"""SFT trainer (post-training stage 1): chat-templates the instruction pairs and fine-tunes the
base checkpoint with loss masked to assistant tokens only.

Template (ChatML, ids reserved at tokenizer build: <|endoftext|>=0 <|pad|>=1 <|im_start|>=2 <|im_end|>=3):

    <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n
    <|im_start|>assistant\n{answer}<|im_end|><|endoftext|>

Supervision: ONLY assistant content + its <|im_end|> (the stop signal); everything else,
including the assistant header, gets target -1 (model.py's cross_entropy ignore_index).
Spans are located via tokenizer offsets, so no BPE-boundary assumptions.

Conversations are packed greedily into ctx+1 rows (never split; <|pad|> tail, mask off),
matching the pretraining regime (causal attention across packed docs, EOT separators).

Run: PYTHONPATH=src python3 -m serblo.train.sft --config configs/sft.yaml
"""
import argparse
import json
import math
import pathlib
import random
import signal
import sys
import time

import torch
import yaml
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from serblo.train.model import GPT, GPTConfig                                  # noqa: E402
from serblo.train.train import (_device_type, evaluate, load_checkpoint,       # noqa: E402
                                lr_at, save_checkpoint, train_step)

EOT, PAD, IM_START, IM_END = 0, 1, 2, 3

CHAT_PROMPTS = [                                                # end-of-run vibe check
    [{"role": "user", "content": "Kako se zoveš?"}],
    [{"role": "user", "content": "Šta je fotosinteza?"}],
    [{"role": "user", "content": "Preporuči mi film za veče."}],
    [{"role": "system", "content": "Ti si Serblo."},
     {"role": "user", "content": "Gde si bio juče, brate?"}],
]


def render_messages(messages):
    """Messages -> (template string, [(start, end) char spans to supervise]).
    A span covers one assistant message's content PLUS its <|im_end|>."""
    parts, spans, cur = [], [], 0

    def emit(s):
        nonlocal cur
        parts.append(s)
        cur += len(s)

    for i, m in enumerate(messages):
        emit("<|im_start|>" + m["role"] + "\n")
        c0 = cur
        emit(m["content"])
        emit("<|im_end|>")
        if m["role"] == "assistant":
            spans.append((c0, cur))
        if i < len(messages) - 1:
            emit("\n")
    return "".join(parts), spans


def _supervise(enc, spans):
    """Token supervision flags: True iff the token lies fully inside a supervised char span."""
    sup = [False] * len(enc.ids)
    for a, b in spans:
        for i, (o0, o1) in enumerate(enc.offsets):
            if o1 > o0 and o0 >= a and o1 <= b:
                sup[i] = True
    return sup


def encode_conversation(tok, messages):
    """-> (ids, sup) with a trailing EOT (never supervised; <|im_end|> is the stop signal)."""
    text, spans = render_messages(messages)
    enc = tok.encode(text)
    return enc.ids + [EOT], _supervise(enc, spans) + [False]


def pack_rows(encoded, ctx):
    """Greedy-pack (ids, sup) conversations into rows of ctx+1 tokens; never splits a
    conversation; pads tails with <|pad|> (unsupervised). Returns (X, Y, stats):
    X = row[:-1], Y = row[1:] with -1 wherever the predicted token is unsupervised."""
    L = ctx + 1
    rows_i, rows_s, cur_i, cur_s = [], [], [], []
    dropped_long = dropped_empty = 0

    def flush():
        pad = L - len(cur_i)
        rows_i.append(cur_i + [PAD] * pad)
        rows_s.append(cur_s + [False] * pad)

    for ids, sup in encoded:
        if len(ids) > L:
            dropped_long += 1
            continue
        if not any(sup):
            dropped_empty += 1
            continue
        if len(cur_i) + len(ids) > L:
            flush()
            cur_i, cur_s = [], []
        cur_i, cur_s = cur_i + ids, cur_s + sup
    if cur_i:
        flush()

    if not rows_i:
        raise ValueError(f"no conversations survived packing (dropped: {dropped_long} overlong, "
                         f"{dropped_empty} without supervised tokens) — check the input file/ctx")
    rows = torch.tensor(rows_i, dtype=torch.long)
    sup_t = torch.tensor(rows_s, dtype=torch.bool)
    X = rows[:, :-1].contiguous()
    Y = rows[:, 1:].clone()
    Y[~sup_t[:, 1:]] = -1                                        # ignore_index in model.py's CE
    return X, Y, {"dropped_long": dropped_long, "dropped_empty": dropped_empty,
                  "sup_frac": float(sup_t[:, 1:].float().mean())}


class Rows:
    """Batch server over packed rows. Default = epoch order (seeded reshuffle each epoch, every
    row exactly once per epoch); with a generator (evaluate()'s contract) = seeded random rows."""

    def __init__(self, X, Y, seed=0):
        self.X, self.Y, self.n = X, Y, X.shape[0]
        if self.n == 0:                                      # would spin forever in get_batch
            raise ValueError("Rows built with zero rows")
        self._gen = torch.Generator().manual_seed(seed)
        self._queue, self.rows_served = [], 0

    @property
    def epoch(self):
        return self.rows_served / self.n

    def get_batch(self, batch_size, device, generator=None):
        if generator is not None:
            idx = torch.randint(0, self.n, (batch_size,), generator=generator)
        else:
            while len(self._queue) < batch_size:
                self._queue += torch.randperm(self.n, generator=self._gen).tolist()
            idx = torch.tensor(self._queue[:batch_size])
            self._queue = self._queue[batch_size:]
            self.rows_served += batch_size
        x, y = self.X[idx], self.Y[idx]
        if str(device).startswith("cuda"):
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)


class SFTData:
    """messages-JSONL -> shuffled conversation-level train/val split -> packed row tensors."""

    def __init__(self, path, tok, ctx, val_frac=0.02, seed=7):
        convs, self.skipped_specials = [], 0
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                msgs = json.loads(line)["messages"]
                if any("<|" in m["content"] for m in msgs):      # defensive: no template injection
                    self.skipped_specials += 1
                    continue
                convs.append(msgs)
        random.Random(seed).shuffle(convs)
        n_val = max(1, int(len(convs) * val_frac))
        self.n_convs, self.n_val_convs = len(convs), n_val

        def build(subset, seed_off):
            texts_spans = [render_messages(m) for m in subset]
            encs = tok.encode_batch([t for t, _ in texts_spans])   # parallel; same result as encode()
            encoded = [(e.ids + [EOT], _supervise(e, s) + [False])
                       for e, (_, s) in zip(encs, texts_spans)]
            X, Y, stats = pack_rows(encoded, ctx)
            return Rows(X, Y, seed + seed_off), stats

        self.val, self.val_stats = build(convs[:n_val], 1)
        self.train, self.train_stats = build(convs[n_val:], 2)


def build_chat_prompt(tok, messages):
    """Generation-side prompt ids: [EOT] + rendered messages + '\\n<|im_start|>assistant\\n'."""
    text, _ = render_messages(messages)
    return [EOT] + tok.encode(text + "\n<|im_start|>assistant\n").ids


@torch.no_grad()
def chat_sample(model, tok, messages, device, max_new=120, temp=0.7, top_k=200, rep_penalty=1.3):
    """Sample an assistant reply; stops at <|im_end|>. Mirrors scripts/sample.py's decoding."""
    ids = torch.tensor([build_chat_prompt(tok, messages)], device=device)
    n_prompt = ids.shape[1]
    for _ in range(max_new):
        logits, _ = model(ids[:, -model.cfg.ctx:])
        logits = logits[:, -1, :]
        if rep_penalty != 1.0:
            score = torch.gather(logits, 1, ids)
            score = torch.where(score > 0, score / rep_penalty, score * rep_penalty)
            logits.scatter_(1, ids, score)
        logits = logits / max(temp, 1e-5)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(torch.softmax(logits, -1), 1)
        if nxt.item() == IM_END:
            break
        ids = torch.cat([ids, nxt], dim=1)
    return tok.decode(ids[0, n_prompt:].tolist(), skip_special_tokens=True).strip()


def run(cfg):
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 7))
    bf16 = cfg.get("bf16", True) and _device_type(device) == "cuda"
    out = pathlib.Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer.from_file(cfg["tokenizer"])
    base = torch.load(cfg["base_ckpt"], map_location="cpu", weights_only=False)
    gcfg = GPTConfig(**base["config"])
    assert gcfg.vocab_size == tok.get_vocab_size(), (
        f"tokenizer vocab {tok.get_vocab_size()} != model vocab {gcfg.vocab_size}")
    model = GPT(gcfg).to(device)
    model.load_state_dict(base["model"])
    print(f"base: step {base.get('step'):,} ({base.get('tokens', 0):,} tokens)", flush=True)

    data = SFTData(cfg["sft_file"], tok, gcfg.ctx, cfg.get("val_frac", 0.02), cfg.get("seed", 7))
    print(f"data: {data.n_convs:,} convs ({data.skipped_specials} skipped on '<|'), "
          f"train rows {data.train.n:,} / val rows {data.val.n:,}, "
          f"sup_frac {data.train_stats['sup_frac']:.3f}, "
          f"dropped long {data.train_stats['dropped_long']} empty {data.train_stats['dropped_empty']}",
          flush=True)

    opt = model.configure_optimizers(cfg["weight_decay"], cfg["peak_lr"], tuple(cfg["betas"]), device)
    micro, accum = cfg["micro_batch"], cfg["grad_accum"]
    max_steps = math.ceil(cfg["epochs"] * data.train.n / (micro * accum))
    warmup = max(5, int(cfg.get("warmup_frac", 0.03) * max_steps))
    step, tokens, best_val, bad_evals = 0, 0, float("inf"), 0

    latest = out / "sft_latest.pt"
    if latest.exists():                                          # resumable
        ck = load_checkpoint(latest, model, opt, map_location="cpu")
        step, tokens, best_val = ck["step"], ck["tokens"], ck["best_val"]
        bad_evals = ck.get("bad_evals", 0)
        print(f"resumed from step {step}", flush=True)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    print(f"SFT: {model.num_params():,} params | {data.train.n:,} rows x {cfg['epochs']} epochs "
          f"= {max_steps} steps (warmup {warmup}) | {micro}x{accum} rows/step | device {device}",
          flush=True)
    model.train()
    t0 = time.time()
    while step < max_steps and not stop["flag"]:
        lr = lr_at(step, warmup, max_steps, cfg["peak_lr"], cfg.get("lr_floor_frac", 0.1))
        loss, gnorm = train_step(model, opt, data.train, micro, accum, device, bf16,
                                 cfg["grad_clip"], lr)
        if not math.isfinite(loss):
            print(f"[FATAL] non-finite loss at step {step + 1} — stopping without saving", flush=True)
            sys.exit(1)
        step += 1
        tokens += micro * accum * gcfg.ctx
        if step % cfg.get("log_interval", 5) == 0:
            print(f"step {step:4d}/{max_steps} | loss {loss:.4f} | lr {lr:.2e} | "
                  f"gnorm {gnorm:.2f} | epoch {data.train.epoch:.2f} | "
                  f"{tokens / (time.time() - t0):,.0f} tok/s", flush=True)
        if step % cfg.get("eval_interval", 20) == 0 and not stop["flag"]:
            vloss = evaluate(model, data.val, micro, device, bf16, cfg.get("eval_iters", 8))
            improved = vloss < best_val - 1e-3
            best_val = min(best_val, vloss)
            bad_evals = 0 if improved else bad_evals + 1
            if improved:                                     # one-shot release: keep the best too
                save_checkpoint(out / "sft_best.pt", model, opt, step, tokens, best_val,
                                base["config"], bad_evals)
            print(f"  [eval] step {step} | val_loss {vloss:.4f} | best {best_val:.4f}"
                  f"{' (saved sft_best)' if improved else ''}", flush=True)
        if step % cfg.get("ckpt_interval", 50) == 0:
            save_checkpoint(latest, model, opt, step, tokens, best_val, base["config"], bad_evals)

    save_checkpoint(latest, model, opt, step, tokens, best_val, base["config"], bad_evals)
    final = out / "sft_final.pt"
    save_checkpoint(final, model, opt, step, tokens, best_val, base["config"], bad_evals)
    print(f"done: step {step}, best_val {best_val:.4f} -> {final}", flush=True)

    model.eval()                                                 # end-of-run vibe check
    for messages in CHAT_PROMPTS:
        try:
            reply = chat_sample(model, tok, messages, device)
        except Exception as e:                                   # sampling must never fail the run
            reply = f"<sampling failed: {e}>"
        print(f"\n>>> {messages[-1]['content']}\n{reply}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    run(yaml.safe_load(open(ap.parse_args().config)))


if __name__ == "__main__":
    main()
