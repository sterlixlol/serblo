"""DPO trainer (post-training stage 2): hand-rolled Direct Preference Optimization on top of the
SFT checkpoint. No reward model, no PPO: train directly on (chosen, rejected) pairs.

    loss = -log sigmoid( beta * [ (logP_pi(chosen) - logP_ref(chosen))
                                 - (logP_pi(rejected) - logP_ref(rejected)) ] )

logP sums run ONLY over supervised tokens (assistant content + <|im_end|>, same masking as SFT,
via sft.encode_conversation). The frozen reference = the SFT weights (the KL leash, built in).

Data: prefs JSONL, one pair per line:
    {"prompt_messages": [{"role": "user", "content": ...}, ...],
     "chosen": "<assistant text>", "rejected": "<assistant text>"}

Pairs are padded per-batch (they're short); never packed. Runs on the 3060 (2 x 125M fits fine).

Run: PYTHONPATH=src python3 -m serblo.train.dpo --config configs/dpo.yaml
"""
import argparse
import copy
import json
import math
import pathlib
import random
import signal
import sys
import time

import torch
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from serblo.train.model import GPT, GPTConfig                                  # noqa: E402
from serblo.train.sft import CHAT_PROMPTS, PAD, chat_sample, encode_conversation  # noqa: E402
from serblo.train.train import _device_type, load_checkpoint, lr_at, save_checkpoint  # noqa: E402


def collate(pairs, device):
    """pairs: list of (c_ids, c_sup, r_ids, r_sup) -> padded (xc, tc, mc, xr, tr, mr).
    x = ids[:-1], t = ids[1:], m = sup[1:] (True where the PREDICTED token is supervised)."""

    def side(idx):
        ids = [p[idx] for p in pairs]
        sup = [p[idx + 1] for p in pairs]
        L = max(len(i) for i in ids)
        x = torch.full((len(ids), L - 1), PAD, dtype=torch.long)
        t = torch.full((len(ids), L - 1), PAD, dtype=torch.long)
        m = torch.zeros((len(ids), L - 1), dtype=torch.bool)
        for b, (i, s) in enumerate(zip(ids, sup)):
            n = len(i) - 1
            x[b, :n] = torch.tensor(i[:-1])
            t[b, :n] = torch.tensor(i[1:])
            m[b, :n] = torch.tensor(s[1:])
        return x.to(device), t.to(device), m.to(device)

    return side(0) + side(2)


def seq_logprob(model, x, t, m, bf16, device):
    """Summed log-probability of the supervised tokens of each row -> [B]."""
    with torch.autocast(device_type=_device_type(device), dtype=torch.bfloat16, enabled=bf16):
        logits, _ = model(x)
    lsm = torch.log_softmax(logits.float(), dim=-1)
    tok_lp = torch.gather(lsm, 2, t.unsqueeze(-1)).squeeze(-1)   # [B, T] logP of each target
    return (tok_lp * m).sum(-1)


def dpo_step(policy, ref, opt, batch, beta, grad_clip, lr, bf16, device):
    """One optimizer step on one batch of pairs. Returns (loss, accuracy, mean margin)."""
    for g in opt.param_groups:
        g["lr"] = lr
    opt.zero_grad(set_to_none=True)
    xc, tc, mc, xr, tr, mr = batch
    with torch.no_grad():
        ref_c = seq_logprob(ref, xc, tc, mc, bf16, device)
        ref_r = seq_logprob(ref, xr, tr, mr, bf16, device)
    pi_c = seq_logprob(policy, xc, tc, mc, bf16, device)
    pi_r = seq_logprob(policy, xr, tr, mr, bf16, device)
    margin = (pi_c - ref_c) - (pi_r - ref_r)                     # implicit reward difference
    loss = -F.logsigmoid(beta * margin).mean()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
    opt.step()
    with torch.no_grad():
        return (float(loss.detach()), float((margin > 0).float().mean()),
                float(margin.mean()), float(gn))


@torch.no_grad()
def dpo_eval(policy, ref, data, batch_size, beta, bf16, device, iters=8, seed=1234):
    gen = torch.Generator().manual_seed(seed)
    losses, accs = [], []
    for _ in range(iters):
        xc, tc, mc, xr, tr, mr = data.get_batch(batch_size, device, generator=gen)
        margin = (seq_logprob(policy, xc, tc, mc, bf16, device)
                  - seq_logprob(ref, xc, tc, mc, bf16, device)
                  - seq_logprob(policy, xr, tr, mr, bf16, device)
                  + seq_logprob(ref, xr, tr, mr, bf16, device))
        losses.append(float(-F.logsigmoid(beta * margin).mean()))
        accs.append(float((margin > 0).float().mean()))
    return sum(losses) / len(losses), sum(accs) / len(accs)


class DPOData:
    """prefs JSONL -> encoded (chosen, rejected) sides, conversation-level train/val split."""

    def __init__(self, path, tok, ctx, val_frac=0.02, seed=7):
        raw, self.skipped = [], 0
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                texts = [m["content"] for m in d["prompt_messages"]] + [d["chosen"], d["rejected"]]
                if any("<|" in t for t in texts) or d["chosen"].strip() == d["rejected"].strip():
                    self.skipped += 1
                    continue
                raw.append(d)
        random.Random(seed).shuffle(raw)

        pairs = []
        for d in raw:
            c = encode_conversation(tok, d["prompt_messages"] + [{"role": "assistant", "content": d["chosen"]}])
            r = encode_conversation(tok, d["prompt_messages"] + [{"role": "assistant", "content": d["rejected"]}])
            if max(len(c[0]), len(r[0])) > ctx + 1 or not (any(c[1]) and any(r[1])):
                self.skipped += 1
                continue
            pairs.append((c[0], c[1], r[0], r[1]))
        if not pairs:
            raise ValueError(f"no usable preference pairs in {path} ({self.skipped} skipped)")
        n_val = max(1, int(len(pairs) * val_frac))
        self.val_pairs, self.train_pairs = pairs[:n_val], pairs[n_val:]
        if not self.train_pairs:
            raise ValueError(f"all {len(pairs)} pairs landed in val — need more pairs "
                             f"or a smaller val_frac")
        self.n = len(pairs)
        self._gen = torch.Generator().manual_seed(seed)
        self._queue, self.rows_served = [], 0

    @property
    def epoch(self):
        return self.rows_served / max(1, len(self.train_pairs))

    def get_batch(self, batch_size, device, generator=None):
        if generator is not None:                                # eval: seeded random from VAL
            idx = torch.randint(0, len(self.val_pairs), (batch_size,), generator=generator).tolist()
            return collate([self.val_pairs[i] for i in idx], device)
        while len(self._queue) < batch_size:                     # train: epoch order, seeded reshuffle
            self._queue += torch.randperm(len(self.train_pairs), generator=self._gen).tolist()
        idx, self._queue = self._queue[:batch_size], self._queue[batch_size:]
        self.rows_served += batch_size
        return collate([self.train_pairs[i] for i in idx], device)


def run(cfg):
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 7))
    bf16 = cfg.get("bf16", True) and _device_type(device) == "cuda"
    out = pathlib.Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer.from_file(cfg["tokenizer"])
    sft = torch.load(cfg["sft_ckpt"], map_location="cpu", weights_only=False)
    gcfg = GPTConfig(**sft["config"])
    assert gcfg.vocab_size == tok.get_vocab_size()
    policy = GPT(gcfg).to(device)
    policy.load_state_dict(sft["model"])
    ref = copy.deepcopy(policy).eval()                           # the frozen KL anchor
    for p in ref.parameters():
        p.requires_grad_(False)

    data = DPOData(cfg["prefs_file"], tok, gcfg.ctx, cfg.get("val_frac", 0.02), cfg.get("seed", 7))
    print(f"data: {data.n:,} pairs kept ({data.skipped} skipped), "
          f"train {len(data.train_pairs):,} / val {len(data.val_pairs):,}", flush=True)

    opt = policy.configure_optimizers(cfg.get("weight_decay", 0.0), cfg["peak_lr"],
                                      tuple(cfg.get("betas", [0.9, 0.95])), device)
    bs, beta = cfg["batch_size"], cfg.get("beta", 0.1)
    max_steps = math.ceil(cfg["epochs"] * len(data.train_pairs) / bs)
    warmup = max(5, int(cfg.get("warmup_frac", 0.05) * max_steps))
    step, best_val = 0, float("inf")

    latest = out / "dpo_latest.pt"
    if latest.exists():
        ck = load_checkpoint(latest, policy, opt, map_location="cpu")
        step, best_val = ck["step"], ck["best_val"]
        print(f"resumed from step {step}", flush=True)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    print(f"DPO: beta {beta} | {len(data.train_pairs):,} pairs x {cfg['epochs']} epochs = "
          f"{max_steps} steps (warmup {warmup}) | batch {bs} pairs | device {device}", flush=True)
    policy.train()
    t0 = time.time()
    while step < max_steps and not stop["flag"]:
        lr = lr_at(step, warmup, max_steps, cfg["peak_lr"], cfg.get("lr_floor_frac", 0.1))
        batch = data.get_batch(bs, device)
        loss, acc, margin, gn = dpo_step(policy, ref, opt, batch, beta,
                                         cfg.get("grad_clip", 1.0), lr, bf16, device)
        if not math.isfinite(loss):
            print(f"[FATAL] non-finite loss at step {step + 1} — stopping without saving", flush=True)
            sys.exit(1)
        step += 1
        if step % cfg.get("log_interval", 5) == 0:
            print(f"step {step:4d}/{max_steps} | loss {loss:.4f} | acc {acc:.2f} | "
                  f"margin {margin:+.3f} | gnorm {gn:.2f} | lr {lr:.2e} | "
                  f"epoch {data.epoch:.2f} | {(time.time() - t0) / step:.1f}s/step", flush=True)
        if step % cfg.get("eval_interval", 20) == 0 and not stop["flag"]:
            vloss, vacc = dpo_eval(policy, ref, data, bs, beta, bf16, device,
                                   cfg.get("eval_iters", 8))
            improved = vloss < best_val
            best_val = min(best_val, vloss)
            if improved:                                     # one-shot release: keep the best too
                save_checkpoint(out / "dpo_best.pt", policy, opt, step, 0, best_val, sft["config"])
            print(f"  [eval] step {step} | val_loss {vloss:.4f} | val_acc {vacc:.2f} | "
                  f"best {best_val:.4f}{' (saved dpo_best)' if improved else ''}", flush=True)
        if step % cfg.get("ckpt_interval", 50) == 0:
            save_checkpoint(latest, policy, opt, step, 0, best_val, sft["config"])

    save_checkpoint(latest, policy, opt, step, 0, best_val, sft["config"])
    final = out / "dpo_final.pt"
    save_checkpoint(final, policy, opt, step, 0, best_val, sft["config"])
    print(f"done: step {step}, best_val {best_val:.4f} -> {final}", flush=True)

    policy.eval()
    for messages in CHAT_PROMPTS:
        try:
            reply = chat_sample(policy, tok, messages, device)
        except Exception as e:
            reply = f"<sampling failed: {e}>"
        print(f"\n>>> {messages[-1]['content']}\n{reply}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    run(yaml.safe_load(open(ap.parse_args().config)))


if __name__ == "__main__":
    main()
