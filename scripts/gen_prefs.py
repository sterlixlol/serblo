#!/usr/bin/env python3
"""Preference-pair stage 1: sample K candidate replies per prompt from the SFT checkpoint.

Prompts come from the SFT train file (messages minus the reference answer). Output is
append-only + resumable by prompt id; judge_prefs.py turns it into
(chosen, rejected) pairs.

Speed: bf16 + length-bucketed prompt batching (prompts with IDENTICAL token length share a
batch, so no padding/attention-mask is needed; the model has none). Rough rates at k=4,
160 new tokens: 3060 --batch-prompts 16 ≈ 4-6h for 8k; A100/H100 --batch-prompts 48-64
≈ ~1h. Ctrl-C-safe, resumes by id, live ETA prints as it runs.

  PYTHONPATH=src python3 scripts/gen_prefs.py --ckpt data/sft/ckpts/sft_final.pt \
      --n-prompts 8000 --k 4 --batch-prompts 16 --out data/dpo/samples.jsonl
"""
import argparse
import json
import pathlib
import sys
import time

import torch
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from serblo.train.model import GPT, GPTConfig            # noqa: E402
from serblo.train.sft import IM_END, build_chat_prompt   # noqa: E402


def truncate_at(ids, stop_id):
    """Token ids up to (excluding) the first stop_id."""
    return ids[:ids.index(stop_id)] if stop_id in ids else ids


def iter_jsonl(path):
    """Parsed JSONL lines; tolerates a torn FINAL line (crash mid-write) but raises on a torn
    middle line (real corruption, not an interrupted append). Mirrors judge_prefs.iter_jsonl."""
    with open(path) as f:
        lines = f.readlines()
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        try:
            yield json.loads(ln)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                print(f"  [warn] ignoring torn final line in {path}", flush=True)
                return
            raise


def build_prompt_pools(prompts_path, exclude_path, reserve_path, n_reserve):
    """-> (excluded_ids, reserved_ids). Preference prompts must be UNSEEN by SFT (else the
    model samples around memorized answers and DPO double-dips the train set), and the first
    n_reserve eligible prompts are materialized to reserve_path as an EVAL-ONLY set, never
    sampled, never trained on, so the final SFT-vs-DPO blind eval is uncontaminated."""
    excluded = set()
    if exclude_path and pathlib.Path(exclude_path).exists():
        excluded = {r.get("id") for r in iter_jsonl(pathlib.Path(exclude_path))}
    reserved = set()
    if reserve_path:
        rp = pathlib.Path(reserve_path)
        if rp.exists():                                      # already materialized: honor it
            reserved = {r["id"] for r in iter_jsonl(rp)}
        else:                                                # materialize once, file order
            rp.parent.mkdir(parents=True, exist_ok=True)
            with open(prompts_path) as fin, open(rp, "w") as fout:
                for i, line in enumerate(fin):
                    if len(reserved) >= n_reserve:
                        break
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    if d["messages"][-1]["role"] != "assistant":
                        continue
                    pid = d.get("id") or f"idx-{i}"
                    if pid in excluded:
                        continue
                    fout.write(json.dumps({"id": pid, "prompt_messages": d["messages"][:-1]},
                                          ensure_ascii=False) + "\n")
                    reserved.add(pid)
            print(f"reserved {len(reserved)} eval-only prompts -> {rp}", flush=True)
    return excluded, reserved


@torch.no_grad()
def sample_batch(model, tok, prompts_ids, k, max_new, temp, top_k, rep_penalty, device):
    """K samples for EACH of several EQUAL-LENGTH prompts in one batch (no padding needed,
    which matters because the model has no attention mask). Sampling math runs in fp32 even
    when the model is bf16. Returns list of k decoded strings per prompt."""
    assert len({len(p) for p in prompts_ids}) == 1, "prompts in a batch must share token length"
    ids = torch.tensor([p for p in prompts_ids for _ in range(k)], device=device)
    n_prompt = ids.shape[1]
    for _ in range(max_new):
        logits, _ = model(ids[:, -model.cfg.ctx:])
        logits = logits[:, -1, :].float()                    # fp32 sampling on bf16 models
        if rep_penalty != 1.0:
            score = torch.gather(logits, 1, ids)
            score = torch.where(score > 0, score / rep_penalty, score * rep_penalty)
            logits.scatter_(1, ids, score)
        logits = logits / max(temp, 1e-5)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        ids = torch.cat([ids, torch.multinomial(torch.softmax(logits, -1), 1)], dim=1)
        if (ids[:, n_prompt:] == IM_END).any(dim=1).all():   # every row has finished
            break
    rows = [tok.decode(truncate_at(r, IM_END), skip_special_tokens=True).strip()
            for r in ids[:, n_prompt:].tolist()]
    return [rows[i * k:(i + 1) * k] for i in range(len(prompts_ids))]


def sample_k(model, tok, messages, k, max_new, temp, top_k, rep_penalty, device):
    """Single-prompt convenience wrapper around sample_batch."""
    prompt = build_chat_prompt(tok, messages)
    return sample_batch(model, tok, [prompt], k, max_new, temp, top_k, rep_penalty, device)[0]


def bucket_by_length(items):
    """[(pid, messages, prompt_ids)] -> {token_length: [items]}, insertion-ordered."""
    buckets = {}
    for item in items:
        buckets.setdefault(len(item[2]), []).append(item)
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=str(ROOT / "data/tokenizer/tokenizer.json"))
    ap.add_argument("--prompts", default=str(ROOT / "data/sft/assembled/serblo-sft-full.jsonl"))
    ap.add_argument("--exclude-from", dest="exclude_from",
                    default=str(ROOT / "data/sft/assembled/serblo-sft-train.jsonl"),
                    help="skip ids present here (default: SFT train file -> only UNSEEN prompts)")
    ap.add_argument("--reserve-eval", dest="reserve_eval",
                    default=str(ROOT / "data/dpo/eval_prompts.jsonl"),
                    help="hold the first N eligible prompts out as the eval-only set")
    ap.add_argument("--n-reserve", type=int, default=200, dest="n_reserve")
    ap.add_argument("--out", default=str(ROOT / "data/dpo/samples.jsonl"))
    ap.add_argument("--n-prompts", type=int, default=8000, dest="n_prompts")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--temp", type=float, default=0.9)      # hotter than demo sampling: we WANT spread
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--rep-penalty", type=float, default=1.3, dest="rep_penalty")
    ap.add_argument("--batch-prompts", type=int, default=16, dest="batch_prompts",
                    help="prompts per forward (x k rows); 16 fits the 3060, 48-64 an A100/H100")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    a = ap.parse_args()

    excluded, reserved = build_prompt_pools(a.prompts, a.exclude_from,
                                            a.reserve_eval, a.n_reserve)
    print(f"pools: {len(excluded)} SFT-seen ids excluded, {len(reserved)} reserved for eval",
          flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(a.tokenizer)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    if dev == "cpu":
        dtype = torch.float32                                # bf16 on CPU is slower, not faster
    model = model.to(dtype).eval().to(dev)
    print(f"# ckpt step {ck['step']:,} | k={a.k} temp {a.temp} rep {a.rep_penalty} "
          f"| batch {a.batch_prompts} prompts | {dtype} | {dev}", flush=True)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {r["id"] for r in iter_jsonl(out)}
    print(f"resume: {len(done)} prompts already sampled", flush=True)

    todo = []                                                # collect eligible, then bucket
    with open(a.prompts) as fin:
        for i, line in enumerate(fin):
            if len(done) + len(todo) >= a.n_prompts:
                break
            if not line.strip():
                continue
            d = json.loads(line)
            msgs = d["messages"]
            if msgs[-1]["role"] != "assistant":              # need a strippable reference answer
                continue
            pid = d.get("id") or f"idx-{i}"
            if pid in done or pid in excluded or pid in reserved:
                continue
            todo.append((pid, msgs[:-1], build_chat_prompt(tok, msgs[:-1])))
    buckets = bucket_by_length(todo)
    print(f"{len(todo)} prompts to sample across {len(buckets)} length-buckets", flush=True)

    n_written, n_new, t0 = len(done), 0, time.time()
    with open(out, "a") as fout:
        for length in sorted(buckets):
            group = buckets[length]
            for c in range(0, len(group), a.batch_prompts):
                chunk = group[c:c + a.batch_prompts]
                results = sample_batch(model, tok, [ids for _, _, ids in chunk], a.k,
                                       a.tokens, a.temp, a.topk, a.rep_penalty, dev)
                for (pid, msgs, _), samples in zip(chunk, results):
                    fout.write(json.dumps({"id": pid, "prompt_messages": msgs,
                                           "samples": samples}, ensure_ascii=False) + "\n")
                    n_written += 1
                    n_new += 1
                fout.flush()                                 # incremental: survive any interrupt
                if n_new <= a.batch_prompts or n_written % 256 < a.batch_prompts:
                    rate = (time.time() - t0) / max(1, n_new)
                    eta_h = rate * (a.n_prompts - n_written) / 3600
                    print(f"{n_written}/{a.n_prompts} | {rate:.2f}s/prompt | ~{eta_h:.1f}h left",
                          flush=True)
    print(f"done: {n_written} prompts in {out}", flush=True)


if __name__ == "__main__":
    main()
