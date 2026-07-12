#!/usr/bin/env python3
"""Blind-eval stage 1: one answer from EACH model (SFT vs DPO) for every reserved eval prompt.

The 200 prompts in data/dpo/eval_prompts.jsonl were held out of SFT training, preference
sampling, and DPO training: virgin ground. Output feeds build_eval_page.py (Stefan, blind)
and judge_eval.py (Opus proxy).

  PYTHONPATH=src python3 scripts/gen_eval_answers.py \
      --sft data/sft/ckpts/sft_final.pt --dpo data/dpo/ckpts/dpo_final.pt
"""
import argparse
import importlib.util
import json
import pathlib
import sys

import torch
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from serblo.train.model import GPT, GPTConfig            # noqa: E402

_spec = importlib.util.spec_from_file_location("gen_prefs", ROOT / "scripts/gen_prefs.py")
assert _spec is not None and _spec.loader is not None
gp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gp)                             # sample_batch / bucket_by_length / iter_jsonl


def answers_for(ckpt_path, prompts, tok, dev, tokens, temp, topk, rep, batch, seed):
    """{id: answer} for one model over all prompts (bucketed batches, deterministic seed)."""
    torch.manual_seed(seed)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model = model.to(torch.bfloat16 if dev == "cuda" else torch.float32).eval().to(dev)
    print(f"# {pathlib.Path(ckpt_path).name}: step {ck['step']:,} best_val {ck.get('best_val'):.4f}",
          flush=True)
    items = [(r["id"], r["prompt_messages"], gp.build_chat_prompt(tok, r["prompt_messages"]))
             for r in prompts]
    out = {}
    buckets = gp.bucket_by_length(items)
    for length in sorted(buckets):
        group = buckets[length]
        for c in range(0, len(group), batch):
            chunk = group[c:c + batch]
            res = gp.sample_batch(model, tok, [ids for _, _, ids in chunk], 1,
                                  tokens, temp, topk, rep, dev)
            for (pid, _, _), rows in zip(chunk, res):
                out[pid] = rows[0]
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default=str(ROOT / "data/sft/ckpts/sft_final.pt"))
    ap.add_argument("--dpo", default=str(ROOT / "data/dpo/ckpts/dpo_final.pt"))
    ap.add_argument("--prompts", default=str(ROOT / "data/dpo/eval_prompts.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/dpo/eval_answers.jsonl"))
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--temp", type=float, default=0.7)   # demo temperature: this is the shipping config
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--rep-penalty", type=float, default=1.3, dest="rep_penalty")
    ap.add_argument("--batch-prompts", type=int, default=16, dest="batch")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="cap prompts (smoke tests)")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(str(ROOT / "data/tokenizer/tokenizer.json"))
    prompts = list(gp.iter_jsonl(pathlib.Path(a.prompts)))
    if a.limit:
        prompts = prompts[:a.limit]
    print(f"{len(prompts)} eval prompts | {dev}", flush=True)

    sft = answers_for(a.sft, prompts, tok, dev, a.tokens, a.temp, a.topk, a.rep_penalty,
                      a.batch, a.seed)
    dpo = answers_for(a.dpo, prompts, tok, dev, a.tokens, a.temp, a.topk, a.rep_penalty,
                      a.batch, a.seed)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:                            # small file: full rewrite is fine
        for r in prompts:
            f.write(json.dumps({"id": r["id"], "prompt_messages": r["prompt_messages"],
                                "sft": sft[r["id"]], "dpo": dpo[r["id"]]},
                               ensure_ascii=False) + "\n")
    print(f"done: {len(prompts)} prompt pairs -> {out}", flush=True)


if __name__ == "__main__":
    main()
