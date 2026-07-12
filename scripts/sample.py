#!/usr/bin/env python3
"""Sample text from a Serblo checkpoint (autoregressive top-k sampling).

  python3 scripts/sample.py <ckpt.pt> [--tokens 120] [--temp 0.8] [--topk 200]
"""
import argparse
import pathlib
import sys

import torch
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from serblo.train.model import GPT, GPTConfig  # noqa: E402


@torch.no_grad()
def generate(model, idx, max_new, temp, top_k, ctx, rep_penalty=1.0):
    for _ in range(max_new):
        cond = idx[:, -ctx:]
        logits, _ = model(cond)
        logits = logits[:, -1, :]
        if rep_penalty != 1.0:                                   # CTRL-style: discourage already-seen tokens
            score = torch.gather(logits, 1, idx)                 # gather seen-token logits (batched, per row)
            score = torch.where(score > 0, score / rep_penalty, score * rep_penalty)
            logits.scatter_(1, idx, score)                       # penalize → scatter back
        logits = logits / max(temp, 1e-5)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = torch.softmax(logits, -1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tokenizer", default=str(ROOT / "data/tokenizer/tokenizer.json"))
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--rep-penalty", type=float, default=1.0, dest="rep_penalty")
    ap.add_argument("--rolls", type=int, default=1)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(a.tokenizer)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval().to(dev)
    print(f"# step {ck['step']:,} | {ck['tokens']:,} tokens seen | best_val {ck.get('best_val')} "
          f"| temp {a.temp} topk {a.topk} rep {a.rep_penalty} rolls {a.rolls} | {dev}\n")

    prompts = ["", "Danas je lep dan i", "U Beogradu je", "Najbolji recept za pitu je",
               "Nauka o klimi pokazuje da", "— Šta ima, gde si bio? — "]
    for p in prompts:
        ids = [0] + (tok.encode(p).ids if p else [])            # id 0 = <|endoftext|> = doc-start
        idx = torch.tensor([ids], device=dev).repeat(a.rolls, 1)
        out = generate(model, idx, a.tokens, a.temp, a.topk, cfg.ctx, a.rep_penalty)
        print("=" * 72)
        print(f"PROMPT: {p!r}")
        for r in range(a.rolls):
            txt = tok.decode(out[r].tolist(), skip_special_tokens=True)
            head = f"  [{r + 1}] " if a.rolls > 1 else ""
            print(f"{head}{txt.strip()}\n")


if __name__ == "__main__":
    main()
