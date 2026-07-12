#!/usr/bin/env python3
"""Chat with a Serblo SFT/DPO checkpoint: tiny streaming REPL.

  PYTHONPATH=src python3 scripts/chat.py                     # defaults to sft_final.pt
  PYTHONPATH=src python3 scripts/chat.py --ckpt <path> --system "Ti si Serblo." --multi

Single-turn by default (matches the training data); --multi carries the whole
conversation in context (fun, but the 125M forgets fast). /reset clears, /exit quits.
"""
import argparse
import pathlib
import sys

import torch
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from serblo.train.model import GPT, GPTConfig          # noqa: E402
from serblo.train.sft import IM_END, build_chat_prompt  # noqa: E402

CYAN, GREEN, DIM, RESET = "\x1b[96m", "\x1b[92m", "\x1b[2m", "\x1b[0m"


@torch.no_grad()
def stream_reply(model, tok, messages, device, max_new, temp, top_k, rep_penalty):
    """Sample one assistant reply, printing tokens as they decode. Returns the reply."""
    ids = torch.tensor([build_chat_prompt(tok, messages)], device=device)
    n_prompt = ids.shape[1]
    printed = ""
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
        text = tok.decode(ids[0, n_prompt:].tolist(), skip_special_tokens=True)
        if not text.endswith("�"):                # don't print half a UTF-8 char
            sys.stdout.write(text[len(printed):])
            sys.stdout.flush()
            printed = text
    print()
    return printed.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "data/sft/ckpts/sft_final.pt"))
    ap.add_argument("--tokenizer", default=str(ROOT / "data/tokenizer/tokenizer.json"))
    ap.add_argument("--system", default=None, help='e.g. "Ti si Serblo, srpski jezički asistent."')
    ap.add_argument("--multi", action="store_true", help="carry conversation history in context")
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--rep-penalty", type=float, default=1.3, dest="rep_penalty")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(a.tokenizer)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.eval().to(dev)
    print(f"{DIM}# {pathlib.Path(a.ckpt).name} | step {ck['step']:,} | best_val {ck.get('best_val'):.4f} "
          f"| {dev} | {'multi-turn' if a.multi else 'single-turn'} | /reset /exit{RESET}")

    base = [{"role": "system", "content": a.system}] if a.system else []
    history = list(base)
    while True:
        try:
            user = input(f"{CYAN}ti>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            history = list(base)
            print(f"{DIM}(očišćeno){RESET}")
            continue
        messages = (history if a.multi else list(base)) + [{"role": "user", "content": user}]
        sys.stdout.write(f"{GREEN}serblo>{RESET} ")
        sys.stdout.flush()
        reply = stream_reply(model, tok, messages, dev, a.tokens, a.temp, a.topk, a.rep_penalty)
        if a.multi:
            history = messages + [{"role": "assistant", "content": reply}]


if __name__ == "__main__":
    main()
