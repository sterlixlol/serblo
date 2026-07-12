#!/usr/bin/env python3
"""Blind-eval stage 2 (proxy judge): Opus votes SFT-vs-DPO on every eval pair via `claude -p`.

Separation rule: prefs were judged by gemini+deepseek, so the EVAL judge must be neither:
Opus (flat-rate Max plan, $0). Sides are randomized per id; the model is pinned to opus.
Append-only + resumable. Stefan's blind read stays the primary
verdict; this is the calibrated proxy.

  PYTHONPATH=src python3 scripts/judge_eval.py            # ~200 calls, ~5-8s each
"""
import argparse
import json
import pathlib
import random
import re
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]

PROMPT = (
    "Ti si strogi ocenjivač odgovora na srpskom jeziku. Dole su pitanje i dva odgovora "
    "manjeg jezičkog modela. Izaberi BOLJI odgovor: prirodniji srpski (nikako prevodilački), "
    "tačnije i korisnije odgovara na pitanje, primeren registar. VAŽNO: duži odgovor NIJE "
    "automatski bolji — kratak, tačan i prirodan odgovor pobeđuje dug i naduvan. "
    'Odgovori ISKLJUČIVO jednim znakom: "A", "B" ili "X" (izjednačeno/oba loša).\n\n'
    "PITANJE:\n{q}\n\nODGOVOR A:\n{a}\n\nODGOVOR B:\n{b}\n\nTvoj izbor (A/B/X):")


def parse_vote(text):
    m = re.search(r"\b([ABX])\b", (text or "").strip().upper())
    return m.group(1) if m else None


def judge_one(rec, model, timeout):
    """-> dict with vote mapped back through the side shuffle."""
    user_q = next((m["content"] for m in rec["prompt_messages"] if m["role"] == "user"), "")
    flip = random.Random(rec["id"]).random() < 0.5       # per-id, reproducible
    a_txt, b_txt = (rec["dpo"], rec["sft"]) if flip else (rec["sft"], rec["dpo"])
    prompt = PROMPT.format(q=user_q, a=a_txt, b=b_txt)
    t = time.time()
    try:
        r = subprocess.run(["claude", "-p", "--model", model, prompt],
                           capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
        vote = parse_vote(r.stdout)
    except subprocess.TimeoutExpired:
        vote = None
    if vote is None:
        return {"id": rec["id"], "error": "no parseable vote", "secs": round(time.time() - t, 1)}
    winner = "tie" if vote == "X" else ("dpo" if (vote == "A") == flip else "sft")
    return {"id": rec["id"], "winner": winner, "vote": vote, "flip": flip,
            "secs": round(time.time() - t, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default=str(ROOT / "data/dpo/eval_answers.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/dpo/eval_votes.jsonl"))
    ap.add_argument("--model", default="opus")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.answers) if l.strip()]
    if a.limit:
        recs = recs[:a.limit]
    out = pathlib.Path(a.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in open(out) if l.strip()}
    todo = [r for r in recs if r["id"] not in done]
    print(f"judging {len(todo)} pairs ({len(done)} done) via claude -p --model {a.model}", flush=True)

    tally = {"dpo": 0, "sft": 0, "tie": 0, "error": 0}
    with open(out, "a") as f:
        for n, rec in enumerate(todo, 1):
            res = judge_one(rec, a.model, a.timeout)
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()                                    # incremental, resumable
            tally[res.get("winner", "error")] += 1
            if n % 10 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} | {tally}", flush=True)
    print(f"done -> {out} | {tally}", flush=True)


if __name__ == "__main__":
    main()
