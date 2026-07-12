#!/usr/bin/env python3
"""Preference-pair stage 2: an LLM judge picks best/worst of the K candidates -> prefs.jsonl
for the DPO trainer ({prompt_messages, chosen, rejected}).

Bias controls: candidates are SHUFFLED per record (seeded on the id) so position bias can't
leak into the dataset, and the judge prompt explicitly forbids preferring longer answers.
Append-only + resumable by id. Reuses run_api's HOSTS/keys/RateLimiter.

  PYTHONPATH=src python3 scripts/judge_prefs.py --samples data/dpo/samples.jsonl \
      --host gemini --model gemini-2.5-flash --rpm-per-key 5 --concurrency 4
"""
import argparse
import json
import pathlib
import random
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from serblo.generate.run_api import BACKOFF, HOSTS, RateLimiter, load_keys   # noqa: E402

JUDGE_SYSTEM = (
    "You are a strict evaluator of Serbian assistant replies. You will see one instruction "
    "and several candidate replies. Judge ONLY reply quality: natural native Serbian (never "
    "translationese), no foreign-script or gibberish characters, actually answers the "
    "instruction, appropriate register. IMPORTANT: do NOT prefer a reply for being longer — "
    "length is not quality; a short, correct, natural reply beats a long padded one. "
    'Respond with ONLY a JSON object: {"best": <number>, "worst": <number>} — no other text.')


def build_judge_prompt(prompt_messages, samples):
    ctx = "\n".join(f"[{m['role']}] {m['content']}" for m in prompt_messages)
    cands = "\n\n".join(f"REPLY {i + 1}:\n{s}" for i, s in enumerate(samples))
    return (f"INSTRUCTION:\n{ctx}\n\n{cands}\n\n"
            f'JSON verdict ({{"best": 1-{len(samples)}, "worst": 1-{len(samples)}}}):')


def parse_verdict(text, k):
    """-> (best_idx0, worst_idx0) or None. Tolerates fences/chatter around the JSON, and tries
    EVERY {...} in the reply (judges sometimes emit a preamble object first)."""
    for m in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(d.get("best"), bool) or isinstance(d.get("worst"), bool):
            continue                                         # int(True) == 1 would silently pass
        try:
            best, worst = int(d["best"]), int(d["worst"])
        except (ValueError, KeyError, TypeError):
            continue
        if best != worst and 1 <= best <= k and 1 <= worst <= k:
            return best - 1, worst - 1
    return None


def iter_jsonl(path):
    """Parsed JSONL lines; tolerates a torn FINAL line (crash mid-write) but raises on a torn
    middle line (real corruption, not an interrupted append)."""
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


def judge_call(url, key, model, prompt, temperature, timeout):
    body = json.dumps({
        "model": model, "temperature": temperature,
        "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                     {"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "User-Agent": "serblo/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return (d.get("choices") or [{}])[0].get("message", {}).get("content") or ""


def judge_one(rec, url, limiter, model, temperature, timeout):
    """-> {"chosen","rejected",...} | {"skip": reason} | {"error": last_err}."""
    samples = [s for s in rec["samples"] if s.strip()]
    if len(samples) < 2 or len(set(samples)) < 2:
        return {"skip": "fewer than 2 distinct non-empty samples"}
    order = list(range(len(samples)))                        # per-id shuffle kills position bias
    random.Random(rec["id"]).shuffle(order)
    shuffled = [samples[j] for j in order]
    prompt = build_judge_prompt(rec["prompt_messages"], shuffled)
    last_err = "no attempt"
    for attempt in range(len(BACKOFF) + 1):
        try:
            text = judge_call(url, limiter.get_key(), model, prompt, temperature, timeout)
            v = parse_verdict(text, len(shuffled))
            if v is None:
                last_err = f"unparseable verdict: {text[:120]!r}"
            else:
                chosen, rejected = shuffled[v[0]], shuffled[v[1]]
                if chosen.strip() == rejected.strip():
                    return {"skip": "judge picked identical texts"}
                return {"id": rec["id"], "prompt_messages": rec["prompt_messages"],
                        "chosen": chosen, "rejected": rejected}
        except Exception as e:                               # 429/timeout/5xx -> backoff + retry
            last_err = f"{type(e).__name__}: {e}"[:200]
        if attempt < len(BACKOFF):
            time.sleep(BACKOFF[attempt])
    return {"error": last_err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(ROOT / "data/dpo/samples.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/dpo/prefs.jsonl"))
    ap.add_argument("--host", default="gemini", choices=sorted(HOSTS))
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--rpm-per-key", type=int, default=5, dest="rpm_per_key")
    ap.add_argument("--max-requests", type=int, default=0, dest="max_requests")
    a = ap.parse_args()

    url = HOSTS[a.host][0]
    limiter = RateLimiter(load_keys(a.host), a.rpm_per_key)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    err_path = out.with_name(out.stem + "-errors.jsonl")

    done = set()
    for p in (out, err_path):                                # errors + skips count as attempted
        if p.exists():                                       # (retry by deleting the errors file)
            done |= {r.get("id") for r in iter_jsonl(p)}

    todo = [r for r in iter_jsonl(pathlib.Path(a.samples)) if r["id"] not in done]
    if a.max_requests:
        todo = todo[:a.max_requests]
    print(f"judging {len(todo)} records ({len(done)} already done) via {a.host}/{a.model}",
          flush=True)

    lock = threading.Lock()
    tally = {"pairs": 0, "skips": 0, "errors": 0}
    with open(out, "a") as fout, open(err_path, "a") as ferr:

        def work(rec):
            res = judge_one(rec, url, limiter, a.model, a.temperature, a.timeout)
            with lock:                                       # incremental, one line per record
                if "chosen" in res:
                    fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                    fout.flush()
                    tally["pairs"] += 1
                else:                                        # skips persist too: never re-judged
                    ferr.write(json.dumps({"id": rec["id"], **res}, ensure_ascii=False) + "\n")
                    ferr.flush()
                    tally["skips" if "skip" in res else "errors"] += 1
                n = sum(tally.values())
                if n % 50 == 0:
                    print(f"{n}/{len(todo)} | {tally}", flush=True)

        with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            futs = [ex.submit(work, r) for r in todo]        # as_completed (NOT ex.map): Ctrl-C
            try:                                             # must cancel the queued thousands,
                for f in as_completed(futs):                 # not run them to completion
                    f.result()
            except KeyboardInterrupt:
                print("\ninterrupted — cancelling queued records (in-flight calls finish)...",
                      flush=True)
                ex.shutdown(wait=True, cancel_futures=True)
                print(f"stopped cleanly: {tally} (resume just re-runs this command)", flush=True)
                raise SystemExit(130)
    print(f"done: {tally} -> {out}", flush=True)


if __name__ == "__main__":
    main()
