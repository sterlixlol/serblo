#!/usr/bin/env python3
"""Gate + eyeball an SFT pairs.jsonl: task coverage, quality flags, sample pairs per task.

  python3 scripts/sft_gate.py data/sft/pilot/pairs.jsonl [--n-samples 2]

Quality flags (diagnostics, not auto-rejects; BCS casual English code-switch is authentic):
  too_long   response > 120 words (a 125M can't produce it; violates 'short')
  refusal    soft refusal / disclaimer / 'as an AI'
  en_bleed   non-translate response that looks English, not Serbian
  format_leak stray <<< / >>> sentinel survived parsing
  short      response < 2 words (fine for yes/no format tasks, flagged elsewhere)
"""
import argparse
import collections
import json
import re
import sys

SR_DIACRITICS = set("čćšžđČĆŠŽĐ")
SR_WORDS = {"i", "je", "da", "na", "se", "u", "za", "ne", "su", "od", "to", "što", "li", "sam",
            "ali", "ili", "kao", "sa", "po", "me", "te", "mi", "ga", "smo", "ste", "će", "bih"}
REFUSAL = re.compile(r"\b(as an ai|kao (veštačka|vjestacka) inteligencij|ne mogu da (ti )?pomognem|"
                     r"žao mi je,? ali|izvinjavam se,? ali|i'?m sorry|i cannot|i can'?t help)\b", re.I)
EN_COMMON = {"the", "and", "you", "for", "with", "this", "that", "your", "have", "are", "is", "to", "of"}
_CYR = re.compile(r"[Ѐ-ӿ]")


def looks_english(text):
    toks = re.findall(r"[a-zA-Z]+", text.lower())
    if len(toks) < 4:
        return False
    en = sum(t in EN_COMMON for t in toks)
    sr = sum(t in SR_WORDS for t in toks)
    return en >= 2 and en > sr and not (set(text) & SR_DIACRITICS)


def flags(task, instr, resp):
    f = []
    words = resp.split()
    if len(words) > 120:
        f.append("too_long")
    if len(words) < 2:
        f.append("short")
    if REFUSAL.search(resp) or REFUSAL.search(instr):
        f.append("refusal")
    if task != "translate" and looks_english(resp):
        f.append("en_bleed")
    if "<<<" in resp or ">>>" in resp or "<<<" in instr:
        f.append("format_leak")
    if any(d in resp or d in instr for d in ("—", "–")):     # em/en-dash = AI tell, strip in post
        f.append("dash")
    if _CYR.search(resp):                                    # stray Cyrillic in Latin-target text
        f.append("cyrillic")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="one or more pairs.jsonl (pooled together)")
    ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--by-source", action="store_true", help="per-source (model) flag-rate table")
    ap.add_argument("--min-n", type=int, default=100, help="hide by-source rows below this count")
    a = ap.parse_args()

    rows = []
    for p in a.paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                r.setdefault("_src", r.get("model") or r.get("source") or p)
                rows.append(r)
    if not rows:
        sys.exit("no pairs")

    cov = collections.Counter(r.get("task", "?") for r in rows)
    flagc = collections.Counter()
    clean = 0
    by_task = collections.defaultdict(list)
    src_total = collections.Counter()
    src_clean = collections.Counter()
    src_flag = collections.defaultdict(collections.Counter)
    for r in rows:
        t = r.get("task", "?")
        fl = flags(t, r.get("instruction", ""), r.get("response", ""))
        r["_flags"] = fl
        by_task[t].append(r)
        src_total[r["_src"]] += 1
        if fl:
            for x in fl:
                flagc[x] += 1
                src_flag[r["_src"]][x] += 1
        else:
            clean += 1
            src_clean[r["_src"]] += 1

    print(f"=== TASK COVERAGE ({len(rows)} pairs) ===")
    for t, n in cov.most_common():
        print(f"  {t:11} {n:4}")
    print(f"\n=== QUALITY ({clean}/{len(rows)} = {100*clean//len(rows)}% flag-free) ===")
    for k, n in flagc.most_common():
        print(f"  {k:12} {n:4}")

    lens = [len(r.get("response", "").split()) for r in rows]
    lens.sort()
    print(f"\n  response words: median {lens[len(lens)//2]}, p90 {lens[int(len(lens)*.9)]}, max {lens[-1]}")

    if a.by_source:
        cols = [k for k, _ in flagc.most_common()]
        print(f"\n=== BY SOURCE (model) — % of pairs carrying each flag (n>={a.min_n}) ===")
        print("  " + f"{'source':24}{'n':>7} {'clean':>6}  " + " ".join(f"{c[:9]:>9}" for c in cols))
        for s, n in src_total.most_common():
            if n < a.min_n:
                continue
            cp = f"{100*src_clean[s]//n}%"
            cells = " ".join(f"{100*src_flag[s][c]/n:>8.1f}%" for c in cols)
            print(f"  {str(s)[:24]:24}{n:>7} {cp:>6}  {cells}")

    print(f"\n=== SAMPLES ({a.n_samples}/task, prefer flag-free) ===")
    for t in cov:
        picks = [r for r in by_task[t] if not r["_flags"]][:a.n_samples]
        picks += [r for r in by_task[t] if r["_flags"]][:max(0, a.n_samples - len(picks))]
        for r in picks[:a.n_samples]:
            fl = f"  ⚑{','.join(r['_flags'])}" if r["_flags"] else ""
            print(f"\n--- {t}  ({r.get('register','')}/{r.get('topic','')}){fl}")
            print(f"  Q: {r.get('instruction','')[:300]}")
            print(f"  A: {r.get('response','')[:400]}")


if __name__ == "__main__":
    main()
