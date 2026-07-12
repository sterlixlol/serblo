#!/usr/bin/env python3
"""Gate a multi-lineage tranche: balanced-pool self-BLEU-4 vs real, + per-lineage.

Reads each lineage's docs.jsonl under --tranche, samples the SAME count from every family
(min across families → balanced, so a fast lineage can't swamp the pool), pools them, and
reports self-BLEU / distinct-n for the pool and each lineage. Compare the pool to real-conv
(self-BLEU-4 ~0.326) and the prior balanced 4-lineage pool (0.195).
"""
import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diversity_baseline import distinct_n, self_bleu, tokenize   # noqa: E402


def load_texts(path):
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line).get("text", "").strip()
                except json.JSONDecodeError:
                    continue
                if t:
                    out.append(t)
    return out


def metrics(texts, seed, cap=None):
    t = list(texts)
    random.Random(seed).shuffle(t)
    if cap:
        t = t[:cap]
    tok = [x for x in (tokenize(d) for d in t) if x]
    if len(tok) < 5:
        return None
    return {"n": len(tok), "sb": self_bleu(tok), "dn": distinct_n(tok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tranche", required=True, help="dir with <lineage>/docs.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--realized", action="store_true",
                    help="also report the REALIZED-proportions pool (all docs at their actual counts, "
                         "sampled to --n) — the at-scale read where a fast lineage can swamp the mix")
    ap.add_argument("--n", type=int, default=500, help="sample size for the realized pool")
    a = ap.parse_args()

    lineages = {}
    for d in sorted(glob.glob(os.path.join(a.tranche, "*/"))):
        name = os.path.basename(d.rstrip("/"))
        texts = load_texts(os.path.join(d, "docs.jsonl"))
        if texts:
            lineages[name] = texts
    if not lineages:
        raise SystemExit(f"no lineage docs.jsonl under {a.tranche}")

    counts = {k: len(v) for k, v in lineages.items()}
    mincount = min(counts.values())
    print(f"lineages: {counts}")
    print(f"balanced sample: {mincount}/lineage\n")

    print("=== per-lineage self-BLEU-4 (lower = more diverse) ===")
    for name, texts in lineages.items():
        m = metrics(texts, a.seed, mincount)
        if m:
            print(f"  {name:16} n={m['n']:3}  self-BLEU-4={m['sb'][4]:.4f}  distinct-2={m['dn'][2]:.3f}")

    pool = []
    for name, texts in lineages.items():
        t = list(texts)
        random.Random(a.seed).shuffle(t)
        pool += t[:mincount]
    m = metrics(pool, a.seed)
    if not m:
        raise SystemExit(f"balanced pool too small ({len(pool)} docs) to gate reliably")
    print(f"\n=== BALANCED POOL  (n={m['n']}, {len(lineages)} families) ===")
    print(f"  self-BLEU 2/3/4 = {m['sb'][2]:.3f} / {m['sb'][3]:.3f} / {m['sb'][4]:.4f}")
    print(f"  distinct 1/2    = {m['dn'][1]:.3f} / {m['dn'][2]:.3f}")
    sb4 = m["sb"][4]
    verdict = ("BEATS real" if sb4 < 0.326 - 0.03 else
               "comparable to real" if sb4 < 0.326 + 0.03 else "WORSE than real")
    print(f"  vs real-conv 0.326 (prior 4-lineage pool 0.195)  ->  {verdict}")

    if a.realized:                                            # AT-SCALE read: pool at ACTUAL counts (a fast
        allt = [t for texts in lineages.values() for t in texts]   # lineage swamps here), sampled to --n
        mr = metrics(allt, a.seed, cap=a.n)
        if mr:
            r4 = mr["sb"][4]
            rv = ("BEATS real" if r4 < 0.296 else "comparable to real" if r4 < 0.356 else "WORSE than real")
            print(f"\n=== REALIZED POOL  (n={mr['n']} of {len(allt)}, actual proportions) ===")
            print(f"  self-BLEU-4 = {r4:.4f}   distinct-2 = {mr['dn'][2]:.3f}   ->  {rv}")
            print("  proportions: " + ", ".join(f"{k} {len(t) * 100 // len(allt)}%"
                                                 for k, t in lineages.items()))


if __name__ == "__main__":
    main()
