#!/usr/bin/env python3
"""distribution_gate.py: token-distribution companion to diversity_baseline.py.

self-BLEU / distinct-n measure PHRASE-level diversity but are BLIND to unigram-frequency
skew: a synthetic source that over-uses one particle (e.g. "ej") passes self-BLEU while
stamping the corpus. (Found the hard way: the prompt's example particles were
copied 6-540x their real rate; self-BLEU never flinched.) This tool compares the unigram
distribution of synthetic vs real (same register) and surfaces WHICH tokens are skewed.

Metrics:
  Jensen-Shannon divergence (bits, 0=identical .. 1=disjoint): symmetric, robust summary,
      needs no smoothing (finite even where one side has zeros).
  KL(synth||real) / KL(real||synth): asymmetric, add-k smoothed.
  Over- / under-represented token tables: ranked by KL contribution; the actionable output.
      "over" surfaces synth's tics; "under" surfaces what synth avoids that real uses.

Calibrate with --control (split real in half) for the sampling-noise JS floor: synthetic
should land NEAR that floor if it's distributionally natural, well above it if it's skewed.
Always compare at the same --n and register, or the numbers aren't comparable.

Usage:
  python scripts/distribution_gate.py --synth data/generated/mimo/docs.jsonl \
      --real real_conv.jsonl --register conversational --n 300
  python scripts/distribution_gate.py --real real_conv.jsonl --control --n 300   # noise floor
"""
import argparse
import glob
import gzip
import json
import math
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diversity_baseline import tokenize  # identical tokenization on both sides (sentinel-strip, lowercase, keep diacritics)


def _open(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def load_texts(path, register=None):
    """Load doc texts from a .jsonl(.gz) file, a dir of shards, or a plain-text file.

    Filters by `register` only when records actually carry that field (synthetic docs.jsonl
    do; real interim shards don't -> taken whole)."""
    files = []
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.jsonl*"), recursive=True))
    else:
        files = [path]
    texts = []
    for f in files:
        with _open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if f.endswith((".jsonl", ".jsonl.gz")) or line[0] == "{":
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue                    # tolerate a partial last line (concurrent append)
                    if register and "register" in r and r["register"] != register:
                        continue
                    texts.append(r.get("text", ""))
                else:
                    texts.append(line)
    return [t for t in texts if t and t.strip()]


def counts(texts):
    c = Counter()
    for t in texts:
        c.update(tokenize(t))
    return c


def js_divergence(cs, cr):
    """Jensen-Shannon divergence in bits. 0 = identical, 1 = disjoint supports. No smoothing."""
    ns, nr = sum(cs.values()), sum(cr.values())
    js = 0.0
    for w in set(cs) | set(cr):
        p = cs.get(w, 0) / ns
        q = cr.get(w, 0) / nr
        m = 0.5 * (p + q)
        if p:
            js += 0.5 * p * math.log2(p / m)
        if q:
            js += 0.5 * q * math.log2(q / m)
    return js


def kl_divergence(cs, cr, k=0.1):
    """KL(cs||cr) in bits, add-k smoothed over the union vocab (handles real's zeros)."""
    ns, nr = sum(cs.values()), sum(cr.values())
    vocab = set(cs) | set(cr)
    V = len(vocab)
    kl = 0.0
    for w in vocab:
        p = (cs.get(w, 0) + k) / (ns + k * V)
        q = (cr.get(w, 0) + k) / (nr + k * V)
        kl += p * math.log2(p / q)
    return kl


def skew_table(cs, cr, direction, top, min_count):
    """Tokens ranked by contribution to the skew.

    direction='over'  -> most over-represented in synth (p_s * log(p_s/p_r)); the tic detector.
    direction='under' -> most under-represented in synth (p_r * log(p_r/p_s)); the gap detector.
    """
    ns, nr = sum(cs.values()), sum(cr.values())
    rows = []
    for w in set(cs) | set(cr):
        c_s, c_r = cs.get(w, 0), cr.get(w, 0)
        p = (c_s + 0.5) / ns                       # smoothed rates so ratios are finite
        q = (c_r + 0.5) / nr
        if direction == "over":
            if c_s < min_count:
                continue
            score = p * math.log2(p / q)
        else:
            if c_r < min_count:
                continue
            score = q * math.log2(q / p)
        rows.append((score, w, c_s, c_r, 1000 * c_s / ns, 1000 * c_r / nr))
    rows.sort(reverse=True)
    return rows[:top]


def _print_table(title, rows):
    print(f"\n{title}")
    print(f"  {'token':16}{'synth/1k':>10}{'real/1k':>10}{'ratio':>9}")
    for _score, w, _cs, _cr, rs, rr in rows:
        ratio = f"{rs / rr:.0f}x" if rr else "inf"
        print(f"  {w[:16]:16}{rs:>10.2f}{rr:>10.2f}{ratio:>9}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", help="synthetic docs (.jsonl/.gz/dir)")
    ap.add_argument("--real", required=True, help="real reference docs (.jsonl/.gz/dir)")
    ap.add_argument("--register", help="filter records to this register (when field present)")
    ap.add_argument("--n", type=int, default=300, help="docs sampled per side")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--min-count", type=int, default=4, help="min token count to rank (noise floor)")
    ap.add_argument("--control", action="store_true",
                    help="split --real in half and compare halves (sampling-noise JS floor)")
    a = ap.parse_args()

    real = load_texts(a.real, a.register)
    random.Random(a.seed).shuffle(real)

    if a.control:
        half = min(a.n, len(real) // 2)
        A, B = real[:half], real[half:2 * half]
        cA, cB = counts(A), counts(B)
        print(f"CONTROL: real vs real, {half} docs each ({a.register or 'all'})")
        print(f"  Jensen-Shannon divergence = {js_divergence(cA, cB):.4f} bits   (this is the noise floor)")
        return

    if not a.synth:
        raise SystemExit("--synth required unless --control")
    synth = load_texts(a.synth, a.register)
    random.Random(a.seed).shuffle(synth)
    synth, real = synth[:a.n], real[:a.n]
    cs, cr = counts(synth), counts(real)

    print(f"synth: {len(synth)} docs, {sum(cs.values()):,} tokens, {len(cs):,} types "
          f"| real: {len(real)} docs, {sum(cr.values()):,} tokens, {len(cr):,} types "
          f"({a.register or 'all'})")
    print(f"\nJensen-Shannon divergence = {js_divergence(cs, cr):.4f} bits  (0=identical, 1=disjoint)")
    print(f"KL(synth||real) = {kl_divergence(cs, cr):.4f} bits   KL(real||synth) = {kl_divergence(cr, cs):.4f} bits")
    _print_table("OVER-represented in synth (the tic detector):", skew_table(cs, cr, "over", a.top, a.min_count))
    _print_table("UNDER-represented in synth (what it avoids):", skew_table(cs, cr, "under", a.top, a.min_count))


if __name__ == "__main__":
    main()
