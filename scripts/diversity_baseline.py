#!/usr/bin/env python3
"""
diversity_baseline.py - anti-homogenization metrics for Serblo.

Computes, over a set of text samples (one doc per .txt file in a folder, OR one
doc per line in a single file):
  - self-BLEU (2,3,4)        : LOWER  = more diverse (samples less like each other)
  - distinct-n (1,2,3)       : HIGHER = more diverse (more unique n-grams)
  - n-gram entropy (1,2,3)   : HIGHER = more diverse (raw bits + normalized 0-1)

Workflow:
  1) Run on REAL Serbian, per register, to get TARGET numbers:
       python diversity_baseline.py --input real_casual/   --n 500
       python diversity_baseline.py --input real_essay/    --n 500
  2) Run on SYNTHETIC, same register, SAME --n:
       python diversity_baseline.py --input synth_casual.txt --n 500
  3) Compare. Synthetic should land NEAR real.

Rule: identical preprocessing + sample size + register on both sides, or the
comparison is meaningless.

Deps: pip install nltk
"""
import argparse, glob, math, os, random, re
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def tokenize(text):
    text = re.sub(r"<<<[^>]*>>>", " ", text)   # strip generation sentinels (<<<DOC>>> etc.)
    text = text.lower()
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)   # keeps Serbian diacritics


def load_docs(path, n, seed=0):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.txt")))
        docs = [open(f, encoding="utf-8").read().strip() for f in files]
    elif path.endswith((".jsonl", ".jsonl.gz")):
        import gzip, json                      # docs.jsonl / shard files: use the "text" field
        op = gzip.open if path.endswith(".gz") else open
        docs = []
        with op(path, "rt", encoding="utf-8") as f:
            for l in f:
                if not l.strip():
                    continue
                try:
                    docs.append(json.loads(l).get("text", "").strip())
                except json.JSONDecodeError:
                    continue                   # partial last line (concurrent append), skip
    else:
        docs = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    docs = [d for d in docs if d]
    random.Random(seed).shuffle(docs)
    return docs[:n]


def ngrams(toks, n):
    return [tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)]


def self_bleu(tok_docs, max_n=4):
    smooth = SmoothingFunction().method1
    out = {}
    for n in range(2, max_n + 1):
        w = tuple([1.0 / n] * n)
        scores = []
        for i, hyp in enumerate(tok_docs):
            refs = [tok_docs[j] for j in range(len(tok_docs)) if j != i]
            scores.append(sentence_bleu(refs, hyp, weights=w, smoothing_function=smooth))
        out[n] = sum(scores) / len(scores)
    return out


def distinct_n(tok_docs, max_n=3):
    out = {}
    for n in range(1, max_n + 1):
        allg = [g for d in tok_docs for g in ngrams(d, n)]
        out[n] = len(set(allg)) / len(allg) if allg else 0.0
    return out


def ngram_entropy(tok_docs, max_n=3):
    out = {}
    for n in range(1, max_n + 1):
        allg = [g for d in tok_docs for g in ngrams(d, n)]
        counts = Counter(allg)
        total = sum(counts.values())
        if total == 0:
            out[n] = (0.0, 0.0); continue
        H = -sum((c / total) * math.log2(c / total) for c in counts.values())
        Hmax = math.log2(len(counts)) if len(counts) > 1 else 1.0
        out[n] = (H, H / Hmax)        # (raw bits, normalized 0-1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder of .txt OR one-doc-per-line file")
    ap.add_argument("--n", type=int, default=500, help="number of docs to sample")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    docs = load_docs(a.input, a.n, a.seed)
    tok = [t for t in (tokenize(d) for d in docs) if t]
    if not tok:
        raise SystemExit("no usable docs found")
    avg = sum(len(t) for t in tok) / len(tok)
    print(f"\n{a.input}: {len(tok)} docs, avg {avg:.0f} tokens/doc")
    print("(if self-BLEU is too slow, drop --n to ~200-300)\n")

    sb = self_bleu(tok)
    dn = distinct_n(tok)
    en = ngram_entropy(tok)
    print("self-BLEU   (lower = more diverse):", {n: round(v, 4) for n, v in sb.items()})
    print("distinct-n  (higher = more diverse):", {n: round(v, 4) for n, v in dn.items()})
    print("entropy r|norm (higher = more diverse):",
          {n: (round(h, 2), round(hn, 3)) for n, (h, hn) in en.items()})


if __name__ == "__main__":
    main()
