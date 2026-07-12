"""Pooled exact + near-duplicate dedup across all M1 sources (PRD M2 stage 4).

Loads every interim shard, exact-dedups by content hash, then MinHash+LSH near-dedups
across the POOLED corpus. Because it runs post-transliteration, it collapses cross-script
Serbian twins (same article once Cyrillic, once Latin) and cross-source overlap (literature
vs FineWeb-2 wiki content, subtitle repeats). Reports dup rates per source; --write emits
survivors to data/processed/dedup/.

Validates the pipeline on the M1 slice; full-scale needs sharding / a high-RAM box (PRD §7.3).

Usage:
  PYTHONPATH=src python -m serblo.dedup.run_dedup --write
"""
import argparse
import glob
import gzip
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import xxhash

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from serblo.dedup.minhash import MinHasher, jaccard, band_keys, UnionFind

SHARD_GLOBS = [
    "fineweb2/*/shard-*.jsonl.gz",
    "opensubtitles/*/shard-*.jsonl.gz",
    "forum/reddit/shard-*.jsonl.gz",
    "literature/shard-*.jsonl.gz",
]


def load_docs(interim):
    docs = []
    for g in SHARD_GLOBS:
        for f in sorted(glob.glob(str(interim / g))):
            with gzip.open(f, "rt", encoding="utf-8") as z:
                for line in z:
                    docs.append(json.loads(line))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interim", default=str(ROOT / "data" / "interim"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "dedup"))
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--bands", type=int, default=16)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    assert a.bands * a.rows == a.num_perm, "bands*rows must equal num_perm"
    interim = Path(a.interim)

    t0 = time.time()
    docs = load_docs(interim)
    orig = Counter(d.get("source") for d in docs)
    print(f"loaded {len(docs):,} docs in {time.time()-t0:.0f}s", flush=True)
    for s, n in orig.most_common():
        print(f"  {s}: {n:,}")

    # exact dedup
    t0 = time.time()
    seen, ex, exact_dups = set(), [], Counter()
    for d in docs:
        h = xxhash.xxh64(d["text"]).intdigest()
        if h in seen:
            exact_dups[d.get("source")] += 1
            continue
        seen.add(h)
        ex.append(d)
    print(f"\nexact-dedup: {len(docs):,} -> {len(ex):,} "
          f"({sum(exact_dups.values()):,} exact dups) in {time.time()-t0:.0f}s", flush=True)

    # minhash signatures
    t0 = time.time()
    mh = MinHasher(a.num_perm, a.k)
    sigs = [mh.signature(d["text"]) for d in ex]
    print(f"minhash: {len(ex):,} signatures in {time.time()-t0:.0f}s", flush=True)

    # LSH bucketing + verified union
    t0 = time.time()
    buckets = defaultdict(list)
    for i, sig in enumerate(sigs):
        if sig is None:
            continue
        for key in band_keys(sig, a.bands, a.rows):
            buckets[key].append(i)
    uf = UnionFind(len(ex))
    comparisons = 0
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        base = idxs[0]
        for j in idxs[1:]:
            comparisons += 1
            if jaccard(sigs[base], sigs[j]) >= a.threshold:
                uf.union(base, j)
    print(f"lsh: {len(buckets):,} buckets, {comparisons:,} verifications in {time.time()-t0:.0f}s", flush=True)

    # keep first per cluster
    cluster_sources = defaultdict(set)
    for i, d in enumerate(ex):
        cluster_sources[uf.find(i)].add(d.get("source"))
    roots, survivors, near_dups = set(), [], Counter()
    for i, d in enumerate(ex):
        r = uf.find(i)
        if r in roots:
            near_dups[d.get("source")] += 1
            continue
        roots.add(r)
        survivors.append(d)
    multi = sum(1 for ss in cluster_sources.values() if len(ss) > 1)

    # report
    surv_by = Counter(d.get("source") for d in survivors)
    print("\n=== dedup summary (M1 slice) ===")
    print(f"{'source':14}{'loaded':>10}{'exact':>9}{'near':>9}{'unique':>10}{'dup%':>8}")
    for s in orig:
        lo, sv = orig[s], surv_by[s]
        print(f"{s:14}{lo:>10,}{exact_dups[s]:>9,}{near_dups[s]:>9,}{sv:>10,}{1-sv/lo:>7.1%}")
    print(f"{'TOTAL':14}{len(docs):>10,}{sum(exact_dups.values()):>9,}"
          f"{sum(near_dups.values()):>9,}{len(survivors):>10,}{1-len(survivors)/len(docs):>7.1%}")
    print(f"cross-source dup clusters (span >1 source): {multi:,}", flush=True)

    if a.write:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        idx = 0
        for i in range(0, len(survivors), 5000):
            with gzip.open(out / f"shard-{idx:05d}.jsonl.gz", "wt", encoding="utf-8") as w:
                for d in survivors[i:i + 5000]:
                    w.write(json.dumps(d, ensure_ascii=False) + "\n")
            idx += 1
        (out / "manifest.json").write_text(json.dumps(
            {"unique_docs": len(survivors), "shards": idx,
             "exact_dups": sum(exact_dups.values()), "near_dups": sum(near_dups.values())},
            ensure_ascii=False, indent=0))
        print(f"\nwrote {len(survivors):,} unique docs to {out} ({idx} shards)", flush=True)


if __name__ == "__main__":
    main()
