"""Streaming, sharded, resumable pooled dedup: the scale replacement for run_dedup.

`run_dedup.py` loads every doc + every MinHash signature + a full UnionFind into RAM. That
validates the pipeline on the ~170k M1 slice but cannot survive the real backbone: at 30-40M
docs the signature array alone is ~20GB and the doc text far more. This does the same job,
pooled exact + MinHash-LSH near-dedup, post-transliteration, keeping the first of each cluster,
without ever holding the corpus text or the whole signature set in memory, and resumable at
shard granularity.

How it stays bounded (four passes over disk, never over RAM):
  1. SIGNATURE PASS (streaming, resumable): assign each doc a global id = its position in the
     canonical file+line order, then append four fixed-width columns to scratch: sigs.dat
     (num_perm uint32 per doc; MinHash values are < 2**31 so uint32 is lossless), content.dat
     (uint64 xxh64 of text, for exact dedup), bands.dat (the `bands` per-band xxh64 keys), and
     valid.dat (0 iff the text had no shingles). Text is never stored. Checkpoint per input
     file; on resume, truncate the columns back to the last completed-file boundary and skip
     done files. doc_id is reproducible purely from stream order, so pass 4 re-derives the
     text by re-streaming, so no doc->text map on disk.
  2. EXACT DEDUP: np.unique(content) keeps the first (lowest) doc_id per content hash.
  3. LSH CLUSTER: for each band, stable-argsort its key column so equal keys group with the
     lowest doc_id first (the bucket "base", matching run_dedup's insertion order), verify
     Jaccard>=threshold against base via the memmapped signatures, union in a numpy UnionFind.
     Exact-dups and no-shingle docs are excluded from bucketing (they aren't in run_dedup's
     `ex`), which makes the surviving components identical to the in-memory tool's.
  4. EMIT (streaming, resumable): survivor = the lowest doc_id of each component that isn't an
     exact-dup (== run_dedup's "keep first per cluster"); re-stream the files and write the
     survivors to rolling shards, tallying per-source dup rates on the way.

Parity: on any input it produces the SAME survivor set as run_dedup.py (proven on the M1 slice
and by a planted-duplicate unit test). Memory ceiling is the index arrays (parent ~num_docs*4B,
a few 8B/doc columns) + memmap paging: gigabytes, not the corpus.

Usage:
  PYTHONPATH=src python -m serblo.dedup.run_dedup_scale --interim data/interim --write
  PYTHONPATH=src python -m serblo.dedup.run_dedup_scale --manifest results/assembly-manifest.json --write
"""
import argparse
import glob
import gzip
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import xxhash

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from serblo.dedup.minhash import MinHasher, band_keys  # noqa: E402

# Same source order run_dedup uses, so doc_ids (and thus the parity target) line up.
SHARD_GLOBS = [
    "fineweb2/*/shard-*.jsonl.gz",
    "opensubtitles/*/shard-*.jsonl.gz",
    "forum/reddit/shard-*.jsonl.gz",
    "literature/shard-*.jsonl.gz",
]
_BUCKET_CHUNK = 10000          # cap sigs materialized per LSH bucket (mega-buckets stay flat-memory)


def resolve_files(interim, manifest):
    """Ordered list of input shard paths, from a census manifest or the interim globs."""
    if manifest:
        spec = json.loads(Path(manifest).read_text())
        return [str((ROOT / f["path"]).resolve()) for f in spec["files"]]
    files = []
    for g in SHARD_GLOBS:
        files.extend(sorted(glob.glob(str(Path(interim) / g))))
    return files


def _open(path):
    """Text reader that matches the file: gzip for shards, plain for synth channel docs.jsonl."""
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") \
        else open(path, encoding="utf-8")


def iter_docs(files):
    """Yield every doc across `files` in canonical order; this defines doc_id (= yield index)."""
    for f in files:
        with _open(f) as z:
            for line in z:
                if line.strip():
                    yield json.loads(line)


# ---- scratch column layout (fixed-width, doc_id = row index) ---------------------------------

class Columns:
    """The four append-only scratch columns, truncatable to a doc-count boundary on resume."""

    def __init__(self, work, num_perm, bands):
        self.work, self.num_perm, self.bands = Path(work), num_perm, bands
        self.sigs = self.work / "sigs.dat"        # (N, num_perm) uint32
        self.content = self.work / "content.dat"  # (N,) uint64
        self.bandk = self.work / "bands.dat"      # (N, bands) uint64
        self.valid = self.work / "valid.dat"      # (N,) uint8

    def _widths(self):
        return {self.sigs: self.num_perm * 4, self.content: 8,
                self.bandk: self.bands * 8, self.valid: 1}

    def truncate_to(self, n):
        """Drop any rows past the first `n` docs (undo a crashed file's partial writes)."""
        for path, w in self._widths().items():
            if path.exists():
                with open(path, "r+b") as fh:
                    fh.truncate(n * w)

    def appenders(self):
        return {k: open(k, "ab") for k in self._widths()}

    def memmap(self, n):
        return (np.memmap(self.sigs, dtype=np.uint32, mode="r", shape=(n, self.num_perm)),
                np.memmap(self.content, dtype=np.uint64, mode="r", shape=(n,)),
                np.memmap(self.bandk, dtype=np.uint64, mode="r", shape=(n, self.bands)),
                np.memmap(self.valid, dtype=np.uint8, mode="r", shape=(n,)))


def write_json(path, obj):
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


# ---- pass 1: signatures ----------------------------------------------------------------------

def stage1_signatures(files, work, cfg):
    """Stream docs -> append sig / content / band-key / valid columns. Resumable per file."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    prog_path = work / "progress.json"
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else {"files_done": [], "docs_done": 0}

    cols = Columns(work, cfg["num_perm"], cfg["bands"])
    cols.truncate_to(prog["docs_done"])           # discard a partial file's rows from a prior crash
    done = set(prog["files_done"])
    mh = MinHasher(cfg["num_perm"], cfg["k"])     # seed=0 default -> identical perms to run_dedup
    t0 = time.time()

    for f in files:
        if f in done:
            continue
        s_buf, c_buf, b_buf, v_buf = [], [], [], []
        with _open(f) as z:
            for line in z:
                if not line.strip():
                    continue
                text = json.loads(line)["text"]
                c_buf.append(xxhash.xxh64(text).intdigest())
                sig = mh.signature(text)
                if sig is None:                    # empty text -> no shingles -> LSH-excluded singleton
                    s_buf.append(np.zeros(cfg["num_perm"], np.uint32))
                    b_buf.append(np.zeros(cfg["bands"], np.uint64))
                    v_buf.append(0)
                else:
                    s_buf.append(sig.astype(np.uint32))   # values < 2**31, lossless
                    b_buf.append(np.fromiter((h for _, h in band_keys(sig, cfg["bands"], cfg["rows"])),
                                             dtype=np.uint64, count=cfg["bands"]))
                    v_buf.append(1)
        fhs = cols.appenders()
        fhs[cols.sigs].write(np.asarray(s_buf, np.uint32).tobytes())
        fhs[cols.content].write(np.asarray(c_buf, np.uint64).tobytes())
        fhs[cols.bandk].write(np.asarray(b_buf, np.uint64).tobytes())
        fhs[cols.valid].write(np.asarray(v_buf, np.uint8).tobytes())
        for fh in fhs.values():
            fh.flush(); os.fsync(fh.fileno()); fh.close()   # data hits disk BEFORE the pointer moves
        prog["files_done"].append(f)
        prog["docs_done"] += len(c_buf)
        write_json(prog_path, prog)
        print(f"  sig {Path(f).name}: +{len(c_buf):,} -> {prog['docs_done']:,} "
              f"({(time.time()-t0)/60:.1f}m)", flush=True)
    return prog["docs_done"]


# ---- pass 2/3: exact dedup + LSH clustering --------------------------------------------------

def stage2_exact(content):
    """Mark every content-hash duplicate except the first (lowest doc_id)."""
    _, first = np.unique(np.asarray(content), return_index=True)   # first = min doc_id per hash
    exact_dup = np.ones(len(content), bool)
    exact_dup[first] = False
    return exact_dup


def _find(parent, x):
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:                       # path compression
        parent[x], x = root, parent[x]
    return root


def stage3_cluster(sigs, bandk, valid, exact_dup, cfg):
    """Union near-duplicates band by band; returns each doc's component root (flattened)."""
    n = len(valid)
    parent = np.arange(n, dtype=np.int64)
    active = (valid.astype(bool)) & (~exact_dup)   # only docs that reach run_dedup's LSH phase
    thr = cfg["threshold"]

    for b in range(cfg["bands"]):
        col = np.asarray(bandk[:, b])
        order = np.argsort(col, kind="stable")     # equal keys grouped, lowest doc_id first
        scol = col[order]
        change = np.ones(len(scol), bool)
        if len(scol) > 1:
            np.not_equal(scol[1:], scol[:-1], out=change[1:])
        starts = np.flatnonzero(change)
        ends = np.append(starts[1:], len(scol))
        for s, e in zip(starts, ends):
            if e - s < 2:
                continue
            grp = order[s:e]
            grp = grp[active[grp]]                  # drop exact-dups / no-shingle docs
            if grp.size < 2:
                continue
            base = int(grp[0])
            others = grp[1:]
            bs = np.asarray(sigs[base])
            for c in range(0, len(others), _BUCKET_CHUNK):     # a boilerplate mega-bucket would
                block = others[c:c + _BUCKET_CHUNK]            # otherwise materialize all sigs at once
                frac = (np.asarray(sigs[block]) == bs).mean(axis=1)
                for j in block[frac >= thr]:
                    rj, rb = _find(parent, int(j)), _find(parent, base)
                    if rj != rb:
                        parent[rj] = rb
    for _ in range(64):                            # pointer-doubling -> every node points at its root
        nxt = parent[parent]
        if np.array_equal(nxt, parent):
            break
        parent = nxt
    return parent


def survivors_mask(roots, exact_dup):
    """Survivor = lowest doc_id of each component that isn't an exact-dup (run_dedup's rule)."""
    keep_candidates = np.flatnonzero(~exact_dup)   # ascending doc_ids still in play
    _, first = np.unique(roots[keep_candidates], return_index=True)
    mask = np.zeros(len(roots), bool)
    mask[keep_candidates[first]] = True
    return mask


# ---- pass 4: emit ----------------------------------------------------------------------------

def stage4_emit(files, out_dir, survivor, exact_dup, shard_docs, do_write):
    """Re-stream in doc_id order, write survivors to rolling shards, tally per-source dup rates."""
    out_dir = Path(out_dir)
    if do_write:
        out_dir.mkdir(parents=True, exist_ok=True)
    loaded, exact_by, near_by, surv_by = Counter(), Counter(), Counter(), Counter()
    buf, seq, n_surv = [], 0, 0

    def flush():
        nonlocal seq, buf
        if not (do_write and buf):
            buf = []
            return
        path = out_dir / f"shard-{seq:05d}.jsonl.gz"
        tmp = path.with_suffix(".gz.tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as w:
            for d in buf:
                w.write(json.dumps(d, ensure_ascii=False) + "\n")
        tmp.replace(path)
        seq += 1
        buf = []

    for doc_id, d in enumerate(iter_docs(files)):
        src = d.get("source")
        loaded[src] += 1
        if survivor[doc_id]:
            n_surv += 1
            surv_by[src] += 1
            if do_write:
                buf.append(d)
                if len(buf) >= shard_docs:
                    flush()
        elif exact_dup[doc_id]:
            exact_by[src] += 1
        else:
            near_by[src] += 1
    flush()

    print("\n=== dedup summary ===")
    print(f"{'source':14}{'loaded':>10}{'exact':>9}{'near':>9}{'unique':>10}{'dup%':>8}")
    tot = sum(loaded.values())
    for s in loaded:
        lo, sv = loaded[s], surv_by[s]
        print(f"{str(s):14}{lo:>10,}{exact_by[s]:>9,}{near_by[s]:>9,}{sv:>10,}{1-sv/max(lo,1):>7.1%}")
    print(f"{'TOTAL':14}{tot:>10,}{sum(exact_by.values()):>9,}"
          f"{sum(near_by.values()):>9,}{n_surv:>10,}{1-n_surv/max(tot,1):>7.1%}")
    if do_write:
        write_json(out_dir / "manifest.json",
                   {"unique_docs": n_surv, "shards": seq,
                    "exact_dups": sum(exact_by.values()), "near_dups": sum(near_by.values())})
        print(f"\nwrote {n_surv:,} unique docs to {out_dir} ({seq} shards)", flush=True)
    return n_surv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interim", default=str(ROOT / "data" / "interim"))
    ap.add_argument("--manifest", default=None, help="census assembly-manifest.json (pooled real+synth)")
    ap.add_argument("--work", default=str(ROOT / "data" / "processed" / "dedup-work"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "dedup"))
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--bands", type=int, default=16)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--shard-docs", type=int, default=5000)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    assert a.bands * a.rows == a.num_perm, "bands*rows must equal num_perm"
    cfg = {"num_perm": a.num_perm, "bands": a.bands, "rows": a.rows,
           "threshold": a.threshold, "k": a.k}

    files = resolve_files(a.interim, a.manifest)
    print(f"input: {len(files)} shards | work {a.work}", flush=True)

    # config guard: refuse to resume scratch built under different params (silent corruption otherwise)
    work = Path(a.work)
    meta_path = work / "meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old.get("cfg") != cfg or old.get("files") != files:
            sys.exit(f"work dir {work} was built with different params/inputs — clear it or pass a new --work")
    else:
        work.mkdir(parents=True, exist_ok=True)
        write_json(meta_path, {"cfg": cfg, "files": files})

    t0 = time.time()
    n = stage1_signatures(files, work, cfg)
    print(f"signatures: {n:,} docs | {(time.time()-t0)/60:.1f}m", flush=True)

    sigs, content, bandk, valid = Columns(work, cfg["num_perm"], cfg["bands"]).memmap(n)
    t0 = time.time()
    exact_dup = stage2_exact(content)
    print(f"exact-dedup: {int(exact_dup.sum()):,} dups | {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    roots = stage3_cluster(sigs, bandk, valid, exact_dup, cfg)
    survivor = survivors_mask(roots, exact_dup)
    print(f"lsh-cluster: {int(survivor.sum()):,} survivors | {(time.time()-t0)/60:.1f}m", flush=True)

    stage4_emit(files, a.out, survivor, exact_dup, a.shard_docs, a.write)


if __name__ == "__main__":
    main()
