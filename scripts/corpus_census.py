#!/usr/bin/env python3
"""Corpus census: pool every source, report composition, emit the assembly manifest.

The corpus is assembled from many trees: real cleaned shards (data/interim-clean/<source>/)
and synthetic channel outputs (data/generated/<channel>/docs.jsonl). This walks them all,
counts docs / words / estimated tokens grouped by source, register, and model-lineage, and
writes an assembly manifest (the ordered file list + per-source stats) that the dedup step
consumes. Read-only over the data; streams each file (no full-corpus buffering).

EXPERIMENT dirs are excluded by default (vertex-flash / vertex-temp10 / vertex-grounded are
A/B runs on DUPLICATE prompt ids, pooling them double-counts; see STATUS §11 corpus rule).

Usage:
  python scripts/corpus_census.py                      # census + write assembly-manifest.json
  python scripts/corpus_census.py --include-experiments
"""
import argparse
import glob
import gzip
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_PER_WORD = 1.577                                     # our 32k BCS-BPE ratio (M4 measured)

# (glob, kind): real = cleaned interim shards; synth = channel docs.jsonl
SOURCES = [
    ("data/interim-clean/*/**/shard-*.jsonl.gz", "real"),
    ("data/interim-clean/*/shard-*.jsonl.gz", "real"),
    ("data/processed/fineweb2-clean/shard-*.jsonl.gz", "real"),
    ("data/generated/mimo/docs.jsonl", "synth"),
    ("data/generated/mimo-mc/docs.jsonl", "synth"),
    ("data/generated/glm/docs.jsonl", "synth"),
    ("data/generated/kimi/docs.jsonl", "synth"),
    ("data/generated/grok/docs.jsonl", "synth"),
    ("data/generated/vertex/docs.jsonl", "synth"),
    ("data/generated/vertex-conv/docs.jsonl", "synth"),
    ("data/generated/vertex-conv-pillar/docs.jsonl", "synth"),
]
EXPERIMENT = ["vertex-flash", "vertex-temp10", "vertex-grounded"]


def _open(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def lineage(rec, kind):
    if kind == "real":
        return rec.get("source", "real")
    m = (rec.get("model") or "").lower()
    for tag in ("mimo", "glm", "kimi", "grok", "gemini", "flash", "gpt", "deepseek"):
        if tag in m:
            return {"flash": "gemini", "glm": "zhipu-glm", "kimi": "moonshot-kimi",
                    "grok": "xai-grok", "mimo": "xiaomi-mimo"}.get(tag, tag)
    return rec.get("source", "synth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-experiments", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "assembly-manifest.json"))
    a = ap.parse_args()

    by_src = defaultdict(lambda: {"docs": 0, "words": 0, "files": 0})
    by_reg = defaultdict(lambda: {"docs": 0, "words": 0})
    by_lin = defaultdict(lambda: {"docs": 0, "words": 0})
    manifest, seen_files = [], set()

    for pattern, kind in SOURCES:
        for path in sorted(glob.glob(os.path.join(ROOT, pattern), recursive=True)):
            if path in seen_files:
                continue
            seen_files.add(path)
            if not a.include_experiments and any(e in path for e in EXPERIMENT):
                continue
            src = path.split("data/")[1].split("/")[1]     # interim-clean/<src> or generated/<channel>
            fdocs = fwords = 0
            try:
                with _open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        w = len(r.get("text", "").split())
                        fdocs += 1; fwords += w
                        by_reg[r.get("register", "n/a")]["docs"] += 1
                        by_reg[r.get("register", "n/a")]["words"] += w
                        lin = lineage(r, kind)
                        by_lin[lin]["docs"] += 1; by_lin[lin]["words"] += w
            except OSError:
                continue
            by_src[src]["docs"] += fdocs; by_src[src]["words"] += fwords; by_src[src]["files"] += 1
            manifest.append({"path": os.path.relpath(path, ROOT), "kind": kind,
                             "source": src, "docs": fdocs, "words": fwords})

    def table(title, d, extra=""):
        print(f"\n{title}")
        print(f"  {'key':22}{'docs':>10}{'words':>14}{'~tokens':>14}{extra}")
        for k, v in sorted(d.items(), key=lambda kv: -kv[1]["words"]):
            print(f"  {k[:22]:22}{v['docs']:>10,}{v['words']:>14,}{int(v['words']*TOK_PER_WORD):>14,}")

    tot_d = sum(v["docs"] for v in by_src.values())
    tot_w = sum(v["words"] for v in by_src.values())
    print(f"=== CORPUS CENSUS === {tot_d:,} docs | {tot_w:,} words | ~{int(tot_w*TOK_PER_WORD):,} tokens")
    table("by source", by_src)
    table("by register", by_reg)
    table("by lineage", by_lin)
    # real vs synthetic split (the 80/20 target)
    real = sum(v["words"] for s in manifest if s["kind"] == "real" for v in [s])
    synth = sum(s["words"] for s in manifest if s["kind"] == "synth")
    print(f"\nreal vs synthetic (target ~80/20): real {real:,}w ({100*real/max(tot_w,1):.1f}%) | "
          f"synth {synth:,}w ({100*synth/max(tot_w,1):.1f}%)")
    print(f"vs 30B-token corpus target: {100*tot_w*TOK_PER_WORD/30e9:.2f}% assembled")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"total_docs": tot_d, "total_words": tot_w, "files": manifest}, f, indent=1)
    print(f"\nassembly manifest -> {a.out} ({len(manifest)} files)")


if __name__ == "__main__":
    main()
