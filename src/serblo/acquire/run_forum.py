"""Process scraped reddit threads into the normalized corpus (PRD M1).

Reads the raw scrape (data/raw/forum/reddit.jsonl), extracts text
(reddit.parse_thread), normalizes (NFC + Cyrillic->Latin), drops thin threads, and
shards to data/interim/forum/reddit/ as gzipped JSONL. Input is a finite local file,
so this just processes it once; shards are flushed incrementally.

No mojibake repair here: reddit is clean UTF-8 (the CP1250/CP1251 fix is subtitle-
specific and would only risk misfiring).

Usage:
  PYTHONPATH=src python -m serblo.acquire.run_forum
"""
import argparse
import gzip
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from serblo.acquire.reddit import parse_thread
from serblo.pipeline.normalize import normalize_text, has_cyrillic, diacritic_ratio


def process(in_path, out_dir, shard_size, min_words):
    out_dir.mkdir(parents=True, exist_ok=True)
    shard, idx, kept, skipped = [], 0, 0, 0
    per_sub = Counter()
    diac_sum, cyr = 0.0, 0

    def flush():
        nonlocal shard, idx, diac_sum, cyr
        if not shard:
            return
        name = f"shard-{idx:05d}.jsonl.gz"
        tmp = out_dir / (name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as w:
            for rec in shard:
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, out_dir / name)
        print(f"  shard {idx} (+{len(shard)}) diac~{diac_sum/len(shard):.3f} cyr_resid={cyr}", flush=True)
        idx += 1
        shard = []
        diac_sum = 0.0
        cyr = 0

    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = normalize_text(parse_thread(rec.get("raw_md", "")))
            if len(text.split()) < min_words:
                skipped += 1
                continue
            out = {"text": text, "source": "reddit",
                   "subreddit": rec.get("subreddit"), "url": rec.get("url")}
            shard.append(out)
            kept += 1
            per_sub[rec.get("subreddit")] += 1
            diac_sum += diacritic_ratio(text)
            if has_cyrillic(text):
                cyr += 1
            if len(shard) >= shard_size:
                flush()
    flush()

    manifest = {"source": "reddit", "docs_written": kept, "skipped_thin": skipped,
                "shards": idx, "per_subreddit": dict(per_sub)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=0))
    print(f"\nkept {kept} threads, skipped {skipped} thin (<{min_words} words), {idx} shards", flush=True)
    print("per subreddit:", dict(per_sub), flush=True)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=str(ROOT / "data" / "raw" / "forum" / "reddit.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "interim" / "forum" / "reddit"))
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--min-words", type=int, default=15)
    a = ap.parse_args()

    in_path = Path(a.in_path)
    if not in_path.exists():
        raise SystemExit(f"no scrape file at {in_path} — run the reddit scraper first")
    print(f"process {in_path} -> {a.out}", flush=True)
    process(in_path, Path(a.out), a.shard_size, a.min_words)


if __name__ == "__main__":
    main()
