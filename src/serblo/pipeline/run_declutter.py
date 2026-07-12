"""Apply declutter across interim shards -> cleaned interim (M2 cleaning pass, PRD §8).

Streams each interim shard, declutters every doc, and writes cleaned shards to a parallel
tree (data/interim-clean/... by default) so the raw interim stays intact for comparison.
A doc whose word-count survival falls below --min-survival (mostly chrome/nav) or whose
cleaned length is below --min-words is DROPPED and counted. Prints the survival histogram
and drop rate; silent cleaning that quietly deletes half the corpus is the nightmare, so
the numbers are always shown.

Idempotent per shard (skips shards already written). Reports let you sanity-check before
pointing dedup + gates at the cleaned tree.

Usage:
  PYTHONPATH=src python -m serblo.pipeline.run_declutter --source fineweb2
"""
import argparse
import glob
import gzip
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from serblo.pipeline.declutter import declutter, survival  # noqa: E402

MIN_SURVIVAL = 0.5
MIN_WORDS = 20


def process(source, interim, out_root, min_survival, min_words):
    shards = sorted(glob.glob(str(interim / source / "**" / "shard-*.jsonl.gz"), recursive=True))
    if not shards:
        print(f"no shards for {source} under {interim/source}", flush=True)
        return
    kept = dropped = 0
    words_in = words_out = 0
    surv_hist = Counter()
    drop_reasons = Counter()
    for shard in shards:
        rel = Path(shard).relative_to(interim)
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():                                     # idempotent resume
            continue
        tmp = dst.with_suffix(".tmp")
        with gzip.open(shard, "rt", encoding="utf-8") as z, gzip.open(tmp, "wt", encoding="utf-8") as w:
            for line in z:
                d = json.loads(line)
                orig = d.get("text", "")
                clean = declutter(orig)
                s = survival(orig, clean)
                words_in += len(orig.split())
                surv_hist[round(s, 1)] += 1
                if len(clean.split()) < min_words:
                    dropped += 1; drop_reasons["too_short_after_clean"] += 1; continue
                if s < min_survival:
                    dropped += 1; drop_reasons["mostly_chrome"] += 1; continue
                d["text"] = clean
                d["decluttered"] = round(s, 3) if s < 0.999 else None
                w.write(json.dumps(d, ensure_ascii=False) + "\n")
                kept += 1
                words_out += len(clean.split())
        tmp.replace(dst)
    total = kept + dropped
    print(f"\n=== declutter {source} ===")
    print(f"docs {total:,} -> kept {kept:,} ({100*kept/max(total,1):.1f}%), dropped {dropped:,}")
    print(f"words {words_in:,} -> {words_out:,} ({100*words_out/max(words_in,1):.1f}% of prose retained)")
    print(f"drop reasons: {dict(drop_reasons)}")
    print("survival histogram (word fraction surviving):")
    for band in sorted(surv_hist):
        print(f"  {band:>4.1f}: {surv_hist[band]:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="interim source dir (fineweb2, opensubtitles, ...)")
    ap.add_argument("--interim", default=str(ROOT / "data" / "interim"))
    ap.add_argument("--out", default=str(ROOT / "data" / "interim-clean"))
    ap.add_argument("--min-survival", type=float, default=MIN_SURVIVAL)
    ap.add_argument("--min-words", type=int, default=MIN_WORDS)
    a = ap.parse_args()
    process(a.source, Path(a.interim), Path(a.out), a.min_survival, a.min_words)


if __name__ == "__main__":
    main()
