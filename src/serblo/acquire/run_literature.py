"""Ingest scraped Serbian literature (folk/ + prose/) into the corpus (PRD M1).

Reads one .txt per work, normalizes (Cyrillic->Latin gajica), drops non-Serbian files
(archive.org keyword-search false positives, e.g. German books) via a Serbian-stopword
gate, strips archive.org/library boilerplate lines, chunks into ~CHUNK-word passages, and
shards to data/interim/literature/. Provenance is tagged (archive.org = scanned OCR, risk;
wikisource = hand-transcribed, clean) for later dosing (folk = small-dose) and quality passes.

Usage:
  PYTHONPATH=src python -m serblo.acquire.run_literature --src ~/Desktop/serbian_scrape
"""
import argparse
import glob
import gzip
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from serblo.pipeline.normalize import normalize_text, diacritic_ratio

SR_STOP = set(("i u je da se na za od koji su nije bio sam ne sto o a ali kao li ga mu to ovo "
               "ono ce bi ili kad sve svoj koja koje jedan bila samo pa te tako gde kako jeste biti").split())
_BOILER = ("digitized by", "archive.org", "funding from", "http://", "https://", "google", "university of")


def _sr_rate(text):
    toks = [x.lower().strip('.,!?;:"\'()»«-') for x in text.split()]
    toks = [x for x in toks if x]
    return (sum(1 for x in toks if x in SR_STOP) / len(toks)) if toks else 0.0


def _strip_boiler(text):
    return "\n".join(ln for ln in text.split("\n") if not any(b in ln.lower() for b in _BOILER))


def _chunks(words, size):
    for i in range(0, len(words), size):
        yield words[i:i + size]


def ingest(src_root, out_dir, chunk, min_sr, shard_size):
    out_dir.mkdir(parents=True, exist_ok=True)
    shard, idx = [], 0
    kept_docs = kept_words = 0
    per, diac_sum, diac_n = Counter(), Counter(), Counter()
    dropped = Counter()
    dropped_examples = []

    def flush():
        nonlocal shard, idx
        if not shard:
            return
        name = f"shard-{idx:05d}.jsonl.gz"
        tmp = out_dir / (name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as w:
            for r in shard:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, out_dir / name)
        idx += 1
        shard = []

    for cat in ("folk", "prose"):
        for f in sorted(glob.glob(f"{src_root}/{cat}/*.txt")):
            prov = "archive.org" if os.path.basename(f).startswith("ia_") else "wikisource"
            norm = normalize_text(_strip_boiler(open(f, encoding="utf-8", errors="replace").read()))
            words = norm.split()
            rate = _sr_rate(norm)
            # Only archive.org is keyword-scraped (false-positive risk, e.g. German books) ->
            # language-gate it. Wikisource is hand-transcribed Serbian -> trust it; drop only
            # empty/stub extractions, and keep short folk poems (the stopword rate is noisy on those).
            if prov == "archive.org":
                drop = len(words) < 20 or rate < min_sr
            else:
                drop = len(words) < 8
            if drop:
                dropped[(cat, prov)] += 1
                if len(dropped_examples) < 20:
                    dropped_examples.append((os.path.basename(f), prov, round(rate, 3), len(words)))
                continue
            for cw in _chunks(words, chunk):
                if len(cw) < 20:
                    continue
                text = " ".join(cw)
                shard.append({"text": text, "source": "literature", "type": cat,
                              "provenance": prov, "file": os.path.basename(f)})
                kept_docs += 1
                kept_words += len(cw)
                per[(cat, prov)] += 1
                diac_sum[(cat, prov)] += diacritic_ratio(text)
                diac_n[(cat, prov)] += 1
                if len(shard) >= shard_size:
                    flush()
    flush()

    print(f"\nkept {kept_docs:,} docs, {kept_words:,} words, {idx} shards", flush=True)
    for k in sorted(per):
        print(f"  {k[0]:5} {k[1]:11}: {per[k]:>6} docs   diac~{diac_sum[k]/max(1,diac_n[k]):.4f}")
    print("dropped files (non-Serbian / too-short):", dict(dropped))
    if dropped_examples:
        print("dropped examples (file, sr_rate, words):", dropped_examples)

    manifest = {"source": "literature", "docs": kept_docs, "words": kept_words, "shards": idx,
                "per_bucket": {f"{a}/{b}": v for (a, b), v in per.items()},
                "dropped": {f"{a}/{b}": v for (a, b), v in dropped.items()}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=0))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.expanduser("~/Desktop/serbian_scrape"))
    ap.add_argument("--out", default=str(ROOT / "data" / "interim" / "literature"))
    ap.add_argument("--chunk", type=int, default=400, help="words per document")
    ap.add_argument("--min-sr", type=float, default=0.05, help="min Serbian-stopword rate to keep a file")
    ap.add_argument("--shard-size", type=int, default=2000)
    a = ap.parse_args()
    print(f"ingest literature from {a.src} -> {a.out} (chunk={a.chunk}, min_sr={a.min_sr})", flush=True)
    ingest(a.src, Path(a.out), a.chunk, a.min_sr, a.shard_size)


if __name__ == "__main__":
    main()
