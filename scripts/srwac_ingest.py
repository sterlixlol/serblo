#!/usr/bin/env python3
"""Ingest srWaC 1.1 (.xml.gz VERT) -> clean docs, normalized EXACTLY like the FineWeb-2 backbone
(normalize_text = NFC + Cyrillic->Latin, then declutter, then the same survival/min-word gates),
so a later dedup against the backbone is apples-to-apples.

VERT layout: <p url=...> = one document; token lines are `word\tlemma\ttag`; <g/> = glue (the next
token attaches with no space, for punctuation); <s>/</s> = sentence. We detokenize with the glue
rule, keep the source url for a cheap URL-overlap signal, then run the backbone's normalize chain.

  python3 scripts/srwac_ingest.py data/raw/srwac/srWaC1.1.0*.xml.gz
Emits data/interim-clean/srwac/shard-*.jsonl.gz {text, url, source} + prints size + internal dup rate.
"""
import argparse
import glob
import gzip
import json
import pathlib
import sys

import xxhash

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from serblo.pipeline.declutter import declutter, survival  # noqa: E402
from serblo.pipeline.normalize import normalize_text        # noqa: E402

MIN_SURVIVAL = 0.5
MIN_WORDS = 30


def iter_docs(path):
    """Yield (url, detokenized_text) per <p> block from a srWaC .xml.gz."""
    url, toks, glue = None, [], False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("<p "):
                if toks:
                    yield url, "".join(toks).strip()
                toks, glue = [], False
                url = None
                i = line.find('url="')
                if i != -1:
                    url = line[i + 5: line.find('"', i + 5)]
            elif line.startswith("</p>") or line.startswith("</corpus>"):
                if toks:
                    yield url, "".join(toks).strip()
                toks, glue = [], False
            elif line == "<g/>":
                glue = True
            elif line and not line.startswith("<"):
                w = line.split("\t", 1)[0]
                toks.append(w if glue else " " + w)
                glue = False
    if toks:
        yield url, "".join(toks).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "data" / "interim-clean" / "srwac"))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    seen = set()
    n_raw = n_clean = n_words = n_dup = 0
    gate_short = gate_survival = 0
    idx = 0
    while (out / f"shard-{idx:05d}.jsonl.gz").exists():      # append a new shard, don't clobber
        idx += 1
    shard = out / f"shard-{idx:05d}.jsonl.gz"
    w = gzip.open(shard, "wt", encoding="utf-8")
    for path in sorted(f for pat in a.files for f in glob.glob(pat)):
        print(f"parsing {path} ...", flush=True)
        for url, raw in iter_docs(path):
            if not raw:
                continue
            n_raw += 1
            text = normalize_text(raw)                       # NFC + Cyrillic->Latin (gajica)
            clean = declutter(text)
            if len(clean.split()) < MIN_WORDS:
                gate_short += 1
                continue
            if survival(text, clean) < MIN_SURVIVAL:
                gate_survival += 1
                continue
            h = xxhash.xxh64(clean).intdigest()              # internal exact-dedup
            if h in seen:
                n_dup += 1
                continue
            seen.add(h)
            w.write(json.dumps({"text": clean, "url": url, "source": "srwac"}, ensure_ascii=False) + "\n")
            n_clean += 1
            n_words += len(clean.split())
    w.close()
    print(f"\n=== srWaC ingest ===")
    print(f"  raw docs parsed : {n_raw:,}")
    print(f"  gated (too short {gate_short:,} / low survival {gate_survival:,}) + internal dups {n_dup:,}")
    print(f"  CLEAN unique docs: {n_clean:,}   words: {n_words:,}   (~{int(n_words * 1.577):,} tokens)")
    print(f"  -> {shard}")


if __name__ == "__main__":
    main()
