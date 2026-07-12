"""Build the grounding reference pools from interim real data (PRD §7.4 grounding).

Streams interim shards, probability-samples candidate docs (seeded), excerpts + gates
them (refpool.py), writes ``data/manifests/refpool-{register}.jsonl`` + a meta sidecar.
The pool is a versioned pipeline artifact: same interim data + same seed = same pool
(the manifest sidecar records the pool hash, so a changed pool can't silently mix).

Essay anchors: FineWeb-2 (wiki/web prose) + literature type=prose (NOT folk verse).
Conversational pool is built for future use but grounding.rate ships 0.0 for conv;
subtitle anchors would drag chat toward dubbed-dialogue register.

Usage:
  PYTHONPATH=src python -m serblo.generate.run_refpool --register essay --target 2000
"""
import argparse
import glob
import gzip
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from serblo.generate.refpool import make_anchor  # noqa: E402

SOURCES = {
    "essay": [
        # (glob, source-tag, record-filter, share)
        ("data/interim/fineweb2/*/shard-*.jsonl.gz", "fineweb2", lambda r: True, 0.7),
        ("data/interim/literature/shard-*.jsonl.gz", "literature", lambda r: r.get("type") == "prose", 0.3),
    ],
    "conversational": [
        ("data/interim/opensubtitles/*/shard-*.jsonl.gz", "opensubtitles", lambda r: True, 0.9),
        ("data/interim/forum/reddit/shard-*.jsonl.gz", "forum", lambda r: True, 0.1),
    ],
}


def stream_candidates(pattern, keep, rng, p_keep):
    for f in sorted(glob.glob(str(ROOT / pattern))):
        with gzip.open(f, "rt", encoding="utf-8") as z:
            for line in z:
                if rng.random() >= p_keep:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if keep(r) and r.get("text"):
                    yield r["text"]


def count_docs(pattern):
    n = 0
    for f in sorted(glob.glob(str(ROOT / pattern))):
        with gzip.open(f, "rt", encoding="utf-8") as z:
            n += sum(1 for _ in z)
    return n


def build(register, target, seed, out_dir):
    rng = random.Random(seed)
    out_path = out_dir / f"refpool-{register}.jsonl"
    anchors, by_source, seq = [], {}, 0
    for pattern, tag, keep, share in SOURCES[register]:
        want = int(target * share)
        total = count_docs(pattern)
        if not total:
            print(f"  {tag}: no docs found — skipped", flush=True)
            continue
        p_keep = min(1.0, (want * 4) / total)            # oversample 4x, gates will thin it
        got = 0
        for text in stream_candidates(pattern, keep, rng, p_keep):
            a = make_anchor(text, rng, register, tag, seq)
            if a is None:
                continue
            anchors.append(a)
            seq += 1
            got += 1
            if got >= want:
                break
        by_source[tag] = got
        print(f"  {tag}: {got}/{want} anchors (pool of {total:,} docs)", flush=True)
    rng.shuffle(anchors)
    with open(out_path, "w", encoding="utf-8") as w:
        for a in anchors:
            w.write(json.dumps(a, ensure_ascii=False) + "\n")
    (out_dir / f"refpool-{register}.meta.json").write_text(json.dumps(
        {"register": register, "n": len(anchors), "seed": seed, "by_source": by_source}, indent=2))
    print(f"{register}: {len(anchors)} anchors -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", choices=["essay", "conversational", "both"], default="both")
    ap.add_argument("--target", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "manifests"))
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    regs = ["essay", "conversational"] if a.register == "both" else [a.register]
    for reg in regs:
        build(reg, a.target, a.seed, out_dir)


if __name__ == "__main__":
    main()
