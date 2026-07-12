"""Decontaminate training shards against the eval/probe set (PRD M5 runner).

Streams the (already deduped) training corpus and drops any doc that shares a 13-gram with an
eval/probe item, writing clean shards + a report. The eval index is tiny (a few hundred short
prompts), so this holds a flat footprint over the full backbone; the training side is streamed
one doc at a time and resumable per shard (a crash re-does at most one shard).

The eval set is whatever the probe set ends up being (PRD §7.8: a few hundred conversational +
essay prompts, hand-built). Point --eval at a .jsonl (uses "text", else "prompt"/"question") or
a .txt (one item per line). This runner exists ahead of the probe set so M5 is one command away
once the set is built.

Usage:
  PYTHONPATH=src python -m serblo.eval.run_decontam --train data/processed/dedup --eval data/eval/probe.jsonl --write
"""
import argparse
import glob
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from serblo.eval.decontam import DEFAULT_N, build_eval_index, contamination  # noqa: E402


def _open(path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") \
        else open(path, encoding="utf-8")


def load_eval_texts(path):
    """Eval/probe items -> list of strings. .jsonl uses text/prompt/question; .txt is one per line."""
    p = Path(path)
    if p.suffix == ".txt":
        return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    texts = []
    with _open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = r.get("text") or r.get("prompt") or r.get("question")
            if t:
                texts.append(t)
    return texts


def train_files(train_dir, manifest):
    if manifest:
        spec = json.loads(Path(manifest).read_text())
        return [str((ROOT / f["path"]).resolve()) for f in spec["files"]]
    return sorted(glob.glob(str(Path(train_dir) / "**" / "shard-*.jsonl.gz"), recursive=True)) \
        or sorted(glob.glob(str(Path(train_dir) / "shard-*.jsonl.gz")))


def write_json(path, obj):
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(ROOT / "data" / "processed" / "dedup"))
    ap.add_argument("--manifest", default=None, help="assembly-manifest.json instead of --train glob")
    ap.add_argument("--eval", required=True, help="probe set: .jsonl (text/prompt) or .txt")
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "decontam"))
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="n-gram size (default 13, GPT-3/Pile)")
    ap.add_argument("--shard-docs", type=int, default=5000)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    eval_texts = load_eval_texts(a.eval)
    index = build_eval_index(eval_texts, a.n)
    print(f"eval set: {len(eval_texts):,} items -> {len(index):,} {a.n}-grams", flush=True)
    if not index:
        sys.exit("empty eval index — nothing to decontaminate against")

    files = train_files(a.train, a.manifest)
    print(f"train: {len(files)} shards", flush=True)
    out_dir = Path(a.out)
    prog_path = out_dir / "progress.json"
    if a.write:
        out_dir.mkdir(parents=True, exist_ok=True)
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else {"files_done": [], "seq": 0}
    done = set(prog["files_done"])

    kept = removed = scanned = 0
    removed_by_src, hit_evalgram = Counter(), Counter()
    buf, seq = [], prog["seq"]

    def flush():
        nonlocal seq, buf
        if a.write and buf:
            path = out_dir / f"shard-{seq:05d}.jsonl.gz"
            tmp = path.with_suffix(".gz.tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as w:
                for d in buf:
                    w.write(json.dumps(d, ensure_ascii=False) + "\n")
            tmp.replace(path)
            seq += 1
        buf = []

    for f in files:
        if f in done:
            continue
        with _open(f) as z:
            for line in z:
                if not line.strip():
                    continue
                d = json.loads(line)
                scanned += 1
                hit = contamination(d.get("text", ""), index, a.n)
                if hit is None:
                    kept += 1
                    if a.write:
                        buf.append(d)
                        if len(buf) >= a.shard_docs:
                            flush()
                else:
                    removed += 1
                    removed_by_src[d.get("source")] += 1
                    hit_evalgram[hit] += 1
        prog["files_done"].append(f)                  # checkpoint per shard
        if a.write:
            flush()                                    # shard boundary = safe resume point
            prog["seq"] = seq
            write_json(prog_path, prog)

    print(f"\n=== decontam ===\nscanned {scanned:,} | kept {kept:,} | removed {removed:,} "
          f"({removed/max(scanned,1):.2%})", flush=True)
    for s, n in removed_by_src.most_common():
        print(f"  removed from {s}: {n:,}")
    if hit_evalgram:                                   # a probe item matching MANY docs is too generic
        print("top eval n-grams by train hits (watch for over-broad probes):")
        for g, n in hit_evalgram.most_common(5):
            print(f"  {n:,}× :: {g[:70]}")
    if a.write:
        write_json(out_dir / "manifest.json", {"kept": kept, "removed": removed, "n": a.n})
        print(f"wrote {kept:,} clean docs -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
