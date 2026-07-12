#!/usr/bin/env python3
"""Multi-core tokenize + pack (P1c). Same output as pack_corpus.py but uses tok.encode_batch(),
which parallelizes tokenization across all cores (Rayon) -> ~Ncores faster. Reuses pack.py's
tested atomic writers. Docs separated by eos_id; every val_every-th doc -> val split.

  PYTHONPATH=src python3 scripts/pack_corpus_fast.py --in 'backbone/*.jsonl.gz' --out tokenized \
      --tokenizer data/tokenizer/tokenizer.json --shard-tokens 100000000 --val-every 200 --batch 16384
"""
import argparse
import glob
import gzip
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from serblo.tokenizer import load_tokenizer                              # noqa: E402
from serblo.train.pack import _atomic_write_json, _atomic_write_u16      # noqa: E402  (tested writers)


def doc_stream(files):
    for f in sorted(files):
        with gzip.open(f, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    t = json.loads(line).get("text", "").strip()
                except json.JSONDecodeError:
                    continue
                if t:
                    yield t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=str(ROOT / "data/tokenizer/tokenizer.json"))
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16384)
    a = ap.parse_args()

    tok = load_tokenizer(a.tokenizer)
    eos = tok.token_to_id("<|endoftext|>")
    sha = hashlib.sha256(pathlib.Path(a.tokenizer).read_bytes()).hexdigest()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = glob.glob(a.inp)
    print(f"packing {len(files)} shards -> {a.out} (eos={eos}, batch={a.batch}, val 1/{a.val_every}) ...", flush=True)

    st = {"shard": 0, "train_tokens": 0, "val_tokens": 0}
    train_buf, val_buf = [], []

    def flush():
        if train_buf:
            _atomic_write_u16(out / f"train_{st['shard']:05d}.bin", train_buf)
            st["train_tokens"] += len(train_buf)
            train_buf.clear()
        if val_buf:
            _atomic_write_u16(out / f"val_{st['shard']:05d}.bin", val_buf)
            st["val_tokens"] += len(val_buf)
            val_buf.clear()
        st["shard"] += 1

    gi = 0
    batch = []

    def do_batch():
        nonlocal batch
        for j, enc in enumerate(tok.encode_batch(batch)):          # PARALLEL across all cores
            ids = enc.ids + [eos]
            (val_buf if (gi + j) % a.val_every == a.val_every - 1 else train_buf).extend(ids)
            if len(train_buf) >= a.shard_tokens:
                flush()

    for d in doc_stream(files):
        batch.append(d)
        if len(batch) >= a.batch:
            do_batch()
            gi += len(batch)
            batch = []
            if st["shard"] and st["shard"] % 20 == 0 and (gi // a.batch) % 50 == 0:
                print(f"  ~{st['shard']} shards, {st['train_tokens']:,} train tok", flush=True)
    if batch:
        do_batch()
        gi += len(batch)
    if train_buf or val_buf:
        flush()

    meta = {"vocab_size": tok.get_vocab_size(), "eos_id": eos, "shard_dtype": "uint16",
            "train_tokens": st["train_tokens"], "val_tokens": st["val_tokens"], "shards": st["shard"],
            "tokenizer_sha256": sha,
            "train_shards": sorted(p.name for p in out.glob("train_*.bin")),
            "val_shards": sorted(p.name for p in out.glob("val_*.bin"))}
    _atomic_write_json(out / "meta.json", meta)
    print(f"DONE: {meta['train_tokens']:,} train + {meta['val_tokens']:,} val tok "
          f"({len(meta['train_shards'])} train shards)")


if __name__ == "__main__":
    main()
