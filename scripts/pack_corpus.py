#!/usr/bin/env python3
"""Tokenize *.jsonl.gz corpus shards into uint16 training shards. Works for the sample now and
the full backbone later (same tool). Resumable via pack.pack_docs.

  PYTHONPATH=src python3 scripts/pack_corpus.py --in 'data/tokenizer-sample/*.jsonl.gz' \
      --out data/tokenized-sample --shard-tokens 20000000 --val-every 200
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
from serblo.tokenizer import load_tokenizer          # noqa: E402
from serblo.train.pack import pack_docs              # noqa: E402


def docs(files):
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
    ap.add_argument("--in", dest="inp", required=True, help="glob of *.jsonl.gz corpus shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=str(ROOT / "data/tokenizer/tokenizer.json"))
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--val-every", type=int, default=200)
    a = ap.parse_args()

    tok = load_tokenizer(a.tokenizer)
    eos = tok.token_to_id("<|endoftext|>")
    sha = hashlib.sha256(pathlib.Path(a.tokenizer).read_bytes()).hexdigest()
    files = glob.glob(a.inp)
    print(f"packing {len(files)} input shards -> {a.out}  (eos_id={eos}, val 1/{a.val_every}) ...", flush=True)

    meta = pack_docs(docs(files), lambda d: tok.encode(d).ids, a.out, eos_id=eos,
                     shard_tokens=a.shard_tokens, val_every=a.val_every,
                     tokenizer_sha256=sha, vocab_size=tok.get_vocab_size())
    print(f"DONE: {meta['train_tokens']:,} train tok + {meta['val_tokens']:,} val tok  "
          f"({len(meta['train_shards'])} train shards, {len(meta['val_shards'])} val shards)")
    print(f"  -> {a.out}/meta.json")


if __name__ == "__main__":
    main()
