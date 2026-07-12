#!/usr/bin/env python3
"""Train the final 32k byte-level BPE on the GCS backbone sample, save data/tokenizer/tokenizer.json,
and verify vocab size + reserved special tokens + token/word ratio (target ~1.5-1.7; de-risk 1.577).

  PYTHONPATH=src python3 scripts/train_tokenizer.py
"""
import glob
import gzip
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from serblo.tokenizer import load_tokenizer                       # noqa: E402
from serblo.tokenizer.train_bpe import SPECIAL_TOKENS, train_bpe  # noqa: E402


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
    files = glob.glob(str(ROOT / "data/tokenizer-sample/*.jsonl.gz"))
    out = str(ROOT / "data/tokenizer/tokenizer.json")
    n_docs = sum(1 for _ in docs(files))
    print(f"training 32k BPE on {len(files)} shards / {n_docs:,} docs ...", flush=True)
    train_bpe(docs(files), vocab_size=32000, out_path=out)

    tok = load_tokenizer(out)
    print(f"\nvocab size: {tok.get_vocab_size()}  (expect 32000)")
    for st in SPECIAL_TOKENS:
        print(f"  special {st!r}: id={tok.token_to_id(st)}")

    sample, words, ntoks = [], 0, 0
    for i, d in enumerate(docs(files)):
        if i >= 3000:
            break
        sample.append(d)
    for d in sample:
        words += len(d.split())
        ntoks += len(tok.encode(d).ids)
    print(f"\ntoken/word ratio on {len(sample)} docs: {ntoks / max(1, words):.3f}  (target ~1.5-1.7)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
