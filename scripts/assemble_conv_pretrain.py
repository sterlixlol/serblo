#!/usr/bin/env python3
"""Assemble the conversational-register bucket that gets mixed into the pretrain corpus (Serblo).

The FineWeb-2 backbone is overwhelmingly formal, Cyrillic-origin web text; the model's stated
priority register is casual conversational BCS, which the web lacks. This bucket supplies it.
It is small (~146M tokens) and gets UPSAMPLED at load time (by listing its shard names N times in
meta.json), not duplicated on disk.

Sources IN:
  - conv pillar  (Gemini 500k, all conversational)  -> the bulk
  - multi-lineage synthetic, conv-* ids only         -> non-Gemini diversity (thin but real)

Sources OUT (deliberate):
  - subtitles : mojibake (c-caron rendered as e-grave) + film-credits cruft -> teaches broken text
  - essay-*   : formal register, redundant with the backbone

Output: one gzipped jsonl of {"text": ...}, consumable directly by pack_corpus_fast.py.
Incremental: writes each doc to the gzip stream as it goes; a crash truncates cleanly and the
whole thing re-derives in seconds from local files.

  python3 scripts/assemble_conv_pretrain.py <pillar_docs.jsonl> <out.jsonl.gz>
"""
import glob
import gzip
import json
import sys

MULTI_GLOBS = [
    "/home/sterlix/Desktop/serblo/data/generated/*/docs.jsonl",
    "/home/sterlix/Desktop/serblo/data/generated/tranche*/*/docs.jsonl",
]


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: assemble_conv_pretrain.py <pillar_docs.jsonl> <out.jsonl.gz>")
    pillar, out_path = sys.argv[1], sys.argv[2]

    n_pillar = n_multi = n_skip_essay = n_empty = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as out:
        # pillar: every doc is conversational (manifest-pillar seed)
        with open(pillar, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    t = json.loads(line).get("text", "").strip()
                except json.JSONDecodeError:
                    continue
                if not t:
                    n_empty += 1
                    continue
                out.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
                n_pillar += 1
        # multi-lineage: conversational ids only
        seen = set()
        for pat in MULTI_GLOBS:
            for f in glob.glob(pat):
                if f in seen:
                    continue
                seen.add(f)
                with open(f, encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        try:
                            d = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not str(d.get("id", "")).startswith("conv"):
                            n_skip_essay += 1
                            continue
                        t = d.get("text", "").strip()
                        if not t:
                            n_empty += 1
                            continue
                        out.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
                        n_multi += 1

    print(f"conv bucket -> {out_path}")
    print(f"  pillar docs        : {n_pillar:,}")
    print(f"  multi-lineage conv : {n_multi:,}")
    print(f"  skipped non-conv   : {n_skip_essay:,}")
    print(f"  skipped empty      : {n_empty:,}")
    print(f"  TOTAL docs         : {n_pillar + n_multi:,}")


if __name__ == "__main__":
    main()
