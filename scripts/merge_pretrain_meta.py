#!/usr/bin/env python3
"""Merge conv + subtitles + srWaC shard metadata into the backbone meta.json (Serblo pretrain).

Upsampling is metadata-level: ShardLoader weights each shard listing by byte size, so listing a
shard name N times gives it N x weight with zero disk duplication.
  - conv (synthetic conversational) and subtitles (real conversational) -> the priority register, xN
  - srWaC (real web) -> volume, x1
The val split is left exactly as the backbone produced it (backbone-only) so the plateau signal
stays clean. Local shards are train_NNNNN.bin; in GCS they carry distinct prefixes (conv_/subs_/
srwac_) so they never collide with the backbone's train_NNNNN.bin.

  python3 scripts/merge_pretrain_meta.py --backbone-meta bb.json --conv-meta tok_conv/meta.json \
      --subs-meta tok_subs/meta.json --srwac-meta tok_srwac/meta.json --upsample 3 --subs-upsample 3 --out out.json
"""
import argparse
import json
import sys


def load(p):
    with open(p) as f:
        return json.load(f)


def renamed(meta, prefix):
    """train_00000.bin -> <prefix>00000.bin, preserving the numeric index."""
    return [f"{prefix}{n.replace('train_', '').replace('.bin', '')}.bin" for n in meta["train_shards"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone-meta", required=True)
    ap.add_argument("--conv-meta", required=True)
    ap.add_argument("--subs-meta", default=None)
    ap.add_argument("--srwac-meta", default=None)
    ap.add_argument("--conv-prefix", default="conv_")
    ap.add_argument("--subs-prefix", default="subs_")
    ap.add_argument("--srwac-prefix", default="srwac_")
    ap.add_argument("--upsample", type=int, default=3)
    ap.add_argument("--subs-upsample", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    bb, cv = load(a.backbone_meta), load(a.conv_meta)
    su = load(a.subs_meta) if a.subs_meta else None
    sw = load(a.srwac_meta) if a.srwac_meta else None

    shas = {bb.get("tokenizer_sha256"), cv.get("tokenizer_sha256")}
    for extra in (su, sw):
        if extra:
            shas.add(extra.get("tokenizer_sha256"))
    if len(shas) != 1 or None in shas:
        sys.exit(f"ABORT: tokenizer sha256 mismatch across metas: {shas}")

    conv_names = renamed(cv, a.conv_prefix)
    subs_names = renamed(su, a.subs_prefix) if su else []
    srwac_names = renamed(sw, a.srwac_prefix) if sw else []
    subs_tok = su["train_tokens"] if su else 0
    srwac_tok = sw["train_tokens"] if sw else 0

    merged = dict(bb)
    merged["train_shards"] = (list(bb["train_shards"]) + conv_names * a.upsample
                              + subs_names * a.subs_upsample + srwac_names)
    merged["train_tokens"] = (bb["train_tokens"] + cv["train_tokens"] * a.upsample
                              + subs_tok * a.subs_upsample + srwac_tok)
    tt = merged["train_tokens"]
    merged["pretrain_mix"] = {
        "backbone_tokens": bb["train_tokens"],
        "conv_tokens_unique": cv["train_tokens"], "conv_upsample": a.upsample,
        "subs_tokens_unique": subs_tok, "subs_upsample": a.subs_upsample,
        "srwac_tokens": srwac_tok,
        "conv_pct_effective": round(cv["train_tokens"] * a.upsample / tt * 100, 2),
        "subs_pct_effective": round(subs_tok * a.subs_upsample / tt * 100, 2) if su else 0.0,
        "srwac_pct_effective": round(srwac_tok / tt * 100, 2) if sw else 0.0,
    }

    with open(a.out, "w") as f:
        json.dump(merged, f, indent=2)

    m = merged["pretrain_mix"]
    print(f"merged meta -> {a.out}")
    print(f"  tokenizer sha256 : {sorted(shas)[0][:16]} (all match)")
    print(f"  backbone {m['backbone_tokens']:,} | conv {m['conv_tokens_unique']:,}x{a.upsample} "
          f"| subs {m['subs_tokens_unique']:,}x{a.subs_upsample} | srwac {m['srwac_tokens']:,}")
    print(f"  effective train_tokens : {tt:,}")
    print(f"  conv {m['conv_pct_effective']}% | subs {m['subs_pct_effective']}% | srwac {m['srwac_pct_effective']}%")
    print(f"  train_shards {len(merged['train_shards'])} (bb {len(bb['train_shards'])} + conv "
          f"{len(conv_names)*a.upsample} + subs {len(subs_names)*a.subs_upsample} + srwac {len(srwac_names)}) "
          f"| val {len(merged['val_shards'])}")


if __name__ == "__main__":
    main()
