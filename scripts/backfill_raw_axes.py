#!/usr/bin/env python3
"""One-off: embed register+axes into existing raw.jsonl records (generation-time provenance).

Why: raw records originally carried no axes; rebuilds looked them up in the CURRENT
manifest. When the conv manifest was regenerated with rotated particles,
a later rebuild stamped the new axes (incl. `particles`) onto docs generated from the
OLD primed manifest, a provenance lie. Drivers now embed axes at generation time; this
script repairs history:

  conv ids on raw lines < --pre  ->  archived v0 manifest (the axes that really primed them)
  conv ids on lines >= --pre     ->  current conv manifest (rotated particles, true)
  essay ids (all)                ->  current essay manifest (byte-identical across regens)

Lines that already carry axes are left untouched. Atomic replace. Run on the ledger
OWNER (the VM) while no generation is running.

Usage:
  python3 scripts/backfill_raw_axes.py --root ~/serblo --pre 630
"""
import argparse
import gzip
import json
from pathlib import Path


def load_manifest(path):
    meta = {}
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            meta[r["id"]] = (r["register"], r["axes"])
    return meta


def backfill(raw_path, resolve, label):
    raw_path = Path(raw_path)
    if not raw_path.exists():
        print(f"{label}: {raw_path} missing — skipped")
        return
    tmp = raw_path.with_suffix(".jsonl.tmp")
    n = filled = had = miss = 0
    with open(raw_path, encoding="utf-8") as f, open(tmp, "w", encoding="utf-8") as w:
        for i, line in enumerate(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                w.write(line)                            # preserve any corrupt tail as-is
                continue
            n += 1
            if "axes" in r and "register" in r:
                had += 1
            else:
                hit = resolve(i, r["id"])
                if hit:
                    r["register"], r["axes"] = hit
                    filled += 1
                else:
                    miss += 1
                    print(f"  warn: {r['id']} (line {i}) unresolvable — left bare")
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(raw_path)
    print(f"{label}: {n} records — {filled} backfilled, {had} already had axes, {miss} unresolved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--pre", type=int, default=630,
                    help="raw lines before this index predate the conv-manifest regen")
    a = ap.parse_args()
    root = Path(a.root).expanduser()
    m = root / "data" / "manifests"

    archive = load_manifest(m / "conversational-v0-primed.jsonl.gz")
    conv = load_manifest(m / "conversational.jsonl")
    essay = load_manifest(m / "essay.jsonl")

    def resolve_mimo(line_i, rid):
        if rid.startswith("essay"):
            return essay.get(rid)
        return (archive if line_i < a.pre else conv).get(rid)

    def resolve_vertex(_line_i, rid):
        return essay.get(rid) or conv.get(rid)

    backfill(root / "data" / "generated" / "mimo" / "raw.jsonl", resolve_mimo, "mimo")
    backfill(root / "data" / "generated" / "vertex" / "raw.jsonl", resolve_vertex, "vertex")


if __name__ == "__main__":
    main()
