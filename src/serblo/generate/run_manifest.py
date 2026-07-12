"""Emit the M3 generation manifest to disk: resumable, incremental, seeded (PRD §7.4).

Writes ``data/manifests/{register}.jsonl`` (one neutral prompt-spec per line) plus a
``{register}.meta.json`` sidecar recording (seed, n, config_hash). The submit steps
(Vertex batch / MiMo / NIM) read these and reshape per backend.

Resumable & incremental: records stream to disk line-by-line; a
re-run with the SAME (seed, n, config) appends only the missing tail. A mismatched
sidecar aborts rather than silently interleaving two different draws.

Usage:
  PYTHONPATH=src python -m serblo.generate.run_manifest --n-conversational 5000 --n-essay 2000
  PYTHONPATH=src python -m serblo.generate.run_manifest --preview 2     # eyeball prompts, no write
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from serblo.generate.manifest import (  # noqa: E402
    axes_for, grounding_rate, iter_records, load_config, space_size,
)

FLUSH_EVERY = 2000


def _config_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_refpool(cfg, register, out_dir):
    """The grounding anchor pool for a register, or None when grounding is off.

    Hard-fails if the config demands grounding but the pool file is missing; silently
    emitting ungrounded prompts when grounding was configured would be a provenance lie."""
    if grounding_rate(cfg, register) <= 0:
        return None
    path = out_dir / f"refpool-{register}.jsonl"
    if not path.exists():
        raise SystemExit(
            f"{register}: grounding.rate > 0 but no anchor pool at {path}. "
            f"Build it first: PYTHONPATH=src python -m serblo.generate.run_refpool")
    pool = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            pool.append(json.loads(line))
    if not pool:
        raise SystemExit(f"{register}: anchor pool {path} is empty")
    return pool


def _count_lines(p):
    if not p.exists():
        return 0
    with open(p, encoding="utf-8") as f:
        return sum(1 for _ in f)


def emit(cfg, register, n, seed, out_dir, config_path, refpool=None):
    if n <= 0:
        return 0
    out_path = out_dir / f"{register}.jsonl"
    meta_path = out_dir / f"{register}.meta.json"
    total = space_size(axes_for(cfg, register))
    cfg_hash = _config_hash(config_path)
    pool_path = out_dir / f"refpool-{register}.jsonl"
    pool_hash = _config_hash(pool_path) if refpool else "none"
    intent = (seed, n, cfg_hash, pool_hash)

    already = _count_lines(out_path)
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
        have = (m.get("seed"), m.get("n"), m.get("config_hash"), m.get("refpool_hash", "none"))
        if have != intent:
            raise SystemExit(
                f"{register}: existing manifest built with different (seed,n,config,refpool) "
                f"{have} != {intent}. Delete {out_path} + {meta_path} or change --out.")
    elif already:
        raise SystemExit(
            f"{register}: {out_path} has {already} lines but no sidecar to verify the "
            f"draw. Delete it or change --out.")

    if already >= n:
        print(f"{register}: already complete ({already:,}/{n:,})")
        return already

    meta_path.write_text(json.dumps(
        {"register": register, "n": n, "seed": seed, "config_hash": cfg_hash,
         "refpool_hash": pool_hash, "grounding_rate": grounding_rate(cfg, register),
         "space_size": total, "coverage": round(n / total, 4)}, indent=2))

    written = already
    mode = "a" if already else "w"
    with open(out_path, mode, encoding="utf-8") as w:
        for rec in iter_records(cfg, register, n, seed, refpool):
            if int(rec["id"].rsplit("-", 1)[1]) < already:   # skip already-written prefix
                continue
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if written % FLUSH_EVERY == 0:
                w.flush()
                print(f"  {register}: {written:,}/{n:,}", flush=True)
    cov = n / total
    tag = "" if n <= total else f"  (space {total:,} exhausted; {n - total:,} repeats)"
    print(f"{register}: wrote {written:,}  [space {total:,}, coverage {cov:.1%}]{tag}")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "generation-config.yaml"))
    ap.add_argument("--out", default=str(ROOT / "data" / "manifests"))
    ap.add_argument("--n-conversational", type=int, default=1000)
    ap.add_argument("--n-essay", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", type=int, default=0,
                    help="print N sample records per register and exit (no write)")
    a = ap.parse_args()
    cfg = load_config(a.config)

    out_dir = Path(a.out)
    if a.preview:
        for reg, n in (("conversational", a.n_conversational), ("essay", a.n_essay)):
            if n <= 0:
                continue
            pool = load_refpool(cfg, reg, out_dir)
            print(f"\n===== {reg}: {a.preview} of {n:,} (space {space_size(axes_for(cfg, reg)):,}) =====")
            shown = 0
            for rec in iter_records(cfg, reg, n, a.seed, pool):
                print(json.dumps(rec, ensure_ascii=False, indent=2))
                shown += 1
                if shown >= a.preview:
                    break
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for reg, n in (("conversational", a.n_conversational), ("essay", a.n_essay)):
        pool = load_refpool(cfg, reg, out_dir) if n > 0 else None
        total += emit(cfg, reg, n, a.seed, out_dir, a.config, pool)
    (out_dir / "index.json").write_text(json.dumps(
        {"conversational": a.n_conversational, "essay": a.n_essay,
         "seed": a.seed, "total_prompts": a.n_conversational + a.n_essay}, indent=2))
    print(f"\nmanifest ready: {total:,} prompt-specs in {out_dir}")


if __name__ == "__main__":
    main()
