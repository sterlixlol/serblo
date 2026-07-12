"""Full-scale FineWeb-2 BCS acquisition: stream → normalize → declutter → shard (PRD M1+M2).

The backbone. FineWeb-2 BCS is ~18.85B words across four configs; this streams them,
transliterates Cyrillic→Latin (so srp_Cyrl collapses onto srp_Latn before dedup; order
matters), strips chrome (declutter), quality-gates, and writes gzip JSONL shards. One
streaming pass does M1 (acquire+normalize) AND the M2 declutter, so we never materialize
the raw 200GB; only the cleaned output lands on disk.

Memory-light: one doc at a time, buffer only the current shard. Resumable: per-config
`skip` + a progress file flushed after every shard, so a crash loses at
most the in-progress shard's docs, which re-stream on resume. Bounded by --target-docs
for a first validation pull; drop the bound for the full backbone.

Storage: local `data/processed/fineweb2-clean/` by default. --gcs-prefix additionally
uploads each finished shard to GCS and deletes the local copy (for the full run, which
won't fit the VM disk). Ends with os._exit(0), a harmless hard exit (the pyarrow reader
has no lingering threads; the earlier datasets-streaming prefetch-thread hang is gone).

Usage:
  PYTHONPATH=src python -m serblo.acquire.run_fineweb_scale --target-docs 2000000
  PYTHONPATH=src python -m serblo.acquire.run_fineweb_scale --gcs-prefix gs://bkt/serblo/corpus/fw2
"""
import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from serblo.acquire.fineweb2 import BCS_CONFIGS, stream_config  # noqa: E402
from serblo.pipeline.declutter import declutter, survival        # noqa: E402
from serblo.pipeline.normalize import normalize_text             # noqa: E402

MIN_WORDS = 30
MIN_LANG_SCORE = 0.60          # FineWeb-2's GlotLID confidence; it pre-filters, this trims the tail
MIN_SURVIVAL = 0.5             # declutter kept < half the words -> was mostly chrome/nav
SHARD_DOCS = 25000
MIN_FREE_GB = 3.0


def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def process(doc):
    """Raw FineWeb-2 doc -> cleaned corpus record, or (None, reason)."""
    if (doc.get("language_score") or 1.0) < MIN_LANG_SCORE:
        return None, "low_lang_score"
    text = normalize_text(doc.get("text") or "")     # NFC + Cyrillic->Latin (gajica)
    clean = declutter(text)
    if len(clean.split()) < MIN_WORDS:
        return None, "too_short"
    if survival(text, clean) < MIN_SURVIVAL:
        return None, "mostly_chrome"
    return {"id": doc.get("id"), "text": clean, "source": "fineweb2",
            "minhash_cluster_size": doc.get("minhash_cluster_size"),
            "language_score": doc.get("language_score")}, "kept"


def load_progress(prog_path):
    if prog_path.exists():
        return json.loads(prog_path.read_text())
    return {"configs": {}, "total_kept": 0}


def save_progress(prog_path, prog):
    tmp = prog_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(prog, indent=1))
    tmp.replace(prog_path)


def flush_shard(buf, out_dir, config, seq, gcs_prefix):
    name = f"shard-{config}-{seq:05d}.jsonl.gz"
    path = out_dir / name
    with gzip.open(path, "wt", encoding="utf-8") as w:
        for rec in buf:
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if gcs_prefix:                                    # offload + reclaim disk (full-run mode)
        up = subprocess.run(["gcloud", "storage", "cp", str(path), f"{gcs_prefix.rstrip('/')}/{name}"],
                            capture_output=True, text=True)
        if up.returncode != 0:
            print(f"  GCS upload FAILED for {name} (kept local):\n{up.stderr[:200]}", flush=True)
        else:
            path.unlink()
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "fineweb2-clean"))
    ap.add_argument("--configs", nargs="+", default=BCS_CONFIGS)
    ap.add_argument("--target-docs", type=int, default=2_000_000,
                    help="stop after this many KEPT docs total (0 = full backbone)")
    ap.add_argument("--shard-docs", type=int, default=SHARD_DOCS)
    ap.add_argument("--gcs-prefix", default=None, help="gs://… : upload+delete each shard (full run)")
    ap.add_argument("--tmp-dir", default=None, help="parquet download dir (default /tmp; use a REAL-disk "
                    "path if /tmp is tmpfs — e.g. the CachyOS desktop RAM-backs /tmp, so ~1GB parquets "
                    "would land in RAM)")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prog_path = out_dir / "progress.json"
    prog = load_progress(prog_path)
    target = a.target_docs or float("inf")
    t0 = time.time()
    print(f"resume: {prog['total_kept']:,} kept so far | target {a.target_docs or 'FULL'} | "
          f"configs {a.configs}", flush=True)

    for config in a.configs:
        cst = prog["configs"].setdefault(config, {"seen": 0, "kept": 0, "shards": 0, "done": False})
        if cst["done"]:
            print(f"{config}: already complete ({cst['kept']:,} kept)", flush=True)
            continue
        if prog["total_kept"] >= target:
            break
        print(f"{config}: streaming from doc {cst['seen']:,} …", flush=True)
        buf, reasons = [], {"kept": 0, "low_lang_score": 0, "too_short": 0, "mostly_chrome": 0}
        try:
            for doc in stream_config(config, skip=cst["seen"], tmp_dir=a.tmp_dir):
                cst["seen"] += 1
                rec, reason = process(doc)
                reasons[reason] = reasons.get(reason, 0) + 1
                if rec is not None:
                    buf.append(rec)
                if len(buf) >= a.shard_docs:
                    if free_gb(out_dir) < MIN_FREE_GB:     # unconditional: in --gcs-prefix mode a
                        print(f"  DISK LOW (<{MIN_FREE_GB}GB free) — stopping cleanly at shard "     # failed
                              f"boundary. In --gcs-prefix mode this means uploads are FAILING and "  # upload
                              f"shards are piling up locally (flush_shard keeps them).", flush=True) # keeps local
                        raise SystemExit(0)
                    name = flush_shard(buf, out_dir, config, cst["shards"], a.gcs_prefix)
                    cst["kept"] += len(buf); prog["total_kept"] += len(buf); cst["shards"] += 1
                    buf = []
                    save_progress(prog_path, prog)
                    print(f"  {config}: {cst['seen']:,} seen, {cst['kept']:,} kept "
                          f"({name}) | total {prog['total_kept']:,} | {(time.time()-t0)/60:.0f}m", flush=True)
                    if prog["total_kept"] >= target:
                        break
            else:
                cst["done"] = True                    # generator exhausted -> config complete
        finally:
            if buf:                                   # flush the partial shard so nothing is lost
                name = flush_shard(buf, out_dir, config, cst["shards"], a.gcs_prefix)
                cst["kept"] += len(buf); prog["total_kept"] += len(buf); cst["shards"] += 1
                save_progress(prog_path, prog)
        save_progress(prog_path, prog)
        print(f"{config}: {cst['kept']:,} kept / {cst['seen']:,} seen | gates {reasons}", flush=True)

    print(f"\n=== acquire summary ===\ntotal kept {prog['total_kept']:,} | "
          f"{(time.time()-t0)/60:.0f} min | out {out_dir}", flush=True)
    sys.stdout.flush()
    os._exit(0)                                       # belt-and-suspenders hard exit (the old datasets
    #                                                   prefetch-thread hang is gone; pyarrow now)


if __name__ == "__main__":
    main()
