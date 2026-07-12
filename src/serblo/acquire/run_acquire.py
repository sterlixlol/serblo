"""Resumable acquire + normalize for FineWeb-2 BCS (PRD M1).

Streams each BCS config, normalizes (NFC + Cyrillic->Latin, diacritics kept), and
writes gzipped JSONL shards to data/interim/fineweb2/<config>/. Resumable via a
per-config manifest that records only *completed* shards, so a crash re-streams at
most the last unfinished shard, and the source (FineWeb-2) is durable, so nothing
is ever lost, only recomputed.

Usage:
  PYTHONPATH=src python -m serblo.acquire.run_acquire --cap 20000
  PYTHONPATH=src python -m serblo.acquire.run_acquire            # full run (needs a disk plan)
"""
import argparse
import gzip
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # .../serblo
sys.path.insert(0, str(ROOT / "src"))

from serblo.acquire.fineweb2 import BCS_CONFIGS, KEEP_META, stream_config
from serblo.pipeline.normalize import normalize_text, has_cyrillic, diacritic_ratio

MIN_FREE_GB = 5.0                                    # stop before we corrupt the disk


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def _load_manifest(cfg_dir):
    mf = cfg_dir / "manifest.json"
    if mf.exists():
        return json.loads(mf.read_text())
    return {"config": cfg_dir.name, "docs_written": 0, "shards": [], "shard_idx": 0}


def _save_manifest(cfg_dir, man):
    mf = cfg_dir / "manifest.json"
    tmp = mf.with_name("manifest.json.tmp")
    tmp.write_text(json.dumps(man, ensure_ascii=False, indent=0))
    os.replace(tmp, mf)                              # atomic


def acquire_config(config, out_root, cap, shard_size):
    cfg_dir = out_root / config
    cfg_dir.mkdir(parents=True, exist_ok=True)
    man = _load_manifest(cfg_dir)
    done = man["docs_written"]
    if cap and done >= cap:
        print(f"[{config}] already at {done} >= cap {cap}, skip", flush=True)
        return man
    print(f"[{config}] resuming from {done} docs (shard {man['shard_idx']})", flush=True)

    buf = []                                         # docs for the current shard
    stats = {"diac": 0.0, "cyr": 0}                  # health signals for this shard
    t0 = time.time()
    n_new = 0

    def flush():
        if not buf:
            return
        if free_gb(out_root) < MIN_FREE_GB:
            raise RuntimeError(f"free disk < {MIN_FREE_GB}GB — stopping before overflow")
        idx = man["shard_idx"]
        name = f"shard-{idx:05d}.jsonl.gz"
        tmp = cfg_dir / (name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as w:
            for rec in buf:
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, cfg_dir / name)              # atomic: shard is all-or-nothing
        man["docs_written"] += len(buf)
        man["shards"].append(name)
        man["shard_idx"] = idx + 1
        _save_manifest(cfg_dir, man)                 # manifest reflects only complete shards
        print(f"[{config}] shard {idx} (+{len(buf)} -> {man['docs_written']}) "
              f"diac~{stats['diac']/len(buf):.3f} cyr_resid={stats['cyr']} "
              f"{time.time()-t0:.0f}s free={free_gb(out_root):.0f}GB", flush=True)
        buf.clear()
        stats["diac"] = 0.0
        stats["cyr"] = 0

    try:
        for ex in stream_config(config, skip=done):
            text = normalize_text(ex.get("text"))
            if not text.strip():
                continue
            rec = {"text": text, "source": "fineweb2", "config": config}
            for k in KEEP_META:
                if k in ex:
                    rec[k] = ex[k]
            buf.append(rec)
            stats["diac"] += diacritic_ratio(text)
            if has_cyrillic(text):
                stats["cyr"] += 1
            n_new += 1
            if len(buf) >= shard_size:
                flush()
            if cap and done + n_new >= cap:
                break
        flush()                                      # final partial shard
    except BaseException as e:                       # incl. KeyboardInterrupt: save first
        flush()
        print(f"[{config}] STOPPED after +{n_new}: {type(e).__name__}: {e}", flush=True)
        raise
    print(f"[{config}] done: {man['docs_written']} docs", flush=True)
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=BCS_CONFIGS)
    ap.add_argument("--cap", type=int, default=None, help="max docs per config (bounded slice)")
    ap.add_argument("--shard-size", type=int, default=10000)
    ap.add_argument("--out", default=str(ROOT / "data" / "interim" / "fineweb2"))
    a = ap.parse_args()

    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"acquire BCS -> {out_root} | cap={a.cap} shard={a.shard_size} "
          f"free={free_gb(out_root):.0f}GB", flush=True)

    grand = 0
    for cfg in a.configs:
        grand += acquire_config(cfg, out_root, a.cap, a.shard_size)["docs_written"]
    print(f"\nTOTAL staged: {grand} docs", flush=True)
    # `datasets` streaming leaves a prefetch thread that blocks interpreter exit
    # after we break out of the stream. Every shard + manifest is already flushed
    # to disk, so force an immediate clean exit instead of hanging on the join.
    os._exit(0)


if __name__ == "__main__":
    main()
