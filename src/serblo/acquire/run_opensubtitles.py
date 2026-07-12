"""Resumable acquire for OpenSubtitles BCS (PRD M1, conversational backbone).

Per language: stream mono lines -> mojibake repair -> normalize (NFC + Cyrillic->
Latin) -> group GROUP_LINES consecutive lines into a dialogue "document" -> shard to
data/interim/opensubtitles/<lang>/ as gzipped JSONL. Resumable via a manifest that
records lines_read at each *completed doc's* boundary, so a crash re-streams only the
tail; the source is durable, so nothing is lost.

Usage:
  PYTHONPATH=src python -m serblo.acquire.run_opensubtitles --cap 20000
  PYTHONPATH=src python -m serblo.acquire.run_opensubtitles --langs sr --cap 500 --group 10
"""
import argparse
import gzip
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from serblo.acquire.opensubtitles import BCS_LANGS, stream_lines
from serblo.pipeline.encoding import fix_mojibake
from serblo.pipeline.normalize import normalize_text, has_cyrillic, diacritic_ratio

MIN_FREE_GB = 5.0


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def _load_manifest(d):
    mf = d / "manifest.json"
    if mf.exists():
        return json.loads(mf.read_text())
    return {"lang": d.name, "docs_written": 0, "lines_read": 0, "shards": [], "shard_idx": 0}


def _save_manifest(d, man):
    mf = d / "manifest.json"
    tmp = mf.with_name("manifest.json.tmp")
    tmp.write_text(json.dumps(man, ensure_ascii=False, indent=0))
    os.replace(tmp, mf)


def acquire_lang(lang, out_root, cap_docs, group_lines, shard_size):
    d = out_root / lang
    d.mkdir(parents=True, exist_ok=True)
    man = _load_manifest(d)
    if cap_docs and man["docs_written"] >= cap_docs:
        print(f"[{lang}] already {man['docs_written']} >= cap {cap_docs}, skip", flush=True)
        return man
    print(f"[{lang}] resuming: {man['docs_written']} docs, {man['lines_read']} lines read", flush=True)

    base = man["lines_read"]
    shard = []              # list of (record, boundary_line) awaiting flush
    group = []              # lines accumulating into the current doc
    seen = 0                # lines consumed from the stream this run
    stats = {"diac": 0.0, "cyr": 0, "fixed": 0}
    t0 = time.time()

    def flush():
        if not shard:
            return
        if free_gb(out_root) < MIN_FREE_GB:
            raise RuntimeError(f"free disk < {MIN_FREE_GB}GB — stopping before overflow")
        idx = man["shard_idx"]
        name = f"shard-{idx:05d}.jsonl.gz"
        tmp = d / (name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as w:
            for rec, _ in shard:
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, d / name)
        man["docs_written"] += len(shard)
        man["lines_read"] = shard[-1][1]        # boundary of last complete doc
        man["shards"].append(name)
        man["shard_idx"] = idx + 1
        _save_manifest(d, man)
        print(f"[{lang}] shard {idx} (+{len(shard)} -> {man['docs_written']} docs) "
              f"diac~{stats['diac']/len(shard):.3f} cyr_resid={stats['cyr']} "
              f"mojibake_fixed={stats['fixed']} lines={man['lines_read']} "
              f"{time.time()-t0:.0f}s", flush=True)
        shard.clear()
        stats.update(diac=0.0, cyr=0, fixed=0)

    try:
        for line in stream_lines(lang, skip=base):
            seen += 1
            fixed = fix_mojibake(line)
            if fixed != line:
                stats["fixed"] += 1
            text = normalize_text(fixed)
            if text.strip():
                group.append(text)
            if len(group) >= group_lines:
                doc = "\n".join(group)
                group = []
                rec = {"text": doc, "source": "opensubtitles", "lang": lang}
                shard.append((rec, base + seen))
                stats["diac"] += diacritic_ratio(doc)
                if has_cyrillic(doc):
                    stats["cyr"] += 1
                if len(shard) >= shard_size:
                    flush()
                if cap_docs and man["docs_written"] + len(shard) >= cap_docs:
                    break
        flush()
    except BaseException as e:
        flush()
        print(f"[{lang}] STOPPED after +{seen} lines: {type(e).__name__}: {e}", flush=True)
        raise
    print(f"[{lang}] done: {man['docs_written']} docs", flush=True)
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=BCS_LANGS)
    ap.add_argument("--cap", type=int, default=None, help="max docs per language")
    ap.add_argument("--group", type=int, default=50, help="subtitle lines per document")
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "data" / "interim" / "opensubtitles"))
    a = ap.parse_args()

    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"acquire OpenSubtitles -> {out_root} | cap={a.cap} group={a.group} "
          f"shard={a.shard_size} free={free_gb(out_root):.0f}GB", flush=True)

    grand = 0
    for lang in a.langs:
        grand += acquire_lang(lang, out_root, a.cap, a.group, a.shard_size)["docs_written"]
    print(f"\nTOTAL staged: {grand} docs", flush=True)
    os._exit(0)     # avoid any lingering urllib/gzip thread hanging interpreter exit


if __name__ == "__main__":
    main()
