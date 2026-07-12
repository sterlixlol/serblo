"""FineWeb-2 BCS acquisition. PRD M1 stage 0 (backbone source).

FineWeb-2 is already GlotLID language/script-filtered and PII-scrubbed upstream, so no
LID is needed here: M1 for this source is just stream -> normalize -> shard.

Streaming strategy (learned the hard way on the 1GB VM):
- `datasets` streaming buffers whole parquet files (~900MB) -> OOM/swap. Rejected.
- HfFileSystem + pyarrow range-reads = thousands of tiny HTTP requests -> anonymous HF
  rate-limits it to a crawl (0% CPU, stalled). Rejected.
- THIS: download each parquet ONCE (hf_hub_download, resumable, one request/file), read
  its row-groups locally with pyarrow (~2MB each), delete. Few requests (HF-friendly) +
  low memory + bounded disk (one ~1GB file at a time). Deterministic file+row order keeps
  skip-based resume correct.

DISABLE XET (load-bearing on a 1GB box): FineWeb-2 is a Xet-backed repo, and an
AUTHENTICATED hf_hub_download otherwise routes through the hf-xet backend, which parallel-
chunk-buffers a ~1GB parquet and spikes to ~916MB RSS -> OOM/swap-thrash on the e2-micro
(the anonymous path never used Xet, which is why it was memory-safe but rate-throttled).
Forcing the classic streaming HTTP path keeps the token's higher rate limits WITHOUT Xet's
memory profile. Set before importing huggingface_hub so the backend choice sees it.
"""
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")           # see module docstring: OOMs the 1GB VM otherwise

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

DATASET = "HuggingFaceFW/fineweb-2"

# The four BCS configs that exist in FineWeb-2 (no cnr/Montenegrin, no bos_Cyrl).
KEEP_META = ("id", "minhash_cluster_size", "language_score", "language_script")

BCS_CONFIGS = ["bos_Latn", "hrv_Latn", "srp_Cyrl", "srp_Latn"]

# Only the columns we use (raw FineWeb-2 parquet has no minhash_cluster_size column).
COLUMNS = ["id", "text", "language_score"]


def config_files(config):
    """Repo-relative parquet paths for a config's train split, in deterministic order."""
    fs = HfFileSystem()
    full = sorted(fs.glob(f"datasets/{DATASET}/data/{config}/train/*.parquet"))
    return [p.split(f"{DATASET}/", 1)[1] for p in full]   # -> "data/bos_Latn/train/000_00000.parquet"


def stream_config(config, skip=0, tmp_dir=None):
    """Yield {id, text, language_score} for `config`, skipping the first `skip`.

    Downloads one parquet at a time to `tmp_dir`, reads row-groups locally (memory-light),
    then deletes it. `skip` fast-forwards whole row-groups (via metadata, no read) before
    the resume point.
    """
    tmp = tmp_dir or "/tmp/serblo-fw2"
    os.makedirs(tmp, exist_ok=True)
    seen = 0
    for fname in config_files(config):
        local = hf_hub_download(DATASET, fname, repo_type="dataset", local_dir=tmp)
        try:
            pf = pq.ParquetFile(local)
            for i in range(pf.num_row_groups):
                rg_rows = pf.metadata.row_group(i).num_rows
                if seen + rg_rows <= skip:                # whole row-group precedes resume point
                    seen += rg_rows
                    continue
                for row in pf.read_row_group(i, columns=COLUMNS).to_pylist():
                    if seen >= skip:
                        yield row
                    seen += 1
        finally:
            if os.path.exists(local):                     # reclaim disk before the next file
                os.remove(local)
