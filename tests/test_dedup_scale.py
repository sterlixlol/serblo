"""run_dedup_scale must produce the SAME survivors as the in-memory run_dedup (the oracle).

The streaming tool trades RAM for disk passes; correctness is defined as byte-for-byte the same
survivor set as run_dedup.py on any input. We plant every duplicate class (exact, cross-source
exact, near-dup above threshold, near-miss below threshold, empty/no-shingle, unique) into a
temp interim tree, run both tools via their real CLIs, and compare the survivor texts.
"""
import gzip
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from serblo.dedup.run_dedup_scale import stage2_exact, survivors_mask  # noqa: E402

BASE = " ".join(f"rec{i}" for i in range(30))                    # 30-word base doc


def _near(base, changed):
    """Copy of base with `changed` trailing words swapped (fewer swaps -> higher Jaccard)."""
    w = base.split()
    for i in range(changed):
        w[-(i + 1)] = f"zzz{i}"
    return " ".join(w)


def _write(path, docs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as z:
        for d in docs:
            z.write(json.dumps(d, ensure_ascii=False) + "\n")


def _survivor_texts(out_dir):
    texts = []
    for shard in sorted(out_dir.glob("shard-*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as z:
            texts += [json.loads(l)["text"] for l in z if l.strip()]
    return texts


def _run(module, interim, out):
    env = {"PYTHONPATH": "src", "PATH": __import__("os").environ["PATH"]}
    cmd = [sys.executable, "-m", module, "--interim", str(interim), "--out", str(out), "--write"]
    if module.endswith("run_dedup_scale"):                     # isolate scratch per run (never the repo default)
        cmd += ["--work", str(out) + "-work"]
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"{module} failed:\n{r.stdout}\n{r.stderr}"
    return _survivor_texts(out)


@pytest.fixture
def interim(tmp_path):
    d = tmp_path / "interim"
    other = " ".join(f"other{i}" for i in range(30))
    _write(d / "fineweb2" / "aa" / "shard-00000.jsonl.gz", [
        {"text": BASE, "source": "fineweb2"},                    # canonical
        {"text": _near(BASE, 1), "source": "fineweb2"},          # near-dup (1 word) -> merged
        {"text": BASE, "source": "fineweb2"},                    # exact dup -> dropped
        {"text": other, "source": "fineweb2"},                   # unique
    ])
    _write(d / "opensubtitles" / "xx" / "shard-00000.jsonl.gz", [
        {"text": BASE, "source": "opensubtitles"},               # cross-source exact -> dropped
        {"text": _near(BASE, 12), "source": "opensubtitles"},    # near-MISS (12 words) -> kept
        {"text": "", "source": "opensubtitles"},                 # empty -> no shingles -> kept singleton
    ])
    _write(d / "forum" / "reddit" / "shard-00000.jsonl.gz", [
        {"text": " ".join(f"uniqueforum{i}" for i in range(30)), "source": "forum"},
    ])
    _write(d / "literature" / "shard-00000.jsonl.gz", [
        {"text": " ".join(f"book{i}" for i in range(30)), "source": "literature"},
    ])
    return d


def test_parity_with_inmemory_oracle(interim, tmp_path):
    old = _run("serblo.dedup.run_dedup", interim, tmp_path / "old")
    new = _run("serblo.dedup.run_dedup_scale", interim, tmp_path / "new")
    # same survivors, and (since both keep first-in-order and emit in doc_id order) same sequence
    assert sorted(new) == sorted(old), f"survivor sets differ\nold={sorted(old)}\nnew={sorted(new)}"
    assert new == old, "survivor ORDER differs (doc_id stream order not preserved)"


def test_resume_idempotent(interim, tmp_path):
    """Re-running against a completed work dir skips done files and re-emits identical survivors."""
    import os
    env = {"PYTHONPATH": "src", "PATH": os.environ["PATH"]}
    work = tmp_path / "work"

    def run(out):
        r = subprocess.run([sys.executable, "-m", "serblo.dedup.run_dedup_scale",
                            "--interim", str(interim), "--work", str(work),
                            "--out", str(out), "--write"],
                           cwd=str(ROOT), env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return r

    run(tmp_path / "r1")
    r2 = run(tmp_path / "r2")
    assert "sig " not in r2.stdout, "stage1 re-processed files it had already done"   # all skipped
    assert _survivor_texts(tmp_path / "r2") == _survivor_texts(tmp_path / "r1")


def test_truncate_recovers_partial_columns(tmp_path):
    """A crash mid-file leaves extra rows; truncate_to(docs_done) discards them exactly."""
    from serblo.dedup.run_dedup_scale import Columns
    cols = Columns(tmp_path, num_perm=4, bands=2)
    tmp_path.mkdir(exist_ok=True)
    # 3 clean docs + 2 "partial-crash" rows appended past the checkpoint
    fhs = cols.appenders()
    fhs[cols.sigs].write(np.arange(5 * 4, dtype=np.uint32).tobytes())
    fhs[cols.content].write(np.arange(5, dtype=np.uint64).tobytes())
    fhs[cols.bandk].write(np.arange(5 * 2, dtype=np.uint64).tobytes())
    fhs[cols.valid].write(np.ones(5, np.uint8).tobytes())
    for fh in fhs.values():
        fh.close()
    cols.truncate_to(3)
    sigs, content, bandk, valid = cols.memmap(3)
    assert sigs.shape == (3, 4) and content.shape == (3,)
    assert bandk.shape == (3, 2) and valid.shape == (3,)
    assert list(content) == [0, 1, 2]              # rows 3,4 gone, first 3 intact


def test_manifest_mixes_gzip_and_plain(tmp_path):
    """--manifest pools gzip shards (real) AND plain .jsonl (synth channels) and dedups across both."""
    import os
    # a gzip shard and a plain synth docs.jsonl, sharing one exact-dup text across formats
    _write(tmp_path / "shard-00000.jsonl.gz", [
        {"text": BASE, "source": "fineweb2"},
        {"text": " ".join(f"real{i}" for i in range(30)), "source": "fineweb2"},
    ])
    plain = tmp_path / "docs.jsonl"                        # NOT gzipped; the case that crashed
    with open(plain, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"text": BASE, "source": "synth"}) + "\n")          # cross-format exact-dup
        fh.write(json.dumps({"text": " ".join(f"gen{i}" for i in range(30)), "source": "synth"}) + "\n")
    manifest = tmp_path / "assembly-manifest.json"
    manifest.write_text(json.dumps({"total_docs": 4, "files": [
        {"path": os.path.relpath(tmp_path / "shard-00000.jsonl.gz", ROOT)},
        {"path": os.path.relpath(plain, ROOT)},
    ]}))
    env = {"PYTHONPATH": "src", "PATH": os.environ["PATH"]}
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "serblo.dedup.run_dedup_scale",
                        "--manifest", str(manifest), "--work", str(tmp_path / "work"),
                        "--out", str(out), "--write"],
                       cwd=str(ROOT), env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"manifest run failed:\n{r.stdout}\n{r.stderr}"
    texts = _survivor_texts(out)
    assert BASE in texts and texts.count(BASE) == 1        # cross-format exact-dup collapsed to one
    assert len(texts) == 3                                 # BASE + two uniques; the plain BASE dropped


def test_lsh_bucket_chunking(monkeypatch):
    """A bucket bigger than _BUCKET_CHUNK still unions every member to base (flat-memory path)."""
    import serblo.dedup.run_dedup_scale as m
    monkeypatch.setattr(m, "_BUCKET_CHUNK", 1)             # force the multi-chunk path
    cfg = {"num_perm": 4, "bands": 1, "rows": 4, "threshold": 0.7, "k": 5}
    sigs = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [9, 9, 9, 9]], dtype=np.uint32)
    bandk = np.array([[100], [100], [100], [200]], dtype=np.uint64)   # 0,1,2 share a bucket; 3 alone
    valid = np.ones(4, np.uint8)
    exact = np.zeros(4, bool)
    roots = m.stage3_cluster(sigs, bandk, valid, exact, cfg)
    mask = m.survivors_mask(roots, exact)
    assert list(mask) == [True, False, False, True]        # 0,1,2 collapse to 0; 3 separate


def test_stage2_exact_keeps_first():
    content = np.array([10, 20, 10, 30, 20, 10], dtype=np.uint64)
    exact = stage2_exact(content)
    # first occurrence of each hash kept (idx 0,1,3); the rest are exact dups
    assert list(exact) == [False, False, True, False, True, True]


def test_survivors_mask_lowest_id_per_component():
    roots = np.array([0, 0, 0, 3, 3], dtype=np.int64)     # {0,1,2} and {3,4}
    exact = np.array([False, False, False, False, False])
    mask = survivors_mask(roots, exact)
    assert list(mask) == [True, False, False, True, False]
    # an exact-dup at the component minimum is skipped; next-lowest becomes the survivor
    exact2 = np.array([True, False, False, False, False])
    mask2 = survivors_mask(roots, exact2)
    assert list(mask2) == [False, True, False, True, False]
