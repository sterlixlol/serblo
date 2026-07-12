"""Tests for the token-shard packer. Uses a fake char-level encoder so the packing logic is
tested independently of the real tokenizer."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from serblo.train.pack import pack_docs  # noqa: E402

EOS = 3


def enc(s):
    # deterministic, uint16-safe, avoids the special-token ids 0-3
    return [ord(c) % 250 + 4 for c in s]


def read_u16(p):
    return np.fromfile(p, dtype=np.uint16)


def cat(dir_, pat):
    files = sorted(pathlib.Path(dir_).glob(pat))
    return np.concatenate([read_u16(f) for f in files]) if files else np.array([], dtype=np.uint16)


def test_roundtrip_and_eos(tmp_path):
    docs = ["abc", "de", "fghij"]
    pack_docs(iter(docs), enc, str(tmp_path), eos_id=EOS, shard_tokens=10_000, val_every=10_000)
    toks = cat(tmp_path, "train_*.bin")
    expected = enc("abc") + [EOS] + enc("de") + [EOS] + enc("fghij") + [EOS]
    assert toks.tolist() == expected
    assert toks.dtype == np.uint16


def test_uint16_dtype(tmp_path):
    pack_docs(iter(["hello world"] * 50), enc, str(tmp_path), eos_id=EOS, shard_tokens=200, val_every=10_000)
    for f in pathlib.Path(tmp_path).glob("train_*.bin"):
        assert read_u16(f).dtype == np.uint16


def test_val_split(tmp_path):
    docs = [f"document number {i} with text" for i in range(1000)]
    meta = pack_docs(iter(docs), enc, str(tmp_path), eos_id=EOS, shard_tokens=100_000, val_every=100)
    total = meta["train_tokens"] + meta["val_tokens"]
    assert meta["val_tokens"] > 0
    # every 100th of 1000 docs -> ~1% in val; disjoint by construction (index routing)
    assert 0.003 < meta["val_tokens"] / total < 0.03
    assert cat(tmp_path, "train_*.bin").size == meta["train_tokens"]
    assert cat(tmp_path, "val_*.bin").size == meta["val_tokens"]


def test_resume_after_crash_is_identical(tmp_path):
    """Crash mid-run then resume must produce byte-identical output to a clean run (no loss, no dup)."""
    docs = [f"doc {i} " * 8 for i in range(300)]
    ref = tmp_path / "ref"
    pack_docs(iter(docs), enc, str(ref), eos_id=EOS, shard_tokens=400, val_every=100_000)
    ref_tokens = cat(ref, "train_*.bin")

    work = tmp_path / "work"

    def crashing():
        for i, d in enumerate(docs):
            if i == 150:
                raise RuntimeError("simulated preemption")
            yield d

    try:
        pack_docs(crashing(), enc, str(work), eos_id=EOS, shard_tokens=400, val_every=100_000)
    except RuntimeError:
        pass
    # resume with the full stream
    pack_docs(iter(docs), enc, str(work), eos_id=EOS, shard_tokens=400, val_every=100_000)
    assert cat(work, "train_*.bin").tolist() == ref_tokens.tolist()


def test_rerun_is_noop(tmp_path):
    docs = [f"line {i}" for i in range(100)]
    m1 = pack_docs(iter(docs), enc, str(tmp_path), eos_id=EOS, shard_tokens=300, val_every=100_000)
    first = cat(tmp_path, "train_*.bin").tolist()
    m2 = pack_docs(iter(docs), enc, str(tmp_path), eos_id=EOS, shard_tokens=300, val_every=100_000)
    assert cat(tmp_path, "train_*.bin").tolist() == first  # idempotent when already complete
    assert m2["train_tokens"] == m1["train_tokens"]
