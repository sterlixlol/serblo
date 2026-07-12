"""Tests for the shard data loader."""
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from serblo.train.data import ShardLoader  # noqa: E402


def _make(tmp_path, n=2000):
    np.arange(n, dtype=np.uint16).tofile(tmp_path / "train_00000.bin")   # token id == position
    (tmp_path / "meta.json").write_text(json.dumps({"train_shards": ["train_00000.bin"], "val_shards": []}))


def test_batch_shapes_and_shift(tmp_path):
    _make(tmp_path)
    ld = ShardLoader(str(tmp_path), ctx=16, split="train")
    x, y = ld.get_batch(8, "cpu", torch.Generator().manual_seed(0))
    assert x.shape == (8, 16) and y.shape == (8, 16)
    assert torch.equal(y[:, :-1], x[:, 1:])            # y is x shifted by one
    # since token id == position, each row is consecutive integers
    assert torch.equal(x[0], torch.arange(int(x[0, 0]), int(x[0, 0]) + 16))


def test_deterministic_with_generator(tmp_path):
    _make(tmp_path)
    ld = ShardLoader(str(tmp_path), ctx=32, split="train")
    x1, _ = ld.get_batch(4, "cpu", torch.Generator().manual_seed(42))
    x2, _ = ld.get_batch(4, "cpu", torch.Generator().manual_seed(42))
    assert torch.equal(x1, x2)


def test_rejects_shards_too_small(tmp_path):
    np.arange(10, dtype=np.uint16).tofile(tmp_path / "train_00000.bin")
    (tmp_path / "meta.json").write_text(json.dumps({"train_shards": ["train_00000.bin"], "val_shards": []}))
    try:
        ShardLoader(str(tmp_path), ctx=64, split="train")
        assert False, "should have raised on too-small shard"
    except ValueError:
        pass
