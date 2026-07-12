"""Tests for the training loop's money-critical helpers: LR schedule, atomic checkpoint
round-trip, grad-accum scaling, and BIT-EXACT resume (the spot-preemption safety proof)."""
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from serblo.train.data import ShardLoader                                    # noqa: E402
from serblo.train.model import GPT, GPTConfig                                 # noqa: E402
from serblo.train.train import (load_checkpoint, lr_at, save_checkpoint,      # noqa: E402
                                train_step)

TINY = dict(n_layer=2, d_model=64, n_head=4, n_kv_head=2, mlp_hidden=128, ctx=32, vocab_size=200)


def _opt(m):
    return m.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")


def test_lr_schedule():
    peak, warmup, total = 4e-4, 100, 1000
    assert lr_at(0, warmup, total, peak) == peak * 1 / warmup            # small at start
    assert abs(lr_at(warmup, warmup, total, peak) - peak) < 1e-12         # peak at end of warmup
    assert abs(lr_at(total, warmup, total, peak) - peak * 0.1) < 1e-12    # floor at the end
    mid = lr_at(warmup + (total - warmup) // 2, warmup, total, peak)
    assert peak * 0.1 < mid < peak                                        # decaying in between


def test_grad_accum_matches_full_batch():
    torch.manual_seed(0)
    m = GPT(GPTConfig(**TINY))
    x = torch.randint(0, TINY["vocab_size"], (8, 16))
    m.zero_grad()
    _, loss = m(x, x)
    loss.backward()
    full = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    for i in range(4):                                                    # 4 micro-batches of 2, each /4
        _, l = m(x[i * 2:(i + 1) * 2], x[i * 2:(i + 1) * 2])
        (l / 4).backward()
    for a, p in zip(full, m.parameters()):
        assert torch.allclose(a, p.grad, atol=1e-5)


def test_checkpoint_roundtrip(tmp_path):
    m = GPT(GPTConfig(**TINY))
    opt = _opt(m)
    x = torch.randint(0, TINY["vocab_size"], (2, 16))
    _, loss = m(x, x)
    loss.backward()
    opt.step()                                                           # give the optimizer state
    p = tmp_path / "ck.pt"
    save_checkpoint(p, m, opt, step=7, tokens=123, best_val=1.5, cfg_dict=TINY)
    m2 = GPT(GPTConfig(**TINY))
    ck = load_checkpoint(p, m2, _opt(m2))
    assert (ck["step"], ck["tokens"], ck["best_val"]) == (7, 123, 1.5)
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.equal(a, b)


def test_bit_exact_resume(tmp_path):
    (np.arange(5000) % TINY["vocab_size"]).astype(np.uint16).tofile(tmp_path / "train_00000.bin")  # ids < vocab
    (tmp_path / "meta.json").write_text(json.dumps({"train_shards": ["train_00000.bin"], "val_shards": []}))

    def five_steps(model, opt):
        ld = ShardLoader(str(tmp_path), 32, "train")
        return [train_step(model, opt, ld, 2, 2, "cpu", False, 1.0, 1e-3)[0] for _ in range(5)]

    torch.manual_seed(123)
    mA = GPT(GPTConfig(**TINY))
    optA = _opt(mA)
    ckp = tmp_path / "ck.pt"
    save_checkpoint(ckp, mA, optA, 0, 0, float("inf"), TINY)             # save at step 0
    losses_a = five_steps(mA, optA)

    mB = GPT(GPTConfig(**TINY))                                          # fresh model, different init
    optB = _opt(mB)
    load_checkpoint(ckp, mB, optB)                                       # restores weights + RNG
    losses_b = five_steps(mB, optB)

    assert losses_a == losses_b, "resume was not bit-exact"
