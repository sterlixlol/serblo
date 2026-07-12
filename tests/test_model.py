"""Tests for the 125M decoder. The overfit-a-batch test is the single most important
correctness check: it proves the whole graph (attention, RoPE, GQA, loss, optimizer) learns."""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from serblo.train.model import GPT, GPTConfig  # noqa: E402


def test_param_count_is_about_125M():
    n = GPT(GPTConfig()).num_params()
    assert 123_000_000 <= n <= 127_000_000, f"expected ~125M, got {n:,}"


def test_forward_shapes_and_loss():
    cfg = GPTConfig(n_layer=2, ctx=64)
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = m(idx, targets=idx)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)
    logits2, none = m(idx)
    assert none is None and logits2.shape == (2, 16, cfg.vocab_size)


def test_tied_embeddings_share_storage():
    m = GPT(GPTConfig(n_layer=1))
    assert m.lm_head.weight is m.tok_emb.weight


def test_gqa_projection_dims():
    cfg = GPTConfig(n_layer=1)
    a = GPT(cfg).blocks[0].attn
    hd = cfg.d_model // cfg.n_head
    assert a.q_proj.weight.shape[0] == cfg.n_head * hd       # 12*64 = 768
    assert a.k_proj.weight.shape[0] == cfg.n_kv_head * hd    # 4*64  = 256
    assert a.v_proj.weight.shape[0] == cfg.n_kv_head * hd


def test_causal_no_future_leak():
    cfg = GPTConfig(n_layer=2, ctx=32)
    m = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 10))
    with torch.no_grad():
        l1, _ = m(idx)
        idx2 = idx.clone()
        idx2[0, -1] = (idx2[0, -1] + 1) % cfg.vocab_size        # perturb ONLY the last token
        l2, _ = m(idx2)
    # positions before the last must be identical (causal: they never see the last token)
    assert torch.allclose(l1[0, :9], l2[0, :9], atol=1e-4)


def test_overfit_single_batch():
    torch.manual_seed(0)
    cfg = GPTConfig(n_layer=2, d_model=128, n_head=4, n_kv_head=2, mlp_hidden=256, ctx=64, vocab_size=256)
    m = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, 32))
    opt = m.configure_optimizers(weight_decay=0.0, lr=1e-3, betas=(0.9, 0.95), device="cpu")
    loss = None
    for _ in range(200):
        _, loss = m(x, targets=x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5, f"failed to overfit a single batch: loss={loss.item():.3f}"
