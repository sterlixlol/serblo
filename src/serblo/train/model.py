"""Serblo decoder. Llama-style, plain PyTorch.

RoPE, grouped-query attention (12 Q / 4 KV heads), RMSNorm, SwiGLU, tied embeddings.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 32000
    n_layer: int = 14
    n_head: int = 12
    n_kv_head: int = 4
    d_model: int = 768
    mlp_hidden: int = 2432  # lands the total at ~125M
    ctx: int = 2048
    tie_embeddings: bool = True
    rope_theta: float = 10000.0
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dt = x.dtype
        x = x.float()  # norm in fp32
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dt) * self.weight


def build_rope(head_dim, ctx, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(ctx).float(), inv_freq)  # [ctx, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [ctx, head_dim]
    return emb.cos(), emb.sin()


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(x, cos, sin):  # x: [B, n, T, hd]; cos/sin: [T, hd]
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nh, self.nkv = cfg.n_head, cfg.n_kv_head
        self.hd = cfg.d_model // cfg.n_head
        self.rep = cfg.n_head // cfg.n_kv_head
        self.q_proj = nn.Linear(cfg.d_model, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.rep > 1:  # repeat KV heads to match Q heads
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, self.nh * self.hd)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.w_down = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.n2, self.mlp = RMSNorm(cfg.d_model), SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.mlp(self.n2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        cos, sin = build_rope(cfg.d_model // cfg.n_head, cfg.ctx, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)  # not saved, rebuilt from cfg
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():  # GPT-2 residual init
            if name.endswith("o_proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.ctx, f"sequence length {T} exceeds ctx {self.cfg.ctx}"
        x = self.tok_emb(idx)
        cos, sin = self.rope_cos[:T].to(x.dtype), self.rope_sin[:T].to(x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=-1)
        return logits, loss

    def num_params(self, non_embedding=False):
        # the tied weight only gets counted once
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def configure_optimizers(self, weight_decay, lr, betas, device):
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": nodecay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=str(device).startswith("cuda"))
