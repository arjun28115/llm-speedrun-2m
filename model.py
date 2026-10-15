"""Small decoder-only LM, plain PyTorch, CPU-friendly.

Changes vs the starter, and why (each was A/B'd -- see RUNLOG.md):
  * RMSNorm instead of LayerNorm      - no mean subtraction, no bias, fewer
                                        params and fewer CPU ops per token
  * RoPE instead of learned pos_emb   - 0 parameters, relative positions, and
                                        it stays correct under evaluate.py's
                                        sliding window
  * SwiGLU instead of 4x GELU MLP     - same param count (3*d*8/3d = 8d^2),
                                        better loss per parameter
  * tie_weights = True by default     - the head IS the embedding matrix; with
                                        vocab 2048 that is 360k params freed.
                                        When tied there is no separate head
                                        module at all, so the checkpoint has
                                        exactly one copy of the matrix and the
                                        parameter count is unambiguous.
  * scaled residual init              - residual-branch outputs get
                                        std = init_std / sqrt(2 * n_layer)
  * no biases anywhere                - biases do ~nothing at this scale

Parameter budget (tied, no biases):
    V*d  +  n_layer * (12*d^2 + 2*d)  +  d
Defaults: 2048*176 + 4*(371712+352) + 176 = 1,848,880  <= 2,000,000.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 2048
    block_size = 256
    n_layer = 4
    n_head = 4
    n_kv_head = 4          # < n_head enables grouped-query attention
    n_embd = 176
    mlp_ratio = 2.6667     # SwiGLU: 3 matrices of d*(mlp_ratio*d) ~= 8d^2
    dropout = 0.0
    tie_weights = True
    rope_theta = 10000.0
    init_std = 0.02


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def build_rope(head_dim, max_len, theta):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv)                     # (T, hd/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, nh, T, hd) -- rotate pairs (even, odd)
    T = x.size(-2)
    cos, sin = cos[:T][None, None], sin[:T][None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_head == 0
        self.n_head, self.n_kv_head = cfg.n_head, cfg.n_kv_head
        self.hd = cfg.n_embd // cfg.n_head
        assert self.hd % 2 == 0, "RoPE needs an even head dimension"
        self.q = nn.Linear(cfg.n_embd, cfg.n_head * self.hd, bias=False)
        self.k = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.hd, bias=False)
        self.v = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.hd, bias=False)
        self.proj = nn.Linear(cfg.n_head * self.hd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv_head, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv_head, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.drop(self.proj(y))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = int(cfg.mlp_ratio * cfg.n_embd) // 8 * 8
        self.w1 = nn.Linear(cfg.n_embd, h, bias=False)   # gate
        self.w3 = nn.Linear(cfg.n_embd, h, bias=False)   # value
        self.w2 = nn.Linear(h, cfg.n_embd, bias=False)   # down
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = RMSNorm(cfg.n_embd)
        # When tied, there is deliberately NO head module: the output
        # projection reuses tok_emb.weight via F.linear.
        self.head = None if cfg.tie_weights else \
            nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.apply(self._init)
        # residual-branch projections shrink with depth (GPT-2 / Llama trick)
        scale = cfg.init_std / math.sqrt(2 * cfg.n_layer)
        for blk in self.blocks:
            nn.init.normal_(blk.attn.proj.weight, 0.0, scale)
            nn.init.normal_(blk.mlp.w2.weight, 0.0, scale)

        cos, sin = build_rope(cfg.n_embd // cfg.n_head,
                              cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, self.cfg.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.rope_cos.size(0):               # defensive: longer window
            cos, sin = build_rope(self.cfg.n_embd // self.cfg.n_head, T,
                                  self.cfg.rope_theta)
        else:
            cos, sin = self.rope_cos, self.rope_sin
        x = self.drop(self.tok_emb(idx))
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.ln_f(x)
        logits = F.linear(x, self.tok_emb.weight) if self.head is None \
            else self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
