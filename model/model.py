"""
Apollo-Helios transformer.

The 'baseline' preset matches Apollo v2's exact model code (manual causal
attention, LayerNorm with bias, dropout=0.1, GELU FFN). Each modern component
is gated by a config flag.

Toggleable components:
    use_rope        : RoPE vs learned absolute position embeddings
    norm_type       : 'rmsnorm' vs 'layernorm'
    use_qk_norm     : RMSNorm on Q and K post-projection
    ffn_type        : 'swiglu' vs 'gelu'

Optimizer choice (muon vs adamw) is handled in train.py, not here.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ApolloHeliosConfig:
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 256
    vocab_size: int = 32000
    dropout: float = 0.1     # Apollo trained with this
    bias: bool = False       # Linear bias=False; LayerNorm uses default bias=True

    use_rope: bool = True
    norm_type: str = 'rmsnorm'
    use_qk_norm: bool = True
    ffn_type: str = 'swiglu'

    swiglu_hidden: int = 1344
    gelu_hidden: int = 2048
    init_std: float = 0.02


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def make_norm(dim, norm_type):
    if norm_type == 'rmsnorm':
        return RMSNorm(dim)
    elif norm_type == 'layernorm':
        # Default bias=True to match Apollo v2's nn.LayerNorm(dim)
        return nn.LayerNorm(dim)
    raise ValueError(f"unknown norm_type: {norm_type}")


def build_rope_cache(seq_len, head_dim, device, base=10000.0):
    assert head_dim % 2 == 0
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, theta)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rot1
    out[..., 1::2] = rot2
    return out


class CausalSelfAttention(nn.Module):
    """
    Manual causal self-attention with masked_fill — matches Apollo v2 exactly.
    Optionally adds RoPE and/or QK-norm.
    """
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_embd = cfg.n_embd
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.use_rope = cfg.use_rope
        self.use_qk_norm = cfg.use_qk_norm

        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        if cfg.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x, rope_cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope and rope_cache is not None:
            cos, sin = rope_cache
            q = apply_rope(q, cos[:T], sin[:T])
            k = apply_rope(k, cos[:T], sin[:T])

        # Manual scaled dot-product attention with causal mask (Apollo's exact code)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class GELU_FFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, cfg.gelu_hidden, bias=cfg.bias)
        self.fc2 = nn.Linear(cfg.gelu_hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class SwiGLU_FFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.swiglu_hidden
        self.w1 = nn.Linear(cfg.n_embd, h, bias=cfg.bias)
        self.w2 = nn.Linear(cfg.n_embd, h, bias=cfg.bias)
        self.w3 = nn.Linear(h, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


def make_ffn(cfg):
    if cfg.ffn_type == 'swiglu':
        return SwiGLU_FFN(cfg)
    elif cfg.ffn_type == 'gelu':
        return GELU_FFN(cfg)
    raise ValueError(f"unknown ffn_type: {cfg.ffn_type}")


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = make_norm(cfg.n_embd, cfg.norm_type)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = make_norm(cfg.n_embd, cfg.norm_type)
        self.ffn = make_ffn(cfg)

    def forward(self, x, rope_cache=None):
        x = x + self.attn(self.norm1(x), rope_cache=rope_cache)
        x = x + self.ffn(self.norm2(x))
        return x


class ApolloHelios(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        if not cfg.use_rope:
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        else:
            self.pos_emb = None
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = make_norm(cfg.n_embd, cfg.norm_type)

        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self._rope_cache_device = None
        self._rope_cache = None

        # GPT-2 init: 0.02 std, residual projections scaled by 1/sqrt(2L)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if (name.endswith('proj.weight')
                    or name.endswith('w3.weight')
                    or name.endswith('fc2.weight')):
                nn.init.normal_(p, mean=0.0, std=cfg.init_std / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def _get_rope_cache(self, T, device):
        if not self.cfg.use_rope:
            return None
        head_dim = self.cfg.n_embd // self.cfg.n_head
        if self._rope_cache_device != device or self._rope_cache is None:
            cos, sin = build_rope_cache(self.cfg.block_size, head_dim, device)
            self._rope_cache = (cos, sin)
            self._rope_cache_device = device
        return self._rope_cache

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size

        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            x = x + self.pos_emb(pos)
        x = self.drop(x)

        rope_cache = self._get_rope_cache(T, idx.device) if self.cfg.use_rope else None

        for block in self.blocks:
            x = block(x, rope_cache=rope_cache)

        x = self.norm_f(x)
        logits = self.head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def num_params(self, exclude_embeddings=True):
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                n -= self.pos_emb.weight.numel()
        return n


ABLATION_PRESETS = {
    'baseline': dict(use_rope=False, norm_type='layernorm', use_qk_norm=False, ffn_type='gelu'),
    'rope':     dict(use_rope=True,  norm_type='layernorm', use_qk_norm=False, ffn_type='gelu'),
    'rmsnorm':  dict(use_rope=False, norm_type='rmsnorm',   use_qk_norm=False, ffn_type='gelu'),
    'qknorm':   dict(use_rope=False, norm_type='layernorm', use_qk_norm=True,  ffn_type='gelu'),
    'swiglu':   dict(use_rope=False, norm_type='layernorm', use_qk_norm=False, ffn_type='swiglu'),
    'modern':   dict(use_rope=True,  norm_type='rmsnorm',   use_qk_norm=True,  ffn_type='swiglu'),
}


def build_model(preset, vocab_size=32000, block_size=256, dropout=0.1):
    if preset not in ABLATION_PRESETS:
        raise ValueError(f"unknown preset {preset}")
    cfg = ApolloHeliosConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        dropout=dropout,
        **ABLATION_PRESETS[preset],
    )
    return ApolloHelios(cfg)


if __name__ == '__main__':
    for name in ABLATION_PRESETS:
        m = build_model(name)
        body = m.num_params(exclude_embeddings=True)
        total = m.num_params(exclude_embeddings=False)
        print(f"{name:10s}  body={body/1e6:6.2f}M  total={total/1e6:6.2f}M")
