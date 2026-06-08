"""Source *text* of the fixed synthetic_data_generator reference model.

Kept in its own module (no torch import at load time) so ``synthetic_arch``
stays small and the validator runs numpy-only until a round execs this string.
See ``local/synthetic_arch.py`` for the version/card wrapper.

~10M params: d_model=384, 6 causal blocks, ff=1280, patch=32 → 16 context
patches + 3 forecast-query patches (horizon 96). tests/test_synth_generator.py
execs this string and asserts the count stays in the ~10M band.
"""

from __future__ import annotations

REFERENCE_ARCH_CODE = '''
"""Reference architecture: a miniature Toto-2.0-style causal patch decoder.

~10M parameters. Channel-independent contiguous patching with an arcsinh robust
scaler (median/MAD, missing values masked out), a residual-MLP patch projection,
learned patch + positional embeddings, and learned forecast-query tokens
appended to the sequence. A stack of causal decoder blocks with PerDimScale
attention runs once; each forecast-query token emits one contiguous output patch
through a residual-MLP head, so the whole horizon is produced in a single pass.

build_model(context_len, prediction_len, num_variates, quantiles) -> nn.Module
returning predictions shaped (batch, prediction_len, num_variates, n_quantiles).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 384
N_LAYERS = 6
N_HEADS = 6
FF_DIM = 1280
PATCH_SIZE = 32
DROPOUT = 0.1
_SOFTPLUS0 = 0.6931471805599453  # softplus(0) = ln 2


class ResidualMLP(nn.Module):
    """Linear in-projection + one residual GELU-MLP block (Toto-2.0 style)."""

    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, out_dim)
        self.fc1 = nn.Linear(out_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = self.in_proj(x)
        return h + self.fc2(F.gelu(self.fc1(h)))


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        # PerDimScale: a learned per-dim multiplier on the queries. At init
        # (zeros) it reproduces the standard 1/sqrt(head_dim) attention scale,
        # but lets the model learn a per-dimension temperature (u-μP style).
        self.per_dim_scale = nn.Parameter(torch.zeros(self.head_dim))

    def forward(self, x, attn_bias):
        b, t, c = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, b, heads, t, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = F.softplus(self.per_dim_scale) / _SOFTPLUS0 / math.sqrt(self.head_dim)
        q = q * scale
        # attn_bias carries the causal mask AND the key-padding (missing-value)
        # mask, so SDPA gets an explicit additive bias (scale folded in above).
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0,
        )
        out = out.transpose(1, 2).reshape(b, t, c)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_bias):
        x = x + self.drop(self.attn(self.ln1(x), attn_bias))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class TotoMiniPatchDecoder(nn.Module):
    def __init__(self, context_len, prediction_len, num_variates, n_quantiles):
        super().__init__()
        self.context_len = int(context_len)
        self.prediction_len = int(prediction_len)
        self.num_variates = int(num_variates)
        self.n_quantiles = int(n_quantiles)
        self.patch_size = PATCH_SIZE
        # Round context + horizon up to whole patches; forward() pads/slices.
        self.n_patches = max(1, math.ceil(self.context_len / self.patch_size))
        self.n_fut = max(1, math.ceil(self.prediction_len / self.patch_size))
        seq_len = self.n_patches + self.n_fut

        self.patch_embed = ResidualMLP(self.patch_size, D_MODEL, D_MODEL)
        # Learned forecast-query tokens: appended to the context so the causal
        # decoder produces every horizon patch in a single forward pass.
        self.forecast_query = nn.Parameter(
            torch.randn(1, self.n_fut, D_MODEL) * 0.02
        )
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, D_MODEL) * 0.02)
        self.blocks = nn.ModuleList(
            [Block(D_MODEL, N_HEADS, FF_DIM, DROPOUT) for _ in range(N_LAYERS)]
        )
        self.ln_f = nn.LayerNorm(D_MODEL)
        # Each forecast-query token emits one contiguous output patch (all
        # quantiles), not the whole flattened horizon off a single token.
        self.head = ResidualMLP(
            D_MODEL, self.patch_size * self.n_quantiles, D_MODEL
        )

    def _robust_scale(self, x, mask):
        # x, mask: (B, L, V). arcsinh robust scaler: median location + mean
        # absolute deviation scale over *observed* timesteps only. Series that
        # are fully missing fall back to loc=0, scale=1.
        obs = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        big = x.masked_fill(~mask.bool(), float("inf"))
        loc = big.median(dim=1, keepdim=True).values
        loc = torch.where(torch.isfinite(loc), loc, torch.zeros_like(loc))
        dev = (x - loc).abs().masked_fill(~mask.bool(), 0.0)
        scale = dev.sum(dim=1, keepdim=True) / obs + 1e-5
        return loc, scale

    def forward(self, x):
        # x: (B, context_len, num_variates). Tolerate a missing variate axis.
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        b, length, v = x.shape

        # Missing-value masking: NaN/inf → masked, never seen by the model.
        mask = torch.isfinite(x).to(x.dtype)
        x = torch.where(mask.bool(), x, torch.zeros_like(x))
        loc, scale = self._robust_scale(x, mask)
        xn = torch.asinh((x - loc) / scale) * mask  # missing → 0

        # Channel-independent: fold variates into the batch.
        xc = xn.permute(0, 2, 1).reshape(b * v, length)
        mc = mask.permute(0, 2, 1).reshape(b * v, length)

        # Pad up to a whole number of patches (left-pad = oldest).
        target_len = self.n_patches * self.patch_size
        if xc.shape[1] < target_len:
            pad = target_len - xc.shape[1]
            xc = F.pad(xc, (pad, 0))
            mc = F.pad(mc, (pad, 0))
        elif xc.shape[1] > target_len:
            xc, mc = xc[:, -target_len:], mc[:, -target_len:]

        patches = xc.reshape(b * v, self.n_patches, self.patch_size)
        # A patch is a usable key if it holds any observed value.
        patch_mask = mc.reshape(b * v, self.n_patches, self.patch_size).amax(-1)

        h = self.patch_embed(patches) + self.pos_embed[:, : self.n_patches]
        q = self.forecast_query + self.pos_embed[:, self.n_patches :]
        h = torch.cat([h, q.expand(b * v, -1, -1)], dim=1)

        attn_bias = self._attn_bias(patch_mask, h)
        for blk in self.blocks:
            h = blk(h, attn_bias)
        h = self.ln_f(h)

        # Single-pass multi-patch forecast: read the forecast-query tokens.
        fut = h[:, self.n_patches :, :]                  # (B*V, n_fut, D)
        out = self.head(fut)                             # (B*V, n_fut, P*Q)
        out = out.reshape(
            b * v, self.n_fut * self.patch_size, self.n_quantiles
        )[:, : self.prediction_len, :]
        out = out.reshape(b, v, self.prediction_len, self.n_quantiles)

        # Inverse arcsinh robust scaler (monotone → preserves quantile order).
        loc_ = loc.permute(0, 2, 1).unsqueeze(-1)
        scale_ = scale.permute(0, 2, 1).unsqueeze(-1)
        out = torch.sinh(out) * scale_ + loc_

        # (B, prediction_len, num_variates, n_quantiles)
        return out.permute(0, 2, 1, 3)

    def _attn_bias(self, patch_mask, h):
        # Additive attention bias = causal mask | key-padding mask. Context
        # patches with no observed value are masked as keys; forecast-query
        # keys are always valid; the diagonal is always kept open so no query
        # row is fully masked (which would NaN the softmax).
        bv, seq = h.shape[0], h.shape[1]
        dev = h.device
        valid = torch.ones(bv, seq, dtype=torch.bool, device=dev)
        valid[:, : self.n_patches] = patch_mask > 0
        causal = torch.triu(
            torch.ones(seq, seq, dtype=torch.bool, device=dev), diagonal=1
        )
        disallow = causal[None] | (~valid)[:, None, :]
        disallow = disallow & ~torch.eye(seq, dtype=torch.bool, device=dev)[None]
        bias = torch.zeros(bv, 1, seq, seq, dtype=h.dtype, device=dev)
        return bias.masked_fill(disallow[:, None], float("-inf"))


def build_model(context_len, prediction_len, num_variates, quantiles):
    return TotoMiniPatchDecoder(
        context_len, prediction_len, num_variates, len(quantiles)
    )


def init_weights(model):
    # u-μP-flavored init: base normal(0, 0.02); residual-output projections
    # (attention out-proj + MLP fc2) depth-scaled by 1/sqrt(2 * N_LAYERS) so the
    # residual stream stays unit-ish as depth grows. Must not change the
    # parameter count (the harness calls this before a strict warm-start load on
    # continuation).
    depth_scale = 1.0 / math.sqrt(2 * N_LAYERS)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    for blk in model.blocks:
        blk.attn.proj.weight.data.mul_(depth_scale)
        blk.mlp[3].weight.data.mul_(depth_scale)


def build_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95)
    )


def training_config():
    return {"batch_size": 64, "grad_accum_steps": 1, "grad_clip": 1.0}


def build_scheduler(optimizer, total_steps):
    total_steps = max(1, int(total_steps))
    warmup = max(1, total_steps // 20)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
'''
