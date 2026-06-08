"""Fixed reference architecture for the ``synthetic_data_generator`` task.

Unlike ``ts_data_pipeline`` — whose frozen architecture is the *current best*
ts_forecasting model and is re-snapshotted every N rounds — the
synthetic-data-generator task pins a **single, never-changing** model. Miners
compete purely on the synthetic data they feed it; the architecture is the same
``code`` every round, so a continuation round is just "keep training the same
weights on (maybe better) data".

The reference model is a miniature (~10M-parameter) Toto-2.0-style **causal
patch decoder**: contiguous non-overlapping patches over the context window, a
learned patch embedding, a stack of causal (decoder) transformer blocks, and a
quantile head that forecasts the horizon from the final patch token. It is held
here as source *text* (not an importable module) so the validator can keep
running with only numpy installed — torch is only imported when the harness
``exec``s the string inside a synthetic_data_generator round.

The ``code`` satisfies the same ``build_model(context_len, prediction_len,
num_variates, quantiles)`` contract every ts_forecasting submission does, so the
existing harness / GIFT-Eval path drives it unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bumping this only makes sense if the reference model itself changes; it is
# stamped onto every experiment's objectives (``synth_arch_version``) and gates
# continuation lineages so a future model swap can't silently stitch
# incomparable checkpoints together.
REFERENCE_ARCH_VERSION = 1
REFERENCE_ARCH_NAME = "toto_mini_patch_decoder_10m"

# ~10M params: d_model=384, 6 causal blocks, ff=1280, patch=32 → 16 patches.
# (See tests/test_synth_generator.py which execs this and asserts the count.)
REFERENCE_ARCH_CODE = '''
"""Reference architecture: a miniature Toto-2.0-style causal patch decoder.

~10M parameters. Channel-independent contiguous patching over the context
window, learned patch + positional embeddings, a stack of causal transformer
decoder blocks (each patch attends only to itself and earlier patches), and a
quantile head that maps the final patch token to the full forecast horizon.

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


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x):
        b, t, c = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, b, heads, t, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Causal mask → patch token t only sees patches <= t (decoder).
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
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

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
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
        # Round the context up to a whole number of patches; forward() pads.
        self.n_patches = max(1, math.ceil(self.context_len / self.patch_size))

        self.patch_embed = nn.Linear(self.patch_size, D_MODEL)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches, D_MODEL) * 0.02
        )
        self.blocks = nn.ModuleList(
            [Block(D_MODEL, N_HEADS, FF_DIM, DROPOUT) for _ in range(N_LAYERS)]
        )
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, self.prediction_len * self.n_quantiles)

    def forward(self, x):
        # x: (B, context_len, num_variates). Be tolerant of a missing variate
        # axis so a (B, L) loader still works.
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        b, length, v = x.shape

        # Instance normalisation (RevIN-lite): per (series) over time. Keeps the
        # model scale-free; we denorm the forecast at the end.
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-5

        xn = (x - mean) / std
        # Channel-independent: fold variates into the batch.
        xc = xn.permute(0, 2, 1).reshape(b * v, length)

        # Pad up to a whole number of patches (left-pad with zeros = oldest).
        target_len = self.n_patches * self.patch_size
        if xc.shape[1] < target_len:
            pad = target_len - xc.shape[1]
            xc = F.pad(xc, (pad, 0))
        elif xc.shape[1] > target_len:
            xc = xc[:, -target_len:]

        patches = xc.reshape(b * v, self.n_patches, self.patch_size)
        h = self.patch_embed(patches) + self.pos_embed
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        # Decoder: forecast from the final (most recent) patch token.
        last = h[:, -1, :]
        out = self.head(last)
        out = out.reshape(b, v, self.prediction_len, self.n_quantiles)

        # Denorm per quantile: (B, V, 1, 1) broadcast.
        mean_ = mean.permute(0, 2, 1).unsqueeze(-1)
        std_ = std.permute(0, 2, 1).unsqueeze(-1)
        out = out * std_ + mean_

        # (B, prediction_len, num_variates, n_quantiles)
        return out.permute(0, 2, 1, 3)


def build_model(context_len, prediction_len, num_variates, quantiles):
    return TotoMiniPatchDecoder(
        context_len, prediction_len, num_variates, len(quantiles)
    )


def init_weights(model):
    # Standard transformer init. Must not change the parameter count (the
    # harness calls this before a strict warm-start load on continuation).
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


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


@dataclass
class ReferenceArch:
    """The fixed architecture every synthetic_data_generator round trains.

    Shaped to plug into the same ``frozen_arch`` challenge card / miner tools
    the ts_data_pipeline task already uses, so the miner-facing surface is
    identical — only the validator-side scoring and the (never-changing)
    version differ.
    """

    version: int
    name: str
    code: str

    def to_card(self) -> dict:
        """Challenge-payload ``frozen_arch`` card (mirrors FrozenArch fields)."""
        return {
            "version": self.version,
            "source_experiment_id": 0,
            "source_metric": 0.0,
            "source_crps": 0.0,
            "source_mase": 0.0,
            "source_flops": 0,
            "source_name": self.name,
            "code": self.code,
            "fixed": True,
        }


def reference_arch() -> ReferenceArch:
    """Return the single fixed reference architecture (cheap; no torch)."""
    return ReferenceArch(
        version=REFERENCE_ARCH_VERSION,
        name=REFERENCE_ARCH_NAME,
        code=REFERENCE_ARCH_CODE,
    )
