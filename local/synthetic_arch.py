"""Fixed reference architecture for the ``synthetic_data_generator`` task.

Unlike ``ts_data_pipeline`` — whose frozen architecture is the *current best*
ts_forecasting model and is re-snapshotted every N rounds — the
synthetic-data-generator task pins a **single, never-changing** model. Miners
compete purely on the synthetic data they feed it; the architecture is the same
``code`` every round, so a continuation round is just "keep training the same
weights on (maybe better) data".

The reference model is a miniature (~10M-parameter) **Toto-2.0-style causal
patch decoder**, implementing the salient pieces of the real design rather than
shortcut stand-ins:

* contiguous channel-independent patching with an **arcsinh robust scaler**
  (median/IQR centering + ``asinh`` squashing) and **missing-value masking**;
* **residual-MLP patch projections** at the input embed and the output head;
* causal (decoder) transformer blocks with **PerDimScale** attention and
  μP-flavored init (u-μP family);
* **single-pass contiguous patch masking**: learnable horizon-query tokens are
  appended after the context patches and decode the forecast as contiguous
  output patches in one forward (no autoregressive rollout);
* a quantile head (pinball loss / quantile sorting live in the harness);
* a **NorMuon** optimizer (Newton–Schulz-orthogonalized Muon update with
  per-neuron RMS normalization) for the 2-D weight matrices, AdamW for the rest.

It is held here as source *text* (not an importable module) so the validator
can keep running with only numpy installed — torch is only imported when the
harness ``exec``s the string inside a synthetic_data_generator round. The
``code`` satisfies the same ``build_model(context_len, prediction_len,
num_variates, quantiles)`` contract every ts_forecasting submission does.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bump only when the reference model itself changes; stamped onto every
# experiment's objectives (``synth_arch_version``) and gates continuation
# lineages so a model swap can't silently stitch incomparable checkpoints
# together. v2 = the faithful Toto-2.0-style rewrite.
REFERENCE_ARCH_VERSION = 2
REFERENCE_ARCH_NAME = "toto_mini_patch_decoder_10m"

# ~10M params: d_model=384, 6 causal blocks, ff=1280, patch=32.
# (tests/test_synth_generator.py execs this and asserts the count + shapes.)
REFERENCE_ARCH_CODE = '''
"""Reference architecture: a miniature Toto-2.0-style causal patch decoder.

~10M parameters. build_model(context_len, prediction_len, num_variates,
quantiles) -> nn.Module returning predictions shaped
(batch, prediction_len, num_variates, n_quantiles).
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
_R_SOFTPLUS_0 = 1.0 / math.log(2.0)  # 1 / softplus(0); PerDimScale calibration


def robust_scale(x, mask):
    """arcsinh robust scaler over observed steps.

    x, mask: (B, L, V). Returns (z, loc, scale) where z is the scaled,
    missing-zeroed signal and loc/scale are (B, V). Stats are taken over
    observed (mask=True, finite) steps only; degenerate/all-missing series
    fall back to loc=0, scale=1.
    """
    x = x.float()
    xn = torch.where(mask, x, torch.full_like(x, float("nan")))
    loc = torch.nanmedian(xn, dim=1).values                      # (B, V)
    qs = torch.tensor([0.25, 0.75], device=x.device, dtype=x.dtype)
    q = torch.nanquantile(xn, qs, dim=1)                          # (2, B, V)
    scale = (q[1] - q[0]) / 1.349
    loc = torch.nan_to_num(loc, nan=0.0)
    scale = torch.nan_to_num(scale, nan=1.0)
    scale = torch.where(scale > 1e-5, scale, torch.ones_like(scale))
    z = torch.asinh((x - loc.unsqueeze(1)) / scale.unsqueeze(1))
    z = torch.where(mask, z, torch.zeros_like(z))
    return z, loc, scale


class ResMLP(nn.Module):
    """Pre-norm residual MLP (patch projection block)."""

    def __init__(self, d, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.ln(x)))))


class CausalSelfAttention(nn.Module):
    """Causal MHA with PerDimScale (learnable per-dim query scaling)."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        # PerDimScale (PaLM): init 0 → softplus(0) → standard 1/sqrt(head_dim).
        self.per_dim_scale = nn.Parameter(torch.zeros(self.head_dim))

    def forward(self, x):
        b, t, c = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, b, heads, t, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = (1.0 / math.sqrt(self.head_dim)) * _R_SOFTPLUS_0 \
            * F.softplus(self.per_dim_scale)
        q = q * scale  # broadcast over (head_dim,)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True, scale=1.0,  # scaling already folded into q
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
        self.n_patches = max(1, math.ceil(self.context_len / self.patch_size))
        # Contiguous output patches that cover the horizon (single-pass).
        self.n_out_patches = max(1, math.ceil(self.prediction_len / self.patch_size))
        n_tokens = self.n_patches + self.n_out_patches

        # Patch embed reads values + observed-mask (channel-independent) → 2*P.
        self.patch_in = nn.Linear(2 * self.patch_size, D_MODEL)
        self.in_res = ResMLP(D_MODEL, DROPOUT)
        # Learnable horizon-query content + shared positional embedding.
        self.horizon_query = nn.Parameter(
            torch.randn(1, self.n_out_patches, D_MODEL) * 0.02
        )
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, D_MODEL) * 0.02)

        self.blocks = nn.ModuleList(
            [Block(D_MODEL, N_HEADS, FF_DIM, DROPOUT) for _ in range(N_LAYERS)]
        )
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.out_res = ResMLP(D_MODEL, DROPOUT)
        # Each output-patch token decodes one contiguous patch × quantiles.
        self.out_head = nn.Linear(D_MODEL, self.patch_size * self.n_quantiles)

    def forward(self, x):
        # x: (B, context_len, num_variates); tolerate a missing variate axis.
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        b, length, v = x.shape

        mask = torch.isfinite(x)
        x = torch.nan_to_num(x, nan=0.0)
        z, loc, scale = robust_scale(x, mask)

        # Channel-independent: fold variates into the batch.
        zc = z.permute(0, 2, 1).reshape(b * v, length)
        mc = mask.permute(0, 2, 1).reshape(b * v, length).to(zc.dtype)

        target_len = self.n_patches * self.patch_size
        if zc.shape[1] < target_len:
            pad = target_len - zc.shape[1]
            zc = F.pad(zc, (pad, 0))
            mc = F.pad(mc, (pad, 0))  # padded steps are "missing"
        elif zc.shape[1] > target_len:
            zc, mc = zc[:, -target_len:], mc[:, -target_len:]

        zp = zc.reshape(b * v, self.n_patches, self.patch_size)
        mp = mc.reshape(b * v, self.n_patches, self.patch_size)
        feat = torch.cat([zp, mp], dim=-1)            # (B*V, n_patches, 2P)
        ctx = self.in_res(self.patch_in(feat))        # (B*V, n_patches, D)

        queries = self.horizon_query.expand(b * v, -1, -1)
        h = torch.cat([ctx, queries], dim=1)          # (B*V, n_tokens, D)
        h = h + self.pos_embed
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)

        # Read the appended output-patch tokens (single-pass decode).
        out_tok = self.out_res(h[:, self.n_patches:, :])
        out = self.out_head(out_tok)                  # (B*V, n_out, P*Q)
        out = out.reshape(b * v, self.n_out_patches, self.patch_size, self.n_quantiles)
        out = out.reshape(b * v, self.n_out_patches * self.patch_size, self.n_quantiles)
        out = out[:, :self.prediction_len, :]         # (B*V, pred, Q)
        out = out.reshape(b, v, self.prediction_len, self.n_quantiles)

        # Invert the robust scaler per series (out is (B, V, pred, Q) here).
        # Clamp the arcsinh-space prediction so sinh can't overflow float32
        # (sinh(±88) ≈ 1.6e38) and blow up the loss into NaN.
        loc_ = loc.unsqueeze(-1).unsqueeze(-1)        # (B, V, 1, 1)
        scale_ = scale.unsqueeze(-1).unsqueeze(-1)
        out = torch.sinh(out.clamp(-30.0, 30.0)) * scale_ + loc_

        # (B, prediction_len, num_variates, n_quantiles)
        return out.permute(0, 2, 1, 3)


# ── NorMuon optimizer ───────────────────────────────────────────────


def _newton_schulz5(G, steps=5, eps=1e-7):
    """Quintic Newton–Schulz orthogonalization of a 2-D matrix (Muon)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.t()
        transposed = True
    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.t()
    return X


class NorMuon(torch.optim.Optimizer):
    """Muon (Newton–Schulz-orthogonalized momentum) with per-neuron RMS
    normalization, for 2-D weight matrices; AdamW for everything else.

    Param groups carry ``use_muon``: the 2-D matrices get the Muon path, the
    rest (biases, norms, embeddings, PerDimScale) get AdamW. The per-neuron
    second-moment normalization (NorMuon) plus a unit-RMS rescale make the
    update scale width-robust (μP-like), so a single ``lr`` transfers.
    """

    def __init__(self, params, lr=2e-2, momentum=0.95, beta2=0.95,
                 betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, use_muon=True):
        defaults = dict(lr=lr, momentum=momentum, beta2=beta2, betas=betas,
                        eps=eps, weight_decay=weight_decay, use_muon=use_muon)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if group["use_muon"] and p.ndim == 2:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                        state["v"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).add_(g)
                    upd = g.add(buf, alpha=group["momentum"])  # Nesterov
                    o = _newton_schulz5(upd).to(p.dtype)
                    # NorMuon: per-neuron 2nd-moment EMA, then unit-RMS rescale.
                    vbuf = state["v"]
                    vbuf.mul_(group["beta2"]).addcmul_(o, o, value=1 - group["beta2"])
                    o = o / (vbuf.sqrt() + eps)
                    o = o / (o.square().mean().sqrt() + eps)
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.add_(o, alpha=-lr)
                else:
                    if "step" not in state:
                        state["step"] = 0
                        state["m"] = torch.zeros_like(g)
                        state["v2"] = torch.zeros_like(g)
                    state["step"] += 1
                    b1, b2 = group["betas"]
                    m, v2 = state["m"], state["v2"]
                    m.mul_(b1).add_(g, alpha=1 - b1)
                    v2.mul_(b2).addcmul_(g, g, value=1 - b2)
                    bc1 = 1 - b1 ** state["step"]
                    bc2 = 1 - b2 ** state["step"]
                    denom = (v2.sqrt() / math.sqrt(bc2)).add_(eps)
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.addcdiv_(m, denom, value=-lr / bc1)
        return loss


# ── hooks ───────────────────────────────────────────────────────────


def build_model(context_len, prediction_len, num_variates, quantiles):
    return TotoMiniPatchDecoder(
        context_len, prediction_len, num_variates, len(quantiles)
    )


def init_weights(model):
    # u-μP-flavored init. Must not change the parameter count (the harness
    # calls this before a strict warm-start load on continuation).
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                fan_in = m.weight.shape[1]
                nn.init.normal_(m.weight, 0.0, (1.0 / fan_in) ** 0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(model.pos_embed, 0.0, 0.02)
        nn.init.normal_(model.horizon_query, 0.0, 0.02)
        for blk in model.blocks:
            nn.init.zeros_(blk.attn.per_dim_scale)
        # Scale residual output projections (GPT-2 style) and the readout (μP).
        res_scale = (2 * N_LAYERS) ** -0.5
        for blk in model.blocks:
            blk.attn.proj.weight.mul_(res_scale)
            blk.mlp[-1].weight.mul_(res_scale)
        model.out_head.weight.mul_(0.5)


def build_optimizer(model):
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim == 2 else no_decay).append(p)
    return NorMuon([
        dict(params=decay, use_muon=True, lr=1e-2, weight_decay=0.0),
        dict(params=no_decay, use_muon=False, lr=3e-4, weight_decay=0.01),
    ])


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
