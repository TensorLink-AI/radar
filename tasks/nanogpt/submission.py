# Experiment: seed_baseline
"""
Minimal nanoGPT training script (seed).

Trains a small transformer language model on the prepared dataset.
The GEPA-Research agent will evolve this file to improve val_loss.

Requirements:
  - Run prepare.py first to generate data/train.bin and data/val.bin
  - PyTorch must be installed

Metric output format (DO NOT REMOVE — the runner extracts these):
  val_loss: <float>
  train_loss: <float>
  iter/s: <float>
  training_seconds: <float>
  peak_vram_mb: <float>
"""

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Configuration ─────────────────────────────────────────────────────

# Model
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
BLOCK_SIZE = 256
DROPOUT = 0.2

# Training
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-1
GRAD_CLIP = 1.0
WARMUP_ITERS = 100

# Budget
TIME_BUDGET = 300  # seconds — the runner enforces this

# Evaluation
EVAL_INTERVAL = 50
EVAL_ITERS = 20

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ── Data Loading ──────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_tokens(split: str) -> np.ndarray:
    """Load pre-tokenized binary data from prepare.py output."""
    path = os.path.join(DATA_DIR, f"{split}.bin")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run 'python prepare.py' first.", file=sys.stderr)
        sys.exit(1)
    data = np.fromfile(path, dtype=np.uint16)
    return data


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: str):
    """Sample a random batch of sequences."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def get_vocab_size() -> int:
    """Read vocab size from metadata, or use GPT-2 default."""
    meta_path = os.path.join(DATA_DIR, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return meta["vocab_size"]
    return 50257  # GPT-2 default


# ── Model ─────────────────────────────────────────────────────────────


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying
        self.tok_emb.weight = self.head.weight
        self.apply(self._init_weights)
        # Scale residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params:,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} > block_size {self.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ── Training ──────────────────────────────────────────────────────────


@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device):
    """Estimate train and val loss over eval_iters batches."""
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            with torch.amp.autocast(device_type=device, dtype=DTYPE):
                _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def main():
    print(f"Device: {DEVICE}")
    print(f"Config: n_layer={N_LAYER}, n_head={N_HEAD}, n_embd={N_EMBD}, block_size={BLOCK_SIZE}")
    print(f"Training: batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, time_budget={TIME_BUDGET}s")

    # Load data
    print("Loading data...")
    train_data = load_tokens("train")
    val_data = load_tokens("val")
    vocab_size = get_vocab_size()
    print(f"Vocab size: {vocab_size}, Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")

    # Build model
    print("Building model...")
    model = GPT(vocab_size, N_LAYER, N_HEAD, N_EMBD, BLOCK_SIZE, DROPOUT).to(DEVICE)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

    # Training loop
    print("Starting training loop...")
    start_time = time.time()
    best_val_loss = float("inf")
    step = 0
    train_loss_accum = 0.0
    log_interval = 10

    while True:
        elapsed = time.time() - start_time
        if elapsed >= TIME_BUDGET:
            break

        # Learning rate warmup then cosine decay
        if step < WARMUP_ITERS:
            lr = LEARNING_RATE * (step + 1) / WARMUP_ITERS
        else:
            progress = (elapsed / TIME_BUDGET)
            lr = LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward + backward
        x, y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
        with torch.amp.autocast(device_type=DEVICE, dtype=DTYPE):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        train_loss_accum += loss.item()
        step += 1

        # Logging
        if step % log_interval == 0:
            avg_loss = train_loss_accum / log_interval
            train_loss_accum = 0.0
            elapsed = time.time() - start_time
            print(f"step {step:>5d} | loss: {avg_loss:.4f} | lr: {lr:.2e} | time: {elapsed:.1f}s")

        # Evaluation
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(
                model, train_data, val_data, EVAL_ITERS, BATCH_SIZE, BLOCK_SIZE, DEVICE
            )
            elapsed = time.time() - start_time
            print(f"step {step:>5d} | train_loss: {losses['train']:.4f} | val_loss: {losses['val']:.4f} | time: {elapsed:.1f}s")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

    # Final evaluation
    total_time = time.time() - start_time
    losses = estimate_loss(model, train_data, val_data, EVAL_ITERS, BATCH_SIZE, BLOCK_SIZE, DEVICE)
    if losses["val"] < best_val_loss:
        best_val_loss = losses["val"]

    iter_per_sec = step / total_time if total_time > 0 else 0

    # Peak memory
    peak_mem = 0.0
    if DEVICE == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    # ── Report results (DO NOT REMOVE — the runner extracts these) ────
    print(f"\nval_loss: {best_val_loss:.6f}")
    print(f"train_loss: {losses['train']:.6f}")
    print(f"iter/s: {iter_per_sec:.1f}")
    print(f"training_seconds: {total_time:.1f}")
    print(f"peak_vram_mb: {peak_mem:.1f}")
    print(f"total_steps: {step}")
    print(f"best_val_loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
