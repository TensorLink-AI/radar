"""Frozen eval. Loads checkpoint, runs validation independently."""

import importlib.util
import os
import random
import sys

import torch
from safetensors.torch import load_file
from prepare import validate, CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES

SEED = int(os.environ.get("SEED", 42))
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

spec = importlib.util.spec_from_file_location("sub", "/workspace/submission.py")
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)

model = sub.build_model(CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES).cuda()
ckpt = "/workspace/checkpoints/model.safetensors"
if os.path.exists(ckpt):
    model.load_state_dict(load_file(ckpt, device="cuda"))
    if hasattr(model, "reset"):
        model.reset()
    model.eval()
    m = validate(model)
    print(f"crps: {m['crps']:.6f}")
    print(f"ncrps: {m['ncrps']:.6f}")
    print(f"mase: {m['mase']:.6f}")
    print(f"peak_vram_mb: {torch.cuda.max_memory_allocated() / 1e6:.1f}")
else:
    print("WARNING: no checkpoint found", file=sys.stderr)
