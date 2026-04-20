"""Phase C eval template for ts_forecasting.

Held in a zero-dep module so the validator can `importlib.import_module`
it without pulling torch/prepare/flops. The trainer-side `train.py`
re-exports EVAL_TEMPLATE from here.
"""

from __future__ import annotations

EVAL_TEMPLATE = '''
import json
import os
import random
import sys

import torch
from safetensors.torch import load_file

from prepare import validate, CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES
from flops import compute_flops_equivalent

random.seed({eval_split_seed})
torch.manual_seed({eval_split_seed})
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

arch_path = "{arch_path}"
checkpoint_path = "{checkpoint_path}"
device = "{device}"

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("submission", arch_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "build_model") or not callable(mod.build_model):
        print(json.dumps({{"crps": float("inf"), "mase": float("inf"), "error": "Missing build_model()"}}))
        sys.exit(0)

    model = mod.build_model(CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES).to(device)
    state_dict = load_file(checkpoint_path, device=device)
    model.load_state_dict(state_dict)

    flops_equiv = 0
    try:
        flops_equiv = compute_flops_equivalent(model, CONTEXT_LEN, NUM_VARIATES, device)
    except Exception:
        pass

    param_count = sum(p.numel() for p in model.parameters())
    if hasattr(model, "reset"):
        model.reset()
    model.eval()

    data_dir = os.environ.get("RADAR_GIFT_EVAL_CACHE", "")
    metrics = validate(model, seed={eval_split_seed},
                       data_dir=data_dir if data_dir else None)

    result = {{
        "crps": metrics["crps"],
        "ncrps": metrics.get("ncrps", float("inf")),
        "mase": metrics["mase"],
        "flops_equivalent_size": flops_equiv,
        "param_count": param_count,
    }}
    if "n_datasets" in metrics:
        result["n_datasets"] = metrics["n_datasets"]
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({{"crps": float("inf"), "mase": float("inf"), "error": str(e)}}))
'''
