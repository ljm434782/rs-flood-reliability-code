"""Synthetic CUDA/CPU execution and checkpoint check; never opens project data."""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scripts.upgrade_train import make_model, _train_epoch, _save_checkpoint


def run(device_name: str) -> dict:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(device_name)
    torch.set_num_threads(2)
    random.seed(20261013); np.random.seed(20261013); torch.manual_seed(20261013)
    if device_name == "cuda":
        torch.cuda.manual_seed_all(20261013)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    config = json.loads((ROOT / "configs/upgrade_20260929.json").read_text(encoding="utf-8"))
    config["_active_seed"] = 20261013
    config["_active_epoch"] = 1
    batch = (torch.randn(4, 2, 256, 256, device=device),
             torch.randn(4, 12, 256, 256, device=device),
             torch.ones(4, 1, 256, 256, dtype=torch.bool, device=device),
             torch.randint(0, 3, (4, 256, 256), device=device),
             torch.zeros(4, device=device))
    batch[3][:, 0, 0] = 255
    rows = [f"synthetic-{i}" for i in range(4)]
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("B2", "B5_whole", "B5_local", "P"):
            model = make_model(name, ROOT, config).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, foreach=False)
            result = _train_epoch(model, name, [(batch, rows)], optimizer,
                                  {"class_weights": [1.0, 2.0, 3.0]}, config, device)
            path = Path(tmp) / f"{name}.pt"
            _save_checkpoint(path, epoch=1, model=model, optimizer=optimizer,
                             history=[result], hashes={"synthetic": True}, seed=20261013, run_id=name)
            expected = torch.rand(8, device=device)
            state = torch.load(path, map_location="cpu", weights_only=False)
            restored = make_model(name, ROOT, config).to(device)
            restored.load_state_dict(state["model_state"])
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.001, foreach=False)
            restored_optimizer.load_state_dict(state["optimizer_state"])
            torch.set_rng_state(state["torch_rng"])
            if device_name == "cuda":
                if state.get("cuda_rng") is None:
                    raise RuntimeError("Missing CUDA RNG state")
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            if not torch.equal(expected, torch.rand(8, device=device)):
                raise RuntimeError("RNG replay mismatch")
            if not all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items()):
                raise RuntimeError("Checkpoint parameter mismatch")
            result["resume_step"] = _train_epoch(restored, name, [(batch, rows)], restored_optimizer,
                {"class_weights": [1.0, 2.0, 3.0]}, config, device)["mean_batch_loss"]
            results[name] = result
    return {"status": "PASS", "device": device_name, "torch": str(torch.__version__),
            "device_name": torch.cuda.get_device_name(0) if device_name == "cuda" else "CPU",
            "precision": "FP32; no AMP; TF32 disabled on CUDA", "batch_size": 4,
            "data_access": "none; synthetic tensors only", "models": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.device)
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded, flush=True)
