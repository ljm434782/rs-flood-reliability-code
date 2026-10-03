"""One synthetic, no-update XPU resource measurement for the SMAGNet adapter."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from flood_reliability.smagnet_adapter import make_smagnet_resnet50, split_s1_s2, two_head_loss, validate_batch


ROOT = Path(__file__).resolve().parents[1]
AUTHOR_SOURCE = ROOT / "third_party" / "smagnet" / "smagnet.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--with-optimizer-step", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not torch.xpu.is_available():
        raise RuntimeError("This resource measurement requires an available XPU")
    device = torch.device("xpu")
    torch.manual_seed(29)
    x = torch.randn(args.batch_size, 6, 256, 256, device=device)
    invalid = torch.zeros(args.batch_size, 1, 256, 256, device=device)
    invalid[:, :, :64, :64] = 1
    label = torch.zeros(args.batch_size, 256, 256, dtype=torch.long, device=device)
    label[:, 64:128, :128] = 1
    label[:, 128:192, 64:192] = 2
    label[:, :64, :64] = 255
    validate_batch(x, invalid, label)
    model = make_smagnet_resnet50(AUTHOR_SOURCE).to(device).train()
    parameter_count = sum(p.numel() for p in model.parameters())
    weights = torch.tensor([1.0, 1.2, 2.0], device=device)
    optimizer = (torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False)
                 if args.with_optimizer_step else None)
    torch.xpu.synchronize()
    torch.xpu.reset_peak_memory_stats(device)
    started = time.perf_counter()
    fused, sar, _ = model(*split_s1_s2(x), invalid)
    loss, parts = two_head_loss(fused, sar, label, weights)
    loss.backward()
    if optimizer is not None:
        optimizer.step()
    torch.xpu.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "synthetic_only": True,
        "optimizer_created": optimizer is not None,
        "optimizer_step_called": optimizer is not None,
        "weights_or_checkpoints_written": False,
        "device": torch.xpu.get_device_name(0),
        "torch": torch.__version__,
        "batch_shape": list(x.shape),
        "invalid_mask_shape": list(invalid.shape),
        "logit_shape": list(fused.shape),
        "parameter_count": parameter_count,
        "elapsed_seconds_forward_backward": elapsed,
        "throughput_samples_per_second": args.batch_size / elapsed,
        "peak_memory_allocated_bytes": torch.xpu.max_memory_allocated(device),
        "loss": float(loss.detach().cpu()),
        "fused_loss": float(parts["fused_loss"].detach().cpu()),
        "sar_loss": float(parts["sar_loss"].detach().cpu()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
