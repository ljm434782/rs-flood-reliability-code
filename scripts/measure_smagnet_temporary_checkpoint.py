"""Measure a synthetic full SMAGNet checkpoint, then delete it in-process."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from flood_reliability.smagnet_adapter import make_smagnet_resnet50, split_s1_s2, two_head_loss, validate_batch


ROOT = Path(__file__).resolve().parents[1]
AUTHOR_SOURCE = ROOT / "third_party" / "smagnet" / "smagnet.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-directory", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--payload", choices=("full", "model-only"), default="full")
    args = parser.parse_args()
    target_name = ("TEMP_DELETE_AFTER_FULL_CHECKPOINT_SIZE.pt" if args.payload == "full"
                   else "TEMP_DELETE_AFTER_MODEL_ONLY_CHECKPOINT_SIZE.pt")
    target = args.audit_directory / target_name
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite temporary audit target {target}")
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    torch.manual_seed(31)
    model = make_smagnet_resnet50(AUTHOR_SOURCE).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False)
    x = torch.randn(1, 6, 256, 256, device=device)
    invalid = torch.zeros(1, 1, 256, 256, device=device)
    label = torch.zeros(1, 256, 256, dtype=torch.long, device=device)
    label[:, 64:128, :128] = 1
    label[:, 128:192, 64:192] = 2
    label[:, :64, :64] = 255
    validate_batch(x, invalid, label)
    fused, sar, _ = model(*split_s1_s2(x), invalid)
    loss, _ = two_head_loss(fused, sar, label, torch.tensor([1.0, 1.2, 2.0], device=device))
    loss.backward()
    optimizer.step()
    args.audit_directory.mkdir(parents=True, exist_ok=True)
    try:
        payload = {"model_state": model.state_dict(),
                   "purpose": "synthetic temporary checkpoint size audit only"}
        if args.payload == "full":
            payload["optimizer_state"] = optimizer.state_dict()
        torch.save(payload, target)
        payload = {"checkpoint_target": str(target), "bytes": target.stat().st_size,
                   "sha256_before_delete": digest(target), "synthetic_only": True,
                   "optimizer_step_called": True, "device": str(device),
                   "payload": args.payload}
    finally:
        if target.exists():
            target.unlink()
    payload["checkpoint_deleted"] = not target.exists()
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
