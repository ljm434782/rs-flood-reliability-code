"""Check a minimal forward, backward, and optimizer step on a chosen device."""

import argparse
import json

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    args = parser.parse_args()

    xpu_available = bool(torch.xpu.is_available())
    if args.device == "xpu" and not xpu_available:
        raise RuntimeError("PyTorch reports that XPU is unavailable")

    device = torch.device(args.device)
    torch.manual_seed(0)
    layer = torch.nn.Linear(4, 2).to(device)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
    inputs = torch.arange(32, dtype=torch.float32, device=device).reshape(8, 4) / 32
    targets = torch.zeros((8, 2), dtype=torch.float32, device=device)
    before = layer.weight.detach().clone()
    loss = torch.nn.functional.mse_loss(layer(inputs), targets)
    loss.backward()
    optimizer.step()
    if args.device == "xpu":
        torch.xpu.synchronize()

    changed = bool(torch.any(before != layer.weight.detach()).item())
    finite = bool(torch.isfinite(loss).item())
    if not (changed and finite):
        raise RuntimeError("Non-finite loss or no parameter update")

    print(
        json.dumps(
            {
                "torch_version": torch.__version__,
                "requested_device": args.device,
                "xpu_available": xpu_available,
                "xpu_device_name": torch.xpu.get_device_name(0) if xpu_available else None,
                "loss": float(loss.detach().cpu().item()),
                "finite_loss": finite,
                "parameters_updated": changed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
