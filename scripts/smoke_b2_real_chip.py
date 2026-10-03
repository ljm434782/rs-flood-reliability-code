"""One diagnostic B2 optimizer step on a fixed ImpactMesh train-only chip.

This is a compute smoke test, not model fitting or a scientific result.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import rasterio
from rasterio.crs import CRS
import torch
import zarr

from flood_reliability import models as model_module
from flood_reliability.models import SmallUNet


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ID = "EMSR424_1_38LQH_x786235_y8282875"  # Fixed train audit frame rank 4.
MANIFEST = ROOT / "data" / "manifests" / "impactmesh_sample_manifest.csv"
OUTPUT = ROOT / "data" / "manifests" / "b2_real_chip_smoke.json"


def verified_path(rows: list[dict[str, str]], modality: str) -> tuple[Path, str]:
    matching = [row for row in rows if row["sample_id"] == SAMPLE_ID and row["split"] == "train" and row["modality"] == modality]
    if len(matching) != 1:
        raise RuntimeError(f"Expected exactly one train {modality} manifest row")
    row = matching[0]
    path = ROOT / row["local_path"]
    if path.stat().st_size != int(row["size_bytes"]) or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise RuntimeError(f"Source size or SHA256 mismatch: {path}")
    return path, row["sha256"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    args = parser.parse_args()
    if args.device == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("XPU unavailable")
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    s1_path, s1_sha = verified_path(rows, "S1RTC")
    label_path, label_sha = verified_path(rows, "MASK")
    group = zarr.open_consolidated(s1_path, mode="r")
    s1_crs = CRS.from_wkt(group["spatial_ref"].attrs["crs_wkt"])
    source_s1_crs = group.attrs["proj_code"].split(";")[2]
    sar = np.asarray(group["bands"][2], dtype=np.float32)
    if sar.shape != (2, 256, 256) or not np.isfinite(sar).all() or np.any(group["nan_mask"][2] != 0):
        raise RuntimeError("Selected event S1 chip is not a complete finite 2-band raster")
    if int(group["data_mask"][2]) != 1:
        raise RuntimeError("Selected S1 acquisition is marked unavailable")
    with rasterio.open(label_path) as source:
        label = source.read(1).astype(np.int64)
        if s1_crs != source.crs:
            raise RuntimeError(f"S1 and label CRS differ: {s1_crs} vs {source.crs}")
        if source.shape != sar.shape[-2:]:
            raise RuntimeError("S1 and label shapes differ")
        expected_x = source.transform.c + (np.arange(source.width) + 0.5) * source.transform.a
        expected_y = source.transform.f + (np.arange(source.height) + 0.5) * source.transform.e
        if not (np.allclose(group["x"][:], expected_x, atol=1e-6, rtol=0) and np.allclose(group["y"][:], expected_y, atol=1e-6, rtol=0)):
            raise RuntimeError("S1 and label grids differ")
    if not set(np.unique(label)) <= {-1, 0, 1, 2}:
        raise RuntimeError("Unexpected label codes")
    ignored = int(np.count_nonzero(label == -1))
    label[label == -1] = 255
    torch.manual_seed(0)
    device = torch.device(args.device)
    model = SmallUNet(2, 3, base_width=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = torch.from_numpy(sar.copy()).unsqueeze(0).to(device)
    target = torch.from_numpy(label.copy()).unsqueeze(0).to(device)
    before = model.head.weight.detach().clone()
    if args.device == "xpu":
        torch.xpu.reset_peak_memory_stats()
        torch.xpu.synchronize()
    start = perf_counter()
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, target, ignore_index=255)
    loss.backward()
    optimizer.step()
    if args.device == "xpu":
        torch.xpu.synchronize()
    elapsed = perf_counter() - start
    finite = bool(torch.isfinite(loss).item())
    updated = bool(torch.any(before != model.head.weight.detach()).item())
    if not finite or not updated or tuple(logits.shape) != (1, 3, 256, 256):
        raise RuntimeError("B2 smoke failed: output, finite loss, or parameter update")
    result = {
        "purpose": "single-step compute smoke only; not formal training or reported segmentation performance",
        "sample_id": SAMPLE_ID,
        "sample_split": "train",
        "s1_sha256": s1_sha,
        "label_sha256": label_sha,
        "s1_and_label_grid_crs": str(s1_crs),
        "s1_event_source_product_crs": source_s1_crs,
        "model_source_sha256": hashlib.sha256(Path(model_module.__file__).read_bytes()).hexdigest(),
        "smoke_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": args.device,
        "device_name": torch.xpu.get_device_name(0) if args.device == "xpu" else "CPU",
        "torch_version": torch.__version__,
        "model": "SmallUNet(2,3,base_width=16); architecture not frozen",
        "random_seed": 0,
        "optimizer": "Adam(lr=0.001); one update only",
        "loss": "cross_entropy(ignore_index=255); original -1 mapped to 255",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "shape": list(logits.shape),
        "ignored_label_pixels": ignored,
        "finite_loss": finite,
        "parameter_updated": updated,
        "one_step_seconds": elapsed,
        "peak_allocated_bytes": int(torch.xpu.max_memory_allocated()) if args.device == "xpu" else None,
        "input_handling": "raw finite train S1 event-index-2 values; no normalization rule selected",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
