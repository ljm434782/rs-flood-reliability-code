"""Run a train-only, process-disjoint B2 development fit on the fixed 24-chip audit frame.

This is an engineering run. Its split, sample size and configuration are not the
formal preregistered study and its metrics must not appear as final results.
"""

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
import torch
import zarr

from flood_reliability.evaluation_counts import confusion_counts
from flood_reliability.models import SmallUNet
from flood_reliability.splits import require_process_disjoint


ROOT = Path(__file__).resolve().parents[1]
FRAME = ROOT / "data" / "manifests" / "impactmesh_temporal_audit_frame.csv"
MANIFEST = ROOT / "data" / "manifests" / "impactmesh_sample_manifest.csv"
OUT = ROOT / "outputs" / "development_b2_24_gridfix"
CHECKPOINT = ROOT / "checkpoints" / "development_b2_24_gridfix.pt"


class CRSGridMismatch(RuntimeError):
    """A source event raster cannot be used with the label as supplied."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_path(records: dict[tuple[str, str], dict], sample_id: str, modality: str) -> Path:
    record = records[(sample_id, modality)]
    if record["split"] != "train":
        raise RuntimeError(f"Development input outside official train: {sample_id}")
    path = ROOT / record["local_path"]
    if path.stat().st_size != int(record["size_bytes"]) or sha256(path) != record["sha256"]:
        raise RuntimeError(f"Source size or SHA256 mismatch: {path}")
    return path


def load_chip(row: dict, records: dict) -> tuple[np.ndarray, np.ndarray]:
    sample_id = row["sample_id"]
    s1_path = verified_path(records, sample_id, "S1RTC")
    label_path = verified_path(records, sample_id, "MASK")
    group = zarr.open_consolidated(s1_path, mode="r")
    sar = np.asarray(group["bands"][2], dtype=np.float32)
    if sar.shape != (2, 256, 256) or not np.isfinite(sar).all():
        raise RuntimeError(f"Invalid event S1 array: {sample_id}")
    if int(group["data_mask"][2]) != 1 or np.any(group["nan_mask"][2] != 0):
        raise RuntimeError(f"Unavailable event S1: {sample_id}")
    with rasterio.open(label_path) as source:
        label = source.read(1).astype(np.int64)
        grid_crs = CRS.from_wkt(group["spatial_ref"].attrs["crs_wkt"])
        if source.crs != grid_crs:
            raise CRSGridMismatch(f"S1 grid CRS {grid_crs} differs from label CRS {source.crs}: {sample_id}")
        if source.shape != (256, 256):
            raise RuntimeError(f"S1/label shape mismatch: {sample_id}")
        expected_x = source.transform.c + (np.arange(256) + 0.5) * source.transform.a
        expected_y = source.transform.f + (np.arange(256) + 0.5) * source.transform.e
        if not (np.allclose(group["x"][:], expected_x, atol=1e-6, rtol=0)
                and np.allclose(group["y"][:], expected_y, atol=1e-6, rtol=0)):
            raise CRSGridMismatch(f"S1/label coordinate mismatch: {sample_id}")
    if not set(np.unique(label)) <= {-1, 0, 1, 2}:
        raise RuntimeError(f"Unexpected label codes: {sample_id}")
    label[label == -1] = 255
    if not np.any(label != 255):
        raise RuntimeError(f"No supervised pixels: {sample_id}")
    return sar, label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    args = parser.parse_args()
    if args.epochs < 1 or (args.device == "xpu" and not torch.xpu.is_available()):
        raise RuntimeError("Invalid epoch count or XPU unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_source = ROOT / "src" / "flood_reliability" / "models.py"
    model_source_sha = sha256(model_source)
    with FRAME.open(newline="", encoding="utf-8") as handle:
        frame = list(csv.DictReader(handle))
    if len(frame) != 24 or len({row["activation"] for row in frame}) != 24:
        raise RuntimeError("Expected the fixed 24-process train audit frame")
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    records = {(row["sample_id"], row["modality"]): row for row in manifest_rows}
    chips = {}
    excluded = []
    for row in frame:
        try:
            chips[row["sample_id"]] = load_chip(row, records)
        except CRSGridMismatch as error:
            excluded.append({"sample_id": row["sample_id"], "activation": row["activation"],
                             "reason": str(error)})
    retained = [row for row in frame if row["sample_id"] in chips]
    train_rows = [row for row in retained if int(row["rank"]) % 5 != 0]
    val_rows = [row for row in retained if int(row["rank"]) % 5 == 0]
    if not train_rows or not val_rows:
        raise RuntimeError("No development training or validation samples after grid checks")
    require_process_disjoint({"development_train": [row["activation"] for row in train_rows],
                              "development_val": [row["activation"] for row in val_rows]})
    train_stack = np.stack([chips[row["sample_id"]][0] for row in train_rows])
    mean = train_stack.mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = train_stack.std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    if np.any(std <= 0) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError("Invalid train-only normalization statistics")
    device = torch.device(args.device)
    model = SmallUNet(2, 3, base_width=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, foreach=False)
    prepared = {sample_id: (torch.from_numpy(((sar - mean[:, None, None]) / std[:, None, None]).copy()).unsqueeze(0),
                            torch.from_numpy(label.copy()).unsqueeze(0))
                for sample_id, (sar, label) in chips.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    split = {"train": [row["sample_id"] for row in train_rows],
             "val": [row["sample_id"] for row in val_rows]}
    (OUT / "split.json").write_text(json.dumps({
        "purpose": "development only, official train source; not preregistered or paper results",
        "rule": "fixed frame rank divisible by 5 -> development validation; all others -> development train",
        "grid_exclusions": excluded,
        "frame_sha256": sha256(FRAME), "manifest_sha256": sha256(MANIFEST),
        "process_key": "activation", "samples": split,
    }, indent=2) + "\n", encoding="utf-8")
    log = []
    for epoch in range(1, args.epochs + 1):
        order = train_rows.copy()
        random.shuffle(order)
        model.train()
        train_loss_sum = 0.0
        for step, row in enumerate(order, start=1):
            inputs, target = prepared[row["sample_id"]]
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs.to(device))
            loss = torch.nn.functional.cross_entropy(logits, target.to(device), ignore_index=255)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss: {row['sample_id']}")
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu())
            if step % 5 == 0:
                print(f"epoch {epoch}: {step}/{len(order)} training chips", flush=True)
        model.eval()
        val_loss_sum = 0.0
        val_confusion = np.zeros((3, 3), dtype=np.int64)
        with torch.inference_mode():
            for row in val_rows:
                inputs, target = prepared[row["sample_id"]]
                logits = model(inputs.to(device))
                loss = torch.nn.functional.cross_entropy(logits, target.to(device), ignore_index=255)
                val_loss_sum += float(loss.cpu())
                prediction = logits.argmax(dim=1).squeeze(0).cpu().numpy()
                val_confusion += confusion_counts(target.squeeze(0).numpy(), prediction, num_classes=3)
        entry = {"epoch": epoch, "train_mean_chip_loss": train_loss_sum / len(train_rows),
                 "development_val_mean_chip_loss": val_loss_sum / len(val_rows),
                 "development_val_confusion": val_confusion.tolist()}
        log.append(entry)
        print(json.dumps(entry), flush=True)
    if sha256(model_source) != model_source_sha:
        raise RuntimeError("Model source changed during the run; refusing provenance record")
    torch.save({"model_state": model.cpu().state_dict(), "optimizer_state": optimizer.state_dict(),
                "epoch": args.epochs, "normalization_mean": mean.tolist(),
                "normalization_std": std.tolist(), "split": split, "seed": args.seed,
                "purpose": "development-only; no formal test or paper result"}, CHECKPOINT)
    (OUT / "run.json").write_text(json.dumps({
        "purpose": "development-only B2 fit on 24 fixed official-train samples; no official validation/test/holdout used",
        "started_source": "fixed train audit frame", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "config": vars(args), "torch_version": torch.__version__,
        "device_name": torch.xpu.get_device_name(0) if args.device == "xpu" else "CPU",
        "model": "SmallUNet(2,3,base_width=16)", "optimizer": "Adam(lr=1e-3,foreach=False)",
        "loss": "cross_entropy(ignore_index=255)",
        "normalization_source": "development_train only", "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(), "frame_sha256": sha256(FRAME),
        "manifest_sha256": sha256(MANIFEST), "script_sha256": sha256(Path(__file__)),
        "grid_exclusions": excluded,
        "model_source_sha256": model_source_sha,
        "checkpoint": CHECKPOINT.relative_to(ROOT).as_posix(), "epochs": log,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Development fit saved: {OUT / 'run.json'}", flush=True)


if __name__ == "__main__":
    main()
