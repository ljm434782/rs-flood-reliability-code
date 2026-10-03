"""Fit exactly one process-weighted global temperature per frozen model."""

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from flood_reliability.impactmesh_io import TrainArchiveReader
from flood_reliability.models import DualUNet, SmallUNet
from flood_reliability.overlap import unique_pixel_mask
from scripts.train_impactmesh_formal import encode_batch


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
SPLIT = MANIFESTS / "impactmesh_process_split_grouped.csv"
ELIGIBILITY = MANIFESTS / "impactmesh_all_train_frozen_eligibility.csv.gz"
GRID = MANIFESTS / "impactmesh_train_mask_grid_index.csv"
EXCLUSIONS = MANIFESTS / "impactmesh_evaluation_process_exclusions.csv"
CONFIG = ROOT / "configs" / "impactmesh_formal_v1.json"
STATS = ROOT / "outputs" / "formal_v1" / "train_only_statistics.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def role_rows(role: str):
    with SPLIT.open(newline="", encoding="utf-8") as handle:
        activations = {r["activation"]: r for r in csv.DictReader(handle)}
    with EXCLUSIONS.open(newline="", encoding="utf-8") as handle:
        excluded = {r["activation"]: r for r in csv.DictReader(handle)}
    if any(activations[a]["role"] != record["original_role"] for a, record in excluded.items()):
        raise RuntimeError("Evaluation exclusion no longer matches frozen process roles")
    with GRID.open(newline="", encoding="utf-8") as handle:
        grids = {r["sample_id"]: r for r in csv.DictReader(handle) if r["split"] == "train"}
    groups = defaultdict(list)
    with gzip.open(ELIGIBILITY, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            group = activations[row["activation"]]
            if (row["paired_eligible"] == "1" and group["role"] == role
                    and row["activation"] not in excluded):
                groups[group["process_id"]].append(row)
    if role in ("calibration", "internal_test") and len(groups) != 19:
        raise RuntimeError(f"Expected 19 independent {role} processes after event exclusions")
    for rows in groups.values():
        rows.sort(key=lambda row: row["sample_id"])
    return dict(sorted(groups.items())), grids


def load_model(model_name: str, device: torch.device, stats: dict, config: dict):
    checkpoint = ROOT / "outputs" / "formal_v1" / model_name / "latest.pt"
    run = ROOT / "outputs" / "formal_v1" / model_name / "run.json"
    if not run.exists():
        raise RuntimeError(f"Formal training incomplete: {model_name}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (saved["model"] != model_name or saved["epoch"] != config["epochs"]
            or saved["split_sha256"] != sha256(SPLIT)
            or saved["eligibility_sha256"] != sha256(ELIGIBILITY)
            or saved["config_sha256"] != sha256(CONFIG)
            or saved["stats_sha256"] != sha256(STATS)):
        raise RuntimeError("Checkpoint is not the frozen final-epoch model")
    model = (SmallUNet(2, 3, config["base_width"]) if model_name == "B2" else
             DualUNet(2, 12, 3, config["base_width"], gated=model_name == "P"))
    model.load_state_dict(saved["model_state"])
    model.to(device).eval()
    return model, checkpoint


def logits_for(model, model_name, batch, reader, stats, device):
    sar, optical, valid, label, offset = encode_batch(reader, batch, stats, device)
    present = torch.ones(len(batch), dtype=torch.bool, device=device)
    with torch.inference_mode():
        logits = (model(sar) if model_name == "B2" else
                  model(sar, optical, valid, present, normalized_time_offset=offset))
    return logits.cpu().numpy(), label.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("B2", "B4", "B5", "P"))
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    if args.device == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("XPU unavailable")
    device = torch.device(args.device)
    model, checkpoint = load_model(args.model, device, stats, config)
    groups, grids = role_rows("calibration")
    margins = {}
    counts = {}
    with TrainArchiveReader(ROOT / "data" / "raw" / "impactmesh" / "train") as reader:
        for process_index, (process, rows) in enumerate(groups.items(), 1):
            parts = []
            preceding = []
            for start in range(0, len(rows), config["batch_size"]):
                batch = rows[start:start + config["batch_size"]]
                logits, labels = logits_for(model, args.model, batch, reader, stats, device)
                for i, row in enumerate(batch):
                    owner = unique_pixel_mask(grids[row["sample_id"]], preceding)
                    preceding.append(grids[row["sample_id"]])
                    selected = (labels[i] != 255) & owner
                    if not selected.any():
                        continue
                    truth = labels[i, selected]
                    score = logits[i, :, selected]
                    if score.shape != (len(truth), 3):
                        raise RuntimeError("Unexpected class-logit arrangement")
                    true_score = score[np.arange(len(truth)), truth]
                    other1 = score[np.arange(len(truth)), (truth + 1) % 3] - true_score
                    other2 = score[np.arange(len(truth)), (truth + 2) % 3] - true_score
                    parts.append(np.stack((other1, other2), axis=1).astype(np.float32))
            if not parts:
                raise RuntimeError(f"No owned pixels for calibration process {process}")
            margins[process] = np.concatenate(parts)
            counts[process] = int(len(margins[process]))
            print(f"{args.model} calibration logits: process {process_index}/{len(groups)}, "
                  f"{process}, {counts[process]} pixels", flush=True)

    def objective(log_temperature):
        temperature = np.exp(log_temperature)
        values = []
        for process in sorted(margins):
            difference = margins[process] / temperature
            nll = np.logaddexp(np.logaddexp(0, difference[:, 0]), difference[:, 1])
            values.append(float(np.mean(nll, dtype=np.float64)))
        return float(np.mean(values))

    optimization = minimize_scalar(objective, bounds=(-3., 3.), method="bounded",
                                   options={"xatol": 1e-3, "maxiter": 100})
    if not optimization.success or not np.isfinite(optimization.fun):
        raise RuntimeError("Global temperature optimization failed")
    result = {
        "model": args.model, "temperature": float(np.exp(optimization.x)),
        "calibration_process_macro_nll_before": objective(0.),
        "calibration_process_macro_nll_after": float(optimization.fun),
        "calibration_process_pixel_counts": counts,
        "objective": "equal-process-mean native-label pixel NLL at 100% original optical availability",
        "optimizer": "scipy bounded scalar minimization of log(T) in [-3,3], xatol=0.001",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": sha256(checkpoint), "stats_sha256": sha256(STATS),
        "split_sha256": sha256(SPLIT), "eligibility_sha256": sha256(ELIGIBILITY),
        "evaluation_exclusions_sha256": sha256(EXCLUSIONS),
        "config_sha256": sha256(CONFIG), "script_sha256": sha256(Path(__file__)),
    }
    output = ROOT / "outputs" / "formal_v1" / args.model / "temperature.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": args.model, "temperature": result["temperature"],
                      "nll_before": result["calibration_process_macro_nll_before"],
                      "nll_after": result["calibration_process_macro_nll_after"]}), flush=True)


if __name__ == "__main__":
    main()
