"""Fit one temperature for validation-selected v2 model on calibration only."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from flood_reliability.impactmesh_io import TrainArchiveReader
from flood_reliability.overlap import unique_pixel_mask
from scripts.calibrate_impactmesh_formal import (
    ELIGIBILITY, EXCLUSIONS, GRID, ROOT, SPLIT, STATS, logits_for, role_rows, sha256,
)
from scripts.train_impactmesh_exploratory_v2 import BASE, CONFIG, make_model


SELECTED = "P_no_sar_gate"


def main():
    comparison = BASE / "selected_validation_comparison.csv"
    with comparison.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5 or min(rows, key=lambda row: float(row["selection_score"]))["candidate"] != SELECTED:
        raise RuntimeError("The predefined validation rule did not select this candidate")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    folder = BASE / SELECTED
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    checkpoint = folder / "selected.pt"
    if run["selected_sha256"] != sha256(checkpoint):
        raise RuntimeError("Selected v2 checkpoint changed")
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    model = make_model(SELECTED, config, device).eval()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state"])
    groups, grids = role_rows("calibration")
    margins = {}
    counts = {}
    with TrainArchiveReader(ROOT / "data" / "raw" / "impactmesh" / "train") as reader:
        for index, (process, rows) in enumerate(groups.items(), 1):
            parts = []
            preceding = []
            for start in range(0, len(rows), config["batch_size"]):
                batch = rows[start:start + config["batch_size"]]
                logits, labels = logits_for(model, "P", batch, reader, stats, device)
                for i, row in enumerate(batch):
                    owner = unique_pixel_mask(grids[row["sample_id"]], preceding)
                    preceding.append(grids[row["sample_id"]])
                    selected = (labels[i] != 255) & owner
                    if not selected.any():
                        continue
                    truth = labels[i, selected]
                    score = logits[i, :, selected]
                    true_score = score[np.arange(len(truth)), truth]
                    other1 = score[np.arange(len(truth)), (truth + 1) % 3] - true_score
                    other2 = score[np.arange(len(truth)), (truth + 2) % 3] - true_score
                    parts.append(np.stack((other1, other2), axis=1).astype(np.float32))
            if not parts:
                raise RuntimeError(f"Calibration process has no usable pixels: {process}")
            margins[process] = np.concatenate(parts)
            counts[process] = len(margins[process])
            print(f"v2 calibration process {index}/{len(groups)} {process}: {counts[process]} pixels", flush=True)

    def objective(log_temperature):
        temperature = np.exp(log_temperature)
        return float(np.mean([np.mean(np.logaddexp(
            np.logaddexp(0, values[:, 0] / temperature),
            values[:, 1] / temperature), dtype=np.float64)
            for values in margins.values()]))

    fit = minimize_scalar(objective, bounds=(-3., 3.), method="bounded",
                          options={"xatol": 1e-3, "maxiter": 100})
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError("v2 temperature optimization failed")
    result = {"candidate": SELECTED, "temperature": float(np.exp(fit.x)),
              "calibration_process_macro_nll_before": objective(0.),
              "calibration_process_macro_nll_after": float(fit.fun),
              "calibration_process_pixel_counts": counts,
              "purpose": "post-v1-test exploratory; only final temperature fitted on calibration",
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "checkpoint_sha256": sha256(checkpoint), "comparison_sha256": sha256(comparison),
              "config_sha256": sha256(CONFIG), "stats_sha256": sha256(STATS),
              "split_sha256": sha256(SPLIT), "eligibility_sha256": sha256(ELIGIBILITY),
              "grid_sha256": sha256(GRID), "exclusions_sha256": sha256(EXCLUSIONS),
              "script_sha256": sha256(Path(__file__))}
    output = folder / "temperature.json"
    if output.exists():
        raise RuntimeError("Refusing to overwrite exploratory calibration")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"temperature": result["temperature"],
                      "nll_before": result["calibration_process_macro_nll_before"],
                      "nll_after": result["calibration_process_macro_nll_after"]}), flush=True)


if __name__ == "__main__":
    main()
