"""One global temperature per v3 candidate on the same calibration19 frame."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from flood_reliability.impactmesh_io import TrainArchiveReader
from flood_reliability.overlap import unique_pixel_mask
from scripts.calibrate_impactmesh_formal import (
    ELIGIBILITY, EXCLUSIONS, GRID, SPLIT, logits_for, role_rows,
)
from scripts.confirm_impactmesh_v3_validation import freeze_selection
from scripts.train_impactmesh_exploratory_v3 import BASE, CONFIG, STATS, model_for
from scripts.train_impactmesh_formal import ROOT, hash_order, sha256

FRAME = BASE / "calibration19_two_tiles_per_process.json"


def calibration_frame(config):
    groups, grids = role_rows("calibration")
    selected = {process: hash_order(rows, config["sample_order_seed"] + "|v3_calibration")[:2]
                for process, rows in groups.items()}
    if len(selected) != 19 or any(len(rows) != 2 for rows in selected.values()):
        raise RuntimeError("Calibration19 fixed frame incomplete")
    frame = {"groups": selected, "purpose": "one global temperature per v3 model only",
             "split_sha256": sha256(SPLIT), "eligibility_sha256": sha256(ELIGIBILITY),
             "grid_sha256": sha256(GRID), "exclusions_sha256": sha256(EXCLUSIONS),
             "config_sha256": sha256(CONFIG)}
    serialized = json.dumps(frame, indent=2) + "\n"
    if FRAME.exists() and FRAME.read_text(encoding="utf-8") != serialized:
        raise RuntimeError("v3 calibration frame changed")
    if not FRAME.exists():
        FRAME.write_text(serialized, encoding="utf-8")
    return selected, grids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", choices=("B4_v3", "B5_v3", "P_sigmoid_v3", "P_residual_v3"))
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frozen = freeze_selection(config)
    groups, grids = calibration_frame(config)
    output = BASE / args.candidate / "temperature.json"
    if output.exists():
        raise RuntimeError("Refusing to overwrite v3 temperature")
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    checkpoint = BASE / args.candidate / "selected.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["candidate"] != args.candidate or payload["stats_sha256"] != sha256(STATS):
        raise RuntimeError("v3 checkpoint/stats mismatch")
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    model = model_for(args.candidate, config, device).eval()
    model.load_state_dict(payload["model_state"])
    margins, counts = {}, {}
    with TrainArchiveReader(ROOT / "data/raw/impactmesh/train") as reader:
        for index, (process, rows) in enumerate(groups.items(), 1):
            parts, preceding = [], []
            logits, labels = logits_for(model, args.candidate, rows, reader, stats, device)
            for i, row in enumerate(rows):
                owner = unique_pixel_mask(grids[row["sample_id"]], preceding)
                preceding.append(grids[row["sample_id"]])
                selected = (labels[i] != 255) & owner
                if not selected.any():
                    continue
                truth = labels[i, selected]
                score = logits[i, :, selected]
                true_score = score[np.arange(len(truth)), truth]
                parts.append(np.stack((score[np.arange(len(truth)), (truth + 1) % 3] - true_score,
                                       score[np.arange(len(truth)), (truth + 2) % 3] - true_score),
                                      axis=1).astype(np.float32))
            if not parts:
                raise RuntimeError(f"No usable calibration pixels for {process}")
            margins[process] = np.concatenate(parts)
            counts[process] = len(margins[process])
            print(f"{args.candidate} calibration {index}/19 {process}: {counts[process]}", flush=True)

    def objective(log_temperature):
        temperature = np.exp(log_temperature)
        return float(np.mean([np.mean(np.logaddexp(
            np.logaddexp(0, values[:, 0] / temperature), values[:, 1] / temperature),
            dtype=np.float64) for values in margins.values()]))

    fit = minimize_scalar(objective, bounds=(-3., 3.), method="bounded",
                          options={"xatol": 1e-3, "maxiter": 100})
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError("v3 temperature fit failed")
    result = {"candidate": args.candidate, "temperature": float(np.exp(fit.x)),
              "calibration_process_macro_nll_before": objective(0.),
              "calibration_process_macro_nll_after": float(fit.fun),
              "calibration_process_pixel_counts": counts,
              "purpose": "one global temperature; no candidate or threshold selection",
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "checkpoint_sha256": sha256(checkpoint), "selection_frozen_sha256": sha256(BASE / "candidate_selection_frozen.json"),
              "selection_choice": frozen["choice"], "frame_sha256": sha256(FRAME),
              "stats_sha256": sha256(STATS), "split_sha256": sha256(SPLIT),
              "eligibility_sha256": sha256(ELIGIBILITY), "grid_sha256": sha256(GRID),
              "exclusions_sha256": sha256(EXCLUSIONS), "script_sha256": sha256(Path(__file__))}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate": args.candidate, "temperature": result["temperature"],
                      "nll_before": result["calibration_process_macro_nll_before"],
                      "nll_after": result["calibration_process_macro_nll_after"]}), flush=True)


if __name__ == "__main__":
    main()
