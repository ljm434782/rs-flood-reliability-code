"""Paired process audit of selected exploratory checkpoints on validation only."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from flood_reliability.impactmesh_io import TrainArchiveReader
from scripts.train_impactmesh_exploratory_v2 import (
    BASE, CONFIG, ROOT, STATS, check_inputs, fixed_frame, make_model, sha256,
    validate,
)


COMPARE = {
    "P_no_sar_gate": "P_no_sar_gate",
    "P_gate_both": "P_gate_both_restart1",
    "B5_drop050": "B5_drop050",
    "B4_bestval": "B4_bestval",
}
PAIRS = (("P_no_sar_gate", "P_gate_both"),
         ("P_no_sar_gate", "B5_drop050"),
         ("P_no_sar_gate", "B4_bestval"))


def paired_interval(left, right, metric, seed=20260928, repetitions=10000):
    if set(left) != set(right):
        raise RuntimeError("Validation process identities differ")
    ids = sorted(left)
    values = np.asarray([np.nan if left[key][metric] is None or right[key][metric] is None
                         else left[key][metric] - right[key][metric] for key in ids])
    if not np.isfinite(values[~np.isnan(values)]).all():
        raise RuntimeError("Nonfinite paired validation difference")
    defined = values[~np.isnan(values)]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(repetitions, len(ids)))
    sampled = values[draws]
    counts = (~np.isnan(sampled)).sum(axis=1)
    bootstrap = np.nansum(sampled, axis=1)[counts > 0] / counts[counts > 0]
    return {"mean_left_minus_right": float(np.mean(defined)),
            "ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "processes": len(ids), "defined_processes": len(defined)}


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    stats = check_inputs(config)
    _, frame, grids, frame_path = fixed_frame(config)
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    evaluated = {}
    hashes = {}
    with TrainArchiveReader(ROOT / "data" / "raw" / "impactmesh" / "train") as reader:
        for candidate, folder_name in COMPARE.items():
            folder = BASE / folder_name
            run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
            checkpoint = folder / "selected.pt"
            if run["selected_sha256"] != sha256(checkpoint):
                raise RuntimeError(f"Selected checkpoint changed: {candidate}")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = make_model(candidate, config, device)
            model.load_state_dict(payload["model_state"])
            evaluated[candidate] = validate(model, reader, frame, grids, stats, config, device)
            hashes[candidate] = sha256(checkpoint)
            print(f"Evaluated {candidate} selected epoch {payload['epoch']} on validation", flush=True)
    comparisons = {}
    for left, right in PAIRS:
        comparisons[f"{left}_minus_{right}"] = {}
        for state in config["selection_states"]:
            comparisons[f"{left}_minus_{right}"][state] = {
                metric: paired_interval(evaluated[left]["processes"][state],
                                        evaluated[right]["processes"][state], metric)
                for metric in ("nll", "flood_iou")}
    output = BASE / "selected_validation_paired_comparisons.json"
    output.write_text(json.dumps({
        "scope": "validation only; descriptive paired process bootstrap; no multiplicity correction",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "frame_sha256": sha256(frame_path), "stats_sha256": sha256(STATS),
        "checkpoint_sha256": hashes, "comparisons": comparisons,
        "validation_summaries": {name: value["summary"] for name, value in evaluated.items()},
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparisons, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
