"""Audit five selected v2 checkpoints against the same validation-only frame."""

import csv
import json
from pathlib import Path

import torch

from scripts.train_impactmesh_exploratory_v2 import BASE, CONFIG, STATS, sha256


OUTPUTS = {
    "B4_bestval": "B4_bestval",
    "B5_drop050": "B5_drop050",
    "B5_drop025": "B5_drop025",
    "P_gate_both": "P_gate_both_restart1",
    "P_no_sar_gate": "P_no_sar_gate",
}


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frame = BASE / "validation_frame.json"
    if not frame.exists():
        raise RuntimeError("Frozen validation frame missing")
    rows = []
    for candidate, folder_name in OUTPUTS.items():
        folder = BASE / folder_name
        run_path = folder / "run.json"
        if not run_path.exists():
            raise RuntimeError(f"Candidate incomplete: {folder_name}")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if (run["candidate"] != candidate or run["config_sha256"] != sha256(CONFIG)
                or run["frame_sha256"] != sha256(frame)
                or run["stats_sha256"] != sha256(STATS)
                or run["selected_sha256"] != sha256(folder / "selected.pt")
                or run["no_test_labels_or_metrics_used_for_selection"] is not True):
            raise RuntimeError(f"Candidate provenance mismatch: {folder_name}")
        selected = torch.load(folder / "selected.pt", map_location="cpu", weights_only=False)
        if selected["candidate"] != candidate or selected["epoch"] != run["selected_epoch"]:
            raise RuntimeError(f"Selected checkpoint mismatch: {folder_name}")
        epochs = [json.loads(line) for line in (folder / "epochs.jsonl").read_text(
            encoding="utf-8").splitlines()]
        if [entry["epoch"] for entry in epochs] != list(range(1, config["epochs"] + 1)):
            raise RuntimeError(f"Incomplete epoch log: {folder_name}")
        choice = epochs[run["selected_epoch"] - 1]
        if (choice["validation_selection_score"] != min(
                entry["validation_selection_score"] for entry in epochs)
                or abs(choice["validation_selection_score"] - run["selected_score"]) > 1e-12):
            raise RuntimeError(f"Selection rule mismatch: {folder_name}")
        row = {"candidate": candidate, "attempt_folder": folder_name,
               "selected_epoch": run["selected_epoch"],
               "selection_score": run["selected_score"]}
        for state in config["selection_states"]:
            state_record = choice["validation"][state]
            row[f"{state}_processes"] = state_record["processes"]
            row[f"{state}_nll"] = state_record["nll"]
            row[f"{state}_flood_iou"] = state_record["flood_iou"]
            row[f"{state}_flood_iou_defined_processes"] = state_record[
                "flood_iou_defined_processes"]
        rows.append(row)
    path = BASE / "selected_validation_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
