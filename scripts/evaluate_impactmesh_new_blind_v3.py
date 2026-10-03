"""Evaluate frozen v3 checkpoints on the independent source-locked val panel."""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import torch

from flood_reliability.impactmesh_io import TrainArchiveReader
from flood_reliability.overlap import unique_pixel_mask
from scripts.calibrate_impactmesh_formal import load_model
from scripts.evaluate_impactmesh_formal import ScoreAccumulator, block_valid
from scripts.train_impactmesh_exploratory_v2 import infer
from scripts.train_impactmesh_exploratory_v3 import BASE, CONFIG, STATS, model_for
from scripts.train_impactmesh_formal import ROOT, encode_batch, sha256

MAN = ROOT / "data/manifests"
LOCKED = MAN / "impactmesh_new_blind_locked_tiles.csv"
META = MAN / "impactmesh_new_blind_locked_tiles.json"
GRID = MAN / "impactmesh_val_mask_grid_index.csv"
STATES = (("full", 100), ("half", 50), ("zero", 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", choices=("B4_v3", "B5_v3", "P_sigmoid_v3", "P_residual_v3",
                                              "B4_v1", "B5_v1"))
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    locked_meta = json.loads(META.read_text(encoding="utf-8"))
    if locked_meta["locked_manifest_sha256"] != sha256(LOCKED) or locked_meta["label_pixels_read"]:
        raise RuntimeError("New blind input-only lock missing or changed")
    choice_file = BASE / "candidate_selection_frozen.json"
    choice = json.loads(choice_file.read_text(encoding="utf-8"))
    if choice["config_sha256"] != sha256(CONFIG) or choice["selection_uses_viewed_validation"]:
        raise RuntimeError("v3 development choice not frozen before blind evaluation")
    for name in config["candidates"]:
        folder = BASE / name
        run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        temp = json.loads((folder / "temperature.json").read_text(encoding="utf-8"))
        if (run["selected_sha256"] != sha256(folder / "selected.pt")
                or temp["checkpoint_sha256"] != run["selected_sha256"]
                or temp["selection_frozen_sha256"] != sha256(choice_file)):
            raise RuntimeError(f"v3 candidate not fully frozen: {name}")
    folder = BASE / args.candidate
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "new_blind_paired.json"
    if output.exists():
        raise RuntimeError("Refusing to repeat new blind evaluation")
    with LOCKED.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups = defaultdict(list)
    for row in rows:
        if row["paired_eligible"] != "1":
            raise RuntimeError("Locked manifest contains ineligible tile")
        groups[row["activation"]].append(row)
    if set(groups) != set(locked_meta["locked_events"]):
        raise RuntimeError("Locked event identities changed")
    with GRID.open(newline="", encoding="utf-8") as handle:
        grids = {row["sample_id"]: row for row in csv.DictReader(handle) if row["split"] == "val"}
    if any(row["sample_id"] not in grids for row in rows):
        raise RuntimeError("Locked tile grid missing")
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    if args.candidate.endswith("_v1"):
        name = args.candidate.split("_")[0]
        model_config_path = ROOT / "configs/impactmesh_formal_v1.json"
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        stats_path = ROOT / "outputs/formal_v1/train_only_statistics.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        model, checkpoint = load_model(name, device, stats, model_config)
        temperature_path = ROOT / "outputs/formal_v1" / name / "temperature.json"
    else:
        model_config_path = CONFIG
        stats_path = STATS
        stats = json.loads(STATS.read_text(encoding="utf-8"))
        checkpoint = BASE / args.candidate / "selected.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = model_for(args.candidate, config, device).eval()
        model.load_state_dict(saved["model_state"])
        temperature_path = BASE / args.candidate / "temperature.json"
    temperature_record = json.loads(temperature_path.read_text(encoding="utf-8"))
    if temperature_record["checkpoint_sha256"] != sha256(checkpoint):
        raise RuntimeError("Reference/model temperature checkpoint mismatch")
    temperature = float(temperature_record["temperature"])
    results = {}
    with TrainArchiveReader(ROOT / "data/raw/impactmesh/val") as reader:
        for process_index, (process, process_rows) in enumerate(sorted(groups.items()), 1):
            process_rows.sort(key=lambda row: row["sample_id"])
            accumulators = {state: ScoreAccumulator() for state, _ in STATES}
            preceding = []
            true_flood_pixels = 0
            for start in range(0, len(process_rows), config["batch_size"]):
                batch = process_rows[start:start + config["batch_size"]]
                sar, optical, valid, target, offset = encode_batch(reader, batch, stats, device)
                original = valid.cpu().numpy()[:, 0]
                labels = target.cpu().numpy()
                owners = []
                for i, row in enumerate(batch):
                    owner = unique_pixel_mask(grids[row["sample_id"]], preceding)
                    preceding.append(grids[row["sample_id"]])
                    owners.append(owner)
                    true_flood_pixels += int(((labels[i] == 2) & owner).sum())
                for state, level in STATES:
                    kept = np.stack([block_valid(original[i], row["sample_id"], level, config["seed"])
                                     for i, row in enumerate(batch)])
                    kept_t = torch.from_numpy(kept[:, None]).to(device)
                    logits = infer(model, sar, optical, kept_t, offset)
                    for i in range(len(batch)):
                        accumulators[state].add(labels[i], logits[i].copy(), owners[i], temperature,
                                                collect_ap=True)
            results[process] = {"tiles": len(process_rows),
                                "true_flood_pixels_owned": true_flood_pixels,
                                "scores": {state: accumulator.finish()
                                           for state, accumulator in accumulators.items()}}
            print(f"{args.candidate} new blind {process_index}/{len(groups)} {process}: "
                  f"{len(process_rows)} tiles", flush=True)
    output.write_text(json.dumps({"candidate": args.candidate,
        "purpose": "post-v1-test exploratory independent-process blind panel",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": sha256(checkpoint), "temperature_sha256": sha256(temperature_path),
        "stats_sha256": sha256(stats_path), "config_sha256": sha256(model_config_path),
        "mask_protocol_config_sha256": sha256(CONFIG),
        "historical_v1_reference": args.candidate.endswith("_v1"),
        "locked_manifest_sha256": sha256(LOCKED), "locked_meta_sha256": sha256(META),
        "selection_frozen_sha256": sha256(choice_file), "grid_sha256": sha256(GRID),
        "states": [state for state, _ in STATES], "results": results},
        indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
