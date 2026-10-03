"""Freeze v3 dev choice, then evaluate selected checkpoints once on viewed validation20."""

import csv
import gzip
import json
from collections import defaultdict
from datetime import datetime, timezone

import torch

from flood_reliability.impactmesh_io import TrainArchiveReader
from scripts.calibrate_impactmesh_formal import GRID
from scripts.train_impactmesh_exploratory_v2 import REASONS, validate
from scripts.train_impactmesh_exploratory_v3 import (
    BASE, CONFIG, STATS, model_for, score_development,
)
from scripts.train_impactmesh_formal import (
    ELIGIBILITY, ROOT, SPLIT, hash_order, prepare_rows, sha256,
)

FROZEN = BASE / "candidate_selection_frozen.json"
FRAME = BASE / "viewed_validation_confirmation_frame.json"
OUT = BASE / "viewed_validation_confirmation.json"


def freeze_selection(config):
    ranked = []
    for candidate in config["candidates"]:
        path = BASE / candidate / "run.json"
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["candidate"] != candidate or run["selected_sha256"] != sha256(BASE / candidate / "selected.pt"):
            raise RuntimeError("Incomplete or changed v3 run")
        ranked.append({"candidate": candidate, "dev_score": run["selected_score"],
                       "epoch": run["selected_epoch"], "checkpoint_sha256": run["selected_sha256"]})
    ranked.sort(key=lambda row: (row["dev_score"], row["candidate"]))
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "choice": ranked[0]["candidate"], "ranking": ranked,
              "basis": "new 16-process training-only development fold",
              "config_sha256": sha256(CONFIG), "selection_uses_viewed_validation": False}
    if FROZEN.exists():
        old = json.loads(FROZEN.read_text(encoding="utf-8"))
        if old["choice"] != result["choice"] or old["ranking"] != ranked:
            raise RuntimeError("Frozen v3 selection changed")
        return old
    FROZEN.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def confirmation_frame(config):
    pool = prepare_rows(config)
    paired = {process: rows[:2] for process, rows in sorted(pool["validation"].items())}
    with SPLIT.open(newline="", encoding="utf-8") as handle:
        activations = {row["activation"]: row for row in csv.DictReader(handle)}
    missing = defaultdict(list)
    with gzip.open(ELIGIBILITY, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            info = activations[row["activation"]]
            if (info["role"] == "validation" and row["paired_eligible"] == "0"
                    and row["failure_reasons"] in REASONS):
                missing[info["process_id"]].append(row)
    missing = {process: hash_order(rows, config["sample_order_seed"] + "|v3_confirmation_missing")[:2]
               for process, rows in sorted(missing.items())}
    if len(paired) != 20 or len(missing) != 13:
        raise RuntimeError("Original validation20 frame changed")
    frame = {"paired": paired, "real_missing": missing,
             "selection_states": config["selection_states"], "seed": config["seed"],
             "split_sha256": sha256(SPLIT), "eligibility_sha256": sha256(ELIGIBILITY),
             "grid_sha256": sha256(GRID), "purpose": "viewed validation20 confirmation only"}
    serialized = json.dumps(frame, indent=2) + "\n"
    if FRAME.exists() and FRAME.read_text(encoding="utf-8") != serialized:
        raise RuntimeError("Confirmation frame changed")
    if not FRAME.exists():
        FRAME.write_text(serialized, encoding="utf-8")
    with GRID.open(newline="", encoding="utf-8") as handle:
        grids = {row["sample_id"]: row for row in csv.DictReader(handle) if row["split"] == "train"}
    if any(row["sample_id"] not in grids for rows in list(paired.values()) + list(missing.values())
           for row in rows):
        raise RuntimeError("Original validation grid index incomplete")
    return frame, grids


def main():
    if OUT.exists():
        raise RuntimeError("Refusing to repeat viewed validation confirmation")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frozen = freeze_selection(config)
    frame, grids = confirmation_frame(config)
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    results = {}
    with TrainArchiveReader(ROOT / "data/raw/impactmesh/train") as reader:
        for candidate in config["candidates"]:
            checkpoint = BASE / candidate / "selected.pt"
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            if payload["candidate"] != candidate or payload["frame_sha256"] != sha256(BASE / "development_frame.json"):
                raise RuntimeError("Unexpected selected v3 checkpoint")
            model = model_for(candidate, config, device)
            model.load_state_dict(payload["model_state"])
            evaluation = validate(model, reader, frame, grids, stats, config, device)
            results[candidate] = {"checkpoint_sha256": sha256(checkpoint),
                                  "epoch": payload["epoch"],
                                  "descriptive_score": score_development(evaluation, config),
                                  "evaluation": evaluation}
            print(candidate, results[candidate]["descriptive_score"], evaluation["summary"], flush=True)
    OUT.write_text(json.dumps({"purpose": "one-time viewed validation20 development confirmation",
        "not_new_blind_test": True, "no_selection_or_tuning_on_these_results": True,
        "selection_frozen_sha256": sha256(FROZEN), "frame_sha256": sha256(FRAME),
        "stats_sha256": sha256(STATS), "chosen_on_dev16": frozen["choice"],
        "results": results}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
