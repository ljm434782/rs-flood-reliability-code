"""Freeze v4 selection and audit chosen checkpoints on the training-only dev fold."""

import json
from datetime import datetime, timezone

import numpy as np
import torch

from flood_reliability.impactmesh_io import TrainArchiveReader
from scripts.confirm_impactmesh_v4_validation import freeze_selection
from scripts.train_impactmesh_exploratory_v2 import validate
from scripts.train_impactmesh_exploratory_v4 import (
    BASE, CONFIG, FRAME, STATS, model_for, prepare,
)
from scripts.train_impactmesh_formal import ROOT, sha256

OUT = BASE / "selected_development_detail.json"


def resampled_score(processes, ids, states):
    nll, iou = [], []
    for state in states:
        values = [processes[state][process] for process in ids
                  if process in processes[state]]
        if not values:
            return None
        state_iou = [v["flood_iou"] for v in values if v["flood_iou"] is not None]
        if not state_iou:
            return None
        nll.append(float(np.mean([v["nll"] for v in values])))
        iou.append(float(np.mean(state_iou)))
    return float(np.mean(nll) - np.mean(iou))


def main():
    if OUT.exists():
        raise RuntimeError("Refusing to overwrite v4 selected-development audit")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frozen = freeze_selection(config)
    _, frame, grids = prepare(config)
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    evaluations = {}
    with TrainArchiveReader(ROOT / "data/raw/impactmesh/train") as reader:
        for candidate in config["candidates"]:
            checkpoint = BASE / candidate / "selected.pt"
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            if (payload["candidate"] != candidate or payload["frame_sha256"] != sha256(FRAME)
                    or payload["stats_sha256"] != sha256(STATS)):
                raise RuntimeError("Selected v4 checkpoint source mismatch")
            model = model_for(candidate, config, device)
            model.load_state_dict(payload["model_state"])
            evaluations[candidate] = validate(model, reader, frame, grids, stats, config, device)
            print(candidate, evaluations[candidate]["summary"], flush=True)
    ids = sorted(frame["paired"])
    states = config["selection_states"]
    for candidate in config["candidates"]:
        actual = resampled_score(evaluations[candidate]["processes"], ids, states)
        saved = json.loads((BASE / candidate / "run.json").read_text(encoding="utf-8"))["selected_score"]
        if not np.isclose(actual, saved, atol=1e-7, rtol=0):
            raise RuntimeError(f"Selected dev score mismatch for {candidate}: {actual}, {saved}")
    rng = np.random.default_rng(config["seed"] + 900)
    pairwise = {}
    for candidate in ("R_noaux_v4", "R_aux_v4"):
        for control in ("B4_v4", "B5_v4"):
            differences = []
            for _ in range(10000):
                sampled = rng.choice(ids, size=len(ids), replace=True)
                a = resampled_score(evaluations[candidate]["processes"], sampled, states)
                b = resampled_score(evaluations[control]["processes"], sampled, states)
                if a is not None and b is not None:
                    differences.append(a - b)
            key = f"{candidate}_minus_{control}"
            pairwise[key] = {"observed": resampled_score(evaluations[candidate]["processes"], ids, states)
                             - resampled_score(evaluations[control]["processes"], ids, states),
                             "bootstrap_95_percentile": np.quantile(differences, [.025, .975]).tolist(),
                             "valid_replicates": len(differences)}
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "basis": "training-only 16-process v4 development; no viewed test used",
              "choice": frozen["choice"], "selection_frozen_sha256": sha256(BASE / "candidate_selection_frozen.json"),
              "config_sha256": sha256(CONFIG), "frame_sha256": sha256(FRAME),
              "stats_sha256": sha256(STATS), "evaluations": evaluations,
              "paired_process_bootstrap": pairwise}
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("frozen choice", frozen["choice"], "pairwise", pairwise, flush=True)


if __name__ == "__main__":
    main()
