"""Freeze two-seed v5 choice, then audit paired development processes and stop rule."""

import json
from datetime import datetime, timezone

import numpy as np
import torch

from flood_reliability.impactmesh_io import TrainArchiveReader
from scripts.train_impactmesh_exploratory_v2 import validate
from scripts.train_impactmesh_exploratory_v4 import score_development
from scripts.train_impactmesh_exploratory_v5 import (
    BASE, CONFIG, FRAME, STATS, model_for, prepare,
)
from scripts.train_impactmesh_formal import ROOT, sha256

FROZEN = BASE / "candidate_selection_frozen.json"
OUT = BASE / "selected_development_detail.json"


def resampled_score(processes, ids, states):
    nll, iou = [], []
    for state in states:
        values = [processes[state][process] for process in ids if process in processes[state]]
        if not values:
            return None
        defined = [row["flood_iou"] for row in values if row["flood_iou"] is not None]
        if not defined:
            return None
        nll.append(float(np.mean([row["nll"] for row in values])))
        iou.append(float(np.mean(defined)))
    return float(np.mean(nll) - np.mean(iou))


def freeze_selection(config):
    ranked = []
    runs = {}
    for candidate in config["candidates"]:
        by_seed = {}
        for seed in config["seeds"]:
            directory = BASE / candidate / f"seed_{seed}"
            run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
            if (run["candidate"] != candidate or run["seed"] != seed
                    or run["config_sha256"] != sha256(CONFIG)
                    or run["selected_sha256"] != sha256(directory / "selected.pt")):
                raise RuntimeError("Incomplete or changed v5 run")
            by_seed[str(seed)] = {"score": run["selected_score"],
                                  "epoch": run["selected_epoch"],
                                  "checkpoint_sha256": run["selected_sha256"],
                                  "parameter_count": run["parameter_count"]}
        runs[candidate] = by_seed
        ranked.append({"candidate": candidate,
                       "mean_score": float(np.mean([v["score"] for v in by_seed.values()])),
                       "seeds": by_seed})
    ranked.sort(key=lambda row: (row["mean_score"], row["candidate"]))
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "choice": ranked[0]["candidate"], "ranking": ranked,
              "basis": "new training-only 16-process v5 fold; mean of two fixed seed scores",
              "config_sha256": sha256(CONFIG), "selection_uses_viewed_data": False}
    if FROZEN.exists():
        old = json.loads(FROZEN.read_text(encoding="utf-8"))
        if old["ranking"] != ranked or old["choice"] != result["choice"]:
            raise RuntimeError("Frozen v5 choice changed")
        return old
    FROZEN.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    if OUT.exists():
        raise RuntimeError("Refusing to overwrite v5 development audit")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frozen = freeze_selection(config)
    _, frame, grids = prepare(config)
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    evaluation_config = {**config, "seed": config["seeds"][0]}
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    evaluations = {}
    with TrainArchiveReader(ROOT / "data/raw/impactmesh/train") as reader:
        for candidate in config["candidates"]:
            evaluations[candidate] = {}
            for seed in config["seeds"]:
                checkpoint = BASE / candidate / f"seed_{seed}" / "selected.pt"
                payload = torch.load(checkpoint, map_location=device, weights_only=False)
                if (payload["candidate"] != candidate or payload["seed"] != seed
                        or payload["frame_sha256"] != sha256(FRAME)
                        or payload["stats_sha256"] != sha256(STATS)):
                    raise RuntimeError("Selected v5 checkpoint source mismatch")
                model = model_for(candidate, config, device)
                model.load_state_dict(payload["model_state"])
                evaluation = validate(model, reader, frame, grids, stats, evaluation_config, device)
                expected = frozen_seed_score(frozen, candidate, seed)
                actual = score_development(evaluation, config)
                if not np.isclose(actual, expected, rtol=0, atol=1e-7):
                    raise RuntimeError(f"V5 score mismatch: {candidate}, {seed}")
                evaluations[candidate][str(seed)] = evaluation
                print(candidate, seed, actual, evaluation["summary"], flush=True)
    ids = sorted(frame["paired"])
    states = config["selection_states"]
    rng = np.random.default_rng(config["seeds"][0] + 900)
    comparisons = {}
    for control in ("B4_v5", "B5_v5"):
        differences = []
        for _ in range(10000):
            sampled = rng.choice(ids, len(ids), replace=True)
            seed_differences = []
            for seed in config["seeds"]:
                a = resampled_score(evaluations["F_aux_v5"][str(seed)]["processes"], sampled, states)
                b = resampled_score(evaluations[control][str(seed)]["processes"], sampled, states)
                if a is None or b is None:
                    break
                seed_differences.append(a - b)
            if len(seed_differences) == len(config["seeds"]):
                differences.append(float(np.mean(seed_differences)))
        means = [row for row in frozen["ranking"] if row["candidate"] in ("F_aux_v5", control)]
        score_for = {row["candidate"]: row["mean_score"] for row in means}
        comparisons[control] = {"mean_difference": score_for["F_aux_v5"] - score_for[control],
                                "bootstrap_95_percentile": np.quantile(differences, [.025, .975]).tolist(),
                                "valid_replicates": len(differences),
                                "both_seeds_better": all(
                                    frozen_seed_score(frozen, "F_aux_v5", seed)
                                    < frozen_seed_score(frozen, control, seed)
                                    for seed in config["seeds"])}
    b5 = evaluations["B5_v5"]
    fusion = evaluations["F_aux_v5"]
    missing_iou_noninferior = {}
    for state in ("zero", "real_missing"):
        f_iou = np.mean([fusion[str(seed)]["summary"][state]["flood_iou"] for seed in config["seeds"]])
        b_iou = np.mean([b5[str(seed)]["summary"][state]["flood_iou"] for seed in config["seeds"]])
        missing_iou_noninferior[state] = {"fusion_mean": float(f_iou), "b5_mean": float(b_iou),
                                          "passes": bool(f_iou >= b_iou)}
    promising = (all(row["mean_difference"] < 0 and row["both_seeds_better"]
                     and row["bootstrap_95_percentile"][1] < 0 for row in comparisons.values())
                 and all(row["passes"] for row in missing_iou_noninferior.values()))
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "basis": "training-only v5 development, selection-conditioned exploratory intervals",
              "selection_frozen_sha256": sha256(FROZEN), "choice": frozen["choice"],
              "config_sha256": sha256(CONFIG), "frame_sha256": sha256(FRAME),
              "stats_sha256": sha256(STATS), "evaluations": evaluations,
              "fusion_minus_controls": comparisons,
              "missing_flood_iou_checks": missing_iou_noninferior,
              "promising_internal_signal": bool(promising),
              "stop_p_or_f_superiority_search_if_false": True}
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("frozen", frozen["choice"], "promising", promising,
          "comparisons", comparisons, "state checks", missing_iou_noninferior, flush=True)


def frozen_seed_score(frozen, candidate, seed):
    return next(row for row in frozen["ranking"] if row["candidate"] == candidate)["seeds"][str(seed)]["score"]


if __name__ == "__main__":
    main()
