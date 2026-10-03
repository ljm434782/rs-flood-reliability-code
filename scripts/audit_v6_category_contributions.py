"""Read-only classwise probability contribution diagnostic for v6 development data."""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flood_reliability.impactmesh_io import TrainArchiveReader
from flood_reliability.overlap import unique_pixel_mask
from scripts.evaluate_impactmesh_formal import block_valid
from scripts.summarize_impactmesh_robustness_v6 import infer, load_grids, model_for, sha256
from scripts.train_impactmesh_formal import encode_batch

CONFIG = ROOT / "configs/impactmesh_robustness_v6.json"
PLAN = ROOT / "docs/V6_CATEGORY_DIAGNOSTIC_PLAN.md"
BASE = ROOT / "outputs/robustness_v6"
FRAME = BASE / "development_frame.json"
STATS = BASE / "fit_only_statistics.json"
SUMMARY = BASE / "v6_summary.json"
FREEZE = BASE / "freeze_manifest.json"
GRID = ROOT / "data/manifests/impactmesh_train_mask_grid_index.csv"
READER = ROOT / "data/raw/impactmesh/train"
STATES = ("full", "half", "zero", "real_missing")
CLASSES = (0, 1, 2)


def class_sums(labels, logits, owners, temperature):
    totals = {c: {"pixels": 0, "nll_sum": 0.0, "brier_sum": 0.0} for c in CLASSES}
    for target, score, owner in zip(labels, logits, owners):
        active = owner & (target != 255)
        shifted = score / temperature
        shifted -= shifted.max(axis=0, keepdims=True)
        probs = np.exp(shifted, dtype=np.float64)
        probs /= probs.sum(axis=0, keepdims=True)
        for c in CLASSES:
            selected = active & (target == c)
            if not selected.any():
                continue
            p = probs[:, selected]
            totals[c]["pixels"] += int(selected.sum())
            totals[c]["nll_sum"] += float(np.sum(-np.log(p[c])))
            onehot = p.copy()
            onehot[c] -= 1.0
            totals[c]["brier_sum"] += float(np.sum(onehot * onehot))
    for c in CLASSES:
        n = totals[c]["pixels"]
        totals[c]["nll_mean"] = None if not n else totals[c]["nll_sum"] / n
        totals[c]["brier_mean"] = None if not n else totals[c]["brier_sum"] / n
    return totals


def bootstrap(values, seed=20261011, repetitions=10000):
    values = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if not len(values):
        return {"mean": None, "ci95_low": None, "ci95_high": None, "defined_processes": 0}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(repetitions, len(values)))
    boot = values[draws].mean(axis=1)
    ci = np.quantile(boot, [0.025, 0.975])
    return {"mean": float(values.mean()), "ci95_low": float(ci[0]), "ci95_high": float(ci[1]),
            "defined_processes": int(len(values))}


def main(device_name="xpu"):
    if device_name == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("XPU unavailable")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frame = json.loads(FRAME.read_text(encoding="utf-8"))
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze["config_sha256"] != sha256(CONFIG) or freeze["v6_frame_sha256"] != sha256(FRAME):
        raise RuntimeError("Frozen v6 config/frame hash mismatch")
    grids = load_grids()
    device = torch.device(device_name)
    state_rows = {state: frame["paired"] for state in ("full", "half", "zero")}
    state_rows["real_missing"] = frame["real_missing"]
    process_details = []

    for candidate in config["candidates"]:
        for seed in config["seeds"]:
            checkpoint = BASE / candidate / f"seed_{seed}" / "selected.pt"
            run = BASE / candidate / f"seed_{seed}" / "run.json"
            if not checkpoint.exists() or not run.exists():
                raise RuntimeError(f"Missing selected v6 checkpoint: {candidate} seed {seed}")
            temp = float(summary["models"][candidate][str(seed)]["temperature"])
            model = model_for(candidate, config, device)
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(payload["model_state"])
            model.eval()
            with TrainArchiveReader(READER) as reader:
                for state in STATES:
                    for process, rows in state_rows[state].items():
                        labels_all, logits_all, owners = [], [], []
                        preceding = []
                        level = {"full": 100, "half": 50, "zero": 0}.get(state)
                        if level is None:
                            means = np.asarray(stats["sar_mean"], dtype=np.float32)[:, None, None]
                            stds = np.asarray(stats["sar_std"], dtype=np.float32)[:, None, None]
                            sars, targets = [], []
                            for row in rows:
                                s1, target = reader.load_s1_label(row["sample_id"])
                                sars.append(((s1 - means) / stds).astype(np.float32)); targets.append(target)
                            sar = torch.from_numpy(np.stack(sars)).to(device)
                            optical = torch.zeros((len(rows), 12, 256, 256), device=device)
                            valid = torch.zeros((len(rows), 1, 256, 256), dtype=torch.bool, device=device)
                            offset = torch.zeros(len(rows), device=device)
                            logits_batch = infer(model, sar, optical, valid, offset, device)
                            for i, row in enumerate(rows):
                                labels_all.append(targets[i]); logits_all.append(logits_batch[i])
                                owners.append(unique_pixel_mask(grids[row["sample_id"]], preceding)); preceding.append(grids[row["sample_id"]])
                        else:
                            for start in range(0, len(rows), config["batch_size"]):
                                batch = rows[start:start + config["batch_size"]]
                                sar, optical, valid, target, offset = encode_batch(reader, batch, stats, device)
                                original = valid.cpu().numpy()[:, 0]
                                kept = np.stack([block_valid(original[i], row["sample_id"], level, config["development_mask_seed"])
                                                 for i, row in enumerate(batch)])
                                logits_batch = infer(model, sar, optical,
                                                     torch.from_numpy(kept[:, None]).to(device), offset, device)
                                targets = target.cpu().numpy()
                                for i, row in enumerate(batch):
                                    labels_all.append(targets[i]); logits_all.append(logits_batch[i])
                                    owners.append(unique_pixel_mask(grids[row["sample_id"]], preceding)); preceding.append(grids[row["sample_id"]])
                        raw = class_sums(labels_all, logits_all, owners, 1.0)
                        scaled = class_sums(labels_all, logits_all, owners, temp)
                        process_details.append({"candidate": candidate, "seed": seed, "state": state,
                                                "process": process, "raw": raw, "scaled": scaled})

    summary_rows = []
    for candidate in config["candidates"]:
        for seed in config["seeds"]:
            for state in STATES:
                subset = [r for r in process_details if r["candidate"] == candidate and r["seed"] == seed and r["state"] == state]
                for c in CLASSES:
                    total_pixels = sum(r["raw"][c]["pixels"] for r in subset)
                    for metric in ("nll", "brier"):
                        for scale in ("raw", "scaled"):
                            total = sum(r[scale][c][f"{metric}_sum"] for r in subset)
                            defined = [r[scale][c][f"{metric}_mean"] for r in subset if r[scale][c][f"{metric}_mean"] is not None]
                            summary_rows.append({"candidate": candidate, "seed": seed, "state": state, "class": c,
                                                 "metric": metric, "scale": scale, "pixels": total_pixels,
                                                 "sum": total, "mean_of_process_means": float(np.mean(defined)) if defined else None,
                                                 "defined_processes": len(defined)})

    interactions = []
    for candidate in config["candidates"]:
        for seed in config["seeds"]:
            for c in CLASSES:
                for metric in ("nll", "brier"):
                    effects = []
                    for process in sorted({r["process"] for r in process_details if r["candidate"] == candidate and r["seed"] == seed and r["state"] == "full"}):
                        get = lambda state, scale: next(r for r in process_details if r["candidate"] == candidate and r["seed"] == seed and r["state"] == state and r["process"] == process)[scale][c][f"{metric}_mean"]
                        vals = [get("full", "raw"), get("full", "scaled"), get("zero", "raw"), get("zero", "scaled")]
                        if all(v is not None for v in vals):
                            effects.append((vals[3] - vals[2]) - (vals[1] - vals[0]))
                    result = bootstrap(effects)
                    result.update({"candidate": candidate, "seed": seed, "class": c, "metric": metric,
                                   "positive_processes": int(sum(x > 0 for x in effects)), "process_count": len(effects),
                                   "total_processes": len(frame["paired"])})
                    interactions.append(result)

    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / "v6_category_contributions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["candidate", "seed", "state", "class", "metric", "scale", "pixels", "sum", "mean_of_process_means", "defined_processes"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(summary_rows)
    (BASE / "v6_category_interactions.json").write_text(json.dumps(interactions, indent=2) + "\n", encoding="utf-8")
    (BASE / "v6_category_process_details.json").write_text(json.dumps(process_details, indent=2) + "\n", encoding="utf-8")
    provenance = {"purpose": "frozen v6 classwise probability contribution diagnostic",
                  "completed_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": sha256(CONFIG),
                  "plan_sha256": sha256(PLAN), "frame_sha256": sha256(FRAME), "stats_sha256": sha256(STATS),
                  "grid_sha256": sha256(GRID), "freeze_manifest_sha256": sha256(FREEZE),
                  "v6_summary_sha256": sha256(SUMMARY), "script_sha256": sha256(Path(__file__)),
                  "summary_csv_sha256": sha256(BASE / "v6_category_contributions.csv"),
                  "interactions_json_sha256": sha256(BASE / "v6_category_interactions.json"),
                  "process_details_sha256": sha256(BASE / "v6_category_process_details.json"),
                  "contains_test_or_holdout_data": False}
    (BASE / "v6_category_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(summary_rows), "interactions": len(interactions), "provenance": str(BASE / "v6_category_provenance.json")}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    main(parser.parse_args().device)
