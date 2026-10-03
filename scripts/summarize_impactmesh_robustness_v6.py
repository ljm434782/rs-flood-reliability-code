"""Evaluate the frozen v6 development supplement and summarize interactions."""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flood_reliability.impactmesh_io import TrainArchiveReader
from flood_reliability.models import DualUNet
from flood_reliability.overlap import unique_pixel_mask
from scripts.evaluate_impactmesh_formal import ScoreAccumulator, block_valid
from scripts.train_impactmesh_formal import encode_batch
from scripts import train_impactmesh_exploratory_v5 as v5

CONFIG = ROOT / "configs/impactmesh_robustness_v6.json"
SOURCE_CONFIG = ROOT / "configs/impactmesh_exploratory_v5.json"
BASE = ROOT / "outputs/robustness_v6"
FRAME = BASE / "development_frame.json"
STATS = BASE / "fit_only_statistics.json"
FREEZE = BASE / "freeze_manifest.json"
GRID = ROOT / "data/manifests/impactmesh_train_mask_grid_index.csv"
READER_PATH = ROOT / "data/raw/impactmesh/train"
STATES = ("full", "half", "zero", "real_missing")
METRICS = ("nll", "brier", "ece15", "flood_iou")


def sha256(path):
    return v5.sha256(Path(path))


def model_for(candidate, config, device):
    spec = config["candidates"][candidate]
    return DualUNet(2, 12, 3, config["base_width"],
                    gated=spec["architecture"] == "dual_unet_gated").to(device)


def load_grids():
    with GRID.open(newline="", encoding="utf-8") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle) if row["split"] == "train"}


def process_metric(labels, logits, owners, temperature):
    acc = ScoreAccumulator()
    for target, score, owner in zip(labels, logits, owners):
        acc.add(target, score, owner, temperature, collect_ap=False)
    result = acc.finish()
    return {key: result[key] for key in METRICS} | {"pixels": result["pixels"]}


def infer(model, sar, optical, valid, offset, device):
    present = valid.flatten(1).any(dim=1)
    safe_optical = torch.where(valid, optical, torch.zeros_like(optical))
    safe_offset = torch.where(present, offset, torch.zeros_like(offset))
    with torch.inference_mode():
        return model(sar, safe_optical, valid, present,
                     normalized_time_offset=safe_offset).cpu().numpy()


def evaluate_state(model, reader, rows_by_process, grids, stats, config, device, state, temperature):
    level = {"full": 100, "half": 50, "zero": 0}.get(state)
    records = {}
    for process, rows in rows_by_process.items():
        labels_all, logits_all, owners = [], [], []
        preceding = []
        if level is None:
            means = np.asarray(stats["sar_mean"], dtype=np.float32)[:, None, None]
            stds = np.asarray(stats["sar_std"], dtype=np.float32)[:, None, None]
            sars, labels = [], []
            for row in rows:
                s1, target = reader.load_s1_label(row["sample_id"])
                sars.append(((s1 - means) / stds).astype(np.float32))
                labels.append(target)
            sar = torch.from_numpy(np.stack(sars)).to(device)
            optical = torch.zeros((len(rows), 12, 256, 256), device=device)
            valid = torch.zeros((len(rows), 1, 256, 256), dtype=torch.bool, device=device)
            offset = torch.zeros(len(rows), device=device)
            logits = infer(model, sar, optical, valid, offset, device)
            for i, row in enumerate(rows):
                labels_all.append(labels[i])
                logits_all.append(logits[i])
                owners.append(unique_pixel_mask(grids[row["sample_id"]], preceding))
                preceding.append(grids[row["sample_id"]])
        else:
            for start in range(0, len(rows), config["batch_size"]):
                batch = rows[start:start + config["batch_size"]]
                sar, optical, valid, target, offset = encode_batch(reader, batch, stats, device)
                original = valid.cpu().numpy()[:, 0]
                kept = np.stack([block_valid(original[i], row["sample_id"], level,
                                             config["development_mask_seed"])
                                 for i, row in enumerate(batch)])
                kept_t = torch.from_numpy(kept[:, None]).to(device)
                logits = infer(model, sar, optical, kept_t, offset, device)
                target_np = target.cpu().numpy()
                for i, row in enumerate(batch):
                    labels_all.append(target_np[i])
                    logits_all.append(logits[i])
                    owners.append(unique_pixel_mask(grids[row["sample_id"]], preceding))
                    preceding.append(grids[row["sample_id"]])
        records[process] = {
            "raw": process_metric(labels_all, logits_all, owners, 1.0),
            "scaled": process_metric(labels_all, logits_all, owners, temperature),
        }
    return records


def fit_temperature(model, reader, fit, grids, stats, config, device):
    margins = {}
    for process, rows in fit.items():
        selected = rows[:config["temperature_fit"]["chips_per_process"]]
        preceding, parts = [], []
        for row in selected:
            sar, optical, valid, target, offset = encode_batch(reader, [row], stats, device)
            logits = infer(model, sar, optical, valid, offset, device)[0]
            owner = unique_pixel_mask(grids[row["sample_id"]], preceding)
            preceding.append(grids[row["sample_id"]])
            active = owner & (target.cpu().numpy()[0] != 255)
            truth = target.cpu().numpy()[0][active]
            score = logits[:, active].T
            true_score = score[np.arange(len(truth)), truth]
            other1 = score[np.arange(len(truth)), (truth + 1) % 3] - true_score
            other2 = score[np.arange(len(truth)), (truth + 2) % 3] - true_score
            parts.append(np.stack((other1, other2), axis=1).astype(np.float32))
        if not parts:
            raise RuntimeError(f"No calibration pixels for fit process {process}")
        margins[process] = np.concatenate(parts)

    def objective(log_temperature):
        temperature = np.exp(log_temperature)
        process_nll = []
        for process in sorted(margins):
            values = margins[process] / temperature
            process_nll.append(float(np.mean(np.logaddexp(
                np.logaddexp(0, values[:, 0]), values[:, 1]), dtype=np.float64)))
        return float(np.mean(process_nll))

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded",
                             options={"xatol": 1e-3, "maxiter": 100})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("v6 temperature optimization failed")
    return float(np.exp(result.x)), float(objective(0.0)), float(result.fun)


def bootstrap(values, seed, repetitions):
    values = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if not len(values):
        return {"mean": None, "ci95": None}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(repetitions, len(values)))
    boot = values[draws].mean(axis=1)
    return {"mean": float(values.mean()),
            "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])]}


def get_fit(config):
    old = (v5.CONFIG, v5.BASE, v5.FRAME, v5.STATS, v5.SPLIT_V5)
    v5.CONFIG = SOURCE_CONFIG
    v5.BASE = ROOT / "outputs/exploratory_v5"
    v5.FRAME = v5.BASE / "development_frame.json"
    v5.STATS = v5.BASE / "fit_only_statistics.json"
    v5.SPLIT_V5 = ROOT / config["source_v5_split"]
    source_config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    fit, _, _ = v5.prepare(source_config)
    v5.CONFIG, v5.BASE, v5.FRAME, v5.STATS, v5.SPLIT_V5 = old
    return fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    args = parser.parse_args()
    if args.device == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("XPU unavailable")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frame = json.loads(FRAME.read_text(encoding="utf-8"))
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze["config_sha256"] != sha256(CONFIG) or freeze["v6_frame_sha256"] != sha256(FRAME):
        raise RuntimeError("v6 freeze hashes do not match")
    grids = load_grids()
    fit = get_fit(config)
    device = torch.device(args.device)
    state_rows = {state: frame["paired"] for state in ("full", "half", "zero")}
    state_rows["real_missing"] = frame["real_missing"]
    all_models = {}
    for candidate in config["candidates"]:
        all_models[candidate] = {}
        for seed in config["seeds"]:
            run = BASE / candidate / f"seed_{seed}" / "run.json"
            checkpoint = BASE / candidate / f"seed_{seed}" / "selected.pt"
            if not run.exists() or not checkpoint.exists():
                raise RuntimeError(f"Incomplete v6 run: {candidate} seed {seed}")
            run_record = json.loads(run.read_text(encoding="utf-8"))
            if run_record["selected_sha256"] != sha256(checkpoint):
                raise RuntimeError(f"Selected checkpoint hash mismatch: {candidate} seed {seed}")
            model = model_for(candidate, config, device)
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(payload["model_state"])
            model.eval()
            with TrainArchiveReader(READER_PATH) as reader:
                temperature, before, after = fit_temperature(model, reader, fit, grids, stats, config, device)
                state_records = {state: evaluate_state(model, reader, state_rows[state], grids, stats, config, device, state, temperature)
                                 for state in STATES}
            all_models[candidate][str(seed)] = {"temperature": temperature,
                "fit_nll_before": before, "fit_nll_after": after, "states": state_records,
                "selected_epoch": run_record["selected_epoch"], "selected_checkpoint_sha256": sha256(checkpoint)}

    rows = []
    interactions = []
    repeat = {}
    for candidate, seeds in all_models.items():
        repeat[candidate] = {}
        for metric in METRICS:
            direction = []
            for seed, record in seeds.items():
                values = {}
                for state in STATES:
                    for scale in ("raw", "scaled"):
                        values[state, scale] = [record["states"][state][process][scale][metric]
                                                for process in sorted(record["states"][state])]
                        summary = bootstrap(values[state, scale], config["bootstrap_seed"], config["bootstrap_resamples"])
                        rows.append({"candidate": candidate, "seed": int(seed), "state": state,
                                     "scale": scale, "metric": metric, **summary,
                                     "process_count": len(values[state, scale])})
                full_raw = values["full", "raw"]
                full_scaled = values["full", "scaled"]
                zero_raw = values["zero", "raw"]
                zero_scaled = values["zero", "scaled"]
                defined = [i for i in range(len(full_raw))
                           if all(value is not None and np.isfinite(value)
                                  for value in (full_raw[i], full_scaled[i], zero_raw[i], zero_scaled[i]))]
                full_delta = np.asarray([full_scaled[i] - full_raw[i] for i in defined])
                zero_delta = np.asarray([zero_scaled[i] - zero_raw[i] for i in defined])
                effect = zero_delta - full_delta
                inter = bootstrap(effect, config["bootstrap_seed"], config["bootstrap_resamples"])
                inter.update({"candidate": candidate, "seed": int(seed), "metric": metric,
                              "positive_processes": int((effect > 0).sum()),
                              "negative_processes": int((effect < 0).sum()), "zero_processes": int((effect == 0).sum()),
                              "process_count": len(effect), "total_processes": len(full_raw)})
                interactions.append(inter)
                direction.append(inter["mean"] is not None and inter["mean"] > 0 and inter["positive_processes"] > 8)
            repeat[candidate][metric] = bool(all(direction))

    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "v6_state_metrics.csv").write_text(
        "candidate,seed,state,scale,metric,mean,ci95_low,ci95_high,process_count\n" + "\n".join(
            f"{r['candidate']},{r['seed']},{r['state']},{r['scale']},{r['metric']},{r['mean']:.9g},{r['ci95'][0]:.9g},{r['ci95'][1]:.9g},{r['process_count']}" for r in rows) + "\n", encoding="utf-8")
    (BASE / "v6_interaction.csv").write_text(
        "candidate,seed,metric,mean,ci95_low,ci95_high,positive_processes,negative_processes,zero_processes,process_count\n" + "\n".join(
            f"{r['candidate']},{r['seed']},{r['metric']},{r['mean']:.9g},{r['ci95'][0]:.9g},{r['ci95'][1]:.9g},{r['positive_processes']},{r['negative_processes']},{r['zero_processes']},{r['process_count']}" for r in interactions) + "\n", encoding="utf-8")
    output = {"purpose": config["purpose"], "completed_utc": datetime.now(timezone.utc).isoformat(),
              "config_sha256": sha256(CONFIG), "freeze_manifest_sha256": sha256(FREEZE),
              "models": all_models, "state_metrics": rows, "interactions": interactions,
              "b4_direction_repeated": repeat.get("B4_v6", {}), "repeat_rule": config["interaction_repeat_rule"]}
    (BASE / "v6_summary.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    (BASE / "v6_provenance.json").write_text(json.dumps({"config_sha256": sha256(CONFIG),
        "freeze_manifest_sha256": sha256(FREEZE), "summary_sha256": sha256(BASE / "v6_summary.json"),
        "state_metrics_sha256": sha256(BASE / "v6_state_metrics.csv"),
        "interaction_sha256": sha256(BASE / "v6_interaction.csv"),
        "script_sha256": sha256(Path(__file__))}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"b4_direction_repeated": repeat.get("B4_v6", {}),
                      "summary": str(BASE / "v6_summary.json")}), flush=True)


if __name__ == "__main__":
    main()
