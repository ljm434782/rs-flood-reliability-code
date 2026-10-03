"""Evaluate frozen neural models on process-disjoint internal test tiles."""

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from flood_reliability.evaluation_counts import confusion_counts
from flood_reliability.formal_metrics import average_precision_binary, macro_with_bootstrap, process_metrics
from flood_reliability.impactmesh_io import TrainArchiveReader
from flood_reliability.overlap import unique_pixel_mask
from flood_reliability.probability_counts import probability_sums
from scripts.calibrate_impactmesh_formal import (
    CONFIG, ELIGIBILITY, EXCLUSIONS, GRID, ROOT, SPLIT, STATS, load_model, role_rows, sha256,
)
from scripts.train_impactmesh_formal import encode_batch


LEVELS = (100, 75, 50, 25, 0)
METRICS = ("flood_iou", "flood_f1", "permanent_iou", "macro_iou",
           "flood_auprc", "nll", "brier", "ece15")


def block_valid(valid: np.ndarray, sample_id: str, level: int, seed: int) -> np.ndarray:
    if level not in LEVELS or valid.shape != (256, 256) or valid.dtype != np.bool_:
        raise ValueError("Expected frozen 256-square optical mask and retention level")
    if level == 100:
        return valid.copy()
    selected = sorted(range(64), key=lambda block: hashlib.sha256(
        f"{seed}|{sample_id}|{block}".encode("ascii")).hexdigest())[:level * 64 // 100]
    blocks = np.zeros((8, 8), dtype=bool)
    blocks.ravel()[selected] = True
    return valid & np.repeat(np.repeat(blocks, 32, axis=0), 32, axis=1)


class ScoreAccumulator:
    def __init__(self):
        self.confusion = np.zeros((3, 3), np.int64)
        self.pixels = 0
        self.nll_sum = 0.
        self.brier_sum = 0.
        self.bin_count = np.zeros(15, np.int64)
        self.bin_conf_sum = np.zeros(15, np.float64)
        self.bin_correct_sum = np.zeros(15, np.float64)
        self.flood_scores = []
        self.flood_truth = []

    def add(self, target: np.ndarray, logits: np.ndarray, owner: np.ndarray,
            temperature: float, *, collect_ap: bool):
        prediction = np.argmax(logits, axis=0).astype(np.int64)
        self.confusion += confusion_counts(target, prediction, num_classes=3,
                                           evaluation_mask=owner)
        shifted = logits / temperature
        shifted -= shifted.max(axis=0, keepdims=True)
        exp = np.exp(shifted, dtype=np.float64)
        probabilities = exp / exp.sum(axis=0, keepdims=True)
        sums = probability_sums(target, np.moveaxis(probabilities, 0, -1),
                                evaluation_mask=owner)
        self.pixels += sums.pixels
        self.nll_sum += sums.nll
        self.brier_sum += sums.brier
        selected = owner & (target != 255)
        if selected.any():
            confidence = probabilities.max(axis=0)[selected]
            correct = (prediction[selected] == target[selected]).astype(np.float64)
            bins = np.minimum((confidence * 15).astype(np.int64), 14)
            self.bin_count += np.bincount(bins, minlength=15)
            self.bin_conf_sum += np.bincount(bins, weights=confidence, minlength=15)
            self.bin_correct_sum += np.bincount(bins, weights=correct, minlength=15)
            if collect_ap:
                self.flood_scores.append(probabilities[2, selected].astype(np.float32))
                self.flood_truth.append((target[selected] == 2).astype(bool))

    def finish(self) -> dict:
        ap = (average_precision_binary(np.concatenate(self.flood_truth),
                                       np.concatenate(self.flood_scores))
              if self.flood_scores else None)
        return process_metrics(self.confusion, self.pixels, self.nll_sum,
                               self.brier_sum, self.bin_count,
                               self.bin_conf_sum, self.bin_correct_sum, ap)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("B2", "B4", "B5", "P"))
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    if args.device == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("XPU unavailable")
    # No internal test use before all four final models and temperatures exist.
    for model_name in ("B2", "B4", "B5", "P"):
        folder = ROOT / "outputs" / "formal_v1" / model_name
        if not (folder / "run.json").exists() or not (folder / "temperature.json").exists():
            raise RuntimeError(f"Model and temperature must be frozen before test: {model_name}")
    device = torch.device(args.device)
    model, checkpoint = load_model(args.model, device, stats, config)
    temperature_file = ROOT / "outputs" / "formal_v1" / args.model / "temperature.json"
    temperature_record = json.loads(temperature_file.read_text(encoding="utf-8"))
    if temperature_record["checkpoint_sha256"] != sha256(checkpoint):
        raise RuntimeError("Temperature fitted to another checkpoint")
    if temperature_record["evaluation_exclusions_sha256"] != sha256(EXCLUSIONS):
        raise RuntimeError("Calibration/test event exclusions differ")
    temperature = float(temperature_record["temperature"])
    groups, grids = role_rows("internal_test")
    by_condition = defaultdict(dict)
    with TrainArchiveReader(ROOT / "data" / "raw" / "impactmesh" / "train") as reader:
        for process_index, (process, rows) in enumerate(groups.items(), 1):
            accumulators = {(level, state): ScoreAccumulator() for level in LEVELS
                            for state in ("raw", "calibrated")}
            preceding = []
            for start in range(0, len(rows), config["batch_size"]):
                batch = rows[start:start + config["batch_size"]]
                sar, optical, valid, target, time_offset = encode_batch(
                    reader, batch, stats, device)
                original = valid.cpu().numpy()[:, 0]
                labels = target.cpu().numpy()
                owners = []
                for row in batch:
                    owners.append(unique_pixel_mask(grids[row["sample_id"]], preceding))
                    preceding.append(grids[row["sample_id"]])
                for level in LEVELS:
                    if args.model == "B2" and level != 100:
                        continue
                    kept = np.stack([block_valid(original[i], row["sample_id"],
                                                 level, config["seed"])
                                     for i, row in enumerate(batch)])
                    kept_t = torch.from_numpy(kept[:, None]).to(device)
                    present = kept_t.flatten(1).any(dim=1)
                    optical_input = torch.where(kept_t, optical, torch.zeros_like(optical))
                    offset = torch.where(present, time_offset, torch.zeros_like(time_offset))
                    with torch.inference_mode():
                        logits_t = (model(sar) if args.model == "B2" else
                                    model(sar, optical_input, kept_t, present,
                                          normalized_time_offset=offset))
                    logits = logits_t.cpu().numpy()
                    for i in range(len(batch)):
                        for state, temp in (("raw", 1.), ("calibrated", temperature)):
                            accumulators[level, state].add(
                                labels[i], logits[i].copy(), owners[i], temp,
                                collect_ap=True)
            for level in LEVELS:
                for state in ("raw", "calibrated"):
                    if args.model == "B2" and level != 100:
                        by_condition[f"{level}_{state}"][process] = by_condition[f"100_{state}"][process]
                    else:
                        by_condition[f"{level}_{state}"][process] = accumulators[level, state].finish()
            print(f"{args.model} internal test process {process_index}/{len(groups)} "
                  f"{process}: {len(rows)} tiles", flush=True)
    summaries = {}
    for condition, process_rows in sorted(by_condition.items()):
        summaries[condition] = {metric: macro_with_bootstrap(process_rows, metric)
                                for metric in METRICS}
    output = ROOT / "outputs" / "formal_v1" / args.model / "internal_test.json"
    output.write_text(json.dumps({
        "model": args.model, "role": "internal_test", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "temperature": temperature, "temperature_sha256": sha256(temperature_file),
        "checkpoint_sha256": sha256(checkpoint), "split_sha256": sha256(SPLIT),
        "eligibility_sha256": sha256(ELIGIBILITY), "grid_sha256": sha256(GRID),
        "evaluation_exclusions_sha256": sha256(EXCLUSIONS),
        "config_sha256": sha256(CONFIG), "stats_sha256": sha256(STATS),
        "script_sha256": sha256(Path(__file__)),
        "conditions": by_condition, "summary": summaries,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
