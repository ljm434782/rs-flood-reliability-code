"""Final external Sen1Floods11 hand-QC water/non-water test only."""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import torch

from flood_reliability.formal_metrics import average_precision_binary, macro_with_bootstrap
from flood_reliability.overlap import unique_pixel_mask
from scripts.calibrate_impactmesh_formal import CONFIG, ROOT, STATS, load_model, sha256


MANIFESTS = ROOT / "data" / "manifests"
AUDIT = MANIFESTS / "sen1floods11_test_audit.csv"
MANIFEST = MANIFESTS / "sen1floods11_manifest.csv"
EXCLUSIONS = MANIFESTS / "sen1_external_process_exclusions.csv"
RAW = ROOT / "data" / "raw" / "sen1floods11" / "data" / "flood_events" / "HandLabeled"


class BinaryAccumulator:
    def __init__(self):
        self.confusion = np.zeros((2, 2), np.int64)
        self.pixels = 0
        self.nll = 0.
        self.brier = 0.
        self.count = np.zeros(15, np.int64)
        self.conf = np.zeros(15)
        self.correct = np.zeros(15)
        self.scores = []
        self.truth = []

    def add(self, label: np.ndarray, probability: np.ndarray,
            support: np.ndarray, *, probability_available: bool):
        truth = label[support].astype(np.int64)
        score = probability[support].astype(np.float64)
        if not len(truth):
            return
        if not set(np.unique(truth)) <= {0, 1} or not np.isfinite(score).all():
            raise RuntimeError("Unexpected external labels or predictions")
        predicted = (score >= .5).astype(np.int64)
        self.confusion += np.bincount(truth * 2 + predicted, minlength=4).reshape(2, 2)
        self.pixels += len(truth)
        if probability_available:
            true_prob = np.where(truth == 1, score, 1 - score)
            with np.errstate(divide="ignore"):
                self.nll += float(np.sum(-np.log(true_prob)))
            self.brier += float(2 * np.square(score - truth).sum())
            confidence = np.maximum(score, 1 - score)
            bins = np.minimum((confidence * 15).astype(np.int64), 14)
            self.count += np.bincount(bins, minlength=15)
            self.conf += np.bincount(bins, weights=confidence, minlength=15)
            self.correct += np.bincount(bins, weights=(predicted == truth), minlength=15)
            self.scores.append(score.astype(np.float32))
            self.truth.append(truth.astype(bool))

    def finish(self, *, probability_available: bool):
        tp = int(self.confusion[1, 1])
        union = int(self.confusion[1].sum() + self.confusion[:, 1].sum() - tp)
        fden = int(self.confusion[1].sum() + self.confusion[:, 1].sum())
        row = {"pixels": self.pixels, "confusion": self.confusion.tolist(),
               "water_iou": tp / union if union else None,
               "water_f1": 2 * tp / fden if fden else None}
        if probability_available:
            occupied = self.count > 0
            ece = np.sum((self.count[occupied] / self.pixels) * np.abs(
                self.correct[occupied] / self.count[occupied]
                - self.conf[occupied] / self.count[occupied]))
            row.update({"water_auprc": average_precision_binary(
                np.concatenate(self.truth), np.concatenate(self.scores)),
                "nll": self.nll / self.pixels, "brier": self.brier / self.pixels,
                "ece15": float(ece)})
        return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("B1", "B2", "B4", "B5", "P"))
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    args = parser.parse_args()
    for name in ("B2", "B4", "B5", "P"):
        folder = ROOT / "outputs" / "formal_v1" / name
        if not (folder / "run.json").exists() or not (folder / "temperature.json").exists():
            raise RuntimeError("External test requires all formal weights and temperatures frozen")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model = None
    checkpoint = None
    temperature_file = None
    temperature = 1.
    threshold_file = None
    if args.model == "B1":
        threshold_file = ROOT / "outputs" / "formal_v1" / "B1" / "thresholds.json"
        baseline = json.loads(threshold_file.read_text(encoding="utf-8"))
        threshold = float(baseline["water_threshold_db"])
    else:
        model, checkpoint = load_model(args.model, device, stats, config)
        temperature_file = ROOT / "outputs" / "formal_v1" / args.model / "temperature.json"
        temp_record = json.loads(temperature_file.read_text(encoding="utf-8"))
        if temp_record["checkpoint_sha256"] != sha256(checkpoint):
            raise RuntimeError("Temperature/checkpoint mismatch")
        temperature = float(temp_record["temperature"])
    with AUDIT.open(newline="", encoding="utf-8") as handle:
        scenes = list(csv.DictReader(handle))
    if len(scenes) != 90 or len({row["scene"] for row in scenes}) != 90:
        raise RuntimeError("Expected fixed 90-scene official hand-QC external test")
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        files = {row["local_path"].replace("\\", "/"): row for row in csv.DictReader(handle)
                 if row["selected_test"] == "True" and row["local_path"]}
    groups = defaultdict(list)
    for scene in scenes:
        groups[scene["event"]].append(scene)
    with EXCLUSIONS.open(newline="", encoding="utf-8") as handle:
        excluded_events = {row["event"] for row in csv.DictReader(handle)}
    if excluded_events != {"Spain", "Nigeria"} or not excluded_events <= set(groups):
        raise RuntimeError("Frozen external same-process exclusions changed")
    results = {}
    for event_index, (event, values) in enumerate(sorted(groups.items()), 1):
        accum = {state: BinaryAccumulator() for state in (
            ("hard",) if args.model == "B1" else ("raw", "calibrated"))}
        preceding = []
        excluded_input_pixels = 0
        for row in sorted(values, key=lambda item: item["scene"]):
            scene = row["scene"]
            sar_path = RAW / "S1Hand" / f"{scene}_S1Hand.tif"
            label_path = RAW / "LabelHand" / f"{scene}_LabelHand.tif"
            for path in (sar_path, label_path):
                key = path.relative_to(ROOT).as_posix()
                source = files.get(key)
                if (source is None or path.stat().st_size != int(source["size_bytes"])
                        or sha256(path) != source["sha256"]):
                    raise RuntimeError(f"External source failed manifest verification: {path}")
            with rasterio.open(sar_path) as image, rasterio.open(label_path) as truth_source:
                if (image.shape != (512, 512) or truth_source.shape != (512, 512)
                        or image.count != 2 or truth_source.count != 1
                        or image.crs != truth_source.crs
                        or not image.transform.almost_equals(truth_source.transform)):
                    raise RuntimeError(f"External S1/label grid mismatch: {scene}")
                sar = image.read().astype(np.float32)
                label = truth_source.read(1).astype(np.int64)
                bounds = truth_source.bounds
                grid = {"grid_crs": str(truth_source.crs), "left": bounds.left,
                        "bottom": bounds.bottom, "right": bounds.right, "top": bounds.top}
            if not set(np.unique(label)) <= {-1, 0, 1}:
                raise RuntimeError(f"External label code changed: {scene}")
            owner = unique_pixel_mask(grid, preceding, shape=(512, 512))
            preceding.append(grid)
            finite = np.isfinite(sar).all(axis=0)
            excluded_input_pixels += int(np.count_nonzero(owner & (label != -1) & ~finite))
            support = owner & (label != -1) & finite
            if args.model == "B1":
                score = (sar[1] <= threshold).astype(np.float64)
                accum["hard"].add(label, score, support, probability_available=False)
            else:
                mean = np.asarray(stats["sar_mean"], dtype=np.float32)[:, None, None]
                std = np.asarray(stats["sar_std"], dtype=np.float32)[:, None, None]
                normalized = np.where(finite[None], (sar - mean) / std, 0).astype(np.float32)
                tensor = torch.from_numpy(normalized[None]).to(device)
                with torch.inference_mode():
                    if args.model == "B2":
                        logits = model(tensor)
                    else:
                        optical = torch.zeros((1, 12, 512, 512), dtype=torch.float32, device=device)
                        valid = torch.zeros((1, 1, 512, 512), dtype=torch.bool, device=device)
                        present = torch.zeros(1, dtype=torch.bool, device=device)
                        offset = torch.zeros(1, dtype=torch.float32, device=device)
                        logits = model(tensor, optical, valid, present,
                                       normalized_time_offset=offset)
                    raw = (1 - torch.softmax(logits, dim=1)[0, 0]).cpu().numpy()
                    calibrated = (1 - torch.softmax(logits / temperature, dim=1)[0, 0]).cpu().numpy()
                accum["raw"].add(label, raw, support, probability_available=True)
                accum["calibrated"].add(label, calibrated, support, probability_available=True)
        results[event] = {"scenes": len(values), "excluded_nonfinite_s1_pixels": excluded_input_pixels,
                          "scores": {state: score.finish(probability_available=state != "hard")
                                     for state, score in accum.items()}}
        print(f"{args.model} external event {event_index}/{len(groups)} {event}: "
              f"{len(values)} scenes", flush=True)
    keys = ("water_iou", "water_f1") if args.model == "B1" else (
        "water_iou", "water_f1", "water_auprc", "nll", "brier", "ece15")
    states = ("hard",) if args.model == "B1" else ("raw", "calibrated")
    summary = {}
    for scope, included in (("independent_excluding_spain_nigeria",
                             [e for e in results if e not in excluded_events]),
                            ("all_with_overlap_sensitivity", list(results))):
        summary[scope] = {}
        for state in states:
            scoped = {event: results[event]["scores"][state] for event in included}
            summary[scope][state] = {key: macro_with_bootstrap(scoped, key) for key in keys}
    output = ROOT / "outputs" / "formal_v1" / args.model / "external_sen1floods11.json"
    output.write_text(json.dumps({
        "model": args.model, "purpose": "final water/non-water hand-QC external test only",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "external_audit_sha256": sha256(AUDIT), "external_manifest_sha256": sha256(MANIFEST),
        "event_exclusions_sha256": sha256(EXCLUSIONS),
        "excluded_main_events": sorted(excluded_events),
        "config_sha256": sha256(CONFIG), "stats_sha256": sha256(STATS),
        "checkpoint_sha256": None if checkpoint is None else sha256(checkpoint),
        "temperature_sha256": None if temperature_file is None else sha256(temperature_file),
        "thresholds_sha256": None if threshold_file is None else sha256(threshold_file),
        "script_sha256": sha256(Path(__file__)), "events": results, "summary": summary,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
