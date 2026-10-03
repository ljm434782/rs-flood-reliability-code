"""Describe paired effects in unchanged v1 outputs; never select models or temperatures.

This summary was specified after viewing v1 results. All paired intervals are
descriptive, unadjusted process-bootstrap intervals, not new confirmatory tests.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/formal_v1"
OUT = ROOT / "outputs/publication_evidence_20260928"
MODELS = ("B2", "B4", "B5", "P")
LEVELS = (100, 75, 50, 25, 0)
METRICS = ("nll", "brier", "ece15")
SEED = 20260927
RESAMPLES = 10000


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(model, comparison, metric, ids, values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite effect: {model}/{comparison}/{metric}")
    draws = np.random.default_rng(SEED).integers(
        0, len(values), size=(RESAMPLES, len(values))
    )
    lo, hi = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
    return {
        "model": model, "comparison": comparison, "metric": metric,
        "process_count": len(values), "mean_difference": float(values.mean()),
        "ci95_low": float(lo), "ci95_high": float(hi),
        "negative_processes": int((values < 0).sum()),
        "positive_processes": int((values > 0).sum()),
        "zero_processes": int((values == 0).sum()),
    }, [
        {"model": model, "comparison": comparison, "metric": metric,
         "process": pid, "difference": float(value)}
        for pid, value in zip(ids, values)
    ]


def main():
    rows, detail, hashes = [], [], {}
    reference_ids = None
    for model in MODELS:
        path = SOURCE / model / "internal_test.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        hashes[path.relative_to(ROOT).as_posix()] = digest(path)
        conditions = data["conditions"]
        ids = sorted(conditions["100_raw"])
        if len(ids) != 19:
            raise ValueError("Expected 19 frozen internal-test processes")
        if reference_ids is None:
            reference_ids = ids
        if ids != reference_ids:
            raise ValueError("Models do not share process IDs")
        for condition in conditions.values():
            if sorted(condition) != ids:
                raise ValueError("Conditions do not share process IDs")
        for metric in METRICS:
            effects = {}
            for level in LEVELS:
                values = [conditions[f"{level}_calibrated"][pid][metric]
                          - conditions[f"{level}_raw"][pid][metric] for pid in ids]
                effects[level] = np.asarray(values)
                row, per_process = describe(
                    model, f"{level}_scaled_minus_raw", metric, ids, values
                )
                rows.append(row)
                detail.extend(per_process)
            row, per_process = describe(
                model, "zero_minus_full_calibration_effect", metric, ids,
                effects[0] - effects[100]
            )
            rows.append(row)
            detail.extend(per_process)
        for suffix in ("raw", "calibrated"):
            for metric in ("flood_iou", "nll", "brier", "ece15"):
                valid_ids = [pid for pid in ids if
                             conditions[f"0_{suffix}"][pid][metric] is not None
                             and conditions[f"100_{suffix}"][pid][metric] is not None]
                values = [conditions[f"0_{suffix}"][pid][metric]
                          - conditions[f"100_{suffix}"][pid][metric] for pid in valid_ids]
                row, per_process = describe(
                    model, f"zero_minus_full_{suffix}", metric, valid_ids, values
                )
                rows.append(row)
                detail.extend(per_process)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, records in (("paired_effects.csv", rows),
                          ("paired_effects_by_process.csv", detail)):
        with (OUT / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (OUT / "provenance.json").write_text(json.dumps({
        "status": "post-test descriptive supplement; no tuning or new blind test",
        "source_sha256": hashes,
        "script_sha256": digest(Path(__file__)),
        "output_sha256": {name: digest(OUT / name) for name in
                          ("paired_effects.csv", "paired_effects_by_process.csv")},
        "seed": SEED, "resamples": RESAMPLES,
        "interval": "paired-process percentile; descriptive; no multiplicity correction",
        "interpretation": {
            "scaled_minus_raw": "negative improves each probability score",
            "zero_minus_full_calibration_effect": "positive means less calibration benefit at zero S2",
            "zero_minus_full": "negative worsens IoU; positive worsens probability scores",
        },
    }, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        if (row["metric"] == "nll" and row["comparison"] in
            ("100_scaled_minus_raw", "0_scaled_minus_raw",
             "zero_minus_full_calibration_effect")) or (
            row["metric"] == "flood_iou" and row["comparison"] == "zero_minus_full_raw"
        ):
            print(json.dumps(row))


if __name__ == "__main__":
    main()
