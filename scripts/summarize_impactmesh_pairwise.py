"""Descriptive paired process differences for the frozen formal evaluations."""

import csv
import json
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parents[1] / "outputs" / "formal_v1"
OUT = BASE / "summary_tables" / "paired_differences.csv"
SEED = 20260927
REPETITIONS = 10000


def load(model, name):
    return json.loads((BASE / model / f"{name}.json").read_text(encoding="utf-8"))


def values(model, dataset, condition, metric):
    record = load(model, dataset)
    if dataset == "internal_test":
        return {key: row[metric] for key, row in record["conditions"][f"{condition}_calibrated"].items()}
    if dataset == "official_holdout":
        return {key: row["scores"]["calibrated"][metric]
                for key, row in record["processes"].items()}
    if dataset == "external_sen1floods11":
        return {key: row["scores"]["calibrated"][metric]
                for key, row in record["events"].items() if key not in {"Spain", "Nigeria"}}
    if dataset == "real_missing_s2":
        return {key: row["scores"]["calibrated"][metric]
                for key, row in record["categories"][condition]["processes"].items()}
    raise ValueError(dataset)


def main():
    comparisons = [
        ("internal_test", "100", "flood_iou", "P", "B4"),
        ("internal_test", "0", "flood_iou", "P", "B4"),
        ("internal_test", "100", "flood_iou", "P", "B5"),
        ("internal_test", "0", "flood_iou", "P", "B5"),
        ("official_holdout", "100", "flood_iou", "P", "B4"),
        ("external_sen1floods11", "0", "water_iou", "P", "B4"),
        ("real_missing_s2", "all_unqualified", "flood_iou", "P", "B4"),
    ]
    result = []
    for dataset, condition, metric, first, second in comparisons:
        left = values(first, dataset, condition, metric)
        right = values(second, dataset, condition, metric)
        if set(left) != set(right):
            raise RuntimeError(f"Process mismatch: {dataset}/{condition}/{first}/{second}")
        ids = sorted(left)
        differences = np.asarray([np.nan if left[key] is None or right[key] is None else
                                  left[key] - right[key] for key in ids], dtype=np.float64)
        if not np.isfinite(differences[~np.isnan(differences)]).all():
            raise RuntimeError("Nonfinite paired difference")
        defined = differences[~np.isnan(differences)]
        rng = np.random.default_rng(SEED)
        draws = rng.integers(0, len(differences), size=(REPETITIONS, len(differences)))
        samples = differences[draws]
        counts = (~np.isnan(samples)).sum(axis=1)
        boot = np.nansum(samples, axis=1)[counts > 0] / counts[counts > 0]
        ci = np.quantile(boot, [.025, .975])
        result.append({"dataset": dataset, "condition": condition, "metric": metric,
                       "first_model": first, "second_model": second,
                       "processes": len(ids), "defined_processes": len(defined),
                       "mean_first_minus_second": float(defined.mean()),
                       "ci95_low": float(ci[0]), "ci95_high": float(ci[1]),
                       "description": "descriptive paired process bootstrap; no multiplicity adjustment"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(json.dumps(result, indent=2))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
