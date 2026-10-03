"""Summarize train-role S2 age/visibility without reading held-out roles.

The input contains mask-only visibility upper bounds. This script reports a
fixed sensitivity grid; it does not choose or freeze an eligibility rule.
"""

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
MASK_INPUT = MANIFESTS / "impactmesh_candidate_train_s2_availability.csv.gz"
MASK_META = MANIFESTS / "impactmesh_candidate_train_s2_availability_meta.json"
FULL_INPUT = MANIFESTS / "impactmesh_candidate_train_optical_full.csv.gz"
FULL_META = MANIFESTS / "impactmesh_candidate_train_optical_full_meta.json"
MASK_OUTPUT = MANIFESTS / "impactmesh_candidate_train_s2_sensitivity.json"
FULL_OUTPUT = MANIFESTS / "impactmesh_candidate_train_optical_full_sensitivity.json"
WINDOW_DAYS = (7, 14, 30, 45)
VISIBLE_THRESHOLDS = (0.0, 0.25, 0.5, 0.75)
PROCESS_MIN_SAMPLES = (1, 5, 10, 20)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="summarize the full 12-band audit")
    args = parser.parse_args()
    input_path = FULL_INPUT if args.full else MASK_INPUT
    meta_path = FULL_META if args.full else MASK_META
    output_path = FULL_OUTPUT if args.full else MASK_OUTPUT
    visible_field = "joint_visible_fraction" if args.full else "mask_joint_visible_upper_fraction"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if sha256(input_path) != meta["output_sha256"]:
        raise RuntimeError("Availability input differs from its audit metadata")
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    with gzip.open(input_path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_sample[row["sample_id"]].append(row)
    if len(by_sample) != int(meta["sample_count"]) or sum(map(len, by_sample.values())) != int(meta["rows"]):
        raise RuntimeError("Availability row/sample count differs from audit metadata")

    activation_samples: dict[str, int] = Counter()
    s1_invalid = Counter()
    for sample_id, rows in by_sample.items():
        if len(rows) != 4 or {int(row["time_index"]) for row in rows} != set(range(4)):
            raise RuntimeError(f"Expected four distinct S2 times: {sample_id}")
        activation = rows[0]["activation"]
        if any(row["activation"] != activation for row in rows):
            raise RuntimeError(f"Activation mismatch: {sample_id}")
        activation_samples[activation] += 1
        first = rows[0]
        if int(first["s1_event_data_mask"]) != 1:
            s1_invalid["data_mask_not_one"] += 1
        if int(first["s1_event_nan_pixels"]) != 0:
            s1_invalid["nan_pixels_nonzero"] += 1
        if int(first["s1_s2_grid_match"]) != 1:
            s1_invalid["s1_s2_grid_mismatch"] += 1

    result = {
        "scope": "candidate training role only; no validation, calibration, internal test, or official holdout read",
        "input_sha256": sha256(input_path),
        "input_meta_sha256": sha256(meta_path),
        "sample_count": len(by_sample),
        "process_count": len(activation_samples),
        "visibility_limit": ("full joint-visible definition including all 12 bands" if args.full else
                             "mask-only upper bound; full S2 band finite/nonfill condition not yet checked"),
        "selection": f"within each pre-S1 window choose maximum {visible_field}; ties choose freshest S2",
        "s1_input_failures": dict(s1_invalid),
        "process_source_sample_counts": {
            "min": min(activation_samples.values()),
            "median": percentile(list(activation_samples.values()), 0.5),
            "max": max(activation_samples.values()),
        },
        "windows": [],
    }
    for days in WINDOW_DAYS:
        selected = []
        for sample_id, rows in by_sample.items():
            first = rows[0]
            s1_ok = (int(first["s1_event_data_mask"]) == 1 and
                     int(first["s1_event_nan_pixels"]) == 0 and
                     int(first["s1_s2_grid_match"]) == 1)
            candidates = [row for row in rows
                          if int(row["s2_at_or_before_s1"]) == 1
                          and 0 <= -float(row["s2_minus_s1_days"]) <= days]
            if not candidates or not s1_ok:
                continue
            best = max(candidates, key=lambda row: (
                float(row[visible_field]),
                float(row["s2_minus_s1_days"])))
            selected.append({
                "sample_id": sample_id,
                "activation": best["activation"],
                "age_days": -float(best["s2_minus_s1_days"]),
                "visible_fraction": float(best[visible_field]),
            })
        window = {
            "window_days": days,
            "samples_with_pre_s1_candidate": len(selected),
            "processes_with_pre_s1_candidate": len({row["activation"] for row in selected}),
            "selected_age_days": {
                "p25": percentile([row["age_days"] for row in selected], 0.25),
                "median": percentile([row["age_days"] for row in selected], 0.5),
                "p75": percentile([row["age_days"] for row in selected], 0.75),
            },
            "selected_visible_upper_fraction": {
                "p25": percentile([row["visible_fraction"] for row in selected], 0.25),
                "median": percentile([row["visible_fraction"] for row in selected], 0.5),
                "p75": percentile([row["visible_fraction"] for row in selected], 0.75),
            },
            "thresholds": [],
        }
        for threshold in VISIBLE_THRESHOLDS:
            eligible = [row for row in selected if row["visible_fraction"] >= threshold]
            per_process = Counter(row["activation"] for row in eligible)
            window["thresholds"].append({
                "minimum_visible_upper_fraction": threshold,
                "eligible_samples": len(eligible),
                "eligible_processes": len(per_process),
                "processes_by_minimum_eligible_samples": {
                    str(minimum): sum(count >= minimum for count in per_process.values())
                    for minimum in PROCESS_MIN_SAMPLES
                },
            })
        result["windows"].append(window)

    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Training role: {result['sample_count']} samples, {result['process_count']} processes")
    print(f"S1 failures: {result['s1_input_failures']}")
    for window in result["windows"]:
        counts = ", ".join(
            f">={item['minimum_visible_upper_fraction']:.2f}: {item['eligible_samples']} samples/{item['eligible_processes']} processes"
            for item in window["thresholds"])
        print(f"{window['window_days']} days: {counts}")
    print(output_path)


if __name__ == "__main__":
    main()
