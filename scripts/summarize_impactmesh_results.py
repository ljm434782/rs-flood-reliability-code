"""Build traceable paper tables from completed, unchanged formal result files."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "formal_v1"
OUT = BASE / "summary_tables"
MODELS = ("B1", "B2", "B4", "B5", "P")
LEVELS = (100, 75, 50, 25, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_fields(prefix: str, summary: dict) -> dict:
    value = summary["mean"]
    ci = summary["ci95"]
    return {prefix: value, f"{prefix}_ci_low": None if ci is None else ci[0],
            f"{prefix}_ci_high": None if ci is None else ci[1],
            f"{prefix}_defined_processes": summary["defined_processes"]}


def validate_comparable_records(records: dict):
    reference = records["B1"]
    internal = set(reference["internal_test"]["processes"])
    holdout = set(reference["official_holdout"]["processes"])
    external = set(reference["external_sen1floods11"]["events"])
    missing = {category: set(record["processes"]) for category, record in
               reference["real_missing_s2"]["categories"].items()}
    if len(internal) != 19 or len(holdout) != 8 or len(external) != 10:
        raise RuntimeError("Unexpected frozen evaluation process/event count")
    for model in MODELS:
        record = records[model]
        current_internal = (set(record["internal_test"]["processes"]) if model == "B1"
                            else set(record["internal_test"]["conditions"]["100_calibrated"]))
        if current_internal != internal or set(record["official_holdout"]["processes"]) != holdout:
            raise RuntimeError(f"Internal or holdout processes differ for {model}")
        if set(record["external_sen1floods11"]["events"]) != external:
            raise RuntimeError(f"External events differ for {model}")
        for category, expected in missing.items():
            if set(record["real_missing_s2"]["categories"][category]["processes"]) != expected:
                raise RuntimeError(f"Real-missing processes differ for {model}: {category}")
        if model != "B1":
            for level in LEVELS:
                for suffix in ("raw", "calibrated"):
                    if set(record["internal_test"]["conditions"][f"{level}_{suffix}"]) != internal:
                        raise RuntimeError(f"Retention process mismatch: {model}/{level}/{suffix}")


def main():
    records = {}
    inputs = {}
    for model in MODELS:
        folder = BASE / model
        records[model] = {}
        for name in ("internal_test", "external_sen1floods11", "official_holdout", "real_missing_s2"):
            path = folder / f"{name}.json"
            if not path.exists():
                raise RuntimeError(f"Formal result missing: {path}")
            records[model][name] = json.loads(path.read_text(encoding="utf-8"))
            inputs[path.relative_to(ROOT).as_posix()] = sha256(path)
    validate_comparable_records(records)
    primary = []
    curve = []
    calibration = []
    missing = []
    for model in MODELS:
        internal = records[model]["internal_test"]
        external = records[model]["external_sen1floods11"]
        holdout = records[model]["official_holdout"]
        real_missing = records[model]["real_missing_s2"]
        if model == "B1":
            internal_full = internal["summary"]
            holdout_full = holdout["summary"]
            external_full = external["summary"]["independent_excluding_spain_nigeria"]["hard"]
        else:
            internal_full = internal["summary"]["100_calibrated"]
            holdout_full = holdout["summary"]["calibrated"]
            external_full = external["summary"]["independent_excluding_spain_nigeria"]["calibrated"]
        row = {"model": model,
               "internal_processes": 19, "holdout_processes": holdout["independent_eligible_process_count"],
               "external_independent_events": 8}
        for prefix, summary in (("internal_flood_iou", internal_full["flood_iou"]),
                                ("internal_macro_iou", internal_full["macro_iou"]),
                                ("holdout_flood_iou", holdout_full["flood_iou"]),
                                ("external_water_iou", external_full["water_iou"])):
            row.update(report_fields(prefix, summary))
        primary.append(row)
        for level in LEVELS:
            if model == "B1":
                score = internal["summary"]["flood_iou"]
            else:
                score = internal["summary"][f"{level}_calibrated"]["flood_iou"]
            curve.append({"model": model, "s2_block_retention_pct": level,
                          **report_fields("flood_iou", score)})
        if model != "B1":
            for metric in ("nll", "brier", "ece15"):
                raw = internal["summary"]["100_raw"][metric]
                calibrated = internal["summary"]["100_calibrated"][metric]
                calibration.append({"model": model, "metric": metric,
                                    **report_fields("raw", raw),
                                    **report_fields("calibrated", calibrated),
                                    "temperature": internal["temperature"]})
        for category, record in real_missing["categories"].items():
            stats = (record["summary"] if model == "B1" else
                     record["summary"]["calibrated"])
            missing.append({"model": model, "category": category,
                            "processes": record["process_count"],
                            "tiles": record["tile_count"],
                            **report_fields("flood_iou", stats["flood_iou"]),
                            **report_fields("macro_iou", stats["macro_iou"])})
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {"primary": OUT / "primary_metrics.csv",
             "curve": OUT / "optical_retention_curve.csv",
             "calibration": OUT / "calibration_metrics.csv",
             "real_missing": OUT / "real_missing_s2.csv"}
    write_csv(paths["primary"], primary)
    write_csv(paths["curve"], curve)
    write_csv(paths["calibration"], calibration)
    write_csv(paths["real_missing"], missing)
    (OUT / "provenance.json").write_text(json.dumps({
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_results_sha256": inputs,
        "table_sha256": {name: sha256(path) for name, path in paths.items()},
        "script_sha256": sha256(Path(__file__)),
        "interpretation": "descriptive extraction only; process bootstrap intervals from frozen evaluation",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved formal summary tables to {OUT}", flush=True)


if __name__ == "__main__":
    main()
