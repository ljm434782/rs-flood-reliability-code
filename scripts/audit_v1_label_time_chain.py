"""Audit label/product time-chain evidence for locked v1 evaluation processes.

This is a metadata-only audit. It reads CEMS activation metadata, the frozen
process manifests, CEMS source-map metadata already indexed locally, and S1
zarr attributes. It does not read ImpactMesh label pixels, S2 pixels, model
scores, or use evaluation results to select processes.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import tarfile
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
RAW = ROOT / "data" / "raw" / "impactmesh"
OUT_CSV = MANIFESTS / "impactmesh_v1_label_time_chain_audit.csv"
OUT_JSON = MANIFESTS / "impactmesh_v1_label_time_chain_audit.json"
OUT_SHA = MANIFESTS / "impactmesh_v1_label_time_chain_audit.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def read_train_times() -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    path = MANIFESTS / "impactmesh_train_event_times.csv.gz"
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[row["activation"]].append(row["s1_event_utc"])
    return result


def read_holdout_times(activations: set[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    path = RAW / "test" / "S1RTC.tar"
    suffix = "_S1RTC.zarr.zip"
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith("S1RTC/") or not member.name.endswith(suffix):
                continue
            sample_id = Path(member.name).name.removesuffix(suffix)
            activation = sample_id.split("_", 1)[0]
            if activation not in activations:
                continue
            with zipfile.ZipFile(io.BytesIO(archive.extractfile(member).read())) as source:
                attrs = json.loads(source.read(".zattrs"))
            times = attrs["datetime"].split(";")
            if len(times) != 4:
                raise RuntimeError(f"Unexpected S1 timestamp count for {sample_id}")
            result[activation].append(times[2])
    return result


def main() -> None:
    split_path = MANIFESTS / "impactmesh_process_split_grouped.csv"
    exclusion_path = MANIFESTS / "impactmesh_evaluation_process_exclusions.csv"
    holdout_path = MANIFESTS / "impactmesh_holdout_frozen_eligibility.csv"
    cems_path = MANIFESTS / "impactmesh_cems_activations.csv"
    cems_index_path = MANIFESTS / "cems_index.csv"
    train_times_path = MANIFESTS / "impactmesh_train_event_times.csv.gz"
    holdout_s1_path = RAW / "test" / "S1RTC.tar"

    exclusions = {}
    with exclusion_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            exclusions[row["activation"]] = row["reason"]

    grouped_roles = {}
    with split_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["role"] in {"internal_test", "calibration", "validation", "train"}:
                grouped_roles[row["activation"]] = row["role"]

    holdout_counts: dict[str, int] = defaultdict(int)
    with holdout_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["paired_eligible"] == "1":
                holdout_counts[row["activation"]] += 1
    holdout_ids = {activation for activation, count in holdout_counts.items() if count >= 5}
    holdout_ids -= set(exclusions)

    internal_ids = {activation for activation, role in grouped_roles.items()
                    if role == "internal_test" and activation not in exclusions}
    if len(internal_ids) != 19:
        raise RuntimeError(f"Expected 19 independent internal processes, found {len(internal_ids)}")
    if len(holdout_ids) != 8:
        raise RuntimeError(f"Expected 8 independent holdout processes, found {len(holdout_ids)}")

    cems = {}
    with cems_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cems[row["activation"]] = row

    cems_products: dict[str, list[dict[str, str]]] = defaultdict(list)
    with cems_index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cems_products[row["activation"]].append(row)

    train_times = read_train_times()
    holdout_times = read_holdout_times(holdout_ids)

    rows = []
    for role, ids, times_by_activation in (
        ("internal_test", sorted(internal_ids), train_times),
        ("official_test_holdout", sorted(holdout_ids), holdout_times),
    ):
        for activation in ids:
            times = sorted(times_by_activation.get(activation, []), key=utc)
            product_rows = cems_products.get(activation, [])
            source_times = sorted({row["source_image_time_utc"] for row in product_rows if row["source_image_time_utc"]})
            source_products = sorted({row["product"] for row in product_rows})
            if activation == "EMSR251" and product_rows:
                classification = "spatial_temporal_match_unverified"
                evidence = (
                    "Convenience sample EMSR251_3_32VML_x473405_y6512885 is documented in "
                    "AOI03; indexed CEMS candidate source-image times are on 2017-10-21 and "
                    "2017-10-22, with S1 event metadata on 2017-10-22. The tile-to-product "
                    "link and product version remain unverified."
                )
            else:
                classification = "unable_to_confirm"
                evidence = (
                    "CEMS activation metadata is available, but no ImpactMesh tile-to-CEMS "
                    "product ID or AOI mapping is available in the frozen evidence. "
                    "Activation time and last update are not satellite reference-image times."
                )
            cems_row = cems.get(activation, {})
            rows.append({
                "evaluation_role": role,
                "activation": activation,
                "classification": classification,
                "s1_event_sample_count": len(times),
                "s1_event_first_utc": times[0] if times else "UNVERIFIED",
                "s1_event_last_utc": times[-1] if times else "UNVERIFIED",
                "cems_name": cems_row.get("name", "UNVERIFIED"),
                "cems_countries": cems_row.get("countries", "UNVERIFIED"),
                "cems_activation_time": cems_row.get("activation_time", "UNVERIFIED"),
                "cems_last_update": cems_row.get("last_update", "UNVERIFIED"),
                "product_delivery_date_utc": "UNVERIFIED",
                "reference_image_times_utc": ";".join(source_times) or "UNVERIFIED",
                "flood_situation_time_utc": "UNVERIFIED",
                "indexed_product_names": ";".join(source_products) or "UNVERIFIED",
                "cems_activation_url": cems_row.get("source_url", f"https://mapping.emergency.copernicus.eu/activations/{activation}/"),
                "cross_split_risk": "none identified in reviewed public metadata; residual risk UNVERIFIED",
                "evidence_note": evidence,
                "exclusion_status": "not excluded from independent primary summary",
            })

    rows.sort(key=lambda row: (row["evaluation_role"], row["activation"]))
    fieldnames = list(rows[0])
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = defaultdict(int)
    for row in rows:
        counts[f"{row['evaluation_role']}|{row['classification']}"] += 1
    payload = {
        "purpose": "Metadata-only label/product/time-chain audit for locked v1 independent evaluation processes",
        "created_utc": datetime.now().astimezone().isoformat(),
        "classification_rule": {
            "direct_product_mapping": "Requires a documented ImpactMesh tile-to-CEMS product identifier; none found in this audit.",
            "spatial_temporal_match_unverified": "Existing AOI and reference-image metadata are close to an S1 event for a documented convenience sample, but the tile-to-product link is not direct.",
            "unable_to_confirm": "No frozen tile-level product mapping; activation dates are not used as satellite acquisition dates.",
        },
        "counts": dict(counts),
        "process_counts": {"internal_test": len(internal_ids), "official_test_holdout": len(holdout_ids)},
        "label_pixels_read": False,
        "model_scores_read": False,
        "s2_pixels_read": False,
        "sources": {
            "impactmesh_process_split_grouped.csv": sha256(split_path),
            "impactmesh_evaluation_process_exclusions.csv": sha256(exclusion_path),
            "impactmesh_holdout_frozen_eligibility.csv": sha256(holdout_path),
            "impactmesh_cems_activations.csv": sha256(cems_path),
            "cems_index.csv": sha256(cems_index_path),
            "impactmesh_train_event_times.csv.gz": sha256(train_times_path),
            "impactmesh_test_S1RTC.tar": sha256(holdout_s1_path),
        },
        "output_sha256": {"csv": sha256(OUT_CSV)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_SHA.write_text(
        f"{sha256(OUT_CSV)}  {OUT_CSV.name}\n"
        f"{sha256(OUT_JSON)}  {OUT_JSON.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
