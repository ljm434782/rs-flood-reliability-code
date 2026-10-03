"""Materialize the audited G3 Rule-A retrospective candidate manifests.

The program uses manifest rows and the exhaustive, header-only spatial-overlap
appendix.  It never opens imagery, raster values, labels, predictions, scores,
or a dataset reader.  Existing ``g3_conservative_candidate_*`` files are inputs
only and are never modified.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests"
OVERLAP_CSV = ROOT / "docs" / "G3_CANDIDATE_SPATIAL_OVERLAP_AUDIT.csv"
STATUS = "CANDIDATE_RETROSPECTIVE_ONLY_NOT_FROZEN"
ROLE_ORDER = ("fit", "calibration", "internal_retro", "holdout_retro")
SOURCE_FILES = {
    "fit": "g3_conservative_candidate_fit.csv",
    "calibration": "g3_conservative_candidate_calibration.csv",
    "internal_retro": "g3_conservative_candidate_internal_retro.csv",
    "holdout_retro": "g3_conservative_candidate_holdout_retro.csv",
}
EXPECTED_SOURCE_SHA256 = {
    "fit": "d9eec725011706e500996a2472144281c4588f26a718aec14112f5fe916d09fc",
    "calibration": "9186134cb5b13126ea870aef1ab78b3b20c0d3f184878f649ea9ae923a96a4be",
    "internal_retro": "596c8019f9062443ed351ae947630792eac8fd44a86029116f9460594bac24fd",
    "holdout_retro": "46a520224ed096fa8712e8def637824dab42f5c7e712982b041d71645480789d",
    "spatial_overlap_audit": "26cfe8be090e8dbb3a6a738ca25c1156da16b611d86b95d637e3366122ef67f6",
}
EXPECTED_COUNTS = {"fit": 3525, "calibration": 2117, "internal_retro": 1906, "holdout_retro": 492}
EXPECTED_UNITS = {"fit": 60, "calibration": 19, "internal_retro": 18, "holdout_retro": 9}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv_with_line_numbers(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"Missing CSV header: {path}")
        rows: list[dict[str, str]] = []
        for source_line_number, row in enumerate(reader, start=2):
            if row["sample_id"] in {item["sample_id"] for item in rows}:
                raise RuntimeError(f"Duplicate source sample ID in {path.name}: {row['sample_id']}")
            row["_source_line_number"] = str(source_line_number)
            rows.append(row)
    return list(reader.fieldnames), rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_hashes() -> dict[str, str]:
    values = {role: sha256(MANIFEST_DIR / filename) for role, filename in SOURCE_FILES.items()}
    values["spatial_overlap_audit"] = sha256(OVERLAP_CSV)
    for name, actual in values.items():
        expected = EXPECTED_SOURCE_SHA256[name]
        if actual != expected:
            raise RuntimeError(f"Unexpected source SHA-256 for {name}: {actual} != {expected}")
    return values


def overlap_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    """Index each audited spatial pair from both endpoint directions."""
    index: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(row["intersection_grid_cells_10m"]) < 1:
            raise RuntimeError("Spatial-overlap audit contains a non-positive intersection")
        left = (row["role_a"], row["sample_id_a"])
        right = (row["role_b"], row["sample_id_b"])
        index[left].append(row)
        index[right].append(row)
    return index


def other_endpoint(row: dict[str, str], role: str, sample_id: str) -> tuple[str, str]:
    if row["role_a"] == role and row["sample_id_a"] == sample_id:
        return row["role_b"], row["sample_id_b"]
    if row["role_b"] == role and row["sample_id_b"] == sample_id:
        return row["role_a"], row["sample_id_a"]
    raise RuntimeError("Overlap index endpoint mismatch")


def make_derived_rows(
    role: str,
    source_filename: str,
    source_fields: list[str],
    source_rows: list[dict[str, str]],
    retained_ids: set[str],
) -> tuple[list[str], list[dict[str, str]]]:
    retained = [row for row in source_rows if row["sample_id"] in retained_ids]
    unit_order: dict[str, int] = {}
    next_unit_order = 1
    within_unit_order: Counter[str] = Counter()
    output: list[dict[str, str]] = []
    for sample_order, source in enumerate(retained, start=1):
        unit = source["conservative_unit_id"]
        if unit not in unit_order:
            unit_order[unit] = next_unit_order
            next_unit_order += 1
        within_unit_order[unit] += 1
        row = {field: source[field] for field in source_fields}
        row.update(
            {
                "derived_status": STATUS,
                "derived_role": role,
                "derived_role_rank": str(ROLE_ORDER.index(role) + 1),
                "derived_sample_order": str(sample_order),
                "derived_unit_order": str(unit_order[unit]),
                "derived_within_unit_order": str(within_unit_order[unit]),
                "source_candidate_filename": source_filename,
                "source_candidate_line_number": source["_source_line_number"],
            }
        )
        output.append(row)
    fields = source_fields + [
        "derived_status",
        "derived_role",
        "derived_role_rank",
        "derived_sample_order",
        "derived_unit_order",
        "derived_within_unit_order",
        "source_candidate_filename",
        "source_candidate_line_number",
    ]
    return fields, output


def main() -> None:
    source_sha = source_hashes()
    source: dict[str, tuple[list[str], list[dict[str, str]]]] = {
        role: read_csv_with_line_numbers(MANIFEST_DIR / filename) for role, filename in SOURCE_FILES.items()
    }
    overlap_rows = list(csv.DictReader(OVERLAP_CSV.open("r", encoding="utf-8", newline="")))
    if len(overlap_rows) != 509:
        raise RuntimeError(f"Expected 509 audited spatial-overlap rows, found {len(overlap_rows)}")
    overlaps = overlap_index(overlap_rows)

    # Rule A: retain fit; remove a later role's whole tile if it shares one or
    # more verified 10 m cells with any retained earlier-role tile.
    retained_ids: dict[str, set[str]] = {}
    excluded_rows: list[dict[str, str]] = []
    for role in ROLE_ORDER:
        _, role_rows = source[role]
        all_ids = {row["sample_id"] for row in role_rows}
        earlier = {
            (earlier_role, sample_id)
            for earlier_role in ROLE_ORDER[: ROLE_ORDER.index(role)]
            for sample_id in retained_ids[earlier_role]
        }
        excluded_ids: set[str] = set()
        for row in role_rows:
            hits = [
                item
                for item in overlaps[(role, row["sample_id"])]
                if other_endpoint(item, role, row["sample_id"]) in earlier
            ]
            if not hits:
                continue
            excluded_ids.add(row["sample_id"])
            prior = sorted({other_endpoint(item, role, row["sample_id"])[0] for item in hits})
            prior_samples = sorted({other_endpoint(item, role, row["sample_id"])[1] for item in hits})
            excluded_rows.append(
                {
                    "derived_status": STATUS,
                    "derived_rule": "whole_tile_exclude_if_shared_10m_cell_with_retained_prior_role",
                    "source_role": role,
                    "source_candidate_filename": SOURCE_FILES[role],
                    "source_candidate_line_number": row["_source_line_number"],
                    "sample_id": row["sample_id"],
                    "activation": row["activation"],
                    "legacy_process_id": row["legacy_process_id"],
                    "conservative_unit_id": row["conservative_unit_id"],
                    "prior_retained_roles": ";".join(prior),
                    "prior_retained_sample_ids": ";".join(prior_samples),
                    "positive_area_pair_count": str(len(hits)),
                    "pairwise_shared_10m_grid_cells": str(
                        sum(int(item["intersection_grid_cells_10m"]) for item in hits)
                    ),
                    "spatial_evidence_file": OVERLAP_CSV.name,
                    "spatial_evidence_sha256": source_sha["spatial_overlap_audit"],
                }
            )
        retained_ids[role] = all_ids - excluded_ids

    output_files: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    role_report: dict[str, dict[str, object]] = {}
    for role in ROLE_ORDER:
        fields, derived = make_derived_rows(
            role, SOURCE_FILES[role], source[role][0], source[role][1], retained_ids[role]
        )
        filename = f"g3_rule_a_derived_candidate_{role}.csv"
        path = MANIFEST_DIR / filename
        write_csv(path, fields, derived)
        output_files[role] = filename
        output_hashes[role] = sha256(path)
        units = Counter(row["conservative_unit_id"] for row in derived)
        activations = Counter(row["activation"] for row in derived)
        if len(derived) != EXPECTED_COUNTS[role] or len(units) != EXPECTED_UNITS[role]:
            raise RuntimeError(f"Unexpected Rule-A count for {role}: {len(derived)} rows, {len(units)} units")
        if [int(row["derived_sample_order"]) for row in derived] != list(range(1, len(derived) + 1)):
            raise RuntimeError(f"Nonconsecutive derived sample order for {role}")
        role_report[role] = {
            "source_rows": len(source[role][1]),
            "retained_rows": len(derived),
            "excluded_rows": len(source[role][1]) - len(derived),
            "units": len(units),
            "activations": len(activations),
            "minimum_tiles_per_activation": min(activations.values()),
            "activations_at_minimum_tiles": sorted(
                activation for activation, count in activations.items() if count == min(activations.values())
            ),
        }

    exclusion_fields = [
        "derived_status",
        "derived_rule",
        "source_role",
        "source_candidate_filename",
        "source_candidate_line_number",
        "sample_id",
        "activation",
        "legacy_process_id",
        "conservative_unit_id",
        "prior_retained_roles",
        "prior_retained_sample_ids",
        "positive_area_pair_count",
        "pairwise_shared_10m_grid_cells",
        "spatial_evidence_file",
        "spatial_evidence_sha256",
    ]
    exclusion_path = MANIFEST_DIR / "g3_rule_a_derived_candidate_excluded_tiles.csv"
    write_csv(exclusion_path, exclusion_fields, excluded_rows)
    output_files["excluded_tiles"] = exclusion_path.name
    output_hashes["excluded_tiles"] = sha256(exclusion_path)
    if Counter(row["source_role"] for row in excluded_rows) != Counter(calibration=29, internal_retro=188):
        raise RuntimeError("Unexpected Rule-A exclusion-role counts")

    # Independent re-check on the full 509-row source audit: retained IDs must
    # have no exact spatial pair in any cross-role combination.
    id_overlap: dict[str, int] = {}
    spatial_overlap: dict[str, dict[str, int]] = {}
    for left_index, left_role in enumerate(ROLE_ORDER):
        for right_role in ROLE_ORDER[left_index + 1 :]:
            key = f"{left_role}__{right_role}"
            id_overlap[key] = len(retained_ids[left_role] & retained_ids[right_role])
            retained_pairs = []
            for row in overlap_rows:
                endpoints = {
                    row["role_a"]: row["sample_id_a"],
                    row["role_b"]: row["sample_id_b"],
                }
                if (
                    left_role in endpoints
                    and right_role in endpoints
                    and endpoints[left_role] in retained_ids[left_role]
                    and endpoints[right_role] in retained_ids[right_role]
                ):
                    retained_pairs.append(row)
            spatial_overlap[key] = {
                "positive_area_tile_pairs": len(retained_pairs),
                "pairwise_shared_10m_grid_cells": sum(
                    int(row["intersection_grid_cells_10m"]) for row in retained_pairs
                ),
            }
    if any(id_overlap.values()) or any(value["positive_area_tile_pairs"] for value in spatial_overlap.values()):
        raise RuntimeError(f"Rule-A isolation failure: IDs={id_overlap}; spatial={spatial_overlap}")

    metadata = {
        "status": STATUS,
        "purpose": "Derived retrospective candidate manifests under independently reviewed G3 Rule A",
        "rule": {
            "name": "whole_tile_exclude_if_shared_10m_cell_with_retained_prior_role",
            "role_order": list(ROLE_ORDER),
            "definition": "Fit is retained. Each later-role tile is excluded if the complete 509-row audited header-only overlap table records one or more shared 10 m cells with an already retained earlier-role tile.",
        },
        "read_boundary": {
            "manifest_and_audited_geometry_metadata_only": True,
            "imagery_pixels_read": False,
            "label_pixels_read": False,
            "predictions_or_scores_read": False,
            "dataset_reader_called": False,
        },
        "source_sha256": source_sha,
        "output_files": output_files,
        "output_sha256": output_hashes,
        "role_report": role_report,
        "cross_role_sample_id_intersections": id_overlap,
        "cross_role_spatial_recheck_from_exhaustive_header_audit": spatial_overlap,
        "unverified_boundaries": [
            "CEMS process-family/source relationship for the overlapping activation pairs remains UNVERIFIED.",
            "Per-tile CEMS label-product and label-reference-time linkage remains UNVERIFIED.",
            "This is retrospective candidate material only and is not authorization for training, calibration, scoring, or confirmation claims.",
        ],
        "reproduction": {
            "script": "scripts/derive_g3_rule_a_candidate.py",
            "script_sha256": sha256(Path(__file__).resolve()),
            "source_order": "Rows preserve source-candidate file order after deterministic whole-tile exclusion; derived_sample_order is consecutive within each role.",
            "unit_order": "derived_unit_order is the first-occurrence order in the retained source order; derived_within_unit_order is consecutive within that unit.",
        },
    }
    metadata_path = MANIFEST_DIR / "g3_rule_a_derived_candidate_meta.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"metadata": metadata_path.name, "metadata_sha256": sha256(metadata_path), **metadata}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
