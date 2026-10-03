"""Build a metadata-only source and label-time audit for Rule-A candidates.

This program intentionally reads CSV/JSON/text manifests only.  It never opens
an image or label raster, never invokes a dataset reader, and never accesses a
prediction, score, calibration result, or candidate manifest for writing.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
DOCS = ROOT / "docs"
STATUS = "CANDIDATE_RETROSPECTIVE_ONLY_NOT_FROZEN"
ROLE_FILES = {
    "fit": "g3_rule_a_derived_candidate_fit.csv",
    "calibration": "g3_rule_a_derived_candidate_calibration.csv",
    "internal_retro": "g3_rule_a_derived_candidate_internal_retro.csv",
    "holdout_retro": "g3_rule_a_derived_candidate_holdout_retro.csv",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def fmt_utc(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def joined(values: list[str]) -> str:
    return ";".join(value for value in values if value)


def main() -> None:
    role_rows = {role: read_csv(MANIFESTS / filename) for role, filename in ROLE_FILES.items()}
    if any(not rows for rows in role_rows.values()):
        raise RuntimeError("A Rule-A role manifest is empty")
    if any(row["derived_status"] != STATUS for rows in role_rows.values() for row in rows):
        raise RuntimeError("Rule-A derived source status mismatch")

    train_event_path = MANIFESTS / "impactmesh_train_event_times.csv.gz"
    train_event = {row["sample_id"]: row["s1_event_utc"] for row in read_gzip_csv(train_event_path)}
    holdout_eligibility_path = MANIFESTS / "impactmesh_holdout_frozen_eligibility.csv"
    holdout_eligibility = {row["sample_id"]: row for row in read_csv(holdout_eligibility_path)}
    cems_index_path = MANIFESTS / "cems_index.csv"
    cems_products: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(cems_index_path):
        cems_products[row["activation"]].append(row)
    label_audit_path = MANIFESTS / "impactmesh_v1_label_time_chain_audit.csv"
    label_audit = {row["activation"]: row for row in read_csv(label_audit_path)}

    train_s1 = json.loads((MANIFESTS / "impactmesh_archive_download_S1RTC.json").read_text(encoding="utf-8"))
    train_s2 = json.loads((MANIFESTS / "impactmesh_archive_download_S2L2A.json").read_text(encoding="utf-8"))
    test_s1 = json.loads((MANIFESTS / "impactmesh_archive_download_test_S1RTC.json").read_text(encoding="utf-8"))
    test_s2 = json.loads((MANIFESTS / "impactmesh_archive_download_test_S2L2A.json").read_text(encoding="utf-8"))
    holdout_meta = json.loads((MANIFESTS / "impactmesh_holdout_frozen_eligibility_meta.json").read_text(encoding="utf-8"))

    # The published repository inventory includes README/splits/TAR assets but
    # no tile-to-CEMS-product mapping file.  The README is a published asset
    # with Git object ID recorded in this immutable local inventory.
    assets_path = MANIFESTS / "impactmesh_assets.json"
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    asset_paths = {item["path"] for item in assets["assets"]}
    if any("product" in item.lower() or "cems" in item.lower() for item in asset_paths):
        raise RuntimeError("Published asset inventory unexpectedly contains a CEMS/product mapping asset")
    readme_oid = next(item["oid"] for item in assets["assets"] if item["path"] == "README.md")

    tile_rows: list[dict[str, str]] = []
    for role, rows in role_rows.items():
        is_train = role != "holdout_retro"
        for row in rows:
            activation = row["activation"]
            if is_train:
                if row["sample_id"] not in train_event:
                    raise RuntimeError(f"Missing train S1 time: {row['sample_id']}")
                s1_anchor = train_event[row["sample_id"]]
                s1_provenance = "impactmesh_train_event_times.csv.gz:s1_event_utc"
                input_split = "train"
                s1_archive = train_s1
                s2_archive = train_s2
                mask_sha = "dc9f2003ac8a1e55511d98f00564f5d967564deb8be41d4270a55192a53c1876"
                mask_evidence = "impactmesh_acquisition_feasibility.json:train/MASK.tar"
                selected_s2_utc = row["selected_s2_utc"]
                selected_s2_age_days = row["selected_age_days"]
                holdout_paired_eligibility = "NOT_APPLICABLE"
            else:
                eligible = holdout_eligibility.get(row["sample_id"])
                if eligible is None:
                    raise RuntimeError(f"Missing holdout metadata: {row['sample_id']}")
                # No direct per-tile test S1-time manifest is available.  This
                # reconstructs the anchor from the frozen selected-S2 UTC and
                # age field and labels that provenance explicitly.
                selected_s2_utc = eligible["selected_s2_utc"]
                selected_s2_age_days = eligible["selected_age_days"]
                holdout_paired_eligibility = eligible["paired_eligible"]
                if selected_s2_utc and selected_s2_age_days:
                    s1_anchor = fmt_utc(
                        parse_utc(selected_s2_utc) + timedelta(days=float(selected_s2_age_days))
                    )
                    s1_provenance = "INFERRED: impactmesh_holdout_frozen_eligibility.csv:selected_s2_utc + selected_age_days; no direct test S1-time manifest"
                else:
                    s1_anchor = "UNAVAILABLE"
                    s1_provenance = "UNVERIFIED: no direct test S1-time manifest, and frozen holdout eligibility has no selected S2/age from which to infer this tile's S1 anchor"
                input_split = "test_holdout"
                s1_archive = test_s1
                s2_archive = test_s2
                mask_sha = holdout_meta["archive_expected_sha256"]["MASK"]
                mask_evidence = "impactmesh_holdout_frozen_eligibility_meta.json:archive_expected_sha256.MASK"

            products = cems_products.get(activation, [])
            product_ids = [item["product"] for item in products]
            product_times = [item["source_image_time_utc"] for item in products]
            product_paths = [item["local_path"] for item in products]
            product_hashes = [item["sha256"] for item in products]
            product_statuses = [item["status"] for item in products]
            offsets: list[str] = []
            for item in products:
                if item["source_image_time_utc"] and s1_anchor != "UNAVAILABLE":
                    hours = (parse_utc(item["source_image_time_utc"]) - parse_utc(s1_anchor)).total_seconds() / 3600
                    offsets.append(f"{item['product']}:{hours:.6f}")
            if products:
                mapping_status = "UNVERIFIED: activation-level CEMS product candidates exist, but no tile-to-product ID/AOI mapping is published locally"
            else:
                mapping_status = "UNVERIFIED: no local CEMS product record for activation and no tile-to-product mapping is published locally"
            if role == "internal_retro":
                label_status = "UNVERIFIED: prior v1 audit classifies this internal activation as unable_to_confirm"
            elif role == "holdout_retro":
                prior = label_audit.get(activation)
                if prior is None:
                    label_status = "UNVERIFIED: activation absent from prior v1 label-time audit"
                else:
                    label_status = f"UNVERIFIED: prior v1 audit evidence note: {prior['evidence_note']}"
            else:
                label_status = "UNVERIFIED: no per-tile CEMS product/reference-time mapping in local published inventory"

            tile_rows.append(
                {
                    "candidate_status": STATUS,
                    "derived_role": role,
                    "conservative_unit_id": row["conservative_unit_id"],
                    "sample_id": row["sample_id"],
                    "activation": activation,
                    "s1_anchor_utc": s1_anchor,
                    "s1_anchor_provenance": s1_provenance,
                    "selected_s2_utc": selected_s2_utc,
                    "selected_s2_age_days": selected_s2_age_days,
                    "holdout_paired_eligibility": holdout_paired_eligibility,
                    "input_split": input_split,
                    "s1_input_archive_path": s1_archive["path"],
                    "s1_input_archive_sha256": s1_archive["sha256"],
                    "s2_input_archive_path": s2_archive["path"],
                    "s2_input_archive_sha256": s2_archive["sha256"],
                    "native_label_archive_path": f"{input_split.split('_')[0]}/MASK.tar",
                    "native_label_archive_sha256": mask_sha,
                    "native_label_archive_evidence": mask_evidence,
                    "dataset_repository": s1_archive["repository"],
                    "dataset_version": s1_archive["version"],
                    "dataset_license": s1_archive["license"],
                    "dataset_license_evidence": f"data/manifests/{'impactmesh_archive_download_S1RTC.json' if is_train else 'impactmesh_archive_download_test_S1RTC.json'}",
                    "cems_product_ids_activation_level_only": joined(product_ids) or "UNMAPPED",
                    "cems_reference_image_times_utc_activation_level_only": joined(product_times) or "UNMAPPED",
                    "cems_reference_minus_s1_hours_activation_level_only": joined(offsets) or "UNMAPPED",
                    "cems_evidence_local_paths": joined(product_paths) or "NONE",
                    "cems_evidence_sha256": joined(product_hashes) or "NONE",
                    "cems_evidence_status": joined(product_statuses) or "NONE",
                    "cems_product_license": "UNVERIFIED in local product evidence; ImpactMesh dataset CC-BY-4.0 does not identify a tile-level CEMS product licence record",
                    "tile_to_cems_product_mapping_status": mapping_status,
                    "label_reference_time_status": label_status,
                    "published_mapping_absence_evidence": "impactmesh_assets.json contains README/splits/TAR assets only; README states MASK annotations from CEMS but no per-tile CEMS product-ID field",
                }
            )

    tile_fields = list(tile_rows[0])
    tile_path = DOCS / "G3_RULE_A_SOURCE_TIME_CHAIN_TILE_AUDIT.csv"
    write_csv(tile_path, tile_fields, tile_rows)

    unit_rows: list[dict[str, str]] = []
    by_unit: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in tile_rows:
        by_unit[(row["derived_role"], row["conservative_unit_id"])].append(row)
    for (role, unit), items in sorted(by_unit.items()):
        anchors = sorted({item["s1_anchor_utc"] for item in items if item["s1_anchor_utc"] != "UNAVAILABLE"})
        products = sorted(
            {
                item["cems_product_ids_activation_level_only"]
                for item in items
                if item["cems_product_ids_activation_level_only"] != "UNMAPPED"
            }
        )
        unit_rows.append(
            {
                "candidate_status": STATUS,
                "derived_role": role,
                "conservative_unit_id": unit,
                "activation_ids": joined(sorted({item["activation"] for item in items})),
                "tile_count": str(len(items)),
                "s1_anchor_first_utc": anchors[0] if anchors else "UNAVAILABLE",
                "s1_anchor_last_utc": anchors[-1] if anchors else "UNAVAILABLE",
                "s1_anchor_provenance": joined(sorted({item["s1_anchor_provenance"] for item in items})),
                "cems_product_candidates_activation_level_only": joined(products) or "UNMAPPED",
                "tile_product_mapping_status": "UNVERIFIED for every tile in unit",
                "label_reference_time_status": "UNVERIFIED for every tile in unit",
                "dataset_license": "CC-BY-4.0",
                "evidence_paths": joined(
                    sorted(
                        {
                            item["dataset_license_evidence"]
                            for item in items
                        }
                    )
                ),
            }
        )
    unit_path = DOCS / "G3_RULE_A_SOURCE_TIME_CHAIN_UNIT_AUDIT.csv"
    write_csv(unit_path, list(unit_rows[0]), unit_rows)

    source_paths = [
        *(MANIFESTS / filename for filename in ROLE_FILES.values()),
        train_event_path,
        holdout_eligibility_path,
        cems_index_path,
        label_audit_path,
        assets_path,
        MANIFESTS / "impactmesh_archive_download_S1RTC.json",
        MANIFESTS / "impactmesh_archive_download_S2L2A.json",
        MANIFESTS / "impactmesh_archive_download_test_S1RTC.json",
        MANIFESTS / "impactmesh_archive_download_test_S2L2A.json",
        MANIFESTS / "impactmesh_holdout_frozen_eligibility_meta.json",
    ]
    summary = {
        "status": STATUS,
        "purpose": "Bounded metadata-only source/product/time-chain audit for Rule-A candidate rows and units",
        "read_boundary": {
            "training_or_holdout_image_values_read": False,
            "training_or_holdout_label_pixels_read": False,
            "prediction_or_score_read": False,
            "dataset_reader_called": False,
        },
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in source_paths},
        "output_sha256": {
            tile_path.name: sha256(tile_path),
            unit_path.name: sha256(unit_path),
        },
        "counts": {
            "candidate_tiles": len(tile_rows),
            "candidate_units": len(unit_rows),
            "train_role_tiles_with_direct_s1_anchor": sum(
                item["derived_role"] != "holdout_retro" and item["s1_anchor_utc"] != "UNAVAILABLE"
                for item in tile_rows
            ),
            "holdout_tiles_with_inferred_s1_anchor": sum(
                item["derived_role"] == "holdout_retro"
                and item["s1_anchor_utc"] != "UNAVAILABLE"
                for item in tile_rows
            ),
            "holdout_tiles_with_unavailable_s1_anchor": sum(
                item["derived_role"] == "holdout_retro"
                and item["s1_anchor_utc"] == "UNAVAILABLE"
                for item in tile_rows
            ),
            "holdout_tiles_paired_eligible": sum(
                item["derived_role"] == "holdout_retro" and item["holdout_paired_eligibility"] == "1"
                for item in tile_rows
            ),
            "holdout_tiles_not_paired_eligible": sum(
                item["derived_role"] == "holdout_retro" and item["holdout_paired_eligibility"] == "0"
                for item in tile_rows
            ),
            "tiles_with_confirmed_tile_to_cems_product_mapping": 0,
            "tiles_with_unverified_tile_to_cems_product_mapping": len(tile_rows),
            "tiles_with_any_activation_level_local_cems_product_candidate": sum(
                item["cems_product_ids_activation_level_only"] != "UNMAPPED" for item in tile_rows
            ),
            "activations_with_any_local_cems_product_candidate": sorted(cems_products.keys() & {item["activation"] for item in tile_rows}),
            "candidate_activation_count": len({item["activation"] for item in tile_rows}),
        },
        "published_mapping_evidence": {
            "impactmesh_published_asset_inventory": "data/manifests/impactmesh_assets.json",
            "readme_git_object_id": readme_oid,
            "official_dataset_card_url": "https://huggingface.co/datasets/ibm-esa-geospatial/ImpactMesh-Flood/blob/main/README.md",
            "observed_dataset_card_scope": "README describes MASK.tar as flood annotations from Copernicus EMS and lists split TAR assets; it does not provide a sample-ID-to-CEMS-product-ID/AOI/reference-time table.",
            "official_cems_api_documentation_url": "https://mapping.emergency.copernicus.eu/about/how-to-harvest-cems-mapping-data/",
            "cems_api_scope": "CEMS product metadata can enumerate activation/AOI products, but that does not identify which product ImpactMesh used for a given tile.",
        },
        "independent_holdout_assessment": {
            "cleared_real_independent_holdout_processes": 0,
            "basis": "Rule-A official holdout is explicitly retrospective; its nine activations lack confirmed per-tile label product/time chains, and existing alternative-source audits have cleared no new independent multi-process G3 source.",
        },
        "conclusion": "No candidate tile has a confirmed CEMS product ID or reference/acquisition time attributable to its native ImpactMesh label. Dataset-level input source/version/licence and S1-anchor evidence are recorded, but label-time and product provenance remain UNVERIFIED.",
        "reproduction_script": "scripts/audit_g3_rule_a_source_time.py",
        "reproduction_script_sha256": sha256(Path(__file__).resolve()),
    }
    summary_path = DOCS / "G3_RULE_A_SOURCE_TIME_CHAIN_AUDIT.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary_sha256": sha256(summary_path), **summary}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
