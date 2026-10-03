"""Recover S1 event-time metadata for Rule-A holdout rows lacking S2-age anchors.

Allowed reads are TAR member names/offsets, each selected S1 Zarr ZIP's
``.zattrs`` text member, and the precomputed MASK TIFF-header grid index.  The
script never opens S1/S2/MASK array members, never reads label pixels, and never
uses predictions, scores, or a dataset reader.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MANIFESTS = ROOT / "data" / "manifests"
RAW = ROOT / "data" / "raw" / "impactmesh" / "test"
STATUS = "CANDIDATE_RETROSPECTIVE_ONLY_NOT_FROZEN"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    source_audit = DOCS / "G3_RULE_A_SOURCE_TIME_CHAIN_TILE_AUDIT.csv"
    derived = MANIFESTS / "g3_rule_a_derived_candidate_holdout_retro.csv"
    grid_index = MANIFESTS / "impactmesh_test_mask_grid_index.csv"
    s1_tar = RAW / "S1RTC.tar"
    mask_tar = RAW / "MASK.tar"
    s1_record = MANIFESTS / "impactmesh_archive_download_test_S1RTC.json"
    holdout_meta = MANIFESTS / "impactmesh_holdout_frozen_eligibility_meta.json"

    derived_by_id = {row["sample_id"]: row for row in read_csv(derived)}
    targets = [
        row
        for row in read_csv(source_audit)
        if row["derived_role"] == "holdout_retro" and row["s1_anchor_utc"] == "UNAVAILABLE"
    ]
    if len(targets) != 11:
        raise RuntimeError(f"Expected 11 unavailable holdout anchors, found {len(targets)}")
    target_ids = {row["sample_id"] for row in targets}
    if not target_ids <= set(derived_by_id):
        raise RuntimeError("Target ID absent from derived holdout candidate")
    grids = {row["sample_id"]: row for row in read_csv(grid_index)}
    if not target_ids <= set(grids):
        raise RuntimeError("Target ID absent from precomputed MASK grid index")

    s1_download = json.loads(s1_record.read_text(encoding="utf-8"))
    holdout = json.loads(holdout_meta.read_text(encoding="utf-8"))
    if s1_download["sha256"] != holdout["archive_expected_sha256"]["S1RTC"]:
        raise RuntimeError("Test S1 archive hash records disagree")

    rows: list[dict[str, str]] = []
    seen_s1: set[str] = set()
    with tarfile.open(s1_tar, "r:") as s1_archive, tarfile.open(mask_tar, "r:") as mask_archive:
        members = {member.name: member for member in s1_archive.getmembers() if member.isfile()}
        mask_members = {member.name: member for member in mask_archive.getmembers() if member.isfile()}
        for source in sorted(targets, key=lambda row: row["sample_id"]):
            sample_id = source["sample_id"]
            s1_name = f"S1RTC/{sample_id}_S1RTC.zarr.zip"
            mask_name = f"MASK/{sample_id}_annotation_flood.tif"
            s1_member = members.get(s1_name)
            mask_member = mask_members.get(mask_name)
            if s1_member is None or mask_member is None:
                raise RuntimeError(f"Missing matched S1/MASK TAR member for {sample_id}")
            # Read only the JSON attributes member; no S1 array member is opened.
            payload = s1_archive.extractfile(s1_member).read()
            with zipfile.ZipFile(io.BytesIO(payload)) as source_zip:
                attr_info = source_zip.getinfo(".zattrs")
                attrs_bytes = source_zip.read(".zattrs")
                attrs = json.loads(attrs_bytes)
            datetimes = attrs.get("datetime", "").split(";")
            if len(datetimes) != 4 or not datetimes[2]:
                raise RuntimeError(f"Unexpected datetime attribute for {sample_id}: {attrs.get('datetime')!r}")
            grid = grids[sample_id]
            candidate = derived_by_id[sample_id]
            rows.append(
                {
                    "candidate_status": STATUS,
                    "derived_role": candidate["derived_role"],
                    "conservative_unit_id": candidate["conservative_unit_id"],
                    "sample_id": sample_id,
                    "activation": candidate["activation"],
                    "s1_event_index": "2",
                    "s1_event_utc_from_zattrs": datetimes[2],
                    "s1_zattrs_datetime_all_four": attrs["datetime"],
                    "s1_zattrs_member_path": ".zattrs",
                    "s1_zattrs_member_sha256": hashlib.sha256(attrs_bytes).hexdigest(),
                    "s1_zattrs_zip_member_crc32": str(attr_info.CRC),
                    "s1_tar_member_path": s1_name,
                    "s1_tar_member_offset_data": str(s1_member.offset_data),
                    "s1_tar_member_size_bytes": str(s1_member.size),
                    "s1_tar_sha256": s1_download["sha256"],
                    "s1_tar_download_record": "data/manifests/impactmesh_archive_download_test_S1RTC.json",
                    "mask_tar_member_path": mask_name,
                    "mask_tar_member_offset_data": str(mask_member.offset_data),
                    "mask_tar_member_size_bytes": str(mask_member.size),
                    "mask_tar_sha256": holdout["archive_expected_sha256"]["MASK"],
                    "mask_grid_index_path": "data/manifests/impactmesh_test_mask_grid_index.csv",
                    "mask_grid_index_sha256": sha256(grid_index),
                    "mask_grid_crs": grid["grid_crs"],
                    "mask_grid_left": grid["left"],
                    "mask_grid_bottom": grid["bottom"],
                    "mask_grid_right": grid["right"],
                    "mask_grid_top": grid["top"],
                    "s1_to_label_grid_anchor_status": "S1 and native MASK have matched sample ID; MASK grid comes from precomputed TIFF-header index. This identifies grid association only, not CEMS product provenance.",
                    "label_product_reference_time_status": "UNVERIFIED: S1 .zattrs does not name a CEMS product ID, AOI, reference image or label production time.",
                }
            )
            seen_s1.add(sample_id)
    if seen_s1 != target_ids:
        raise RuntimeError("Incomplete selected S1 attribute audit")

    output = DOCS / "G3_RULE_A_HOLDOUT_S1_ANCHOR_METADATA_AUDIT.csv"
    write_csv(output, list(rows[0]), rows)
    summary = {
        "status": STATUS,
        "purpose": "Metadata-only direct S1 event-anchor recovery for 11 Rule-A holdout rows without frozen S2/age anchors",
        "read_boundary": {
            "s1_s2_mask_array_values_read": False,
            "label_pixels_read": False,
            "prediction_or_score_read": False,
            "dataset_reader_called": False,
            "allowed_metadata": ["TAR member names/offsets", "S1 ZIP .zattrs", "precomputed MASK TIFF-header grid index"],
        },
        "counts": {"target_rows": len(rows), "direct_s1_event_timestamps_recovered": len(rows), "confirmed_cems_product_mapping": 0},
        "source_sha256": {
            "docs/G3_RULE_A_SOURCE_TIME_CHAIN_TILE_AUDIT.csv": sha256(source_audit),
            "data/manifests/g3_rule_a_derived_candidate_holdout_retro.csv": sha256(derived),
            "data/manifests/impactmesh_test_mask_grid_index.csv": sha256(grid_index),
            "data/manifests/impactmesh_archive_download_test_S1RTC.json": sha256(s1_record),
            "data/manifests/impactmesh_holdout_frozen_eligibility_meta.json": sha256(holdout_meta),
        },
        "output_sha256": {output.name: sha256(output)},
        "conclusion": "All 11 direct S1 event timestamps are recoverable from the selected S1 ZIP .zattrs datetime attribute at event index 2. The matched MASK sample IDs and precomputed grid headers provide a grid association. Neither source supplies a tile-to-CEMS-product mapping or label reference/acquisition time; those remain UNVERIFIED.",
        "reproduction_script": "scripts/audit_g3_holdout_s1_anchor_metadata.py",
        "reproduction_script_sha256": sha256(Path(__file__).resolve()),
    }
    summary_path = DOCS / "G3_RULE_A_HOLDOUT_S1_ANCHOR_METADATA_AUDIT.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary_sha256": sha256(summary_path), **summary}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
