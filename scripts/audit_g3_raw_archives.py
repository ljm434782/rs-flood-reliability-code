"""Verify every ImpactMesh train/test raw archive against publisher LFS SHA-256."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET_INDEX = ROOT / "data" / "manifests" / "impactmesh_assets.json"
OUTPUT = ROOT / "docs" / "G3_RAW_ARCHIVE_CHECKSUM_AUDIT.json"
TARGETS = ("train/MASK.tar", "train/S1RTC.tar", "train/S2L2A.tar",
           "test/MASK.tar", "test/S1RTC.tar", "test/S2L2A.tar")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit() -> dict:
    source = json.loads(ASSET_INDEX.read_text(encoding="utf-8"))
    by_path = {entry["path"]: entry for entry in source["assets"] if entry.get("path") in TARGETS}
    if set(by_path) != set(TARGETS):
        raise RuntimeError("publisher asset manifest does not enumerate all six archives")
    records = []
    mismatches = []
    for relative in TARGETS:
        entry = by_path[relative]
        lfs = entry.get("lfs", {})
        raw = ROOT / "data" / "raw" / "impactmesh" / relative
        if not raw.is_file():
            mismatches.append({"path": relative, "error": "missing"})
            continue
        actual_size = raw.stat().st_size
        actual_sha = file_sha256(raw)
        record = {"path": f"data/raw/impactmesh/{relative}", "size_bytes": actual_size,
                  "expected_size_bytes": int(lfs.get("size", entry.get("size", -1))),
                  "sha256": actual_sha, "publisher_lfs_sha256": lfs.get("oid"),
                  "publisher_git_oid": entry.get("oid")}
        records.append(record)
        print(json.dumps({"checked": relative, "size_bytes": actual_size,
                          "sha256": actual_sha}, sort_keys=True), flush=True)
        if (actual_sha != lfs.get("oid") or actual_size != int(lfs.get("size", entry.get("size", -1)))):
            mismatches.append(record)
    report = {"status": "PASS" if not mismatches and len(records) == len(TARGETS) else "FAIL",
              "checked_utc": datetime.now(timezone.utc).isoformat(),
              "source_repository": source["repository"], "source_version": "ImpactMesh-Flood v1",
              "license": "CC-BY-4.0", "publisher_asset_manifest": "data/manifests/impactmesh_assets.json",
              "publisher_asset_manifest_sha256": file_sha256(ASSET_INDEX),
              "raw_archives_overwritten": False, "archives": records, "mismatches": mismatches}
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps({"status": report["status"], "report": str(OUTPUT),
                      "records": len(records), "mismatches": len(mismatches)}, indent=2), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError("One or more raw archives differ from publisher records")
    return report


if __name__ == "__main__":
    audit()
