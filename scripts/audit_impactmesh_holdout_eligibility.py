"""Apply already-frozen pairing criteria to official holdout inputs, label-grid only."""

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rasterio.crs import CRS
from rasterio.io import MemoryFile

from flood_reliability.impactmesh_io import TrainArchiveReader, _group
from scripts.calibrate_impactmesh_formal import ROOT, sha256


MANIFESTS = ROOT / "data" / "manifests"
RAW = ROOT / "data" / "raw" / "impactmesh" / "test"
OUTPUT = MANIFESTS / "impactmesh_holdout_frozen_eligibility.csv"
META = MANIFESTS / "impactmesh_holdout_frozen_eligibility_meta.json"


def timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def grid_matches(group, crs, transform) -> bool:
    x = transform.c + (np.arange(256) + .5) * transform.a
    y = transform.f + (np.arange(256) + .5) * transform.e
    return (CRS.from_wkt(group["spatial_ref"].attrs["crs_wkt"]) == crs
            and np.allclose(group["x"][:], x, atol=1e-6, rtol=0)
            and np.allclose(group["y"][:], y, atol=1e-6, rtol=0))


def main():
    assets = {item["path"]: item for item in json.loads(
        (MANIFESTS / "impactmesh_assets.json").read_text(encoding="utf-8"))["assets"]}
    for modality in ("S1RTC", "S2L2A", "MASK"):
        path = RAW / f"{modality}.tar"
        if path.stat().st_size != int(assets[f"test/{modality}.tar"]["size"]):
            raise RuntimeError(f"Holdout archive size changed: {path}")
    official = set((MANIFESTS / "impactmesh_flood_test_holdout.txt").read_text(encoding="utf-8").splitlines())
    if len(official) != 562:
        raise RuntimeError("Official holdout ID list changed")
    rows = []
    with TrainArchiveReader(RAW) as reader:
        archive_ids = set(reader.index["S1RTC"])
        if len(archive_ids) != 2959 or not official <= archive_ids:
            raise RuntimeError("Official test archive or holdout subset changed")
        for position, sample_id in enumerate(sorted(official), 1):
            s1 = _group(reader.blob("S1RTC", sample_id))
            s2 = _group(reader.blob("S2L2A", sample_id))
            with MemoryFile(reader.blob("MASK", sample_id)) as memory:
                with memory.open() as source:
                    if source.shape != (256, 256) or source.count != 1:
                        raise RuntimeError(f"Holdout label grid shape changed: {sample_id}")
                    s1_grid = grid_matches(s1, source.crs, source.transform)
                    s2_grid = grid_matches(s2, source.crs, source.transform)
            s1_times = s1.attrs["datetime"].split(";")
            s2_times = s2.attrs["datetime"].split(";")
            if len(s1_times) != 4 or len(s2_times) != 4:
                raise RuntimeError(f"Unexpected holdout times: {sample_id}")
            event = timestamp(s1_times[2])
            sar = np.asarray(s1["bands"][2], dtype=np.float32)
            sar_good = (int(s1["data_mask"][2]) == 1
                        and not np.any(s1["nan_mask"][2] != 0)
                        and sar.shape == (2, 256, 256) and np.isfinite(sar).all())
            candidates = []
            for index, value in enumerate(s2_times):
                age = (event - timestamp(value)).total_seconds() / 86400
                if not 0 <= age <= 30:
                    continue
                optical = np.asarray(s2["bands"][index], dtype=np.float32)
                valid = (np.isin(np.asarray(s2["SCL"][index]), (4, 5, 6))
                         & (np.asarray(s2["cloud_mask"][index]) == 0)
                         & (np.asarray(s2["nan_mask"][index]) == 0)
                         & np.all(np.isfinite(optical) & (optical != -9999), axis=0)
                         & (int(s2["data_mask"][index]) == 1))
                candidates.append((float(valid.mean()), -age, index, value, age))
            best = max(candidates) if candidates else None
            reasons = []
            if not sar_good:
                reasons.append("invalid_event_s1")
            if not s1_grid or not s2_grid:
                reasons.append("grid_mismatch")
            if best is None:
                reasons.append("no_pre_s1_s2_within_30d")
            elif best[0] < .5:
                reasons.append("visible_below_0.5")
            rows.append({
                "sample_id": sample_id, "activation": sample_id.split("_", 1)[0],
                "paired_eligible": int(not reasons), "selected_time_index": "" if best is None else best[2],
                "selected_s2_utc": "" if best is None else best[3],
                "selected_age_days": "" if best is None else best[4],
                "selected_joint_visible_fraction": "" if best is None else best[0],
                "failure_reasons": ";".join(reasons),
            })
            if position % 100 == 0:
                print(f"Frozen holdout input audit: {position}/{len(official)}", flush=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["activation"] for row in rows if row["paired_eligible"])
    META.write_text(json.dumps({
        "purpose": "frozen 30d/50% rules applied to locked official holdout inputs",
        "official_holdout_tiles": len(official), "paired_eligible_tiles": sum(r["paired_eligible"] for r in rows),
        "activation_eligible_counts": dict(sorted(counts.items())),
        "paired_eligible_activations_at_least_5": sum(n >= 5 for n in counts.values()),
        "test_label_pixels_read": False, "test_quality_threshold_chosen": False,
        "archive_expected_sha256": {modality: assets[f"test/{modality}.tar"]["lfs"]["oid"]
                                    for modality in ("S1RTC", "S2L2A", "MASK")},
        "official_ids_sha256": sha256(MANIFESTS / "impactmesh_flood_test_holdout.txt"),
        "output_sha256": sha256(OUTPUT), "completed_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(META.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
