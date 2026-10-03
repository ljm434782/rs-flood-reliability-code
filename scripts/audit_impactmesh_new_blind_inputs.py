"""Apply frozen input-only rules to six prequalified official-val flood events."""

import csv
import json
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from rasterio.io import MemoryFile

from flood_reliability.impactmesh_io import TrainArchiveReader, _group
from scripts.audit_impactmesh_holdout_eligibility import grid_matches, timestamp
from scripts.train_impactmesh_formal import ROOT, sha256

MAN = ROOT / "data/manifests"
RAW = ROOT / "data/raw/impactmesh/val"
SOURCE = MAN / "impactmesh_new_blind_source_inventory.csv"
OUTPUT = MAN / "impactmesh_new_blind_input_eligibility.csv"
LOCKED = MAN / "impactmesh_new_blind_locked_tiles.csv"
META = MAN / "impactmesh_new_blind_locked_tiles.json"
PREQUALIFIED = ("EMSR410", "EMSR411", "EMSR496", "EMSR559", "EMSR788", "EMSR843")


def main():
    if any(path.exists() for path in (OUTPUT, LOCKED, META)):
        raise RuntimeError("Refusing to overwrite blind input audit")
    source = {row["activation"]: row for row in csv.DictReader(SOURCE.open(newline="", encoding="utf-8"))}
    if set(PREQUALIFIED) - source.keys():
        raise RuntimeError("Prequalified event missing from input-only source inventory")
    assets = {item["path"]: item for item in json.loads((MAN / "impactmesh_assets.json").read_text(
        encoding="utf-8"))["assets"]}
    for modality in ("S1RTC", "S2L2A", "MASK"):
        asset = assets[f"val/{modality}.tar"]
        path = RAW / f"{modality}.tar"
        if path.stat().st_size != int(asset["size"]) or sha256(path) != asset["lfs"]["oid"]:
            raise RuntimeError(f"Official val raw SHA mismatch: {modality}")
    official = [line for line in (MAN / "impactmesh_flood_val.txt").read_text(
        encoding="utf-8").splitlines() if line.split("_", 1)[0] in PREQUALIFIED]
    if len(official) != 202:
        raise RuntimeError(f"Prequalified source tile count changed: {len(official)}")
    rows = []
    with TrainArchiveReader(RAW) as reader:
        if not set(official) <= set(reader.index["S1RTC"]):
            raise RuntimeError("Official val IDs missing from raw archive")
        for position, sample_id in enumerate(sorted(official), 1):
            s1 = _group(reader.blob("S1RTC", sample_id))
            s2 = _group(reader.blob("S2L2A", sample_id))
            with MemoryFile(reader.blob("MASK", sample_id)) as memory:
                with memory.open() as label_source:
                    if label_source.shape != (256, 256) or label_source.count != 1:
                        raise RuntimeError(f"Label-grid shape changed: {sample_id}")
                    s1_grid = grid_matches(s1, label_source.crs, label_source.transform)
                    s2_grid = grid_matches(s2, label_source.crs, label_source.transform)
            s1_times = s1.attrs["datetime"].split(";")
            s2_times = s2.attrs["datetime"].split(";")
            if len(s1_times) != 4 or len(s2_times) != 4:
                raise RuntimeError(f"Unexpected acquisition time count: {sample_id}")
            event_time = timestamp(s1_times[2])
            sar = np.asarray(s1["bands"][2], dtype=np.float32)
            sar_good = (int(s1["data_mask"][2]) == 1
                        and not np.any(s1["nan_mask"][2] != 0)
                        and sar.shape == (2, 256, 256) and np.isfinite(sar).all())
            candidates = []
            for index, value in enumerate(s2_times):
                age = (event_time - timestamp(value)).total_seconds() / 86400
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
            rows.append({"sample_id": sample_id, "activation": sample_id.split("_", 1)[0],
                "event_s1_utc": s1_times[2], "paired_eligible": int(not reasons),
                "selected_time_index": "" if best is None else best[2],
                "selected_s2_utc": "" if best is None else best[3],
                "selected_age_days": "" if best is None else best[4],
                "selected_joint_visible_fraction": "" if best is None else best[0],
                "failure_reasons": ";".join(reasons)})
            if position % 25 == 0:
                print(f"New blind input-only audit {position}/{len(official)}", flush=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["activation"] for row in rows if row["paired_eligible"])
    locked = [row for row in rows if row["paired_eligible"] and counts[row["activation"]] >= 5]
    with LOCKED.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(locked)
    META.write_text(json.dumps({"completed_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "source-prequalified val process blind panel, input-only lock",
        "source_inventory_sha256": sha256(SOURCE),
        "official_val_ids_sha256": sha256(MAN / "impactmesh_flood_val.txt"),
        "raw_sha256": {modality: assets[f"val/{modality}.tar"]["lfs"]["oid"]
                        for modality in ("S1RTC", "S2L2A", "MASK")},
        "prequalified_events": list(PREQUALIFIED), "prequalified_tiles": len(official),
        "eligible_counts_before_process_minimum": dict(sorted(counts.items())),
        "locked_events": sorted(set(row["activation"] for row in locked)),
        "locked_tiles": len(locked), "eligibility_sha256": sha256(OUTPUT),
        "locked_manifest_sha256": sha256(LOCKED), "label_pixels_read": False,
        "model_outputs_read": False, "event_selection_using_label_distribution": False},
        indent=2) + "\n", encoding="utf-8")
    print(META.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
