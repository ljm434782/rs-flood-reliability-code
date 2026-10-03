"""Inspect an ImpactMesh S1/S2/label triplet without model selection."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import zarr


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "raw" / "impactmesh" / "samples"
OUTPUT = ROOT / "data" / "manifests" / "impactmesh_sample_audit.json"


def counts(values: np.ndarray) -> dict[str, int]:
    unique, n = np.unique(values, return_counts=True)
    return {str(key): int(value) for key, value in zip(unique.tolist(), n.tolist())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id")
    args = parser.parse_args()
    with (ROOT / "data" / "manifests" / "impactmesh_sample_manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    sample_id = args.sample_id or next(row["sample_id"] for row in manifest if row["tar_path"].startswith("val/"))
    folder = SAMPLES / sample_id
    source = next(row for row in manifest if row["sample_id"] == sample_id)
    split = source.get("split", "val")
    output = OUTPUT if split == "val" else OUTPUT.with_name(f"impactmesh_sample_audit_{sample_id}.json")
    frame_path = ROOT / "data" / "manifests" / "impactmesh_temporal_audit_frame.csv"
    frame_rank = None
    if frame_path.exists():
        with frame_path.open(newline="", encoding="utf-8") as handle:
            frame_row = next((row for row in csv.DictReader(handle) if row["sample_id"] == sample_id), None)
        if frame_row is not None:
            if split != "train" or int(source["tar_ordinal"]) != int(frame_row["train_mask_tar_ordinal"]):
                raise RuntimeError("Frozen temporal-audit frame disagrees with downloaded sample manifest")
            frame_rank = int(frame_row["rank"])
    result = {
        "sample_id": folder.name,
        "split": split,
        "selection": (
            f"fixed train-only temporal-audit frame rank {frame_rank}; member ordinal {source.get('tar_ordinal', '0')} in {source['tar_path']}"
            if frame_rank is not None else
            f"member ordinal {source.get('tar_ordinal', '0')} in {source['tar_path']}; convenience sample only"
        ),
    }
    timestamps = {}
    coordinates = {}
    for modality in ("S1RTC", "S2L2A"):
        path = folder / f"{folder.name}_{modality}.zarr.zip"
        group = zarr.open_consolidated(path, mode="r")
        bands = group["bands"]
        attrs = dict(group.attrs)
        times = attrs["datetime"].split(";")
        timestamps[modality] = [datetime.fromisoformat(value.replace("Z", "+00:00")) for value in times]
        coordinates[modality] = (group["x"][:], group["y"][:])
        event_band = bands[2, 0]
        finite = event_band[np.isfinite(event_band)]
        entry = {
            "bands_shape": list(bands.shape),
            "band_names": group["band"][:].tolist(),
            "timestamps_utc": times,
            "data_mask_values": group["data_mask"][:].tolist(),
            "nan_mask_value_counts_by_time": [counts(group["nan_mask"][index]) for index in range(4)],
            "event_index_2_band0_finite_fraction": float(np.isfinite(event_band).mean()),
            "event_index_2_band0_quantiles_finite": np.quantile(finite, [0, 0.5, 0.99, 1]).tolist(),
        }
        if modality == "S2L2A":
            entry["cloud_mask_attributes"] = dict(group["cloud_mask"].attrs)
            entry["cloud_mask_value_counts_by_time"] = [counts(group["cloud_mask"][index]) for index in range(4)]
            entry["SCL_value_counts_by_time"] = [counts(group["SCL"][index]) for index in range(4)]
            entry["scene_cloud_metadata"] = {
                name: attrs.get(name, "").split(";")
                for name in (
                    "s2_high_proba_clouds_percentage",
                    "s2_medium_proba_clouds_percentage",
                    "s2_cloud_shadow_percentage",
                    "s2_nodata_pixel_percentage",
                )
            }
        result[modality] = entry
    result["s2_minus_s1_days_by_index"] = [
        (s2 - s1).total_seconds() / 86400
        for s1, s2 in zip(timestamps["S1RTC"], timestamps["S2L2A"])
    ]
    x1, y1 = coordinates["S1RTC"]
    x2, y2 = coordinates["S2L2A"]
    result["s1_s2_coordinates_match"] = bool(np.array_equal(x1, x2) and np.array_equal(y1, y2))
    with rasterio.open(folder / f"{folder.name}_annotation_flood.tif") as label:
        values = label.read(1)
        if label.transform.b != 0 or label.transform.d != 0:
            raise RuntimeError("Sample label has a rotated grid; extend the coordinate check")
        expected_x = label.transform.c + (np.arange(label.width) + 0.5) * label.transform.a
        expected_y = label.transform.f + (np.arange(label.height) + 0.5) * label.transform.e
        result["label"] = {
            "shape": list(label.shape),
            "crs": str(label.crs),
            "transform": list(label.transform),
            "class_counts": counts(values),
            "sensor_grid_matches_label_centers": bool(
                np.allclose(x1, expected_x, rtol=0, atol=1e-6)
                and np.allclose(y1, expected_y, rtol=0, atol=1e-6)
            ),
        }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Sample {folder.name}: S1/S2 coordinates match={result['s1_s2_coordinates_match']}, label centers match={result['label']['sensor_grid_matches_label_centers']}")
    print("S2-S1 days by time index:", result["s2_minus_s1_days_by_index"])
    print("Label class counts:", result["label"]["class_counts"])
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
