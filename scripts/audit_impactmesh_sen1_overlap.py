"""Screen actual ImpactMesh mask grids against Sen1Floods11 hand-test grids.

All ImpactMesh footprints come from official TIFF headers in verified MASK.tar
archives. No label pixel values are read. This remains a spatial screen, not
proof that flood processes or source Sentinel scenes are independent.
"""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import rasterio
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.strtree import STRtree


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
SEN_LABELS = ROOT / "data" / "raw" / "sen1floods11" / "data" / "flood_events" / "HandLabeled" / "LabelHand"


@lru_cache(maxsize=None)
def transformer(crs: str) -> Transformer:
    return Transformer.from_crs(crs, "EPSG:4326", always_xy=True)


def corners_to_wgs(crs: str, corners: list[tuple[float, float]]) -> Polygon:
    lon, lat = transformer(crs).transform([p[0] for p in corners], [p[1] for p in corners])
    polygon = Polygon(zip(lon, lat))
    if not polygon.is_valid or polygon.is_empty:
        raise ValueError("Invalid transformed image footprint")
    return polygon


def grid_footprint(row: dict[str, str]) -> Polygon:
    left, bottom, right, top = (float(row[name]) for name in ("left", "bottom", "right", "top"))
    if not (left < right and bottom < top):
        raise ValueError(f"Invalid TIFF bounds: {row['sample_id']}")
    return corners_to_wgs(row["grid_crs"], [(left, top), (right, top), (right, bottom), (left, bottom)])


def label_footprint(path: Path) -> Polygon:
    with rasterio.open(path) as raster:
        t = raster.transform
        corners = [t * (col, row) for col, row in ((0, 0), (raster.width, 0), (raster.width, raster.height), (0, raster.height))]
        return corners_to_wgs(str(raster.crs), corners)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_grids() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    grids: dict[str, dict[str, str]] = {}
    input_hashes = {}
    assets = {item["path"]: item for item in json.loads((MANIFESTS / "impactmesh_assets.json").read_text(encoding="utf-8"))["assets"]}
    for split in ("train", "val", "test"):
        path = MANIFESTS / f"impactmesh_{split}_mask_grid_index.csv"
        meta = json.loads((MANIFESTS / f"impactmesh_{split}_mask_grid_index_meta.json").read_text(encoding="utf-8"))
        archive = ROOT / "data" / "raw" / "impactmesh" / split / "MASK.tar"
        published = assets[f"{split}/MASK.tar"]
        if (
            meta["split"] != split
            or sha256(path) != meta["grid_index_sha256"]
            or sha256(MANIFESTS / f"impactmesh_{split}_mask_index.csv") != meta["source_index_sha256"]
            or meta["source_tar_sha256"] != published["lfs"]["oid"]
            or archive.stat().st_size != published["size"]
            or sha256(archive) != published["lfs"]["oid"]
        ):
            raise RuntimeError(f"Grid index or verified source archive changed: {split}")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        official = (MANIFESTS / f"impactmesh_flood_{split}.txt").read_text(encoding="utf-8").splitlines()
        if {row["sample_id"] for row in rows} != set(official) or len(rows) != len(official):
            raise RuntimeError(f"Actual grid index differs from official {split} sample IDs")
        for row in rows:
            if row["sample_id"] in grids or row["split"] != split:
                raise RuntimeError(f"Duplicate or misfiled sample: {row['sample_id']}")
            grids[row["sample_id"]] = row
        input_hashes[split] = sha256(path)
    return grids, input_hashes


def verified_sen_labels() -> list[Path]:
    with (MANIFESTS / "sen1floods11_manifest.csv").open(newline="", encoding="utf-8") as handle:
        selected = [row for row in csv.DictReader(handle) if "/LabelHand/" in row["name"] and row["selected_test"] == "True"]
    labels = sorted(SEN_LABELS.glob("*_LabelHand.tif"))
    if len(selected) != 90 or len(labels) != 90 or {Path(row["name"]).name for row in selected} != {path.name for path in labels}:
        raise RuntimeError("Official Sen1 hand-test label names differ from verified local files")
    for row in selected:
        path = ROOT / row["local_path"]
        if row["state"] != "verified" or path.stat().st_size != int(row["size_bytes"]) or sha256(path) != row["sha256"]:
            raise RuntimeError(f"Sen1 hand-test label differs from manifest: {row['name']}")
    return labels


def validate_local_labels(grids: dict[str, dict[str, str]]) -> list[str]:
    with (MANIFESTS / "impactmesh_sample_manifest.csv").open(newline="", encoding="utf-8") as handle:
        records = [row for row in csv.DictReader(handle) if row["modality"] == "MASK"]
    checked = []
    for record in records:
        sample_id = record["sample_id"]
        path = ROOT / record["local_path"]
        if path.stat().st_size != int(record["size_bytes"]) or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Local label differs from member manifest: {sample_id}")
        expected = grids[sample_id]
        with rasterio.open(path) as raster:
            actual = (raster.bounds.left, raster.bounds.bottom, raster.bounds.right, raster.bounds.top)
            recorded = tuple(float(expected[key]) for key in ("left", "bottom", "right", "top"))
            if str(raster.crs) != expected["grid_crs"] or any(abs(a - b) > 1e-6 for a, b in zip(actual, recorded)):
                raise RuntimeError(f"Full TAR header index disagrees with local label: {sample_id}")
        checked.append(sample_id)
    return checked


def main() -> None:
    grids, grid_hashes = load_grids()
    checked = validate_local_labels(grids)
    labels = verified_sen_labels()
    scenes = [path.name.removesuffix("_LabelHand.tif") for path in labels]
    polygons = [label_footprint(path) for path in labels]
    index = STRtree(polygons)
    holdout = set((MANIFESTS / "impactmesh_flood_test_holdout.txt").read_text(encoding="utf-8").splitlines())
    if not holdout <= grids.keys() or any(grids[sample_id]["split"] != "test" for sample_id in holdout):
        raise RuntimeError("Official holdout IDs must be a subset of test")
    count_by_split = Counter()
    activation_by_split = defaultdict(set)
    hit_ids = defaultdict(set)
    hit_activations = defaultdict(set)
    hit_scenes = defaultdict(set)
    overlaps = []
    for sample_id, row in grids.items():
        split = row["split"]
        activation = row["activation"]
        candidate = grid_footprint(row)
        groups = [split] + (["test_holdout"] if sample_id in holdout else [])
        for group in groups:
            count_by_split[group] += 1
            activation_by_split[group].add(activation)
        for idx in index.query(candidate, predicate="intersects"):
            if candidate.intersection(polygons[int(idx)]).area <= 0:
                continue
            scene = scenes[int(idx)]
            for group in groups:
                hit_ids[group].add(sample_id)
                hit_activations[group].add(activation)
                hit_scenes[group].add(scene)
            overlaps.append({
                "split": split,
                "test_holdout": sample_id in holdout,
                "sample_id": sample_id,
                "activation": activation,
                "sen1_scene": scene,
                "sen1_event": scene.split("_", 1)[0],
            })
    output = {
        "source": "official ImpactMesh mask TIFF headers from fully verified train/val/test MASK.tar, official ID lists, and locally verified Sen1Floods11 hand-test label rasters",
        "source_list_sha256": {name: sha256(MANIFESTS / f"impactmesh_flood_{name}.txt") for name in ("train", "val", "test", "test_holdout")},
        "mask_grid_index_sha256": grid_hashes,
        "sen1_manifest_sha256": sha256(MANIFESTS / "sen1floods11_manifest.csv"),
        "impactmesh_geometry_rule": "Actual TIFF header CRS and bounds, never inferred ID MGRS zone",
        "geometry_checked_against_local_labels": checked,
        "id_zone_crs_mismatch_count_by_split": {
            split: sum(row["id_zone_matches_grid"] == "0" for row in grids.values() if row["split"] == split)
            for split in ("train", "val", "test")
        },
        "sen1_hand_test_scene_count": len(scenes),
        "impactmesh_sample_count_by_split": dict(count_by_split),
        "impactmesh_activation_count_by_split": {key: len(value) for key, value in activation_by_split.items()},
        "overlap_sample_count_by_split": {key: len(hit_ids[key]) for key in count_by_split},
        "overlap_activation_count_by_split": {key: len(hit_activations[key]) for key in count_by_split},
        "overlap_sen1_scene_count_by_split": {key: len(hit_scenes[key]) for key in count_by_split},
        "overlap_pair_count": len(overlaps),
        "overlaps": overlaps,
        "limits": "Footprint comparison only. No label pixels read. Spatial nonoverlap does not prove distinct flood processes or Sentinel source imagery. Final selected assets require renewed process-level review.",
    }
    target = MANIFESTS / "impactmesh_sen1_overlap_audit.json"
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"ImpactMesh samples screened: {len(grids)}; Sen1 test scenes: {len(scenes)}")
    print("Overlapping samples by split:", output["overlap_sample_count_by_split"])
    print("Overlapping activations by split:", output["overlap_activation_count_by_split"])
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
