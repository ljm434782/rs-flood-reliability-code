"""Find cross-activation time/space neighbors before formal process isolation.

Uses official train S1 acquisition metadata and TIFF georeferencing only. It
does not read label pixels or any official test imagery. Candidate pairs need
CEMS event review; proximity is not proof of a shared flood process.
"""

import csv
import gzip
import hashlib
import io
import json
import math
import tarfile
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
S1 = ROOT / "data" / "raw" / "impactmesh" / "train" / "S1RTC.tar"
GRID = MANIFESTS / "impactmesh_train_mask_grid_index.csv"
SPLIT = MANIFESTS / "impactmesh_process_split_frozen.csv"
TIMES = MANIFESTS / "impactmesh_train_event_times.csv.gz"
PAIRS = MANIFESTS / "impactmesh_process_neighbor_candidates.csv"
META = MANIFESTS / "impactmesh_process_neighbor_candidates_meta.json"
SUFFIX = "_S1RTC.zarr.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(a)))


def main() -> None:
    assets = {item["path"]: item for item in json.loads(
        (MANIFESTS / "impactmesh_assets.json").read_text(encoding="utf-8"))["assets"]}
    expected = assets["train/S1RTC.tar"]
    if S1.stat().st_size != int(expected["size"]) or sha256(S1) != expected["lfs"]["oid"]:
        raise RuntimeError("Official S1 archive failed size/SHA256 verification")
    official_ids = set((MANIFESTS / "impactmesh_flood_train.txt").read_text(encoding="utf-8").splitlines())
    with SPLIT.open(newline="", encoding="utf-8") as handle:
        roles = {row["activation"]: row["role"] for row in csv.DictReader(handle)}
    if len(official_ids) != 19448 or len(roles) != 194:
        raise RuntimeError("Official IDs or frozen roles changed")
    seen = set()
    moments: dict[str, list[datetime]] = defaultdict(list)
    temp = TIMES.with_suffix(".csv.gz.part")
    with gzip.open(temp, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "activation", "role", "s1_event_utc"))
        writer.writeheader()
        with tarfile.open(S1, "r:") as archive:
            for index, member in enumerate(archive, start=1):
                if not member.isfile() or not member.name.startswith("S1RTC/") or not member.name.endswith(SUFFIX):
                    raise RuntimeError(f"Unexpected S1 TAR member: {member.name}")
                sample_id = Path(member.name).name.removesuffix(SUFFIX)
                if sample_id not in official_ids or sample_id in seen:
                    raise RuntimeError(f"Unexpected or duplicate S1 sample: {sample_id}")
                seen.add(sample_id)
                activation = sample_id.split("_", 1)[0]
                with zipfile.ZipFile(io.BytesIO(archive.extractfile(member).read())) as source:
                    attrs = json.loads(source.read(".zattrs"))
                times = attrs["datetime"].split(";")
                if len(times) != 4:
                    raise RuntimeError(f"Unexpected S1 acquisition count: {sample_id}")
                stamp = utc(times[2])
                moments[activation].append(stamp)
                writer.writerow({"sample_id": sample_id, "activation": activation,
                                 "role": roles[activation], "s1_event_utc": times[2]})
                if index % 2000 == 0:
                    print(f"S1 event timestamps: {index}/{len(official_ids)}", flush=True)
    if seen != official_ids:
        raise RuntimeError("S1 TAR and official split do not match")
    temp.replace(TIMES)
    envelopes: dict[str, list[float]] = {}
    transformers = {}
    grid_seen = set()
    with GRID.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_id = row["sample_id"]
            if row["split"] != "train" or sample_id not in official_ids or sample_id in grid_seen:
                raise RuntimeError(f"Unexpected or duplicate grid record: {sample_id}")
            grid_seen.add(sample_id)
            activation = row["activation"]
            crs = row["grid_crs"]
            if crs not in transformers:
                transformers[crs] = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            west, south, east, north = transformers[crs].transform_bounds(
                float(row["left"]), float(row["bottom"]), float(row["right"]), float(row["top"]), densify_pts=4)
            if activation not in envelopes:
                envelopes[activation] = [west, south, east, north]
            else:
                env = envelopes[activation]
                env[0], env[1] = min(env[0], west), min(env[1], south)
                env[2], env[3] = max(env[2], east), max(env[3], north)
    if grid_seen != official_ids or set(moments) != set(envelopes):
        raise RuntimeError("Grid and S1 acquisition activation sets differ")
    activations = sorted(moments)
    summary = {}
    for activation in activations:
        values = sorted(moments[activation])
        west, south, east, north = envelopes[activation]
        summary[activation] = {"first": values[0], "last": values[-1],
                               "mid": values[len(values) // 2],
                               "lon": (west + east) / 2, "lat": (south + north) / 2,
                               "longitude_span": east - west, "role": roles[activation]}
    rows = []
    for i, left in enumerate(activations):
        a = summary[left]
        for right in activations[i + 1:]:
            b = summary[right]
            day_gap = max(0.0, (b["first"] - a["last"]).total_seconds() / 86400,
                          (a["first"] - b["last"]).total_seconds() / 86400)
            if day_gap > 30:
                continue
            km = haversine_km(a["lon"], a["lat"], b["lon"], b["lat"])
            if km > 500:
                continue
            rows.append({"activation_a": left, "role_a": a["role"],
                         "activation_b": right, "role_b": b["role"],
                         "cross_role": int(a["role"] != b["role"]),
                         "minimum_time_gap_days": round(day_gap, 6),
                         "median_s1_gap_days": round(abs((a["mid"] - b["mid"]).total_seconds()) / 86400, 6),
                         "centroid_distance_km": round(km, 3),
                         "a_first_s1": a["first"].isoformat(), "a_last_s1": a["last"].isoformat(),
                         "b_first_s1": b["first"].isoformat(), "b_last_s1": b["last"].isoformat(),
                         "a_lon_span_degrees": round(a["longitude_span"], 3),
                         "b_lon_span_degrees": round(b["longitude_span"], 3)})
    rows.sort(key=lambda row: (-row["cross_role"], row["minimum_time_gap_days"], row["centroid_distance_km"]))
    with PAIRS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
            "activation_a", "role_a", "activation_b", "role_b", "cross_role",
            "minimum_time_gap_days", "median_s1_gap_days", "centroid_distance_km",
            "a_first_s1", "a_last_s1", "b_first_s1", "b_last_s1",
            "a_lon_span_degrees", "b_lon_span_degrees"])
        writer.writeheader()
        writer.writerows(rows)
    META.write_text(json.dumps({
        "source": "Official verified train S1 archive and train MASK TIFF georeferencing only",
        "purpose": "Conservative candidate pairs for manual CEMS same-event review; proximity is not a grouping rule",
        "activation_count": len(activations), "candidate_pair_count": len(rows),
        "cross_role_candidate_count": sum(int(row["cross_role"]) for row in rows),
        "candidate_window_days": 30, "candidate_centroid_distance_km": 500,
        "s1_archive_sha256": expected["lfs"]["oid"],
        "official_train_split_sha256": sha256(MANIFESTS / "impactmesh_flood_train.txt"),
        "grid_index_sha256": sha256(GRID), "frozen_split_sha256": sha256(SPLIT),
        "times_output_sha256": sha256(TIMES), "pairs_output_sha256": sha256(PAIRS),
        "labels_read": False, "official_test_imagery_read": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Found {len(rows)} nearby activation pairs, {sum(int(row['cross_role']) for row in rows)} cross-role", flush=True)


if __name__ == "__main__":
    main()
