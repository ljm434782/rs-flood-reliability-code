"""Read only event S1 timestamps for six source-prequalified val events."""

import csv
import json
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from flood_reliability.impactmesh_io import _group
from scripts.train_impactmesh_formal import ROOT, sha256

MAN = ROOT / "data/manifests"
ARCHIVE = ROOT / "data/raw/impactmesh/val/S1RTC.tar"
OUT = MAN / "impactmesh_new_blind_s1_times.csv"
META = OUT.with_suffix(".json")
IDS = {"EMSR410", "EMSR411", "EMSR496", "EMSR559", "EMSR788", "EMSR843"}


def main():
    if OUT.exists() or META.exists():
        raise RuntimeError("Refusing to overwrite new-blind S1 time audit")
    assets = {x["path"]: x for x in json.loads((MAN / "impactmesh_assets.json").read_text(
        encoding="utf-8"))["assets"]}
    if sha256(ARCHIVE) != assets["val/S1RTC.tar"]["lfs"]["oid"]:
        raise RuntimeError("Official val S1 hash mismatch")
    expected = {line for line in (MAN / "impactmesh_flood_val.txt").read_text(
        encoding="utf-8").splitlines() if line.split("_", 1)[0] in IDS}
    rows = []
    with tarfile.open(ARCHIVE, "r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            sample_id = Path(member.name).name.removesuffix("_S1RTC.zarr.zip")
            if sample_id not in expected:
                continue
            s1 = _group(archive.extractfile(member).read())
            times = s1.attrs["datetime"].split(";")
            if len(times) != 4 or s1["bands"].shape != (4, 2, 256, 256):
                raise RuntimeError(f"Unexpected S1 structure {sample_id}")
            rows.append({"sample_id": sample_id, "activation": sample_id.split("_", 1)[0],
                         "event_s1_utc": times[2]})
    if {row["sample_id"] for row in rows} != expected:
        raise RuntimeError("Incomplete val S1 source time audit")
    rows.sort(key=lambda row: row["sample_id"])
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["activation"]].append(row["event_s1_utc"])
    summary = {event: {"tiles": len(values), "first": min(values), "last": max(values)}
               for event, values in sorted(groups.items())}
    META.write_text(json.dumps({"completed_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "event chronology and source independence review; no labels",
        "raw_sha256": sha256(ARCHIVE), "ids_sha256": sha256(MAN / "impactmesh_flood_val.txt"),
        "output_sha256": sha256(OUT), "event_ranges": summary,
        "label_pixels_read": False, "model_outputs_read": False},
        indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
