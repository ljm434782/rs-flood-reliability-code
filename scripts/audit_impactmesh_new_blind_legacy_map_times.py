"""Extract only source-map situation minutes from locked legacy CEMS PDFs."""

import csv
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone

from scripts.train_impactmesh_formal import ROOT, sha256

MAN = ROOT / "data/manifests"
SOURCE = MAN / "impactmesh_new_blind_legacy_cems_map_sources.csv"
TIMES = MAN / "impactmesh_new_blind_s1_times.csv"
OUT = MAN / "impactmesh_new_blind_legacy_map_time_audit.json"
LOCKED = ("EMSR411", "EMSR559")


def main():
    if OUT.exists():
        raise RuntimeError("Refusing to overwrite legacy map time audit")
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        maps = list(csv.DictReader(handle))
    with TIMES.open(newline="", encoding="utf-8") as handle:
        source_times = defaultdict(set)
        for row in csv.DictReader(handle):
            if row["activation"] in LOCKED:
                source_times[row["activation"]].add(row["event_s1_utc"])
    records = {}
    for code in LOCKED:
        matching = [row for row in maps if row["activation"] == code and "DEL_PRODUCT" in row["map_url"]]
        if len(matching) != 1 or len(source_times[code]) != 1:
            raise RuntimeError(f"Unexpected locked legacy product or S1 times: {code}")
        item = matching[0]
        path = ROOT / item["local_path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"CEMS PDF hash changed: {code}")
        text = subprocess.run(["pdftotext", "-raw", str(path), "-"],
                              capture_output=True, text=True, check=True).stdout
        compact = re.sub(r"\s+", "", text)
        instant = datetime.fromisoformat(next(iter(source_times[code])).replace("Z", "+00:00"))
        minute = instant.strftime("%d/%m/%Y%H:%M")
        first = compact.find(minute)
        # The selected minute must occur in the map's relevant situation date section.
        match = first >= 0 and "Situationasof" in compact[max(0, first - 80):first]
        records[code] = {"s1_event_utc": instant.isoformat(),
                         "map_situation_minute_utc": minute,
                         "situation_minute_matches_s1": match,
                         "map_url": item["map_url"], "map_sha256": item["sha256"]}
        print(code, "situation_minute_matches_s1", match)
    OUT.write_text(json.dumps({"completed_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "locked legacy CEMS map situation-time audit only",
        "map_source_manifest_sha256": sha256(SOURCE),
        "s1_times_sha256": sha256(TIMES), "events": records,
        "limitation": "map time match does not establish ImpactMesh per-tile product identity",
        "impactmesh_label_pixels_read": False, "model_outputs_read": False},
        indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
