"""Locate, download, and audit a small fixed batch of train-only pilot samples."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
SCRIPTS = ROOT / "scripts"


def run(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(SCRIPTS / script), *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ranks", type=int, nargs="+", help="One to four distinct ranks from the frozen 24-row train pilot")
    args = parser.parse_args()
    if not 1 <= len(args.ranks) <= 4 or len(set(args.ranks)) != len(args.ranks):
        parser.error("Provide one to four distinct frozen-frame ranks")
    with (MANIFESTS / "impactmesh_temporal_audit_frame.csv").open(newline="", encoding="utf-8") as handle:
        frame = {int(row["rank"]): row for row in csv.DictReader(handle)}
    if not set(args.ranks) <= set(frame):
        parser.error("Every rank must exist in the frozen train-only frame")
    for rank in args.ranks:
        sample_id = frame[rank]["sample_id"]
        print(f"--- fixed pilot rank {rank}: {sample_id} ---", flush=True)
        for modality in ("S1RTC", "S2L2A"):
            run("locate_impactmesh_tar.py", "--frame-rank", str(rank), "--modality", modality)
        run("download_impactmesh_pilot.py", "--frame-rank", str(rank))
        run("audit_impactmesh_sample.py", "--sample-id", sample_id)
        output = MANIFESTS / f"impactmesh_sample_audit_{sample_id}.json"
        audit = json.loads(output.read_text(encoding="utf-8"))
        if (audit["sample_id"] != sample_id or audit["split"] != "train" or
                f"frame rank {rank}" not in audit["selection"] or
                not audit["s1_s2_coordinates_match"] or
                not audit["label"]["sensor_grid_matches_label_centers"]):
            raise RuntimeError(f"Pilot audit verification failed for {sample_id}")
        print(f"Verified fixed rank {rank}: grid and identity match", flush=True)


if __name__ == "__main__":
    main()
