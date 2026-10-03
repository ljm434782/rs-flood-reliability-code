"""List STURM Zenodo ZIP members through HTTP ranges without fetching the archive."""

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "data" / "manifests" / "sturm_zenodo_record.json"
INDEX = ROOT / "data" / "manifests" / "sturm_zip_index.csv"


class RangeReader(io.RawIOBase):
    def __init__(self, url: str, size: int, block_size: int = 8 * 1024 * 1024):
        self.url = url
        self.size = size
        self.block_size = block_size
        self.position = 0
        self.block_start = -1
        self.block = b""
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, connect=3, read=3, backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
        )))

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError("Invalid seek mode")
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, count: int = -1) -> bytes:
        if count < 0:
            count = self.size - self.position
        output = bytearray()
        while count and self.position < self.size:
            block_start = (self.position // self.block_size) * self.block_size
            if block_start != self.block_start:
                block_end = min(block_start + self.block_size, self.size) - 1
                response = self.session.get(
                    self.url,
                    headers={"Range": f"bytes={block_start}-{block_end}"},
                    timeout=(30, 120),
                )
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError("Server ignored HTTP range request")
                self.block = response.content
                self.block_start = block_start
            within = self.position - self.block_start
            take = min(count, len(self.block) - within)
            if take <= 0:
                raise RuntimeError("Invalid HTTP range response length")
            output.extend(self.block[within:within + take])
            self.position += take
            count -= take
        return bytes(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-metadata", action="store_true", help="Save the two original CSV metadata files")
    parser.add_argument("--sample-tile", metavar="TILE_ID", help="Fetch four TIFFs for one tile ID")
    parser.add_argument("--sample-activation-representatives", action="store_true", help="Fetch one same-name candidate per activation")
    parser.add_argument("--candidate-masks", action="store_true", help="Fetch both masks for all same-name candidates")
    parser.add_argument("--unmatched-activation-masks", action="store_true", help="Fetch masks for dual-modality activations without same-name tiles")
    parser.add_argument("--external-overlap-country-masks", action="store_true", help="Fetch S1/S2 masks in countries shared with the Sen1Floods11 hand test events")
    args = parser.parse_args()
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    asset = next(item for item in record["files"] if item["key"] == "Dataset.zip")
    block_size = 1024 * 1024 if args.sample_tile else 8 * 1024 * 1024
    with zipfile.ZipFile(RangeReader(asset["links"]["self"], asset["size"], block_size)) as archive:
        infos = archive.infolist()
        if args.extract_metadata:
            destination = ROOT / "data" / "raw" / "sturm" / "metadata"
            destination.mkdir(parents=True, exist_ok=True)
            for name in ("Dataset/Sentinel1_metadata.csv", "Dataset/Sentinel2_metadata.csv"):
                target = destination / Path(name).name
                if not target.exists():
                    target.write_bytes(archive.read(name))
                    print(f"Extracted {target}")
        sample_ids = []
        if args.sample_tile:
            sample_ids.append(args.sample_tile)
        if args.sample_activation_representatives:
            table = ROOT / "data" / "manifests" / "sturm_candidate_pairs.csv"
            with table.open(newline="", encoding="utf-8") as handle:
                representatives = {}
                for row in csv.DictReader(handle):
                    representatives.setdefault(row["ems_code_s1"], row["tile_id"])
            sample_ids.extend(representatives.values())
        targets = []
        for tile_id in dict.fromkeys(sample_ids):
            if Path(tile_id).name != tile_id or not tile_id.endswith(".tif"):
                raise ValueError("Expected a TIFF tile basename")
            destination = ROOT / "data" / "raw" / "sturm" / "samples" / Path(tile_id).stem
            for sensor, folders in (("S1", ("Sentinel1", "S1")), ("S2", ("Sentinel2", "S2"))):
                targets.append((f"Dataset/{folders[0]}/{folders[1]}/{tile_id}", destination / f"{sensor}.tif"))
                targets.append((f"Dataset/{folders[0]}/Floodmaps/{tile_id}", destination / f"{sensor}_floodmap.tif"))
        if args.candidate_masks:
            table = ROOT / "data" / "manifests" / "sturm_candidate_pairs.csv"
            with table.open(newline="", encoding="utf-8") as handle:
                candidate_ids = [row["tile_id"] for row in csv.DictReader(handle)]
            for tile_id in candidate_ids:
                if Path(tile_id).name != tile_id or not tile_id.endswith(".tif"):
                    raise ValueError("Expected a TIFF tile basename")
                for sensor, folder in (("S1", "Sentinel1"), ("S2", "Sentinel2")):
                    name = f"Dataset/{folder}/Floodmaps/{tile_id}"
                    target = ROOT / "data" / "raw" / "sturm" / "candidate_masks" / sensor / tile_id
                    targets.append((name, target))
        if args.unmatched_activation_masks:
            metadata = ROOT / "data" / "raw" / "sturm" / "metadata"
            tables = {}
            for sensor, folder in (("S1", "Sentinel1"), ("S2", "Sentinel2")):
                with (metadata / f"{folder}_metadata.csv").open(newline="", encoding="utf-8") as handle:
                    tables[sensor] = list(csv.DictReader(handle))
            both = {row["ems_code"] for row in tables["S1"]} & {row["ems_code"] for row in tables["S2"]}
            with (ROOT / "data" / "manifests" / "sturm_activation_summary.csv").open(newline="", encoding="utf-8") as handle:
                same_name = {row["ems_code_s1"] for row in csv.DictReader(handle)}
            unmatched = both - same_name
            if unmatched != {"EMSR424", "EMSR444", "EMSR554", "EMSR567"}:
                raise RuntimeError(f"Unexpected activations without same-name tiles: {sorted(unmatched)}")
            for sensor, folder in (("S1", "Sentinel1"), ("S2", "Sentinel2")):
                for row in tables[sensor]:
                    if row["ems_code"] not in unmatched:
                        continue
                    tile_id = row["tile_id"]
                    if Path(tile_id).name != tile_id or not tile_id.endswith(".tif"):
                        raise ValueError("Expected a TIFF tile basename")
                    targets.append((
                        f"Dataset/{folder}/Floodmaps/{tile_id}",
                        ROOT / "data" / "raw" / "sturm" / "unmatched_activation_masks" / sensor / tile_id,
                    ))
        if args.external_overlap_country_masks:
            for sensor, folder in (("S1", "Sentinel1"), ("S2", "Sentinel2")):
                source = ROOT / "data" / "raw" / "sturm" / "metadata" / f"{folder}_metadata.csv"
                with source.open(newline="", encoding="utf-8") as handle:
                    external_candidates = [row for row in csv.DictReader(handle) if row["country"] in {"Spain", "Pakistan"}]
                for row in external_candidates:
                    tile_id = row["tile_id"]
                    if Path(tile_id).name != tile_id or not tile_id.endswith(".tif"):
                        raise ValueError("Expected a TIFF tile basename")
                    targets.append((
                        f"Dataset/{folder}/Floodmaps/{tile_id}",
                        ROOT / "data" / "raw" / "sturm" / f"external_overlap_{sensor.lower()}_masks" / tile_id,
                    ))
        for name, target in sorted(targets, key=lambda item: archive.getinfo(item[0]).header_offset):
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
            print(f"Extracted {target}", flush=True)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    with INDEX.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "size_bytes", "compressed_bytes", "crc32"])
        for item in infos:
            writer.writerow([item.filename, item.file_size, item.compress_size, f"{item.CRC:08x}"])
    print(f"Indexed {len(infos)} members; uncompressed {sum(item.file_size for item in infos) / 1e9:.2f} GB")
    print(f"Saved {INDEX}")


if __name__ == "__main__":
    main()
