"""Locate one fixed ImpactMesh sample inside a large remote TAR using range reads.

Only TAR headers are read. The exact member ID is checked before any payload
download. The method assumes S1/S2 member order matches the verified MASK TAR;
it fails closed if that assumption is contradicted by observed headers.
"""

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
REPO = "ibm-esa-geospatial/ImpactMesh-Flood"
HEADER = 512
PROBE_BYTES = 4 * 1024 * 1024
FINAL_WALK_BYTES = 8 * 1024 * 1024
MAX_PROBE_BYTES = 32 * 1024 * 1024
SUFFIX = {"MASK": "_annotation_flood.tif", "S1RTC": "_S1RTC.zarr.zip", "S2L2A": "_S2L2A.zarr.zip"}


@dataclass(frozen=True)
class Member:
    offset: int
    name: str
    size: int

    @property
    def data_offset(self) -> int:
        return self.offset + HEADER

    @property
    def next_offset(self) -> int:
        return self.data_offset + ((self.size + HEADER - 1) // HEADER) * HEADER


def parse_header(block: bytes, offset: int) -> Member | None:
    if len(block) != HEADER or block[257:262] != b"ustar" or block[156:157] not in (b"0", b"\0"):
        return None
    try:
        saved_checksum = int(block[148:156].strip(b"\0 "), 8)
        size = int(block[124:136].strip(b"\0 "), 8)
        name = block[:100].split(b"\0", 1)[0].decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    check = sum(block[:148]) + 8 * ord(" ") + sum(block[156:])
    if check != saved_checksum or not name or size < 0:
        return None
    return Member(offset, name, size)


class RangeSource:
    def __init__(self, size: int):
        self.size = size
        self.calls = 0
        self.bytes_read = 0

    def read(self, start: int, length: int) -> bytes:
        raise NotImplementedError


class FileSource(RangeSource):
    def __init__(self, path: Path):
        self.path = path
        super().__init__(path.stat().st_size)

    def read(self, start: int, length: int) -> bytes:
        if start < 0 or length < 0 or start + length > self.size:
            raise ValueError("Invalid local range")
        with self.path.open("rb") as handle:
            handle.seek(start)
            result = handle.read(length)
        self.calls += 1
        self.bytes_read += len(result)
        if len(result) != length:
            raise RuntimeError("Incomplete local read")
        return result


class HttpSource(RangeSource):
    def __init__(self, url: str, size: int):
        super().__init__(size)
        self.url = url
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, connect=3, read=3, backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
        )))

    def read(self, start: int, length: int) -> bytes:
        if start < 0 or length <= 0 or start + length > self.size:
            raise ValueError("Invalid HTTP range")
        end = start + length - 1
        response = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=(30, 120))
        response.raise_for_status()
        content_range = response.headers.get("Content-Range", "")
        expected = f"bytes {start}-{end}/{self.size}"
        if response.status_code != 206 or content_range != expected or len(response.content) != length:
            raise RuntimeError(f"Untrusted range response: status={response.status_code}, Content-Range={content_range}, bytes={len(response.content)}")
        self.calls += 1
        self.bytes_read += length
        return response.content


def headers_in(data: bytes, base: int, modality: str):
    suffix = SUFFIX[modality]
    for relative in range(0, len(data) - HEADER + 1, HEADER):
        member = parse_header(data[relative:relative + HEADER], base + relative)
        if member and member.name.startswith(f"{modality}/") and member.name.endswith(suffix):
            yield member


def member_ordinal(member: Member, modality: str, ordinal_by_id: dict[str, int]) -> int:
    sample_id = Path(member.name).name.removesuffix(SUFFIX[modality])
    if sample_id not in ordinal_by_id:
        raise RuntimeError(f"Remote TAR member absent from verified MASK index: {member.name}")
    return ordinal_by_id[sample_id]


def next_header(source: RangeSource, start: int, stop: int, modality: str) -> Member | None:
    start = ((start + HEADER - 1) // HEADER) * HEADER
    limit = min(stop, source.size)
    length = min(PROBE_BYTES, limit - start)
    while length >= HEADER:
        data = source.read(start, length)
        member = next(headers_in(data, start, modality), None)
        if member is not None:
            return member
        if length >= MAX_PROBE_BYTES or start + length >= limit:
            return None
        length = min(2 * length, MAX_PROBE_BYTES, limit - start)
    return None


def previous_header(source: RangeSource, stop: int, modality: str) -> Member:
    length = min(PROBE_BYTES, stop)
    while length >= HEADER:
        start = ((stop - length) // HEADER) * HEADER
        data = source.read(start, stop - start)
        members = list(headers_in(data, start, modality))
        if members:
            return members[-1]
        if length >= MAX_PROBE_BYTES or length >= stop:
            break
        length = min(2 * length, MAX_PROBE_BYTES, stop)
    raise RuntimeError("Could not find TAR header before upper bound")


def walk_from(source: RangeSource, start: int, stop: int, modality: str, ordinal_by_id: dict[str, int], target: int) -> Member:
    data = source.read(start, stop - start)
    offset = start
    while offset + HEADER <= stop:
        relative = offset - start
        member = parse_header(data[relative:relative + HEADER], offset)
        if member is None or not member.name.startswith(f"{modality}/"):
            raise RuntimeError(f"TAR sequence cannot be validated at offset {offset}")
        ordinal = member_ordinal(member, modality, ordinal_by_id)
        if ordinal == target:
            return member
        if ordinal > target:
            raise RuntimeError("Target ordinal skipped: S1/S2 order differs from MASK index")
        offset = member.next_offset
    raise RuntimeError("Target member absent from bounded TAR interval")


def locate(source: RangeSource, modality: str, ordinal_by_id: dict[str, int], target: int, trace: bool = False) -> Member:
    total = len(ordinal_by_id)
    if not 0 <= target < total:
        raise ValueError("Invalid target ordinal")
    lower_offset, lower_ordinal = 0, 0
    upper_offset, upper_ordinal = source.size, total
    for _ in range(40):
        if trace:
            print(f"bounds offset={lower_offset}..{upper_offset} ordinal={lower_ordinal}..{upper_ordinal}", flush=True)
        if upper_offset - lower_offset <= FINAL_WALK_BYTES:
            return walk_from(source, lower_offset, upper_offset, modality, ordinal_by_id, target)
        predicted = lower_offset + (upper_offset - lower_offset) * (target - lower_ordinal) // max(1, upper_ordinal - lower_ordinal)
        predicted = max(lower_offset + HEADER, min(predicted, upper_offset - HEADER))
        member = next_header(source, predicted, upper_offset, modality)
        if trace:
            print(f"probe={predicted} found={member}", flush=True)
        if member is None:
            member = previous_header(source, upper_offset, modality)
            ordinal = member_ordinal(member, modality, ordinal_by_id)
            if trace:
                print(f"previous={member} ordinal={ordinal}", flush=True)
            if ordinal == target:
                return member
            if ordinal > target:
                upper_offset, upper_ordinal = member.offset, ordinal
            else:
                lower_offset, lower_ordinal = member.next_offset, ordinal + 1
            continue
        ordinal = member_ordinal(member, modality, ordinal_by_id)
        if ordinal == target:
            return member
        if ordinal < target:
            if member.next_offset <= lower_offset or member.next_offset >= upper_offset:
                raise RuntimeError("Locator lower bound made no progress")
            lower_offset, lower_ordinal = member.next_offset, ordinal + 1
        else:
            if member.offset == upper_offset:
                raise RuntimeError("Locator upper bound made no progress")
            else:
                upper_offset, upper_ordinal = member.offset, ordinal
        if lower_offset >= upper_offset:
            raise RuntimeError("Locator bounds crossed; TAR ordering may differ")
    raise RuntimeError("TAR locator did not converge")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-rank", type=int)
    parser.add_argument("--modality", choices=("MASK", "S1RTC", "S2L2A"))
    parser.add_argument("--local-mask-check", action="store_true", help="Check algorithm against the verified local train/MASK.tar")
    parser.add_argument("--verify-frame-mask", action="store_true", help="Check all 24 fixed-frame locations against the full verified local MASK index")
    parser.add_argument("--trace", action="store_true", help="Print locator bounds for debugging")
    args = parser.parse_args()
    if args.verify_frame_mask:
        if args.frame_rank is not None or args.modality is not None or args.local_mask_check:
            parser.error("--verify-frame-mask does not take rank, modality, or --local-mask-check")
    elif args.frame_rank is None or args.modality is None:
        parser.error("--frame-rank and --modality are required unless --verify-frame-mask is used")
    with (MANIFESTS / "impactmesh_temporal_audit_frame.csv").open(newline="", encoding="utf-8") as handle:
        frame = {int(row["rank"]): row for row in csv.DictReader(handle)}
    with (MANIFESTS / "impactmesh_train_mask_index.csv").open(newline="", encoding="utf-8") as handle:
        mask_rows = list(csv.DictReader(handle))
    ordinal_by_id = {row["sample_id"]: int(row["ordinal"]) for row in mask_rows}
    if len(ordinal_by_id) != len(mask_rows):
        raise RuntimeError("Duplicate sample IDs in verified MASK member index")
    assets = {item["path"]: item for item in json.loads((MANIFESTS / "impactmesh_assets.json").read_text(encoding="utf-8"))["assets"]}
    if args.verify_frame_mask:
        local_tar = ROOT / "data" / "raw" / "impactmesh" / "train" / "MASK.tar"
        digest = hashlib.sha256()
        with local_tar.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        asset = assets["train/MASK.tar"]
        if local_tar.stat().st_size != asset["size"] or digest.hexdigest() != asset["lfs"]["oid"]:
            raise RuntimeError("Local MASK TAR differs from the official verified asset")
        source = FileSource(local_tar)
        for rank in sorted(frame):
            selected = frame[rank]
            target = int(selected["train_mask_tar_ordinal"])
            member = locate(source, "MASK", ordinal_by_id, target)
            reference = mask_rows[target]
            if (reference["sample_id"] != selected["sample_id"] or
                    member.data_offset != int(reference["member_offset_bytes"]) or
                    member.size != int(reference["member_size_bytes"])):
                raise RuntimeError(f"Locator differs from verified MASK index at frame rank {rank}")
        report = {
            "source_tar": "train/MASK.tar",
            "source_tar_sha256_verified": digest.hexdigest(),
            "fixed_frame_rank_count": len(frame),
            "matched_offsets_and_sizes": len(frame),
            "range_reads": source.calls,
            "range_bytes_read": source.bytes_read,
            "limit": "validates locator against local MASK TAR, not full remote S1/S2 TAR integrity or order",
        }
        output = MANIFESTS / "impactmesh_tar_locator_mask_validation.json"
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Verified {len(frame)}/{len(frame)} fixed-frame MASK locations against official TAR and full member index")
        print(output)
        return
    if args.frame_rank not in frame:
        parser.error("Unknown fixed-frame rank")
    selected = frame[args.frame_rank]
    target = int(selected["train_mask_tar_ordinal"])
    if ordinal_by_id.get(selected["sample_id"]) != target:
        raise RuntimeError("Fixed frame differs from verified MASK member index")
    path = f"train/{args.modality}.tar"
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"
    if args.local_mask_check:
        if args.modality != "MASK":
            parser.error("--local-mask-check requires --modality MASK")
        local_tar = ROOT / "data" / "raw" / "impactmesh" / "train" / "MASK.tar"
        digest = hashlib.sha256()
        with local_tar.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if local_tar.stat().st_size != assets[path]["size"] or digest.hexdigest() != assets[path]["lfs"]["oid"]:
            raise RuntimeError("Local MASK TAR differs from the official verified asset")
        source = FileSource(local_tar)
    else:
        source = HttpSource(url, assets[path]["size"])
    member = locate(source, args.modality, ordinal_by_id, target, trace=args.trace)
    actual_id = Path(member.name).name.removesuffix(SUFFIX[args.modality])
    if actual_id != selected["sample_id"]:
        raise RuntimeError(f"Located wrong member: {actual_id} != {selected['sample_id']}")
    if args.modality == "MASK":
        reference = mask_rows[target]
        if member.data_offset != int(reference["member_offset_bytes"]) or member.size != int(reference["member_size_bytes"]):
            raise RuntimeError("MASK locator disagrees with full verified TAR index")
    result = {
        "frame_rank": args.frame_rank,
        "sample_id": actual_id,
        "tar_path": path,
        "tar_source_url": url,
        "tar_lfs_sha256": assets[path]["lfs"]["oid"],
        "tar_full_sha256_verified": args.local_mask_check,
        "tar_header_offset_bytes": member.offset,
        "member_offset_bytes": member.data_offset,
        "member_size_bytes": member.size,
        "range_requests": source.calls,
        "range_bytes_read": source.bytes_read,
        "state": "location-only record; consult sample manifest for payload download status",
    }
    output = MANIFESTS / f"impactmesh_tar_location_rank{args.frame_rank}_{args.modality}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Matched {member.name}: offset={member.data_offset}, size={member.size}")
    print(f"Range reads: {source.calls}, bytes: {source.bytes_read}")
    print(output)


if __name__ == "__main__":
    main()
