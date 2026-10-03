"""Read verified ImpactMesh train tiles by TAR member without altering raw data."""

import io
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import rasterio
import zarr
from rasterio.crs import CRS
from rasterio.io import MemoryFile


SUFFIXES = {"S1RTC": "_S1RTC.zarr.zip", "S2L2A": "_S2L2A.zarr.zip", "MASK": "_annotation_flood.tif"}


def _group(blob: bytes):
    store = zarr.storage.MemoryStore()
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for name in archive.namelist():
            store[name] = archive.read(name)
    return zarr.open_consolidated(store, mode="r")


class TrainArchiveReader:
    """Single-process random access; each instance owns its own file handles."""

    def __init__(self, raw_train: Path):
        self.paths = {key: raw_train / f"{key}.tar" for key in SUFFIXES}
        self.handles = {key: path.open("rb") for key, path in self.paths.items()}
        self.index = {}
        for modality, path in self.paths.items():
            entries = {}
            with tarfile.open(path, "r:") as archive:
                for member in archive:
                    suffix = SUFFIXES[modality]
                    if not member.isfile() or not member.name.startswith(f"{modality}/") or not member.name.endswith(suffix):
                        raise RuntimeError(f"Unexpected {modality} TAR member: {member.name}")
                    sample_id = Path(member.name).name.removesuffix(suffix)
                    if sample_id in entries:
                        raise RuntimeError(f"Duplicate {modality} sample: {sample_id}")
                    entries[sample_id] = (member.offset_data, member.size)
            self.index[modality] = entries
        sample_sets = [set(items) for items in self.index.values()]
        if any(sample_sets[0] != other for other in sample_sets[1:]):
            raise RuntimeError("Train S1/S2/MASK archives contain different sample IDs")

    def close(self):
        for handle in self.handles.values():
            handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def blob(self, modality: str, sample_id: str) -> bytes:
        offset, size = self.index[modality][sample_id]
        handle = self.handles[modality]
        handle.seek(offset)
        blob = handle.read(size)
        if len(blob) != size:
            raise RuntimeError(f"Truncated archive member: {modality} {sample_id}")
        return blob

    def load(self, sample_id: str, s2_index: int):
        if s2_index not in range(4):
            raise ValueError("Selected S2 time index must be 0..3")
        s1 = _group(self.blob("S1RTC", sample_id))
        s2 = _group(self.blob("S2L2A", sample_id))
        with MemoryFile(self.blob("MASK", sample_id)) as memory:
            with memory.open() as source:
                label = source.read(1).astype(np.int64)
                if source.shape != (256, 256) or source.count != 1:
                    raise RuntimeError(f"Unexpected label shape: {sample_id}")
                grid_crs = CRS.from_wkt(s1["spatial_ref"].attrs["crs_wkt"])
                if source.crs != grid_crs:
                    raise RuntimeError(f"S1/label CRS mismatch: {sample_id}")
                expected_x = source.transform.c + (np.arange(256) + .5) * source.transform.a
                expected_y = source.transform.f + (np.arange(256) + .5) * source.transform.e
                if not (np.allclose(s1["x"][:], expected_x, atol=1e-6, rtol=0)
                        and np.allclose(s1["y"][:], expected_y, atol=1e-6, rtol=0)
                        and np.allclose(s2["x"][:], expected_x, atol=1e-6, rtol=0)
                        and np.allclose(s2["y"][:], expected_y, atol=1e-6, rtol=0)
                        and CRS.from_wkt(s2["spatial_ref"].attrs["crs_wkt"]) == source.crs):
                    raise RuntimeError(f"S1/S2/label grid mismatch: {sample_id}")
        sar = np.asarray(s1["bands"][2], dtype=np.float32)
        optical = np.asarray(s2["bands"][s2_index], dtype=np.float32)
        if sar.shape != (2, 256, 256) or optical.shape != (12, 256, 256):
            raise RuntimeError(f"Unexpected image array shape: {sample_id}")
        if int(s1["data_mask"][2]) != 1 or np.any(s1["nan_mask"][2] != 0) or not np.isfinite(sar).all():
            raise RuntimeError(f"Invalid S1 event tile: {sample_id}")
        valid = (np.isin(np.asarray(s2["SCL"][s2_index]), (4, 5, 6))
                 & (np.asarray(s2["cloud_mask"][s2_index]) == 0)
                 & (np.asarray(s2["nan_mask"][s2_index]) == 0)
                 & np.all(np.isfinite(optical) & (optical != -9999), axis=0)
                 & (int(s2["data_mask"][s2_index]) == 1))
        if valid.mean() < .5:
            raise RuntimeError(f"Frozen optical eligibility failed at read: {sample_id}")
        if not set(np.unique(label)) <= {-1, 0, 1, 2}:
            raise RuntimeError(f"Unexpected label values: {sample_id}")
        label[label == -1] = 255
        optical = np.where(valid[None], optical, 0).astype(np.float32, copy=False)
        return sar, optical, valid, label

    def load_s1_label(self, sample_id: str):
        """Read a frozen real-S2-unavailable tile without inspecting S2 pixels."""
        s1 = _group(self.blob("S1RTC", sample_id))
        with MemoryFile(self.blob("MASK", sample_id)) as memory:
            with memory.open() as source:
                if source.shape != (256, 256) or source.count != 1:
                    raise RuntimeError(f"Unexpected label grid: {sample_id}")
                if CRS.from_wkt(s1["spatial_ref"].attrs["crs_wkt"]) != source.crs:
                    raise RuntimeError(f"S1/label CRS mismatch: {sample_id}")
                expected_x = source.transform.c + (np.arange(256) + .5) * source.transform.a
                expected_y = source.transform.f + (np.arange(256) + .5) * source.transform.e
                if not (np.allclose(s1["x"][:], expected_x, atol=1e-6, rtol=0)
                        and np.allclose(s1["y"][:], expected_y, atol=1e-6, rtol=0)):
                    raise RuntimeError(f"S1/label coordinate mismatch: {sample_id}")
                label = source.read(1).astype(np.int64)
        sar = np.asarray(s1["bands"][2], dtype=np.float32)
        if (sar.shape != (2, 256, 256) or int(s1["data_mask"][2]) != 1
                or np.any(s1["nan_mask"][2] != 0) or not np.isfinite(sar).all()):
            raise RuntimeError(f"Invalid event S1: {sample_id}")
        if not set(np.unique(label)) <= {-1, 0, 1, 2}:
            raise RuntimeError(f"Unexpected label codes: {sample_id}")
        label[label == -1] = 255
        return sar, label
