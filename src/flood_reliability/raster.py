"""Raster alignment and label handling with explicit spatial metadata."""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


def align_to_reference(source_path: str, reference_path: str, *, categorical: bool = False) -> np.ndarray:
    """Reproject one band to a reference grid; preserve class values for labels."""
    with rasterio.open(source_path) as source, rasterio.open(reference_path) as reference:
        if source.crs is None or reference.crs is None:
            raise ValueError("Both rasters require a CRS")
        if source.count != 1:
            raise ValueError("Expected a single-band source raster")
        if categorical:
            destination = np.full((reference.height, reference.width), 255, dtype=np.uint8)
            dst_nodata = 255
            resampling = Resampling.nearest
        else:
            destination = np.full((reference.height, reference.width), np.nan, dtype=np.float32)
            dst_nodata = np.nan
            resampling = Resampling.bilinear
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=reference.transform,
            dst_crs=reference.crs,
            dst_nodata=dst_nodata,
            resampling=resampling,
        )
        return destination
