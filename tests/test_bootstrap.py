import json
import logging
import random

import numpy as np
import rasterio
from rasterio.transform import from_origin

from flood_reliability.raster import align_to_reference
from flood_reliability.repro import configure_logging, load_json_config, seed_everything, sha256_file


def test_config_logging_and_hash(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"seed": 23}), encoding="utf-8")
    assert load_json_config(path) == {"seed": 23}
    assert len(sha256_file(path)) == 64
    logger = configure_logging("DEBUG")
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1
    assert configure_logging().handlers == logger.handlers


def test_seed_repeats_python_and_numpy():
    seed_everything(23)
    first = (random.random(), np.random.random())
    seed_everything(23)
    assert first == (random.random(), np.random.random())


def test_categorical_alignment_preserves_ignore(tmp_path):
    transform = from_origin(0, 2, 1, 1)
    profile = dict(driver="GTiff", height=2, width=2, count=1, dtype="uint8", crs="EPSG:4326", transform=transform, nodata=255)
    source = tmp_path / "label.tif"
    reference = tmp_path / "reference.tif"
    with rasterio.open(source, "w", **profile) as dst:
        dst.write(np.array([[0, 1], [2, 255]], dtype=np.uint8), 1)
    with rasterio.open(reference, "w", **profile) as dst:
        dst.write(np.zeros((2, 2), dtype=np.uint8), 1)
    result = align_to_reference(str(source), str(reference), categorical=True)
    np.testing.assert_array_equal(result, [[0, 1], [2, 255]])
