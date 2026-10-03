"""Audit STURM's published CSV metadata before any image-level pairing claim."""

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "sturm" / "metadata"
OUT = ROOT / "data" / "manifests"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    s1 = pd.read_csv(RAW / "Sentinel1_metadata.csv").drop(columns=["Unnamed: 0"])
    s2 = pd.read_csv(RAW / "Sentinel2_metadata.csv").drop(columns=["Unnamed: 0"])
    assert s1.tile_id.is_unique and s2.tile_id.is_unique
    pairs = s1.merge(s2, on="tile_id", suffixes=("_s1", "_s2"), validate="one_to_one")
    for column in ("ems_code", "aoi_code", "epsg_code"):
        if not (pairs[f"{column}_s1"] == pairs[f"{column}_s2"]).all():
            raise ValueError(f"Same tile_id has conflicting {column}")
    pairs["s1_time"] = pd.to_datetime(pairs["sentinel_date"])
    pairs["s2_time"] = pd.to_datetime(pairs["sentinel_timestamp"])
    pairs["delta_s2_minus_s1_days"] = (pairs.s2_time - pairs.s1_time).dt.total_seconds() / 86400
    pairs["same_floodmap_id"] = pairs.floodmap_id_s1 == pairs.floodmap_id_s2
    pairs["same_floodmap_date"] = pairs.floodmap_date_s1 == pairs.floodmap_date_s2
    pairs["candidate_only"] = True
    output_columns = [
        "tile_id", "ems_code_s1", "aoi_code_s1", "country_s1", "epsg_code_s1",
        "s1_time", "s2_time", "delta_s2_minus_s1_days",
        "floodmap_id_s1", "floodmap_id_s2", "floodmap_date_s1", "floodmap_date_s2",
        "same_floodmap_id", "same_floodmap_date", "candidate_only",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    pairs[output_columns].to_csv(OUT / "sturm_candidate_pairs.csv", index=False)
    summary = pairs.groupby("ems_code_s1").agg(
        candidate_pairs=("tile_id", "size"),
        aois=("aoi_code_s1", "nunique"),
        same_floodmap_pairs=("same_floodmap_id", "sum"),
        s1_floodmaps=("floodmap_id_s1", "nunique"),
        s2_floodmaps=("floodmap_id_s2", "nunique"),
        median_abs_delta_days=("delta_s2_minus_s1_days", lambda values: values.abs().median()),
    )
    summary.to_csv(OUT / "sturm_activation_summary.csv")
    print(f"S1 assets: {len(s1)}; S2 assets: {len(s2)}")
    print(f"Exact tile-ID candidate pairs: {len(pairs)} in {len(summary)} activations")
    print(f"Pairs using the same CEMS map ID: {pairs.same_floodmap_id.sum()}")
    print(f"Largest activation share: {summary.candidate_pairs.max() / len(pairs):.1%}")
    print("These are metadata-level candidates, not verified spatial/semantic pairs.")


if __name__ == "__main__":
    main()
