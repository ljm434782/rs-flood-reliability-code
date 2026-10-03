# Data and checkpoint access

Raw imagery, native label rasters and model tensors/checkpoints are excluded.
There is no checkpoint download link in this release. Complete inference
cannot be reproduced from this package alone; cached main tables/charts can
be reproduced. Full raw-checkpoint recomputation remains unverified.

ImpactMesh-Flood v1 source/version/license records remain in
`data/manifests/impactmesh_assets.json` and
`docs/G3_RAW_ARCHIVE_CHECKSUM_AUDIT.json`. Scientific configuration projections
retain the exact original asset paths, sizes and SHA-256 values. These are
historical publisher/provenance records, not new verification of live download
servers. Frozen assets must not be silently replaced by another version.

Sen1Floods11 source records and fixed exclusions remain in
`data/manifests/sen1floods11_manifest.csv` and
`data/manifests/sen1_external_process_exclusions.csv`. This external test is
water/non-water only; it does not validate the three-class native labels.

No repository, checkpoint, DOI or journal supplement URL was invented.
If the authors later distribute checkpoints, genuine external release-asset
URLs should be accompanied by SHA-256 and model/seed/epoch/temperature mappings.
No tensors were added during this sanitation task. Data licenses remain
those of their publishers; this project's author-code MIT License does not
cover the datasets.
