# Sentinel-1/2 flood-mapping reliability — public research release

This repository provides the analysis code, fixed scientific configurations,
process manifests, provenance records and cached numerical outputs supporting
the study. The main reported tables and figures can be regenerated from the
included cached summaries. Model checkpoints and raw imagery are not included,
so complete end-to-end inference reproduction is not claimed by this release.

## Level 1 — immediately reproducible

Use Python 3.11 or newer in your own virtual environment:

```console
python -m pip install -e ".[dev,impactmesh_audit]"
python scripts/render_rs_cached_main.py --output regenerated_main_artifacts
```

The renderer uses saved CSV summaries and their provenance checks, with no
training, temperature fitting or raw-data inference. It recreates the main
result tables, main result plots and their presentation from the included
cached summaries. The two reference tables are in `examples/expected_tables/`.
Original summaries, all process/seed scores, class decomposition and negative
results remain unchanged. Table text and numbers are identical; LF/CRLF
differences depend on the operating system. Plot bytes may vary with fonts
and plotting-library versions.

## Level 2 — scientific implementation available

`src/flood_reliability/` and `scripts/` provide model definitions, optical masks,
metrics, process/pixel ownership, and historical training/calibration/evaluation
entry points. Fixed scientific parameters and process memberships are in
`configs/` and `data/manifests/`. Unique CUDA implementations remain in the
historical migration subtree; byte-identical copies were omitted and inventoried.
The public configuration projections omit one administrative free-text field
only; their new hashes do not equal or replace the original experiment hashes.

The public distribution excludes historical execution-control attestations.
Its administrative execution gate always refuses full historical execution;
no public approval, new freeze, independent review or bypass is inferred.
This supports scientific inspection and cached reproduction, not an immediately
executable historical full-data training/inference workflow.

## Level 3 — not self-contained in this archive

Complete historical computation requires ImpactMesh-Flood raw imagery and
native labels, Sen1Floods11 source data, model tensors and the specific
historical hardware/runtime and execution-control records. Those resources
are not supplied together here. No checkpoint download link has been released
in this package. `full_raw_checkpoint_recomputation_verified` remains false.

## Interpretation and provenance

The main B4/B2 comparison uses three seeds (20261013–20261015) and fixed
100/75/50/25/0% optical retention. The upgrade roles remain 60 fit, 19
calibration, 18 internal retrospective and 9 holdout retrospective units.
The analysis is retrospective descriptive; no new blind confirmation was
obtained. Independent pre-run review did not pass; execution was
researcher-directed. Two completed B2 fits were imported from XPU and 16
remaining fits used CUDA. Hardware heterogeneity and historical technical
failures remain disclosed. The native label-to-CEMS reference-time chain is
UNVERIFIED. Sen1Floods11 evaluates water/non-water only.

See [reproduction instructions](docs/CODE_REPRODUCTION.md),
[public provenance](docs/EXECUTION_PROVENANCE_PUBLIC.md),
[data access](docs/DATA_ACCESS.md), [changes](docs/PUBLIC_RELEASE_CHANGES.md)
and [verification](RELEASE_VERIFICATION.json).

## License

MIT License applies to this project's author code; see the root `LICENSE`.
Copyright (c) 2026 Zhiyong Guo et al.

SMAGNet and other third-party code remain subject to their respective licenses.
Dataset licenses are separate and are not covered by this project's MIT License.
See `THIRD_PARTY_NOTICES.md`. This release was prepared locally; no repository
URL, DOI, journal supplementary URL or automatic upload has been assigned.
