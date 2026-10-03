# Public reproduction scope

## Installation and immediately reproducible artifacts

Create and activate a suitable Python 3.11+ environment. From this directory:

```console
python -m pip install -e ".[dev,impactmesh_audit]"
python scripts/render_rs_cached_main.py --output regenerated_main_artifacts
```

The renderer reproduces presentation from saved summaries only. It reads no
raw imagery, labels or model tensors. Compare each generated `.tex` table
with its counterpart in `examples/expected_tables/`, normalizing LF/CRLF
line terminators. All remaining text and numerical tokens must be identical.
The class decomposition must reconstruct the same reported Brier gains.
Plot byte identity is not claimed across plotting libraries/fonts.

## Verified existing test scope

```console
python -m pytest -q tests/test_availability.py tests/test_split_isolation.py tests/test_evaluation_counts.py tests/test_probability_counts.py tests/test_overlap.py tests/test_formal_metrics.py tests/test_external_binary.py tests/test_optical_benefit.py tests/test_upgrade_masks.py tests/test_upgrade_metrics.py tests/test_upgrade_20260929_runner.py tests/test_upgrade_b1_eval.py -k "not xpu_budget_blocks_unresolved_live_child"
```

Baseline and public checks use exactly these 12 existing synthetic/unit modules.
They produce 59 passed and 1 deselected on the verification host. The excluded
self-PID probe exercises a historical Windows incompatibility: the POSIX
`os.kill(pid,0)` process probe can terminate a Windows target. The original
scientific source was not patched as part of release sanitation. Training,
temperature-fitting and inference smoke/lifecycle tests are not part of this
verification command. No scientific experiment is re-executed.

## Scientific code versus historical execution control

Public release supports cached-result reproduction and scientific code
inspection, while historical execution-control metadata was excluded from
the public distribution. The root `upgrade_execution.py` is explicitly a
public derivative: only the administrative `require_dual_freeze` function
is replaced with an unconditional refusal. Every other AST node remains
identical. Scientific model, loss, mask, metric and ownership implementations
are unchanged. Configurations omit an administrative instruction field only;
their historical status values describe the original experiment, not a new
public executable freeze or authorization.

Historical training/calibration/evaluation scripts remain for inspection;
full execution requiring the excluded controls is intentionally unavailable.
Saved-result aggregators that bind the historical raw config hash likewise
cannot use the projected public configuration as an original attestation.
Use `render_rs_cached_main.py` for the verified public reproduction path.
No bypass or reconstructed historical approval is provided.

The unique CUDA implementation files remain under
`outputs/server_migration_20260929/package/`. This is a partial historical
source record after byte-identical duplicates and administrative records were
omitted; it is not a standalone executable repository. The omitted files and
their original hashes are documented in `docs/PUBLIC_RELEASE_OMISSIONS.json`
and `docs/PUBLIC_RELEASE_CHANGES.md`.

## Environments and external resources

Original Windows/XPU and Linux/CUDA dependency locks are historical environment
records. Their local build/install paths describe past environments rather than
current install locations; no private identity or credential is needed by the
public renderer. Exact wheel provenance and bitwise cross-device recomputation
are not generally verified. Two completed B2 models originated on XPU; the
other 16 neural fits used CUDA, with the original precision restrictions.

`pyproject.toml` gives base dependencies. XPU historical locks are not portable
install instructions. Research extras are listed in
`requirements/research-extra.txt`; select suitable PyTorch/torchvision builds
when inspecting hardware-specific implementations. This archive does not
promise that every historical runtime can be recreated from those extras.

Complete inference requires external raw imagery/native labels, model tensors,
historical runtime and excluded execution-control records. No real checkpoint
download URL is available in this distribution. See `docs/DATA_ACCESS.md`.

## Meaning of hashes

`PACKAGE_MANIFEST.json` binds actual public distribution bytes. It is a public
release manifest, not an experiment freeze manifest. Original digests in
cached provenance, run metadata and the public change inventories identify
historical original bytes; they are not assertions about a redacted public
file. The original source ZIP remains unchanged and is retained privately.
