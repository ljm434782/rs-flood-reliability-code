# G1 data-source and flood-process audit

**Audit date:** 2026-09-29  
**Scope:** metadata, source provenance, CEMS activation records and local catalogues only. No new candidate label pixels, label-derived target proportions, predictions, scores, or test results were opened. No new raster was downloaded. `UNVERIFIED` means the required evidence is absent from the inspected metadata, not that a dataset is faulty.

## G1 decision

**No new GEOID-Flood test process is cleared or frozen.** The available metadata supports, at most, a future S1-only three-class external sensitivity. It cannot support the study's single-time Sentinel-2 availability question, and it does not yet support a new fit/development/calibration/test partition.

Stop before acquisition, label access, inference, or confirmatory experiment design. This finding does not change the completed v1--v5 studies.

## Source-level feasibility

| Source | Provenance, version, licence and integrity record | Relevant semantics | G3 disposition |
|---|---|---|---|
| ImpactMesh-Flood v1, including ordinary validation/test and the prior four-event panel | Existing frozen manifests and CEMS product-time audits; CC-BY-4.0 source family and verified raw archives are recorded in the v1 manifests | Event-time S1 and native CEMS-derived `0/1/2/Ignore` masks; exact per-tile CEMS product linkage remains `UNVERIFIED` | **Not a new source.** Already-used/evaluated events cannot become a new blind role. |
| Sen1Floods11 v1.1 hand-labelled test | Official repository/GCS; 90 pairs, GCS MD5 and local SHA256 are listed in `data/manifests/sen1floods11_manifest.csv` | Binary water/non-water and `-1` Ignore only | **Not eligible** for a new three-class panel; it has been used for final external testing. Spain and Nigeria are excluded from the independent primary summary because of associated-event evidence. |
| STURM-Flood | Zenodo record `12748983`, version `1.0.0`, CC-BY-4.0; `Dataset.zip` 3,985,215,183 bytes, publisher MD5 `0e4172e74a4cf2e4c608c14d1588d1d9` | No product-level processing or target-time fields. At most 12 activations may contain spatially paired S1/S2 candidates; only three pass the same-product/pre-S1/common-valid-area diagnostic. CEMS product conflicts remain. | **Not feasible** for G3: insufficient verified independent processes and no verified single-time S2 state. |
| GEOID-Flood main tree | [Dataset card](https://huggingface.co/datasets/links-ads/geoid-flood), [author repository](https://github.com/links-ads/geoid-flood), local CEMS snapshot. Revision `868407460bf3db492f50730a57585916baa71dc6`; CC-BY-4.0 compilation/splits/labels, MIT code, Copernicus notices. `tile_catalog.parquet`: 633,270 bytes, SHA256 `c6ec7e3a3c252977069e9fa1613f07a0c01f7a121a0f55e9868f0645aad38022`; CEMS snapshot: 657,942 bytes, SHA256 `cee96a5dca221f5b716c58845633bc25c99439756058d5fcad9292304046ec23`; both retrieved 2026-09-27. | 10 m, 1024x1024; pre/post S1 GRD and RTC VV/VH, stored linear sigma0; `0/1/2/255` labels. S2 is a cloud-filtered **pre-event composite**, not a single selected observation. | **Potential future S1-only source, not feasible now.** No test process is clear; S2 cannot represent the optical-availability states. |
| GEOID-Flood `heldout` tree | Same revision and licence. Local heldout catalogue: 93,195 bytes, SHA256 `7b47920e779a6d561de1b855e5adf6826f9c734bd6d5a76a10d4016715ae03e6` | Same S1 and composite-S2 semantics | **At most one descriptive S1-only process** after exclusions; not a multi-process confirmation set. |

The GEOID card reports 219 CEMS activations in 65 countries, manually validated three-class labels, and post-event S1 GRD/RTC with two polarizations. It says the loader converts linear S1 to dB. Those facts permit a later input-only compatibility check, but do not link each target raster to a specific post-S1 acquisition.

## Historical event isolation

The audit compared GEOID against the frozen ImpactMesh process inventory, including all fit, development, calibration, internal-test, holdout, excluded, and previously evaluated four-panel activation IDs. Exact ID overlap is sufficient to reject a new blind role; a distinct ID can still be the same flood process.

### Heldout tree

The heldout catalogue has nine activations and 1,429 publisher-eligible tiles. Eight are previously used and excluded: `EMSR857`, `EMSR858`, `EMSR861`, `EMSR864`, `EMSR865`, `EMSR866`, `EMSR868`, `EMSR870`. Only `EMSR871` (45 eligible tiles) is absent. It is the 2026-03-31 Abruzzo/Molise/Basilicata flood, but still one process and its catalogue supplies delineation times rather than a verified target-product-to-post-S1 relationship. It is not a test panel.

### Main tree

From 212 activation IDs, applying the publisher's `is_valid AND invalid_pixel_frac <= 0.95` rule, excluding exact prior IDs and the four already scored panel events (`EMSR411`, `EMSR559`, `EMSR788`, `EMSR843`), then requiring at least five eligible tiles yields **51 activation candidates / 1,738 tiles**. This is only a source count.

The local CEMS time/centroid chain screen flags 22 candidates within 30 days and 500 km of a prior process. They cannot enter a new role without documented event-identity review:

`EMSR151`, `EMSR167`, `EMSR267`, `EMSR292`, `EMSR319`, `EMSR359`, `EMSR441`, `EMSR467`, `EMSR470`, `EMSR485`, `EMSR520`, `EMSR631`, `EMSR659`, `EMSR665`, `EMSR697`, `EMSR721`, `EMSR722`, `EMSR755`, `EMSR759`, `EMSR761`, `EMSR796`, `EMSR854`.

The 29 no-hit IDs are also unconfirmed: `EMSR154`, `EMSR174`, `EMSR176`, `EMSR188`, `EMSR199`, `EMSR205`, `EMSR235`, `EMSR257`, `EMSR258`, `EMSR326`, `EMSR337`, `EMSR349`, `EMSR410`, `EMSR416`, `EMSR431`, `EMSR446`, `EMSR483`, `EMSR548`, `EMSR557`, `EMSR562`, `EMSR568`, `EMSR572`, `EMSR577`, `EMSR646`, `EMSR662`, `EMSR720`, `EMSR758`, `EMSR776`, `EMSR794`.

No-hit is not independence evidence. `EMSR349` belongs to the Cyclone Idai chain with previously seen `EMSR347/348`; `EMSR758` belongs to the Storm Boris chain; both are excluded. CEMS states that `EMSR446` continued floods and repeated delineation over training `EMSR438` AOIs, so it is excluded. `EMSR483` is a Calabria cyclone one week before training `EMSR486` in Sardinia; their weather-system relation is `UNVERIFIED` and it is held out.

The publisher's split must not be used as a project role assignment. The local main catalogue puts 52 of 212 activations into more than one publisher split. Ten of the 51 candidates straddle publisher splits: `EMSR199`, `EMSR258`, `EMSR349`, `EMSR431`, `EMSR441`, `EMSR446`, `EMSR485`, `EMSR557`, `EMSR562`, `EMSR796`. The project must place every evidence-linked flood process in exactly one role.

## Test-split chain check

The prior input-only test screen had seven exact-ID-unseen candidates with at least five eligible tiles: `EMSR349`, `EMSR446`, `EMSR483`, `EMSR557`, `EMSR572`, `EMSR662`, `EMSR758`. `EMSR349`, `EMSR446`, `EMSR758`, and now `EMSR662` are excluded; `EMSR483` is held. `EMSR557` and `EMSR572` are unconfirmed.

| Activation | CEMS evidence checked | Blocking gap |
|---|---|---|
| [EMSR572](https://mapping.emergency.copernicus.eu/activations/EMSR572/) | 2022-04-01 Bosnia and Herzegovina heavy-rain/river-overflow flood; one AOI and one delineation product, delivered 2022-04-03 | No exact post-S1 acquisition or raster-target-to-product/source-image link. |
| [EMSR557](https://mapping.emergency.copernicus.eu/activations/EMSR557/) | Norway extreme-weather Gyda, rainfall and snowmelt; 13 AOIs and multiple delineation deliveries | Multi-AOI/product chain cannot yet identify each target's mapping-time reference. |
| [EMSR662](https://mapping.emergency.copernicus.eu/activations/EMSR662/) | Later public-detail metadata identifies one CEMS delineation product sourced from Sentinel-2 at 2023-05-14 07:56 UTC | Excluded: the source image is optical, so the GEOID post-S1 input cannot be the label's S1 mapping anchor. |

Thus the currently cleared main-tree test list has **0 processes**. “At most three” was a pre-check upper bound, not a cleared count.

## Whether roles can be constructed

| Intended use | Answer now | Evidence needed before assignment |
|---|---|---|
| GEOID S2 for full/partial/missing/asynchronous optical states | **No** | The released S2 is a composite and has no single acquisition/availability state aligned with post-S1. |
| Existing S1-only checkpoint on GEOID as three-class external sensitivity | **Not yet** | For each frozen event: post-S1RTC band order, dB conversion, nodata, grid/CRS/transform, exact S1 time, label reference/product time and process independence. |
| GEOID fit/development/calibration/test programme | **Not constructible yet** | CEMS AOI/narrative process graph for the candidates, grouping cross-split activations and allocating all linked IDs to exactly one role without using labels or model outcomes. |
| Existing ImpactMesh fit/development/calibration plus GEOID external test | **Not constructible yet** | A multi-process independent GEOID test list first; the global temperature source must be declared before testing and cannot be fitted from GEOID test labels. |
| EMSR871 alone | **Descriptive only if separately frozen** | Never a substitute for multi-process comparative evidence. |

The number of **confirmed new processes is 0**. The only conditional single-process possibility is EMSR871. G1 cannot estimate a precision interval from development-effect variance without reading model outputs, which is outside this audit. G2 must provide the frozen development-process variance/effect estimate and the G3 target interval; G1 can then check whether the independently confirmed process count meets that predeclared requirement. No role count should be reverse-engineered from a test result.

## Storage and acquisition boundary

The D: drive has 370,895,155,200 bytes free (about 370.9 GB). No GEOID data were acquired here.

| Future acquisition, only after G3 freeze | Publisher/audited size | Current implication |
|---|---:|---|
| Full GEOID release | about 584--588 GB | Does not fit free space. |
| Main test S1RTC plus labels | about 36.15 GB S1RTC + 0.02 GB labels | Physically possible, but no test is frozen and labels must remain untouched until locked G5. |
| Heldout S1RTC plus labels | about 20 GB | Physically possible, but supports only EMSR871. |
| Main all-split S1RTC plus labels | about 192.2 GB + 0.1 GB | May fit with few download workers for a separately approved S1-only programme, but is unjustified while roles and timing are unresolved. |

The publisher warns that download peak includes selected data plus roughly 2 GB per in-flight worker. Any later acquisition must remeasure free space, pin URLs/revision/hashes/licence notices, and preserve raw files immutably.

## Stop recommendation

Do not acquire candidate rasters, access labels, run inference, or create G3 training plans from GEOID now. The next minimal G1 work is a metadata-only CEMS event graph for the 29 no-hit IDs, beginning with the two remaining test IDs (`EMSR557`, `EMSR572`) and held `EMSR483`, followed by an input-only source-time check for any survivor. Only if that yields enough independent processes for the G3 predeclared role counts and precision target should the coordinator freeze revision, process roles, tile manifest, checkpoint, temperature source, metrics and stopping rule. Then, and only then, recommend bounded S1RTC-input acquisition; test labels belong solely to the locked G5 evaluation.

## Addendum: bounded CEMS graph and alternative-source prospecting (2026-09-29)

This addendum used only CEMS public activation metadata and source descriptions. It did not download a raster or open label, target-proportion, prediction, or score data.

### Remaining GEOID test candidates

The CEMS public-detail endpoint returned a complete record for `EMSR662` but returned HTTP 403 for `EMSR483`, `EMSR557`, and `EMSR572` during this pass. That access difference prevents treating a missing field as evidence of a source-time match.

| Activation | Additional public record | Process conclusion | S1-anchor / target-time conclusion |
|---|---|---|---|
| `EMSR483` | The preserved CEMS index gives 3 AOIs and 3 products for the 2020-11-22 Calabria flood. Its [activation record](https://mapping.emergency.copernicus.eu/activations/EMSR483/) identifies a November cyclone; project training `EMSR486` began six days later in eastern Sardinia. Detailed API was 403. | A common Mediterranean storm system cannot be excluded. **Hold.** | No source-image record can be tied to a GEOID tile. |
| `EMSR557` | The [activation record](https://mapping.emergency.copernicus.eu/activations/EMSR557/) describes Gyda, rainfall and snowmelt across 13 AOIs/products. Detailed API was 403. | No positive overlap is known, but independence is **unconfirmed**. | Multiple AOIs/products and no tile-to-product field prevent a target reference-time claim. |
| `EMSR572` | The [activation record](https://mapping.emergency.copernicus.eu/activations/EMSR572/) describes one Bosnia and Herzegovina AOI/product after heavy rain. Detailed API was 403. | No positive overlap is known, but independence is **unconfirmed**. | Catalogue event/delineation times do not prove the post-S1 acquisition or target product. |
| `EMSR662` | The [CEMS public-detail record](https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/?code=EMSR662) gives event time 2023-05-11, one AOI/product, a Sentinel-2 source image at 2023-05-14 07:56 UTC, and delivery at 2023-05-16 18:04 UTC. It records cancelled monitoring because cloud-free optical imagery could not be acquired. | No documented historical-process link was found. | **Exclude from S1-anchored testing:** its delineation source is optical S2, so a GEOID post-S1 input is not the target mapping anchor. |

Positive links remain `EMSR349`--Idai (`EMSR347/348`), `EMSR446`--training `EMSR438`, and `EMSR758`--Storm Boris (`EMSR756/757/763/766`). `EMSR662` is now excluded on time/modality grounds. `EMSR483` is held; `EMSR557` and `EMSR572` are unconfirmed; the frozen-test count remains **zero**.

For the other no-hit IDs, the available CEMS index supplies only activation time, centroid, country, AOI/product counts and short narratives. It supplies neither an all-event storm/process graph nor a GEOID-raster-to-CEMS-product key. A manual expansion would add names without establishing the required identity or target-time proof, so this bounded metadata pass stops here.

### Three alternative open sources considered

| Source | Metadata evidence | Compatibility result |
|---|---|---|
| [Kuro Siwo labelled GRD webdataset](https://huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-Webdataset) | CC-BY-4.0; 43 events (2015--2022), 10 designated test events; one post-flood and two pre-flood Sentinel-1 VV/VH scenes; labels `0` no water, `1` permanent water, `2` flood, plus validity mask. Per-sample `info.json` holds metadata. Published total: 1.96 TB. | **Future S1-only possibility only.** It has no S2 state, exceeds local capacity as published, and this pass cannot establish revision/checksum, event/CEMS IDs, source-label time, or all historical-project overlaps. No event is frozen. |
| [SenForFlood](https://github.com/GEOHUM-PLUS/SenForFlood) | Public description: 353 events, before/during S1 and S2, CEMS/DFO flood masks, about 254 GB. It explicitly reports flood detection **without permanent water bodies**. A machine-readable release licence and target/Ignore specification were not established. | **Incompatible at this gate.** It does not preserve the project `non-water/permanent-water/flood` target; licence, event identity, reference timing and split isolation remain unverified. |
| [CEMS Global Flood Monitoring layer catalogue](https://confluence.ecmwf.int/spaces/CEMS/pages/242067378/Web%2BLayers) | Documentation describes S1 observed-flood extent, reference-water mask, S1 footprint and acquisition metadata. | **Not an independent reference-label source.** The flood/water layers are S1-derived from the same observation, with no packaged paired raw-input/three-class/Ignore event benchmark or event-isolation manifest. |

The official [CEMS product manual](https://mapping.emergency.copernicus.eu/about/rapid-mapping-manual/product-overview/what-is-delivered-in-a-product/) confirms that products are maps plus geospatial datasets and may have new versions. It is useful for provenance but not a ready independent benchmark; reconstruction would need a separately authorised label-construction protocol and full event/S1/S2 audit.

**Prospecting result:** no alternative source is ready for G3. Kuro Siwo is the sole additional source with released three-class S1 label semantics, but it remains a future S1-only candidate pending a pinned metadata manifest, historical-overlap audit, source-time linkage and capacity-bounded acquisition plan. The stop recommendation remains in force.

## Addendum: Kuro Siwo pinned-release feasibility (2026-09-29)

This bounded follow-up inspected only the author repository, its catalogue YAML, the Hugging Face revision and its LFS file manifest. It did not download a WebDataset/GeoTIFF shard, open `mask.npy`, `valid_mask.npy`, `info.json`, target proportions, predictions or model outputs.

### Release identity, licence and file manifest

| Item | Metadata finding | Audit consequence |
|---|---|---|
| WebDataset release | [Hugging Face Kuro-Siwo-Webdataset](https://huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-Webdataset) revision `e8e61b7b1254b04bfa2b5e26d43db90cb86b5004`, last modified 2026-07-29 | This is a usable revision pin for a future source manifest. The API LFS object IDs are SHA-256 content IDs for every published shard. |
| Author repository | [KuroSiwo commit `4347ed173c4e48f5a9d578bb5fe8453706b08e5e`](https://github.com/Orion-AI-Lab/KuroSiwo/tree/4347ed173c4e48f5a9d578bb5fe8453706b08e5e); catalogue YAML blob `c96366ff0150345a8ec7083adf96675c970fa710` | The catalogue exposes event keys, AOIs and a `ref_date`, but no released mapping of WebDataset test shards to event IDs. |
| Licence | The dataset card declares CC-BY-4.0 and the repository README calls the data CC-BY. The repository `LICENSE` is MIT for code, while the 2024 paper states MIT for “dataset and BlackBench.” | Treat **CC-BY-4.0 as the intended Hugging Face asset licence only after publisher confirmation**. The contradictory paper wording prevents a fully closed licence record today. |
| Test GRD storage | 12 LFS TAR shards, 122,774,292,480 bytes = 114.34 GiB; every shard has a published LFS SHA-256 object ID. The 5 train-GRD shards total 43.55 GiB; both total 157.90 GiB. | The entire GRD test archive can fit the current free 345.4 GiB, leaving about 231 GB decimal before extraction. A raw-plus-extracted workflow is also within raw capacity but must remeasure disk before any freeze. **Selective test-event retrieval is not possible from the published shard manifest** because shard-to-event membership is absent. |

The author card describes 43 events and 10 designated test events. Its YAML catalogue instead has 45 event keys, including non-EMSR field-labelled identifiers (`1111002`--`1111013`). Neither source identifies which ten events are in `test_GRD`; the current Hugging Face tree lists only `shard-00000.tar` through `shard-00011.tar`. This event-count/split mismatch is an independent blocker for a blind-role manifest.

### Process-overlap result

Matching the 45 YAML event keys against every previously used ImpactMesh process and the two formerly scored four-panel events finds **21 exact activation overlaps**:

`EMSR273`, `EMSR275`, `EMSR277`, `EMSR279`, `EMSR324`, `EMSR411`, `EMSR417`, `EMSR421`, `EMSR437`, `EMSR445`, `EMSR492`, `EMSR497`, `EMSR498`, `EMSR502`, `EMSR514`, `EMSR518`, `EMSR554`, `EMSR555`, `EMSR559`, `EMSR561`, `EMSR567`.

This alone prevents treating Kuro Siwo's publisher test split as independent from this project. The 24 remaining catalogue keys include CEMS events and non-EMSR identifiers, but their process relationships cannot be assessed without a pinned event-to-shard mapping and a CEMS AOI/narrative review. No event is cleared. In particular, the known Kuro catalogue also contains `EMSR470`, `EMSR520`, and `EMSR562`, which have already required caution in GEOID/STURM source screening even though they are not exact IDs in the frozen ImpactMesh process table.

### Time, labels and modality compatibility

The [dataset card](https://huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-Webdataset) publishes two pre-event and one post-flood Sentinel-1 VV/VH scene, a DEM, three target values (`0` no water, `1` permanent water, `2` flood), and a separate validity mask. The author paper says experts photointerpreted processed GRD images to create the three classes, and that the post-event image was selected as close as possible to the actual event date; it reports a mean post-event gap of 3.6 days and median of one day. These are source-level descriptions, not an event/tile-level target-time proof.

The author catalogue's `ref_date` values are event-level timestamps and its code schema names `flood_date`, `source_date`, `s1_ids`, `revision` and `version`, but the metadata-only source files do not define an exact relation among a label, its photointerpretation inputs, a post-S1 acquisition, and an authoritative flood reference time. The published WebDataset card does not specify a numeric Ignore code or the evaluation treatment for invalid pixels beyond providing `valid_mask.npy`. Therefore:

- The released class taxonomy is compatible in name with the study's three classes.
- The required per-event S1-anchor and label-reference-time relationship is **UNVERIFIED**.
- The labels are manual SAR photointerpretations, so they are not an independent non-SAR reference for the same post-S1 input; the degree of per-tile dependence cannot be determined from the inspected metadata.
- Kuro Siwo has no released Sentinel-2 observation for the study's full/partial/missing/asynchronous optical states. It can only ever be considered as an S1-only sensitivity.

### Final Kuro Siwo disposition

Do not acquire even the 114.34 GiB test GRD archive now. A future re-audit would first need an author-published, revision-pinned event-to-shard/index manifest; a reconciled data licence; every event's exact acquisition, target-reference and validity/Ignore semantics; and CEMS process isolation against the complete project history. Only after those checks could a coordinator freeze a subset before accessing its labels. Kuro Siwo is consequently **not a feasible G3 source at this gate**.

## Addendum: ImpactMesh S2 band order and technical-preflight boundary (2026-09-29)

This addendum is a read-only metadata check for the frozen ImpactMesh fit set. It inspected existing JSON/CSV manifests and the reader source only. It did not open raw imagery, pixels, labels, predictions, scores, or any test/holdout archive.

### Established band mapping

Every applicable pre-existing sample-audit JSON reports the same `S2L2A.band_names` order:

`[B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12]`.

The band-axis index is zero-based, so the required SMAGNet optical order `B04, B03, B02, B08` maps exactly to **`(3, 2, 1, 7)`**. The same JSONs report S2 shape `(4, 12, 256, 256)`. There is one observed order and one observed shape; no conflicting band name or band count appears in the applicable audit evidence.

### Coverage and evidence limits

| Evidence | Coverage/result | SHA-256 |
|---|---|---|
| `impactmesh_sample_audit_EMSR*.json` applicable to the train-eligibility manifest | 25 sample IDs occur in the frozen train eligibility table; 18 are paired eligible, including **16** in the frozen `train` role. All 25 have the order above. The deterministic SHA-256 of sorted `filename SHA256` lines for the 27-file audit collection is `b233834fe35f87d60c3ed69dd3afcff4fcb3c925f366af2cdcb70bd2255bf77b`. | Per-file hashes are represented by that deterministic review aggregate. |
| `impactmesh_sample_audit.json` | One additional convenience audit with the same S2 order and shape. It is corroborating evidence only. | `ac87f47373aec09f16f47a21688bb4b431bf0908f946d01ef2cf4fd65f8d802f` |
| `impactmesh_all_train_frozen_eligibility.csv.gz` | 19,448 official-train rows, 11,538 paired eligible. Joining to the frozen process-role table yields **4,751 paired-eligible frozen-train samples**. It records a selected S2 time index, but no band-name field. | `92925c8289c81041d99a8b67131de7e025861094386881daf44d12511b30d9a0` |
| `impactmesh_all_train_frozen_eligibility_meta.json` | Pins the S2 source archive as `train/S2L2A.tar`, SHA-256 `0a58335297971c4a6e5e4f4e4aa8d2afe9201dedaad35e08babd03f3a1bb0c6f`. | `5d0e4865729b518427ff6fab715e15829e9fef8fee7470c5ea9db02fada65acd` |
| `impactmesh_process_split_frozen.csv` | Supplies the role join used for the 4,751-count calculation. | `f0a687ac62581e139a27a7e747261cc6a0f18d45a05728ec2893442f386a696e` |

The 16 audited, paired-eligible frozen-train samples are only a small metadata sample of the 4,751 frozen fit samples. The frozen eligibility table has no per-sample S2 `band_names`, so this evidence **does not establish the band order for every fit sample**. Its selected-time field only takes indices 1 and 2 among paired-eligible rows; it does not describe the channel axis. The current reader validates the 12-channel array shape but does not validate the named channel sequence. No inconsistency was observed; complete coverage is nevertheless missing.

### Required technical-preflight gate

Before any candidate training, run a metadata-only preflight over **all 4,751 frozen fit sample IDs** that reads each S2 Zarr `band` coordinate and fails closed unless it equals the exact 12-name sequence above. The preflight must record the frozen manifest revision and output an audited count plus a SHA-256 of its per-sample metadata result. A missing coordinate, a different order, a duplicate name, or any count other than 12 is a blocking failure; it must not be repaired by positional guessing.

The preflight is a technical check, not training: it **must not call `optimizer.step()` or otherwise update model weights**. It must not save a checkpoint, fit normalisation or calibration statistics, tune a threshold, or write a result usable for model selection. A synthetic-tensor forward/backward resource check is permitted only under the same no-update constraint; any real fit-batch throughput check may read inputs only and must not read labels. These constraints match the existing upgrade-protocol gate and keep the fixed training sample set and all evaluation partitions outside this check.

## Addendum: full frozen-fit S2 `band` coordinate verification (2026-09-29)

This is the required full-coverage, metadata-only closure of the preceding band-order gate. The frozen eligibility table was joined to the frozen role table and selected exactly the rows with `paired_eligible=1` and `role=train`. The result was **4,751 unique sample IDs**. Only `data/raw/impactmesh/train/S2L2A.tar` was opened; the procedure enumerated TAR headers, then read only each selected ZIP's `band/.zarray`, `band/.zattrs`, and `band/0` members. It did not read any `bands/*` member, S1 archive, label, pixel, prediction, score, calibration or evaluation-split asset.

| Check | Result |
|---|---:|
| Frozen fit IDs selected / unique | 4,751 / 4,751 |
| S2 TAR members indexed | 19,448 |
| Missing frozen-fit S2 members | 0 |
| Duplicate frozen-fit IDs / duplicate S2 TAR members | 0 / 0 |
| Coordinate I/O or Zarr-format errors | 0 |
| `band` coordinate with duplicate names | 0 |
| Coordinate differing from `[B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12]` | 0 |
| Exact matches | **4,751** |

Thus every frozen fit sample has the required named 12-channel sequence. The SMAGNet input extraction remains `B04,B03,B02,B08 = (3,2,1,7)` with zero-based indexing. This establishes the metadata requirement only; it does not validate image values, labels, training behaviour or any result.

### Reproducibility evidence

| Evidence item | SHA-256 |
|---|---|
| Frozen eligibility CSV | `92925c8289c81041d99a8b67131de7e025861094386881daf44d12511b30d9a0` |
| Frozen process-role CSV | `f0a687ac62581e139a27a7e747261cc6a0f18d45a05728ec2893442f386a696e` |
| Frozen-eligibility metadata file | `5d0e4865729b518427ff6fab715e15829e9fef8fee7470c5ea9db02fada65acd` |
| Source `train/S2L2A.tar` identity recorded by that metadata | `0a58335297971c4a6e5e4f4e4aa8d2afe9201dedaad35e08babd03f3a1bb0c6f` |
| Deterministic verification result: SHA-256 of sorted `sample_id + TAB + observed comma-separated band names + LF` lines | `1fbbf076be758b8f925dee778db1838a6dc6328cbda60d2bd0d18fa6a9d93a1b` |

The verification was fail-closed: it would have stopped and reported the first selected ID if a required coordinate member, ZIP/Zarr decode, TAR member, or expected name sequence failed. It passed without an exception. This closes the S2-name/order entry gate for the frozen 4,751-sample fit set. The no-update boundary in the preceding section remains mandatory for all technical preflights: no `optimizer.step()`, weight mutation, checkpoint, fitted statistic, calibration or threshold selection is permitted.

## Addendum: current frozen fit pool versus the G2/v5 3,531-tile pool (2026-09-29)

This is a manifest-ID-only reconciliation. It read the frozen eligibility CSV and the current, legacy-grouped and v5 process-role CSVs. It did not open imagery, coordinates, pixels, labels, predictions, scores or any evaluation asset.

The two numbers describe different declared sample sets, even though both use the same eligibility CSV:

| Set | Selection rule reproduced from its own manifest | Process/unit count | Unique paired-eligible sample IDs |
|---|---|---:|---:|
| Current frozen fit set (A) | `paired_eligible=1` and `role=train` in `impactmesh_process_split_frozen.csv` | 85 current activation-level roles | **4,751** |
| Historical G2/v5 fit set (B) | `paired_eligible=1` and the activation's `process_id` in a `v5_role=fit` row, using `impactmesh_process_split_grouped.csv` plus `impactmesh_v5_training_dev_processes.csv` | 65 legacy grouped processes | **3,531** |

The v5 file itself declares 3,531 tiles, and exact ID reconstruction produces the same count. The set relation is:

| ID relation | Count | SHA-256 of sorted sample IDs, one per LF-terminated line |
|---|---:|---|
| A (current frozen) | 4,751 | `e2947d7dd348de2f289e949b5c5d02f146a80716a8d45d5ea7b67729fa97e989` |
| B (historical G2/v5) | 3,531 | `92d651073d502831a8a22e383f0476c9c489941e7bbf7ecdc0239bcbb2b53be0` |
| A ∩ B | 3,496 | `fc74932f4092da54f828a44c087169a858bc196bd1e07f301eb285a905865b9b` |
| A − B | 1,255 | `bb300ab9df1f878beb021d3492e0e04ea9dc7e302df064129e849d80ee1daf59` |
| B − A | 35 | `1234d2fb01ba4c2311cc4169e0b5cd0580756471fb6657a6d261f173a93d8a6d` |

`A − B` contains 20 current activations: `EMSR265` (74), `EMSR273` (12), `EMSR279` (166), `EMSR347` (64), `EMSR388` (109), `EMSR422` (52), `EMSR427` (150), `EMSR445` (12), `EMSR477` (25), `EMSR489` (10), `EMSR514` (5), `EMSR555` (169), `EMSR565` (11), `EMSR639` (149), `EMSR680` (21), `EMSR728` (107), `EMSR778` (21), `EMSR850` (39), `EMSR855` (13), and `EMSR868` (46). `B − A` contains `EMSR231` (5) and `EMSR550` (30).

The difference is explained by **manifest revision and process grouping**, not by duplicate tiles or a changed paired-eligibility predicate. The historical grouped table joined `EMSR231` to process `EMSR228` and `EMSR554` to process `EMSR550`; v5 therefore counted those tiles in its 65-process fit set. The current frozen table assigns `EMSR231` and `EMSR550` to validation while assigning `EMSR554` to train. It also makes current train roles of activations that the grouped table placed within validation/calibration processes, for example `EMSR422 → EMSR421` and `EMSR778 → EMSR777`. Sixteen legacy v5-development processes also occur in A as current train roles. These are role/grouping changes, not a numerical rounding issue.

### Consequence for the G2 diagnostic

The historical 3,531-tile coverage calculation is internally reproducible for its old 65-group v5 fit set. It is **not a coverage statement for the current 4,751-ID frozen fit set**. In particular, its 65-process, 24-epoch process-balanced schedule and the reported 3,247/3,531 unique-tile coverage cannot be presented as the schedule or coverage of the current frozen training set without a new, ID-explicit schedule calculation.

Before G3 begins, the coordinator must select one role/grouping manifest as the formal fit definition, attach its set hash to the protocol and derive the process-balanced schedule from that exact ID set. If the current 85-role manifest is chosen, the 3,531/65-process G2 diagnostic remains historical context only. If the legacy 65-group set is proposed instead, its process grouping and every cross-activation role need to be re-frozen and checked against the study's independent-flood-process rule. This audit does not determine which process definition is scientifically valid.

| Input manifest | SHA-256 |
|---|---|
| Frozen eligibility CSV | `92925c8289c81041d99a8b67131de7e025861094386881daf44d12511b30d9a0` |
| Current frozen process-role CSV | `f0a687ac62581e139a27a7e747261cc6a0f18d45a05728ec2893442f386a696e` |
| Legacy grouped process-role CSV | `537617bdabeaa2e0fd1c01a11282d425d31c1d477315159aad0b68eb62e52e95` |
| v5 process-role CSV | `6ab6186699e2e22bc672e54ec5f3ba0efdc4e7edb3fc17a7667b29a4f080e0a9` |
| v5 process-role metadata JSON | `0d9e6ce6ed9e9847ff56111cc0bd9727b762b7226a0bf29414a5d36d63be661a` |

## Addendum: known process-family graph and role-isolation re-audit (2026-09-29)

This re-opens G1's process-isolation check because the current activation-level role table conflicts with already recorded shared-event evidence. The audit used only existing CEMS-backed event notes, source/role manifests and sample IDs. It did not open imagery, pixels, labels, predictions, calibration outputs or scores.

### Evidence-supported family graph

The four two-activation edges in `impactmesh_process_groups.csv` are explicitly backed by the CEMS/JRC evidence recorded in `docs/PROCESS_IDENTITY_AUDIT.md`: North-West Ireland (`EMSR228`--`EMSR231`), Storm Gloria (`EMSR421`--`EMSR422`), Australia floods (`EMSR550`--`EMSR554`) and Tropical Storm Sara (`EMSR777`--`EMSR778`). They are confirmed family components, not proximity-screen guesses.

| Confirmed family | Current activation roles | Historical grouped role / G2-v5 status | Isolation finding |
|---|---|---|---|
| `EMSR228; EMSR231` | train; validation | one process `EMSR228`, train / v5 fit | **Conflict in current table.** |
| `EMSR421; EMSR422` | validation; train | one process `EMSR421`, validation | **Conflict in current table.** |
| `EMSR550; EMSR554` | validation; train | one process `EMSR550`, train / v5 fit | **Conflict in current table.** |
| `EMSR777; EMSR778` | calibration; train | one process `EMSR777`, calibration | **Conflict in current table.** |
| Hurricane Irma: `EMSR232; EMSR233` | validation; calibration | validation; calibration | Confirmed after the grouped split. Existing primary-evaluation exclusion removes `EMSR233`; the source role CSV still encodes the cross-role pair. |
| Storm Boris: `EMSR756; EMSR762; EMSR757; EMSR763; EMSR766` | calibration; internal test; holdout; holdout; holdout | same roles | Confirmed after the grouped split. Existing primary-evaluation exclusion removes `EMSR762`, `EMSR757`, `EMSR763`, and `EMSR766`; it retains calibration `EMSR756`. |

Thus the current 85 activation-level table cannot be a process-disjoint formal role definition. It places all four previously merged family pairs across active roles. The historical grouped table keeps those four components intact, but it alone does not encode the later Irma/Boris exclusions.

The following recorded associations involve past or candidate evaluation sources. They are not permitted to create a new independent evaluation role: `EMSR349` belongs to Cyclone Idai with historical `EMSR347/EMSR348`; `EMSR446` continues the `EMSR438` Uganda flood; and candidate `EMSR758` belongs to Storm Boris. They are already blocked in the G1 source screen. Separately, the existing Sen1Floods11 primary external summary excludes its Spain and Nigeria events because the records link them to ImpactMesh training `EMSR388` and `EMSR314`, respectively. `EMSR348`, `EMSR349`, `EMSR446` and `EMSR758` are not active members of the current train-role manifest; this audit records the links so that they cannot be reintroduced as supposedly independent data.

`real_missing` is an input-availability state, not an independent process role. In the historical v5 split it is drawn from the 16 historical development processes: 1,279 input-only rows across 11 process IDs, with sample-ID-set hash `a2d3267aa0f878f49f267ac7eaf1e91d23ef52b83bd6193b5a388ac16ede9deb`. It must inherit the process role of its parent tile and cannot be used to bypass a family exclusion.

### Minimum conservative role repair, derived without outcomes

The smallest available mechanical repair is to start from the existing CEMS-supported grouped manifest rather than the current 85-role activation table, then apply the existing post-training primary-evaluation exclusion closure. This is a **containment proposal**, not a newly frozen split and not evidence that every remaining component is independent:

| Proposed status | Rule | Processes | Paired-eligible sample IDs | Sample-ID-set SHA-256 |
|---|---|---:|---:|---|
| Fit | Historical v5 `fit` process groups | 65 | 3,531 | `92d651073d502831a8a22e383f0476c9c489941e7bbf7ecdc0239bcbb2b53be0` |
| Historical development, no access this round | Historical v5 `dev` process groups | 16 | 867 | `55707d6ece4c0caee79a4e5810906c30258b079a1bba5f1d95e3c90c144213d7` |
| Calibration, effective primary set | Grouped calibration minus `EMSR233` | 19 | 2,146 | `9901a1a564b772018cb94731bf360459535a877343d6589773a6494e6613f9bf` |
| Retrospective internal test, effective primary set | Grouped internal test minus `EMSR762` | 19 | 2,120 | `dd74f19fa0d39279d5d121305174897204d3f3de20234a9a18506f4aa71713ba` |
| Retrospective official holdout, effective primary set | Holdout manifest minus `EMSR757`, `EMSR763`, `EMSR766` | 9 activations | 492 metadata-listed samples | `0da66078bb76868a86cc49bf1230826c42cb055823b25db3e5a9cf43690b4c21` |
| Excluded paired, pre-existing | Grouped `excluded_paired` role | 20 | 53 | `40b3fa0d775574fd537ca509b26382440412c817ad6d494c4ff6a573be4a018f` |

For reference, the historical grouped validation inventory has 20 processes and 2,805 paired-eligible IDs (`8063636d6fff75df8b08380d8b1b717fd248d745a95fc66ce37391cf5f1f390c`); it is not a G3 fit or calibration source under the current draft. The 16-process historical development pool and every retrospective evaluation inventory must remain unavailable for training, checkpoint selection, normalization, threshold selection and model choice.

The repair preserves every one of the four confirmed two-activation components in a single role. It also gives the Irma/Boris components only one active primary role: validation keeps `EMSR232` while `EMSR233` is excluded; calibration keeps `EMSR756` while `EMSR762/757/763/766` are excluded. The fit pool contains neither family. The holdout count and hash above are from the sample-ID manifest only; no holdout asset was opened.

### Freeze disposition

**Do not freeze either the current 85-role table or the containment proposal as a scientifically complete process split yet.** The source process audit screened 37 proximity/time candidates but could establish only four within-train shared-event edges, and it explicitly warns that wider or more distant common disasters can be missed. The G1 source screen also has unresolved event identities and no newly cleared independent test process. The minimum repair removes every *known* active-role conflict, but it cannot prove absence of undiscovered family links or certify the legacy 65-group definition without a complete, versioned CEMS process-family review.

Before G3, the coordinator must freeze one graph that includes the six confirmed components above, the external/candidate exclusion edges, and any subsequently verified CEMS links; assign each component to exactly one active role or `excluded`; then regenerate every role's ID set and schedule from that graph. No historical role file should be overwritten, and no prior evaluation output may guide those assignments.

| Evidence file | SHA-256 |
|---|---|
| `docs/PROCESS_IDENTITY_AUDIT.md` | `105dcf0ca727142760371dafe363d9789a4cc3dbe9b0e639ba3b1f13a9fdf5cd` |
| `docs/NEW_BLIND_SOURCE_AUDIT.md` | `08625bb57aff1f8a3679fe547c89557aa77e94cd5bfa958d8949338ed14334e3` |
| `docs/EXTERNAL_TEST_AUDIT.md` | `f8b71e6376c26c955a5250bf50ea33baa4f246abdfdb849f50d26891df8a8671` |
| Grouped process roles | `537617bdabeaa2e0fd1c01a11282d425d31c1d477315159aad0b68eb62e52e95` |
| Process-family components | `b4859cc26b37a2c5e4343f14e628318170e17633d958edc65100f7c82d70165d` |
| Current activation roles | `f0a687ac62581e139a27a7e747261cc6a0f18d45a05728ec2893442f386a696e` |
| Historical v5 roles | `6ab6186699e2e22bc672e54ec5f3ba0efdc4e7edb3fc17a7667b29a4f080e0a9` |
| Primary evaluation exclusions | `9db408f7762065e736c9dc3ef14388d6af2836232892bca89f894f0e13ff4b4b` |
| External-process exclusions | `d7e5498b191084b5ec81326527d68f23b3f078247dd5488e250466f961179978` |

## Addendum: bounded CEMS source scan for all role activations (2026-09-29)

This is a finite, falsifiable source scan for the role-isolation issue. It uses the two locally preserved CEMS activation-API snapshots, the normalized CEMS activation manifest, the fixed 37-pair proximity screen, the recorded CEMS/JRC/public-bulletin decisions, and role/sample-ID manifests. It does not use a label, image pixel, model output or previously viewed metric.

### Fixed scope and completion criterion

The scan universe is the union of all current and historical grouped roles: **194 activation IDs** (182 official-train plus 12 official-holdout IDs). All 194 occur in the full local CEMS API snapshot with a non-empty event name, event date, country field, AOI count and product count. The snapshot has a narrative/search excerpt for 116/194; 78 lack one. It supplies AOI/product **counts**, not a complete per-AOI geometry, product/source-image chain or hydrological event graph.

“Relationship evidence scan complete” has the limited meaning that (1) each of these 194 activation records was found in the versioned local snapshot; (2) all 37 pre-existing time/centroid candidate edges have one of the three statuses below and a recorded source class; (3) all distinctive storm/cyclone names in this fixed corpus were checked for repeated names, and the already recorded Hurricane Irma, Tropical Storm Sara and Storm Boris bulletin links were added even where the proximity screen did not capture them. It **does not** mean that CEMS globally proves no other shared flood process exists.

### All 37 proximity candidates

The status hash below is SHA-256 of sorted `activation_a,activation_b<TAB>status<LF>` lines: `7bb2c417c020527413d24383f1914d406c56a22b517377f5f83cf1c75d4ece3e`.

| Status | Candidate edges | Evidence class |
|---|---|---|
| Same process, evidence-supported | `228--231`; `421--422`; `444--445`; `550--554`; `777--778` | The first, second, fourth and fifth are the CEMS/JRC-supported merges in `PROCESS_IDENTITY_AUDIT.md`. For `444--445`, local official CEMS narratives describe the same eastern-European heavy-rain flood: Ukraine in one record and the Upper Prut catchment continuing through Ukraine, Romania and Moldova in the other. This edge is evidence-supported but has not been added to the legacy grouped manifest. |
| Different processes, evidence-supported | `271--273`; `280--283`; `402--405`; `472--475`; `481--482`; `564--565`; `567--570`; `762--768`; `762--771`; `855--858`; `233--243`; `329--333` | Nine are explicit distinct-event decisions in `PROCESS_IDENTITY_AUDIT.md`. The remaining three are named distinct cyclones in the CEMS snapshot: Matmo/Nakri, Batsirai/Emnati and Eta/Iota. |
| **UNRESOLVED** | `215--261`; `273--275`; `323--324`; `323--329`; `330--333`; `469--472`; `469--475`; `477--481`; `477--482`; `486--487`; `517--518`; `683--684`; `706--708`; `756--764`; `762--764`; `762--770`; `764--768`; `768--770`; `768--771`; `861--864` | The fixed snapshot gives name/date/country/AOI count, but no sufficient event-family proof. Near time, country, centroid or a generic “flood/storm” title is deliberately not promoted to a merge. |

The cross-region name scan confirms the known components `EMSR232--EMSR233` (Hurricane Irma) and `EMSR777--EMSR778` (Tropical Storm Sara). The pre-existing CEMS Storm Boris bulletin component is `EMSR756; EMSR762; EMSR757; EMSR763; EMSR766`. Candidate/source exclusions already recorded in this audit remain: Idai `EMSR347/348/349`, Uganda `EMSR438/446`, and source candidate `EMSR758` with Storm Boris. The old external primary summary also retains its Spain--`EMSR388` and Nigeria--`EMSR314` exclusions. These links must stay in the graph even when one endpoint is absent from the official-train role table.

### Consequence for role closure

The CEMS-supported `EMSR444--EMSR445` component is an additional confirmed family. It does not create a current active-role collision because `EMSR444` is already `excluded_paired`; it must nevertheless be encoded as one component in any replacement graph. The 20 unresolved edges cannot be silently split across active roles. Most already have an `excluded_paired` endpoint under the legacy grouped roles. The two unresolved `EMSR477--EMSR481` and `EMSR477--EMSR482` edges are the remaining historical fit/development boundary: `EMSR477` is in the historical v5 development pool and `EMSR481/482` are v5 fit processes.

The smallest conservative closure for that known unresolved boundary is to retain the fit pool and mark the historical `EMSR477` development component excluded from any current or future use. It removes 25 paired-eligible IDs from the historical development inventory; it is a safety exclusion, not an assertion that the three activations are one event.

| Role/status under this strict closure | Processes | Sample IDs | SHA-256 of sorted sample IDs |
|---|---:|---:|---|
| Fit | 65 | 3,531 | `92d651073d502831a8a22e383f0476c9c489941e7bbf7ecdc0239bcbb2b53be0` |
| Historical development, no access, after `EMSR477` exclusion | 15 | 842 | `1923eaee7fc3dafec7506600ada79a65015a39ca2d6ceb1776621a281a5d4581` |
| Historical real-missing state within that no-access development inventory | 10 | 1,213 | `f4aff4760c589c57690d2799af1d9986f8b5502a727278d44f1d259f9f382c58` |
| Effective calibration | 19 | 2,146 | `9901a1a564b772018cb94731bf360459535a877343d6589773a6494e6613f9bf` |
| Retrospective internal test | 19 | 2,120 | `dd74f19fa0d39279d5d121305174897204d3f3de20234a9a18506f4aa71713ba` |
| Retrospective official holdout | 9 activations | 492 metadata-listed IDs | `0da66078bb76868a86cc49bf1230826c42cb055823b25db3e5a9cf43690b4c21` |
| Existing excluded-paired inventory | 20 | 53 | `40b3fa0d775574fd537ca509b26382440412c817ad6d494c4ff6a573be4a018f` |

The strict closure removes `EMSR477`'s 25 historical-development IDs (`cd4645f8ee9045c306784159c9f1073bf7db9b401a690bdf3bbd133ecd3cd613`) and changes neither the fit IDs nor any past output. It also requires the previous grouped-role repair and the Irma/Boris primary-evaluation exclusions from the preceding addendum.

### Audit disposition

The finite source scan is complete for its stated corpus, but the process graph is **not sufficient to freeze confirmation-facing roles**. Twenty candidate relationships remain `UNRESOLVED`; the available local CEMS source lacks narrative detail for 78 role activation records and lacks per-AOI/event-family records for all 194. The closure above protects against the known active-boundary conflicts, yet it cannot establish that remaining distinct components are independent floods. Therefore G1 remains **not cleared for confirmation-role freeze**. A later freeze requires a versioned CEMS event-family adjudication for every unresolved edge and an explicit decision to merge or exclude it; it may not use labels, predictions or scores to make those decisions.

| Source/manifest | SHA-256 |
|---|---|
| Full local CEMS activation API snapshot | `859ab6a22017f9ef53d1ca1f8d9a7e467f2d945fd4b9f4e844dd54947beda11e` |
| Normalized ImpactMesh CEMS activation manifest | `dcadc14a4a7353c1169ed057ea83a3e455285fa97b77a5738ad1d83cf988a531` |
| 37-pair candidate screen | `861a22ce27cf140f69c45ca194f7f34a6bd499331e6047406e73e87a719f4c96` |
| Grouped role manifest | `537617bdabeaa2e0fd1c01a11282d425d31c1d477315159aad0b68eb62e52e95` |
| Frozen eligibility manifest | `92925c8289c81041d99a8b67131de7e025861094386881daf44d12511b30d9a0` |

## Addendum: conservative potential-dependence closure for retrospective work (2026-09-29)

This addendum turns the finite source scan into an operational, conservative boundary. It does not treat a near-time/near-place edge as proof of one flood. `confirmed same` is merged into one process family; `screened distinct` remains separate; `likely linked but unresolved` is used only to prevent an unexamined cross-role dependency.

For this closure, the only active roles are **fit**, **calibration**, **retrospective internal test**, and **retrospective holdout**. Historical development and legacy validation are inactive and must not be accessed for training, selection, calibration or current scoring. `excluded` is inactive. Existing Irma/Boris and external-process exclusions remain in force.

### Edge-by-edge role check for the 20 unresolved candidates

| Likely-linked but unresolved edge | Current 85-role endpoints | Legacy operational endpoints | Closure treatment |
|---|---|---|---|
| `215--261` | train / train | fit / fit | One potential-dependence fit cluster. |
| `273--275` | train / excluded | historical dev / excluded | Development remains inaccessible; excluded endpoint remains inactive. |
| `323--324` and `323--329` | excluded / internal; excluded / train | excluded / internal; excluded / fit | One ambiguous component crosses internal and fit: exclude `324` and `329` from primary roles. |
| `330--333` | excluded / validation | excluded / inactive legacy validation | Keep validation inactive. |
| `469--472` and `469--475` | excluded / train; excluded / validation | excluded / fit; excluded / inactive legacy validation | Only `472` has an active endpoint; retain it as one fit unit and keep the other endpoints inactive. |
| `477--481` and `477--482` | train / train; train / train | historical dev / fit; historical dev / fit | Development is inaccessible. `481` and `482` remain separate fit units because their direct CEMS names, Eta/Iota, are screened distinct. |
| `486--487` | train / train | fit / fit | One potential-dependence fit cluster. |
| `517--518` | train / excluded | fit / excluded | Keep the excluded endpoint inactive. |
| `683--684` | calibration / excluded | calibration / excluded | Keep the excluded endpoint inactive. |
| `706--708` | train / train | fit / fit | One potential-dependence fit cluster. |
| `756--764`; `762--764`; `762--770`; `764--768`; `768--770`; `768--771` | calibration/excluded/internal/validation endpoints | calibration or inactive-legacy-validation endpoints, otherwise excluded | Existing Boris exclusions keep `762/757/763/766` inactive. `764/770` are excluded and `768/771` stay inactive legacy validation; only `756` remains active calibration. |
| `861--864` | train / train | fit / fit | One potential-dependence fit cluster. |

No unresolved edge leaves two different primary active roles after this closure except the connected `323;324;329` ambiguity component. The closure excludes its two active endpoints rather than declaring it a real merged flood family. The `screened distinct` edges continue to block a merger even where an inactive, unresolved hub is adjacent.

### Resulting process units and sample sets

Confirmed components `228;231` and `550;554` were already single v5 fit process groups. Within the remaining fit set, potential-dependence clusters are `215;261`, `486;487`, `706;708`, and `861;864`. Starting with 65 v5 fit process groups, excluding `EMSR329` leaves 64 raw fit process IDs; those four within-fit clusters yield **60 fit process units** for any process-equal reporting or future process-balanced sampling.

| Active or inactive status | Process units | Sample IDs | SHA-256 of sorted sample IDs |
|---|---:|---:|---|
| Fit, conservative closure | 60 potential-dependence units (64 raw process IDs) | 3,525 | `20f2961a727f31ae7aa1fcd2786c4cdf9b2bb8d8a2fe6847250e1dc955659ae8` |
| Calibration | 19 | 2,146 | `9901a1a564b772018cb94731bf360459535a877343d6589773a6494e6613f9bf` |
| Retrospective internal test | 18 | 2,094 | `0dce714d1981db60ba8fe3f33413f37cca503823536827472fa44277df143864` |
| Retrospective official holdout | 9 activations | 492 metadata-listed IDs | `0da66078bb76868a86cc49bf1230826c42cb055823b25db3e5a9cf43690b4c21` |
| Historical development | Inaccessible | 867 | `55707d6ece4c0caee79a4e5810906c30258b079a1bba5f1d95e3c90c144213d7` |
| Train-archive exclusions (existing plus `324` and `329`) | 24 raw process IDs | 101 | `41a38e661d4811a8e1de70addc44f7ac11c2c2ae79ef31ed7a0ad8881452f9f8` |

`EMSR324` contributes 26 paired-eligible IDs (`1581bab9094c12fa94debeaf8a91cdfa3c6dfe8bb3ea02e16ac6e0bd36e27926`) and `EMSR329` contributes six (`2ff0050be9692c758ff259f9bef83b494b787350039ed75f3af0cc8d86bd90b9`). The historical development inventory is retained unchanged as an inaccessible record; no historical outcome is used to choose this closure.

### Implementation consequences and claim boundary

Before any later run, regenerate from the 3,525-ID fit set: the process-unit membership map; its ordered sample manifest; fit-only normalization and class-weight statistics; cluster-equal sampling rule; per-unit tile counts; epoch draw count; batch/update count; unique-tile coverage; and all run/config hashes. The prior 65-process, 3,531-ID, 3,120-update schedule is not valid for the 60-unit closure. Calibration and retrospective aggregation must process-equal weight the listed units and attach the exclusion list to every output.

This closure removes all **known or unresolved cross-primary-role** dependencies in the bounded local-source graph. It supports only a retrospective, descriptive analysis subject to that source-audit boundary. It does **not** support a confirmatory claim, a claim that the CEMS graph is globally complete, an independent blind-test claim, or a generalization claim. No new independent confirmation test is available; that absence must be reported rather than replaced with a selected or reconstructed source.

## Addendum: G3 conservative candidate manifest assets (2026-09-29)

The following new assets materialize the preceding conservative closure. They are **candidate-only** (`candidate_only_not_frozen`), do not replace a historical manifest, and may not be used for training until the experiment and coordination reviews independently reproduce and freeze them. Generation read only existing IDs, role fields and frozen eligibility metadata; it did not read imagery, labels, predictions or scores.

| Candidate asset | Contents | SHA-256 |
|---|---|---|
| `g3_conservative_candidate_relationships.csv` | All 194 role activations: current/legacy role, confirmed family, potential-dependence cluster, conservative status/unit, evidence reason, and every screened-distinct edge endpoint. | `195e923e7b9ccadc0ce36e874cfcff5fd579b9e197b3d6beed92966af03e2ffa` |
| `g3_conservative_candidate_fit.csv` | 3,525 paired fit IDs, their 60 conservative units, selected S2 index/time/age/visibility fields, and deterministic unit/tile order. | `d9eec725011706e500996a2472144281c4588f26a718aec14112f5fe916d09fc` |
| `g3_conservative_candidate_calibration.csv` | 2,146 IDs in 19 calibration units. | `9186134cb5b13126ea870aef1ab78b3b20c0d3f184878f649ea9ae923a96a4be` |
| `g3_conservative_candidate_internal_retro.csv` | 2,094 IDs in 18 retrospective internal-test units. | `596c8019f9062443ed351ae947630792eac8fd44a86029116f9460594bac24fd` |
| `g3_conservative_candidate_holdout_retro.csv` | 492 metadata-listed IDs from nine retrospective holdout activations. S2 fields are intentionally blank because no holdout imagery/eligibility asset was read. | `46a520224ed096fa8712e8def637824dab42f5c7e712982b041d71645480789d` |
| `g3_conservative_candidate_historical_dev_inaccessible.csv` | 867 IDs in the 16 historical development processes, explicitly inaccessible this round. | `c6347d9a1dd28f10c211deaa50bba9b132f37c0a0a3cb9b7bf09f9d28f4592fe` |
| `g3_conservative_candidate_excluded.csv` | 101 train-archive metadata IDs plus 70 excluded-holdout IDs, with source manifest and exclusion status. | `8bcd85a0db223a390adfeeb72879d1105258dc48a848c1e940014326db5a1a1e` |
| `g3_conservative_candidate_meta.json` | Candidate status, sources, SHA-256 values, count checks, ordering seeds and closure rules. | `1464b829e26457a58b3bce8a2679bedbc21a96692caa7d645f35d5ba7bce783c` |

The fit order is deterministic but remains a **candidate** convention: units sort by SHA-256 of `g3-conservative-candidate-unit-order-v1|unit_id`; samples within a unit sort by SHA-256 of `g3-conservative-candidate-tile-order-v1|unit_id|sample_id`, with lexical ID as the tie-breaker. The source hashes and exact rules are stored in the metadata JSON. The assets reject the historical `EMSR324/329` cross-role ambiguity and retain all evaluation-family exclusions.

Consistency checks found zero sample-ID overlap among fit, calibration, retrospective internal, retrospective holdout, historical development and excluded candidate lists. The output supports a later review/recomputation only. It does not itself close the CEMS process-identity limitation, authorize model training, or convert retrospective assets into independent confirmation data.

### Metadata timestamp correction (2026-09-29)

The candidate metadata initially contained the placeholder `generated_utc=2026-09-29T00:00:00Z`. It was corrected at the local UTC clock time to `2026-09-29T03:47:19.360Z`. No role, sample ID, ordering seed, source hash or output CSV content changed. The metadata SHA-256 changed from `1464b829e26457a58b3bce8a2679bedbc21a96692caa7d645f35d5ba7bce783c` to **`62c115be4e4577ae5ef7d18e2a9c903fc3bc2bce4b4edfe4064a5317e04a09c5`**.

## Addendum: independent G3 candidate-manifest recomputation (2026-09-29)

This is an independent, read-only check of the G3 conservative candidate assets. It read IDs, manifest metadata and Zarr metadata/name arrays only. It did not open S1/S2 reflectance arrays, masks/labels, predictions, calibration outputs or scores. It does not freeze a split and does not authorize training.

### Recomputed structural checks

Each candidate file's SHA-256 agrees with `g3_conservative_candidate_meta.json` (metadata SHA-256: `62c115be4e4577ae5ef7d18e2a9c903fc3bc2bce4b4edfe4064a5317e04a09c5`): relationships `195e923e7b9ccadc0ce36e874cfcff5fd579b9e197b3d6beed92966af03e2ffa`; fit `d9eec725011706e500996a2472144281c4588f26a718aec14112f5fe916d09fc`; calibration `9186134cb5b13126ea870aef1ab78b3b20c0d3f184878f649ea9ae923a96a4be`; internal `596c8019f9062443ed351ae947630792eac8fd44a86029116f9460594bac24fd`; holdout `46a520224ed096fa8712e8def637824dab42f5c7e712982b041d71645480789d`; inaccessible development `c6347d9a1dd28f10c211deaa50bba9b132f37c0a0a3cb9b7bf09f9d28f4592fe`; excluded `8bcd85a0db223a390adfeeb72879d1105258dc48a848c1e940014326db5a1a1e`.

| Check | Independent result |
|---|---|
| Candidate row counts | Relationships 194; fit 3,525; calibration 2,146; internal retrospective 2,094; holdout retrospective 492; inaccessible development 867; excluded 171. All agree with candidate metadata. |
| Active process units | Fit has 60 units; calibration 19; internal retrospective 18; holdout 9 activation units. The relationship table has 66 fit activations because confirmed/potential families collapse to 60 units, and 20 calibration activations because the Sara component is one unit. |
| Role/status closure | Relationship statuses recompute to fit 66, calibration 20, retrospective internal 18, retrospective holdout 9, historical development inaccessible 16, inactive legacy validation 21, excluded paired 37, known evaluation-family exclusions 5, and cross-primary ambiguity exclusions 2. |
| Cross-list isolation | No duplicate sample ID occurs within any candidate list and no sample-ID intersection occurs among fit, calibration, retrospective internal, retrospective holdout, inaccessible development and excluded lists. The fit, calibration, internal and inaccessible-development rows reproduce the corresponding frozen-eligibility metadata fields. |
| Cluster-role isolation | No confirmed-family or potential-dependence cluster has active endpoints in more than one of fit, calibration, retrospective internal or retrospective holdout. |
| Fit ordering | The 3,525 rows reproduce the declared deterministic unit and within-unit ordering. |

### All-fit Sentinel-2 band-name check

For **all 3,525** fit sample IDs, the audit opened only the `band` Zarr metadata/name members in the verified train `S2L2A.tar` archive (19,448 archive member headers inspected; no reflectance array opened). Each ID has exactly one band-name array and the identical ordered vector:

`[B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12]`.

Thus the declared B4/B3/B2/B8 channel indices remain `(3, 2, 1, 7)` for every candidate fit row. There were zero missing names, duplicate name arrays or inconsistent vectors. The SHA-256 of sorted `sample_id<TAB>ordered-band-vector<LF>` records is `c5af4a1b5eb9eba769ec205f2573e131d12c1b1479a6378aff97ae24b2fd1a0d`.

### Source, label-time and spatial-support audit

The fit input archives are traceable to ImpactMesh-Flood v1: train S1RTC SHA-256 `f742b9b2f9362c0635d372129b640b525e0d300d062d72cc647f864fc20a28ee` and train S2L2A SHA-256 `0a58335297971c4a6e5e4f4e4aa8d2afe9201dedaad35e08babd03f3a1bb0c6f`. The associated acquisition records name version `v1`, source repository `ibm-esa-geospatial/ImpactMesh-Flood`, and license `CC-BY-4.0`. The retrospective holdout input metadata pins the official test S1RTC/S2L2A/MASK archive hashes; it records `test_label_pixels_read=false` and does not establish a tile-level label provenance chain.

The candidate metadata itself pins six role/eligibility sources, but does **not** pin the CEMS source snapshot, the CEMS relationship screen, archive-acquisition records, label/MASK archive hashes, a dataset revision/commit, or a license notice. Those records exist elsewhere in this workspace, but must be added to any eventual freeze binder so the candidate is reproducible as one provenance closure.

The pre-existing metadata-only label/product/time audit covers all 18 current internal retrospective activations and reports `unable_to_confirm` for each: frozen evidence has no ImpactMesh tile-to-CEMS product ID or AOI mapping, and CEMS activation/update timestamps are not satellite reference-image timestamps. Of the nine candidate holdout activations, eight occur in that audit: `EMSR251` has only a spatial/temporal *candidate* match with its tile-to-product link/version unverified, while `EMSR321`, `358`, `556`, `844`, `845`, `860` and `867` are `unable_to_confirm`; `EMSR841` is absent from the older eight-activation audit. Therefore **the label reference-time boundary is UNVERIFIED for all G3 retrospective evaluation candidates.**

No candidate asset defines a unique-pixel-owner mask, tile-footprint union or within-role pixel-overlap audit. Existing grid-header audits are limited to prior release/hand-test checks and cannot establish unique pixel support for the new candidate units. Unique pixel/tile support is consequently **UNVERIFIED** and must be regenerated from final selected masks using CRS, transform, shape, resolution and nodata checks before any process-level reporting.

### Disposition

The candidate manifests pass this structural and all-fit-band-name recomputation. They do **not** pass a data-provenance freeze gate: per-tile label product/reference-time evidence is unverified, one current holdout activation is outside the old label-time audit, unique pixel ownership is not defined, and the candidate metadata lacks a complete source/license/version and CEMS-evidence closure. They remain `candidate_only_not_frozen`; no training, calibration or retrospective scoring may start from these assets until those items are resolved and the coordinator freezes the resulting version.

## Addendum: G3 candidate MASK-grid coverage and spatial-overlap check (2026-09-29)

This supplementary independent check used only the precomputed MASK TIFF-header grid indexes and candidate sample-ID lists. It did not read MASK pixels, image pixels, labels, predictions or scores. It supersedes the preceding addendum's limited `UNVERIFIED` statement about spatial support with a concrete footprint-level finding.

### Grid-source coverage and geometry compatibility

All candidate IDs map once, and only once, to the expected header-only grid index:

| Candidate role | IDs | Expected source index | Covered | Split/holdout fields | CRS count | Common footprint size |
|---|---:|---|---:|---|---:|---|
| Fit | 3,525 | `impactmesh_train_mask_grid_index.csv` | 3,525 | `train` / `0` | 31 | 2,560 m × 2,560 m |
| Calibration | 2,146 | `impactmesh_train_mask_grid_index.csv` | 2,146 | `train` / `0` | 13 | 2,560 m × 2,560 m |
| Retrospective internal | 2,094 | `impactmesh_train_mask_grid_index.csv` | 2,094 | `train` / `0` | 12 | 2,560 m × 2,560 m |
| Retrospective holdout | 492 | `impactmesh_test_mask_grid_index.csv` | 492 | `test` / `1` | 9 | 2,560 m × 2,560 m |

There are no candidate IDs in the wrong train/test index or missing from both indexes. The train index has SHA-256 `1319ba52972add2ea4f5069830bb69eae748263b7021d7dc7218006a0f9c132b` and is traceable to MASK TAR `dc9f2003ac8a1e55511d98f00564f5d967564deb8be41d4270a55192a53c1876`; the test index has SHA-256 `076f6cb08b7a76ca9de37fc3853225ae12099bba5a2b9005158c4d9ea5ab61ca` and is traceable to MASK TAR `8e2b805020e9ef7adda33856e579fdb454a07f7c6469f30cfb8a92a5b52597aa`. Every rectangle has positive bounds. CRS was taken from the TIFF header, not inferred from the ID: respectively 230/3,525 fit, 194/2,146 calibration, 5/2,094 internal and 33/492 holdout IDs have an ID-zone/actual-grid-CRS mismatch, which confirms that ID-zone inference would be invalid.

The ordered ID-to-grid records have these independent SHA-256 bindings: fit `8ea75e84a8be2e1b8f6d53283439a14b55a470df772eed6641a02b55199c7409`; calibration `0e3fdf8fac94eb2f94ad62c78127b06f948382a0510f0afb556b42b5909345d3`; internal `89cb5fe075db36280601c62247912b29c2af1714005fc26fb550671753478897`; holdout `98e6442975a4c1a216c3806fd190f791df132170d0ac54d488082d7830d39501`. Each line is `candidate-order sample_id`, followed by the source header's CRS and bounds.

### Cross-role footprint overlaps

The audit transformed each header rectangle from its recorded CRS to EPSG:6933 and tested positive-area intersections. Among 8,257 candidate tiles it found **509 positive-area cross-primary-role footprint pairs**:

| Role pair | Tile-pair overlaps | Activation-pair overlaps | Total intersection area | Largest single intersection |
|---|---:|---|---:|---:|
| Fit / calibration | 48 | `EMSR260--487`, `EMSR429--849`, `EMSR482--778`, `EMSR486--858`, `EMSR705--795` | 97,977,015.858 m² | 5,715,098.342 m² |
| Fit / retrospective internal | 461 | `EMSR417--864`, `EMSR424--645`, `EMSR504--586` | 729,037,379.697 m² | 4,541,970.995 m² |
| Calibration / retrospective internal | 0 | — | 0 m² | 0 m² |
| Any role / retrospective holdout | 0 | — | 0 m² | 0 m² |

The deterministic SHA-256 of the 509 sorted cross-role records (role pair, activations, tile IDs and equal-area intersection area) is `0184990a207b40a19a6f1d8bea43d94dce3ede72453f583e12ea10cd443b1b42`. This is a geometry-level evidence screen, not a comparison of label values.

The grid-index fields (`grid_crs`, `left`, `bottom`, `right`, `top`) suffice for that footprint screen and for binding each candidate ID to the correct MASK archive. They do **not** contain raster shape, affine transform, pixel resolution or nodata; consequently they are insufficient by themselves to rebuild a pixel-exact `unique_pixel_mask`. A later owner-mask calculation needs those header fields from the frozen source and a deterministic precedence rule.

### Blocking disposition

The 509 cross-role intersections show that the present candidate roles do not have disjoint spatial support. A future unique-owner mask could assign each intersecting pixel to one list only after a frozen rule is defined, but it cannot retrospectively make the current lists spatially independent. In particular, the fit/calibration and fit/internal intersections are active-role leakage risks. G3 remains **blocked** for freeze, training, calibration and retrospective scoring until the coordinator resolves those overlapping tiles and regenerates all affected candidate manifests, process counts, hashes and support audits.

## Addendum: exact 10 m MASK-header overlap audit and pre-specified repair comparisons (2026-09-29)

This follow-up replaces the preceding footprint approximation with a header-only exact-grid audit. It opened the TIFF headers, but no bands or values, for all 8,257 candidate MASK members. It read CRS, width, height, affine transform and bounds only; `pixel_values_read=false`, `label_values_read=false`, and `model_outputs_read=false`.

All 8,257 headers match their selected train/test grid-index CRS and bounds. Every tile is 256 × 256 pixels with 10 m × 10 m resolution, north-up transform `(10, 0, left; 0, -10, top)`, and bounds on its recorded CRS's 10 m lattice. The direct headers therefore supply the shape/transform information that the earlier index CSV alone lacked. Every one of the 509 cross-role intersections has a shared recorded CRS and an integer number of 10 m cells. This makes either of the following fixed rules geometrically reproducible when its order and source archive hashes are frozen.

The complete 509-row, metadata-only audit appendix is [G3_CANDIDATE_SPATIAL_OVERLAP_AUDIT.csv](G3_CANDIDATE_SPATIAL_OVERLAP_AUDIT.csv), SHA-256 `26cfe8be090e8dbb3a6a738ca25c1156da16b611d86b95d637e3366122ef67f6`. It gives both sample IDs, role, conservative unit, activation, CRS, exact intersection area, 10 m cell count and fraction of each tile for every pair. The accompanying reproducibility summary is [G3_CANDIDATE_SPATIAL_OVERLAP_AUDIT.json](G3_CANDIDATE_SPATIAL_OVERLAP_AUDIT.json), SHA-256 `0bb4d8d04889663d407b7a251f3979c547cc67f7f3b90e435cf661dcfa71465d`; its canonical tabular-record hash is `b9ae636afb50895ea820ba012a897bcf46fea51eded2e638c54d06540f20e757`.

| Cross-role pair | Exact tile pairs | Exact 10 m cells in pairwise intersections | Affected activations | Unique affected sample IDs |
|---|---:|---:|---|---:|
| Fit / calibration | 48 | 979,317 | `260--487`, `429--849`, `778--482`, `795--705`, `858--486` | 54 |
| Fit / retrospective internal | 461 | 7,290,174 | `424--645`, `586--504`, `864--417` | 385 |
| Calibration / retrospective internal; any role / holdout | 0 | 0 | — | 0 |

The counts above are pairwise cell intersections. They intentionally are not interpreted as valid or water/non-water label pixels: no label values or Ignore masks were read.

### Fixed rule A: drop an evaluation tile if any of its 10 m pixel centres overlaps fit support

Because every relevant header is aligned on the same 10 m lattice in a shared CRS, positive cell overlap is equivalent to the stated pixel-centre test. This rule removes the whole evaluation tile, rather than selecting its labels or outcomes. It is the more conservative of the two operational rules below.

| Role | Candidate tiles / units | Removed tiles | Affected activations | Retained tiles / units | Retained unique 10 m grid cells |
|---|---:|---:|---|---:|---:|
| Fit | 3,525 / 60 | 0 | — | 3,525 / 60 | 227,902,756 |
| Calibration | 2,146 / 19 | 29 | `260`, `429`, `778`, `795`, `858` | 2,117 / 19 | 137,276,776 |
| Retrospective internal | 2,094 / 18 | 188 | `417`, `504`, `645` | 1,906 / 18 | 116,600,926 |
| Retrospective holdout | 492 / 9 | 0 | — | 492 / 9 | 31,960,502 |

Thus no whole process unit is eliminated under rule A, but its retained support changes and all candidate manifests, process-level support counts and hashes would have to be regenerated. The counts are geometric cells, before any Ignore/nodata handling.

### Fixed rule B: spatial owner mask with role precedence

The alternative is a deterministic support-only mask: fit owns all of its geometry; calibration retains cells not owned by fit; retrospective internal retains cells not owned by fit or calibration; retrospective holdout retains cells not owned by the preceding three roles. This fixed order (`fit → calibration → internal → holdout`) contains no outcomes, labels or scores.

| Evaluation role | Candidate tiles / units | Tiles with any owned cell | Units with any owned cell | Unique owned 10 m grid cells |
|---|---:|---:|---:|---:|
| Calibration | 2,146 / 19 | 2,146 | 19 | 138,198,003 |
| Retrospective internal | 2,094 / 18 | 2,047 | 18 | 121,637,454 |
| Retrospective holdout | 492 / 9 | 492 | 9 | 31,960,502 |

Rule B is also exactly reproducible from the verified archive headers, selected lists and stated precedence. It retains partial tiles, so it has greater geometric support than rule A. It does not, however, remove the process/imagery-dependence concern indicated by the original cross-role overlaps; it merely gives a coordinate one owner for evaluation accounting.

### Recommendation and boundary

Neither rule clears G3. If a later, separately frozen retrospective protocol permits a mechanical support repair, rule A is the defensible candidate because it excludes any evaluation tile sharing fit spatial support and has no partial-tile interpretation. Rule B is useful as a sensitivity/support accounting calculation only. Before either rule can be used, the coordinator must decide whether the affected activation pairs reveal common flood/source dependence, regenerate the role manifests from the chosen pre-specified rule, and rerun process, source, label-time and unique-support audits. No candidate asset was changed by this audit.

## Addendum: complete role-pair matrix, sequential whole-tile screen, and CEMS context (2026-09-29)

This section is a read-only extension of the exact 10 m header audit. The audit appendix JSON was extended with this calculation (new SHA-256 `0bb4d8d04889663d407b7a251f3979c547cc67f7f3b90e435cf661dcfa71465d`); the candidate manifests and the 509-row overlap-pair CSV remain unchanged. “Shared cells” below is the sum over positive-area tile pairs, so it is deliberately a pairwise overlap count rather than a count of unique valid label pixels.

### All six original cross-role pairs

| Role pair | Positive-area tile pairs | Pairwise shared 10 m cells | Unique affected tiles in first / second role |
|---|---:|---:|---:|
| Fit / calibration | 48 | 979,317 | 25 / 29 |
| Fit / retrospective internal | 461 | 7,290,174 | 197 / 188 |
| Fit / retrospective holdout | 0 | 0 | 0 / 0 |
| Calibration / retrospective internal | 0 | 0 | 0 / 0 |
| Calibration / retrospective holdout | 0 | 0 | 0 / 0 |
| Retrospective internal / retrospective holdout | 0 | 0 | 0 / 0 |

Rule A's current fit-only screen consequently has the support listed in the preceding section: fit 3,525 tiles/60 units; calibration 2,117/19 after 29 tile removals; internal 1,906/18 after 188 removals; and holdout 492/9 with no removals. This screen has not been materialized as a candidate-manifest edit.

### Deterministic serial whole-tile screen

For a stricter geometric check, the audit applied the fixed role order `fit → calibration → retrospective internal → retrospective holdout`. A tile in each later role is rejected if it shares one or more 10 m cells with **any retained** earlier-role tile. The rule has no data-, label- or score-dependent branch.

| Role | Retained tiles | Retained units | Retained activations | Minimum retained tiles in an activation |
|---|---:|---:|---:|---|
| Fit | 3,525 | 60 | 66 | 5 (`EMSR231`) |
| Calibration | 2,117 | 19 | 20 | 7 (`EMSR858`) |
| Retrospective internal | 1,906 | 18 | 18 | 5 (`EMSR385`) |
| Retrospective holdout | 492 | 9 | 9 | 2 (`EMSR841`) |

It removes the same 29 calibration and 188 internal tiles as the fit-only screen because the other four original role-pair intersections are empty. Recomputing the complete six-pair matrix after the serial screen gives **zero positive-area tile pairs and zero pairwise shared 10 m cells in every pair**. It is therefore a deterministic way to obtain geographically non-overlapping retained tiles from these four lists, subject to all other unresolved validity constraints.

### CEMS event context for the eight spatially overlapping activation pairs

No one of the eight pairs has a direct, documented same-process CEMS edge in the fixed relationship audit. CEMS metadata supports different dates and names for each pair, but that is not proof of independent flood processes, source imagery, labels or products. The corresponding tile-level source and label-product links remain **UNVERIFIED**.

| Overlapping activations | Metadata-only context | CEMS relation disposition |
|---|---|---|
| `260--487` | Northern Italy / Emilia-Romagna; 2017-12-12 and 2020-12-06 | Repeated broad region, different dates; no direct edge. |
| `429--849` | Ireland; 2020-02-24 and Storm Claudia, 2025-11-14 | Different dates/names; no direct edge. |
| `778--482` | Tropical Storm Sara, 2024-11-16 and Tropical Cyclone Iota, 2020-11-17 | Different named storms/dates; no direct edge. `778`'s known Sara family link is to `777`, not `482`. |
| `795--705` | Italy/Tuscany; 2025-03-14 and 2023-11-03 | Repeated broad region, different dates; no direct edge. |
| `858--486` | Southern Italy, 2026-01-19 and eastern Sardinia, 2020-11-28 | Different dates; no direct edge. `858` has a separate screened-distinct edge to `855`, not `486`. |
| `424--645` | Madagascar; 2020-01-30 tropical disturbance and Tropical Storm Cheneso, 2023-01-25 | Different dates/names; no direct edge. |
| `586--504` | New South Wales; 2022-07-03 and 2021-03-20 | Repeated region, different dates; no direct edge. |
| `864--417` | Portugal/Spain, 2026-02-03 and Portugal, 2019-12-23 | Repeated region, different dates; no direct edge. `864` has a separate unresolved edge to `861`, not `417`. |

The CEMS observations classify the source relationship of all eight overlap pairs as **UNVERIFIED**, rather than declaring them independent. A serial spatial screen is a prospective mechanical support rule; it cannot substitute for process-family adjudication or per-tile source/label provenance.

## Addendum: Rule-A derived retrospective candidate manifests (2026-09-29)

Following independent review of the header-only spatial audit, the project now contains a **new derived candidate set**. It does not modify any `g3_conservative_candidate_*` source file. Its status is exactly `CANDIDATE_RETROSPECTIVE_ONLY_NOT_FROZEN`; it is not a frozen experimental split and does not authorize training, calibration, scoring or confirmation claims.

The reproducible generator is [derive_g3_rule_a_candidate.py](../scripts/derive_g3_rule_a_candidate.py), SHA-256 `7541bde55b8aee65c77a831b0e222412f0bea978a1ea89485a78c515e6f66ef7`. It reads only four source candidate CSVs and the exhaustive 509-row header-only overlap appendix. It does not call a dataset reader or open imagery, label pixels, predictions or scores. Fit is retained; calibration, retrospective internal and retrospective holdout are processed in that fixed order, with a whole tile excluded if it shares at least one already audited 10 m cell with a retained earlier-role tile. In this data, the rule removes only 29 calibration and 188 internal tiles because all other original role-pair intersections are zero.

| Derived asset | Rows / units | SHA-256 |
|---|---:|---|
| `g3_rule_a_derived_candidate_fit.csv` | 3,525 / 60 | `c6c71467f9d535ff07493e7356f107aa3d99d07721ba48facc2358d53758ad9c` |
| `g3_rule_a_derived_candidate_calibration.csv` | 2,117 / 19 | `6d8f0fa9270bda3098aaca87ffb63defc72223d07b29591ebba2adfda3656b28` |
| `g3_rule_a_derived_candidate_internal_retro.csv` | 1,906 / 18 | `91f5d87c8fad9c491cfe1612b76641f5a71f4ffca893e22610ff3ea5325dfc0e` |
| `g3_rule_a_derived_candidate_holdout_retro.csv` | 492 / 9 | `bed3cba0a85a2da4d24a26e1b38065fe47a211f61f93e88752361628849e4df3` |
| `g3_rule_a_derived_candidate_excluded_tiles.csv` | 217 excluded tiles | `66265b63812bc7d733034ab9ccb381c4deb3915f84ad08787ee347df85920e23` |
| `g3_rule_a_derived_candidate_meta.json` | Sources, rules, counts and rechecks | `d2eeb5f5ada0b00a4edb7844ec4381d944a228d7952a1c80176ac7d0a04129fe` |

Every retained row records its `derived_role`, role rank, consecutive derived sample order, first-occurrence unit order, within-unit order, source filename and original source CSV line number. Every excluded row records the source role/line, conservative unit, prior retained sample IDs, pair count and shared-cell count that triggered its exclusion. Independent recomputation confirmed the source-line mapping, order continuity, 0/29/188/0 exclusions, expected 60/19/18/9 units, zero cross-role sample-ID intersections and zero remaining pairs in the exhaustive spatial-overlap audit.

The derived metadata preserves the existing CEMS process-family/source boundary and per-tile label product/reference-time boundary as **UNVERIFIED**. Any later runner must use the derived file hashes only after coordination review; it must not infer that geographic de-overlap establishes process independence.

## Addendum: bounded Rule-A source, product and label-time re-audit (2026-09-29)

This bounded follow-up covers every Rule-A derived candidate row and unit using manifests, archive-download records, the local CEMS product index and published-asset metadata only. The reproducible script is [audit_g3_rule_a_source_time.py](../scripts/audit_g3_rule_a_source_time.py), SHA-256 `414020f41d8fa142aef02dc7f9a4f8ad4749be33b48a6a9d5651e82db3bb371f`. It does not open imagery or label raster values, call a dataset reader, or read predictions, scores or calibration outputs.

The complete per-tile result is [G3_RULE_A_SOURCE_TIME_CHAIN_TILE_AUDIT.csv](G3_RULE_A_SOURCE_TIME_CHAIN_TILE_AUDIT.csv), SHA-256 `26ea9a9b444b8981ce9dccba793659da82ca0fea493d846c442065df7e1aacee`; it contains the candidate role/unit/ID, S1 anchor and its provenance, selected S2 metadata when available, verified S1/S2/MASK archive hashes, dataset version/licence, all local activation-level CEMS product candidates and their local paths/hashes, and the mapping/time status. Its 106-unit aggregation is [G3_RULE_A_SOURCE_TIME_CHAIN_UNIT_AUDIT.csv](G3_RULE_A_SOURCE_TIME_CHAIN_UNIT_AUDIT.csv), SHA-256 `7a59348f343288931c65acba94a52bb3668adeb380d75317f20c65a391345724`. The machine-readable audit summary is [G3_RULE_A_SOURCE_TIME_CHAIN_AUDIT.json](G3_RULE_A_SOURCE_TIME_CHAIN_AUDIT.json), SHA-256 `12a51605b9fe0bdc81ed309515e5f4bb8269f218ee23bef6a5e49a9e881da701`.

| Audit item | Result |
|---|---|
| Candidate coverage | 8,040 tiles in 106 units, all represented once in the tile audit. |
| Dataset input provenance | Train roles use verified ImpactMesh-Flood v1 `train/S1RTC.tar` / `train/S2L2A.tar` and native `train/MASK.tar`; retrospective holdout uses their `test/` counterparts. Archive paths, SHA-256 values and the `CC-BY-4.0` dataset licence are recorded per row. |
| S1 anchor | All 7,548 train-role tiles have direct `impactmesh_train_event_times.csv.gz` anchors. Of 492 holdout metadata IDs, 481 anchors can only be reconstructed from frozen selected-S2 UTC plus age; **11 lack both a direct test-S1 time manifest and selected-S2/age values**, so their anchor is explicitly `UNAVAILABLE`. |
| Holdout optical eligibility | The 492-ID retrospective holdout list contains 432 paired-eligible and 60 non-eligible metadata IDs. It is therefore not an inference-ready paired evaluation set. |
| Local CEMS product candidates | Only EMSR424 (287 fit tiles), EMSR640 (45 fit tiles) and EMSR251 (121 holdout tiles) have any local CEMS product candidate: 453 tiles total. The product IDs, reference-image times, file paths and file hashes are recorded, but only at activation level. |
| Confirmed label source/time chain | **0 / 8,040 tiles.** No candidate has a CEMS product ID or reference/acquisition time attributable to its native ImpactMesh MASK label. Every tile remains `UNVERIFIED`. |

The published [ImpactMesh README](https://huggingface.co/datasets/ibm-esa-geospatial/ImpactMesh-Flood/blob/main/README.md) describes `MASK.tar` as CEMS flood annotations and lists split-level TAR files, but contains no sample-ID-to-CEMS-product/AOI/reference-time table. The locally preserved full published-asset inventory (`data/manifests/impactmesh_assets.json`, SHA-256 `e9a3c894d178d77bd5e63a2a1ccaa981fe0a53c637543b3c0408360b59c698db`) likewise contains README, split lists and modality TARs, but no mapping asset; its README Git object ID is `961e6cf8d064c6734a1e34ef0f3e3c4b3e767e34`. CEMS' [official API documentation](https://mapping.emergency.copernicus.eu/about/how-to-harvest-cems-mapping-data/) explains how activation/AOI product metadata and reference imagery can be enumerated. That capability cannot identify which one ImpactMesh used for a particular tile without a publisher mapping.

Thus the local and published evidence supports dataset-level source/version/licence tracing, and partial activation-level CEMS product enumeration. It does not close per-tile label-product provenance or the label-reference-time-to-S1-anchor relationship. The **real independently cleared holdout count remains 0**: the nine official-holdout activations are retrospective candidates only, and the existing alternative-source audits have cleared no new independent multi-process G3 source. Different activation names or different dates were not treated as independence evidence.

## Addendum: final bounded holdout anchor recovery and retrospective-use boundary (2026-09-29)

This is the final narrow metadata check in the current data-audit pass. No further CEMS, source, product or time-chain investigation is planned from this point. It reads only TAR member metadata, selected S1 Zarr `.zattrs` text and the existing MASK TIFF-header grid index; no image/S2/MASK array value, label pixel, prediction, score or dataset reader was accessed.

The 11 Rule-A holdout rows previously lacking a frozen S2/age-derived anchor all belong to `EMSR556`. Their event S1 timestamp is directly recoverable as `2021-12-21T09:48:59.163333Z` from the selected S1 Zarr `.zattrs.datetime` field at event index 2. For each row, [G3_RULE_A_HOLDOUT_S1_ANCHOR_METADATA_AUDIT.csv](G3_RULE_A_HOLDOUT_S1_ANCHOR_METADATA_AUDIT.csv) records the S1 TAR member path/offset, `.zattrs` hash and CRC, exact timestamp, matched native MASK TAR member path/offset, and MASK-header grid CRS/bounds. Its SHA-256 is `f0500699958b7285d55b89ba5e2726c6cede5c2872fb729b8e54f622612236a1`; summary JSON SHA-256 is `c1be977eebb164f041206803c588d9152414b197d75a419e9b85928b941097b8`; the reproducibility script SHA-256 is `5d512bdcced928072601d2684a49c8b1a6c645b739b35aaa5909fcd6212a8eac`.

The S1/MASK sample-ID and grid association establishes only that these entries are co-indexed native dataset assets. The S1 attributes contain no CEMS product ID, AOI, reference-image time or label production time. Their tile-to-product and label-reference-time relationship therefore remains **UNVERIFIED**.

### Fixed retrospective reporting groups

| Group | Tiles | Use boundary |
|---|---:|---|
| Paired eligible | 432 | The only support for the predeclared five optical-availability states. Report as retrospective/descriptive because the nine holdout activations are not a cleared blind test and the native label product/time chain is unverified. |
| Low visible (`visible_below_0.5`) | 49 | Separate descriptive real-optical-availability group; not part of the controlled five-state paired comparison. |
| No qualifying pre-S1 S2 within 30 days | 11 | Direct S1 anchor is now known, but there is no selected qualifying pre-S1 S2. Exclude from S2 time-difference/sensitivity analyses; if shown, label only as a descriptive no-qualifying-optical-metadata group. |

The Rule-A derived roles retain 3,525 / 2,117 / 1,906 / 492 tiles and 60 / 19 / 18 / 9 units. They have zero known cross-role sample-ID intersections and zero remaining pairs in the reviewed 509-row spatial-overlap audit. These properties permit the stated **retrospective descriptive** use of the audited metadata boundaries. They do not establish a blind holdout, independent confirmation set, CEMS-product-linked label time, or a confirmation-facing generalization claim. The unresolved product/reference-time item is retained as a limitation of that description; it is not a reason to reopen the completed data audit before the bounded retrospective work.
