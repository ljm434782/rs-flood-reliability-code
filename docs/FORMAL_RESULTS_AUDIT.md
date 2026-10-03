# Formal results audit (2026-09-27)

All five models completed the frozen training/calibration/evaluation sequence. The
numeric tables are extracted from `outputs/formal_v1/*/*.json` by
`scripts/summarize_impactmesh_results.py`; the table provenance file records
SHA256 of every result file. The extractor checks that compared models use the
same process and event IDs. The 36 project tests pass.

| Model | Internal flood IoU, 19 processes | Official holdout flood IoU, 8 processes | Sen1Floods11 water IoU, 8 independent events |
|---|---:|---:|---:|
| B1 | 0.206 | 0.176 | 0.441 |
| B2 | 0.164 | 0.218 | 0.466 |
| B4 | 0.297 | 0.262 | 0.496 |
| B5 | 0.287 | 0.271 | 0.448 |
| P | 0.252 | 0.230 | 0.499 |

These are process/event-equal means, not pooled-pixel means. External water IoU
is binary and cannot establish the three-class distinction. Spain and Nigeria
events are excluded from that independent-event summary due to documented
process-overlap risk, with all-event sensitivity retained in original JSON.

The internal paired 100% to 0% S2 block-retention flood IoU changes are B4
0.297 to 0.270, B5 0.287 to 0.252, and P 0.252 to 0.251. B1 and B2 are
invariant to S2 masking by construction. P does not outperform B4 in the
internal complete-observation state: paired P minus B4 mean difference -0.045,
descriptive 95% process bootstrap interval [-0.064, -0.026]. At 0% S2 the
difference is -0.019 [-0.071, 0.028]. These intervals have no multiplicity
adjustment and do not imply causal attribution to the gate alone.

On 651 genuinely unqualified S2 tiles from 15 internal-test processes, flood
IoU is B1 0.253, B2 0.147, B4 0.319, B5 0.252, P 0.239. The categories are
380 tiles with selected S2 visibility below 50% and 271 with no pre-S1 S2 in
30 days; they are not treated as controlled missingness. Some processes occur
in both categories, so category process counts must not be summed.

Global temperature scaling fitted only on 19 calibration processes reduces
internal 100% S2 NLL for B2 0.310 to 0.292, B4 0.270 to 0.250, B5 0.342
to 0.317, and P 0.293 to 0.272. The same temperature is used at every S2
retention level. For B4 at 0% S2, NLL **increases** from 0.327 to 0.331,
showing that full-observation calibration is not uniformly beneficial under
sensor loss. All ECE results use the fixed 15-bin process-level convention.

The selected pre-S1 S2 age median is 18.80 days for the internal paired test
and 19.26 days for the eligible official holdout, computed from frozen input
metadata without labels. It is an acquisition-age description, not evidence of
label synchrony. The dataset card does not map every tile to an exact CEMS
product/version; correspondence of all three-class labels to the event S1
instant remains `UNVERIFIED`.

Official sources: [ImpactMesh-Flood v1 dataset card](https://huggingface.co/datasets/ibm-esa-geospatial/ImpactMesh-Flood),
[Sen1Floods11 author repository](https://github.com/cloudtostreet/Sen1Floods11),
[Hurricane Irma CEMS bulletin](https://mapping.emergency.copernicus.eu/news/copernicus-ems-is-mapping-the-impact-of-hurricane-irma-over-the-carribbean-islands/),
[Storm Boris CEMS bulletin](https://mapping.emergency.copernicus.eu/news/information-bulletin-173-the-copernicus-emergency-management-service-monitors-floods-in-central-and-eastern-europe/),
[Nigeria CEMS activation](https://mapping.emergency.copernicus.eu/activations/EMSR314/).
