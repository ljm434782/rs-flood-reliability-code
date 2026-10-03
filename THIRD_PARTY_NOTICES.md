# Separate code and data licenses

## Author project code

MIT License in the repository root applies to this project's author code.
Copyright (c) 2026 Zhiyong Guo et al.
Third-party code remains governed by its respective licenses. Dataset licenses
are not covered or superseded by this project's MIT License.

## SMAGNet third-party source

ASU CICI Lab's SMAGNet source is distributed under its original MIT License
(copyright 2025 ASU CICI Lab). `third_party/smagnet/LICENSE` and `SOURCE.json`
are byte-identical to the source archive. `SOURCE.json` records upstream
repository https://github.com/ASUcicilab/SMAGNet.git and commit
4371df08e6ca3b9d71c0385ad57b589830469a0c, plus exact source/license SHA-256.
This license covers that third-party source, not the entire author project.
The three-class adaptation boundary remains documented separately.

## ImpactMesh-Flood data

The preserved v1 publisher/provenance records identify CC-BY-4.0. Raw imagery
and native labels are not redistributed here. Publisher license and attribution
requirements remain separate from any code license.

## Sen1Floods11 data

Original source and access records remain in
`data/manifests/sen1floods11_manifest.csv`. Data use remains subject to the
publisher's own license/terms. This package does not infer a new data license
from SMAGNet or the authors' code-license decision.

## Other dependencies

Python/PyTorch, scientific/geospatial libraries, segmentation-models-pytorch,
timm and other external dependencies retain their own licenses. They are
dependency requirements, not vendored author code. Historical environment
lock records describe their original installations; the author project's
MIT License cannot override those external terms.
