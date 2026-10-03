# Third-party notices

DuetRNA contains adapted or vendored source from open-source research
projects. No third-party checkpoint, dataset, generated result, or compiled
evaluation binary is redistributed here.

## MIT-licensed components

The root `LICENSE` contains the MIT terms used by the original DuetRNA code and
the following adapted components:

| Component | Upstream | Copyright / notice | Use in this repository |
| --- | --- | --- | --- |
| RNA-FrameFlow | <https://github.com/rish-16/rna-backbone-design> | 2024 Rishabh Anand and Chaitanya K. Joshi | Starting point for RNA preprocessing, flow matching, atom reconstruction, and benchmark organization; substantially extended for coupled base/sugar frames and joint sequence-structure generation. |
| protein-frame-flow | <https://github.com/microsoft/protein-frame-flow> | Microsoft Corporation | Node, edge, time, and positional embedding utilities. |
| SE(3) diffusion | <https://github.com/jasonkyuyim/se3_diffusion> | 2022 Jason Yim, Brian L. Trippe, Valentin De Bortoli, and Emile Mathieu | Rigid-body, diffusion, and atom-reconstruction utilities. |
| MMDiff | <https://github.com/Profluent-Internships/MMDiff> | 2024 Profluent Bio | Nucleotide constants and nucleic-acid atom reconstruction. |
| Geomstats | <https://github.com/geomstats/geomstats> | 2018 Nina Miolane | Numerical methods used through the upstream SO(3) implementation. |
| Graphein | <https://github.com/a-r-j/graphein> | Graphein contributors | Selected angle-computation helpers identified in source comments. |

`external_tools/grnade_api/` contains the benchmark-compatible gRNAde v0.3
inference subset from <https://github.com/chaitjo/geometric-rna-design>,
copyright 2025 Chaitanya K. Joshi, under the MIT License. Its exact license is
reproduced in `LICENSES/gRNAde-MIT.txt`. No gRNAde weights are included.

The RhoFold source subset includes RNA-FM/ESM-derived files retaining
copyright notices for Meta Platforms, Inc. and affiliates under the MIT
License. Those terms are reproduced in `LICENSES/ESM-MIT.txt`.

## Apache-2.0 components

The following source remains under Apache License 2.0:

- `external_tools/rhofold_api/`, adapted from
  <https://github.com/ml4bio/RhoFold>;
- invariant point attention and related layers adapted from
  <https://github.com/aqlaboratory/openfold>; and
- AlphaFold-derived constants and geometry utilities from
  <https://github.com/google-deepmind/alphafold>.

Relevant files retain their upstream headers. The complete Apache License 2.0
text is reproduced in `LICENSES/Apache-2.0.txt`. No RhoFold weights are
included.

## Separately obtained artifacts and tools

The following are not part of this distribution and remain governed by their
own providers' terms:

- DuetRNA, gRNAde, and RhoFold checkpoints;
- RNASolo, RNA3DB, and any other datasets;
- US-align, qTMclust, and TMscore executables;
- Phenix/MolProbity; and
- generated structures, evaluation outputs, and experiment logs.

Users are responsible for verifying the provenance and licenses of every
separately obtained artifact.
