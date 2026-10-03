# Third-party notices

See [LICENSE](LICENSE) for DuetRNA's MIT license. Adapted files retain the
licenses documented below and in their headers.

This list records the sources of included code. Python installation
dependencies are listed in [pyproject.toml](pyproject.toml).

## MIT source

| Upstream source | Included code and source history | Copyright notice |
| --- | --- | --- |
| [RNA-FrameFlow](https://github.com/rish-16/rna-backbone-design) | RNA parsing and preprocessing in [chain_parser.py](duetrna/data/chain_parser.py), [structure_parser.py](duetrna/data/structure_parser.py), and [process_rna_pdb_files.py](duetrna/data/process_rna_pdb_files.py); atom reconstruction and [benchmark evaluation](benchmark/evalsuite.py). These files retain the upstream credits described below. | 2024 Rishabh Anand and Chaitanya K. Joshi |
| [protein-frame-flow](https://github.com/microsoft/protein-frame-flow) | [Node embeddings](duetrna_modeling/node_embedder.py), [edge embeddings](duetrna_modeling/edge_embedder.py), and [time/position embedding helpers](duetrna_modeling/utils.py), inherited through RNA-FrameFlow. | Microsoft Corporation |
| [SE(3)-diffusion](https://github.com/jasonkyuyim/se3_diffusion) | [Rigid transforms](duetrna_shared_core/geometry/rigid_utils.py), [SO(3) utilities](duetrna_training/diffusion/so3_utils.py), and parts of [atom reconstruction](duetrna_shared_core/reconstruction/all_atom.py), inherited through RNA-FrameFlow. | 2022 Jason Yim, Brian L. Trippe, Valentin De Bortoli, and Emile Mathieu |
| [MMDiff](https://github.com/Profluent-Internships/MMDiff) | [Nucleotide constants](duetrna_shared_core/chemistry/nucleotide_constants.py), parsing helpers, and `of_na_torsion_angles_to_frames` / `na_frames_to_atom23_pos` in [all_atom.py](duetrna_shared_core/reconstruction/all_atom.py), inherited through RNA-FrameFlow. | 2024 Profluent Bio |
| [Geomstats](https://github.com/geomstats/geomstats) | `rotmat_to_rotvec` in [so3_utils.py](duetrna_training/diffusion/so3_utils.py) records its adaptation through SE(3)-diffusion and RNA-FrameFlow. | 2018 Nina Miolane |
| [Graphein](https://github.com/a-r-j/graphein) | Dihedral and bond-angle helpers in [evalsuite.py](benchmark/evalsuite.py), inherited through RNA-FrameFlow; `pdb_to_tensor`, `df_to_tensor`, `remove_insertions`, `filter_dataframe`, and `get_full_atom_coords` in [gRNAde data_utils.py](external_tools/grnade_api/src/data/data_utils.py), inherited through gRNAde. | 2019–2023 Arian Jamasb; [license text](LICENSES/Graphein-MIT.txt) |
| [gRNAde](https://github.com/chaitjo/geometric-rna-design/tree/v0.3.2) | The inverse-folding adapter and inference source in [external_tools/grnade_api](external_tools/grnade_api/). | 2025 Chaitanya K. Joshi; [license text](LICENSES/gRNAde-MIT.txt) |
| [RNA-FM](https://github.com/ml4bio/RNA-FM) / [ESM](https://github.com/facebookresearch/esm) | Language-model source bundled by RhoFold in [rna_fm](external_tools/rhofold_api/rhofold/model/rna_fm/). | Meta Platforms, Inc. and affiliates; [license text](LICENSES/ESM-MIT.txt) |

The MMDiff, Geomstats, and Graphein entries identify inherited helper code.
Their packages and pretrained models are not required by the installation.
The MIT notices above are retained in [LICENSE](LICENSE) and the linked
component license files.

## Apache-2.0 source

| Upstream source | Included code |
| --- | --- |
| [RhoFold](https://github.com/ml4bio/RhoFold) | Folding-model source in [external_tools/rhofold_api](external_tools/rhofold_api/); its RNA-FM/ESM-derived files retain the MIT notices listed above. |
| [OpenFold](https://github.com/aqlaboratory/openfold) | Invariant point attention in [ipa_pytorch.py](duetrna_modeling/ipa_pytorch.py) and torsion-network layers in [torsion_net.py](duetrna_modeling/torsion_net.py), as identified by their headers. |
| [AlphaFold](https://github.com/google-deepmind/alphafold) | Constants, geometry, and attention code identified by retained DeepMind copyright headers. |

These files retain their upstream headers. The Apache License 2.0 text is
included in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

## Separately obtained files and tools

Model checkpoints, datasets, US-align, qTMclust, TMscore, and Phenix/MolProbity
are not included in this source distribution. Obtain them under their
providers' terms; see the [evaluation setup](README.md#required-external-artifacts)
for the required files.
