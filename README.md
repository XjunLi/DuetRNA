# DuetRNA

**Base-and-Sugar Dual-Frame Flow Matching for RNA Co-Design**

[Paper](https://www.biorxiv.org/content/10.64898/2026.08.18.745543v1) ·
[Code](https://github.com/XjunLi/DuetRNA) ·
[Project Page](https://xjunli.github.io/DuetRNA/) ·
[Blog (中文)](https://xjunli.github.io/DuetRNA/blog/zh/) ·
[Blog (English)](https://xjunli.github.io/DuetRNA/blog/en/)

DuetRNA jointly generates RNA sequences and three-dimensional structures from
a target length. It uses coupled base and sugar frames with flow matching to
model RNA interactions and backbone geometry.

## Installation

Use Python 3.11, PyTorch 2.4, and a CUDA GPU. Install the
[PyTorch](https://pytorch.org/get-started/locally/) and
[PyG extension wheels](https://data.pyg.org/whl/) matching your CUDA environment.

```bash
git clone https://github.com/XjunLi/DuetRNA.git
cd DuetRNA
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Run the commands below from the repository root.

## Data preparation

Convert RNA PDB files into training features and a metadata CSV:

```bash
python -m duetrna.data.process_rna_pdb_files \
  --pdb_dir /path/to/rnasolo/pdb \
  --write_dir /path/to/rnasolo_processed \
  --num_processes 16
```

The output includes feature pickles and `rna_metadata.csv`.
For RNA3DB mmCIF input, see
`python -m duetrna.data.process_rna3db_mmcif_files --help`.

## Training

Training settings are in [config.yaml](duetrna/configs/config.yaml).
The default budget is 180,000 optimizer steps. For two GPUs:

```bash
python -m duetrna.train \
  data_cfg.csv_path=/path/to/rnasolo_processed/rna_metadata.csv \
  experiment.num_devices=2 \
  experiment.trainer.accumulate_grad_batches=2 \
  experiment.wandb.name=rnasolo_run1
```

Checkpoints are saved under `checkpoints/`. To resume a run, add
`experiment.warm_start=/path/to/model.ckpt`.

## Generation

Generate four 80-nt designs:

```bash
python -m duetrna.inference \
  inference.ckpt_path=/path/to/model.ckpt \
  inference.training_config_path=/path/to/config.yaml \
  inference.output_dir=generated \
  inference.name=length80 \
  inference.samples.length_subset='[80]' \
  +inference.samples.samples_per_length=4 \
  inference.evalsuite.run_eval=false
```

Each sample produces a PDB structure, FASTA sequence, and trajectory PDB under
`generated/length80/`. Sampling settings are in
[inference.yaml](duetrna/configs/inference.yaml).

## Evaluation

IF evaluates generated structures using gRNAde-designed sequences and RhoFold.
GS evaluates DuetRNA's generated sequences with RhoFold.

Place the evaluation tools and weights at the following paths:

| Resource | Location |
| --- | --- |
| [RhoFold](https://proj.cse.cuhk.edu.hk/aihlab/RhoFold/api/download?filename=RhoFold_pretrained.pt) | `benchmark/checkpoints/RhoFold_pretrained.pt` |
| [gRNAde](https://github.com/chaitjo/geometric-rna-design/tree/v0.3.2/checkpoints) | `external_tools/grnade_api/checkpoints/gRNAde_ARv1_1state_all.h5` |
| [US-align, qTMclust, TMscore](https://github.com/rish-16/rna-backbone-design/tree/main/rna_backbone_design/tools/cpp) | `external_tools/cpp/` |

Make the comparison tools executable and set reference-data paths in
`inference.evalsuite.paths`.

Run generation and IF evaluation:

```bash
python -m duetrna.inference \
  inference.ckpt_path=/path/to/model.ckpt \
  inference.training_config_path=/path/to/config.yaml \
  inference.output_dir=generated \
  inference.name=eval_if \
  inference.benchmark_preset=benchmark/protocols/duetrna_structure_first.yaml \
  inference.evalsuite.run_eval=true
```

For GS, use `benchmark/protocols/duetrna_joint.yaml` as the preset.
To evaluate existing samples, keep their output directory and run name and add
`inference.run_inference=false`.

The presets generate 50 samples per length from 40 to 150 nt in steps of 10.
Metrics are saved under the run's `evaluation/` directory.

Additional structure-quality, diversity, and novelty analyses are available in
[evaluation_analysis/](evaluation_analysis/), with settings in
[config.example.toml](evaluation_analysis/config.example.toml).

## Code layout

| Directory | Contents |
| --- | --- |
| `duetrna/` | Model, training, generation, and data preprocessing |
| `duetrna/configs/` | Training and inference settings |
| `benchmark/` | IF/GS evaluation and structural metrics |
| `evaluation_analysis/` | Structure-quality, diversity, and novelty analysis |
| `external_tools/` | gRNAde and RhoFold adapters |
| `Docs/` | Paper and method figures |

## Citation

```bibtex
@article{li2026duetrna,
  title   = {Base-and-Sugar Dual-Frame Flow Matching for RNA Co-Design},
  author  = {Li, Junzhe and Peng, Lijian and Li, Yuhao and Zhou, Yize and
             Cao, Hanqun and Tan, Cheng and Liu, Shengchao},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.08.18.745543},
  url     = {https://www.biorxiv.org/content/10.64898/2026.08.18.745543v1}
}
```

## License

[MIT License](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party notices.
