"""
RNA evaluation adapted from RNA-FrameFlow.

Protocol corrections relative to upstream evalsuite.py at 5f3efa1:
https://github.com/rish-16/rna-backbone-design/blob/5f3efa169791d76adf121d35028bf3a34635639c/rna_backbone_design/analysis/evalsuite.py#L509
The upstream selector ANDs C4', O3', and C3' names, selecting no atoms;
its comments describe C4' metrics, but TMscore does not specify an atom.
An OR selection retains PDB atom order and can cross-pair C3' with C4'.
Here scRMSD/scGDT match C4' by residue/name, independent of atom order.
TMscore, USalign, and qTMclust explicitly use RNA/C4' (RNA default: C3').

Some helper methods for angle computation have been taken from `graphein`:
https://github.com/a-r-j/graphein/blob/master/graphein/protein/tensor/angles.py#L448
Credits: Arian R. Jamasb
"""

from __future__ import annotations

import os, gc, json, math, shutil, subprocess, torch
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import torch.nn.functional as F

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# external tools (gRNAde, RhoFold)
from external_tools.rhofold_api.rhofold.rf import RhoFold
from external_tools.rhofold_api.rhofold.config import rhofold_config
from external_tools.rhofold_api.rhofold.utils.alphabet import get_features
from benchmark import metrics
from evaluation_analysis.common import canonical_sample_id
from duetrna.data.io import filter_metadata_by_split

ALIGNMENT_ATOM = "C4'"
BACKBONE_ATOMS = (ALIGNMENT_ATOM,)


class EvalSuite:
    def __init__(self,
            save_dir,
            paths=None,
            constants=None,
            gpu_id1=0,
            gpu_id2=1,
            use_invfold=True,
            prefer_generated_fasta=False,
            load_models=True,
        ):

        """
        Params:
            save_dir (str) : target directory to save temp files and metrics in torch's .pt format

            paths (cfg) : sub-config containining the following paths:
                - training_data_dir (str) : directory containing the training reference samples in PDB format (to compute novelty)
                - usalign_path (str) : filepath to ./USalign to compute TM-scores between generated samples and ground truth samples (novelty)
                - qtmclust_path (str) : filepath to ./qTMclust to compute TM-scores among generated samples (diversity)
                - tmscore_path (str) : filepath to ./TMscore to compute TM-scores between generated samples and predicted structures from structure predictor (smTM)
                - training_metadata_path (str) : filepath to the training metadata CSV for bookkeeping

            constants (cfg) : sub-config containing the following constants:
                - metadata_len_filter (list[int, int]) : min-max sequence lengths to compare generated samples to
                - tm_thresh (float) : threshold in [0, 1] to consider two structures similar using TM-score
                - rmsd_thresh (float) : maximum RMSD between generated sample and predicted structure to be considered designable

            gpu_id1 (int) : GPU index for inverse folding model (gRNAde)
            gpu_id2 (int) : GPU index for structure prediction model (RhoFold)

            use_invfold (bool) : flag to decide whether to inverse fold the backbones or rely on existing sequence inside the sample's PDB file
            prefer_generated_fasta (bool) : when True, use a generated FASTA found beside a PDB before considering inverse folding
            load_models (bool) : flag to decide whether to load models to GPU or not (in case you want to use some of the helper functions without overhead from loading)

        Files/folders saved locally:
            - fwdfold_pdbs/ : directory containing all forward-folded samples in PDB format
            - invfold_seqs/ : directory containing all inverse-folded samples in FASTA format
            - final_metrics.pt : pytorch file of all global+local metrics computed throughout EvalSuite run
            - final_metrics_local.pt : pytorch file containing all local metrics (stored in case global metrics fail)
            - final_metrics_global.pt : pytorch file containing all global metrics (stored in case local metrics fail)
            - gen_list.list : text file containing names of all generated samples
            - gt_list.list : all metadata reference structures in the selected split and length range
            - MASTER_QTM_RESULTS.txt : text file (tsv) containing qTMclust module outputs
            - MASTER_USALIGN_RESULTS.txt : text file (tsv) containing US-align module outputs
            - TMSCORE_TMP_RESULTS.txt: temporary text file containing computed TM-scores using TMscore module outputs
        """

        os.makedirs(save_dir, exist_ok=True)
        print (f"Created savedir: {save_dir}")

        grnade_save_path = os.path.join(save_dir, "invfold_seqs")
        rhofold_save_path = os.path.join(save_dir, "fwdfold_pdbs")
        tmp_save_dict = save_dir

        self.save_dir = save_dir
        self.use_invfold = use_invfold
        self.prefer_generated_fasta = prefer_generated_fasta
        self.rhofold_checkpoint = Path("benchmark/checkpoints/RhoFold_pretrained.pt").resolve()

        self.usalign_path = paths.usalign_path
        self.qtmclust_path = paths.qtmclust_path
        self.tmscore_path = paths.tmscore_path
        self.metadata_path = paths.training_metadata_path
        self.gt_dir = paths.training_data_dir
        self.split_dir = getattr(paths, "split_dir", None)
        self.reference_split = getattr(paths, "reference_split", "train")
        self.novelty_reference_scope = "all_metadata_structures_in_selected_split_and_length_range"

        self.tm_thresh = constants.tm_thresh # threshold for qTMclust
        self.rmsd_thresh = constants.rmsd_thresh # threshold for scRMSD
        self.metadata_len_filter = constants.seqlen_range

        # temp files
        self.grnade_save_path = grnade_save_path
        self.rhofold_save_path = rhofold_save_path
        self.tmp_save_dict = tmp_save_dict

        # tools
        self.gRNAde_module = None
        if load_models:
            if self.use_invfold:
                from external_tools.grnade_api.gRNAde import gRNAde

                self.gRNAde_module = gRNAde(split='all', max_num_conformers=1, gpu_id=gpu_id1)
            self.rhofold = RhoFold(rhofold_config)

            self.rhofold.load_state_dict(torch.load(self.rhofold_checkpoint, map_location=torch.device('cpu'))['model'])
            self.rhofold.eval()
            self.device = f"cuda:{gpu_id2}"
            self.rhofold = self.rhofold.to(self.device)

        if not os.path.exists(grnade_save_path):
            print (f"Creating directory for gRNAde generated sequences: {grnade_save_path}")
            os.mkdir(grnade_save_path)

        if not os.path.exists(rhofold_save_path):
            print (f"Creating directory for RhoFold predicted structures: {rhofold_save_path}")
            os.mkdir(rhofold_save_path)

    def _fold_cache_identity(self, sequence):
        """A prediction is reusable only for the same sequence and predictor."""
        stat = self.rhofold_checkpoint.stat()
        return {
            "sequence": str(sequence),
            "predictor": "RhoFold-single-sequence-v1",
            "checkpoint": str(self.rhofold_checkpoint),
            "checkpoint_size": stat.st_size,
            "checkpoint_mtime_ns": stat.st_mtime_ns,
        }

    @staticmethod
    def _fold_cache_matches(pdb_path, identity):
        pdb_path = Path(pdb_path)
        try:
            if not pdb_path.is_file() or pdb_path.stat().st_size == 0:
                return False
            saved = json.loads(pdb_path.with_suffix(".json").read_text(encoding="utf-8"))
            return saved == identity
        except (OSError, ValueError):
            return False

    @staticmethod
    def _ordered_backbone_positions(universe):
        """Match sugar-backbone atoms by residue order and name, not PDB line order."""
        positions = []
        for residue in universe.residues:
            for name in BACKBONE_ATOMS:
                atoms = residue.atoms.select_atoms(f"name {name}")
                if len(atoms) != 1:
                    raise ValueError(
                        f"Residue {residue.resid} requires one {name} atom; found {len(atoms)}"
                    )
                positions.append(atoms.positions[0])
        if not positions:
            raise ValueError("No RNA backbone atoms found")
        return np.asarray(positions)

    def _find_generated_fasta(self, gen_fp):
        fasta_fp = os.path.splitext(gen_fp)[0] + ".fasta"
        return fasta_fp if os.path.exists(fasta_fp) else None

    def _is_eval_pdb_name(self, name):
        """Return True for final generated sample PDBs, excluding trajectories."""
        if not name.endswith(".pdb"):
            return False
        stem = os.path.splitext(os.path.basename(name))[0]
        return "traj" not in stem

    def _list_eval_pdb_names(self, gen_dir):
        return [
            name
            for name in sorted(os.listdir(gen_dir))
            if self._is_eval_pdb_name(name) and os.path.isfile(os.path.join(gen_dir, name))
        ]

    def _write_pdb_name_list(self, path, names):
        with open(path, "w", encoding="utf-8") as handle:
            for name in names:
                handle.write(f"{name}\n")

    def _qtm_cluster_count(self, gen_dir, names, label):
        """Run qTMclust on the specified generated PDBs."""
        if not names:
            return 0
        list_path = os.path.join(self.tmp_save_dict, f"gen_list_{label}.list")
        result_path = os.path.join(self.tmp_save_dict, f"MASTER_QTM_RESULTS_{label}.txt")
        self._write_pdb_name_list(list_path, names)
        subprocess.run(
            [
                self.qtmclust_path,
                "-mol",
                "RNA",
                "-atom",
                f"{ALIGNMENT_ATOM:>4}",
                "-dir",
                gen_dir,
                list_path,
                "-TMcut",
                str(self.tm_thresh),
                "-o",
                result_path,
            ],
            check=True,
        )
        clusters = []
        with open(result_path, "r", encoding="utf-8") as handle:
            for line in handle:
                members = [column.strip() for column in line.strip().split("\t") if column.strip()]
                if members:
                    clusters.append(members)
        observed_members = [
            canonical_sample_id(member) + ".pdb"
            for cluster in clusters
            for member in cluster
        ]
        if len(observed_members) != len(names) or len(set(observed_members)) != len(names):
            raise ValueError(
                "qTMclust output does not contain every input exactly once: "
                f"expected={len(names)}, entries={len(observed_members)}, "
                f"unique={len(set(observed_members))}"
            )
        if set(observed_members) != set(names):
            raise ValueError("qTMclust output membership differs from the explicit input list")
        return len(clusters)

    @staticmethod
    def _record_sctm(record):
        value = record.get("max_scTM")
        if value in (None, ""):
            value = record.get("min_scTM")
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        return score if math.isfinite(score) else None

    def _compute_three_way_diversity(self, gen_dir, evaluated_records):
        """Compute raw diversity, valid-only diversity, and unique-valid yield.

        G is every generated PDB, V is the scTM-valid subset, N=|G|, and
        M=|V|. Records that fail the folding/self-consistency pipeline remain in
        N but cannot enter V.
        """
        generated_names = self._list_eval_pdb_names(gen_dir)
        generated_set = set(generated_names)
        successful_set = set()
        valid_set = set()
        for record in evaluated_records:
            name = f"{record['pdb_name']}.pdb"
            score = self._record_sctm(record)
            if name not in generated_set or score is None:
                continue
            successful_set.add(name)
            if score >= self.tm_thresh:
                valid_set.add(name)
        successful_names = sorted(successful_set)
        valid_names = sorted(valid_set)

        n_generated = len(generated_names)
        n_successful = len(successful_names)
        n_valid = len(valid_names)
        if n_generated == 0:
            raise ValueError("Cannot compute diversity: no generated PDB files were found")

        clusters_generated = self._qtm_cluster_count(gen_dir, generated_names, "all_generated")
        clusters_valid = self._qtm_cluster_count(gen_dir, valid_names, "sctm_valid")
        clusters_successful = self._qtm_cluster_count(gen_dir, successful_names, "successful_eval")
        summary = {
            "definition": {
                "validity": f"max scTM >= {self.tm_thresh}",
                "D_raw": "K(G) / N",
                "D_valid": "K(V) / M",
                "Y_UV": "K(V) / N",
            },
            "N_generated": n_generated,
            "S_successful_self_consistency": n_successful,
            "M_valid": n_valid,
            "K_generated": clusters_generated,
            "K_successful_self_consistency": clusters_successful,
            "K_valid": clusters_valid,
            "D_raw": clusters_generated / n_generated,
            "D_valid": None if n_valid == 0 else clusters_valid / n_valid,
            "Y_UV": clusters_valid / n_generated,
            "D_legacy_successful": None if n_successful == 0 else clusters_successful / n_successful,
            "self_consistency_success_rate": n_successful / n_generated,
            "valid_yield": n_valid / n_generated,
        }
        with open(os.path.join(self.save_dir, "diversity_three_way.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return summary

    def _load_generated_sequence_records(self, fasta_fp, n_seqs_per_bb):
        records = list(SeqIO.parse(fasta_fp, "fasta"))
        if not records:
            raise ValueError(f"Generated FASTA is empty: {fasta_fp}")
        return records[:n_seqs_per_bb]

    def _audit_invfold_input(self, gen_fp):
        atom_names = set()
        residue_names = set()
        with open(gen_fp, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                if not raw_line.startswith("ATOM"):
                    continue
                atom_names.add(raw_line[12:16].strip())
                residue_names.add(raw_line[17:20].strip())
        return {
            "has_base_anchor": bool({"N1", "N9"} & atom_names),
            "residue_names": sorted(residue_names),
        }

    def perform_eval(self, gen_dir, flatten_dir=False, plot=False, exclude=None, source=None):
        """Write final metrics only when every requested evaluation succeeds."""
        status_path = Path(self.save_dir) / "evaluation_status.json"
        final_path = Path(self.save_dir) / "final_metrics.pt"
        final_path.unlink(missing_ok=True)
        if source is None:
            for name in (
                "final_metrics_scrmsd_ranked_samples_dict.pt",
                "final_metrics_scrmsd_ranked_samples_arr.pt",
                "final_metrics_local.pt", "final_metrics_global.pt",
                "diversity_three_way.json", "partial_self_consistency.pt",
            ):
                (Path(self.save_dir) / name).unlink(missing_ok=True)
        status = {"state": "running", "generated_directory": str(Path(gen_dir).resolve())}
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        try:
            result = self._perform_eval(gen_dir, flatten_dir, plot, exclude or [], source)
        except Exception as error:
            status.update(state="failed", error=f"{type(error).__name__}: {error}")
            status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            raise
        status["state"] = "complete"
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        return result

    def _perform_eval(self, gen_dir, flatten_dir=False, plot=False, exclude=(), source=None):
        """
        Params:
            - gen_dir : path of the generated samples (inference or validation)
                -> This directory must be 1-dimensional without any subdirectories.
                   Ensure the PDB files are named uniquely to avoid overwriting them.
            - flatten_dir : boolean to flatten directory in case samples are separated by sequence length (default: False).
                -> If set to True, if `gen_dir` has subdirectories based on length, a new
                   directory called "flattened_samples/" is created which contains all the samples
                   with a new naming convention (seqlen_na_samples_idx.pdb)
            - exclude : list of strings corresponding to which evals not to perform
                -> pick from ["struct_dihedrals", "struct_bond_angles", "struct_bond_dist", "batch_clusters", "batch_pdbTM"]
            - plot : boolean to plot histograms of sturctural metrics (eg: distance, angles, dihedrals)

        Notes:
            structural metrics computed:
                1. Histogram of adjacent C4' bond distances averaged over entire backbone
                2. Histogram of consecutive C4' pseudo-dihedral angles

            batch metrics computed:
                1. diversity percent (using qTMclust)
                2. novelty percent (using USAlign)
                3. designability percent (using gRNAde + E2EFold/RhoFold)

        - This is done regardless of sequence length. There is no extra bucketing for length.
        - Only run this method, do not use any of the other helper methods with a `_compute_xyz()` name
            - Use the `exclude` param if want to ignore specific evaluations

        Returns:
            dictionary of all metrics
        """

        structural_eval_dict = {
            "struct_dihedrals": self._compute_dihedral_angles,
            "struct_bond_angles": self._compute_bond_angles,
            "struct_bond_dist": self._compute_c4_to_c4_distance,
        }

        batch_eval_dict = {
            "batch_pdbTM": self._compute_usalign_metrics # uses gen_dir and self.gt_dir
        }

        all_keys = list(structural_eval_dict.keys()) + list(batch_eval_dict.keys())
        final_metric_dict = {key:[] for key in all_keys}

        if flatten_dir:
            """
            To be used if `gen_dir` looks like this:
            |- gen_dir
            |--- length_a/
            |----- na_sample_1.pdb
            |----- na_sample_2.pdb
            |----- na_sample_3.pdb
            |--- length_b/
            |----- na_sample_1.pdb
            |----- na_sample_2.pdb
            |----- na_sample_3.pdb
            |--- length_c/
            |----- na_sample_1.pdb
            |----- na_sample_2.pdb
            |----- na_sample_3.pdb

            The final result will be:

            |- flattened_samples/ # copied + renamed files
            |--- a_na_sample_1.pdb
            |--- a_na_sample_2.pdb
            |--- a_na_sample_3.pdb
            |--- b_na_sample_1.pdb
            |--- b_na_sample_2.pdb
            |--- b_na_sample_3.pdb
            |--- c_na_sample_1.pdb
            |--- c_na_sample_2.pdb
            |--- c_na_sample_3.pdb
            |- gen_dir/ # originals not affected
            |--- length_a/
            |--- length_b/
            |--- length_c/
            """
            print ("Flattening directory ...")
            FWD_PATH = f"{gen_dir}/flattened_samples/" # assume `gen_dir` and `flattened_samples` are at the same level
            self._flatten_dir(gen_dir, FWD_PATH)
            gen_dir = FWD_PATH # set new location as `gen_dir`

        os.makedirs(self.save_dir, exist_ok=True)

        metadata_df = pd.read_csv(self.metadata_path)
        metadata_df = filter_metadata_by_split(
            metadata_df,
            split_dir=self.split_dir,
            split_name=self.reference_split,
        )
        metadata_df = metadata_df[metadata_df['modeled_na_seq_len'] >= self.metadata_len_filter[0]]
        metadata_df = metadata_df[metadata_df['modeled_na_seq_len'] <= self.metadata_len_filter[1]]
        reference_names = sorted(set(metadata_df["pdb_name"].astype(str)))
        if not reference_names:
            raise ValueError("No reference structures remain after split and length filtering")
        missing = [name for name in reference_names if not os.path.isfile(
            os.path.join(self.gt_dir, f"{name}.pdb")
        )]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} reference PDBs required by the metadata: {missing[:5]}"
            )
        self._write_pdb_name_list(
            os.path.join(self.tmp_save_dict, "gt_list.list"),
            [f"{name}.pdb" for name in reference_names],
        )
        print(f"Loaded all {len(reference_names)} reference structures in the selected split and length range")

        if source:
            print ("Loading rankings from source ...")
            ranked_designable_seqlen_sample_dict = torch.load(source)
            flattened_ranked_designable_seqlen_sample_arr = []
            for _, sample_arr in ranked_designable_seqlen_sample_dict.items():
                flattened_ranked_designable_seqlen_sample_arr.extend(sample_arr)
        else:
            scrmsd_sample_list = self._get_designable_samples(gen_dir)
            ranked_designable_seqlen_sample_dict = self._rank_designable_topk(scrmsd_sample_list)

            flattened_ranked_designable_seqlen_sample_arr = []
            for _, sample_arr in ranked_designable_seqlen_sample_dict.items():
                flattened_ranked_designable_seqlen_sample_arr.extend(sample_arr)

        if source and any(
            record.get("tm_atom") != ALIGNMENT_ATOM
            or record.get("backbone_atoms") != list(BACKBONE_ATOMS)
            for record in flattened_ranked_designable_seqlen_sample_arr
        ):
            raise ValueError("Cached self-consistency metrics use a different or unspecified atom protocol; recompute them")
        expected_names = set(self._list_eval_pdb_names(gen_dir))
        scored_names = [f"{record['pdb_name']}.pdb" for record in flattened_ranked_designable_seqlen_sample_arr]
        if set(scored_names) != expected_names or len(scored_names) != len(expected_names):
            raise ValueError("Self-consistency records must cover each generated sample exactly once")
        for record in flattened_ranked_designable_seqlen_sample_arr:
            score = self._record_sctm(record)
            if score is None or not 0 <= score <= 1 or not math.isfinite(float(record["min_scRMSD"])):
                raise ValueError("Self-consistency records contain invalid scores")
        torch.save(ranked_designable_seqlen_sample_dict,
                   Path(self.save_dir) / "final_metrics_scrmsd_ranked_samples_dict.pt")
        torch.save(flattened_ranked_designable_seqlen_sample_arr,
                   Path(self.save_dir) / "final_metrics_scrmsd_ranked_samples_arr.pt")
        valid_names = sorted({
            f"{record['pdb_name']}.pdb"
            for record in flattened_ranked_designable_seqlen_sample_arr
            if (score := self._record_sctm(record)) is not None and score >= self.tm_thresh
        })
        self._write_pdb_name_list(
            os.path.join(self.tmp_save_dict, "gen_list.list"), valid_names
        )

        diversity_summary = self._compute_three_way_diversity(
            gen_dir,
            flattened_ranked_designable_seqlen_sample_arr,
        )
        # Report raw diversity, valid diversity, and unique valid yield.
        final_metric_dict["batch_clusters"] = [diversity_summary["D_raw"]]
        final_metric_dict["batch_clusters_valid"] = [diversity_summary["D_valid"]]
        final_metric_dict["unique_valid_yield"] = [diversity_summary["Y_UV"]]
        final_metric_dict["diversity_counts"] = diversity_summary
        final_metric_dict["alignment_protocol"] = {
            "tm_molecule": "RNA",
            "tm_atom": ALIGNMENT_ATOM,
            "tm_normalization": "second_structure_length (TM2)",
            "scrmsd_scgdt_atoms": list(BACKBONE_ATOMS),
            "structural_geometry_atom": "C4'",
        }
        final_metric_dict["batch_pdbTM_reference_scope"] = self.novelty_reference_scope
        final_metric_dict["batch_pdbTM_generated_scope"] = "sctm_valid_samples"
        final_metric_dict["batch_pdbTM_reference_count"] = len(reference_names)
        final_metric_dict["excluded_metrics"] = list(exclude)

        try:
            # structural metrics
            for eval_name, func in structural_eval_dict.items():
                if eval_name not in exclude: # do not compute excluded eval metrics
                    print (f"#### Performing eval: '{eval_name}'")
                    for sample_name in self._list_eval_pdb_names(gen_dir):
                        fp = os.path.join(gen_dir, sample_name)
                        uni = mda.Universe(fp, format="PDB")
                        c4p_atoms = uni.select_atoms("name C4'") # only select
                        c4p_coords = torch.from_numpy(c4p_atoms.positions)

                        ret_arr = func(c4p_coords) # collect from samples
                        final_metric_dict[eval_name].extend(ret_arr)
                print (f"\tComputed {len(final_metric_dict[eval_name])}-long array ...")
        except Exception as e:
            raise RuntimeError("Structural metric computation failed") from e

        final_metric_dict_struct = {key: final_metric_dict[key] for key in list(structural_eval_dict.keys())}
        torch.save(
            final_metric_dict_struct,
            os.path.join(self.save_dir, "final_metrics_local.pt")
        )

        try:
            # whole-batch metrics
            for eval_name, func in batch_eval_dict.items():
                if eval_name not in exclude: # do not compute excluded eval metrics
                    print (f"#### Performing eval: '{eval_name}'")
                    ret_arr = func(gen_dir)
                    final_metric_dict[eval_name].extend(ret_arr)
        except Exception as e:
            raise RuntimeError("External-tool metric computation failed") from e

        final_metric_dict_tools = {
            key: final_metric_dict[key]
            for key in [
                "batch_clusters",
                "batch_clusters_valid",
                "unique_valid_yield",
                "diversity_counts",
                "alignment_protocol",
                "batch_pdbTM_reference_scope",
                "batch_pdbTM_generated_scope",
                "batch_pdbTM_reference_count",
                *list(batch_eval_dict.keys()),
            ]
        }
        torch.save(
            final_metric_dict_tools,
            os.path.join(self.save_dir, "final_metrics_global.pt")
        )

        # save final metric dict
        torch.save(
            final_metric_dict,
            os.path.join(self.save_dir, "final_metrics.pt")
        )

        return final_metric_dict

    def _flatten_dir(self, gen_dir, FWD_PATH):
        if os.path.exists(FWD_PATH):
            shutil.rmtree(FWD_PATH)
        print ("Created flattened directory ...")
        os.mkdir(FWD_PATH)

        for pdb_size_dir in os.listdir(gen_dir):
            fp = os.path.join(gen_dir, pdb_size_dir)
            if os.path.isdir(fp) and pdb_size_dir.startswith("length_") and pdb_size_dir[7:].isdigit():
                length = int(pdb_size_dir.split("_")[-1])
                for pdb_name in os.listdir(fp):
                    if self._is_eval_pdb_name(pdb_name): # ignore trajectories, only keep final structures
                        try:
                            idx = int(os.path.splitext(pdb_name)[0].split("_")[-1])
                            fpp = os.path.join(gen_dir, pdb_size_dir, pdb_name)
                            fasta_src = os.path.join(gen_dir, pdb_size_dir, f"sample_{idx}.fasta")
                            pdb_dst = os.path.join(FWD_PATH, f"{length}_na_sample_{idx}.pdb")
                            fasta_dst = os.path.join(FWD_PATH, f"{length}_na_sample_{idx}.fasta")

                            shutil.copy2(fpp, pdb_dst)
                            if os.path.exists(fasta_src):
                                shutil.copy2(fasta_src, fasta_dst)
                        except Exception as e:
                            raise RuntimeError(f"Cannot flatten sample {fp}/{pdb_name}") from e

    def _compute_c4_to_c4_distance(self, coords):
        """
        gen_dir : path of the generated samples (inference or validation)

        Returns:
            Array of all adjacent-residue C4' bond distances over the entire backbone, for all samples in `gen_dir`
        """
        rolled_coords = torch.roll(coords, shifts=1, dims=0)
        dist = torch.linalg.norm((coords - rolled_coords), dim=-1, ord=2)[1:].mean().item() # we add [1:] to ignore the first column (rolled up from the bottom)
        return [dist]

    def _compute_dihedral_angles(self, coords):
        """
        gen_dir : path of the generated samples (inference or validation)

        Returns:
            C4' pseudo-dihedral angles for consecutive quadruplets in each sample
        """
        def _dihedral_angle_helper(a, b, c, d, eps = 1e-7):
            eps = torch.tensor(eps, device=a.device) # type: ignore

            # bc = F.normalize(b - c, dim=2)
            bc = F.normalize(b - c, dim=-1)
            # n1 = torch.cross(F.normalize(a - b, dim=2), bc)
            n1 = torch.cross(F.normalize(a - b, dim=-1), bc, dim=-1)
            # n2 = torch.cross(bc, F.normalize(c - d, dim=2))
            n2 = torch.cross(bc, F.normalize(c - d, dim=-1), dim=-1)
            # x = (n1 * n2).sum(dim=2)
            x = (n1 * n2).sum(dim=-1)
            x = torch.clamp(x, -1 + eps, 1 - eps)
            x[x.abs() < eps] = eps

            y = (torch.cross(n1, bc, dim=-1) * n2).sum(dim=-1)
            return torch.atan2(y, x).numpy().tolist()

        C4_i0 = coords[0:-3, :]
        C4_i1 = coords[1:-2, :]
        C4_i2 = coords[2:-1, :]
        C4_i3 = coords[3:, :]
        angles = _dihedral_angle_helper(C4_i0, C4_i1, C4_i2, C4_i3)
        return angles

    def _compute_bond_angles(self, coords):
        """
        gen_dir : path of the generated samples (inference or validation)

        Returns:
            Array of all bond angles between C4' atom triplets between adjacent residues
        """
        def _to_ang_helper(a, b, c):
            if a.ndim == 1:
                a = a.unsqueeze(0)
            if b.ndim == 1:
                b = b.unsqueeze(0)
            if c.ndim == 1:
                c = c.unsqueeze(0)

            ba = b - a
            bc = b - c
            denominator = torch.norm(ba, dim=-1) * torch.norm(bc, dim=-1)
            if torch.any(denominator == 0):
                raise ValueError("Bond angle is undefined for coincident C4' atoms")
            cosine = (ba * bc).sum(dim=-1) / denominator
            return torch.acos(cosine.clamp(-1.0, 1.0)).numpy().tolist()

        C4_i0 = coords[0:-2, :]
        C4_i1 = coords[1:-1, :]
        C4_i2 = coords[2:, :]
        angles = _to_ang_helper(C4_i0, C4_i1, C4_i2)
        return angles

    @torch.no_grad()
    def _get_designable_samples(self, gen_dir, n_seqs_per_bb=8):
        """
        For each generated PDB,
            1. inverse fold using gRNAde
            2. forward fold using RhoFold
            3. compute metrics between generated and predicted structures

        Metrics include:
            scTM, scRMSD, scGDT

        Return:
            final array of designable samples and respective metrics
        """

        gen_pdb_rmsds = []
        failures = []

        # inverse-fold all generated samples
        pbar = tqdm(self._list_eval_pdb_names(gen_dir))
        for pdb_name in pbar: # outer loop: generated PDBs
            pbar.set_description(pdb_name)
            gen_fp = os.path.join(gen_dir, pdb_name)

            try:
                seqlen = int(pdb_name.split("_")[0])
                pdb_stem = os.path.splitext(pdb_name)[0]

                # inner loop 1: generate sequences
                out_fp = os.path.join(self.grnade_save_path, f"{pdb_stem}.fasta")

                generated_fasta_fp = self._find_generated_fasta(gen_fp)

                # Use the generated FASTA when requested; otherwise inverse-fold
                # a sequence from the backbone.
                if self.prefer_generated_fasta and generated_fasta_fp is not None:
                    gen_seqs = self._load_generated_sequence_records(
                        generated_fasta_fp,
                        n_seqs_per_bb=n_seqs_per_bb,
                    )
                elif self.use_invfold:
                    audit = self._audit_invfold_input(gen_fp)
                    if not audit["has_base_anchor"]:
                        raise ValueError(
                            "gRNAde-compatible inverse-fold input is missing N1/N9 base-anchor atoms; "
                            f"residue names seen: {audit['residue_names']}"
                        )
                    if self.gRNAde_module is None:
                        raise RuntimeError("gRNAde module was not loaded although use_invfold=True.")
                    gen_seqs, _, _, _ = self.gRNAde_module.design_from_pdb_file(
                        pdb_filepath=gen_fp,
                        output_filepath=out_fp,
                        n_samples=n_seqs_per_bb,
                        temperature=0.1,
                    )
                elif generated_fasta_fp is not None:
                    gen_seqs = self._load_generated_sequence_records(
                        generated_fasta_fp,
                        n_seqs_per_bb=n_seqs_per_bb,
                    )
                else:
                    raise ValueError(
                        "No generated FASTA found beside the PDB and use_invfold=False. "
                        "Enable inverse folding or provide generated FASTA files."
                    )

                if not gen_seqs:
                    raise ValueError("No sequences available for forward folding")
                for seq in gen_seqs:
                    sequence = str(seq.seq)
                    if len(sequence) != seqlen or set(sequence) - set("ACGU"):
                        raise ValueError(f"Expected {seqlen} canonical RNA bases, received {sequence!r}")
                torch.cuda.empty_cache()
                gc.collect()

                seq_files = []
                # split sequences across N_seq different FASTA files
                for seq_idx, seq in enumerate(gen_seqs):
                    updated_fasta_stem = f"{pdb_stem}_{seq_idx}"
                    new_out_fp = os.path.join(self.grnade_save_path, f"{updated_fasta_stem}.fasta")
                    seq_files.append([
                        new_out_fp,
                        updated_fasta_stem,
                        seq_idx,
                        self._fold_cache_identity(seq.seq),
                    ])
                    SeqIO.write([seq], new_out_fp, "fasta")

                # inner loop 2: for each predicted sequence, predict the structure
                for rec in seq_files:
                    seq_fp, fasta_fn, seq_idx, cache_identity = rec

                    # Never reuse a prediction solely because its sample ID matches.
                    rhofold_out_fp = os.path.join(self.rhofold_save_path, f'{fasta_fn}.pdb')
                    if self._fold_cache_matches(rhofold_out_fp, cache_identity):
                        continue
                    cache_path = Path(rhofold_out_fp).with_suffix(".json")
                    cache_path.unlink(missing_ok=True)

                    input_a3m = input_fas = seq_fp

                    data_dict = get_features(input_fas, input_a3m)

                    outputs = self.rhofold(
                        tokens=data_dict['tokens'].to(self.device),
                        rna_fm_tokens=data_dict['rna_fm_tokens'].to(self.device),
                        seq=data_dict['seq'],
                    )

                    output = outputs[-1]
                    unrelaxed_model = rhofold_out_fp
                    node_cords_pred = output['cord_tns_pred'][-1].squeeze(0)

                    self.rhofold.structure_module.converter.export_pdb_file(
                        data_dict['seq'],
                        node_cords_pred.data.cpu().numpy(),
                        path=unrelaxed_model,
                        chain_id=None,
                        confidence=output['plddt'][0].data.cpu().numpy(),
                        logger=None
                    )
                    cache_path.write_text(json.dumps(cache_identity, indent=2) + "\n", encoding="utf-8")

                    torch.cuda.empty_cache()
                    gc.collect()

                # inner loop 3: compute RMSDs between predicted and original generated backbone
                gen_uni = mda.Universe(gen_fp, format="PDB")
                gen_backbone = self._ordered_backbone_positions(gen_uni)

                # temp metrics
                cur_min_rmsd = float("inf") # very large number
                cur_max_scTM = float("-inf")
                cur_max_scGDT = float("-inf")
                cur_scGDT_at_min_rmsd = float("-inf")
                cur_best_pred_pdb_name = "" # RhoFold PDB with the closest RMSD match
                cur_best_tm_pred_pdb_name = "" # RhoFold PDB with the best scTM match
                n_res = len(gen_backbone) // len(BACKBONE_ATOMS)

                for k in range(len(gen_seqs)):
                    # rhofold_pdb_name : <seq_len>_na_sample_<idx>_<seq_idx>.pdb
                    rhofold_fp = os.path.join(self.rhofold_save_path, f"{pdb_stem}_{k}.pdb")
                    if not os.path.exists(rhofold_fp):
                        raise FileNotFoundError(rhofold_fp)
                    rf_uni = mda.Universe(rhofold_fp, format="PDB")
                    pred_backbone = self._ordered_backbone_positions(rf_uni)

                    if gen_backbone.shape != pred_backbone.shape or len(gen_backbone) != seqlen:
                        raise ValueError("Generated and folded C4' coordinates must match the requested length")

                    # compute scTM
                    tm = self._compute_tm_score(
                                gen_path=gen_fp,
                                pred_path=rhofold_fp,
                                gen_pdb_name=pdb_name,
                                pred_pdb_name=f"{pdb_stem}_{k}.pdb"
                            ) # C4' TM-score, normalized by the predicted RNA length

                    # compute scGDT
                    gdt = self._compute_gdt(
                                gen_backbone,
                                pred_backbone,
                            ) # Corresponding C4' atoms

                    # compute scRMSD
                    r_score = rms.rmsd(
                                gen_backbone,
                                pred_backbone,
                                center=True,
                                superposition=True
                            ) # Corresponding C4' atoms

                    # RNA-FrameFlow reports max scTM over Nseq and min scRMSD
                    # over Nseq independently. Do not tie TM validity to the
                    # sequence that happened to minimize RMSD.
                    if cur_max_scTM < tm:
                        cur_max_scTM = tm
                        cur_best_tm_pred_pdb_name = f"{pdb_stem}_{k}"
                    if cur_max_scGDT < gdt:
                        cur_max_scGDT = gdt

                    # use scRMSD to track the best structure among all RhoFold predictions
                    if cur_min_rmsd > r_score:
                        cur_min_rmsd = r_score
                        cur_best_pred_pdb_name = f"{pdb_stem}_{k}" # store the best predicted match's PDB filename from RhoFold
                        cur_scGDT_at_min_rmsd = gdt

                if not all(math.isfinite(float(value)) for value in (cur_min_rmsd, cur_max_scTM, cur_max_scGDT)):
                    raise ValueError("Self-consistency metrics contain a non-finite value")
                # store record for faster future retrieval
                gen_pdb_rmsds.append({
                    "gen_dir": gen_dir,
                    "pdb_name": pdb_stem,
                    "min_scRMSD": cur_min_rmsd,
                    "n_res": n_res,
                    "cur_best_pred_pdb_name": cur_best_pred_pdb_name,
                    "cur_best_tm_pred_pdb_name": cur_best_tm_pred_pdb_name,
                    "min_scTM": cur_max_scTM,
                    "max_scTM": cur_max_scTM,
                    "min_scGDT": cur_scGDT_at_min_rmsd,
                    "max_scGDT": cur_max_scGDT,
                    "seqlen": seqlen,
                    "backbone_atoms": list(BACKBONE_ATOMS),
                    "tm_atom": ALIGNMENT_ATOM,
                })
            except Exception as e:
                failures.append({"sample": pdb_name, "error": f"{type(e).__name__}: {e}"})
                print(f"[EvalSuite] Failed {pdb_name}: {e}")
                torch.cuda.empty_cache()
                gc.collect()

        failure_path = Path(self.save_dir) / "failed_samples.json"
        failure_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        if failures:
            torch.save(gen_pdb_rmsds, Path(self.save_dir) / "partial_self_consistency.pt")
            raise RuntimeError(f"Self-consistency failed for {len(failures)} samples; see {failure_path}")
        if not gen_pdb_rmsds:
            raise ValueError("No generated RNA samples were evaluated")
        return gen_pdb_rmsds

    def _rank_designable_topk(self, sample_arr, rankby="min_scRMSD"):
        """
        Params:
            sample_arr (list) : list of samples with respective self-consistency details
            rankby (str): select from ["min_scRMSD", "max_scTM", "max_scGDT"].
                Aliases "min_scTM" and "min_scGDT" are also accepted.

        Remarks:
            Sorts samples according to given metric (scTM, scRMSD, scGDT) to compute validity

        Returns:
            Sorted list of valid samples
        """

        """
        NOTE:
        The older version of this method also extracted the best "K" samples (K was a method parameter).
        We've removed this functionality and take all samples, even if they aren't among the best "K".
        """

        ranked_samples = {}
        for sample in sample_arr:
            # _, _, _, num_res, _ = sample
            num_res = sample['n_res']
            if num_res not in ranked_samples:
                ranked_samples[num_res] = [sample]
            else:
                ranked_samples[num_res].append(sample)

        for n_res, seqlen_sample_arr in ranked_samples.items():
            # pdb_name_wo_ext doesn't have the .pdb extension

            if rankby in {"min_scTM", "min_scGDT", "max_scTM", "max_scGDT"}:
                legacy_rankby = rankby.replace("max_", "min_")
                ranked_seqlen_sample_arr = sorted(
                    seqlen_sample_arr,
                    key=lambda rec: rec.get(rankby, rec[legacy_rankby]),
                    reverse=True,
                ) # sort in descending order by scTM/scGDT
            elif rankby == "min_scRMSD":
                ranked_seqlen_sample_arr = sorted(seqlen_sample_arr, key=lambda rec : rec[rankby], reverse=False) # sort in ascending order by scRMSD

            filtered_seqlen_sample_arr = ranked_seqlen_sample_arr
            ranked_samples[n_res] = filtered_seqlen_sample_arr # extract the top-K and assign back to that size bucket

        return ranked_samples # return final ranked+filtered size buckets

    def _compute_gdt(self, gen_pos, pred_pos):
        return metrics.gdt_score(
                                x=torch.from_numpy(gen_pos).unsqueeze(0),
                                y=torch.from_numpy(pred_pos).unsqueeze(0)
                            )

    def _compute_tm_score(self, gen_path, pred_path, gen_pdb_name, pred_pdb_name):
        tmscore_tmp_dir = "tmscore_tmp_dir/"
        os.makedirs(os.path.join(self.tmp_save_dict, tmscore_tmp_dir), exist_ok=True)

        gen_tmp_dir = os.path.join(self.tmp_save_dict, tmscore_tmp_dir, "gen")
        pred_tmp_dir = os.path.join(self.tmp_save_dict, tmscore_tmp_dir, "pred")

        os.makedirs(gen_tmp_dir, exist_ok=True)
        os.makedirs(pred_tmp_dir, exist_ok=True)
        shutil.copy2(gen_path, gen_tmp_dir)
        shutil.copy2(pred_path, pred_tmp_dir)

        gen_list_path = os.path.join(self.tmp_save_dict, tmscore_tmp_dir, "gen_tmscore.list")
        pred_list_path = os.path.join(self.tmp_save_dict, tmscore_tmp_dir, "pred_tmscore.list")
        with open(gen_list_path, "w") as f:
            f.write(gen_pdb_name)
        with open(pred_list_path, "w") as f:
            f.write(pred_pdb_name)

        ret_path = os.path.join(self.tmp_save_dict, "TMSCORE_TMP_RESULTS.txt")
        with open(ret_path, "w", encoding="utf-8") as output_handle:
            subprocess.run(
                [
                    self.tmscore_path,
                    "-mol",
                    "RNA",
                    "-atom",
                    f"{ALIGNMENT_ATOM:>4}",
                    "-dir1",
                    gen_tmp_dir,
                    gen_list_path,
                    "-dir2",
                    pred_tmp_dir,
                    pred_list_path,
                    "-outfmt",
                    "2",
                ],
                check=True,
                stdout=output_handle,
            )

        df = pd.read_csv(ret_path, delimiter='\t')
        df = df[['#PDBchain1', "PDBchain2", "TM2", "Lali"]]
        if len(df) != 1 or not np.isfinite(df["TM2"]).all() or not df["TM2"].between(0, 1).all():
            raise ValueError("TMscore must return exactly one finite score in [0, 1]")
        tmscore = float(df["TM2"].iloc[0])

        # delete all temp files
        shutil.rmtree(os.path.join(self.tmp_save_dict, tmscore_tmp_dir))
        # os.system(f"rm {ret_path}")

        return tmscore

    def compute_tmscore_from_usalign(self, usalign_report_path):
        df = pd.read_csv(usalign_report_path, delimiter="\t")
        df = df[['#PDBchain1', 'PDBchain2', 'TM2']]

        df_ = df.groupby("#PDBchain1")
        max_tmscore_per_pdb = df_['TM2'].max()

        tm_json = max_tmscore_per_pdb.to_json()
        tm_json = json.loads(tm_json)

        seqlens = []
        pdb_names = []
        tm_scores = []
        for pdb_name, tm2 in tm_json.items():
            seqlen = int(pdb_name.split("_")[0].strip("/"))
            seqlens.append(seqlen)
            pdb_names.append(pdb_name)
            tm_scores.append(tm2)

        dct = {
            "seqlen": seqlens,
            "pdb_name": pdb_names,
            "tm2": tm_scores
        }

        tm_report_df = pd.DataFrame.from_dict(dct)
        return tm_report_df

    @staticmethod
    def _validate_usalign_coverage(frame, generated_names, reference_names):
        generated = {canonical_sample_id(name) for name in generated_names}
        references = {canonical_sample_id(name) for name in reference_names}
        pairs = frame[["#PDBchain1", "PDBchain2"]].copy()
        pairs["#PDBchain1"] = pairs["#PDBchain1"].map(canonical_sample_id)
        pairs["PDBchain2"] = pairs["PDBchain2"].map(canonical_sample_id)
        if (set(pairs["#PDBchain1"]) != generated
                or set(pairs["PDBchain2"]) != references
                or len(pairs) != len(generated) * len(references)
                or pairs.duplicated().any()):
            raise ValueError("USalign must compare every valid sample with every reference exactly once")
        if not np.isfinite(frame["TM2"]).all() or not frame["TM2"].between(0, 1).all():
            raise ValueError("USalign returned a non-finite or out-of-range TM2 score")

    def _compute_usalign_metrics(self, gen_dir):
        """
        gen_dir : path of the generated samples (inference or validation)
        """

        ret_path = os.path.join(self.tmp_save_dict, "MASTER_USALIGN_RESULTS.txt")
        gen_path = os.path.join(gen_dir)
        gt_path = os.path.join(self.gt_dir)
        gen_list_path = os.path.join(self.tmp_save_dict, "gen_list.list")
        gt_list_path = os.path.join(self.tmp_save_dict, "gt_list.list")

        # cancel if nothing is considered designable
        with open(gen_list_path, "r") as f:
            d = f.readlines()
        if len(d) < 1:
            return []

        with open(ret_path, "w", encoding="utf-8") as output_handle:
            subprocess.run(
                [
                    self.usalign_path,
                    "-mol",
                    "RNA",
                    "-atom",
                    f"{ALIGNMENT_ATOM:>4}",
                    "-dir1",
                    gen_dir,
                    gen_list_path,
                    "-dir2",
                    gt_path,
                    gt_list_path,
                    "-outfmt",
                    "2",
                ],
                check=True,
                stdout=output_handle,
            )

        df = pd.read_csv(ret_path, delimiter="\t")
        df = df[['#PDBchain1', 'PDBchain2', 'TM2']]
        with open(gt_list_path, encoding="utf-8") as handle:
            reference_names = [line.strip() for line in handle if line.strip()]
        self._validate_usalign_coverage(df, d, reference_names)

        # For each generated X, look at Y={y1, ..., yn}, get max across all (X,yi) pairs and then take mean of |X| generated samples
        df_ = df.groupby("#PDBchain1")
        avg_pdbTM = df_['TM2'].max().mean()
        return [avg_pdbTM]

    def _compute_qTMclust_metrics(self, gen_dir):
        """
        gen_dir : path of the generated samples (inference or validation)
        """

        names = self._list_eval_pdb_names(gen_dir)
        if not names:
            return [-1]
        clusters = self._qtm_cluster_count(gen_dir, names, "all_generated")
        return [clusters / len(names)]

    def load_from_metric_dict(self, metric_dict_path):
        return torch.load(metric_dict_path)

    def print_metrics(self, metric_dict):
        def _format_ratio(numerator, denominator):
            if denominator == 0:
                return "N/A"
            return f"{numerator / denominator:.5f}"

        print ("\n\nMETRICS:")
        batch_clusters = metric_dict["batch_clusters"]
        batch_clusters_valid = metric_dict.get("batch_clusters_valid")
        unique_valid_yield = metric_dict.get("unique_valid_yield")
        diversity_counts = metric_dict.get("diversity_counts")
        batch_pdbtm = metric_dict["batch_pdbTM"]
        pdbtm_scope = metric_dict.get(
            "batch_pdbTM_reference_scope",
            "historical_scope_not_recorded",
        )

        if batch_clusters == [-1]:
            print ("D_raw = K(G) / N: N/A | 0 generated samples")
        else:
            print (f"D_raw = K(G) / N: {batch_clusters}")
        if batch_clusters_valid is not None:
            print (f"D_valid = K(V) / M: {batch_clusters_valid}")
        if unique_valid_yield is not None:
            print (f"Y_UV = K(V) / N: {unique_valid_yield}")
        if diversity_counts is not None:
            print(
                "Diversity counts: "
                f"N={diversity_counts['N_generated']}, "
                f"S={diversity_counts['S_successful_self_consistency']}, "
                f"M={diversity_counts['M_valid']}, "
                f"K(G)={diversity_counts['K_generated']}, "
                f"K(V)={diversity_counts['K_valid']}"
            )
            print(
                "Overall scTM-valid yield M/N: "
                f"{diversity_counts['valid_yield']:.5f}; "
                "conditional scTM-validity M/S: "
                f"{_format_ratio(diversity_counts['M_valid'], diversity_counts['S_successful_self_consistency'])}"
            )

        if "batch_pdbTM" in metric_dict.get("excluded_metrics", []):
            print(f"Reference-set pdbTM ({pdbtm_scope}): excluded")
        elif len(batch_pdbtm) == 0:
            print (f"Reference-set pdbTM ({pdbtm_scope}): N/A | no scTM-valid samples")
        else:
            print (f"Reference-set pdbTM ({pdbtm_scope}; scTM-valid subset): {batch_pdbtm}")

        sorted_samples_path = f"{self.save_dir}/final_metrics_scrmsd_ranked_samples_dict.pt"
        sorted_scrmsd = torch.load(sorted_samples_path)

        temp = []
        total = 0
        filt = 0

        for l, v in sorted_scrmsd.items():
            rmsd = list(map(lambda x : x['min_scRMSD'], v))
            rmsd = list(sorted(rmsd))
            filt_rmsd = list(filter(lambda x : x <= self.rmsd_thresh, rmsd))
            temp.extend(rmsd)
            total += len(rmsd)
            filt += len(filt_rmsd)
        print (
            f"Validity (% <= {self.rmsd_thresh}A): "
            f"{_format_ratio(filt, total)} | {filt} / {total}"
        )

        temp = []
        total = 0
        filt = 0

        for _, v in sorted_scrmsd.items():
            tm = list(map(lambda x : x.get('max_scTM', x['min_scTM']), v))
            tm = list(sorted(tm))
            filt_tm = list(filter(lambda x : x >= self.tm_thresh, tm))
            temp.extend(tm)
            total += len(tm)
            filt += len(filt_tm)
        print (
            f"Validity (% >= {self.tm_thresh}): "
            f"{_format_ratio(filt, total)} | {filt} / {total}"
        )


class DuetRNAEvalSuite(EvalSuite):
    """Joint/direct benchmark wrapper for dual-frame DuetRNA models."""

    def __init__(
        self,
        save_dir,
        paths=None,
        constants=None,
        gpu_id1=0,
        gpu_id2=1,
        use_invfold: bool = False,
        prefer_generated_fasta: bool = True,
        load_models: bool = True,
    ):
        super().__init__(
            save_dir=save_dir,
            paths=paths,
            constants=constants,
            gpu_id1=gpu_id1,
            gpu_id2=gpu_id2,
            use_invfold=use_invfold,
            prefer_generated_fasta=prefer_generated_fasta,
            load_models=load_models,
        )
