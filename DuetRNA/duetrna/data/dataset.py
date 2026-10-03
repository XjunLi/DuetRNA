from __future__ import annotations

import functools as fn
import hashlib
import math
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import tree
import warnings
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch.utils.data.distributed import dist

from duetrna_shared_core.chemistry import aatype9_to_aatype4, convert_na_aatype6_to_aatype9
from duetrna_shared_core.data_utils import pad_feats
from duetrna_shared_core.geometry import Rigid
from duetrna_shared_core.io import parse_processed_feats, read_processed_pickle
from duetrna.data import data_transforms


NUM_NA_RESIDUE_ATOMS = 23
FILTER_CACHE_VERSION = 2


def _dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _select_na_window(processed_feats: dict) -> dict:
    modeled_idx = processed_feats["modeled_idx"]
    min_idx = int(np.min(modeled_idx))
    max_idx = int(np.max(modeled_idx))
    del processed_feats["modeled_idx"]
    if processed_feats.get("protein_modeled_idx") is None:
        processed_feats.pop("protein_modeled_idx", None)
    if processed_feats.get("na_modeled_idx") is None:
        processed_feats.pop("na_modeled_idx", None)
    return tree.map_structure(lambda x: x[min_idx : (max_idx + 1)], processed_feats)


class PDBNABaseDataset(Dataset):
    def __init__(
        self,
        data_conf,
        is_training: bool,
        filter_eval_split: bool = False,
        inference_cfg: Optional[DictConfig] = None,
    ):
        del filter_eval_split, inference_cfg
        self._data_conf = data_conf
        self._is_training = is_training
        self._init_metadata_and_splits()

    @property
    def data_conf(self):
        return self._data_conf

    @property
    def is_training(self):
        return self._is_training

    def _init_metadata_and_splits(self):
        pdb_csv = pd.read_csv(self.data_conf.csv_path)
        pdb_csv.fillna(
            {"helix_percent": 0, "coil_percent": 0, "strand_percent": 0, "radius_gyration": 0},
            inplace=True,
        )
        filt = self.data_conf.filtering
        pdb_csv = pdb_csv[pdb_csv.modeled_na_seq_len <= filt.max_len]
        pdb_csv = pdb_csv[pdb_csv.modeled_na_seq_len >= filt.min_len]
        pdb_csv = pdb_csv[pdb_csv.quaternary_category == "homomer"]
        pdb_csv = pdb_csv[pdb_csv.num_protein_chains == 0]
        pdb_csv = pdb_csv.sort_values("modeled_na_seq_len", ascending=False)
        pdb_csv = self._filter_zero_mask_rows(pdb_csv)

        if self.is_training:
            frac = float(self.data_conf.get("short_train_fraction", 1.0))
            seed = int(self.data_conf.get("short_train_seed", self.data_conf.get("seed", 123)))
            if 0.0 < frac < 1.0 and len(pdb_csv) > 0:
                sample_n = max(1, int(round(len(pdb_csv) * frac)))
                pdb_csv = pdb_csv.sample(n=sample_n, random_state=seed, replace=False)
                pdb_csv = pdb_csv.sort_values("modeled_na_seq_len", ascending=False)
            self.csv = pdb_csv
        else:
            eval_csv = pdb_csv
            all_lengths = pdb_csv["modeled_na_seq_len"].unique()
            length_indices = (len(all_lengths) - 1) * np.linspace(0.0, 1.0, self.data_conf.num_eval_lengths)
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_csv = eval_csv[eval_csv.modeled_na_seq_len.isin(eval_lengths)]
            eval_csv = eval_csv.groupby("modeled_na_seq_len").sample(
                self.data_conf.samples_per_eval_length,
                replace=True,
                random_state=int(self.data_conf.get("eval_seed", 123)),
            )
            self.csv = eval_csv.sort_values(["modeled_na_seq_len"], ascending=False)

    def _filter_cache_path(self, pdb_csv: pd.DataFrame) -> Path:
        csv_path = Path(self.data_conf.csv_path).resolve()
        repo_root = csv_path.parent.parent if csv_path.parent.name == "metadata" else csv_path.parent
        cache_dir = repo_root / ".cache" / "duetrna_filter"
        cache_dir.mkdir(parents=True, exist_ok=True)
        frame_cfg = self.data_conf.dual_frame if "dual_frame" in self.data_conf else {}
        processed_digest = hashlib.sha256()
        for raw_path in pdb_csv["processed_path"].astype(str):
            processed_path = Path(raw_path)
            if not processed_path.is_absolute():
                processed_path = Path.cwd() / processed_path
            try:
                stat = processed_path.stat()
                fingerprint = f"{raw_path}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
            except FileNotFoundError:
                fingerprint = f"{raw_path}\0MISSING\n"
            processed_digest.update(fingerprint.encode("utf-8"))
        signature = {
            "cache_version": FILTER_CACHE_VERSION,
            "csv_path": str(csv_path),
            "csv_mtime_ns": csv_path.stat().st_mtime_ns if csv_path.exists() else 0,
            "num_rows": int(len(pdb_csv)),
            "processed_files": processed_digest.hexdigest(),
            "mode": frame_cfg.get("mode", None) if hasattr(frame_cfg, "get") else None,
            "base_origin_mode": frame_cfg.get("base_origin_mode", None) if hasattr(frame_cfg, "get") else None,
            "sugar_origin_mode": frame_cfg.get("sugar_origin_mode", None) if hasattr(frame_cfg, "get") else None,
        }
        digest = hashlib.md5(repr(signature).encode("utf-8")).hexdigest()
        return cache_dir / f"{digest}.pt"

    def _filter_zero_mask_rows(self, pdb_csv: pd.DataFrame) -> pd.DataFrame:
        cache_path = self._filter_cache_path(pdb_csv)
        if cache_path.exists():
            cache_payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            valid_paths = set(cache_payload.get("valid_paths", []))
            invalid_rows = cache_payload.get("invalid_rows", [])
            keep_rows = pdb_csv["processed_path"].isin(valid_paths)
            if invalid_rows:
                warnings.warn(
                    f"Loaded cached DuetRNA zero-mask filtering with {len(invalid_rows)} dropped samples: "
                    + ", ".join(path for path, _ in invalid_rows[:5]),
                    RuntimeWarning,
                )
            return pdb_csv.loc[keep_rows].copy()

        keep_rows = []
        dropped_rows = []
        for row in pdb_csv.itertuples(index=False):
            feats = self._process_csv_row(row.processed_path)
            valid_res_count = int(torch.sum(feats["res_mask"]).item())
            keep = valid_res_count > 0
            keep_rows.append(keep)
            if not keep:
                dropped_rows.append((row.processed_path, int(row.modeled_na_seq_len)))

        if dropped_rows:
            warnings.warn(
                f"Filtered {len(dropped_rows)} DuetRNA samples with zero valid dual-frame residues: "
                + ", ".join(path for path, _ in dropped_rows[:5]),
                RuntimeWarning,
            )
        cache_payload = {
            "valid_paths": pdb_csv.loc[keep_rows, "processed_path"].tolist(),
            "invalid_rows": dropped_rows,
        }
        # Every DDP rank constructs the dataset.  Write to a rank/process-local
        # temporary file and atomically publish it so no rank can observe a
        # partially serialized torch payload from another rank.
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=cache_path.parent,
                prefix=f".{cache_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            torch.save(cache_payload, temporary_path)
            os.replace(temporary_path, cache_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return pdb_csv.loc[keep_rows].copy()

    @fn.lru_cache(maxsize=100)
    def _process_csv_row(self, processed_file_path):
        processed_feats = parse_processed_feats(read_processed_pickle(processed_file_path))
        processed_feats["is_na_residue_mask"] = (
            processed_feats["molecule_type_encoding"][:, 1] == 1
        ) | (processed_feats["molecule_type_encoding"][:, 2] == 1)
        processed_feats = _select_na_window(processed_feats)

        chain_feats = {
            "aatype": torch.tensor(processed_feats["aatype"]).long(),
            "all_atom_positions": torch.tensor(processed_feats["atom_positions"]).double(),
            "all_atom_mask": torch.tensor(processed_feats["atom_mask"]).double(),
            "atom_deoxy": torch.tensor(processed_feats["atom_deoxy"]).bool(),
        }
        na_mask = torch.as_tensor(processed_feats["is_na_residue_mask"]).bool()
        atom23_pos = chain_feats["all_atom_positions"][na_mask][:, :NUM_NA_RESIDUE_ATOMS]
        atom23_mask = chain_feats["all_atom_mask"][na_mask][:, :NUM_NA_RESIDUE_ATOMS]
        atom_deoxy = chain_feats["atom_deoxy"][na_mask]
        aatype = chain_feats["aatype"][na_mask]
        aatype9 = convert_na_aatype6_to_aatype9(aatype.clone(), deoxy_offset_mask=atom_deoxy)
        aatype4 = aatype9_to_aatype4(aatype9)

        na_chain_feats = {
            "aatype": aatype.clone(),
            "all_atom_positions": atom23_pos.clone(),
            "all_atom_mask": atom23_mask.clone(),
            "atom_deoxy": atom_deoxy.clone(),
            "atom23_gt_positions": atom23_pos.clone(),
        }
        dual_frame_cfg = self.data_conf.dual_frame if "dual_frame" in self.data_conf else None
        na_chain_feats = data_transforms.prepare_oracle_dual_frame_batch(
            na_chain_feats,
            frame_cfg=dual_frame_cfg,
        )

        base_rigids = Rigid.from_tensor_4x4(na_chain_feats["base_rigidgroups_gt_frames"])
        sugar_rigids = Rigid.from_tensor_4x4(na_chain_feats["sugar_rigidgroups_gt_frames"])
        rel_rigids = Rigid.from_tensor_4x4(na_chain_feats["base_to_sugar_gt_frames"])

        final_feats = {
            "aatype4": aatype4,
            "aatype9": aatype9,
            "torsion_angles_sin_cos": na_chain_feats["torsion_angles_sin_cos"],
            "base_rotmats_1": base_rigids.get_rots().get_rot_mats(),
            "base_trans_1": base_rigids.get_trans(),
            "rel_rotmats_1": rel_rigids.get_rots().get_rot_mats(),
            "rel_trans_1": rel_rigids.get_trans(),
            "sugar_rotmats_1": sugar_rigids.get_rots().get_rot_mats(),
            "sugar_trans_1": sugar_rigids.get_trans(),
            "rotmats_1": base_rigids.get_rots().get_rot_mats(),
            "trans_1": base_rigids.get_trans(),
            "res_mask": na_chain_feats["base_to_sugar_gt_exists"].int(),
            "is_na_residue_mask": torch.ones_like(aatype9, dtype=torch.int32),
            "atom23_gt_positions": atom23_pos,
            "atom23_gt_mask": atom23_mask,
            "base_exists": na_chain_feats["base_rigidgroups_gt_exists"],
            "sugar_exists": na_chain_feats["sugar_rigidgroups_gt_exists"],
            "rel_exists": na_chain_feats["base_to_sugar_gt_exists"],
        }
        return final_feats

    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        final_feats = self._process_csv_row(row["processed_path"])
        if int(torch.sum(final_feats["res_mask"]).item()) <= 0:
            raise ValueError(
                f"DuetRNA dataset produced a zero-mask sample after filtering: {row['processed_path']}"
            )
        final_feats = pad_feats(final_feats, row["modeled_na_seq_len"])
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x),
            final_feats,
        )
        final_feats = tree.map_structure(
            lambda x: x.float() if torch.is_tensor(x) and x.dtype == torch.float64 else x,
            final_feats,
        )
        return final_feats

    def __len__(self):
        return len(self.csv)


class LengthDataset(torch.utils.data.Dataset):
    def __init__(self, samples_cfg):
        all_sample_lengths = range(
            samples_cfg.min_length,
            samples_cfg.max_length + 1,
            samples_cfg.length_step,
        )
        if samples_cfg.length_subset is not None:
            all_sample_lengths = [int(x) for x in samples_cfg.length_subset]
        all_sample_ids = []
        for length in all_sample_lengths:
            for sample_id in range(samples_cfg.samples_per_length):
                all_sample_ids.append((length, sample_id))
        self._all_sample_ids = all_sample_ids

    def __len__(self):
        return len(self._all_sample_ids)

    def __getitem__(self, idx):
        num_res, sample_id = self._all_sample_ids[idx]
        return {"num_res": num_res, "sample_id": sample_id}


class RNALengthBatcher:
    def __init__(
        self,
        sampler_cfg,
        metadata_csv,
        seed=123,
        shuffle=True,
        num_replicas=None,
        rank=None,
    ):
        self._sampler_cfg = sampler_cfg
        self._data_csv = metadata_csv.copy()
        self._data_csv["index"] = list(range(len(self._data_csv)))
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size = sampler_cfg.max_batch_size
        self.length_bucket_size = max(1, int(sampler_cfg.get("length_bucket_size", 1)))
        self.epoch_size_mode = str(sampler_cfg.get("epoch_size_mode", "legacy_num_samples")).lower()
        self._length_by_index = self._data_csv["modeled_na_seq_len"].to_numpy(dtype=np.int64)
        if num_replicas is None or rank is None:
            if _dist_available_and_initialized():
                auto_num_replicas = dist.get_world_size()
                auto_rank = dist.get_rank()
            else:
                auto_num_replicas = 1
                auto_rank = 0
        if num_replicas is None:
            self.num_replicas = auto_num_replicas
        else:
            self.num_replicas = num_replicas
        if rank is None:
            self.rank = auto_rank
        else:
            self.rank = rank
        # Lightning calls ``set_epoch`` through ``dataloader.sampler`` or
        # ``dataloader.batch_sampler.sampler``. Expose this custom batch
        # sampler through the latter route so checkpoint resumes use the
        # restored epoch instead of silently restarting at epoch zero.
        self.sampler = self
        self._num_batches = math.ceil(len(self._data_csv) / self.num_replicas)

    def _epoch_indices(self):
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self._data_csv), generator=rng).tolist()
        else:
            indices = list(range(len(self._data_csv)))
        return indices, rng

    def _replica_csv_from_indices(self, indices, rank):
        if len(self._data_csv) > self.num_replicas:
            return self._data_csv.iloc[indices[rank :: self.num_replicas]]
        return self._data_csv

    def _length_bucket_id(self, seq_len):
        seq_len = int(seq_len)
        if self.length_bucket_size <= 1:
            return seq_len
        return (seq_len - 1) // self.length_bucket_size

    def _bucketed_groups(self, replica_csv):
        bucket_ids = replica_csv["modeled_na_seq_len"].map(self._length_bucket_id)
        bucketed_csv = replica_csv.assign(length_bucket=bucket_ids)
        return bucketed_csv.groupby("length_bucket", sort=True)

    def _bucket_max_batch_size(self, bucket_df):
        seq_len = int(bucket_df["modeled_na_seq_len"].max())
        linear_effect = bool(self._sampler_cfg.get("linear_effect", False))
        if linear_effect:
            return min(self.max_batch_size, self._sampler_cfg.max_num_res_squared // seq_len + 1)
        return min(self.max_batch_size, self._sampler_cfg.max_num_res_squared // seq_len**2 + 1)

    def _build_sample_order(self, replica_csv, rng, *, shuffle_batches=True):
        sample_order = []
        for _, bucket_df in self._bucketed_groups(replica_csv):
            bucket_df = bucket_df.sort_values("modeled_na_seq_len", ascending=False, kind="mergesort")
            max_batch_size = self._bucket_max_batch_size(bucket_df)
            num_batches = math.ceil(len(bucket_df) / max_batch_size)
            for i in range(num_batches):
                batch_df = bucket_df.iloc[i * max_batch_size : (i + 1) * max_batch_size]
                sample_order.append(batch_df["index"].tolist())
        if sample_order and shuffle_batches:
            order = torch.randperm(len(sample_order), generator=rng).tolist()
            sample_order = [sample_order[i] for i in order]
        return sample_order

    def _batch_alignment_key(self, batch_indices):
        """Approximate padded pairwise work for cross-rank step alignment."""
        lengths = self._length_by_index[np.asarray(batch_indices, dtype=np.int64)]
        padded_len = int(lengths.max())
        return len(batch_indices) * padded_len**2, padded_len, len(batch_indices)

    def _aligned_replica_batches(self):
        """Build equal-count batches whose work quantiles line up across ranks.

        DDP waits for the slowest rank on every optimizer step.  Independently
        shuffling length buckets gives every rank similar epoch-level work but
        pairs a long batch on one rank with short batches on the others.  Build
        all deterministic rank-local batch lists, pad them to the same count,
        sort each list by its padded pairwise cost, and finally apply one shared
        epoch permutation.  Samples remain rank-disjoint and epoch-shuffled,
        while expensive batches execute at the same DDP step.
        """
        indices, _ = self._epoch_indices()
        target_batches = self._replica_batch_target()
        batches_by_rank = []

        for replica_rank in range(self.num_replicas):
            replica_csv = self._replica_csv_from_indices(indices, replica_rank)
            replica_batches = self._build_sample_order(
                replica_csv,
                rng=None,
                shuffle_batches=False,
            )
            if not replica_batches:
                raise ValueError(f"Replica {replica_rank} produced no batches.")

            # Preserve the legacy repeat-to-target behavior, but choose repeated
            # batches with a deterministic rank-local permutation before cost
            # alignment so the same sample is not always duplicated.
            padded_batches = list(replica_batches)
            pad_rng = torch.Generator()
            pad_rng.manual_seed(self.seed + self.epoch + 1_000_003 * replica_rank)
            while len(padded_batches) < target_batches:
                pad_order = torch.randperm(len(replica_batches), generator=pad_rng).tolist()
                padded_batches.extend(replica_batches[i] for i in pad_order)
            padded_batches = padded_batches[:target_batches]
            padded_batches.sort(key=self._batch_alignment_key, reverse=True)
            batches_by_rank.append(padded_batches)

        shared_rng = torch.Generator()
        shared_rng.manual_seed(self.seed + self.epoch + 2_000_003)
        shared_order = torch.randperm(target_batches, generator=shared_rng).tolist()
        return [
            [replica_batches[i] for i in shared_order]
            for replica_batches in batches_by_rank
        ]

    def _count_batches_for_csv(self, replica_csv):
        total_batches = 0
        for _, bucket_df in self._bucketed_groups(replica_csv):
            max_batch_size = self._bucket_max_batch_size(bucket_df)
            total_batches += math.ceil(len(bucket_df) / max_batch_size)
        return total_batches

    def _replica_epoch_batches(self):
        indices, rng = self._epoch_indices()
        replica_csv = self._replica_csv_from_indices(indices, self.rank)
        return self._build_sample_order(replica_csv, rng)

    def _replica_batch_target(self):
        indices, _ = self._epoch_indices()
        return max(
            self._count_batches_for_csv(self._replica_csv_from_indices(indices, replica_rank))
            for replica_rank in range(self.num_replicas)
        )

    def _create_batches(self):
        if self.epoch_size_mode == "replica_batches":
            batches_by_rank = self._aligned_replica_batches()
            self.sample_order = batches_by_rank[self.rank]
            self._num_batches = len(self.sample_order)
            return

        all_batches = []
        num_augments = -1
        while len(all_batches) < self._num_batches:
            all_batches.extend(self._replica_epoch_batches())
            num_augments += 1
            if num_augments > 1000:
                raise ValueError("Exceeded number of augmentations.")
        self.sample_order = all_batches[: self._num_batches]

    def __iter__(self):
        # Keep this a generator: Lightning creates the DataLoader iterator
        # before it calls ``set_epoch`` on the first/resumed epoch.  Deferring
        # batch materialization until the first ``next`` lets the restored
        # epoch take effect.
        self._create_batches()
        self.epoch += 1
        yield from self.sample_order

    def set_epoch(self, epoch):
        """Standard distributed-sampler hook, including checkpoint resumes."""
        self.epoch = int(epoch)
        if hasattr(self, "sample_order"):
            del self.sample_order

    def __len__(self):
        if hasattr(self, "sample_order"):
            return len(self.sample_order)
        if self.epoch_size_mode == "replica_batches":
            # Lightning queries the DataLoader length before the first call to
            # ``__iter__``. Report the same equalized per-rank batch count that
            # ``_create_batches`` will materialize; the legacy sample-count
            # estimate makes epoch/checkpoint accounting incorrect.
            return self._replica_batch_target()
        return self._num_batches


__all__ = ["LengthDataset", "PDBNABaseDataset", "RNALengthBatcher"]
