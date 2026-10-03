from __future__ import annotations

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler, dist

from duetrna_shared_core.data_utils import pad_feats
from duetrna.data.dataset import LengthDataset, PDBNABaseDataset, RNALengthBatcher


def _dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _pad_collate(batch):
    max_len = max(int(sample["res_mask"].shape[0]) for sample in batch)
    collated = default_collate([pad_feats(sample, max_len, use_torch=True) for sample in batch])
    collated["is_na_residue_mask"] = torch.ones_like(
        collated["res_mask"],
        dtype=collated["is_na_residue_mask"].dtype,
    )
    return collated


class PDBNABaseDataModule(LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_cfg = data_cfg
        self.data_train = None
        self.data_val = None

    def setup(self, stage=None):
        del stage
        self.data_train = PDBNABaseDataset(self.data_cfg, is_training=True)
        self.data_val = PDBNABaseDataset(self.data_cfg, is_training=False)

    def train_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.data_cfg.num_workers
        lb = RNALengthBatcher(
            self.data_cfg,
            self.data_train.csv,
            seed=int(self.data_cfg.get("seed", 123)),
            rank=rank,
            num_replicas=num_replicas,
        )
        return DataLoader(
            self.data_train,
            batch_sampler=lb,
            collate_fn=_pad_collate,
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.data_cfg.prefetch_factor,
            pin_memory=bool(getattr(self.data_cfg, "pin_memory", False)),
            persistent_workers=True if num_workers > 0 else False,
        )

    def val_dataloader(self):
        num_workers = int(getattr(self.data_cfg, "val_num_workers", min(2, int(self.data_cfg.num_workers))))
        if _dist_available_and_initialized():
            val_samp = DistributedSampler(self.data_val, shuffle=False)
        else:
            val_samp = None
        return DataLoader(
            self.data_val,
            sampler=val_samp,
            shuffle=False,
            batch_size=int(getattr(self.data_cfg, "eval_batch_size", 1)),
            collate_fn=_pad_collate,
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else int(getattr(self.data_cfg, "val_prefetch_factor", 2)),
            pin_memory=bool(getattr(self.data_cfg, "pin_memory", False)),
            persistent_workers=True if num_workers > 0 else False,
        )


__all__ = ["LengthDataset", "PDBNABaseDataModule", "RNALengthBatcher"]
