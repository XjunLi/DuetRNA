from __future__ import annotations

import math
import os
import time

import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger

from duetrna_shared_core import data_utils as du
from duetrna_shared_core.metrics import calc_rna_c4_c4_metrics, compute_rmsd
from duetrna_metrics.metrics import (
    chi_angle_loss,
    inter_residue_clash_loss,
)
from duetrna_training.diffusion import so3_utils
from duetrna_training.diffusion.interpolant import (
    Interpolant,
    _centered_gaussian,
    _rots_diffuse_mask,
    _trans_diffuse_mask,
    _uniform_so3,
)
from duetrna_shared_core.chemistry import AATYPE4_TO_RESTYPE, COMMON_ATOM_ORDER, aatype4_to_sequence, compact_atom_index
from duetrna.runtime.analysis.pdb_io import write_atom23_to_pdb
from duetrna.runtime.models.flow_model import DuetRNAFlowModel
from duetrna.runtime.models.reconstruction import (
    base_atom_class_mask,
    reconstruct_atom23,
    reconstruct_soft_base_atoms,
    reconstruct_atom23_trajectory,
)


O3_INDEX = COMMON_ATOM_ORDER["O3'"]
P_INDEX = COMMON_ATOM_ORDER["P"]
C1_INDEX = COMMON_ATOM_ORDER["C1'"]


def _is_purine(restype: str) -> bool:
    return restype in {"A", "G", "DA", "DG"}


def _atom_loss_profile_mask(aatype4: torch.Tensor, profile: str, *, dtype: torch.dtype) -> torch.Tensor:
    profile_key = str(profile).lower()
    device = aatype4.device
    num_classes = len(AATYPE4_TO_RESTYPE)
    if profile_key == "full23":
        table = torch.ones(num_classes, 23, device=device, dtype=dtype)
    elif profile_key == "origin7_plus_base_plane2":
        table = torch.zeros(num_classes, 23, device=device, dtype=dtype)
        common_indices = [
            COMMON_ATOM_ORDER["C3'"],
            COMMON_ATOM_ORDER["C4'"],
            COMMON_ATOM_ORDER["O4'"],
            COMMON_ATOM_ORDER["C1'"],
            COMMON_ATOM_ORDER["C5'"],
            COMMON_ATOM_ORDER["O3'"],
            COMMON_ATOM_ORDER["P"],
            COMMON_ATOM_ORDER["OP1"],
            COMMON_ATOM_ORDER["OP2"],
            11,  # Glycosidic N: N9 for purines, N1 for pyrimidines.
        ]
        for class_idx, restype in AATYPE4_TO_RESTYPE.items():
            base_plane_atoms = ("C4", "C8") if _is_purine(restype) else ("C4", "C2")
            indices = common_indices + [compact_atom_index(restype, name) for name in base_plane_atoms]
            table[int(class_idx), indices] = 1.0
    else:
        raise ValueError(
            "Unsupported DuetRNA atom_loss_profile "
            f"`{profile}`. Expected `origin7_plus_base_plane2` or `full23`."
        )
    return table.index_select(0, aatype4.long().reshape(-1)).reshape(*aatype4.shape, 23)


def _base_geometry_index_lookup() -> tuple[torch.Tensor, torch.Tensor]:
    glycosidic = torch.zeros(len(AATYPE4_TO_RESTYPE), dtype=torch.long)
    plane = torch.zeros(len(AATYPE4_TO_RESTYPE), 2, dtype=torch.long)
    for aatype, restype in AATYPE4_TO_RESTYPE.items():
        if _is_purine(restype):
            anchor_name, plane_names = "N9", ("C4", "C8")
        else:
            anchor_name, plane_names = "N1", ("C4", "C2")
        glycosidic[int(aatype)] = compact_atom_index(restype, anchor_name)
        plane[int(aatype), 0] = compact_atom_index(restype, plane_names[0])
        plane[int(aatype), 1] = compact_atom_index(restype, plane_names[1])
    return glycosidic, plane


BASE_GLYCOSIDIC_INDEX_LOOKUP, BASE_PLANE_INDEX_LOOKUP = _base_geometry_index_lookup()


def _gather_atom_coords(atom23: torch.Tensor, atom_indices: torch.Tensor) -> torch.Tensor:
    gather_indices = atom_indices[..., None, None].expand(*atom_indices.shape, 1, 3)
    return torch.gather(atom23, dim=2, index=gather_indices).squeeze(2)


def _gather_atom_mask(atom23_mask: torch.Tensor, atom_indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(atom23_mask, dim=2, index=atom_indices[..., None]).squeeze(2)


def _scatter_sample_losses(reference: torch.Tensor, active_idx: torch.Tensor, active_values: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(reference)
    if active_idx.numel() > 0:
        out.index_copy_(0, active_idx, active_values)
    return out


def _derive_rel_from_absolute(base_trans, base_rotmats, sugar_trans, sugar_rotmats):
    base_rigids = du.create_rigid(base_rotmats, base_trans)
    sugar_rigids = du.create_rigid(sugar_rotmats, sugar_trans)
    rel_rigids = base_rigids.invert().compose(sugar_rigids)
    return rel_rigids.get_trans(), rel_rigids.get_rots().get_rot_mats()


def _runtime_step_seed(
    base_seed: int,
    global_rank: int,
    global_step: int,
    microbatch_index: int = 0,
) -> int:
    """Return a deterministic, rank/microbatch-disjoint training RNG seed."""
    return (
        int(base_seed)
        + 1_000_003 * int(global_rank)
        + 10_000_019 * int(global_step)
        + 1_000_000_007 * int(microbatch_index)
    ) % (2**31 - 1)


def _shared_step_uniform(
    base_seed: int,
    global_step: int,
    stream: int = 0,
    *,
    microbatch_index: int = 0,
) -> float:
    """Return a deterministic rank-independent uniform value without GPU sync."""
    mask = (1 << 64) - 1
    value = (
        (int(base_seed) & mask)
        ^ (((int(global_step) + 1) * 0x9E3779B97F4A7C15) & mask)
        ^ (((int(stream) + 1) * 0xD1B54A32D192ED03) & mask)
    )
    if microbatch_index:
        value ^= ((int(microbatch_index) * 0x94D049BB133111EB) & mask)
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return float(value >> 11) * (1.0 / (1 << 53))


def _shared_step_bernoulli(
    base_seed: int,
    global_step: int,
    probability: float,
    *,
    stream: int = 0,
    microbatch_index: int = 0,
) -> bool:
    """Make one reproducible Bernoulli decision shared by every DDP rank."""
    probability = float(probability)
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    return (
        _shared_step_uniform(
            base_seed,
            global_step,
            stream,
            microbatch_index=microbatch_index,
        )
        < probability
    )


def _inference_sample_seed(base_seed: int, sample_length: int, sample_id: int) -> int:
    """Return a deterministic per-sample seed independent of DDP assignment."""
    return (
        int(base_seed)
        + 1_000_003 * int(sample_length)
        + int(sample_id)
    ) % (2**31 - 1)


class DuetRNAInterpolant(Interpolant):
    def _sample_absolute_trans_prior(self, trans_1, res_mask):
        trans_0 = _centered_gaussian(*res_mask.shape, self._device) * du.NM_TO_ANG_SCALE
        trans_0 = _trans_diffuse_mask(trans_0, trans_1, res_mask)
        return trans_0 * res_mask[..., None]

    def _sample_absolute_rot_prior(self, rotmats_1, res_mask):
        num_batch, num_res = res_mask.shape
        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        return _rots_diffuse_mask(rotmats_0, rotmats_1, res_mask)

    def _interpolate_absolute_trans(self, trans_0, trans_1, t, res_mask):
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
        return trans_t * res_mask[..., None]

    def _interpolate_absolute_rotmats(self, rotmats_0, rotmats_1, t, res_mask):
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        return _rots_diffuse_mask(rotmats_t, rotmats_1, res_mask)

    def corrupt_batch(self, batch):
        noisy_batch = dict(batch)
        res_mask = batch["res_mask"]
        num_batch, _ = res_mask.shape
        t = self.sample_t(num_batch)[:, None]
        noisy_batch["t"] = t
        base_trans_0 = self._sample_absolute_trans_prior(batch["base_trans_1"], res_mask)
        base_rotmats_0 = self._sample_absolute_rot_prior(batch["base_rotmats_1"], res_mask)
        sugar_trans_0 = self._sample_absolute_trans_prior(batch["sugar_trans_1"], res_mask)
        sugar_rotmats_0 = self._sample_absolute_rot_prior(batch["sugar_rotmats_1"], res_mask)
        noisy_batch["base_trans_t"] = self._interpolate_absolute_trans(
            base_trans_0,
            batch["base_trans_1"],
            t,
            res_mask,
        )
        noisy_batch["base_rotmats_t"] = self._interpolate_absolute_rotmats(
            base_rotmats_0,
            batch["base_rotmats_1"],
            t,
            res_mask,
        )
        noisy_batch["sugar_trans_t"] = self._interpolate_absolute_trans(
            sugar_trans_0,
            batch["sugar_trans_1"],
            t,
            res_mask,
        )
        noisy_batch["sugar_rotmats_t"] = self._interpolate_absolute_rotmats(
            sugar_rotmats_0,
            batch["sugar_rotmats_1"],
            t,
            res_mask,
        )
        noisy_batch["rel_trans_t"], noisy_batch["rel_rotmats_t"] = _derive_rel_from_absolute(
            noisy_batch["base_trans_t"],
            noisy_batch["base_rotmats_t"],
            noisy_batch["sugar_trans_t"],
            noisy_batch["sugar_rotmats_t"],
        )
        noisy_batch["trans_t"] = noisy_batch["base_trans_t"]
        noisy_batch["rotmats_t"] = noisy_batch["base_rotmats_t"]
        return noisy_batch

    def sample_joint(self, num_batch, num_res, model):
        res_mask = torch.ones(num_batch, num_res, device=self._device)
        batch = {"res_mask": res_mask}
        base_trans_0 = _centered_gaussian(num_batch, num_res, self._device) * du.NM_TO_ANG_SCALE
        base_rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        sugar_trans_0 = _centered_gaussian(num_batch, num_res, self._device) * du.NM_TO_ANG_SCALE
        sugar_rotmats_0 = _uniform_so3(num_batch, num_res, self._device)

        ts = torch.linspace(self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps, device=self._device)
        t_1 = ts[0]
        base_traj = [(base_trans_0, base_rotmats_0)]
        sugar_traj = [(sugar_trans_0, sugar_rotmats_0)]
        clean_base_traj = []
        clean_rel_traj = []
        pred_torsions = None
        pred_base_logits = None

        for t_2 in ts[1:]:
            base_trans_t_1, base_rotmats_t_1 = base_traj[-1]
            sugar_trans_t_1, sugar_rotmats_t_1 = sugar_traj[-1]
            batch["base_trans_t"] = base_trans_t_1
            batch["base_rotmats_t"] = base_rotmats_t_1
            batch["sugar_trans_t"] = sugar_trans_t_1
            batch["sugar_rotmats_t"] = sugar_rotmats_t_1
            batch["rel_trans_t"], batch["rel_rotmats_t"] = _derive_rel_from_absolute(
                base_trans_t_1,
                base_rotmats_t_1,
                sugar_trans_t_1,
                sugar_rotmats_t_1,
            )
            batch["t"] = torch.ones((num_batch, 1), device=self._device) * t_1
            with torch.no_grad():
                model_out = model(batch)

            pred_base_trans = model_out["pred_base_trans"]
            pred_base_rotmats = model_out["pred_base_rotmats"]
            pred_sugar_trans = model_out["pred_sugar_trans"]
            pred_sugar_rotmats = model_out["pred_sugar_rotmats"]
            pred_rel_rigids = du.create_rigid(pred_base_rotmats, pred_base_trans).invert().compose(
                du.create_rigid(pred_sugar_rotmats, pred_sugar_trans)
            )
            pred_rel_trans = pred_rel_rigids.get_trans()
            pred_rel_rotmats = pred_rel_rigids.get_rots().get_rot_mats()
            pred_torsions = model_out["pred_torsions"]
            pred_base_logits = model_out["pred_base_logits"]
            clean_base_traj.append((pred_base_trans.detach().cpu(), pred_base_rotmats.detach().cpu()))
            clean_rel_traj.append((pred_rel_trans.detach().cpu(), pred_rel_rotmats.detach().cpu()))
            if self._cfg.self_condition:
                batch["base_trans_sc"] = pred_base_trans
                batch["sugar_trans_sc"] = pred_sugar_trans

            d_t = t_2 - t_1
            next_base_trans = self._trans_euler_step(d_t, t_1, pred_base_trans, base_trans_t_1)
            next_base_rots = self._rots_euler_step(d_t, t_1, pred_base_rotmats, base_rotmats_t_1)
            next_sugar_trans = self._trans_euler_step(d_t, t_1, pred_sugar_trans, sugar_trans_t_1)
            next_sugar_rots = self._rots_euler_step(d_t, t_1, pred_sugar_rotmats, sugar_rotmats_t_1)
            base_traj.append((next_base_trans, next_base_rots))
            sugar_traj.append((next_sugar_trans, next_sugar_rots))
            t_1 = t_2

        base_trans_t_1, base_rotmats_t_1 = base_traj[-1]
        sugar_trans_t_1, sugar_rotmats_t_1 = sugar_traj[-1]
        batch["base_trans_t"] = base_trans_t_1
        batch["base_rotmats_t"] = base_rotmats_t_1
        batch["sugar_trans_t"] = sugar_trans_t_1
        batch["sugar_rotmats_t"] = sugar_rotmats_t_1
        batch["rel_trans_t"], batch["rel_rotmats_t"] = _derive_rel_from_absolute(
            base_trans_t_1,
            base_rotmats_t_1,
            sugar_trans_t_1,
            sugar_rotmats_t_1,
        )
        batch["t"] = torch.ones((num_batch, 1), device=self._device) * ts[-1]
        with torch.no_grad():
            model_out = model(batch)

        pred_base_trans = model_out["pred_base_trans"]
        pred_base_rotmats = model_out["pred_base_rotmats"]
        pred_sugar_trans = model_out["pred_sugar_trans"]
        pred_sugar_rotmats = model_out["pred_sugar_rotmats"]
        pred_rel_rigids = du.create_rigid(pred_base_rotmats, pred_base_trans).invert().compose(
            du.create_rigid(pred_sugar_rotmats, pred_sugar_trans)
        )
        pred_rel_trans = pred_rel_rigids.get_trans()
        pred_rel_rotmats = pred_rel_rigids.get_rots().get_rot_mats()
        pred_torsions = model_out["pred_torsions"]
        pred_base_logits = model_out["pred_base_logits"]
        clean_base_traj.append((pred_base_trans.detach().cpu(), pred_base_rotmats.detach().cpu()))
        clean_rel_traj.append((pred_rel_trans.detach().cpu(), pred_rel_rotmats.detach().cpu()))
        base_traj.append((pred_base_trans, pred_base_rotmats))
        sugar_traj.append((pred_sugar_trans, pred_sugar_rotmats))
        rel_traj = [
            _derive_rel_from_absolute(base_t, base_r, sugar_t, sugar_r)
            for (base_t, base_r), (sugar_t, sugar_r) in zip(base_traj, sugar_traj)
        ]

        pred_aatype4 = torch.argmax(pred_base_logits, dim=-1)
        atom23_traj = reconstruct_atom23_trajectory(
            base_traj,
            sugar_traj,
            pred_torsions,
            torch.ones(num_batch, num_res, device=self._device, dtype=torch.bool),
            pred_aatype4,
        )
        return atom23_traj, clean_base_traj, clean_rel_traj, pred_aatype4


class DuetRNAFlowModule(LightningModule):
    def __init__(self, cfg, folding_cfg=None):
        del folding_cfg
        super().__init__()
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._interpolant_cfg = cfg.interpolant
        self.model = DuetRNAFlowModel(cfg.model)
        chi_weight = float(getattr(self._exp_cfg.training, "chi_loss_weight", 0.25))
        chi_enabled = bool(
            getattr(self._exp_cfg.training, "enable_chi_loss", chi_weight > 0.0)
        )
        if not chi_enabled:
            # The default training recipe explicitly disables the chi objective.
            # Retain the state-dict keys, but do not carry an untrained head as
            # trainable parameters merely because DDP can tolerate unused ones.
            self.model.chi_head.requires_grad_(False)
        self.interpolant = DuetRNAInterpolant(cfg.interpolant)
        self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(self._sample_write_dir, exist_ok=True)
        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self._saved_cfg = cfg
        self._c4_idx = COMMON_ATOM_ORDER["C4'"]
        ema_cfg = getattr(self._exp_cfg, "ema", None)
        self._ema_enabled = bool(getattr(ema_cfg, "enable", False))
        self._ema_decay = float(getattr(ema_cfg, "decay", 0.9999))
        self._ema_warmup_steps = int(getattr(ema_cfg, "warmup_steps", 0))
        self._ema_state: dict[str, torch.Tensor] = {}
        self._ema_backup: dict[str, torch.Tensor] | None = None
        self._ema_num_updates = 0

    def on_train_start(self):
        self._epoch_start_time = time.time()
        self._validation_start_time = None
        self._prev_train_batch_end_perf = None
        self._pending_train_gap_ms = None
        if self._ema_enabled:
            self._ensure_ema_state()

    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        sync_dist = self.trainer is not None and self.trainer.world_size > 1
        self.log("train/epoch_time_minutes", epoch_time, on_step=False, on_epoch=True, prog_bar=False, sync_dist=sync_dist)
        num_batches = getattr(self.trainer, "num_training_batches", None)
        if num_batches is not None:
            self.log("train/epoch_num_batches", float(num_batches), on_step=False, on_epoch=True, prog_bar=False)
        self._epoch_start_time = time.time()

    def on_validation_epoch_start(self):
        self._validation_start_time = time.time()
        self._swap_to_ema_weights()

    def on_train_batch_start(self, batch, batch_idx):
        del batch, batch_idx
        if not self._timing_enabled():
            self._pending_train_gap_ms = None
            return
        now = time.perf_counter()
        if self._prev_train_batch_end_perf is None:
            self._pending_train_gap_ms = None
        else:
            self._pending_train_gap_ms = 1000.0 * (now - self._prev_train_batch_end_perf)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        del outputs, batch_idx
        self._update_ema()
        if not self._timing_enabled():
            return
        self._sync_timing_device(batch["res_mask"].device)
        self._prev_train_batch_end_perf = time.perf_counter()

    def _ema_parameters(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and torch.is_floating_point(param):
                yield name, param

    def _ensure_ema_state(self):
        if self._ema_state:
            for name, param in self._ema_parameters():
                if name in self._ema_state:
                    self._ema_state[name] = self._ema_state[name].to(device=param.device, dtype=param.dtype)
            return
        self._ema_state = {
            name: param.detach().clone()
            for name, param in self._ema_parameters()
        }

    def _current_ema_decay(self) -> float:
        if self._ema_warmup_steps <= 0:
            return self._ema_decay
        ramp = min(1.0, float(max(self._ema_num_updates, 1)) / float(self._ema_warmup_steps))
        return self._ema_decay * ramp

    def _update_ema(self):
        if not self._ema_enabled or not self.training:
            return
        self._ensure_ema_state()
        decay = self._current_ema_decay()
        with torch.no_grad():
            for name, param in self._ema_parameters():
                ema_param = self._ema_state[name].to(device=param.device, dtype=param.dtype)
                ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
                self._ema_state[name] = ema_param
        self._ema_num_updates += 1

    def _swap_to_ema_weights(self):
        if not self._ema_enabled or not self._ema_state or self._ema_backup is not None:
            return
        self._ema_backup = {}
        with torch.no_grad():
            for name, param in self._ema_parameters():
                ema_param = self._ema_state.get(name)
                if ema_param is None:
                    continue
                self._ema_backup[name] = param.detach().clone()
                param.copy_(ema_param.to(device=param.device, dtype=param.dtype))

    def _restore_training_weights(self):
        if self._ema_backup is None:
            return
        with torch.no_grad():
            for name, param in self._ema_parameters():
                backup = self._ema_backup.get(name)
                if backup is not None:
                    param.copy_(backup.to(device=param.device, dtype=param.dtype))
        self._ema_backup = None

    def on_save_checkpoint(self, checkpoint):
        if self._ema_enabled and self._ema_state:
            checkpoint["ema_state_dict"] = {
                name: value.detach().cpu()
                for name, value in self._ema_state.items()
            }
            checkpoint["ema_num_updates"] = int(self._ema_num_updates)

    def on_load_checkpoint(self, checkpoint):
        ema_state = checkpoint.get("ema_state_dict")
        if ema_state:
            self._ema_state = {
                name: value.detach().clone()
                for name, value in ema_state.items()
            }
            self._ema_num_updates = int(checkpoint.get("ema_num_updates", 0))

    def _frame_losses(self, noisy_batch, pred_trans, pred_rotmats, gt_trans, gt_rotmats, trans_key_prefix: str):
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch["res_mask"]
        t = noisy_batch["t"]
        norm_scale = 1 - torch.min(t[..., None], torch.tensor(training_cfg.t_normalize_clip, device=t.device))
        rotmats_t = noisy_batch[f"{trans_key_prefix}_rotmats_t"]
        gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, gt_rotmats.type(torch.float32))
        pred_rot_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats)

        trans_weight = float(
            getattr(training_cfg, f"{trans_key_prefix}_translation_loss_weight", training_cfg.translation_loss_weight)
        )
        rot_weight = float(
            getattr(training_cfg, f"{trans_key_prefix}_rotation_loss_weight", training_cfg.rotation_loss_weights)
        )

        trans_error = (gt_trans - pred_trans) / norm_scale * training_cfg.trans_scale
        trans_loss = trans_weight * torch.sum(
            trans_error**2 * loss_mask[..., None],
            dim=(-1, -2),
        ) / (torch.sum(loss_mask, dim=-1) * 3).clamp(min=1.0)
        rot_error = (gt_rot_vf - pred_rot_vf) / norm_scale
        rot_loss = rot_weight * torch.sum(
            rot_error**2 * loss_mask[..., None],
            dim=(-1, -2),
        ) / (torch.sum(loss_mask, dim=-1) * 3).clamp(min=1.0)
        return trans_loss, rot_loss

    def _rel_terminal_losses(self, pred_rel_trans, pred_rel_rotmats, gt_rel_trans, gt_rel_rotmats, loss_mask):
        training_cfg = self._exp_cfg.training
        trans_weight = float(getattr(training_cfg, "rel_translation_loss_weight", training_cfg.translation_loss_weight))
        rot_weight = float(getattr(training_cfg, "rel_rotation_loss_weight", training_cfg.rotation_loss_weights))

        rel_trans_error = (pred_rel_trans - gt_rel_trans) * training_cfg.trans_scale
        rel_trans_loss = trans_weight * torch.sum(
            rel_trans_error**2 * loss_mask[..., None],
            dim=(-1, -2),
        ) / (torch.sum(loss_mask, dim=-1) * 3).clamp(min=1.0)

        rel_rot_error = so3_utils.calc_rot_vf(pred_rel_rotmats, gt_rel_rotmats.type(torch.float32))
        rel_rot_loss = rot_weight * torch.sum(
            rel_rot_error**2 * loss_mask[..., None],
            dim=(-1, -2),
        ) / (torch.sum(loss_mask, dim=-1) * 3).clamp(min=1.0)
        return rel_trans_loss, rel_rot_loss

    def _chain_loss(self, pred_atom23, gt_atom23, atom23_mask, loss_mask):
        edge_mask = (
            atom23_mask[:, :-1, O3_INDEX]
            * atom23_mask[:, 1:, P_INDEX]
            * loss_mask[:, :-1]
            * loss_mask[:, 1:]
        )
        pred_dist = torch.linalg.norm(
            pred_atom23[:, :-1, O3_INDEX] - pred_atom23[:, 1:, P_INDEX],
            dim=-1,
        )
        gt_dist = torch.linalg.norm(
            gt_atom23[:, :-1, O3_INDEX] - gt_atom23[:, 1:, P_INDEX],
            dim=-1,
        )
        return torch.sum(((pred_dist - gt_dist) ** 2) * edge_mask, dim=-1) / edge_mask.sum(dim=-1).clamp(min=1.0)

    def _local_geometry_losses(self, pred_atom23, gt_atom23, atom23_mask, loss_mask, aatype4):
        glyco_lookup = BASE_GLYCOSIDIC_INDEX_LOOKUP.to(device=aatype4.device)
        plane_lookup = BASE_PLANE_INDEX_LOOKUP.to(device=aatype4.device)
        glyco_idx = glyco_lookup.index_select(0, aatype4.long().reshape(-1)).reshape(*aatype4.shape)
        plane_idx = plane_lookup.index_select(0, aatype4.long().reshape(-1)).reshape(*aatype4.shape, 2)
        plane_idx_0 = plane_idx[..., 0]
        plane_idx_1 = plane_idx[..., 1]

        pred_c1 = pred_atom23[:, :, C1_INDEX]
        gt_c1 = gt_atom23[:, :, C1_INDEX]
        pred_n = _gather_atom_coords(pred_atom23, glyco_idx)
        gt_n = _gather_atom_coords(gt_atom23, glyco_idx)
        pred_plane_0 = _gather_atom_coords(pred_atom23, plane_idx_0)
        pred_plane_1 = _gather_atom_coords(pred_atom23, plane_idx_1)
        gt_plane_0 = _gather_atom_coords(gt_atom23, plane_idx_0)
        gt_plane_1 = _gather_atom_coords(gt_atom23, plane_idx_1)

        c1_mask = atom23_mask[:, :, C1_INDEX]
        glyco_mask = _gather_atom_mask(atom23_mask, glyco_idx)
        plane_mask_0 = _gather_atom_mask(atom23_mask, plane_idx_0)
        plane_mask_1 = _gather_atom_mask(atom23_mask, plane_idx_1)
        anchor_mask = loss_mask * c1_mask * glyco_mask
        plane_mask = anchor_mask * plane_mask_0 * plane_mask_1

        pred_c1_to_n = pred_n - pred_c1
        gt_c1_to_n = gt_n - gt_c1
        anchor_vector_sq = torch.sum((pred_c1_to_n - gt_c1_to_n) ** 2, dim=-1)
        pred_glyco_dist = torch.linalg.norm(pred_c1_to_n, dim=-1)
        gt_glyco_dist = torch.linalg.norm(gt_c1_to_n, dim=-1)
        glyco_distance_sq = (pred_glyco_dist - gt_glyco_dist) ** 2

        pred_n_to_plane_0 = pred_plane_0 - pred_n
        pred_n_to_plane_1 = pred_plane_1 - pred_n
        gt_n_to_plane_0 = gt_plane_0 - gt_n
        gt_n_to_plane_1 = gt_plane_1 - gt_n
        base_shape_sq = 0.5 * (
            torch.sum((pred_n_to_plane_0 - gt_n_to_plane_0) ** 2, dim=-1)
            + torch.sum((pred_n_to_plane_1 - gt_n_to_plane_1) ** 2, dim=-1)
        )

        anchor_denom = anchor_mask.sum(dim=-1).clamp(min=1.0)
        plane_denom = plane_mask.sum(dim=-1).clamp(min=1.0)
        return {
            "local_geometry_anchor_vector_loss": torch.sum(anchor_vector_sq * anchor_mask, dim=-1) / anchor_denom,
            "local_geometry_glycosidic_distance_loss": torch.sum(glyco_distance_sq * anchor_mask, dim=-1) / anchor_denom,
            "local_geometry_base_shape_loss": torch.sum(base_shape_sq * plane_mask, dim=-1) / plane_denom,
        }

    def _require_finite(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if torch.isfinite(value).all():
            return value
        bad_entries = torch.nonzero(~torch.isfinite(value), as_tuple=False)[:10].tolist()
        raise FloatingPointError(f"DuetRNA produced non-finite `{name}` at entries {bad_entries}.")

    def _timing_enabled(self) -> bool:
        return bool(getattr(self._exp_cfg.training, "enable_step_timing_logs", False))

    def _sync_timing_device(self, device: torch.device) -> None:
        if device.type != "cuda":
            return
        if not bool(getattr(self._exp_cfg.training, "timing_log_sync_cuda", True)):
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize(device=device)

    @staticmethod
    def _timing_scalar(device: torch.device, value: float) -> torch.Tensor:
        return torch.tensor(float(value), device=device, dtype=torch.float32)

    def model_step(self, noisy_batch):
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch["res_mask"]
        timing_enabled = self._timing_enabled()
        timing_device = loss_mask.device
        timing_metrics = {}
        valid_counts = torch.sum(loss_mask, dim=-1)
        if torch.any(valid_counts <= 0):
            bad_batch = torch.nonzero(valid_counts <= 0, as_tuple=False).flatten().tolist()
            raise ValueError(
                f"DuetRNA batch contains samples with zero valid residues at indices {bad_batch}."
            )
        aatype4 = noisy_batch["aatype4"]
        atom23_gt = noisy_batch["atom23_gt_positions"]
        atom23_mask = noisy_batch["atom23_gt_mask"]
        num_batch, num_res = loss_mask.shape

        gt_base_trans = noisy_batch["base_trans_1"]
        gt_base_rotmats = noisy_batch["base_rotmats_1"]
        gt_sugar_trans = noisy_batch["sugar_trans_1"]
        gt_sugar_rotmats = noisy_batch["sugar_rotmats_1"]
        gt_rel_trans = noisy_batch["rel_trans_1"]
        gt_rel_rotmats = noisy_batch["rel_rotmats_1"]
        gt_torsions = noisy_batch["torsion_angles_sin_cos"][:, :, :8, :].reshape(num_batch, num_res, 16)
        gt_chi = noisy_batch["torsion_angles_sin_cos"][:, :, 9, :]

        if timing_enabled:
            self._sync_timing_device(timing_device)
            model_forward_start = time.perf_counter()
        model_output = self.model(noisy_batch)
        if timing_enabled:
            self._sync_timing_device(timing_device)
            timing_metrics["timing_model_forward_ms"] = self._timing_scalar(
                timing_device,
                1000.0 * (time.perf_counter() - model_forward_start),
            )
        pred_base_trans = model_output["pred_base_trans"]
        pred_base_rotmats = model_output["pred_base_rotmats"]
        pred_sugar_trans = model_output["pred_sugar_trans"]
        pred_sugar_rotmats = model_output["pred_sugar_rotmats"]
        pred_rel_trans = model_output["pred_rel_trans"]
        pred_rel_rotmats = model_output["pred_rel_rotmats"]
        pred_torsions = model_output["pred_torsions"].reshape(num_batch, num_res, 16)
        pred_base_logits = model_output["pred_base_logits"]
        pred_chi = model_output["pred_chi"]
        atom_weight = float(getattr(training_cfg, "atom_loss_weight", 1.0))
        soft_base_weight = float(getattr(training_cfg, "soft_base_atom_loss_weight", 0.5))
        chain_weight = float(getattr(training_cfg, "chain_loss_weight", 1.0))
        chi_weight = float(getattr(training_cfg, "chi_loss_weight", 0.25))
        clash_weight = float(getattr(training_cfg, "clash_loss_weight", 0.25))
        local_geometry_weight = float(getattr(training_cfg, "local_geometry_loss_weight", 1.0))
        local_geometry_anchor_weight = float(getattr(training_cfg, "local_geometry_anchor_weight", 1.0))
        local_geometry_distance_weight = float(getattr(training_cfg, "local_geometry_glycosidic_distance_weight", 0.25))
        local_geometry_base_shape_weight = float(getattr(training_cfg, "local_geometry_base_shape_weight", 0.25))
        clash_normalization = str(getattr(training_cfg, "clash_loss_normalization", "pairs"))
        clash_exclude_neighbor_residues = int(getattr(training_cfg, "clash_exclude_neighbor_residues", 0))
        chi_enabled = bool(getattr(training_cfg, "enable_chi_loss", chi_weight > 0.0))
        clash_enabled = bool(getattr(training_cfg, "enable_clash_loss", clash_weight > 0.0))
        local_geometry_enabled = bool(
            getattr(training_cfg, "enable_local_geometry_loss", False)
            and local_geometry_weight > 0.0
        )

        base_trans_loss, base_rot_loss = self._frame_losses(
            noisy_batch,
            pred_base_trans,
            pred_base_rotmats,
            gt_base_trans,
            gt_base_rotmats,
            "base",
        )
        sugar_trans_loss, sugar_rot_loss = self._frame_losses(
            noisy_batch,
            pred_sugar_trans,
            pred_sugar_rotmats,
            gt_sugar_trans,
            gt_sugar_rotmats,
            "sugar",
        )
        rel_terminal_trans_loss, rel_terminal_rot_loss = self._rel_terminal_losses(
            pred_rel_trans,
            pred_rel_rotmats,
            gt_rel_trans,
            gt_rel_rotmats,
            loss_mask,
        )
        base_trans_loss = self._require_finite("base_trans_loss", base_trans_loss)
        base_rot_loss = self._require_finite("base_rot_loss", base_rot_loss)
        rel_terminal_trans_loss = self._require_finite("rel_terminal_trans_loss", rel_terminal_trans_loss)
        rel_terminal_rot_loss = self._require_finite("rel_terminal_rot_loss", rel_terminal_rot_loss)
        sugar_trans_loss = self._require_finite("sugar_trans_loss", sugar_trans_loss)
        sugar_rot_loss = self._require_finite("sugar_rot_loss", sugar_rot_loss)

        flat_logits = pred_base_logits.reshape(num_batch * num_res, 4)
        flat_targets = aatype4.reshape(num_batch * num_res)
        base_ce = F.cross_entropy(flat_logits, flat_targets, reduction="none").reshape(num_batch, num_res)
        base_ce_loss = float(getattr(training_cfg, "base_ce_loss_weight", 1.0)) * torch.sum(
            base_ce * loss_mask,
            dim=-1,
        ) / torch.sum(loss_mask, dim=-1).clamp(min=1.0)
        base_ce_loss = self._require_finite("base_ce_loss", base_ce_loss)

        aux_active = noisy_batch["t"][:, 0] > training_cfg.aux_loss_t_pass
        active_idx = torch.nonzero(aux_active, as_tuple=False).flatten()
        active_fraction = aux_active.float()

        torsion_loss = torch.zeros_like(base_ce_loss)
        atom_loss = torch.zeros_like(base_ce_loss)
        soft_base_atom_loss = torch.zeros_like(base_ce_loss)
        chain_loss = torch.zeros_like(base_ce_loss)
        chi_loss = torch.zeros_like(base_ce_loss)
        clash_loss = torch.zeros_like(base_ce_loss)
        local_geometry_loss = torch.zeros_like(base_ce_loss)
        local_geometry_anchor_vector_loss = torch.zeros_like(base_ce_loss)
        local_geometry_glycosidic_distance_loss = torch.zeros_like(base_ce_loss)
        local_geometry_base_shape_loss = torch.zeros_like(base_ce_loss)
        atom_rmsd = torch.zeros_like(loss_mask, dtype=base_ce_loss.dtype)
        if timing_enabled:
            timing_metrics["timing_atom_block_ms"] = self._timing_scalar(timing_device, 0.0)
            timing_metrics["timing_soft_base_ms"] = self._timing_scalar(timing_device, 0.0)
            timing_metrics["timing_chi_ms"] = self._timing_scalar(timing_device, 0.0)
            timing_metrics["timing_chain_ms"] = self._timing_scalar(timing_device, 0.0)
            timing_metrics["timing_clash_ms"] = self._timing_scalar(timing_device, 0.0)
            timing_metrics["timing_local_geometry_ms"] = self._timing_scalar(timing_device, 0.0)

        if active_idx.numel() > 0:
            active_loss_mask = loss_mask.index_select(0, active_idx)
            active_aatype4 = aatype4.index_select(0, active_idx)
            active_atom23_gt = atom23_gt.index_select(0, active_idx)
            active_atom23_mask = atom23_mask.index_select(0, active_idx)

            pred_torsions_view = pred_torsions.index_select(0, active_idx).reshape(active_idx.numel(), num_res, 8, 2)
            gt_torsions_view = gt_torsions.index_select(0, active_idx).reshape(active_idx.numel(), num_res, 8, 2)
            active_torsion_loss = float(training_cfg.tors_loss_scale) * torch.sum(
                torch.linalg.norm(pred_torsions_view - gt_torsions_view, dim=-1) ** 2 * active_loss_mask[..., None],
                dim=(-1, -2),
            ) / (torch.sum(active_loss_mask, dim=-1) * 8).clamp(min=1.0)
            torsion_loss = _scatter_sample_losses(torsion_loss, active_idx, active_torsion_loss)
            torsion_loss = self._require_finite("torsion_loss", torsion_loss)

            if timing_enabled:
                self._sync_timing_device(timing_device)
                atom_block_start = time.perf_counter()
            pred_atom23_active = reconstruct_atom23(
                pred_base_trans.index_select(0, active_idx),
                pred_base_rotmats.index_select(0, active_idx),
                pred_sugar_trans.index_select(0, active_idx),
                pred_sugar_rotmats.index_select(0, active_idx),
                pred_torsions.index_select(0, active_idx),
                noisy_batch["is_na_residue_mask"].bool().index_select(0, active_idx),
                active_aatype4,
            )
            pred_atom23_active = self._require_finite("pred_atom23", pred_atom23_active)
            atom_profile = _atom_loss_profile_mask(
                active_aatype4,
                getattr(training_cfg, "atom_loss_profile", "origin7_plus_base_plane2"),
                dtype=active_atom23_mask.dtype,
            )
            atom_loss_mask = active_atom23_mask * active_loss_mask[..., None] * atom_profile
            active_atom_loss = atom_weight * torch.sum(
                ((pred_atom23_active - active_atom23_gt) ** 2) * atom_loss_mask[..., None],
                dim=(-1, -2, -3),
            ) / atom_loss_mask.sum(dim=(-1, -2)).clamp(min=1.0)
            atom_loss = _scatter_sample_losses(atom_loss, active_idx, active_atom_loss)
            atom_loss = self._require_finite("atom_loss", atom_loss)
            active_atom_rmsd = compute_rmsd(pred_atom23_active, active_atom23_gt, mask=active_atom23_mask)
            atom_rmsd = _scatter_sample_losses(atom_rmsd, active_idx, active_atom_rmsd)
            atom_rmsd = self._require_finite("atom23_rmsd", atom_rmsd)
            if timing_enabled:
                self._sync_timing_device(timing_device)
                timing_metrics["timing_atom_block_ms"] = self._timing_scalar(
                    timing_device,
                    1000.0 * (time.perf_counter() - atom_block_start),
                )

            if soft_base_weight > 0.0:
                if timing_enabled:
                    self._sync_timing_device(timing_device)
                    soft_base_start = time.perf_counter()
                active_base_probs = F.softmax(pred_base_logits.index_select(0, active_idx), dim=-1)
                pred_soft_base_atoms = reconstruct_soft_base_atoms(
                    pred_base_trans.index_select(0, active_idx),
                    pred_base_rotmats.index_select(0, active_idx),
                    active_base_probs,
                )
                target_base_atom_mask = base_atom_class_mask(active_atom23_gt.device)
                target_base_atom_mask = target_base_atom_mask.index_select(
                    0,
                    active_aatype4.long().reshape(-1),
                ).reshape(active_idx.numel(), num_res, 23)
                soft_base_atom_mask = active_atom23_mask * active_loss_mask[..., None] * target_base_atom_mask.to(active_atom23_mask.dtype)
                active_soft_base_atom_loss = soft_base_weight * torch.sum(
                    ((pred_soft_base_atoms - active_atom23_gt) ** 2) * soft_base_atom_mask[..., None],
                    dim=(-1, -2, -3),
                ) / soft_base_atom_mask.sum(dim=(-1, -2)).clamp(min=1.0)
                soft_base_atom_loss = _scatter_sample_losses(
                    soft_base_atom_loss,
                    active_idx,
                    active_soft_base_atom_loss,
                )
                if timing_enabled:
                    self._sync_timing_device(timing_device)
                    timing_metrics["timing_soft_base_ms"] = self._timing_scalar(
                        timing_device,
                        1000.0 * (time.perf_counter() - soft_base_start),
                    )
            soft_base_atom_loss = self._require_finite("soft_base_atom_loss", soft_base_atom_loss)

            if chi_enabled and chi_weight > 0.0:
                if timing_enabled:
                    self._sync_timing_device(timing_device)
                    chi_start = time.perf_counter()
                active_chi_loss = chi_weight * chi_angle_loss(
                    pred_chi.index_select(0, active_idx),
                    gt_chi.index_select(0, active_idx),
                    active_loss_mask,
                )
                chi_loss = _scatter_sample_losses(chi_loss, active_idx, active_chi_loss)
                if timing_enabled:
                    self._sync_timing_device(timing_device)
                    timing_metrics["timing_chi_ms"] = self._timing_scalar(
                        timing_device,
                        1000.0 * (time.perf_counter() - chi_start),
                    )
            chi_loss = self._require_finite("chi_loss", chi_loss)

            if timing_enabled:
                self._sync_timing_device(timing_device)
                chain_start = time.perf_counter()
            active_chain_loss = chain_weight * self._chain_loss(
                pred_atom23_active,
                active_atom23_gt,
                active_atom23_mask,
                active_loss_mask,
            )
            chain_loss = _scatter_sample_losses(chain_loss, active_idx, active_chain_loss)
            if timing_enabled:
                self._sync_timing_device(timing_device)
                timing_metrics["timing_chain_ms"] = self._timing_scalar(
                    timing_device,
                    1000.0 * (time.perf_counter() - chain_start),
                )
            chain_loss = self._require_finite("chain_loss", chain_loss)

            if local_geometry_enabled:
                if timing_enabled:
                    self._sync_timing_device(timing_device)
                    local_geometry_start = time.perf_counter()
                active_local_geometry_losses = self._local_geometry_losses(
                    pred_atom23_active,
                    active_atom23_gt,
                    active_atom23_mask,
                    active_loss_mask,
                    active_aatype4,
                )
                active_anchor_vector_loss = active_local_geometry_losses["local_geometry_anchor_vector_loss"]
                active_glycosidic_distance_loss = active_local_geometry_losses[
                    "local_geometry_glycosidic_distance_loss"
                ]
                active_base_shape_loss = active_local_geometry_losses["local_geometry_base_shape_loss"]
                active_local_geometry_loss = local_geometry_weight * (
                    local_geometry_anchor_weight * active_anchor_vector_loss
                    + local_geometry_distance_weight * active_glycosidic_distance_loss
                    + local_geometry_base_shape_weight * active_base_shape_loss
                )
                local_geometry_anchor_vector_loss = _scatter_sample_losses(
                    local_geometry_anchor_vector_loss,
                    active_idx,
                    active_anchor_vector_loss,
                )
                local_geometry_glycosidic_distance_loss = _scatter_sample_losses(
                    local_geometry_glycosidic_distance_loss,
                    active_idx,
                    active_glycosidic_distance_loss,
                )
                local_geometry_base_shape_loss = _scatter_sample_losses(
                    local_geometry_base_shape_loss,
                    active_idx,
                    active_base_shape_loss,
                )
                local_geometry_loss = _scatter_sample_losses(
                    local_geometry_loss,
                    active_idx,
                    active_local_geometry_loss,
                )
                if timing_enabled:
                    self._sync_timing_device(timing_device)
                    timing_metrics["timing_local_geometry_ms"] = self._timing_scalar(
                        timing_device,
                        1000.0 * (time.perf_counter() - local_geometry_start),
                    )
            local_geometry_anchor_vector_loss = self._require_finite(
                "local_geometry_anchor_vector_loss",
                local_geometry_anchor_vector_loss,
            )
            local_geometry_glycosidic_distance_loss = self._require_finite(
                "local_geometry_glycosidic_distance_loss",
                local_geometry_glycosidic_distance_loss,
            )
            local_geometry_base_shape_loss = self._require_finite(
                "local_geometry_base_shape_loss",
                local_geometry_base_shape_loss,
            )
            local_geometry_loss = self._require_finite("local_geometry_loss", local_geometry_loss)

            if clash_enabled and clash_weight > 0.0:
                if timing_enabled:
                    self._sync_timing_device(timing_device)
                    clash_start = time.perf_counter()
                active_clash_loss = clash_weight * inter_residue_clash_loss(
                    pred_atom23_active,
                    active_atom23_mask,
                    active_loss_mask,
                    o3_index=O3_INDEX,
                    p_index=P_INDEX,
                    normalization=clash_normalization,
                    exclude_neighbor_residues=clash_exclude_neighbor_residues,
                )
                clash_loss = _scatter_sample_losses(clash_loss, active_idx, active_clash_loss)
                if timing_enabled:
                    self._sync_timing_device(timing_device)
                    timing_metrics["timing_clash_ms"] = self._timing_scalar(
                        timing_device,
                        1000.0 * (time.perf_counter() - clash_start),
                    )
            clash_loss = self._require_finite("clash_loss", clash_loss)
        else:
            torsion_loss = self._require_finite("torsion_loss", torsion_loss)
            atom_loss = self._require_finite("atom_loss", atom_loss)
            soft_base_atom_loss = self._require_finite("soft_base_atom_loss", soft_base_atom_loss)
            chi_loss = self._require_finite("chi_loss", chi_loss)
            chain_loss = self._require_finite("chain_loss", chain_loss)
            clash_loss = self._require_finite("clash_loss", clash_loss)
            local_geometry_loss = self._require_finite("local_geometry_loss", local_geometry_loss)

        # A partial length-bucket batch can contain no samples above the aux
        # t-threshold. Keep the torsion head in DDP's autograd graph with an
        # exactly zero contribution so ordinary DDP cannot diverge across
        # ranks when only some ranks have active auxiliary supervision.
        torsion_loss = torsion_loss + 0.0 * pred_torsions[:, 0, 0]

        base_frame_loss = base_trans_loss + base_rot_loss
        rel_terminal_loss = rel_terminal_trans_loss + rel_terminal_rot_loss
        sugar_frame_loss = sugar_trans_loss + sugar_rot_loss
        total_loss = (
            base_frame_loss
            + rel_terminal_loss
            + sugar_frame_loss
            + base_ce_loss
            + soft_base_atom_loss
            + torsion_loss
            + atom_loss
            + chain_loss
        )
        if chi_enabled:
            total_loss = total_loss + chi_loss
        if clash_enabled:
            total_loss = total_loss + clash_loss
        if local_geometry_enabled:
            total_loss = total_loss + local_geometry_loss
        total_loss = self._require_finite("total_loss", total_loss)

        pred_base_label = torch.argmax(pred_base_logits, dim=-1)
        base_accuracy = torch.sum((pred_base_label == aatype4).float() * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1).clamp(min=1.0)

        out = {
            "base_trans_loss": base_trans_loss,
            "base_rot_loss": base_rot_loss,
            "base_frame_loss": base_frame_loss,
            "rel_terminal_trans_loss": rel_terminal_trans_loss,
            "rel_terminal_rot_loss": rel_terminal_rot_loss,
            "rel_terminal_loss": rel_terminal_loss,
            "sugar_trans_loss": sugar_trans_loss,
            "sugar_rot_loss": sugar_rot_loss,
            "sugar_frame_loss": sugar_frame_loss,
            "torsion_loss": torsion_loss,
            "base_ce_loss": base_ce_loss,
            "soft_base_atom_loss": soft_base_atom_loss,
            "atom_loss": atom_loss,
            "chain_loss": chain_loss,
            "atom23_rmsd": atom_rmsd,
            "base_accuracy": base_accuracy,
            "aux_active_fraction": active_fraction,
            "total_loss": total_loss,
        }
        if chi_enabled:
            out["chi_loss"] = chi_loss
        if clash_enabled:
            out["clash_loss"] = clash_loss
        if local_geometry_enabled:
            out["local_geometry_loss"] = local_geometry_loss
            out["local_geometry_anchor_vector_loss"] = local_geometry_anchor_vector_loss
            out["local_geometry_glycosidic_distance_loss"] = local_geometry_glycosidic_distance_loss
            out["local_geometry_base_shape_loss"] = local_geometry_base_shape_loss
        out.update(timing_metrics)
        return out

    def training_step(self, batch, batch_idx):
        # Keep corruption distinct across DDP ranks and reproducible at the
        # same global step after a checkpoint resume.  With gradient
        # accumulation, ``global_step`` is shared by several microbatches, so
        # include their within-accumulation index instead of replaying identical
        # corruption. Self-conditioning remains rank-shared for each microbatch.
        trainer = getattr(self, "_trainer", None)
        accumulation = max(1, int(getattr(trainer, "accumulate_grad_batches", 1)))
        microbatch_index = int(batch_idx) % accumulation if accumulation > 1 else 0
        torch.manual_seed(
            _runtime_step_seed(
                int(getattr(self._exp_cfg, "seed", 123)),
                int(getattr(self, "global_rank", 0)),
                int(self.global_step),
                microbatch_index,
            )
        )
        timing_enabled = self._timing_enabled()
        timing_device = batch["res_mask"].device
        if timing_enabled:
            self._sync_timing_device(timing_device)
            step_start = time.perf_counter()
            corrupt_start = step_start
        self.interpolant.set_device(batch["res_mask"].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)
        if timing_enabled:
            self._sync_timing_device(timing_device)
            corrupt_ms = 1000.0 * (time.perf_counter() - corrupt_start)
        self_condition_hit = 0.0
        self_condition_dropped = 0.0
        self_condition_ms = 0.0
        sc_cfg = getattr(self._model_cfg, "self_conditioning", None)
        sc_apply_prob = float(getattr(sc_cfg, "train_apply_prob", 0.5))
        sc_dropout_prob = float(getattr(sc_cfg, "train_dropout_prob", 0.0))
        shared_sc_hit = _shared_step_bernoulli(
            int(getattr(self._exp_cfg, "seed", 123)),
            int(self.global_step),
            sc_apply_prob,
            stream=0,
            microbatch_index=microbatch_index,
        )
        if self._interpolant_cfg.self_condition and shared_sc_hit:
            self_condition_hit = 1.0
            if timing_enabled:
                self._sync_timing_device(timing_device)
                self_condition_start = time.perf_counter()
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch["base_trans_sc"] = model_sc["pred_base_trans"]
                noisy_batch["sugar_trans_sc"] = model_sc["pred_sugar_trans"]
                if _shared_step_bernoulli(
                    int(getattr(self._exp_cfg, "seed", 123)),
                    int(self.global_step),
                    sc_dropout_prob,
                    stream=1,
                    microbatch_index=microbatch_index,
                ):
                    self_condition_dropped = 1.0
                    noisy_batch["base_trans_sc"] = torch.zeros_like(noisy_batch["base_trans_sc"])
                    noisy_batch["sugar_trans_sc"] = torch.zeros_like(noisy_batch["sugar_trans_sc"])
            if timing_enabled:
                self._sync_timing_device(timing_device)
                self_condition_ms = 1000.0 * (time.perf_counter() - self_condition_start)
        if timing_enabled:
            self._sync_timing_device(timing_device)
            model_step_start = time.perf_counter()
        batch_losses = self.model_step(noisy_batch)
        if timing_enabled:
            self._sync_timing_device(timing_device)
            model_step_ms = 1000.0 * (time.perf_counter() - model_step_start)
        total_loss = torch.mean(batch_losses["total_loss"])
        if timing_enabled:
            self._sync_timing_device(timing_device)
            batch_losses["timing_corrupt_batch_ms"] = self._timing_scalar(timing_device, corrupt_ms)
            batch_losses["timing_self_condition_ms"] = self._timing_scalar(timing_device, self_condition_ms)
            batch_losses["timing_model_step_ms"] = self._timing_scalar(timing_device, model_step_ms)
            batch_losses["timing_step_total_ms"] = self._timing_scalar(
                timing_device,
                1000.0 * (time.perf_counter() - step_start),
            )
            batch_losses["timing_interbatch_gap_ms"] = self._timing_scalar(
                timing_device,
                0.0 if self._pending_train_gap_ms is None else self._pending_train_gap_ms,
            )
            batch_losses["self_condition_hit"] = self._timing_scalar(timing_device, self_condition_hit)
            batch_losses["self_condition_dropped"] = self._timing_scalar(timing_device, self_condition_dropped)
            batch_losses["actual_batch_size"] = self._timing_scalar(timing_device, float(batch["res_mask"].shape[0]))
            batch_losses["mean_valid_residues"] = batch["res_mask"].sum(dim=-1).float().mean()
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["res_mask"].shape[0])
        for key, value in batch_losses.items():
            self.log(f"train/{key}", torch.mean(value), on_step=True, on_epoch=True, prog_bar=False, batch_size=batch["res_mask"].shape[0])
        return total_loss

    def validation_step(self, batch, batch_idx):
        res_mask = batch["res_mask"]
        self.interpolant.set_device(res_mask.device)
        val_interpolant = DuetRNAInterpolant(self._interpolant_cfg)
        val_interpolant.set_device(res_mask.device)
        sync_dist = self.trainer is not None and self.trainer.world_size > 1
        with torch.no_grad():
            noisy_val_batch = val_interpolant.corrupt_batch(batch)
            val_losses = self.model_step(noisy_val_batch)
        for key, value in val_losses.items():
            if key.startswith("timing_"):
                continue
            metric_key = "loss" if key == "total_loss" else key
            self.log(
                f"valid_objective/{metric_key}",
                torch.mean(value),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=res_mask.shape[0],
                sync_dist=sync_dist,
            )
        num_batch = res_mask.shape[0]
        valid_lengths = res_mask.sum(dim=-1).to(torch.int64)
        num_res = int(valid_lengths.max().item())
        atom23_traj, _, _, pred_aatype4 = val_interpolant.sample_joint(
            num_batch,
            num_res,
            self.model,
        )
        final_samples = du.to_numpy(atom23_traj[-1])
        batch_metrics = []
        rank = int(getattr(self, "global_rank", 0))
        for i in range(num_batch):
            valid_len = int(valid_lengths[i].item())
            final_pos = final_samples[i, :valid_len]
            aatype4 = pred_aatype4[i, :valid_len].detach().cpu().numpy()
            saved_rna_path = write_atom23_to_pdb(
                final_pos,
                aatype4,
                os.path.join(self._sample_write_dir, f"sample_rank_{rank}_{i}_idx_{batch_idx}_len_{valid_len}"),
            )
            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_samples.append([saved_rna_path, self.global_step, wandb.Molecule(saved_rna_path)])
            batch_metrics.append(calc_rna_c4_c4_metrics(final_pos[:, self._c4_idx]))
        self.validation_epoch_metrics.append(pd.DataFrame(batch_metrics))

    def on_validation_epoch_end(self):
        if not self.validation_epoch_metrics:
            self._validation_start_time = None
            self._restore_training_weights()
            return
        sync_dist = self.trainer is not None and self.trainer.world_size > 1
        is_global_zero = self.trainer is None or self.trainer.is_global_zero
        if len(self.validation_epoch_samples) > 0 and is_global_zero and isinstance(self.logger, WandbLogger):
            self.logger.log_table(
                key="valid/samples",
                columns=["sample_path", "global_step", "RNA"],
                data=self.validation_epoch_samples,
            )
        self.validation_epoch_samples.clear()
        val_epoch_metrics = pd.concat(self.validation_epoch_metrics)
        for key, metric_val in val_epoch_metrics.mean().to_dict().items():
            self.log(
                f"valid/{key}",
                float(metric_val),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics),
                sync_dist=sync_dist,
            )
        if self._validation_start_time is not None:
            val_epoch_time = (time.time() - self._validation_start_time) / 60.0
            self.log("valid/epoch_time_minutes", val_epoch_time, on_step=False, on_epoch=True, prog_bar=False, sync_dist=sync_dist)
            self._validation_start_time = None
        self.validation_epoch_metrics.clear()
        self._restore_training_weights()

    def configure_optimizers(self):
        optimizer_cfg = OmegaConf.to_container(self._exp_cfg.optimizer, resolve=True)
        scheduler_name = optimizer_cfg.pop("scheduler", None)
        warmup_steps = int(optimizer_cfg.pop("warmup_steps", 0) or 0)
        min_lr_ratio = float(optimizer_cfg.pop("min_lr_ratio", 0.0) or 0.0)
        optimizer = torch.optim.AdamW(params=self.model.parameters(), **optimizer_cfg)

        if scheduler_name is None or str(scheduler_name).lower() in {"", "none", "null"}:
            return optimizer

        scheduler_name = str(scheduler_name).lower()
        if scheduler_name != "cosine":
            raise ValueError(
                "Unsupported optimizer scheduler "
                f"`{scheduler_name}`. Expected `cosine` or `none`."
            )

        max_steps = int(getattr(self._exp_cfg.trainer, "max_steps", -1))
        if max_steps <= 0:
            raise ValueError("Cosine scheduler requires experiment.trainer.max_steps > 0.")

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            decay_steps = max(1, max_steps - warmup_steps)
            progress = min(1.0, float(max(0, step - warmup_steps)) / float(decay_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def predict_step(self, batch):
        device = next(self.model.parameters()).device
        interpolant = DuetRNAInterpolant(self._infer_cfg.interpolant)
        interpolant.set_device(device)
        sample_length = batch["num_res"].item()
        sample_id = batch["sample_id"].item()
        # Make each generated structure deterministic and unique independent of
        # DDP rank assignment.  A single process-global seed can otherwise give
        # different ranks identical priors for same-length samples.
        base_seed = int(getattr(self._infer_cfg, "seed", 123))
        sample_seed = _inference_sample_seed(base_seed, int(sample_length), int(sample_id))
        torch.manual_seed(sample_seed)
        sample_dir = os.path.join(self._output_dir, f"length_{sample_length}")
        os.makedirs(sample_dir, exist_ok=True)

        atom23_traj, base_clean_traj, rel_clean_traj, pred_aatype4 = interpolant.sample_joint(
            1,
            sample_length,
            self.model,
        )
        atom23_np = du.to_numpy(torch.concat(atom23_traj, dim=0))
        pred_aatype4_single = pred_aatype4[0].detach().cpu()
        sequence = aatype4_to_sequence(pred_aatype4_single)

        sample = atom23_np[-1]
        sample_prefix = os.path.join(sample_dir, f"sample_{sample_id}")
        saved_rna_path = write_atom23_to_pdb(sample, pred_aatype4_single.numpy(), sample_prefix)
        saved_rna_traj_path = write_atom23_to_pdb(atom23_np, pred_aatype4_single.numpy(), sample_prefix + "_traj.pdb")
        sequence_path = sample_prefix + ".fasta"
        with open(sequence_path, "w", encoding="utf-8") as handle:
            handle.write(f">length_{sample_length}_sample_{sample_id}\n{sequence}\n")
        return {
            "sample_path": saved_rna_path,
            "traj_path": saved_rna_traj_path,
            "sequence": sequence,
            "sequence_path": sequence_path,
            "sample_seed": sample_seed,
            "num_base_steps": len(base_clean_traj),
            "num_rel_steps": len(rel_clean_traj),
        }
