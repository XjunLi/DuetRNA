from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from duetrna_shared_core import data_utils as du
from duetrna_modeling import torsion_net
from duetrna_modeling.edge_embedder import EdgeEmbedder
from duetrna_modeling.ipa_pytorch import (
    BackboneUpdate,
    EdgeTransition,
    InvariantPointAttention,
    Linear,
    StructureModuleTransition,
)
from duetrna_modeling.node_embedder import NodeEmbedder


def _rotmat_from_6d(x: torch.Tensor) -> torch.Tensor:
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def _rigids_ang_to_nm(x):
    """Convert rigid translations from Angstroms to nanometres. Module-level so pickle works."""
    return x.apply_trans_fn(lambda y: y * du.ANG_TO_NM_SCALE)


def _rigids_nm_to_ang(x):
    """Convert rigid translations from nanometres to Angstroms. Module-level so pickle works."""
    return x.apply_trans_fn(lambda y: y * du.NM_TO_ANG_SCALE)


class DuetRNAFlowModel(nn.Module):
    def __init__(self, model_conf):
        super().__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self._object_mixing_every = max(int(getattr(self._ipa_conf, "object_mixing_every", 1)), 1)
        self._object_mixing_blocks = frozenset(
            block
            for block in range(int(self._ipa_conf.num_blocks))
            if (block % self._object_mixing_every == 0)
            or (block == int(self._ipa_conf.num_blocks) - 1)
        )
        self._cross_edge_use_relative_rotation = bool(
            getattr(getattr(model_conf, "cross_edge", None), "use_relative_rotation", False)
        )
        pose_conditioning_conf = getattr(model_conf, "pose_conditioning", None)
        self._pose_conditioning_mode = str(
            getattr(pose_conditioning_conf, "mode", "absolute_mlp")
        ).lower()
        if self._pose_conditioning_mode not in {"none", "absolute_mlp"}:
            raise ValueError(
                "model.pose_conditioning.mode must be 'none' or 'absolute_mlp', "
                f"got {self._pose_conditioning_mode!r}"
            )
        base_logit_head_conf = getattr(model_conf, "base_logit_head", None)
        self._base_logit_use_attention_pool = bool(
            getattr(base_logit_head_conf, "use_attention_pool", False)
        )
        self._base_logit_attention_heads = max(int(getattr(base_logit_head_conf, "num_attention_heads", 1)), 1)
        self._base_logit_attention_cutoff = float(
            getattr(base_logit_head_conf, "distance_cutoff", 15.0)
        )
        self._base_logit_attention_qk_mode = str(
            getattr(base_logit_head_conf, "attention_qk_mode", "vector")
        ).lower()
        if self._base_logit_attention_qk_mode not in {"vector", "scalar"}:
            raise ValueError(
                "model.base_logit_head.attention_qk_mode must be 'vector' or 'scalar', "
                f"got {self._base_logit_attention_qk_mode!r}"
            )
        self.rigids_ang_to_nm = _rigids_ang_to_nm
        self.rigids_nm_to_ang = _rigids_nm_to_ang

        self.node_embedder = NodeEmbedder(model_conf.node_features)
        self.base_edge_embedder = EdgeEmbedder(model_conf.edge_features)
        self.sugar_edge_embedder = EdgeEmbedder(model_conf.edge_features)
        self.cross_edge_embedder = EdgeEmbedder(model_conf.edge_features)

        c_s = self._ipa_conf.c_s
        c_p = self._model_conf.edge_embed_size
        if self._pose_conditioning_mode == "absolute_mlp":
            self.base_pose_proj = nn.Sequential(
                nn.Linear(12, c_s),
                nn.ReLU(),
                nn.Linear(c_s, c_s),
            )
            self.sugar_pose_proj = nn.Sequential(
                nn.Linear(12, c_s),
                nn.ReLU(),
                nn.Linear(c_s, c_s),
            )
        self.frame_type_embed = nn.Embedding(2, c_s)
        self.edge_fuse = nn.Sequential(
            nn.Linear(3 * c_p, c_p),
            nn.ReLU(),
            nn.Linear(c_p, c_p),
            nn.LayerNorm(c_p),
        )
        self.summary_proj = nn.Sequential(
            nn.Linear(2 * c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )
        if self._cross_edge_use_relative_rotation:
            self.bridge_rot_proj = nn.Linear(6, c_s)
        self.base_pair_pool_proj = nn.Sequential(
            nn.Linear(c_p, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )
        self.sugar_pair_pool_proj = nn.Sequential(
            nn.Linear(c_p, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )
        self.fused_pair_pool_proj = nn.Sequential(
            nn.Linear(c_p, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )
        if self._base_logit_use_attention_pool:
            if self._base_logit_attention_qk_mode == "scalar":
                self._base_logit_attention_head_dim = 1
            else:
                self._base_logit_attention_head_dim = max(c_s // self._base_logit_attention_heads, 1)
            qk_out_dim = self._base_logit_attention_heads * self._base_logit_attention_head_dim
            self.base_logit_attn_q = nn.Linear(
                c_s,
                qk_out_dim,
                bias=False,
            )
            self.base_logit_attn_k = nn.Linear(
                c_s,
                qk_out_dim,
                bias=False,
            )
            self.base_logit_attn_out = nn.Linear(self._base_logit_attention_heads * c_s, c_s)

        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f"base_ipa_{b}"] = InvariantPointAttention(self._ipa_conf)
            self.trunk[f"sugar_ipa_{b}"] = InvariantPointAttention(self._ipa_conf)
            self.trunk[f"base_ipa_ln_{b}"] = nn.LayerNorm(c_s)
            self.trunk[f"sugar_ipa_ln_{b}"] = nn.LayerNorm(c_s)
            obj_tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=c_s,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=c_s,
                batch_first=True,
                dropout=0.0,
                norm_first=False,
            )
            self.trunk[f"obj_tfmr_{b}"] = torch.nn.TransformerEncoder(
                obj_tfmr_layer,
                self._ipa_conf.seq_tfmr_num_layers,
                enable_nested_tensor=False,
            )
            self.trunk[f"post_obj_tfmr_{b}"] = Linear(c_s, c_s, init="final")
            self.trunk[f"obj_ln_{b}"] = nn.LayerNorm(c_s)
            if b not in self._object_mixing_blocks:
                # Keep the legacy state-dict keys for checkpoint compatibility,
                # but do not advertise parameters from a branch forward never
                # executes as trainable. This also stops DDP's unused-parameter
                # mode from hiding accidental dead modules.
                self.trunk[f"obj_tfmr_{b}"].requires_grad_(False)
                self.trunk[f"post_obj_tfmr_{b}"].requires_grad_(False)
                self.trunk[f"obj_ln_{b}"].requires_grad_(False)
            self.trunk[f"base_transition_{b}"] = StructureModuleTransition(c=c_s)
            self.trunk[f"sugar_transition_{b}"] = StructureModuleTransition(c=c_s)
            self.trunk[f"base_bb_update_{b}"] = BackboneUpdate(c_s, use_rot_updates=True)
            self.trunk[f"sugar_bb_update_{b}"] = BackboneUpdate(c_s, use_rot_updates=True)
            if b < self._ipa_conf.num_blocks - 1:
                self.trunk[f"edge_transition_{b}"] = EdgeTransition(
                    node_embed_size=c_s,
                    edge_embed_in=self._model_conf.edge_embed_size,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

        self.base_logit_head = nn.Sequential(
            nn.Linear(4 * c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, 4),
        )
        self.angle_pred_net = torsion_net.TorsionAngleHead(
            c_in=c_s,
            c_hidden=128,
            no_blocks=2,
            no_angles=8,
            epsilon=1e-12,
        )
        self.chi_head = nn.Sequential(
            nn.Linear(2 * c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, 2),
        )

    def _pose_condition(self, trans: torch.Tensor, rotmats: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        pose = torch.cat([trans, rotmats.reshape(*rotmats.shape[:-2], 9)], dim=-1)
        return proj(pose)

    def _flatten_objects(self, base_node_embed: torch.Tensor, sugar_node_embed: torch.Tensor) -> torch.Tensor:
        return torch.stack([base_node_embed, sugar_node_embed], dim=2).reshape(
            base_node_embed.shape[0], base_node_embed.shape[1] * 2, base_node_embed.shape[2]
        )

    def _unflatten_objects(self, obj_tokens: torch.Tensor, num_res: int) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, dim = obj_tokens.shape
        obj_tokens = obj_tokens.reshape(bsz, num_res, 2, dim)
        return obj_tokens[:, :, 0], obj_tokens[:, :, 1]

    def _pool_pair_features(self, pair_feat: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        denom = edge_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return torch.sum(pair_feat * edge_mask[..., None], dim=2) / denom

    def _distance_attention_pool(
        self,
        node: torch.Tensor,
        trans: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        dist = torch.linalg.norm(trans[:, :, None, :] - trans[:, None, :, :], dim=-1)
        pair_mask = node_mask[:, :, None] * node_mask[:, None, :]
        num_res = node.shape[1]
        eye = torch.eye(num_res, device=node.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = pair_mask.bool() & (~eye) & (dist <= self._base_logit_attention_cutoff)
        distance_score = -dist / max(self._base_logit_attention_cutoff, 1e-6)
        q = self.base_logit_attn_q(node).view(
            node.shape[0],
            num_res,
            self._base_logit_attention_heads,
            self._base_logit_attention_head_dim,
        )
        k = self.base_logit_attn_k(node).view(
            node.shape[0],
            num_res,
            self._base_logit_attention_heads,
            self._base_logit_attention_head_dim,
        )
        qk_score = torch.einsum("bihd,bjhd->bijh", q, k) / (self._base_logit_attention_head_dim ** 0.5)
        scores = distance_score[..., None] + qk_score
        mask_value = -1.0e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1.0e9
        scores = scores.masked_fill(~pair_mask[..., None], mask_value)
        weights = torch.softmax(scores, dim=2) * pair_mask[..., None].to(node.dtype)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp(min=1e-8)
        pooled = torch.einsum("bijh,bjd->bihd", weights, node)
        pooled = pooled.reshape(node.shape[0], num_res, self._base_logit_attention_heads * node.shape[-1])
        return self.base_logit_attn_out(pooled) * node_mask[..., None]

    def forward(self, input_feats):
        node_mask = input_feats["res_mask"].float()
        continuous_t = input_feats["t"]
        base_trans_t = input_feats["base_trans_t"]
        base_rotmats_t = input_feats["base_rotmats_t"]
        sugar_trans_t = input_feats["sugar_trans_t"]
        sugar_rotmats_t = input_feats["sugar_rotmats_t"]
        base_trans_sc = input_feats.get("base_trans_sc", torch.zeros_like(base_trans_t))
        sugar_trans_sc = input_feats.get("sugar_trans_sc", torch.zeros_like(base_trans_t))

        base_rigids_t = du.create_rigid(base_rotmats_t, base_trans_t)
        sugar_rigids_t = du.create_rigid(sugar_rotmats_t, sugar_trans_t)

        h0 = self.node_embedder(continuous_t, node_mask)
        base_type = self.frame_type_embed.weight[0].view(1, 1, -1)
        sugar_type = self.frame_type_embed.weight[1].view(1, 1, -1)

        if self._pose_conditioning_mode == "none":
            # IPA already receives the rigid frames and therefore sees geometry
            # equivariantly. Scalar node features must not encode absolute world
            # translations or rotation-matrix entries.
            base_init = h0 + base_type
            sugar_init = h0 + sugar_type
        else:
            base_init = h0 + self._pose_condition(base_trans_t, base_rotmats_t, self.base_pose_proj) + base_type
            sugar_init = h0 + self._pose_condition(sugar_trans_t, sugar_rotmats_t, self.sugar_pose_proj) + sugar_type
        base_init = base_init * node_mask[..., None]
        sugar_init = sugar_init * node_mask[..., None]

        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        base_edge = self.base_edge_embedder(base_init, base_trans_t, base_trans_sc, edge_mask)
        sugar_edge = self.sugar_edge_embedder(sugar_init, sugar_trans_t, sugar_trans_sc, edge_mask)
        bridge_node = self.summary_proj(torch.cat([base_init, sugar_init], dim=-1)) * node_mask[..., None]
        if self._cross_edge_use_relative_rotation:
            bridge_rot = torch.matmul(base_rotmats_t.transpose(-1, -2), sugar_rotmats_t)
            bridge_rot_6d = bridge_rot[..., :, :2].transpose(-1, -2).reshape(*bridge_rot.shape[:-2], 6)
            bridge_node = bridge_node + self.bridge_rot_proj(bridge_rot_6d) * node_mask[..., None]
        bridge_trans_t = sugar_trans_t - base_trans_t
        bridge_trans_sc = sugar_trans_sc - base_trans_sc
        cross_edge = self.cross_edge_embedder(bridge_node, bridge_trans_t, bridge_trans_sc, edge_mask)
        edge_embed = self.edge_fuse(torch.cat([base_edge, sugar_edge, cross_edge], dim=-1)) * edge_mask[..., None]
        base_pair_context = self.base_pair_pool_proj(self._pool_pair_features(base_edge, edge_mask)) * node_mask[..., None]
        sugar_pair_context = self.sugar_pair_pool_proj(self._pool_pair_features(sugar_edge, edge_mask)) * node_mask[..., None]

        base_node = base_init
        sugar_node = sugar_init
        curr_base_rigids = self.rigids_ang_to_nm(base_rigids_t)
        curr_sugar_rigids = self.rigids_ang_to_nm(sugar_rigids_t)

        num_batch, num_res = node_mask.shape
        obj_mask = node_mask.unsqueeze(-1).repeat(1, 1, 2).reshape(num_batch, num_res * 2)

        for b in range(self._ipa_conf.num_blocks):
            base_ipa = self.trunk[f"base_ipa_{b}"](base_node, edge_embed, curr_base_rigids, node_mask)
            sugar_ipa = self.trunk[f"sugar_ipa_{b}"](sugar_node, edge_embed, curr_sugar_rigids, node_mask)
            base_node = self.trunk[f"base_ipa_ln_{b}"](base_node + base_ipa * node_mask[..., None])
            sugar_node = self.trunk[f"sugar_ipa_ln_{b}"](sugar_node + sugar_ipa * node_mask[..., None])

            run_object_mixing = b in self._object_mixing_blocks
            if run_object_mixing:
                obj_tokens = self._flatten_objects(base_node, sugar_node)
                obj_tokens = obj_tokens + self._flatten_objects(
                    base_type.expand(num_batch, num_res, -1),
                    sugar_type.expand(num_batch, num_res, -1),
                )
                obj_tfmr_out = self.trunk[f"obj_tfmr_{b}"](
                    obj_tokens,
                    src_key_padding_mask=(1 - obj_mask).bool(),
                )
                obj_tokens = self.trunk[f"obj_ln_{b}"](
                    obj_tokens + self.trunk[f"post_obj_tfmr_{b}"](obj_tfmr_out)
                )
                obj_tokens = obj_tokens * obj_mask[..., None]
                base_node, sugar_node = self._unflatten_objects(obj_tokens, num_res)

            base_node = self.trunk[f"base_transition_{b}"](base_node) * node_mask[..., None]
            sugar_node = self.trunk[f"sugar_transition_{b}"](sugar_node) * node_mask[..., None]

            base_update = self.trunk[f"base_bb_update_{b}"](base_node)
            sugar_update = self.trunk[f"sugar_bb_update_{b}"](sugar_node)
            curr_base_rigids = curr_base_rigids.compose_q_update_vec(base_update, node_mask[..., None])
            curr_sugar_rigids = curr_sugar_rigids.compose_q_update_vec(sugar_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                summary_node = self.summary_proj(torch.cat([base_node, sugar_node], dim=-1)) * node_mask[..., None]
                edge_embed = self.trunk[f"edge_transition_{b}"](summary_node, edge_embed)
                edge_embed = edge_embed * edge_mask[..., None]

        curr_base_rigids = self.rigids_nm_to_ang(curr_base_rigids)
        curr_sugar_rigids = self.rigids_nm_to_ang(curr_sugar_rigids)
        curr_rel_rigids = curr_base_rigids.invert().compose(curr_sugar_rigids)

        pred_base_trans = curr_base_rigids.get_trans()
        pred_base_rotmats = curr_base_rigids.get_rots().get_rot_mats()
        pred_sugar_trans = curr_sugar_rigids.get_trans()
        pred_sugar_rotmats = curr_sugar_rigids.get_rots().get_rot_mats()
        pred_rel_trans = curr_rel_rigids.get_trans()
        pred_rel_rotmats = curr_rel_rigids.get_rots().get_rot_mats()
        fused_pair_context = (
            self.fused_pair_pool_proj(self._pool_pair_features(edge_embed, edge_mask)) * node_mask[..., None]
        )
        if self._base_logit_use_attention_pool:
            neighbor_pool = self._distance_attention_pool(base_node, pred_base_trans, node_mask)
            base_logit_input = torch.cat(
                [base_node, neighbor_pool, base_pair_context, sugar_pair_context],
                dim=-1,
            )
        else:
            base_logit_input = torch.cat(
                [base_node, base_pair_context, sugar_pair_context, fused_pair_context],
                dim=-1,
            )
        pred_base_logits = self.base_logit_head(base_logit_input) * node_mask[..., None]
        _, pred_torsions = self.angle_pred_net(sugar_node, sugar_init)
        pred_chi = self.chi_head(torch.cat([base_node, sugar_node], dim=-1))
        pred_chi = F.normalize(pred_chi, dim=-1, eps=1e-8) * node_mask[..., None]

        return {
            "pred_base_trans": pred_base_trans,
            "pred_base_rotmats": pred_base_rotmats,
            "pred_sugar_trans": pred_sugar_trans,
            "pred_sugar_rotmats": pred_sugar_rotmats,
            "pred_rel_trans": pred_rel_trans,
            "pred_rel_rotmats": pred_rel_rotmats,
            "pred_torsions": pred_torsions,
            "pred_base_logits": pred_base_logits,
            "pred_chi": pred_chi,
        }
