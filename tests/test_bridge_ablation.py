"""Checkpoint-compatible bridge ablation checks; no data or GPU required."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from duetrna.models.flow_model import DuetRNAFlowModel


def model_config(enabled=True, relative_rotation=False, blocks=2):
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "duetrna/configs/config.yaml")
    cfg.model.node_embed_size = 32
    cfg.model.edge_embed_size = 16
    cfg.model.node_features.c_pos_emb = 16
    cfg.model.node_features.c_timestep_emb = 16
    cfg.model.edge_features.feat_dim = 8
    cfg.model.ipa.c_hidden = 8
    cfg.model.ipa.no_heads = 2
    cfg.model.ipa.no_qk_points = 2
    cfg.model.ipa.no_v_points = 2
    cfg.model.ipa.num_blocks = blocks
    cfg.model.ipa.seq_tfmr_num_layers = 1
    cfg.model.cross_edge.enabled = enabled
    cfg.model.cross_edge.use_relative_rotation = relative_rotation
    return cfg.model


def rotations(*shape):
    q, _ = torch.linalg.qr(torch.randn(*shape, 3, 3))
    q[..., :, -1] *= torch.linalg.det(q).unsqueeze(-1)
    return q


def batch():
    torch.manual_seed(8)
    return {
        "res_mask": torch.tensor([[1., 1., 1., 1., 0.], [1., 1., 1., 0., 0.]]),
        "t": torch.tensor([[0.3], [0.7]]),
        "base_trans_t": torch.randn(2, 5, 3) * 4,
        "sugar_trans_t": torch.randn(2, 5, 3) * 4,
        "base_rotmats_t": rotations(2, 5),
        "sugar_rotmats_t": rotations(2, 5),
        "base_trans_sc": torch.randn(2, 5, 3) * 4,
        "sugar_trans_sc": torch.randn(2, 5, 3) * 4,
    }


def model(enabled=True, relative_rotation=False, blocks=2):
    torch.manual_seed(31)
    np.random.seed(31)  # IPA also initializes through scipy.stats.truncnorm.
    net = DuetRNAFlowModel(model_config(enabled, relative_rotation, blocks)).eval()
    # Activate zero-initialized outputs: otherwise a fresh model can ignore all
    # pair features, making the bridge-sensitivity check vacuous.
    with torch.no_grad():
        for name, param in net.named_parameters():
            if (
                "_bb_update_" in name
                or ("_ipa_" in name and ".linear_out." in name)
                or ("edge_transition_" in name and ".final_layer." in name)
            ):
                param.normal_(std=0.02)
    return net


def assert_outputs_close(actual, expected, mask, atol=3e-5, rtol=3e-5):
    for key in actual:
        torch.testing.assert_close(actual[key][mask], expected[key][mask], atol=atol, rtol=rtol)


def test_old_config_defaults_to_enabled():
    cfg = model_config()
    del cfg.cross_edge.enabled
    old_config_model = DuetRNAFlowModel(cfg).eval()
    enabled = model()
    old_config_model.load_state_dict(enabled.state_dict(), strict=True)
    inputs = batch()
    with torch.no_grad():
        assert_outputs_close(old_config_model(inputs), enabled(inputs), inputs["res_mask"].bool(), 0, 0)


@pytest.mark.parametrize("relative_rotation", [False, True])
def test_checkpoint_keys_values_and_parameter_order_are_unchanged(relative_rotation):
    enabled = model(True, relative_rotation)
    disabled = model(False, relative_rotation)
    assert enabled.state_dict().keys() == disabled.state_dict().keys()
    assert [n for n, _ in enabled.named_parameters()] == [n for n, _ in disabled.named_parameters()]
    for name, value in enabled.state_dict().items():
        torch.testing.assert_close(value, disabled.state_dict()[name], atol=0, rtol=0)
    disabled.load_state_dict(enabled.state_dict(), strict=True)
    enabled.load_state_dict(disabled.state_dict(), strict=True)


@pytest.mark.parametrize("relative_rotation", [False, True])
def test_disabled_branch_equals_zeroing_legacy_encoder_output(relative_rotation):
    enabled = model(True, relative_rotation)
    disabled = model(False, relative_rotation)
    inputs = batch()
    mask = inputs["res_mask"].bool()
    with torch.no_grad():
        normal_output = enabled(inputs)
        handle = enabled.cross_edge_embedder.register_forward_hook(
            lambda _module, _args, output: torch.zeros_like(output)
        )
        try:
            zeroed_output = enabled(inputs)
        finally:
            handle.remove()
        disabled_output = disabled(inputs)
    assert_outputs_close(disabled_output, zeroed_output, mask, 0, 0)
    assert (normal_output["pred_base_logits"][mask] - disabled_output["pred_base_logits"][mask]).abs().max() > 1e-6


def test_disabled_encoder_is_not_executed_and_its_weights_have_no_effect():
    net = model(False, True)
    inputs = batch()
    with torch.no_grad():
        original = net(inputs)
        for module in (net.cross_edge_embedder, net.bridge_rot_proj):
            for param in module.parameters():
                param.fill_(float("nan"))
        changed = net(inputs)
    assert_outputs_close(changed, original, inputs["res_mask"].bool(), 0, 0)


@pytest.mark.parametrize("self_condition", [False, True])
def test_disabled_model_preserves_global_se3_equivariance(self_condition):
    net = model(False)
    inputs = batch()
    if not self_condition:
        del inputs["base_trans_sc"], inputs["sugar_trans_sc"]
    moved = copy.deepcopy(inputs)
    g, shift = rotations(), torch.tensor([6., -2., 4.])
    for name in inputs:
        if "trans_" in name:
            moved[name] = inputs[name] @ g.T + shift
        elif "rotmats_" in name:
            moved[name] = g @ inputs[name]
    with torch.no_grad():
        original, actual = net(inputs), net(moved)
    expected = dict(original)
    for kind in ("base", "sugar"):
        expected[f"pred_{kind}_trans"] = original[f"pred_{kind}_trans"] @ g.T + shift
        expected[f"pred_{kind}_rotmats"] = g @ original[f"pred_{kind}_rotmats"]
    assert_outputs_close(actual, expected, inputs["res_mask"].bool())


def test_padding_cannot_change_valid_predictions():
    net = model(False)
    inputs = batch()
    changed = copy.deepcopy(inputs)
    mask = inputs["res_mask"].bool()
    for name in inputs:
        if "trans_" in name:
            changed[name][~mask] = torch.randn_like(changed[name][~mask]) * 100
        elif "rotmats_" in name:
            changed[name][~mask] = rotations(int((~mask).sum()))
    with torch.no_grad():
        assert_outputs_close(net(inputs), net(changed), mask)


@pytest.mark.parametrize("blocks", [1, 2])
def test_backward_and_optimizer_state_restore(blocks):
    original = model(True, True, blocks)
    original_optimizer = torch.optim.AdamW(original.parameters(), lr=1e-4)
    inputs = batch()
    mask = inputs["res_mask"].bool()
    loss = sum(value[mask].square().mean() for value in original(inputs).values())
    loss.backward()
    original_optimizer.step()

    disabled = model(False, True, blocks).train()
    disabled.load_state_dict(original.state_dict(), strict=True)
    # Match flow_module.configure_optimizers: retain all parameters, including
    # frozen parameters, so an existing optimizer's groups still align.
    optimizer = torch.optim.AdamW(disabled.parameters(), lr=1e-4)
    optimizer.load_state_dict(original_optimizer.state_dict())
    frozen_before = {
        name: param.detach().clone()
        for name, param in disabled.named_parameters() if not param.requires_grad
    }
    loss = sum(value[mask].square().mean() for value in disabled(inputs).values())
    loss.backward()
    for name, param in disabled.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name
        else:
            assert param.grad is None, name
    optimizer.step()
    for name, param in disabled.named_parameters():
        if name in frozen_before:
            torch.testing.assert_close(param, frozen_before[name], atol=0, rtol=0)
