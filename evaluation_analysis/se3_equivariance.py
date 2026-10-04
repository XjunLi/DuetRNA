"""Numerically measure checkpoint-level global SE(3) equivariance error."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from evaluation_analysis.common import file_fingerprint, load_toml, resolve_path, write_json


def _checkpoint_state_mapping(checkpoint: Any) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint root must be a mapping")
    raw = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    if not isinstance(raw, Mapping):
        raise TypeError("Checkpoint has no mapping state_dict")
    return raw


def _checkpoint_has_pose_projection(checkpoint: Any) -> bool:
    raw = _checkpoint_state_mapping(checkpoint)
    return any(
        "base_pose_proj." in str(key) or "sugar_pose_proj." in str(key)
        for key in raw
    )


def _extract_model_state(checkpoint: Any, model_keys: set[str]) -> dict[str, Any]:
    raw = _checkpoint_state_mapping(checkpoint)
    prefixes = ("", "model.", "module.", "module.model.")
    candidates: list[tuple[int, dict[str, Any]]] = []
    for prefix in prefixes:
        transformed = {
            key[len(prefix) :] if prefix and key.startswith(prefix) else key: value
            for key, value in raw.items()
            if not prefix or key.startswith(prefix)
        }
        overlap = len(model_keys.intersection(transformed))
        candidates.append((overlap, transformed))
    overlap, selected = max(candidates, key=lambda item: item[0])
    if overlap != len(model_keys):
        missing = sorted(model_keys - set(selected))
        raise RuntimeError(
            f"Checkpoint does not cover the model state: matched {overlap}/{len(model_keys)}, "
            f"missing={missing[:10]}"
        )
    return {key: selected[key] for key in model_keys}


def _random_rotations(torch: Any, shape: tuple[int, ...], *, device: Any) -> Any:
    matrix = torch.randn(*shape, 3, 3, device=device)
    orthogonal, _ = torch.linalg.qr(matrix)
    sign = torch.where(torch.linalg.det(orthogonal) < 0, -1.0, 1.0)
    orthogonal[..., :, -1] *= sign[..., None]
    return orthogonal


def _move_translations(torch: Any, value: Any, rotation: Any, shift: Any) -> Any:
    return torch.einsum("ij,blj->bli", rotation, value) + shift


def _move_rotations(torch: Any, value: Any, rotation: Any) -> Any:
    return torch.einsum("ij,bljk->blik", rotation, value)


def _error_statistics(torch: Any, observed: Any, expected: Any) -> dict[str, float]:
    difference = (observed - expected).detach().float()
    return {
        "max_abs": float(difference.abs().max().cpu()),
        "rms": float(torch.sqrt(torch.mean(difference.square())).cpu()),
    }


def run_from_config(config_path: str | Path, *, output_override: str | Path | None = None) -> Path:
    # Keep the training stack lazy so diversity/geometry table consumers can run
    # without PyTorch installed.
    import torch
    from omegaconf import OmegaConf

    from duetrna.models.flow_model import (
        DuetRNAFlowModel,
    )

    config_file, root = load_toml(config_path)
    config = root.get("se3_equivariance")
    if not isinstance(config, Mapping):
        raise KeyError("Config must contain a [se3_equivariance] table")
    config_dir = config_file.parent
    model_config_path = resolve_path(
        config.get("model_config"), base_dir=config_dir, must_exist=True
    )
    checkpoint_path = resolve_path(
        config.get("checkpoint"), base_dir=config_dir, must_exist=True
    )
    output_path = resolve_path(
        output_override or config.get("output"), base_dir=config_dir
    )
    if model_config_path is None or checkpoint_path is None or output_path is None:
        raise KeyError("model_config, checkpoint, and output are required")

    device_name = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_name)
    seed = int(config.get("seed", 7))
    trials = int(config.get("trials", 8))
    batch_size = int(config.get("batch_size", 1))
    length = int(config.get("length", 64))
    shift_scale = float(config.get("shift_scale", 20.0))
    if trials < 1 or batch_size < 1 or length < 1:
        raise ValueError("trials, batch_size, and length must be positive")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, int(config.get("torch_threads", 1))))

    model_config = OmegaConf.load(model_config_path)
    model_section = model_config.model if "model" in model_config else model_config
    if "pose_conditioning_mode" in config:
        model_section.pose_conditioning.mode = str(config["pose_conditioning_mode"])
    model = DuetRNAFlowModel(model_section).to(device).eval()
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    expected_pose_projection = str(model_section.pose_conditioning.mode).lower() == "absolute_mlp"
    checkpoint_has_pose_projection = _checkpoint_has_pose_projection(checkpoint)
    if checkpoint_has_pose_projection != expected_pose_projection:
        raise RuntimeError(
            "Checkpoint/model pose-conditioning mismatch: "
            f"checkpoint_has_pose_projection={checkpoint_has_pose_projection}, "
            f"configured_mode={model_section.pose_conditioning.mode!s}. "
            "Do not disable an old checkpoint's pose MLP only at evaluation time."
        )
    model_state = _extract_model_state(checkpoint, set(model.state_dict()))
    model.load_state_dict(model_state, strict=True)

    trial_rows: list[dict[str, Any]] = []
    equivariant_keys = ("base", "sugar")
    invariant_keys = (
        "pred_rel_trans",
        "pred_rel_rotmats",
        "pred_torsions",
        "pred_base_logits",
        "pred_chi",
    )
    with torch.no_grad():
        for trial in range(trials):
            features = {
                "res_mask": torch.ones(batch_size, length, device=device),
                "t": torch.rand(batch_size, 1, device=device),
                "base_trans_t": torch.randn(batch_size, length, 3, device=device),
                "sugar_trans_t": torch.randn(batch_size, length, 3, device=device),
                "base_rotmats_t": _random_rotations(
                    torch, (batch_size, length), device=device
                ),
                "sugar_rotmats_t": _random_rotations(
                    torch, (batch_size, length), device=device
                ),
                "base_trans_sc": torch.randn(batch_size, length, 3, device=device),
                "sugar_trans_sc": torch.randn(batch_size, length, 3, device=device),
            }
            global_rotation = _random_rotations(torch, (), device=device)
            global_shift = shift_scale * torch.randn(1, 1, 3, device=device)
            moved = {key: value.clone() for key, value in features.items()}
            for key in (
                "base_trans_t",
                "sugar_trans_t",
                "base_trans_sc",
                "sugar_trans_sc",
            ):
                moved[key] = _move_translations(
                    torch, features[key], global_rotation, global_shift
                )
            for key in ("base_rotmats_t", "sugar_rotmats_t"):
                moved[key] = _move_rotations(torch, features[key], global_rotation)

            original_output = model(features)
            moved_output = model(moved)
            errors: dict[str, dict[str, float]] = {}
            for prefix in equivariant_keys:
                translation_key = f"pred_{prefix}_trans"
                rotation_key = f"pred_{prefix}_rotmats"
                errors[translation_key] = _error_statistics(
                    torch,
                    moved_output[translation_key],
                    _move_translations(
                        torch, original_output[translation_key], global_rotation, global_shift
                    ),
                )
                errors[rotation_key] = _error_statistics(
                    torch,
                    moved_output[rotation_key],
                    _move_rotations(torch, original_output[rotation_key], global_rotation),
                )
            for key in invariant_keys:
                errors[key] = _error_statistics(torch, moved_output[key], original_output[key])
            trial_rows.append(
                {
                    "trial": trial,
                    "global_shift": global_shift.detach().cpu().flatten().tolist(),
                    "errors": errors,
                    "max_error": max(value["max_abs"] for value in errors.values()),
                }
            )

    corrected_max_error = max(float(row["max_error"]) for row in trial_rows)
    all_rms = [
        statistics["rms"]
        for row in trial_rows
        for statistics in row["errors"].values()
    ]
    report = {
        "corrected_max_error": corrected_max_error,
        "mean_component_rms_error": sum(all_rms) / len(all_rms),
        "pose_conditioning_mode": str(model_section.pose_conditioning.mode),
        "checkpoint_has_pose_projection": checkpoint_has_pose_projection,
        "trials": trials,
        "batch_size": batch_size,
        "length": length,
        "seed": seed,
        "device": str(device),
        "model_config": file_fingerprint(model_config_path, include_sha256=True),
        "checkpoint": file_fingerprint(
            checkpoint_path, include_sha256=bool(config.get("hash_checkpoint", False))
        ),
        "trial_results": trial_rows,
    }
    write_json(output_path, report)
    failure_threshold = config.get("fail_above")
    if failure_threshold is not None and corrected_max_error > float(failure_threshold):
        raise RuntimeError(
            f"SE(3) equivariance error {corrected_max_error:.6g} exceeds "
            f"configured threshold {float(failure_threshold):.6g}; report written to {output_path}"
        )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure global SE(3) equivariance error")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    output = run_from_config(arguments.config, output_override=arguments.output)
    print(f"Wrote SE(3) equivariance report to {output}")


if __name__ == "__main__":
    main()
