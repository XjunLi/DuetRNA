from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_benchmark_preset(preset: str | None) -> Path | None:
    if not preset:
        return None
    preset_path = Path(str(preset))
    if not preset_path.is_absolute():
        preset_path = _repo_root() / preset_path
    if not preset_path.exists():
        raise FileNotFoundError(f"Benchmark preset not found: {preset_path}")
    return preset_path


def apply_benchmark_preset(cfg: DictConfig) -> DictConfig:
    preset_path = resolve_benchmark_preset(cfg.inference.get("benchmark_preset", None))
    if preset_path is None:
        return cfg
    preset_cfg = OmegaConf.load(preset_path)
    OmegaConf.set_struct(cfg, False)
    # Treat presets as benchmark defaults while preserving explicit Hydra
    # command-line overrides such as inference.samples.samples_per_length=20.
    return OmegaConf.merge(preset_cfg, cfg)
