from __future__ import annotations

import os
from pathlib import Path
import sys

# Resolve local packages for both direct-file and module execution.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time
from glob import glob

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from duetrna import training as eu
from duetrna.data.dataset import LengthDataset
from duetrna.models.flow_module import DuetRNAFlowModule


log = eu.get_pylogger(__name__)


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


def _evalsuite_gpu_ids(infer_cfg: DictConfig) -> tuple[int, int]:
    if not infer_cfg.use_gpu or int(infer_cfg.num_gpus) <= 0:
        return 0, 0
    if int(infer_cfg.num_gpus) == 1:
        return 0, 0
    return 0, 1


def _load_ckpt_cfg(ckpt_dir, explicit_path=None):
    if explicit_path:
        requested_path = os.path.expanduser(str(explicit_path))
        candidates = [requested_path]
        if not os.path.isabs(requested_path):
            candidates.append(os.path.join(ckpt_dir, requested_path))
        for path in candidates:
            if os.path.isfile(path):
                return OmegaConf.load(path), path
        raise FileNotFoundError(
            "The requested checkpoint training configuration does not exist: "
            f"{explicit_path}"
        )

    public_config_path = os.path.join(ckpt_dir, "config.yaml")
    if os.path.isfile(public_config_path):
        return OmegaConf.load(public_config_path), public_config_path

    research_config_paths = sorted(glob(os.path.join(ckpt_dir, "config_*.yaml")))
    if len(research_config_paths) == 1:
        path = research_config_paths[0]
        return OmegaConf.load(path), path
    if len(research_config_paths) > 1:
        raise FileNotFoundError(
            "Multiple checkpoint training configurations were found. Set "
            "inference.training_config_path to the intended file."
        )
    raise FileNotFoundError(
        f"No checkpoint training configuration found in {ckpt_dir}. Expected "
        "config.yaml or one unambiguous config_*.yaml file."
    )


def _set_attention_qk_mode_from_checkpoint(state_dict, cfg: DictConfig) -> None:
    base_head_cfg = getattr(getattr(cfg, "model", None), "base_logit_head", None)
    if base_head_cfg is None or not bool(getattr(base_head_cfg, "use_attention_pool", False)):
        return

    q_key = "model.base_logit_attn_q.weight"
    k_key = "model.base_logit_attn_k.weight"
    if q_key not in state_dict or k_key not in state_dict:
        return

    q_weight = state_dict[q_key]
    k_weight = state_dict[k_key]
    if q_weight.shape != k_weight.shape or q_weight.ndim != 2:
        return

    num_heads = max(int(getattr(base_head_cfg, "num_attention_heads", 1)), 1)
    if int(q_weight.shape[0]) == num_heads:
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(base_head_cfg, False)
        base_head_cfg.attention_qk_mode = "scalar"
        log.warning(
            "Detected legacy scalar attention-pool q/k checkpoint weights "
            f"with shape {tuple(q_weight.shape)}; using attention_qk_mode=scalar."
        )


def _load_flow_module(ckpt_path: str, cfg: DictConfig) -> DuetRNAFlowModule:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    _set_attention_qk_mode_from_checkpoint(state_dict, cfg)
    module = DuetRNAFlowModule(cfg)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning(f"DuetRNA checkpoint missing keys: {len(missing)}")
    if unexpected:
        log.warning(f"DuetRNA checkpoint unexpected keys: {len(unexpected)}")
    use_ema_weights = bool(getattr(cfg.inference, "use_ema_weights", False))
    if use_ema_weights:
        ema_state = ckpt.get("ema_state_dict") if isinstance(ckpt, dict) else None
        if not ema_state:
            raise ValueError(
                "inference.use_ema_weights=true, but checkpoint does not contain `ema_state_dict`."
            )
        model_params = dict(module.model.named_parameters())
        copied = 0
        with torch.no_grad():
            for name, value in ema_state.items():
                param = model_params.get(name)
                if param is None:
                    continue
                param.copy_(value.to(device=param.device, dtype=param.dtype))
                copied += 1
        log.info(f"Loaded EMA weights for DuetRNA inference: {copied} tensors")
    return module


def _trainer_device_config(infer_cfg: DictConfig) -> tuple[str, int, str]:
    requested = int(infer_cfg.num_gpus)
    if not infer_cfg.use_gpu or requested <= 0:
        return "cpu", 1, "auto"
    visible = torch.cuda.device_count()
    if visible <= 0:
        raise RuntimeError(
            "No visible CUDA devices found but inference.use_gpu=true. "
            "Set `inference.use_gpu=false` for CPU-only smoke tests."
        )
    devices = min(requested, visible)
    return "gpu", devices, "ddp" if devices > 1 else "auto"


def _prepare_runtime_cfg(cfg: DictConfig) -> tuple[DictConfig, str]:
    ckpt_path = cfg.inference.ckpt_path
    ckpt_dir = os.path.dirname(ckpt_path)
    ckpt_cfg, ckpt_cfg_path = _load_ckpt_cfg(
        ckpt_dir,
        explicit_path=cfg.inference.get("training_config_path"),
    )

    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(ckpt_cfg, False)
    cfg = OmegaConf.merge(ckpt_cfg, cfg)
    cfg = apply_benchmark_preset(cfg)
    cfg.experiment.checkpointer.dirpath = "./"
    return cfg, ckpt_cfg_path


class DuetRNASampler:
    def __init__(self, cfg: DictConfig):
        cfg, ckpt_cfg_path = _prepare_runtime_cfg(cfg)
        ckpt_path = cfg.inference.ckpt_path

        self._cfg = cfg
        self._infer_cfg = cfg.inference
        self._samples_cfg = self._infer_cfg.samples
        self._output_dir = os.path.join(self._infer_cfg.output_dir, self._infer_cfg.name)
        os.makedirs(self._output_dir, exist_ok=True)

        merged_cfg_path = os.path.join(self._output_dir, "config.yaml")
        with open(merged_cfg_path, "w", encoding="utf-8") as handle:
            OmegaConf.save(config=self._cfg, f=handle)
        log.info(f"Loaded checkpoint config from {ckpt_cfg_path}")
        log.info(f"Saving inference config to {merged_cfg_path}")

        self._flow_module = _load_flow_module(ckpt_path=ckpt_path, cfg=cfg)
        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg
        self._flow_module._samples_cfg = self._samples_cfg
        self._flow_module._output_dir = self._output_dir

    @property
    def cfg(self) -> DictConfig:
        return self._cfg

    def run_sampling(self):
        accelerator, devices, strategy = _trainer_device_config(self._infer_cfg)

        eval_dataset = LengthDataset(self._samples_cfg)
        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        trainer = Trainer(accelerator=accelerator, strategy=strategy, devices=devices)
        start_time = time.time()
        predictions = trainer.predict(self._flow_module, dataloaders=dataloader)
        elapsed_time = time.time() - start_time
        log.info(f"Finished DuetRNA inference in {elapsed_time:.2f}s with {len(predictions)} samples")


@hydra.main(version_base=None, config_path="configs", config_name="inference")
def run(cfg: DictConfig) -> None:
    runtime_cfg = None
    if cfg.inference.run_inference:
        log.info(f"Starting DuetRNA inference with {cfg.inference.num_gpus} GPUs")
        sampler = DuetRNASampler(cfg)
        runtime_cfg = sampler.cfg
        sampler.run_sampling()
    elif cfg.inference.evalsuite.run_eval:
        runtime_cfg, _ = _prepare_runtime_cfg(cfg)

    eval_cfg = runtime_cfg if runtime_cfg is not None else cfg
    if eval_cfg.inference.evalsuite.run_eval:
        from benchmark.evalsuite import DuetRNAEvalSuite

        rna_samples_dir = f"{eval_cfg.inference.output_dir}/{eval_cfg.inference.name}"
        saving_dir = eval_cfg.inference.evalsuite.eval_save_dir
        gpu_id1, gpu_id2 = _evalsuite_gpu_ids(eval_cfg.inference)
        evalsuite = DuetRNAEvalSuite(
            save_dir=saving_dir,
            paths=eval_cfg.inference.evalsuite.paths,
            constants=eval_cfg.inference.evalsuite.constants,
            gpu_id1=gpu_id1,
            gpu_id2=gpu_id2,
            use_invfold=bool(eval_cfg.inference.evalsuite.get("use_invfold", True)),
            prefer_generated_fasta=bool(eval_cfg.inference.evalsuite.get("prefer_generated_fasta", False)),
        )
        evalsuite.perform_eval(rna_samples_dir, flatten_dir=True)
        metrics_fp = os.path.join(saving_dir, "final_metrics.pt")
        metric_dict = evalsuite.load_from_metric_dict(metrics_fp)
        evalsuite.print_metrics(metric_dict)


if __name__ == "__main__":
    run()
