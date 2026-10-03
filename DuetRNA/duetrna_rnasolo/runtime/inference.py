from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig

import duetrna_training as eu
from duetrna.runtime.inference import (
    DuetRNASampler,
    _evalsuite_gpu_ids,
    _prepare_runtime_cfg,
)


log = eu.get_pylogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def run(cfg: DictConfig) -> None:
    runtime_cfg = None
    if cfg.inference.run_inference:
        log.info(f"Starting DuetRNA RNASolo inference with {cfg.inference.num_gpus} GPUs")
        sampler = DuetRNASampler(cfg)
        runtime_cfg = sampler.cfg
        sampler.run_sampling()
    elif cfg.inference.evalsuite.run_eval:
        runtime_cfg, _ = _prepare_runtime_cfg(cfg)

    eval_cfg = runtime_cfg if runtime_cfg is not None else cfg
    if eval_cfg.inference.evalsuite.run_eval:
        from benchmark.evalsuite_duetrna import DuetRNAEvalSuite

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
