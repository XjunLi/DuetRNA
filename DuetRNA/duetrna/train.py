"""DuetRNA training entrypoint."""

import os
from pathlib import Path
import sys

# Resolve local packages for both direct-file and module execution.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import GPUtil
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, Timer
from pytorch_lightning.loggers.wandb import WandbLogger

from duetrna import training as eu
from duetrna.training import NanGradientCallback
from duetrna.data.dataset import PDBNABaseDataModule
from duetrna.models.flow_module import DuetRNAFlowModule


log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class FinalStepModelCheckpoint(ModelCheckpoint):
    """Write one optimizer-resumable checkpoint when training ends normally."""

    CHECKPOINT_NAME_LAST = "final-step-{step:08d}"

    def __init__(self, *, required_completed_epochs: int, **kwargs):
        super().__init__(**kwargs)
        self.required_completed_epochs = int(required_completed_epochs)

    def on_exception(self, trainer, pl_module, exception):
        del trainer, pl_module, exception

    def on_train_end(self, trainer, pl_module):
        completed_epochs = int(trainer.fit_loop.epoch_progress.current.completed)
        if completed_epochs != self.required_completed_epochs:
            log.warning(
                "Not writing the final snapshot: completed %d/%d epochs.",
                completed_epochs,
                self.required_completed_epochs,
            )
            return
        super().on_train_end(trainer, pl_module)


def _is_external_distributed_child() -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return (
        world_size > 1
        and "LOCAL_RANK" in os.environ
        and os.environ.get("DUETRNA_EXTERNAL_TORCHRUN") == "1"
    )


class DuetRNATrainer:
    def __init__(self, *, cfg: DictConfig):
        seed_everything(int(cfg.experiment.seed), workers=True)
        self._cfg = cfg
        self._data_cfg = cfg.data_cfg
        self._exp_cfg = cfg.experiment
        self._model = DuetRNAFlowModule(self._cfg)
        self._datamodule = PDBNABaseDataModule(data_cfg=self._data_cfg)

    def train(self):
        callbacks = []
        trainer_kwargs = OmegaConf.to_container(self._exp_cfg.trainer, resolve=True)

        if self._exp_cfg.debug:
            logger = None
            self._exp_cfg.num_devices = 1
            self._data_cfg.num_workers = 0
        else:
            logger = WandbLogger(**self._exp_cfg.wandb)
            checkpointer_cfg = OmegaConf.to_container(self._exp_cfg.checkpointer, resolve=True)
            snapshot_every_n_steps = int(checkpointer_cfg.pop("step_snapshot_every_n_train_steps", 0) or 0)
            snapshot_dirname = str(checkpointer_cfg.pop("step_snapshot_dirname", "step_snapshots"))
            save_final_on_train_end = bool(checkpointer_cfg.pop("save_final_on_train_end", False))
            final_snapshot_dirname = str(
                checkpointer_cfg.pop("final_snapshot_dirname", "final_snapshot")
            )
            ckpt_dir = checkpointer_cfg["dirpath"]
            os.makedirs(ckpt_dir, exist_ok=True)
            callbacks.append(ModelCheckpoint(**checkpointer_cfg))
            if snapshot_every_n_steps > 0:
                callbacks.append(
                    ModelCheckpoint(
                        dirpath=os.path.join(ckpt_dir, snapshot_dirname),
                        filename="step-{step:08d}",
                        monitor=None,
                        save_top_k=-1,
                        save_last=False,
                        every_n_train_steps=snapshot_every_n_steps,
                        every_n_epochs=None,
                        auto_insert_metric_name=False,
                    )
                )
            if save_final_on_train_end:
                required_completed_epochs = int(trainer_kwargs["max_epochs"])
                if required_completed_epochs <= 0:
                    raise ValueError(
                        "save_final_on_train_end requires a positive trainer.max_epochs"
                    )
                callbacks.append(
                    FinalStepModelCheckpoint(
                        required_completed_epochs=required_completed_epochs,
                        dirpath=os.path.join(ckpt_dir, final_snapshot_dirname),
                        monitor=None,
                        save_top_k=0,
                        save_last=True,
                        save_on_exception=False,
                        every_n_epochs=0,
                        every_n_train_steps=None,
                        save_on_train_epoch_end=True,
                        auto_insert_metric_name=False,
                    )
                )
            callbacks.append(Timer(duration=self._exp_cfg.trainer.max_time))
            callbacks.append(
                NanGradientCallback(
                    fail_on_nonfinite=bool(
                        getattr(self._exp_cfg.training, "fail_on_nonfinite_gradients", True)
                    )
                )
            )
            cfg_path = os.path.join(ckpt_dir, "config.yaml")
            with open(cfg_path, "w", encoding="utf-8") as handle:
                OmegaConf.save(config=self._cfg, f=handle.name)
            cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
            flat_cfg = dict(eu.flatten_dict(cfg_dict))
            if isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
                logger.experiment.config.update(flat_cfg)

        accelerator = str(self._exp_cfg.trainer.accelerator).lower()
        if accelerator == "cpu":
            devices = 1 if self._exp_cfg.num_devices is None else int(self._exp_cfg.num_devices)
            trainer_kwargs["strategy"] = "auto"
        else:
            requested_devices = 1 if self._exp_cfg.num_devices is None else int(self._exp_cfg.num_devices)
            if _is_external_distributed_child() and requested_devices > 1:
                devices = 1
            elif requested_devices <= 1:
                try:
                    devices = GPUtil.getAvailable(order="memory", limit=8)[:requested_devices]
                except Exception:
                    devices = 1 if torch.cuda.is_available() else []
                if isinstance(devices, list) and len(devices) == 0:
                    if torch.cuda.is_available():
                        devices = 1
                    else:
                        raise RuntimeError(
                            "No GPU device available but trainer accelerator is set to GPU. "
                            "Set `experiment.trainer.accelerator=cpu` for CPU debug, or run on a GPU node."
                        )
                trainer_kwargs["strategy"] = "auto"
            else:
                devices = requested_devices

        trainer = Trainer(
            **trainer_kwargs,
            callbacks=callbacks,
            logger=logger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=devices,
        )
        trainer.fit(
            model=self._model,
            datamodule=self._datamodule,
            ckpt_path=self._exp_cfg.warm_start,
        )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    trainer = DuetRNATrainer(cfg=cfg)
    trainer.train()


if __name__ == "__main__":
    main()
