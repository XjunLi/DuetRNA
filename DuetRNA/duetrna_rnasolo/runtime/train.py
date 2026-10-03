"""DuetRNA training entrypoint with the RNASolo configuration."""

import hydra
from omegaconf import DictConfig

from duetrna.runtime.train import DuetRNATrainer


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    trainer = DuetRNATrainer(cfg=cfg)
    trainer.train()


if __name__ == "__main__":
    main()
