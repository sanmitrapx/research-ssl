import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src.data.sonata_datamodule import SonataDataModule
from src.models.sonata_cp_classifier import SonataCpClassifier


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.get("seed", 42))

    print("Loading data...")
    dm = SonataDataModule(**cfg.data)

    print("Loading model...")
    model_cfg = dict(cfg.model)
    model_cfg["bin_centers"] = dm.bin_centers
    model = SonataCpClassifier(**model_cfg)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    trainer = pl.Trainer(**cfg.trainer)

    print("Starting training...")
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
