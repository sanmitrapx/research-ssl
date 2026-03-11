import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src.data.sonata_datamodule import SonataDataModule


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.get("seed", 42))

    print("Loading data...")
    data_cfg = {k: v for k, v in cfg.data.items() if not k.startswith("_")}
    dm = SonataDataModule(**data_cfg)

    print("Loading model...")
    model_target = cfg.model.get("_target_", "src.models.sonata_cp_classifier.SonataCpClassifier")
    print(f"  Model class: {model_target}")
    model = hydra.utils.instantiate(cfg.model, bin_centers=dm.bin_centers, _convert_="partial")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    callbacks = []
    if "callbacks" in cfg.trainer:
        for cb_cfg in cfg.trainer.callbacks:
            callbacks.append(hydra.utils.instantiate(cb_cfg))

    logger = None
    if "logger" in cfg.trainer:
        logger = hydra.utils.instantiate(cfg.trainer.logger)

    trainer_cfg = {
        k: v for k, v in cfg.trainer.items()
        if not k.startswith("_") and k not in ("callbacks", "logger")
    }
    trainer = pl.Trainer(**trainer_cfg, callbacks=callbacks, logger=logger)

    print("Starting training...")
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
