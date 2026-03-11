import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import sonata
from sonata.model import PointTransformerV3


class SonataFineTuneModel(pl.LightningModule):
    """Sonata model with encoder finetuning for pressure prediction"""

    def __init__(
        self,
        pretrained_repo: str = "facebook/sonata",
        freeze_encoder: bool = False,  # Set to False for finetuning
        encoder_lr: float = 1e-5,  # Lower LR for pretrained encoder
        decoder_lr: float = 1e-4,  # Higher LR for new decoder
        sonata_dim: int = 512,
        decoder_dims: list = [512, 256, 128, 64],
        output_dim: int = 1,
        weight_decay: float = 1e-2,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        custom_config = dict(
            enc_mode=False,  # Keep decoder
            enable_flash=False,
        )

        model = PointTransformerV3.from_pretrained(
            "facebook/sonata",
            strict=False,  # ← Add this to ignore missing keys
            **custom_config,
        )

        # Load pretrained Sonata model
        custom_config = dict(
            enc_mode=False,  # Full encoder-decoder mode
            freeze_encoder=False,  # Enable encoder finetuning
            enable_flash=True,
        )

        self.sonata = sonata.model.load(
            "sonata", repo_id=pretrained_repo, custom_config=custom_config
        )

        # Add custom decoder head for pressure prediction
        self.pressure_head = self._build_pressure_head(
            decoder_dims, output_dim, dropout
        )

        self.loss_fn = nn.MSELoss()

    def _build_pressure_head(self, decoder_dims, output_dim, dropout):
        """Build pressure prediction head"""
        layers = []
        in_dim = 512  # Sonata's output dimension

        for dim in decoder_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = dim

        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    def configure_optimizers(self):
        """Configure optimizers with different learning rates for encoder and decoder"""

        # Separate encoder and decoder parameters
        encoder_params = []
        decoder_params = []

        for name, param in self.named_parameters():
            if "sonata" in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)

        # Create parameter groups with different learning rates
        param_groups = [
            {
                "params": encoder_params,
                "lr": self.hparams.encoder_lr,
                "name": "encoder",
            },
            {
                "params": decoder_params,
                "lr": self.hparams.decoder_lr,
                "name": "decoder",
            },
        ]

        optimizer = AdamW(param_groups, weight_decay=self.hparams.weight_decay)

        # Learning rate scheduler with warmup
        scheduler = {
            "scheduler": CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
                eta_min=1e-6,
            ),
            "interval": "epoch",
        }

        return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        """Implement gradual unfreezing (optional)"""
        if self.current_epoch < self.hparams.warmup_epochs:
            # Optionally freeze encoder during initial warmup
            for name, param in self.sonata.named_parameters():
                param.requires_grad = False
        else:
            # Unfreeze encoder after warmup
            for name, param in self.sonata.named_parameters():
                param.requires_grad = True
