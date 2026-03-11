import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import sonata
from omegaconf import DictConfig


class SonataPointCloudModel(pl.LightningModule):
    """Sonata model with frozen encoder for pressure prediction"""

    def __init__(
        self,
        pretrained_repo: str = "facebook/sonata",
        freeze_encoder: bool = True,
        decoder_dims: list = [512, 256, 128, 64],
        output_dim: int = 1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained Sonata model with encoder-decoder configuration
        custom_config = dict(
            enc_mode=False,  # Enable full encoder-decoder mode
            freeze_encoder=freeze_encoder,  # Freeze encoder weights
            enable_flash=True,  # Use FlashAttention if available
        )

        # Load pretrained model
        self.sonata = sonata.model.load(
            "sonata", repo_id=pretrained_repo, custom_config=custom_config
        )

        # Freeze encoder parameters
        if freeze_encoder:
            self._freeze_encoder()

        # Add custom decoder head for pressure prediction
        self.pressure_head = self._build_pressure_head(
            decoder_dims, output_dim, dropout
        )

        self.loss_fn = nn.MSELoss()

    def _freeze_encoder(self):
        """Freeze encoder parameters"""
        for name, param in self.sonata.named_parameters():
            if "enc" in name or "encoder" in name:
                param.requires_grad = False

        # Log frozen parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {frozen_params:,}")

    def _build_pressure_head(self, decoder_dims, output_dim, dropout):
        """Build pressure prediction head"""
        layers = []

        # Input from Sonata decoder
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

        # Final prediction layer
        layers.append(nn.Linear(in_dim, output_dim))

        return nn.Sequential(*layers)

    def forward(self, point_dict):
        """Forward pass through Sonata + pressure head"""
        # Process through Sonata
        encoded_features = self.sonata(point_dict)

        # Extract features (Sonata returns a Point object)
        if hasattr(encoded_features, "feat"):
            features = encoded_features.feat
        else:
            features = encoded_features

        # Predict pressure
        pressure_pred = self.pressure_head(features)

        return pressure_pred

    def _prepare_batch(self, batch):
        """Prepare batch for Sonata model"""
        # Convert H5 batch format to Sonata's expected format
        point_dict = {
            "coord": batch["vertices"],  # [B, N, 3]
            "feat": torch.cat(
                [
                    batch["vertices"],  # Use coordinates as features
                    batch.get(
                        "normals", torch.zeros_like(batch["vertices"])
                    ),  # Add normals if available
                ],
                dim=-1,
            ),  # [B, N, 6]
        }

        # Add optional fields if available
        if "faces" in batch:
            point_dict["faces"] = batch["faces"]

        return point_dict

    def training_step(self, batch, batch_idx):
        # Prepare input for Sonata
        point_dict = self._prepare_batch(batch)

        # Get ground truth pressure
        pressure_gt = batch["pressure"]  # [B, N] or [B, N, 1]
        if pressure_gt.dim() == 2:
            pressure_gt = pressure_gt.unsqueeze(-1)

        # Forward pass
        pressure_pred = self.forward(point_dict)

        # Compute loss
        loss = self.loss_fn(pressure_pred, pressure_gt)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        point_dict = self._prepare_batch(batch)
        pressure_gt = batch["pressure"]
        if pressure_gt.dim() == 2:
            pressure_gt = pressure_gt.unsqueeze(-1)

        pressure_pred = self.forward(point_dict)
        loss = self.loss_fn(pressure_pred, pressure_gt)

        # Additional metrics
        mae = torch.mean(torch.abs(pressure_pred - pressure_gt))

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_mae", mae, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        # Only optimize decoder parameters
        decoder_params = [p for p in self.parameters() if p.requires_grad]

        optimizer = AdamW(
            decoder_params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        # Cosine annealing with warmup
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
        """Learning rate warmup"""
        if self.current_epoch < self.hparams.warmup_epochs:
            # Linear warmup
            lr_scale = (self.current_epoch + 1) / self.hparams.warmup_epochs
            for pg in self.trainer.optimizers[0].param_groups:
                pg["lr"] = self.hparams.learning_rate * lr_scale
