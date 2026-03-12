"""Plain regression with rL2 loss.

Directly predicts continuous Cp values using a residual MLP head.
No binning or classification -- the output is a single scalar per point
and the loss is the relative L2 norm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW

import sonata

from ..utils.losses import LpLoss


class _ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.drop(F.relu(self.fc1(x)))
        out = self.fc2(out)
        return x + out


class SonataCpRegression(pl.LightningModule):
    """Sonata encoder + residual MLP for direct Cp regression with rL2 loss."""

    DEFAULT_UPCAST_DIM = 1088

    def __init__(
        self,
        sonata_repo: str = "facebook/sonata",
        decoder_dims: list = (512, 512, 256, 256),
        upcast_dim: int = 1088,
        num_concat_levels: int = 2,
        use_geometric_features: bool = False,
        learning_rate: float = 0.0001,
        head_lr: float = 0.001,
        weight_decay: float = 0.05,
        lr_decay_rate: float = 0.65,
        max_epochs: int = 100,
        eta_min: float = 1e-6,
        dropout: float = 0.1,
        cp_mean: float = 0.0,
        cp_std: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        kwargs.clear()
        self.save_hyperparameters()

        self.register_buffer("cp_mean", torch.tensor(cp_mean, dtype=torch.float32))
        self.register_buffer("cp_std", torch.tensor(cp_std, dtype=torch.float32))

        custom_config = dict(enc_mode=True, enable_flash=True)
        self.sonata = sonata.model.load(
            "sonata", repo_id=sonata_repo, custom_config=custom_config
        )

        point_feat_dim = upcast_dim
        if use_geometric_features:
            point_feat_dim += 9

        self.regression_head = self._build_head(
            point_feat_dim, list(decoder_dims), dropout
        )

        self.rl2_loss = LpLoss()

    def _build_head(self, input_dim, decoder_dims, dropout):
        layers = []
        in_dim = input_dim

        for i, dim in enumerate(decoder_dims):
            if dim == in_dim and i > 0:
                layers.extend([
                    _ResidualBlock(dim, dropout),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
            else:
                layers.extend([
                    nn.Linear(in_dim, dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
            in_dim = dim

        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    @staticmethod
    def _upcast_features(point, num_concat_levels):
        for _ in range(num_concat_levels):
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat(
                [parent.feat, point.feat[inverse]], dim=-1
            )
            point = parent
        return point

    def forward(self, point_data):
        encoded = self.sonata(point_data)
        encoded = self._upcast_features(encoded, self.hparams.num_concat_levels)
        feat = encoded.feat

        inverse = point_data["inverse"]
        feat = feat[inverse]

        if self.hparams.use_geometric_features:
            geo = torch.cat([
                point_data["uncentered_coord"],
                point_data["untransformed_normal"],
                point_data["untransformed_deltas"],
            ], dim=-1)
            feat = torch.cat([feat, geo], dim=-1)

        cp_hat = self.regression_head(feat).squeeze(-1)
        return dict(cp_hat=cp_hat)

    def _denormalize(self, cp_standardized):
        return cp_standardized * self.cp_std + self.cp_mean

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        cp_hat_std = out["cp_hat"]
        target_std = batch["pressure"]
        loss = self.rl2_loss(cp_hat_std.unsqueeze(0), target_std.unsqueeze(0))
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        cp_hat_std = out["cp_hat"]
        target_std = batch["pressure"]

        train_loss = self.rl2_loss(cp_hat_std.unsqueeze(0), target_std.unsqueeze(0))

        cp_hat_raw = self._denormalize(cp_hat_std)
        pressure_raw = batch["pressure_raw"]
        rl2 = self.rl2_loss(cp_hat_raw.unsqueeze(0), pressure_raw.unsqueeze(0))
        mae = (cp_hat_raw - pressure_raw).abs().mean()

        self.log("val_loss", train_loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log("val_rl2", rl2, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log("val_mae", mae, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return train_loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def _get_encoder_param_groups(self):
        lr = self.hparams.learning_rate
        decay = self.hparams.lr_decay_rate
        num_stages = 5

        groups = []
        emb_params = list(self.sonata.embedding.parameters())
        groups.append({"params": emb_params, "lr": lr * (decay ** num_stages)})

        for s in range(num_stages):
            stage = getattr(self.sonata.enc, f"enc{s}", None)
            if stage is not None:
                stage_params = list(stage.parameters())
                stage_lr = lr * (decay ** (num_stages - 1 - s))
                groups.append({"params": stage_params, "lr": stage_lr})

        return groups

    def configure_optimizers(self):
        head_lr = self.hparams.head_lr
        wd = self.hparams.weight_decay

        param_groups = self._get_encoder_param_groups()
        param_groups.append({
            "params": self.regression_head.parameters(),
            "lr": head_lr,
        })

        optimizer = AdamW(param_groups, weight_decay=wd)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.eta_min,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
