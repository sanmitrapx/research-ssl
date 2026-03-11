"""Variant 3: Multi-scale classification.

Two classification heads at different bin resolutions:
  - Coarse (num_cp_bins_coarse, default 32): captures global Cp structure
  - Fine   (num_cp_bins):                    captures local precision

Combined loss trains both; inference uses the fine head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .sonata_cp_classifier import SonataCpClassifier, _ResidualBlock


class SonataCpMultiScale(SonataCpClassifier):

    def __init__(self, num_cp_bins_coarse: int = 32, lambda_coarse: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        K_coarse = num_cp_bins_coarse
        coarse_centers = torch.linspace(
            self.bin_centers[0].item(),
            self.bin_centers[-1].item(),
            K_coarse,
        )
        self.register_buffer("coarse_bin_centers", coarse_centers)

        last_hidden = list(self.hparams.decoder_dims)[-1]
        self.coarse_head = nn.Sequential(
            nn.Linear(last_hidden, last_hidden // 2),
            nn.ReLU(),
            nn.Linear(last_hidden // 2, K_coarse),
        )
        self.coarse_ce = nn.CrossEntropyLoss()

    def _build_head(self, input_dim, decoder_dims, num_bins, dropout):
        layers = []
        in_dim = input_dim
        for i, dim in enumerate(decoder_dims):
            if dim == in_dim and i > 0:
                layers.extend([_ResidualBlock(dim, dropout), nn.ReLU(), nn.Dropout(dropout)])
            else:
                layers.extend([nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = dim
        self._backbone = nn.Sequential(*layers)
        self._cls_proj = nn.Linear(in_dim, num_bins)
        return nn.Sequential(self._backbone, self._cls_proj)

    def forward(self, point_data):
        encoded = self.sonata(point_data)

        encoded = self._upcast_features(encoded, self.hparams.num_concat_levels)
        feat = encoded.feat
        feat = feat[point_data["inverse"]]

        if self.hparams.use_geometric_features:
            geo = torch.cat([
                point_data["uncentered_coord"],
                point_data["untransformed_normal"],
                point_data["untransformed_deltas"],
            ], dim=-1)
            feat = torch.cat([feat, geo], dim=-1)

        hidden = self._backbone(feat)
        logits_fine = self._cls_proj(hidden)
        logits_coarse = self.coarse_head(hidden)

        probs = F.softmax(logits_fine, dim=-1)
        cp_hat = (probs * self.bin_centers).sum(dim=-1)

        return dict(
            logits=logits_fine,
            logits_coarse=logits_coarse,
            probs=probs,
            cp_hat=cp_hat,
        )

    def _compute_loss(self, logits, target_bins, pressure_raw, cp_hat,
                      logits_coarse=None, target_bins_coarse=None):
        fine_loss = super()._compute_loss(logits, target_bins, pressure_raw, cp_hat)

        if logits_coarse is not None and target_bins_coarse is not None:
            coarse_loss = self.coarse_ce(logits_coarse, target_bins_coarse)
            return fine_loss + self.hparams.lambda_coarse * coarse_loss
        return fine_loss

    def _get_coarse_bins(self, pressure_raw):
        """Map raw Cp values to coarse bin indices."""
        centers = self.coarse_bin_centers
        dists = (pressure_raw.unsqueeze(-1) - centers.unsqueeze(0)).abs()
        return dists.argmin(dim=-1)

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        target_bins_coarse = self._get_coarse_bins(batch["pressure_raw"])
        loss = self._compute_loss(
            out["logits"], batch["pressure_bin"], batch["pressure_raw"], out["cp_hat"],
            logits_coarse=out["logits_coarse"],
            target_bins_coarse=target_bins_coarse,
        )
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        target_bins = batch["pressure_bin"]
        pressure_raw = batch["pressure_raw"]
        target_bins_coarse = self._get_coarse_bins(pressure_raw)
        loss = self._compute_loss(
            out["logits"], target_bins, pressure_raw, out["cp_hat"],
            logits_coarse=out["logits_coarse"],
            target_bins_coarse=target_bins_coarse,
        )
        pred_bins = out["logits"].argmax(dim=-1)
        mbe = (pred_bins - target_bins).float().abs().mean()
        rl2 = self.rl2_loss(out["cp_hat"].unsqueeze(0), pressure_raw.unsqueeze(0))

        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log("val_mbe", mbe, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log("val_rl2", rl2, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return loss
