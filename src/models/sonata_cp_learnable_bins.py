"""Variant 2: Learnable bin centers.

Bin centers are nn.Parameters that shift during training to concentrate
resolution where the model needs it most (stagnation, suction peaks).
A regularisation term keeps bins ordered and prevents collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .sonata_cp_classifier import SonataCpClassifier


class SonataCpLearnableBins(SonataCpClassifier):

    def __init__(self, bin_spread_reg: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # Promote buffer to parameter so optimizer can update it
        centers = self.bin_centers.data.clone()
        del self._buffers["bin_centers"]
        self.bin_centers_param = nn.Parameter(centers)

    @property
    def _sorted_centers(self):
        """Return sorted centers to guarantee monotonicity."""
        return torch.sort(self.bin_centers_param)[0]

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

        logits = self.cp_classifier_head(feat)
        probs = F.softmax(logits, dim=-1)
        centers = self._sorted_centers
        cp_hat = (probs * centers).sum(dim=-1)

        return dict(logits=logits, probs=probs, cp_hat=cp_hat)

    def _compute_loss(self, logits, target_bins, pressure_raw, cp_hat):
        base_loss = super()._compute_loss(logits, target_bins, pressure_raw, cp_hat)

        # Regularise: penalise bins collapsing together
        centers = self._sorted_centers
        diffs = centers[1:] - centers[:-1]
        collapse_penalty = torch.relu(0.001 - diffs).sum()

        return base_loss + self.hparams.bin_spread_reg * collapse_penalty

    def configure_optimizers(self):
        optimizers, schedulers = super().configure_optimizers()
        # Add bin centers to the head param group (highest LR)
        for pg in optimizers[0].param_groups:
            if any(p is self.bin_centers_param for p in pg["params"]):
                return optimizers, schedulers
        optimizers[0].add_param_group({
            "params": [self.bin_centers_param],
            "lr": self.hparams.head_lr,
        })
        return optimizers, schedulers
