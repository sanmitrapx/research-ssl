"""Variant 1: Sub-bin regression.

Predicts bin classification + a per-point scalar offset within the
predicted bin, eliminating quantization error from discrete binning.

    cp_hat = soft_expectation + tanh(offset) * half_bin_width
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..sonata_cp_classifier import SonataCpClassifier, _ResidualBlock


class SonataCpSubBin(SonataCpClassifier):

    def __init__(self, lambda_offset: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        last_hidden = list(self.hparams.decoder_dims)[-1]
        self.offset_head = nn.Sequential(
            nn.Linear(last_hidden, last_hidden // 2),
            nn.ReLU(),
            nn.Linear(last_hidden // 2, 1),
        )

        bin_widths = torch.cat([
            self.bin_centers[1:] - self.bin_centers[:-1],
            (self.bin_centers[-1:] - self.bin_centers[-2:-1]),
        ])
        self.register_buffer("half_bin_widths", bin_widths / 2.0)

    def _build_head(self, input_dim, decoder_dims, num_bins, dropout):
        """Build shared backbone + classification head, expose penultimate features."""
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
        logits = self._cls_proj(hidden)
        probs = F.softmax(logits, dim=-1)
        soft_cp = (probs * self.bin_centers).sum(dim=-1)

        raw_offset = self.offset_head(hidden).squeeze(-1)
        pred_bins = logits.argmax(dim=-1)
        half_w = self.half_bin_widths[pred_bins]
        offset = torch.tanh(raw_offset) * half_w
        cp_hat = soft_cp + offset

        return dict(logits=logits, probs=probs, cp_hat=cp_hat, offset=offset)

    def _compute_loss(self, logits, target_bins, pressure_raw, cp_hat):
        cls_loss = super()._compute_loss(logits, target_bins, pressure_raw, cp_hat)
        offset_loss = F.smooth_l1_loss(cp_hat, pressure_raw)
        return cls_loss + self.hparams.lambda_offset * offset_loss
