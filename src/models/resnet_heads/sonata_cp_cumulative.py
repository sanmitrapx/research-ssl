"""Variant 6: Cumulative ordinal regression (CORAL-style).

Instead of predicting K-class probabilities, the model predicts K-1
cumulative binary indicators: P(Cp > c_k) for each threshold k.
This enforces ordinal consistency by construction and produces
well-calibrated probabilities that respect bin ordering.

    logit_k = shared_feat @ w + b_k   (shared weights, per-bin bias)

Reference: Cao et al., "Rank Consistent Ordinal Regression for Neural
Networks with Application to Age Estimation", Pattern Recognition 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..sonata_cp_classifier import SonataCpClassifier, _ResidualBlock


class CumulativeOrdinalHead(nn.Module):
    """CORAL head: shared linear + K-1 bias terms."""

    def __init__(self, input_dim, num_bins):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(num_bins - 1))

    def forward(self, x):
        """Returns (N, K-1) cumulative logits."""
        shared = self.fc(x)  # (N, 1)
        return shared + self.biases  # broadcast: (N, K-1)


class SonataCpCumulative(SonataCpClassifier):

    def __init__(self, lambda_rank_consistency: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        K = len(self.bin_centers)
        last_hidden = list(self.hparams.decoder_dims)[-1]
        self.cumulative_head = CumulativeOrdinalHead(last_hidden, K)

    def _build_head(self, input_dim, decoder_dims, num_bins, dropout):
        """Build shared backbone (no final classification layer -- replaced by cumulative_head)."""
        layers = []
        in_dim = input_dim
        for i, dim in enumerate(decoder_dims):
            if dim == in_dim and i > 0:
                layers.extend([_ResidualBlock(dim, dropout), nn.ReLU(), nn.Dropout(dropout)])
            else:
                layers.extend([nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = dim

        self._backbone = nn.Sequential(*layers)
        # Keep a dummy final proj so parent __init__ doesn't break
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
        cum_logits = self.cumulative_head(hidden)  # (N, K-1)

        # Convert cumulative probabilities to per-bin probabilities
        cum_probs = torch.sigmoid(cum_logits)
        # P(bin=0) = 1 - cum_probs[:, 0]
        # P(bin=k) = cum_probs[:, k-1] - cum_probs[:, k]   for 0 < k < K-1
        # P(bin=K-1) = cum_probs[:, K-2]
        ones = torch.ones(cum_probs.shape[0], 1, device=cum_probs.device)
        zeros = torch.zeros(cum_probs.shape[0], 1, device=cum_probs.device)
        extended = torch.cat([ones, cum_probs, zeros], dim=-1)  # (N, K+1)
        probs = extended[:, :-1] - extended[:, 1:]  # (N, K)
        probs = probs.clamp(min=1e-7)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        cp_hat = (probs * self.bin_centers).sum(dim=-1)

        # Build standard logits for metrics compatibility
        logits = torch.log(probs + 1e-8)

        return dict(
            logits=logits,
            cum_logits=cum_logits,
            probs=probs,
            cp_hat=cp_hat,
        )

    def _compute_loss(self, logits, target_bins, pressure_raw, cp_hat,
                      cum_logits=None):
        K = len(self.bin_centers)

        if cum_logits is not None:
            # CORAL binary CE: for each threshold k, target is 1 if bin > k
            targets_cum = (
                target_bins.unsqueeze(-1)
                > torch.arange(K - 1, device=target_bins.device).unsqueeze(0)
            ).float()
            coral_loss = F.binary_cross_entropy_with_logits(
                cum_logits, targets_cum, reduction="mean"
            )

            # Rank consistency: penalise non-monotone cumulative probs
            cum_probs = torch.sigmoid(cum_logits)
            rank_violations = F.relu(cum_probs[:, 1:] - cum_probs[:, :-1])
            rank_loss = rank_violations.mean()

            return coral_loss + self.hparams.lambda_rank_consistency * rank_loss

        return super()._compute_loss(logits, target_bins, pressure_raw, cp_hat)

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = self._compute_loss(
            out["logits"], batch["pressure_bin"], batch["pressure_raw"], out["cp_hat"],
            cum_logits=out["cum_logits"],
        )
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        target_bins = batch["pressure_bin"]
        pressure_raw = batch["pressure_raw"]
        loss = self._compute_loss(
            out["logits"], target_bins, pressure_raw, out["cp_hat"],
            cum_logits=out["cum_logits"],
        )
        pred_bins = out["logits"].argmax(dim=-1)
        mbe = (pred_bins - target_bins).float().abs().mean()
        rl2 = self.rl2_loss(out["cp_hat"].unsqueeze(0), pressure_raw.unsqueeze(0))

        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log("val_mbe", mbe, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log("val_rl2", rl2, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return loss
