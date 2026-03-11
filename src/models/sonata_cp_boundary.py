"""Variant 5: Boundary-aware loss weighting.

Points near Cp discontinuities (suction peaks, separation points)
are harder to predict and carry disproportionate importance for
aerodynamic accuracy.  This variant detects boundary regions by
computing the local Cp gradient magnitude via KNN, then upweights
the loss at those points.

    w_i = 1 + alpha * |grad_Cp|_i / max(|grad_Cp|)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .sonata_cp_classifier import SonataCpClassifier
from .utils.losses import EMDLoss


class SonataCpBoundary(SonataCpClassifier):

    def __init__(
        self,
        boundary_alpha: float = 2.0,
        boundary_k: int = 16,
        boundary_max_points: int = 80000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        K = len(self.bin_centers)
        self.weighted_ce = nn.CrossEntropyLoss(reduction="none")
        widths = torch.cat([
            self.bin_centers[1:] - self.bin_centers[:-1],
            (self.bin_centers[-1:] - self.bin_centers[-2:-1]),
        ])
        widths = widths / widths.sum()
        self.weighted_emd = EMDLoss(K, reduction="none", bin_widths=widths)

    def _compute_gradient_weights(self, coords, cp_values):
        """Compute per-point weight based on local Cp gradient magnitude."""
        N = coords.shape[0]
        max_pts = self.hparams.boundary_max_points
        k = self.hparams.boundary_k

        if N > max_pts:
            perm = torch.randperm(N, device=coords.device)[:max_pts]
            sub_coords = coords[perm]
            sub_cp = cp_values[perm]

            dists = torch.cdist(sub_coords, sub_coords)
            _, knn_idx = dists.topk(k + 1, dim=-1, largest=False)
            knn_idx = knn_idx[:, 1:]  # exclude self

            cp_neighbours = sub_cp[knn_idx]
            grad_mag_sub = (cp_neighbours - sub_cp.unsqueeze(-1)).abs().mean(dim=-1)

            # Map back to full resolution
            full_dists = torch.cdist(coords, sub_coords)
            nearest = full_dists.argmin(dim=-1)
            grad_mag = grad_mag_sub[nearest]
        else:
            dists = torch.cdist(coords, coords)
            _, knn_idx = dists.topk(k + 1, dim=-1, largest=False)
            knn_idx = knn_idx[:, 1:]

            cp_neighbours = cp_values[knn_idx]
            grad_mag = (cp_neighbours - cp_values.unsqueeze(-1)).abs().mean(dim=-1)

        grad_max = grad_mag.max().clamp(min=1e-6)
        weights = 1.0 + self.hparams.boundary_alpha * (grad_mag / grad_max)
        return weights

    def _compute_loss(self, logits, target_bins, pressure_raw, cp_hat,
                      coords=None):
        if coords is not None:
            weights = self._compute_gradient_weights(coords, pressure_raw)
        else:
            weights = torch.ones(logits.shape[0], device=logits.device)

        if self.hparams.loss_type == "ordinal_kl":
            loss = self._ordinal_kl_loss_weighted(logits, target_bins, weights)
        else:
            ce = self.weighted_ce(logits, target_bins)
            loss = (ce * weights).mean()
            if self.hparams.lambda_emd > 0:
                emd = self.weighted_emd(logits, target_bins)
                loss = loss + self.hparams.lambda_emd * (emd * weights).mean()

        return loss

    def _ordinal_kl_loss_weighted(self, logits, target_bins, weights):
        K = logits.shape[-1]
        sigma = self.hparams.label_smoothing_sigma
        bins = torch.arange(K, device=logits.device, dtype=torch.float32)
        diff = bins.unsqueeze(0) - target_bins.float().unsqueeze(-1)
        soft_targets = torch.exp(-0.5 * (diff / sigma) ** 2)
        soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)
        log_probs = F.log_softmax(logits, dim=-1)
        per_point = -(soft_targets * log_probs).sum(dim=-1)
        return (per_point * weights).mean()

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = self._compute_loss(
            out["logits"], batch["pressure_bin"], batch["pressure_raw"], out["cp_hat"],
            coords=batch.get("uncentered_coord"),
        )
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        target_bins = batch["pressure_bin"]
        pressure_raw = batch["pressure_raw"]
        loss = self._compute_loss(
            out["logits"], target_bins, pressure_raw, out["cp_hat"],
            coords=batch.get("uncentered_coord"),
        )
        pred_bins = out["logits"].argmax(dim=-1)
        mbe = (pred_bins - target_bins).float().abs().mean()
        rl2 = self.rl2_loss(out["cp_hat"].unsqueeze(0), pressure_raw.unsqueeze(0))

        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log("val_mbe", mbe, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log("val_rl2", rl2, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return loss
