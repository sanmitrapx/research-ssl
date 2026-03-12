"""Learnable anisotropic diffusion head on KNN graph.

Features are iteratively smoothed over the surface via a multi-step
diffusion process.  At each step, position-dependent edge weights
determine how much information flows between neighbours, and a
learned gate controls the update magnitude.  This encourages
spatially coherent Cp predictions while preserving sharp transitions
where the data supports them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sonata_cp_classifier import SonataCpClassifier
from ..utils.knn import chunked_knn


class DiffusionBlock(nn.Module):
    """Multi-step gated diffusion with position-aware edge weights."""

    def __init__(self, dim, k=16, num_steps=5):
        super().__init__()
        self.k = k
        self.num_steps = num_steps

        self.edge_weight_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.Sigmoid(),
            )
            for _ in range(num_steps)
        ])

        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_steps)])

    def forward(self, feat, coords, knn_idx):
        """
        feat:    (N, C)
        coords:  (N, 3)
        knn_idx: (N, k)
        """
        rel_pos = coords[knn_idx] - coords.unsqueeze(1)    # (N, k, 3)
        edge_w = self.edge_weight_net(rel_pos).squeeze(-1)  # (N, k)
        edge_w = F.softmax(edge_w, dim=-1)                  # (N, k)

        for t in range(self.num_steps):
            neighbor_feat = feat[knn_idx]                    # (N, k, C)
            agg = (edge_w.unsqueeze(-1) * neighbor_feat).sum(dim=1)

            gate = self.gates[t](torch.cat([feat, agg], dim=-1))
            feat = self.norms[t](feat + gate * (agg - feat))

        return feat


class SonataCpDiffusion(SonataCpClassifier):
    """Sonata encoder + learnable diffusion smoothing + MLP classifier."""

    def __init__(
        self,
        diff_k: int = 16,
        diff_steps: int = 5,
        diff_dim: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        upcast_dim = self.hparams.upcast_dim
        self.diff_proj = nn.Linear(upcast_dim, diff_dim)
        self.diffusion = DiffusionBlock(diff_dim, k=diff_k, num_steps=diff_steps)

        K = len(self.bin_centers)
        self.cp_classifier_head = self._build_head(
            diff_dim, [diff_dim, diff_dim], K, self.hparams.dropout,
        )

    def _get_extra_param_groups(self):
        params = list(self.diff_proj.parameters()) + list(self.diffusion.parameters())
        return [{"params": params, "lr": self.hparams.head_lr}]

    def forward(self, point_data):
        encoded = self._encode(point_data)
        feat = encoded.feat
        coord = encoded.coord

        feat = self.diff_proj(feat)

        knn_idx = chunked_knn(coord, k=self.hparams.diff_k)

        feat = self.diffusion(feat, coord, knn_idx)

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
        cp_hat = (probs * self.bin_centers).sum(dim=-1)

        return dict(logits=logits, probs=probs, cp_hat=cp_hat)
