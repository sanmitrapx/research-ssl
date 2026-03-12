"""KNN Graph Convolution head (EdgeConv / DGCNN-style).

Message passing at grid-sampled resolution using KNN graph.
Each EdgeConv layer computes edge features h(x_i, x_j - x_i),
max-pools over k neighbours, and adds a residual skip.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sonata_cp_classifier import SonataCpClassifier
from ..utils.knn import chunked_knn


class EdgeConvBlock(nn.Module):
    """Single EdgeConv layer with residual connection."""

    def __init__(self, in_dim, out_dim, k=16):
        super().__init__()
        self.k = k
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, feat, knn_idx):
        """
        feat: (N, C_in)
        knn_idx: (N, k)
        Returns: (N, C_out)
        """
        neighbor_feat = feat[knn_idx]                          # (N, k, C)
        center_feat = feat.unsqueeze(1).expand_as(neighbor_feat)
        edge_feat = torch.cat([center_feat, neighbor_feat - center_feat], dim=-1)

        N, k, C2 = edge_feat.shape
        edge_out = self.edge_mlp(edge_feat.view(N * k, C2)).view(N, k, -1)
        agg = edge_out.max(dim=1)[0]                           # (N, C_out)

        return self.norm(agg + self.skip(feat))


class SonataCpGCN(SonataCpClassifier):
    """Sonata encoder + EdgeConv message passing + MLP classifier."""

    def __init__(
        self,
        gcn_k: int = 16,
        gcn_layers: int = 3,
        gcn_dim: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        upcast_dim = self.hparams.upcast_dim
        self.mp_layers = nn.ModuleList()
        in_dim = upcast_dim
        for _ in range(gcn_layers):
            self.mp_layers.append(EdgeConvBlock(in_dim, gcn_dim, k=gcn_k))
            in_dim = gcn_dim

        K = len(self.bin_centers)
        self.cp_classifier_head = self._build_head(
            gcn_dim, [gcn_dim, gcn_dim], K, self.hparams.dropout,
        )

    def _get_extra_param_groups(self):
        return [{"params": self.mp_layers.parameters(), "lr": self.hparams.head_lr}]

    def forward(self, point_data):
        encoded = self._encode(point_data)
        feat = encoded.feat
        coord = encoded.coord

        knn_idx = chunked_knn(coord, k=self.hparams.gcn_k)

        for layer in self.mp_layers:
            feat = layer(feat, knn_idx)

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
