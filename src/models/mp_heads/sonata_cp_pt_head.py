"""Point Transformer head – local multi-head self-attention at grid resolution.

After Sonata encodes features, lightweight Point Transformer blocks
refine them using KNN-based local attention with relative position
encoding before the final classifier MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sonata_cp_classifier import SonataCpClassifier
from ..utils.knn import chunked_knn


class PointTransformerBlock(nn.Module):
    """Local multi-head attention with relative position encoding."""

    def __init__(self, dim, k=16, num_heads=4):
        super().__init__()
        self.k = k
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.pos_enc = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, feat, coords, knn_idx):
        """
        feat:    (N, C)
        coords:  (N, 3)
        knn_idx: (N, k)
        """
        N, C = feat.shape
        k = knn_idx.shape[1]
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(feat)                                # (N, C)
        k_feat = self.k_proj(feat)
        v_feat = self.v_proj(feat)

        k_nbr = k_feat[knn_idx]                             # (N, k, C)
        v_nbr = v_feat[knn_idx]

        rel_pos = coords[knn_idx] - coords.unsqueeze(1)     # (N, k, 3)
        pe = self.pos_enc(rel_pos)                           # (N, k, C)

        k_nbr = k_nbr + pe
        v_nbr = v_nbr + pe

        q = q.view(N, 1, H, D).permute(0, 2, 1, 3)         # (N, H, 1, D)
        k_nbr = k_nbr.view(N, k, H, D).permute(0, 2, 1, 3) # (N, H, k, D)
        v_nbr = v_nbr.view(N, k, H, D).permute(0, 2, 1, 3)

        attn = (q @ k_nbr.transpose(-2, -1)) / (D ** 0.5)   # (N, H, 1, k)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v_nbr).squeeze(2).reshape(N, C)        # (N, C)
        out = self.out_proj(out)

        feat = self.norm1(feat + out)
        feat = self.norm2(feat + self.ffn(feat))
        return feat


class SonataCpPTHead(SonataCpClassifier):
    """Sonata encoder + Point Transformer local attention + MLP classifier."""

    def __init__(
        self,
        pt_k: int = 16,
        pt_layers: int = 2,
        pt_dim: int = 256,
        pt_heads: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        upcast_dim = self.hparams.upcast_dim
        self.input_proj = nn.Linear(upcast_dim, pt_dim)

        self.pt_blocks = nn.ModuleList([
            PointTransformerBlock(pt_dim, k=pt_k, num_heads=pt_heads)
            for _ in range(pt_layers)
        ])

        K = len(self.bin_centers)
        self.cp_classifier_head = self._build_head(
            pt_dim, [pt_dim, pt_dim], K, self.hparams.dropout,
        )

    def _get_extra_param_groups(self):
        mp_params = list(self.input_proj.parameters()) + list(self.pt_blocks.parameters())
        return [{"params": mp_params, "lr": self.hparams.head_lr}]

    def forward(self, point_data):
        encoded = self._encode(point_data)
        feat = encoded.feat
        coord = encoded.coord

        feat = self.input_proj(feat)

        knn_idx = chunked_knn(coord, k=self.hparams.pt_k)

        for block in self.pt_blocks:
            feat = block(feat, coord, knn_idx)

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
