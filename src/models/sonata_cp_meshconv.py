"""Mesh Convolution head – message passing on actual surface topology.

Faces from the original mesh are remapped to grid-sampled resolution
via the inverse mapping, producing a sparse adjacency at the same
resolution as the Sonata encoder output.  Message passing layers then
operate on this mesh-derived graph, respecting the true surface
connectivity rather than arbitrary KNN neighbourhoods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

from .sonata_cp_classifier import SonataCpClassifier
from .utils.knn import faces_to_edge_index


class MeshConvBlock(nn.Module):
    """Message-passing layer on mesh edges with relative position features."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_dim + 3, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, feat, edge_index, coord):
        """
        feat:       (N, C_in)
        edge_index: (2, E)  src → dst
        coord:      (N, 3)
        """
        src, dst = edge_index

        rel_pos = coord[dst] - coord[src]                   # (E, 3)
        msg_in = torch.cat([feat[src], feat[dst] - feat[src], rel_pos], dim=-1)
        messages = self.msg_mlp(msg_in)                     # (E, C_out)

        agg = torch_scatter.scatter_mean(messages, dst, dim=0, dim_size=feat.shape[0])

        return self.norm(agg + self.skip(feat))


class SonataCpMeshConv(SonataCpClassifier):
    """Sonata encoder + mesh-topology message passing + MLP classifier."""

    def __init__(
        self,
        meshconv_layers: int = 3,
        meshconv_dim: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        upcast_dim = self.hparams.upcast_dim
        self.mc_layers = nn.ModuleList()
        in_dim = upcast_dim
        for _ in range(meshconv_layers):
            self.mc_layers.append(MeshConvBlock(in_dim, meshconv_dim))
            in_dim = meshconv_dim

        K = len(self.bin_centers)
        self.cp_classifier_head = self._build_head(
            meshconv_dim, [meshconv_dim, meshconv_dim], K, self.hparams.dropout,
        )

    def _get_extra_param_groups(self):
        return [{"params": self.mc_layers.parameters(), "lr": self.hparams.head_lr}]

    def forward(self, point_data):
        encoded = self._encode(point_data)
        feat = encoded.feat
        coord = encoded.coord

        inverse = point_data["inverse"]
        faces = point_data["faces"]
        edge_index = faces_to_edge_index(faces, inverse, coord.shape[0])

        for layer in self.mc_layers:
            feat = layer(feat, edge_index, coord)

        feat = feat[inverse]

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
