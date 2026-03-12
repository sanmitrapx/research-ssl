"""Variant 4: Spatial smoothing via KNN-CRF post-processing.

After the classification head, a lightweight mean-field CRF (implemented
as message-passing over KNN graph) refines per-point logits using spatial
neighbours.  This encourages Cp predictions to be spatially coherent
along the mesh surface.

The CRF is differentiable and trained end-to-end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..sonata_cp_classifier import SonataCpClassifier


class KNNCRFLayer(nn.Module):
    """Lightweight CRF via iterative KNN message passing.

    At each iteration:
        q <- softmax(unary + compat @ aggregate_neighbours(q))
    """

    def __init__(self, num_classes: int, k: int = 16, iterations: int = 3):
        super().__init__()
        self.k = k
        self.iterations = iterations
        self.compat = nn.Linear(num_classes, num_classes, bias=False)
        nn.init.eye_(self.compat.weight)
        self.compat.weight.data *= -0.1

    def _knn_graph(self, coords):
        """Build KNN indices from point coordinates. Returns (N, k) index tensor."""
        dists = torch.cdist(coords, coords)
        _, idx = dists.topk(self.k, dim=-1, largest=False)
        return idx

    def forward(self, logits, coords):
        """
        logits: (N, K) raw unary logits
        coords: (N, 3) point positions
        """
        knn_idx = self._knn_graph(coords)
        q = F.softmax(logits, dim=-1)

        for _ in range(self.iterations):
            neighbour_q = q[knn_idx]                  # (N, k, K)
            msg = neighbour_q.mean(dim=1)              # (N, K)
            refined = logits + self.compat(msg)
            q = F.softmax(refined, dim=-1)

        return refined, q


class SonataCpCRF(SonataCpClassifier):

    def __init__(
        self,
        crf_k: int = 16,
        crf_iterations: int = 3,
        crf_max_points: int = 50000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        K = len(self.bin_centers)
        self.crf = KNNCRFLayer(K, k=crf_k, iterations=crf_iterations)

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

        coords = point_data["uncentered_coord"]
        N = coords.shape[0]
        max_pts = self.hparams.crf_max_points

        if N > max_pts:
            # Subsample for CRF to keep memory bounded, but refine all points
            # by mapping each point to its nearest sampled neighbour
            perm = torch.randperm(N, device=coords.device)[:max_pts]
            sub_logits, sub_q = self.crf(logits[perm], coords[perm])

            # Map back: find nearest sampled point for each original point
            dists = torch.cdist(coords, coords[perm])
            nearest = dists.argmin(dim=-1)
            refined_logits = sub_logits[nearest]
            probs = sub_q[nearest]
        else:
            refined_logits, probs = self.crf(logits, coords)

        cp_hat = (probs * self.bin_centers).sum(dim=-1)

        return dict(logits=refined_logits, probs=probs, cp_hat=cp_hat)
