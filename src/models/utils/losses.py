import torch
import torch.nn as nn
import torch.nn.functional as F


class LpLoss(object):
    """Relative Lp loss: ||pred - target||_p / ||target||_p"""

    def __init__(self, p=2, reduction='mean'):
        self.p = p
        self.reduction = reduction

    def __call__(self, pred, target):
        diff_norms = torch.norm(pred.reshape(pred.shape[0], -1) - target.reshape(target.shape[0], -1), p=self.p, dim=1)
        target_norms = torch.norm(target.reshape(target.shape[0], -1), p=self.p, dim=1)
        relative = diff_norms / (target_norms + 1e-8)
        if self.reduction == 'mean':
            return relative.mean()
        return relative


class EMDLoss(nn.Module):
    """Earth Mover's Distance (Wasserstein-1) loss for ordinal classification.
    Cp-weighted: weights CDF differences by actual Cp distances between bins."""

    def __init__(self, num_bins, reduction="mean", bin_centers=None, bin_widths=None):
        super().__init__()
        self.num_bins = num_bins
        self.reduction = reduction
        if bin_widths is not None:
            self.register_buffer('bin_widths', bin_widths)
        elif bin_centers is not None:
            bin_centers_t = torch.as_tensor(bin_centers, dtype=torch.float32)
            widths = torch.diff(bin_centers_t)
            widths = torch.cat([widths[:1], widths])
            widths = widths / widths.sum()
            self.register_buffer('bin_widths', widths)
        else:
            self.register_buffer('bin_widths', torch.ones(num_bins) / num_bins)

    def forward(self, logits, targets):
        """
        logits: (N, K) raw logits
        targets: (N,) integer bin indices
        """
        probs = F.softmax(logits, dim=-1)
        target_one_hot = F.one_hot(targets, self.num_bins).float()
        cdf_pred = torch.cumsum(probs, dim=-1)
        cdf_target = torch.cumsum(target_one_hot, dim=-1)
        cdf_diff = torch.abs(cdf_pred - cdf_target)
        weighted_diff = cdf_diff * self.bin_widths.unsqueeze(0)
        return weighted_diff.sum(dim=-1).mean()
