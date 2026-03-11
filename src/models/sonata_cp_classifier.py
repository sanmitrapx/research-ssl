import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
import sonata

from .utils.losses import EMDLoss, LpLoss

LOSS_TYPES = ("ce_emd", "ordinal_kl")


class _ResidualBlock(nn.Module):
    """Two-layer residual block: Linear-ReLU-Dropout-Linear + skip."""

    def __init__(self, dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.drop(F.relu(self.fc1(x)))
        out = self.fc2(out)
        return x + out


class SonataCpClassifier(pl.LightningModule):
    """Sonata encoder + per-point MLP for Cp classification.

    Follows the original Sonata paper's approach: encoder features are
    up-cast back to grid-sample resolution via the pooling hierarchy,
    then mapped to original-resolution points using the GridSample
    inverse mapping.  A residual MLP predicts per-point Cp bin logits.

    Fine-tuning: layerwise LR decay (lower LR for early encoder stages).

    Loss types (``loss_type``):
        ce_emd      -- CrossEntropy + lambda_emd * EMD
        ordinal_kl  -- Gaussian-smoothed KL divergence (sigma controlled by
                       ``label_smoothing_sigma``)
    """

    DEFAULT_UPCAST_DIM = 1088

    def __init__(
        self,
        sonata_repo: str = "facebook/sonata",
        decoder_dims: list = (512, 512, 256, 256),
        num_cp_bins: int = 64,
        bin_centers: list = None,
        upcast_dim: int = 1088,
        num_concat_levels: int = 2,
        use_geometric_features: bool = False,
        learning_rate: float = 0.0001,
        head_lr: float = 0.001,
        weight_decay: float = 0.01,
        lr_decay_rate: float = 0.65,
        max_epochs: int = 100,
        dropout: float = 0.1,
        loss_type: str = "ce_emd",
        weighted_emd: bool = True,
        lambda_emd: float = 0.1,
        lambda_recon: float = 0.0,
        label_smoothing_sigma: float = 2.0,
    ):
        super().__init__()
        if loss_type not in LOSS_TYPES:
            raise ValueError(
                f"Unknown loss_type '{loss_type}', choose from {LOSS_TYPES}"
            )
        self.save_hyperparameters()

        if bin_centers is not None:
            self.register_buffer(
                "bin_centers", torch.tensor(bin_centers, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "bin_centers", torch.linspace(-2.5, 1.0, num_cp_bins)
            )

        custom_config = dict(enc_mode=True, enable_flash=True)
        self.sonata = sonata.model.load(
            "sonata", repo_id=sonata_repo, custom_config=custom_config
        )

        K = len(self.bin_centers)
        point_feat_dim = upcast_dim
        if use_geometric_features:
            point_feat_dim += 9

        self.cp_classifier_head = self._build_head(
            point_feat_dim, list(decoder_dims), K, dropout
        )

        widths = torch.cat([
            self.bin_centers[1:] - self.bin_centers[:-1],
            (self.bin_centers[-1:] - self.bin_centers[-2:-1]),
        ])
        widths = widths / widths.sum()

        self.emd_loss = EMDLoss(
            K, reduction="mean", bin_widths=widths
        )
        self.ce_loss = nn.CrossEntropyLoss()
        self.rl2_loss = LpLoss()

    def _build_head(self, input_dim, decoder_dims, num_bins, dropout):
        """Build residual MLP head.

        Consecutive dims of the same size become a _ResidualBlock;
        a change in width is handled by a Linear projection.
        """
        layers = []
        in_dim = input_dim

        for i in range(len(decoder_dims)):
            dim = decoder_dims[i]
            if dim == in_dim and i > 0:
                layers.extend([
                    _ResidualBlock(dim, dropout),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
            else:
                layers.extend([
                    nn.Linear(in_dim, dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
            in_dim = dim

        layers.append(nn.Linear(in_dim, num_bins))
        return nn.Sequential(*layers)

    @staticmethod
    def _upcast_features(point, num_concat_levels):
        """Up-cast encoder features to grid-sample resolution.

        Walks up the pooling hierarchy: at each level, upsample the
        coarse features via the inverse mapping and concat with the
        finer parent features, then move to the parent level.
        """
        for _ in range(num_concat_levels):
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat(
                [parent.feat, point.feat[inverse]], dim=-1
            )
            point = parent
        return point

    def forward(self, point_data):
        encoded = self.sonata(point_data)

        encoded = self._upcast_features(encoded, self.hparams.num_concat_levels)
        feat = encoded.feat

        # Map grid-sampled features back to original mesh resolution
        inverse = point_data["inverse"]
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

    def _ordinal_kl_loss(self, logits, target_bins):
        """Gaussian-smoothed ordinal KL divergence loss."""
        K = logits.shape[-1]
        sigma = self.hparams.label_smoothing_sigma
        bins = torch.arange(K, device=logits.device, dtype=torch.float32)
        diff = bins.unsqueeze(0) - target_bins.float().unsqueeze(-1)
        soft_targets = torch.exp(-0.5 * (diff / sigma) ** 2)
        soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)
        log_probs = F.log_softmax(logits, dim=-1)
        return F.kl_div(log_probs, soft_targets, reduction="batchmean")

    def _compute_loss(self, logits, target_bins, pressure_raw, cp_hat):
        """Compute training loss based on configured loss_type."""
        if self.hparams.loss_type == "ordinal_kl":
            loss = self._ordinal_kl_loss(logits, target_bins)
        else:
            loss = self.ce_loss(logits, target_bins)
            if self.hparams.lambda_emd > 0:
                loss = loss + self.hparams.lambda_emd * self.emd_loss(logits, target_bins)
            if self.hparams.lambda_recon > 0:
                loss = loss + self.hparams.lambda_recon * F.smooth_l1_loss(
                    cp_hat, pressure_raw
                )
        return loss

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = self._compute_loss(
            out["logits"],
            batch.get("pressure_bin"),
            batch.get("pressure_raw"),
            out["cp_hat"],
        )
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        target_bins = batch["pressure_bin"]
        pressure_raw = batch["pressure_raw"]
        loss = self._compute_loss(out["logits"], target_bins, pressure_raw, out["cp_hat"])

        pred_bins = out["logits"].argmax(dim=-1)
        mbe = (pred_bins - target_bins).float().abs().mean()
        rl2 = self.rl2_loss(out["cp_hat"].unsqueeze(0), pressure_raw.unsqueeze(0))

        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log("val_mbe", mbe, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log("val_rl2", rl2, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def _get_encoder_param_groups(self):
        """Build per-stage param groups with decaying LR for layerwise_lr strategy."""
        lr = self.hparams.learning_rate
        decay = self.hparams.lr_decay_rate
        num_stages = 5

        groups = []
        emb_params = list(self.sonata.embedding.parameters())
        groups.append({"params": emb_params, "lr": lr * (decay ** num_stages)})

        for s in range(num_stages):
            stage = getattr(self.sonata.enc, f"enc{s}", None)
            if stage is not None:
                stage_params = list(stage.parameters())
                stage_lr = lr * (decay ** (num_stages - 1 - s))
                groups.append({"params": stage_params, "lr": stage_lr})

        return groups

    def configure_optimizers(self):
        head_lr = self.hparams.head_lr
        wd = self.hparams.weight_decay

        param_groups = self._get_encoder_param_groups()
        param_groups.append({
            "params": self.cp_classifier_head.parameters(),
            "lr": head_lr,
        })

        optimizer = AdamW(param_groups, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
