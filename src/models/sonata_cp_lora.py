"""LoRA fine-tuning of Sonata for Cp classification using HuggingFace PEFT.

Freezes the entire Sonata encoder and injects lightweight low-rank
adapters into the attention QKV and projection layers via peft's
LoraConfig + get_peft_model.  Only the adapters + classification head
are trained.

Typical parameter budget at rank=8:
    Encoder (frozen):  ~108M params  (0 trainable)
    LoRA adapters:     ~0.5-1.5M    (trainable)
    Classifier head:   ~1-2M        (trainable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
import sonata
from peft import LoraConfig, get_peft_model

from .utils.losses import EMDLoss, LpLoss
from .sonata_cp_classifier import _ResidualBlock

LOSS_TYPES = ("ce_emd", "ordinal_kl")


class SonataCpLoRA(pl.LightningModule):
    """Sonata encoder with PEFT LoRA adapters + per-point Cp classification head."""

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
        learning_rate: float = 1e-3,
        weight_decay: float = 0.05,
        max_epochs: int = 100,
        pct_start: float = 0.05,
        div_factor: float = 10.0,
        final_div_factor: float = 1000.0,
        dropout: float = 0.1,
        loss_type: str = "ce_emd",
        weighted_emd: bool = True,
        lambda_emd: float = 0.1,
        lambda_recon: float = 0.0,
        label_smoothing_sigma: float = 2.0,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_targets: list = ("attn.qkv", "attn.proj"),
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
        base_sonata = sonata.model.load(
            "sonata", repo_id=sonata_repo, custom_config=custom_config
        )

        # Apply PEFT LoRA to the Sonata encoder
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(lora_targets),
            bias="none",
        )
        self.sonata = get_peft_model(base_sonata, lora_config)
        self.sonata.print_trainable_parameters()

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

        self.emd_loss = EMDLoss(K, reduction="mean", bin_widths=widths)
        self.ce_loss = nn.CrossEntropyLoss()
        self.rl2_loss = LpLoss()

    def _build_head(self, input_dim, decoder_dims, num_bins, dropout):
        layers = []
        in_dim = input_dim
        for i, dim in enumerate(decoder_dims):
            if dim == in_dim and i > 0:
                layers.extend([
                    _ResidualBlock(dim, dropout), nn.ReLU(), nn.Dropout(dropout),
                ])
            else:
                layers.extend([
                    nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(dropout),
                ])
            in_dim = dim
        layers.append(nn.Linear(in_dim, num_bins))
        return nn.Sequential(*layers)

    @staticmethod
    def _upcast_features(point, num_concat_levels):
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

    def _ordinal_kl_loss(self, logits, target_bins):
        K = logits.shape[-1]
        sigma = self.hparams.label_smoothing_sigma
        bins = torch.arange(K, device=logits.device, dtype=torch.float32)
        diff = bins.unsqueeze(0) - target_bins.float().unsqueeze(-1)
        soft_targets = torch.exp(-0.5 * (diff / sigma) ** 2)
        soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)
        log_probs = F.log_softmax(logits, dim=-1)
        return F.kl_div(log_probs, soft_targets, reduction="batchmean")

    def _compute_loss(self, logits, target_bins, pressure_raw, cp_hat):
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

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        wd = self.hparams.weight_decay

        # PEFT marks LoRA params as requires_grad, everything else is frozen
        lora_params = [p for p in self.sonata.parameters() if p.requires_grad]
        head_params = list(self.cp_classifier_head.parameters())

        param_groups = [
            {"params": lora_params, "lr": lr},
            {"params": head_params, "lr": lr},
        ]

        optimizer = AdamW(param_groups, weight_decay=wd)

        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=self.hparams.pct_start,
            anneal_strategy="cos",
            div_factor=self.hparams.div_factor,
            final_div_factor=self.hparams.final_div_factor,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
