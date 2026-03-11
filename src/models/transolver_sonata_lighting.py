# src/models/transolver_sonata_lightning.py
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import sonata
from sonata.structure import Point
from .transolver_sonata import Transolver_block_with_Sonata
from .utils.losses import LpLoss


class TransolverSonataModel(pl.LightningModule):
    """Transolver with Sonata cross-attention for point cloud processing"""

    def __init__(
        self,
        # Sonata parameters
        sonata_repo: str = "facebook/sonata",
        freeze_sonata: bool = True,
        sonata_dim: int = 512,
        # Transolver parameters
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        slice_num: int = 32,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        # Output parameters
        output_dim: int = 1,
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
    ):
        super().__init__()
        self.save_hyperparameters()
        grid_ref = 8
        fun_dim = 3
        self.grid_ref = grid_ref
        print("lr is ", learning_rate)
        # Load pretrained Sonata encoder
        custom_config = dict(
            enc_mode=False,  # Encoder only mode
            enable_flash=True,
        )
        self.sonata_encoder = sonata.model.load(
            "sonata", repo_id=sonata_repo, custom_config=custom_config
        ).cuda()
        # Freeze Sonata if requested
        if freeze_sonata:
            for param in self.sonata_encoder.parameters():
                param.requires_grad = False
        # Input projection
        self.input_proj = nn.Linear(
            fun_dim + grid_ref * grid_ref * grid_ref, hidden_dim
        )  # 3D coordinates to hidden_dim
        # Transolver blocks with Sonata cross-attention
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            is_last = i == num_layers - 1
            self.blocks.append(
                Transolver_block_with_Sonata(
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                    slice_num=slice_num,
                    sonata_dim=sonata_dim,
                    last_layer=is_last,
                    out_dim=output_dim,
                )
            )
        self.loss_fn = LpLoss(size_average=True)

    def get_grid(self, my_pos):
        # my_pos 1 N 3
        batchsize = my_pos.shape[0]
        gridx = torch.tensor(np.linspace(-1.5, 1.5, self.grid_ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.grid_ref, 1, 1, 1).repeat(
            [batchsize, 1, self.grid_ref, self.grid_ref, 1]
        )
        gridy = torch.tensor(np.linspace(0, 2, self.grid_ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.grid_ref, 1, 1).repeat(
            [batchsize, self.grid_ref, 1, self.grid_ref, 1]
        )
        gridz = torch.tensor(np.linspace(-4, 4, self.grid_ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, self.grid_ref, 1).repeat(
            [batchsize, self.grid_ref, self.grid_ref, 1, 1]
        )
        ref = (
            torch.cat((gridx, gridy, gridz), dim=-1)
            .cuda()
            .reshape(batchsize, self.grid_ref**3, 3)
        )  # B 4 4 4 3
        pos = (
            torch.sqrt(
                torch.sum((my_pos[:, :, None, :] - ref[:, None, :, :]) ** 2, dim=-1)
            )
            .reshape(
                batchsize,
                my_pos.shape[1],
                self.grid_ref * self.grid_ref * self.grid_ref,
            )
            .contiguous()
        )
        return pos

    def prepare_sonata_input(self, data):
        """Prepare point cloud for Sonata encoder"""
        # B, N, _ = point_data["uncentered_coord"].shape
        # # Create Point object expected by Sonata
        # point_dict = point_data | {
        #     "batch": torch.arange(B).repeat_interleave(N).to(point_data["coord"].device)
        # }
        transolver_point = {
            "pos": data["uncentered_coord"],
            "fx": data["feat"].unsqueeze(0),
        }
        sonata_point = data
        target = data["pressure"]
        return sonata_point, transolver_point, target

    def forward(self, point_data):
        """
        Args:
            vertices: Point cloud vertices [B, N, 3]
        Returns:
            predictions: Output predictions [B, N, output_dim]
        """
        # Get Sonata encoder features
        with torch.no_grad() if self.hparams.freeze_sonata else torch.enable_grad():
            sonata_point, transolver_point, target = self.prepare_sonata_input(
                point_data
            )
            sonata_output = self.sonata_encoder(sonata_point)
            B = transolver_point["pos"].shape[0]
            # Extract features from Sonata output
            if hasattr(sonata_output, "feat"):
                sonata_features = sonata_output.feat.reshape(
                    B, -1, self.hparams.sonata_dim
                )
            else:
                sonata_features = sonata_output.reshape(B, -1, self.hparams.sonata_dim)
        # Project input vertices
        new_pos = self.get_grid(transolver_point["pos"])
        transolver_features = torch.cat((transolver_point["pos"], new_pos), dim=-1)
        x = self.input_proj(transolver_features)  # [B, N, hidden_dim]
        # Pass through Transolver blocks with Sonata cross-attention
        for block in self.blocks:
            x = block(x, sonata_features)
        return x  # [B, N, output_dim]

    def training_step(self, batch, batch_idx):
        pressure = batch["pressure"]
        # Forward pass
        pred = self.forward(batch)
        loss = self.loss_fn(pred, pressure)
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        pressure = batch["pressure"]
        pred = self.forward(batch)
        loss = self.loss_fn(pred, pressure)
        # Additional metrics
        mae = torch.mean(torch.abs(pred - pressure))
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_mae",
            mae,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        # Reuse validation logic for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        # Only optimize non-frozen parameters
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = AdamW(
            params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        # scheduler = {
        #     "scheduler": CosineAnnealingLR(
        #         optimizer, T_max=self.hparams.max_steps, eta_min=1e-6
        #     ),
        #     "interval": "step",
        # }
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                # max_lr=5 * 0.0001,
                total_steps=(581 + 1) * 200,
                final_div_factor=1000.0,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]
