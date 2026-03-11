# src/models/transolver_sonata_lightning.py
from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
import sonata
from .transolver_sonata import Transolver_block_with_Sonata, GeometryAwareAggregator
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
        # Caching parameters
        cache_dir: str = "./data/sonata_cache",
        enable_caching: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        grid_ref = 8
        fun_dim = 9
        self.grid_ref = grid_ref
        # Setup caching
        self.enable_caching = enable_caching
        self.cache_dir = Path(cache_dir)
        if self.enable_caching:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.feature_cache = {}  # In-memory cache using batch_idx as key
        self.cache_populated = False
        # Load pretrained Sonata encoder
        custom_config = dict(
            enc_mode=True,  # Encoder only mode
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
            apply_film = (i < 2) or (i >= num_layers - 2)
            self.blocks.append(
                Transolver_block_with_Sonata(
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                    slice_num=slice_num,
                    sonata_dim=sonata_dim,
                    use_film=apply_film,
                    last_layer=(i == num_layers - 1),
                    out_dim=output_dim,
                )
            )
        self.sonata_feature_aggregator = GeometryAwareAggregator()
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

    def _get_cache_key(self, batch_idx, is_training=True):
        """Generate cache key using batch_idx and split type"""
        split = "train" if is_training else "val"
        return f"{split}_{batch_idx}"

    def _batch_sonata_features(self, sonata_output):
        sonata_features = sonata_output.feat
        sonata_batch_indices = sonata_output.batch
        return sonata_features, sonata_batch_indices

    def _get_sonata_features(self, sonata_point, batch_idx, is_training=True):
        """Get Sonata features either from cache or by encoding"""
        cache_key = self._get_cache_key(batch_idx, is_training)

        if not self.enable_caching or not self.cache_populated:
            # Encode with Sonata
            with torch.no_grad():
                sonata_output = self.sonata_encoder(sonata_point)
                # Extract features from Sonata output
                if hasattr(sonata_output, "feat"):
                    sonata_features, sonata_indices = self._batch_sonata_features(
                        sonata_output
                    )
                # Cache the features if caching is enabled
                if self.enable_caching:
                    self.feature_cache[cache_key] = [
                        sonata_features.cpu(),
                        sonata_indices.cpu(),
                    ]
                return sonata_features, sonata_indices
        else:
            # Retrieve from cache
            if cache_key in self.feature_cache:
                sonata_features, sonata_indices = self.feature_cache[cache_key]
                return sonata_features.cuda(), sonata_indices.cuda()
            else:
                raise ValueError(f"Warning: Cache miss for key {cache_key}")
                # with torch.no_grad():
                #     sonata_output = self.sonata_encoder(sonata_point)
                #     if hasattr(sonata_output, "feat"):
                #         sonata_features = self._batch_sonata_features(sonata_output)
                #     # Add to cache
                #     self.feature_cache[cache_key] = sonata_features.cpu()
                #     return sonata_features

    def prepare_sonata_input(self, data):
        """Prepare point cloud for Sonata encoder"""
        transolver_point = {
            "pos": data["uncentered_coord"].unsqueeze(0),
            "fx": torch.cat(
                (
                    data["uncentered_coord"],
                    data["untransformed_normal"],
                    data["untransformed_deltas"],
                ),
                dim=-1,
            ).unsqueeze(0),
        }
        sonata_point = data
        target = data["pressure"].unsqueeze(0)
        return sonata_point, transolver_point, target

    def forward(self, point_data, batch_idx=None, is_training=True):
        """
        Args:
            point_data: Point cloud data dict
            batch_idx: Batch index for caching
            is_training: Whether in training mode (for cache key generation)
        Returns:
            predictions: Output predictions [B, N, output_dim]
        """
        sonata_point, transolver_point, target = self.prepare_sonata_input(point_data)

        # Get Sonata features (from cache if available)
        if batch_idx is not None:
            sonata_features, sonata_indices = self._get_sonata_features(
                sonata_point, batch_idx, is_training
            )
        else:
            # Fallback to no caching if batch_idx not provided
            with torch.no_grad():
                sonata_output = self.sonata_encoder(sonata_point)
                sonata_features, sonata_indices = self._batch_sonata_features(
                    sonata_output
                )

        # Project input vertices
        new_pos = self.get_grid(transolver_point["pos"])  # needs to be batchified
        transolver_features = torch.cat((transolver_point["fx"], new_pos), dim=-1)
        x = self.input_proj(transolver_features)  # [B, N, hidden_dim]

        # Pass through Transolver blocks with Sonata cross-attention
        g_vec = self.sonata_feature_aggregator(sonata_features)
        # sonata_mean = sonata_features.mean(0).reshape(1,-1)
        # sonata_max = sonata_features.max(0)[0].reshape(1,-1)
        # g_vec = torch.cat((sonata_mean,sonata_max),dim=1)
        all_film_stats = {}
        for id, block in enumerate(self.blocks):
            # if id < self.hparams.num_layers // 2:
            #     x = block(x, sonata_features=None)
            # else:
            # x, batch_indices, sonata_features=None, sonata_batch_indices=None
            
            x, stats = block(
                x,
                sonata_features=g_vec,
            )
            if stats is not None:
                all_film_stats[f"L{id+1}"] = stats
        # x = x[sonata_point["inverse"]]
        # x = self.upsample_transolver(x)
        return x, all_film_stats  # [B, N, output_dim]

    def training_step(self, batch, batch_idx):
        pressure = batch["pressure"].unsqueeze(0)
        # Forward pass with batch_idx for caching
        pred, film_metadata = self.forward(batch, batch_idx, is_training=True)
        loss = self.loss_fn(pred, pressure)
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        # Log LRs for both parameter groups
        # Group 0 = Model, Group 1 = Bridge (film_gen)
        lrs = self.lr_schedulers().get_last_lr()
        self.log("lr/model", lrs[0], on_step=True)
        self.log("lr/bridge", lrs[1], on_step=True)
        # Monitor every 50 optimizer updates
        if self.global_step % 50 == 0:
            for layer_name, stats in film_metadata.items():
                self.log(f"film/gamma_{layer_name}", stats["gamma"], on_step=True)
                self.log(f"film/beta_{layer_name}", stats["beta"], on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pressure = batch["pressure"].unsqueeze(0)
        # Forward pass with batch_idx for caching (validation split)
        pred, _ = self.forward(batch, batch_idx, is_training=False)
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

    def on_before_optimizer_step(self, optimizer):
        # This computes the L2 norm of the gradients for all parameters
        norms = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float('inf'))
        self.log("diag/grad_norm", norms)
        
    def on_train_epoch_end(self):
        """Mark cache as populated after first epoch"""
        if self.enable_caching and not self.cache_populated and self.current_epoch == 0:
            self.cache_populated = True
            print(
                f"Sonata feature cache populated with {len(self.feature_cache)} entries"
            )
            # Save cache to disk with batch_idx based keys
            cache_file = self.cache_dir / "sonata_features_cache_indexed.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(self.feature_cache, f)
            print(f"Cache saved to {cache_file}")

    def on_train_start(self):
        """Load cache from disk if available"""
        if self.enable_caching:
            cache_file = self.cache_dir / "sonata_features_cache_indexed.pkl"
            if cache_file.exists():
                print(f"Loading cache from {cache_file}")
                with open(cache_file, "rb") as f:
                    self.feature_cache = pickle.load(f)
                self.cache_populated = True
                print(f"Loaded {len(self.feature_cache)} cached entries")
                # Delete Sonata encoder to free memory
                if self.cache_populated and self.hparams.freeze_sonata:
                    del self.sonata_encoder
                    torch.cuda.empty_cache()
                    print("Sonata encoder deleted to free memory")

    # def configure_optimizers(self):
    #     # Only optimize non-frozen parameters
    #     params = [p for p in self.parameters() if p.requires_grad]
    #     optimizer = AdamW(
    #         params,
    #         lr=self.hparams.learning_rate,
    #         weight_decay=self.hparams.weight_decay,
    #     )
    #     scheduler = {
    #         "scheduler": torch.optim.lr_scheduler.OneCycleLR(
    #             optimizer,
    #             max_lr=self.hparams.learning_rate,
    #             total_steps=(581 + 1) * 200,
    #             final_div_factor=1000.0,
    #         ),
    #         "interval": "step",
    #     }
    #     return [optimizer], [scheduler]

    def configure_optimizers(self):
        bridge_params = []
        model_params = []
        
        # Separate frozen from unfrozen, and bridge from backbone
        for name, p in self.named_parameters():
            if not p.requires_grad: 
                continue 
            if "film_gen" in name or "aggregator" in name:
                bridge_params.append(p)
            else:
                model_params.append(p)

        # 5x higher max_lr for the geometry bridge
        max_lrs = [self.hparams.learning_rate, self.hparams.learning_rate * 5]

        optimizer = AdamW([
            {"params": model_params},
            {"params": bridge_params}
        ], lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

        # Automatically accounts for 500 epochs and accumulation=8
        total_steps = self.trainer.estimated_stepping_batches

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lrs,
                total_steps=total_steps,
                pct_start=0.2,           # Warm up for 20% of 37,500 steps
                div_factor=25.0,         # Start at max_lr / 25
                final_div_factor=1000.0, # End at max_lr / 1000
                anneal_strategy='cos'
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]