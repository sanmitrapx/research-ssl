# src/models/transolver_sonata.py
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from timm.models.layers import trunc_normal_

ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


class Enhanced_Physics_Attention_with_Sonata(nn.Module):
    """Physics attention with integrated Sonata cross-attention"""

    def __init__(
        self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64, sonata_dim=512
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.slice_num = slice_num
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        # Projections for slicing
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)

        # Self-attention on slices
        self.to_q = nn.Linear(dim_head, dim_head)
        self.to_k = nn.Linear(dim_head, dim_head)
        self.to_v = nn.Linear(dim_head, dim_head)

        # Cross-attention with Sonata
        if sonata_dim != inner_dim:
            self.sonata_proj = nn.Linear(sonata_dim, inner_dim)
        else:
            self.sonata_proj = nn.Identity()

        self.cross_k = nn.Linear(dim_head, dim_head)
        self.cross_v = nn.Linear(dim_head, dim_head)

        # Output projection
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, sonata_features=None):
        """
        Args:
            x: Input features [B, N, C]
            sonata_features: Sonata encoded features [B, M, D]
        """
        B, N, C = x.shape

        # Project input
        x_mid = (
            self.in_project_x(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
        )
        fx_mid = (
            self.in_project_fx(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
        )

        # (1) Compute slice assignments [6]
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / self.temperature
        )  # [B, H, N, G]
        slice_norm = slice_weights.sum(2)  # [B, H, G]

        # Create slice tokens [6]
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None])

        # (2) Self-attention among slice tokens [6]
        q_slice = self.to_q(slice_token)
        k_slice = self.to_k(slice_token)
        v_slice = self.to_v(slice_token)

        dots = torch.matmul(q_slice, k_slice.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice)  # [B, H, G, D]

        # (3) Cross-attention with Sonata features (on slice tokens)
        if sonata_features is not None:
            # Project Sonata features
            sonata_features = self.sonata_proj(sonata_features)  # [B, M, inner_dim]
            sonata_features = sonata_features.reshape(B, -1, self.heads, self.dim_head)
            sonata_features = sonata_features.permute(0, 2, 1, 3)  # [B, H, M, D]

            # Cross-attention: slice tokens query, Sonata features as key/value
            k_sonata = self.cross_k(sonata_features)
            v_sonata = self.cross_v(sonata_features)

            cross_dots = (
                torch.matmul(out_slice_token, k_sonata.transpose(-1, -2)) * self.scale
            )
            cross_attn = self.softmax(cross_dots)  # [B, H, G, M]
            cross_out = torch.matmul(cross_attn, v_sonata)  # [B, H, G, D]

            # Combine with self-attention output
            out_slice_token = out_slice_token + cross_out

        # (4) De-slice: redistribute to original points [6]
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")

        return self.to_out(out_x)


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu"):
        super().__init__()

        if n_layers == 0:
            self.layers = nn.ModuleList([nn.Linear(n_input, n_output)])
        else:
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(n_input, n_hidden))
            for _ in range(n_layers - 1):
                self.layers.append(nn.Linear(n_hidden, n_hidden))
            self.layers.append(nn.Linear(n_hidden, n_output))

        self.act = ACTIVATION[act]

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.act(x)
        return x


class Transolver_block_with_Sonata(nn.Module):
    """Transformer encoder block with Sonata cross-attention"""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
        sonata_dim: int = 512,
        use_film: bool = False
    ):
        super().__init__()
        self.last_layer = last_layer
        self.hidden_dim = hidden_dim
        self.use_film = use_film

        # Layer norms
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)

        # Enhanced Physics attention with Sonata
        self.Attn = Enhanced_Physics_Attention_with_Sonata(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            sonata_dim=sonata_dim,
        )
        # FiLM Generator with Zero-Init
        if self.use_film:
            self.film_gen = nn.Sequential(
                nn.Linear(sonata_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim * 2)
            )
            # ZERO-INIT: Final linear layer starts at 0
            nn.init.constant_(self.film_gen[-1].weight, 0)
            nn.init.constant_(self.film_gen[-1].bias, 0)
        # MLP
        self.mlp = MLP(
            hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, act=act
        )

        # Final layer projections if needed
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # def forward(self, fx, sonata_features=None):
    #     """
    #     Args:
    #         fx: Point cloud features [B, N, C]
    #         sonata_features: Encoded features from Sonata [B, M, D]
    #     """

    #     fx = fx + self.Attn(self.ln_1(fx), sonata_features)

    #     # MLP block
    #     fx = fx + self.mlp(self.ln_2(fx))

    #     # Optional final projection for last layer
    #     if self.last_layer:
    #         fx = self.ln_3(fx)
    #         return self.mlp2(fx)

    #     return fx
    def forward(self, fx, sonata_features=None):
        # --- Physics Attention Branch ---
        # Note: If use_film is True, we pass sonata_features as None to Attn 
        # to ensure the modulation happens via FiLM, not cross-attention.
        film_stats = None
        attn_out = self.Attn(self.ln_1(fx), sonata_features=None)

        if self.use_film and sonata_features is not None:
            # Generate gamma and beta
            # If sonata_features is [B, D], unsqueeze to [B, 1, 2*D]
            params = self.film_gen(sonata_features)
            if params.dim() == 2:
                params = params.unsqueeze(1)
            
            gamma, beta = torch.chunk(params, 2, dim=-1)
            film_stats = {"gamma": gamma.abs().mean(), "beta": beta.abs().mean()}
            
            # Apply FiLM to the attention output
            attn_out = attn_out * (1 + gamma) + beta

        # Add back to residual
        fx = fx + attn_out

        # --- MLP Branch ---
        fx = fx + self.mlp(self.ln_2(fx))

        if self.last_layer:
            fx = self.ln_3(fx)
            return self.mlp2(fx), film_stats

        return fx, film_stats

class GeometryAwareAggregator(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        # 1. Local Context Layer (Curvature Detector)
        # Using a kernel size of 3 to look at immediate neighbors
        self.local_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim), # Depthwise
            nn.Conv1d(dim, dim, kernel_size=1), # Pointwise
            nn.SiLU()
        )
        
        # 2. Global Pooling (Attention)
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        
        # Final projection to "clean" the global vector
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, z):
        # z: [151, 512]
        
        # Step 1: Extract Local Curvature
        # Transpose for Conv1d: [B, 512, 151]
        z = z.unsqueeze(0)
        z_local = z.transpose(1, 2)
        z_local = self.local_conv(z_local)
        z_local = z_local.transpose(1, 2) # Back to [B, 151, 512]
        
        # Residual connection to keep original features
        z = z + z_local 
        
        # Step 2: Global Attention Pooling
        # Query: [B, 1, 512], Key/Value: [B, 151, 512]
        z_global, _ = self.attn(self.query.repeat(z.size(0), 1, 1), z, z)
        
        return self.post_norm(z_global.squeeze(1)) # [B, 512]