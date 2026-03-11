# src/models/transolver_sonata.py
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from timm.models.layers import trunc_normal_

# Try to import flash attention
try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input

    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    print("Flash Attention not available. Using standard attention.")

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
    """Physics attention with integrated Sonata cross-attention and Flash Attention support"""

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        slice_num=64,
        sonata_dim=512,
        use_flash=True,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.slice_num = slice_num
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.use_flash = use_flash and FLASH_AVAILABLE

        # Projections for slicing
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)

        # Self-attention on slices - combined QKV for flash attention efficiency
        if self.use_flash:
            self.qkv = nn.Linear(dim_head, dim_head * 3, bias=False)
        else:
            self.to_q = nn.Linear(dim_head, dim_head)
            self.to_k = nn.Linear(dim_head, dim_head)
            self.to_v = nn.Linear(dim_head, dim_head)

        # Cross-attention with Sonata
        if sonata_dim != inner_dim:
            self.sonata_proj = nn.Linear(sonata_dim, inner_dim)
        else:
            self.sonata_proj = nn.Identity()

        # For cross-attention, we keep separate projections as flash_attn works best with self-attention
        self.cross_k = nn.Linear(dim_head, dim_head)
        self.cross_v = nn.Linear(dim_head, dim_head)

        # Output projection
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def _flash_self_attention(self, slice_token):
        """Perform self-attention using Flash Attention"""
        B, H, G, D = slice_token.shape

        # Compute QKV
        qkv = self.qkv(slice_token)  # [B, H, G, 3*D]
        qkv = qkv.reshape(B, H, G, 3, D).permute(3, 0, 1, 2, 4)  # [3, B, H, G, D]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is [B, H, G, D]

        # Reshape for flash attention: [B*H, G, D]
        q = q.reshape(B * H, G, D)
        k = k.reshape(B * H, G, D)
        v = v.reshape(B * H, G, D)

        # Apply flash attention
        # Note: flash_attn_func expects [batch, seqlen, num_heads, head_dim]
        # We need to reshape accordingly
        q = q.reshape(B, H, G, D).transpose(1, 2)  # [B, G, H, D]
        k = k.reshape(B, H, G, D).transpose(1, 2)  # [B, G, H, D]
        v = v.reshape(B, H, G, D).transpose(1, 2)  # [B, G, H, D]

        # Apply flash attention
        out = flash_attn_func(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,
        )  # [B, G, H, D]

        # Reshape back
        out = out.transpose(1, 2)  # [B, H, G, D]

        return out

    def _standard_self_attention(self, slice_token):
        """Standard self-attention implementation"""
        q_slice = self.to_q(slice_token)
        k_slice = self.to_k(slice_token)
        v_slice = self.to_v(slice_token)

        dots = torch.matmul(q_slice, k_slice.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice)

        return out_slice_token

    def _flash_cross_attention(self, queries, keys, values):
        """Perform cross-attention using Flash Attention"""
        B, H, G, D = queries.shape
        _, _, M, _ = keys.shape

        # Reshape for flash attention
        q = queries.reshape(B, H, G, D).transpose(1, 2)  # [B, G, H, D]
        k = keys.reshape(B, H, M, D).transpose(1, 2)  # [B, M, H, D]
        v = values.reshape(B, H, M, D).transpose(1, 2)  # [B, M, H, D]

        # Apply flash attention for cross-attention
        out = flash_attn_func(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,
        )  # [B, G, H, D]

        # Reshape back
        out = out.transpose(1, 2)  # [B, H, G, D]

        return out

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

        # (2) Self-attention among slice tokens using Flash Attention or standard
        if self.use_flash:
            out_slice_token = self._flash_self_attention(slice_token)
        else:
            out_slice_token = self._standard_self_attention(slice_token)

        # (3) Cross-attention with Sonata features (on slice tokens)
        if sonata_features is not None:
            # Project Sonata features
            sonata_features = self.sonata_proj(sonata_features)  # [B, M, inner_dim]
            sonata_features = sonata_features.reshape(B, -1, self.heads, self.dim_head)
            sonata_features = sonata_features.permute(0, 2, 1, 3)  # [B, H, M, D]

            # Cross-attention: slice tokens query, Sonata features as key/value
            k_sonata = self.cross_k(sonata_features)
            v_sonata = self.cross_v(sonata_features)

            if self.use_flash:
                # Use Flash Attention for cross-attention
                cross_out = self._flash_cross_attention(
                    out_slice_token, k_sonata, v_sonata
                )
            else:
                # Standard cross-attention
                cross_dots = (
                    torch.matmul(out_slice_token, k_sonata.transpose(-1, -2))
                    * self.scale
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
    """Transformer encoder block with Sonata cross-attention and Flash Attention support"""

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
        use_flash: bool = True,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.hidden_dim = hidden_dim

        # Layer norms
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)

        # Enhanced Physics attention with Sonata and Flash Attention
        self.Attn = Enhanced_Physics_Attention_with_Sonata(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            sonata_dim=sonata_dim,
            use_flash=use_flash,
        )

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

    def forward(self, fx, pos=None, sonata_features=None):
        """
        Args:
            fx: Point cloud features [B, N, C]
            pos: Position features (optional)
            sonata_features: Encoded features from Sonata [B, M, D]
        """
        # Physics attention with Sonata cross-attention
        fx = fx + self.Attn(self.ln_1(fx), sonata_features)

        # MLP block
        fx = fx + self.mlp(self.ln_2(fx))

        # Optional final projection for last layer
        if self.last_layer:
            fx = self.ln_3(fx)
            return self.mlp2(fx)

        return fx
