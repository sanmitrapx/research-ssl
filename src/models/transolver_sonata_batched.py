# src/models/transolver_sonata_batched.py
import torch
import torch.nn as nn
from torch_scatter import scatter_softmax, scatter_sum
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
    """Physics attention with integrated Sonata cross-attention using scatter operations"""

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
        self.to_out = self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),  # Remove bias to save memory
            nn.Dropout(dropout, inplace=True),  # Use inplace dropout
        )

    # def forward(
    #     self, x, batch_indices, sonata_features=None, sonata_batch_indices=None
    # ):
    #     # Process in chunks if input is too large
    #     N_total = x.shape[0]

    #     if N_total > 50000:  # Threshold for chunking
    #         chunk_size = 25000
    #         outputs = []

    #         for i in range(0, N_total, chunk_size):
    #             end_idx = min(i + chunk_size, N_total)
    #             chunk_out = self.forward(
    #                 x[i:end_idx],
    #                 batch_indices[i:end_idx],
    #                 sonata_features[i:end_idx],
    #                 sonata_batch_indices[i:end_idx],
    #             )
    #             outputs.append(chunk_out)

    #         return torch.cat(outputs, dim=0)
    #     else:
    #         return self._forward_impl(
    #             x, batch_indices, sonata_features, sonata_batch_indices
    #         )

    def forward(
        self, x, batch_indices, sonata_features=None, sonata_batch_indices=None
    ):
        """Efficient forward using scatter operations with einsum-based cross-attention"""

        N_total, C = x.shape
        dtype = x.dtype
        batch_size = batch_indices.max().item() + 1

        # (1) Project input
        fx_mid = self.in_project_fx(x).reshape(N_total, self.heads, self.dim_head)
        x_mid = self.in_project_x(x).reshape(N_total, self.heads, self.dim_head)

        # (2) Compute slice weights - softmax over slice dimension [6]
        slice_logits = self.in_project_slice(x_mid)  # [N_total, H, G]
        temperature_expanded = self.temperature.squeeze(-1)  # [1, H, 1]
        slice_weights = self.softmax(
            slice_logits / temperature_expanded
        )  # [N_total, H, G]

        # (3) Create weighted features for slicing
        weighted_fx = fx_mid.unsqueeze(2) * slice_weights.unsqueeze(
            -1
        )  # [N_total, H, G, D]

        # (4) Flatten for scatter operations
        weighted_fx_flat = weighted_fx.reshape(N_total, -1)  # [N_total, H*G*D]
        slice_weights_flat = slice_weights.reshape(N_total, -1)  # [N_total, H*G]

        # (5) Scatter sum to create slice tokens per batch [6]
        slice_tokens_flat = scatter_sum(
            weighted_fx_flat,
            batch_indices,
            dim=0,
            dim_size=batch_size,
        )  # [B, H*G*D]
        slice_norms_flat = scatter_sum(
            slice_weights_flat,
            batch_indices,
            dim=0,
            dim_size=batch_size,
        )  # [B, H*G]

        # (6) Reshape back to structured format
        slice_tokens = slice_tokens_flat.reshape(
            batch_size, self.heads, self.slice_num, self.dim_head
        )
        slice_norms = slice_norms_flat.reshape(batch_size, self.heads, self.slice_num)

        # (7) Normalize slice tokens [6]
        slice_tokens = slice_tokens / (slice_norms.unsqueeze(-1) + 1e-5)

        # (8) Self-attention among slice tokens [6]
        q_slice = self.to_q(slice_tokens)  # [B, H, G, D]
        k_slice = self.to_k(slice_tokens)  # [B, H, G, D]
        v_slice = self.to_v(slice_tokens)  # [B, H, G, D]

        dots = (
            torch.matmul(q_slice, k_slice.transpose(-1, -2)) * self.scale
        )  # [B, H, G, G]
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice)  # [B, H, G, D]

        # (9) Cross-attention with Sonata using einsum method
        if sonata_features is not None and sonata_batch_indices is not None:
            out_slice_token = self._block_diagonal_cross_attention_einsum(
                out_slice_token,  # [B, H, G, D] - queries from slice tokens
                sonata_features,  # [M_total, D_sonata] - all Sonata features
                batch_indices,  # [N_total] - batch assignment for points (not directly used here)
                sonata_batch_indices,  # [M_total] - batch assignment for Sonata features
            )

        # (10) De-slice: redistribute to original points
        # Expand slice tokens back to all points
        out_x = torch.zeros(
            N_total, self.heads, self.dim_head, device=x.device, dtype=dtype
        )

        for b in range(batch_size):
            mask = batch_indices == b
            if mask.any():
                # Get slice weights for points in this batch
                weights_batch = slice_weights[mask]  # [N_b, H, G]
                # Apply de-slicing using einsum
                result = torch.einsum("hgd,nhg->nhd", out_slice_token[b], weights_batch)
                out_x[mask] = result.to(dtype)
        # Reshape and project output
        out_x = out_x.reshape(N_total, -1)  # [N_total, H*D]
        return self.to_out(out_x)

    def _block_diagonal_cross_attention_einsum(
        self, queries, keys_values, batch_indices, kv_batch_indices
    ):
        """
        Optimized cross-attention using einsum for maximum efficiency

        Args:
            queries: [B, H, G, D] slice tokens as queries
            keys_values: [M_total, D_sonata] all Sonata features
            batch_indices: [N_total] batch assignments for points (not used here)
            kv_batch_indices: [M_total] batch assignments for Sonata features
        """
        B, H, G, D = queries.shape
        M_total = keys_values.shape[0]
        device = queries.device

        # Project Sonata features if needed
        if hasattr(self, "sonata_proj"):
            keys_values = self.sonata_proj(keys_values)  # [M_total, inner_dim]

        # Reshape for multi-head attention
        keys_values = keys_values.reshape(M_total, H, D)  # [M_total, H, D]

        # Compute keys and values
        k_sonata = self.cross_k(keys_values)  # [M_total, H, D]
        v_sonata = self.cross_v(keys_values)  # [M_total, H, D]

        # Create batch assignment matrix (one-hot encoding)
        # This creates a sparse matrix where batch_mask[b, m] = 1 if Sonata feature m belongs to batch b
        batch_mask = torch.zeros(B, M_total, device=device)
        batch_mask[kv_batch_indices, torch.arange(M_total, device=device)] = 1.0

        # Gather keys and values per batch using einsum
        k_batched = torch.einsum(
            "bm,mhd->bhmd", batch_mask, k_sonata
        )  # [B, H, M_max, D]
        v_batched = torch.einsum(
            "bm,mhd->bhmd", batch_mask, v_sonata
        )  # [B, H, M_max, D]

        # Compute attention scores using einsum
        scores = (
            torch.einsum("bhgd,bhmd->bhgm", queries, k_batched) * self.scale
        )  # [B, H, G, M_max]

        # Create attention mask from batch_mask
        attn_mask = batch_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, M_total]

        # Masked softmax (set scores to -inf where mask is 0)
        scores = scores - (1.0 - attn_mask) * 1e9
        attn_weights = torch.softmax(scores, dim=-1)

        # Handle numerical issues (NaN from all-masked rows)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)

        # Apply attention to values using einsum
        cross_out = torch.einsum(
            "bhgm,bhmd->bhgd", attn_weights, v_batched
        )  # [B, H, G, D]

        # Add residual connection (queries + cross_out)
        return queries + cross_out


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
    ):
        super().__init__()
        self.last_layer = last_layer
        self.hidden_dim = hidden_dim

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

    def forward(
        self, fx, batch_indices, sonata_features=None, sonata_batch_indices=None
    ):
        """
        Forward pass for variable-sized batches

        Args:
            fx: Concatenated features [N_total, C]
            batch_indices: Batch assignment for each point [N_total]
            sonata_features: Sonata features (concatenated or list)
        """
        # Apply attention with batch indices
        fx_out = fx + self.Attn(
            self.ln_1(fx), batch_indices, sonata_features, sonata_batch_indices
        )

        # MLP (works element-wise, so no changes needed)
        fx_out = fx_out + self.mlp(self.ln_2(fx_out))

        # Final projection if last layer
        if self.last_layer:
            fx_out = self.ln_3(fx_out)
            fx_out = self.mlp2(fx_out)

        return fx_out
