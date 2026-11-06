import torch
import torch.nn as nn
from mamba_ssm import Mamba3

from architecture.time_embedding import TimeEmbeddingLayer
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        expand: int,
        layer_idx: int,
        d_state: int,
        headdim: int,
        mimo_rank: int,
        chunk_size: int,
        dropout: float,
    ):
        super().__init__()
        self.mamba = Mamba3(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            expand=expand,
            is_mimo=False,
            mimo_rank=mimo_rank,
            chunk_size=chunk_size,
            is_outproj_norm=False,
            dropout=dropout,
            layer_idx=layer_idx,
            n_layer=n_layer,
        )

        # FiLM layers: Generate scale and shift from the time embedding
        self.film_gate = nn.Linear(d_model, d_model * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, layer_idx: int) -> torch.Tensor:
        # 1. Process via Mamba first
        residual = x
        out = self.mamba(x)
        # 2. FiLM Modulation: Inject time into the flow
        # This forces the model to scale/shift features based on 't'
        gate = self.film_gate(t_emb).unsqueeze(1) # (B, 1, 2*d_model)
        scale, shift = gate.chunk(2, dim=-1)

        # Modulate the Mamba output based on time
        out = out * (1 + scale) + shift

        # 3. Residual connection
        return residual + out

class Mamba3FlowMatchingModel(nn.Module):
    """
    Velocity prediction backbone v_θ(x_t, t) using Mamba.
    """
    def __init__(
        self,
        num_features: int,
        d_model: int,
        n_layers: int,
        expand: int, 
        d_state: int,
        headdim: int,
        mimo_rank: int,
        chunk_size: int,
        dropout: float,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        # Input projection

        self.input_proj = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.SiLU(),
            nn.Identity())

        # Time embedding
        self.time_embed = TimeEmbeddingLayer(d_model)

        # Mamba Backbone
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                n_layer=n_layers,
                expand=expand,
                layer_idx=i,
                d_state=d_state,
                headdim=headdim,
                mimo_rank=mimo_rank,
                chunk_size=chunk_size,
                dropout=dropout,
            )
            for i in range(n_layers)
        ])

        # Output projection back to feature space
        #self.norm_out = nn.LayerNorm(d_model)
        self.norm_out = nn.Identity()
        self.output_proj = nn.Linear(d_model, num_features)

        # Initialize output layer to zero for stable training
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_emb = self.input_proj(x)
        t_emb = self.time_embed(t)

        h = x_emb
        for i, layer in enumerate(self.layers):
            # Pass the index 'i' to the block for the prints to work
            h = layer(h, t_emb, layer_idx=i)

        v = self.output_proj(h)
        return v