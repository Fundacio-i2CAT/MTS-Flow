import math
import torch
import torch.nn as class_nn

class SinusoidalPositionEmbeddings(torch.nn.Module):
    """
    Standard sinusoidal embeddings for time steps.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: (B,)
        Returns:
            embeddings: (B, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # Handle odd dimensions
        if self.dim % 2 == 1:
            embeddings = torch.nn.functional.pad(embeddings, (0, 1))
            
        return embeddings

class TimeEmbeddingLayer(torch.nn.Module):
    """
    Projects scalar time into an embedding vector.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.time_embed = torch.nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            torch.nn.Linear(dim, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) time steps
        Returns:
            (B, dim)
        """
        return self.time_embed(t)
