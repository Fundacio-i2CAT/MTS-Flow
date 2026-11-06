import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class SlidingDFT(nn.Module):
    """
    Computes the Short-Time Fourier Transform (STFT) recursively across time steps.
    Operates in O(T) complexity (linear time) by updating the DFT spectrum from
    step t-1 to step t.
    """
    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = window_size
        self.num_bins = window_size // 2 + 1
        
        # Precompute the twiddle factors: e^(j * 2 * pi * k / W)
        k = torch.arange(self.num_bins, dtype=torch.float32)
        twiddle = torch.exp(1j * 2.0 * np.pi * k / window_size)
        self.register_buffer("twiddle", twiddle)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [Batch, T, 1]
            
        Returns:
            Output tensor of shape [Batch, T, num_bins] representing the absolute magnitude.
        """
        batch_size, T, _ = x.shape
        device = x.device
        
        # Pad the sequence on the left by window_size zeros
        # This allows us to retrieve entering and leaving samples at any step t
        # without boundary checks.
        x_pad = torch.nn.functional.pad(x, (0, 0, self.window_size, 0), mode='constant', value=0.0)
        
        # Initialize spectrum X_{-1} to zeros
        X = torch.zeros((batch_size, self.num_bins), dtype=torch.complex64, device=device)
        
        # Output tensor for absolute magnitude
        out = torch.empty((batch_size, T, self.num_bins), dtype=torch.float32, device=device)
        
        # Recursive SDFT update:
        # X_t[k] = e^(j * 2 * pi * k / W) * (X_{t-1}[k] + x_in - x_out)
        for t in range(T):
            x_in = x_pad[:, t + self.window_size]   # shape: [Batch, 1]
            x_out = x_pad[:, t]                     # shape: [Batch, 1]
            delta = x_in - x_out                    # shape: [Batch, 1]
            
            # Update the DFT spectrum
            X = self.twiddle * (X + delta)
            
            # Store the absolute magnitude
            out[:, t] = X.abs()
            
        return out
