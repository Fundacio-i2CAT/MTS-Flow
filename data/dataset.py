import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional

from utils.stft_features import SlidingDFT
class BatchDict(dict):
    """
    Custom dictionary subclass returned by the dataset/dataloader.
    Delegates .to() directly to the primary tensor to maintain compatibility
    with training loops that call batch.to(device) and expect a Tensor.
    """
    def to(self, device, *args, **kwargs):
        key = "x" if "x" in self else "inputs"
        return self[key].to(device, *args, **kwargs)

class TimeSeriesDataset(Dataset):
    """
    Dataset for multivariate time series.
    Expects data of shape (num_samples, sequence_length, num_features).
    """
    def __init__(
        self,
        data: np.ndarray,
        normalization: str = 'none',
        stats: Optional[dict] = None,
        window_size: int = 64,
        num_gated_bins: int = 0,
 
    ):
        self.data = data.astype(np.float32)
        if len(self.data.shape) != 3:
            raise ValueError(f"Expected 3D data (samples, seq_len, features), got {self.data.shape}")
        
        self.num_gated_bins = num_gated_bins
        self.normalization = normalization
        self.window_size = window_size
        self.stats = stats
 
        
        # Precompute sliding DFT features on CPU/GPU
        if num_gated_bins > 0:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sdf_module = SlidingDFT(window_size=window_size).to(device)
            sdf_module.eval()
            
            num_samples = self.data.shape[0]
            num_features = self.data.shape[2]  # Detects total time-series channels
            batch_size = 512
            
            all_channels_list = []
            
            # Loop through each original time-series feature channel independently
            for c in range(num_features):
                # 1. Extract single channel raw time series: shape [num_samples, T, 1]
                x_raw_c = torch.tensor(self.data[:, :, c:c+1], dtype=torch.float32)
                x_tensor_gpu = x_raw_c.to(device)
                
                features_list = []
                # 2. Process chunks through GPU to avoid VRAM explosion
                with torch.no_grad():
                    for i in range(0, num_samples, batch_size):
                        chunk = x_tensor_gpu[i : i + batch_size]
                        chunk_features = sdf_module(chunk)  # shape: [chunk_size, T, num_bins]
                        features_list.append(chunk_features.cpu())
                
                # 3. Consolidate spectral features for channel 'c' back on CPU
                spectral_features_c = torch.cat(features_list, dim=0)  # shape: [num_samples, T, num_bins]
                
                # 4. Group this raw time feature with its own spectral bins side-by-side
                # shape: [num_samples, T, 1 + num_bins]
                channel_block = torch.cat([x_raw_c, spectral_features_c], dim=-1)
                all_channels_list.append(channel_block)
                
            # 5. Interleave all channels perfectly across the feature axis
            # Resulting feature dimension order: [Time_0, Spec_0_0..., Time_1, Spec_1_0...]
            # Final Shape: [num_samples, T, num_features * (1 + num_bins)]
            self.joint_data = torch.cat(all_channels_list, dim=-1)
        else:
            self.joint_data = torch.tensor(self.data, dtype=torch.float32)
        self._normalize()

    def _normalize(self):
            x = self.joint_data
            norm = self.normalization
            
            if norm == 'none' or norm is None:
                return

            # Ensure x is a PyTorch Tensor
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
    
            if norm == 'min_max':
                min_vals = x.amin(dim=1, keepdim=True)
                max_vals = x.amax(dim=1, keepdim=True)
                self.joint_data = (x - min_vals) / (max_vals - min_vals + 1e-5)
    
            elif norm == 'global_min_max':
                if self.stats is not None and 'min' in self.stats and 'max' in self.stats:
                    min_val, max_val = self.stats['min'], self.stats['max']
                else:
                    if self.stats is None:
                        self.stats = {}
                    # Calculate global min/max across samples (dim 0) and sequence length (dim 1)
                    # Resulting shape: [1, 1, total_features] for clean broadcasting
                    min_val = torch.amin(x, dim=(0, 1), keepdim=True)
                    max_val = torch.amax(x, dim=(0, 1), keepdim=True)
                    self.stats['min'], self.stats['max'] = min_val, max_val
    
                self.joint_data = (x - min_val) / (max_val - min_val + 1e-5)
    
            elif norm in ['standarization', 'standardization', 'standard']:
                if self.stats is not None and 'mean' in self.stats and 'std' in self.stats:
                    mean, std = self.stats['mean'], self.stats['std']
                else:
                    if self.stats is None:
                        self.stats = {}
                    mean = x.mean(dim=1, keepdim=True)
                    std = x.std(dim=1, keepdim=True)
                    self.stats['mean'], self.stats['std'] = mean, std
    
                # FIXED: Assigned to self.joint_data and updated epsilon to 1e-5
                self.joint_data = (x - mean) / (std + 1e-5)
            else:
                raise ValueError(f"Unknown normalization: {norm}")

    def __len__(self) -> int:
        return len(self.joint_data)

    def __getitem__(self, idx: int) -> BatchDict:
        x = self.joint_data[idx]
        return BatchDict({
            "x": x,
            "inputs": x
        })
