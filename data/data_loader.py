from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from .dataset import TimeSeriesDataset, BatchDict
from utils.stft_features import SlidingDFT  # Import as requested by instruction 1
import h5py

def collate_fn(batch):
    """
    Collates a list of samples into a batch, wrapping the resulting dictionary
    in BatchDict so that .to() behaves correctly during training.
    """
    collated = default_collate(batch)
    if isinstance(collated, dict):
        return BatchDict(collated)
    return collated

def get_dataloader(config: dict[str, any],  split: str = "train",
                   stats: dict | None = None) -> DataLoader:
    """Builds and returns a dataloader based on the configuration."""
    dataset_cfg = config.get("dataset", {})
    
    if split == "train": 
        split = "train"
        stats = None
        
    data_path = dataset_cfg.get("data_path")
    
    if not data_path:
        raise ValueError("dataset.data_path must be specified in the configuration.")
    
    with h5py.File(data_path, 'r') as f:
        if split not in f:
            raise KeyError(f"Split '{split}' not found in h5 file. Available keys: {list(f.keys())}")
        data = f[split][:]

    # Extract configurable SlidingDFT parameters
    window_size = dataset_cfg.get("window_size", 64)
    num_gated_bins = dataset_cfg.get("num_gated_bins", 0)
    
    dataset = TimeSeriesDataset(
        data=data,
        normalization=dataset_cfg.get("normalization", "global_min_max"),
        stats=stats,
        window_size=window_size,
        num_gated_bins=num_gated_bins,
    )
    
    return DataLoader(
        dataset,
        batch_size=dataset_cfg.get("batch_size", 64),
        shuffle=(split == "train"),
        num_workers=dataset_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=(split == "train"),
        collate_fn=collate_fn
    )
