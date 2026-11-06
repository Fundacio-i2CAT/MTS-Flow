import torch.nn as nn
from architecture.mts_gan_architecture import Mamba3FlowMatchingModel


def build_model(config: dict[str, any]) -> nn.Module:
    """
    Factory function for the model backbone.
    Allows easy extension to Transformers or other architectures.
    """
    model_cfg = config.get("model", {})
    arch_type = model_cfg.get("architecture", "mamba_flow_matching")
    dataset_cfg = config.get("dataset", {})
    num_features = config.get("dataset", {}).get("num_features", 10)
    
    if arch_type == "mamba_3_flow_matching":
        return Mamba3FlowMatchingModel(
            num_features=num_features,
            d_model=model_cfg.get("d_model", 128),
            n_layers=model_cfg.get("n_layers", 4),
            expand=model_cfg.get("expand", 2), 
            d_state=model_cfg.get("d_state", 16),
            headdim=model_cfg.get("headdim", 32),
            mimo_rank=model_cfg.get("mimo_rank", 2),
            chunk_size=model_cfg.get("chunk_size", 16),
            dropout=model_cfg.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown architecture: {arch_type}")