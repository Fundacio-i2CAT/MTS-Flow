import torch

class FlowMatcher:
    """
    Flow Matching logic.
    Provides methods to sample intermediate noisy states and target velocities.
    """
    def __init__(self, sigma_min: float = 1e-4):
        self.sigma_min = sigma_min

    def compute_target(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Interpolates between x_0 (noise) and x_1 (data) at time t.
        Args:
            x_0: (B, seq_len, num_features) Noise
            x_1: (B, seq_len, num_features) Real Data
            t:   (B, 1, 1) Timestep in [0, 1]
            
        Returns:
            x_t: Intermediate state
            u_t: Target velocity
        """
        t_expanded = t.view(-1, 1, 1).expand_as(x_1)

        # x_t =  t * x_1 + (1 - t) * x_0
        x_t = t_expanded * x_1 + (1 - t_expanded) * x_0
        u_t = x_1 - x_0

        return x_t, u_t

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Samples uniform time t in [0, 1]"""
        return torch.rand((batch_size,), device=device)

    def sample_noise(self, shape, device):
        return torch.randn(shape, device=device)
