"""
loss.py  Configurable loss factory for Flow Matching on multivariate time series.
"""
from __future__ import annotations
import abc
import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruct_clean_sample(
    x_t: torch.Tensor,
    v_pred: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct the clean sample x1_hat from the noisy interpolation x_t,
    the predicted velocity v_pred, and the timestep t.

    Uses the Flow identity:
        x_t = t * x_1 + (1 - t) * x_0
        v   = x_1 - x_0
        => x_1 = x_t + (1 - t) * v

    NOTE: Used only for auxiliary spectral losses. The ODE solver is unchanged.

    Args:
        x_t:    (B, L, F) Noisy interpolated state.
        v_pred: (B, L, F) Model-predicted velocity.
        t:      (B,) Scalar timestep per sample.

    Returns:
        x1_hat: (B, L, F) Reconstructed clean sample.
    """
    # Broadcast t to (B, 1, 1) so it covers (L, F) automatically
    t_bcast = t.view(-1, 1, 1)
    return x_t + (1.0 - t_bcast) * v_pred


def extract_raw_channels(tensor: torch.Tensor, num_features: int, num_bins: int) -> torch.Tensor:
    """
    Extract only the raw time-series channels from an interleaved feature tensor.

    Channel layout per feature block (stride = 1 + num_bins):
        [raw_ch, spec_ch_0, spec_ch_1, ..., spec_ch_{num_bins-1}]

    Args:
        tensor:       (B, L, num_features * (1 + num_bins)) Interleaved tensor.
        num_features: Number of original time-series variables.
        num_bins:     Number of spectral bins per variable.

    Returns:
        (B, L, num_features) Tensor containing only the raw channels.
    """
    stride = 1 + num_bins
    raw_indices = [f * stride for f in range(num_features)]
    return tensor[:, :, raw_indices]


def fft_magnitude_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """
    Compute L1 loss between the rfft magnitude spectra of two signals.

    FFT is applied along the time dimension (dim=1).

    Args:
        real: (B, L, C) Ground-truth raw channels.
        fake: (B, L, C) Reconstructed raw channels (x1_hat).

    Returns:
        Scalar L1 loss between |rfft(real)| and |rfft(fake)|.
    """
    real_fft = torch.fft.rfft(real, dim=1).abs()
    fake_fft = torch.fft.rfft(fake, dim=1).abs()
    return F.l1_loss(fake_fft, real_fft)


def _infer_num_bins(model: nn.Module, config: dict) -> int:
    """
    Resolve num_bins from the model object (preferred) or from the YAML config.

    Returns 0 when the model has no spectral channels (plain Mamba variants),
    which gracefully degrades extract_raw_channels to a full-channel identity.
    """
    if hasattr(model, "num_bins"):
        return int(model.num_bins)

    dataset_cfg = config.get("dataset", {})
    model_cfg   = config.get("model",   {})

    num_bins = dataset_cfg.get("num_bins", dataset_cfg.get("number_bins", 0))
    if not num_bins and model_cfg.get("architecture") == "mamba_3_stft":
        window_size = dataset_cfg.get("window_size", 64)
        num_bins = window_size // 2 + 1

    return int(num_bins)


def _infer_num_features(model: nn.Module, config: dict) -> int:
    """Resolve the number of original time-series variables."""
    if hasattr(model, "num_features"):
        return int(model.num_features)
    return int(config.get("dataset", {}).get("num_features", 1))


class BaseLoss(abc.ABC):
    """
    Interface that every concrete loss must implement.

    Subclasses receive all relevant training quantities and return a dictionary
    containing the total loss (used for backpropagation) together with any
    auxiliary losses that should be logged.
    """

    @abc.abstractmethod
    def __call__(
        self,
        *,
        model: nn.Module,
        x_t: torch.Tensor,
        x_1: torch.Tensor,
        u_t: torch.Tensor,
        v_pred: torch.Tensor,
        t: torch.Tensor,
        epoch: int,
    ) -> dict[str, torch.Tensor | float]:
        """
        Compute the training loss.

        Args:
            model:  The velocity model (used to infer spectral layout at runtime).
            x_t:    (B, L, F) Noisy interpolated state.
            x_1:    (B, L, F) Ground-truth clean sample.
            u_t:    (B, L, F) Target velocity (x_1 - x_0).
            v_pred: (B, L, F) Model-predicted velocity.
            t:      (B,) Timestep per sample.
            epoch:  Current training epoch.

        Returns:
            Dictionary with at least:

                {
                    "loss": Tensor,          # used for backward()
                    "loss_raw": Tensor,      # logging
                    "loss_fft": Tensor,      # logging
                    "fft_weight": float,     # effective FFT weight
                }

            Additional loss components may be added by subclasses.
        """

class NormalLoss(BaseLoss):
    """
    Standard Flow MSE loss.

        loss = MSE(v_pred, u_t)

    Reproduces the original training loop behaviour exactly.
    Backward-compatible default when loss_type is omitted from the config.
    """

    def __call__(self, *, model, x_t, x_1, u_t, v_pred, t, epoch) -> torch.Tensor:
        loss = F.mse_loss(v_pred, u_t)

        return {
            "loss": loss,
            "loss_raw": loss.detach(),
            "loss_fft": torch.tensor(
                0.0, device=loss.device, dtype=loss.dtype
            ),
            "fft_weight": 0.0,
        }


class SpectralFFTLoss(BaseLoss):
    """
    Experiment 1  FFT Consistency Loss.

    Supervises ONLY the raw velocity channels (channel 0 of every feature block).
    No spectral-channel MSE is applied; that is reserved for future experiments.

    Objective:
        loss_raw = MSE(v_pred_raw, u_t_raw)

        x1_hat   = x_t + (1 - t) * v_pred

        loss_fft = L1( |rfft(x1_hat_raw, dim=1)|,
                       |rfft(x1_raw,     dim=1)| )

        loss     = raw_loss_weight  * loss_raw
                 + spectral_fft_weight * loss_fft

    Configuration keys (under training:):
        raw_loss_weight:    Weight for the raw-channel velocity MSE.  (default: 1.0)
        spectral_fft_weight: Weight for the FFT magnitude auxiliary.  (default: 0.02)

    Intentionally disabled for this experiment:
        spectral_loss_weight  would weight MSE on spectral-bin velocity channels.
    """

    def __init__(
        self,
        config: dict,
        raw_loss_weight: float,
        fft_loss_weight: float,
    ):
        self.config  = config
        self.raw_w   = raw_loss_weight
        self.fft_w   = fft_loss_weight

    def __call__(self, *, model, x_t, x_1, u_t, v_pred, t, epoch) -> torch.Tensor:
        nf = _infer_num_features(model, self.config)
        nb = _infer_num_bins(model, self.config)
        fft_start_epoch = int(
            self.config["training"].get("fft_start_epoch", 0)
        )
        # --- Raw-channel velocity MSE ---
        u_raw  = extract_raw_channels(u_t,    nf, nb)
        vp_raw = extract_raw_channels(v_pred, nf, nb)
        loss_raw = F.mse_loss(vp_raw, u_raw)

        # --- FFT consistency auxiliary loss ---
        x1_hat    = reconstruct_clean_sample(x_t, v_pred, t)
        x1_raw    = extract_raw_channels(x_1,    nf, nb)
        x1hat_raw = extract_raw_channels(x1_hat, nf, nb)
        loss_fft  = fft_magnitude_loss(x1_raw, x1hat_raw)

        fft_weight = self.fft_w if epoch >= fft_start_epoch else 0.0
        total_loss = (
            self.raw_w * loss_raw
            + fft_weight * loss_fft
        )

        return {
            "loss": total_loss,
            "loss_raw": loss_raw.detach(),
            "loss_fft": loss_fft.detach(),
            "fft_weight": fft_weight,
        }


# Active loss modes only. Future modes are commented out below.
_ACTIVE_LOSS_REGISTRY: dict[str, type] = {
    "normal":       NormalLoss,
    "spectral_fft": SpectralFFTLoss,
}


def build_loss(config: dict) -> BaseLoss:
    """
    Factory function for the training loss.

    Reads ``config["training"]["loss_type"]`` (default: ``"normal"``).
    Existing YAML configs that omit loss_type default to NormalLoss and are
    fully backward compatible.

    Active configuration keys (all under ``training:``):
        loss_type:           normal | spectral_fft         (default: normal)
        raw_loss_weight:     Weight for the raw-channel velocity MSE.  (default: 1.0)
        spectral_fft_weight: Weight for the FFT magnitude auxiliary.   (default: 0.02)

    Preserved but currently ignored keys:
        spectral_loss_weight  reserved for future spectral-channel MSE supervision.
        spectral_sdft_weight  reserved for future SlidingDFT consistency loss.

    Args:
        config: Full training configuration dictionary.

    Returns:
        A callable ``BaseLoss`` instance.

    Raises:
        ValueError:          If an unknown loss_type is specified.
        NotImplementedError: If a valid but currently disabled mode is requested.

    Example::

        loss_fn = build_loss(config)
        loss = loss_fn(
            model=model,
            x_t=x_t, x_1=x_1, u_t=u_t,
            v_pred=v_pred, t=t,
        )
    """
    training_cfg = config.get("training", {})

    loss_type = training_cfg.get("loss_type", "normal").lower()

    # --- Guard: unknown loss type ---
    _all_known = set(_ACTIVE_LOSS_REGISTRY) | {"spectral_sdft", "spectral_combined"}
    if loss_type not in _all_known:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. "
            f"Known types: {sorted(_all_known)}"
        )

    # --- Guard: disabled future modes ---
    if loss_type == "spectral_sdft":
        raise NotImplementedError(
            "loss_type='spectral_sdft' is reserved for a future experiment "
            "(Experiment 2  Sliding-DFT consistency loss). "
            "To enable it, uncomment SpectralSDFTLoss and its factory branch in loss.py."
        )
    if loss_type == "spectral_combined":
        raise NotImplementedError(
            "loss_type='spectral_combined' is reserved for a future experiment "
            "(Experiment 3  combined FFT + SDFT consistency). "
            "To enable it, uncomment SpectralCombinedLoss and its factory branch in loss.py."
        )

    # --- normal: exact reproduction of original training loop ---
    if loss_type == "normal":
        return NormalLoss()

    # --- spectral_fft: Experiment 1 (ACTIVE) ---
    if loss_type == "spectral_fft":
        raw_w = float(training_cfg.get("raw_loss_weight",    1.0))
        fft_w = float(training_cfg.get("spectral_fft_weight", 0.02))

        return SpectralFFTLoss(
            config=config,
            raw_loss_weight=raw_w,
            fft_loss_weight=fft_w,
        )

    raise ValueError(f"Unhandled loss_type: '{loss_type}'")
