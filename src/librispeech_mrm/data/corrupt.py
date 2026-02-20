"""Audio corruption utilities (additive noise + RIR convolution)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_noise_and_rir(
    clean: torch.Tensor,
    noise: torch.Tensor,
    rir: torch.Tensor,
    *,
    snr_db: float,
) -> torch.Tensor:
    """Apply additive noise at a given SNR and convolve with a room impulse response.

    All inputs are 1-D float tensors (mono waveforms at the same sample rate).

    Steps:
    1. Convolve the clean signal with the RIR.
    2. Scale the noise to achieve the desired SNR relative to the reverberant clean signal.
    3. Add scaled noise and return.
    """
    # 1. Convolve clean with RIR.
    # Use 1-d convolution (scipy-style full convolution, then truncate to original length).
    if rir.ndim == 1 and rir.numel() > 0:
        # Normalize RIR to unit energy so that convolution preserves overall level.
        rir_norm = rir / (rir.norm() + 1e-12)
        # F.conv1d expects [batch, channels, length]
        reverb = F.conv1d(
            clean.view(1, 1, -1),
            rir_norm.flip(0).view(1, 1, -1),
            padding=rir_norm.numel() - 1,
        ).squeeze()
        reverb = reverb[: clean.numel()]
    else:
        reverb = clean

    # 2. Scale noise to target SNR.
    sig_power = (reverb ** 2).mean().clamp(min=1e-12)
    noise_power = (noise ** 2).mean().clamp(min=1e-12)

    snr_linear = 10.0 ** (snr_db / 10.0)
    scale = (sig_power / (noise_power * snr_linear)).sqrt()

    # Ensure noise length matches reverb length.
    n = noise
    if n.numel() < reverb.numel():
        n = F.pad(n, (0, reverb.numel() - n.numel()))
    n = n[: reverb.numel()]

    return reverb + scale * n
