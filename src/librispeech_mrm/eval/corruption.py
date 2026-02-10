from __future__ import annotations

from typing import Any

import torch

from librispeech_mrm.data.audio import load_audio_mono, load_audio_segment_mono
from librispeech_mrm.data.corrupt import apply_noise_and_rir


def maybe_apply_corruption(
    wav: torch.Tensor,
    *,
    sample_rate: int,
    corruption: dict[str, Any] | None,
) -> torch.Tensor:
    if not corruption:
        return wav

    ctype = str(corruption.get("type", ""))
    if ctype != "noise+rir":
        raise ValueError(f"Unknown corruption.type: {ctype}")

    snr_db = float(corruption["snr_db"])
    noise_path = str(corruption["noise_path"])
    noise_offset_s = float(corruption.get("noise_offset_s", 0.0))
    rir_path = str(corruption["rir_path"])

    rir = load_audio_mono(rir_path, target_sr=sample_rate)

    offset_frames = int(noise_offset_s * sample_rate)
    num_frames = int(wav.shape[0])
    noise = load_audio_segment_mono(
        noise_path,
        target_sr=sample_rate,
        offset_frames=offset_frames,
        num_frames=num_frames,
    )
    if noise.shape[0] < num_frames:
        # If segment hits EOF, pad with zeros (deterministic).
        noise = torch.nn.functional.pad(noise, (0, num_frames - noise.shape[0]))

    return apply_noise_and_rir(wav, noise, rir, snr_db=snr_db)
