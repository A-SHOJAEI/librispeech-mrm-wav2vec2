"""Audio loading utilities."""
from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch


def load_audio_mono(path: str | Path, target_sr: int = 16000) -> torch.Tensor:
    """Load audio file to a 1D float32 tensor at target_sr."""
    wav, sr = sf.read(str(path), dtype="float32")
    wav = torch.from_numpy(wav).float()
    if wav.ndim > 1:
        wav = wav.mean(dim=-1)
    if sr != target_sr:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def load_audio_segment_mono(
    path: str | Path,
    *,
    target_sr: int = 16000,
    offset_frames: int = 0,
    num_frames: int = -1,
) -> torch.Tensor:
    """Load a segment of audio to a 1D float32 tensor at target_sr.

    Parameters
    ----------
    path : path to audio file
    target_sr : desired sample rate for output
    offset_frames : start frame in the *native* sample rate of the file
    num_frames : number of frames to read (-1 = read to end)
    """
    wav, sr = sf.read(
        str(path),
        start=max(0, offset_frames),
        frames=num_frames,
        dtype="float32",
    )
    wav = torch.from_numpy(wav).float()
    if wav.ndim > 1:
        wav = wav.mean(dim=-1)
    if sr != target_sr:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav
