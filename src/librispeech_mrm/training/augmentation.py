from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torchaudio
import torch

from librispeech_mrm.data.audio import load_audio_mono, load_audio_segment_mono
from librispeech_mrm.data.corrupt import apply_noise_and_rir


@dataclass(frozen=True)
class NoiseRIRAugmentConfig:
    enabled: bool
    probability: float
    snr_db_min: float
    snr_db_max: float
    noise_paths: list[str]
    rir_paths: list[str]


class NoiseRIRAugment:
    def __init__(self, cfg: NoiseRIRAugmentConfig, *, sample_rate: int, seed: int):
        self.cfg = cfg
        self.sample_rate = int(sample_rate)
        self.seed = int(seed)
        self._noise_info: dict[str, tuple[int, int]] = {}  # path -> (num_frames, sr)

    def _info(self, path: str) -> tuple[int, int]:
        if path not in self._noise_info:
            info = torchaudio.info(path)
            self._noise_info[path] = (int(info.num_frames), int(info.sample_rate))
        return self._noise_info[path]

    def _load_noise_segment(self, path: str, *, offset_s: float, target_len: int) -> torch.Tensor:
        nf, nsr = self._info(path)
        offset_frames = int(offset_s * nsr)

        # Load enough source frames so that after resample we have at least target_len.
        approx_src = int((target_len * nsr) / self.sample_rate) + 8
        max_offset = max(0, nf - approx_src - 1)
        offset_frames = min(max(0, offset_frames), max_offset)

        n = load_audio_segment_mono(
            path,
            target_sr=self.sample_rate,
            offset_frames=offset_frames,
            num_frames=approx_src,
        )
        if n.shape[0] < target_len:
            n = torch.nn.functional.pad(n, (0, target_len - n.shape[0]))
        return n[:target_len]

    def apply(self, wav: torch.Tensor, wav_lengths: torch.Tensor, *, step: int) -> torch.Tensor:
        if not self.cfg.enabled:
            return wav
        if not self.cfg.noise_paths or not self.cfg.rir_paths:
            return wav

        out = wav.clone()
        bsz = out.shape[0]

        for i in range(bsz):
            rng = random.Random((self.seed & 0xFFFFFFFF) ^ ((step + 1) * 1000003) ^ (i * 9176))
            if rng.random() > float(self.cfg.probability):
                continue

            L = int(wav_lengths[i].item())
            if L <= 0:
                continue

            noise_path = rng.choice(self.cfg.noise_paths)
            rir_path = rng.choice(self.cfg.rir_paths)
            snr_db = rng.uniform(float(self.cfg.snr_db_min), float(self.cfg.snr_db_max))

            # Uniform offset in seconds based on noise duration.
            nf, nsr = self._info(noise_path)
            noise_dur_s = float(nf) / float(nsr)
            max_off = max(0.0, noise_dur_s - (float(L) / float(self.sample_rate)) - 0.01)
            offset_s = rng.uniform(0.0, max_off) if max_off > 0 else 0.0

            clean = out[i, :L].contiguous()
            noise = self._load_noise_segment(noise_path, offset_s=offset_s, target_len=L)
            rir = load_audio_mono(rir_path, target_sr=self.sample_rate)

            y = apply_noise_and_rir(clean, noise, rir, snr_db=snr_db)
            out[i, :L] = y

        return out
