"""Synthetic (smoke-test) dataset generation for end-to-end validation."""
from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from librispeech_mrm.utils.io import ensure_dir, write_jsonl


@dataclass
class SmokeConfig:
    data_root: Path
    manifests_dir: Path
    sample_rate: int = 16000
    n_train: int = 24
    n_dev: int = 8
    n_test: int = 8
    min_utt_seconds: float = 1.0
    max_utt_seconds: float = 2.0
    corruption_enabled: bool = False
    snrs_db: list[int] | None = None
    seed: int = 0


def _random_text(rng: random.Random, min_words: int = 2, max_words: int = 6) -> str:
    n_words = rng.randint(min_words, max_words)
    words = []
    for _ in range(n_words):
        wlen = rng.randint(2, 6)
        words.append("".join(rng.choices(string.ascii_lowercase, k=wlen)))
    return " ".join(words)


def _generate_wav(
    rng: random.Random,
    sample_rate: int,
    duration_s: float,
) -> np.ndarray:
    """Generate a synthetic waveform (sum of a few random sine tones)."""
    n_samples = int(sample_rate * duration_s)
    t = np.linspace(0.0, duration_s, n_samples, dtype=np.float32)
    wav = np.zeros(n_samples, dtype=np.float32)
    n_tones = rng.randint(1, 3)
    for _ in range(n_tones):
        freq = rng.uniform(100.0, 800.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(0.05, 0.3)
        wav += amp * np.sin(2.0 * np.pi * freq * t + phase).astype(np.float32)
    # Add a tiny bit of noise so the signal is never perfectly periodic.
    wav += 0.001 * np.random.default_rng(rng.randint(0, 2**31)).standard_normal(n_samples).astype(np.float32)
    return wav


def _generate_noise_wav(rng: random.Random, sample_rate: int, duration_s: float) -> np.ndarray:
    """Generate a noise waveform (white noise)."""
    n_samples = int(sample_rate * duration_s)
    return 0.1 * np.random.default_rng(rng.randint(0, 2**31)).standard_normal(n_samples).astype(np.float32)


def _generate_rir(rng: random.Random, sample_rate: int) -> np.ndarray:
    """Generate a minimal synthetic room impulse response."""
    length = int(0.05 * sample_rate)  # 50 ms
    rir = np.zeros(length, dtype=np.float32)
    rir[0] = 1.0
    # Add a few reflections.
    n_reflections = rng.randint(2, 5)
    for _ in range(n_reflections):
        pos = rng.randint(1, length - 1)
        amp = rng.uniform(0.05, 0.3)
        rir[pos] += amp
    return rir


def generate_smoke_dataset(cfg: SmokeConfig) -> None:
    """Generate synthetic audio and JSONL manifests for smoke testing."""
    rng = random.Random(cfg.seed)

    audio_dir = cfg.data_root / "smoke_audio"
    ensure_dir(audio_dir)
    ensure_dir(cfg.manifests_dir)

    splits = {
        "train": cfg.n_train,
        "dev": cfg.n_dev,
        "test": cfg.n_test,
    }

    # Generate noise and RIR files for corruption (shared across all utterances).
    noise_path: str | None = None
    rir_path: str | None = None
    if cfg.corruption_enabled and cfg.snrs_db:
        noise_dir = cfg.data_root / "smoke_noise"
        rir_dir = cfg.data_root / "smoke_rir"
        ensure_dir(noise_dir)
        ensure_dir(rir_dir)

        noise_wav = _generate_noise_wav(rng, cfg.sample_rate, 10.0)
        noise_file = noise_dir / "noise_000.wav"
        sf.write(str(noise_file), noise_wav, cfg.sample_rate)
        noise_path = str(noise_file)

        rir_wav = _generate_rir(rng, cfg.sample_rate)
        rir_file = rir_dir / "rir_000.wav"
        sf.write(str(rir_file), rir_wav, cfg.sample_rate)
        rir_path = str(rir_file)

    all_rows: dict[str, list[dict[str, Any]]] = {}

    for split_name, n_utts in splits.items():
        rows: list[dict[str, Any]] = []
        for i in range(n_utts):
            utt_id = f"smoke_{split_name}_{i:04d}"
            duration_s = round(rng.uniform(cfg.min_utt_seconds, cfg.max_utt_seconds), 4)
            text = _random_text(rng)
            speaker_id = f"spk_{i % 4:02d}"
            chapter_id = f"ch_{i % 2:02d}"

            wav = _generate_wav(rng, cfg.sample_rate, duration_s)
            wav_path = audio_dir / f"{utt_id}.wav"
            sf.write(str(wav_path), wav, cfg.sample_rate)

            row: dict[str, Any] = {
                "id": utt_id,
                "audio_path": str(wav_path),
                "text": text,
                "duration_s": duration_s,
                "sample_rate": cfg.sample_rate,
                "split": split_name,
                "speaker_id": speaker_id,
                "chapter_id": chapter_id,
            }
            rows.append(row)

        manifest_name = f"smoke_{split_name}.jsonl"
        write_jsonl(cfg.manifests_dir / manifest_name, rows)
        all_rows[split_name] = rows

    # Generate corrupted test manifests.
    if cfg.corruption_enabled and cfg.snrs_db and noise_path and rir_path:
        test_rows = all_rows.get("test", [])
        for snr_db in cfg.snrs_db:
            corrupt_rows: list[dict[str, Any]] = []
            for row in test_rows:
                crow = dict(row)
                crow["corruption"] = {
                    "type": "noise+rir",
                    "snr_db": snr_db,
                    "noise_path": noise_path,
                    "noise_offset_s": 0.0,
                    "rir_path": rir_path,
                }
                corrupt_rows.append(crow)
            manifest_name = f"smoke_test_corrupt_snr{snr_db}.jsonl"
            write_jsonl(cfg.manifests_dir / manifest_name, corrupt_rows)
