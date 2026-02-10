from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from librispeech_mrm.data.audio import load_audio_mono
from librispeech_mrm.utils.io import read_jsonl


@dataclass(frozen=True)
class ManifestRow:
    id: str
    audio_path: str
    text: str
    duration_s: float
    sample_rate: int
    split: str
    speaker_id: str
    chapter_id: str
    corruption: dict[str, Any] | None = None


def _row_from_dict(d: dict[str, Any]) -> ManifestRow:
    return ManifestRow(
        id=str(d["id"]),
        audio_path=str(d["audio_path"]),
        text=str(d.get("text", "")),
        duration_s=float(d.get("duration_s", 0.0)),
        sample_rate=int(d.get("sample_rate", 16000)),
        split=str(d.get("split", "")),
        speaker_id=str(d.get("speaker_id", "")),
        chapter_id=str(d.get("chapter_id", "")),
        corruption=d.get("corruption"),
    )


class ManifestAudioDataset(Dataset):
    def __init__(self, manifest_path: str | Path | list[str | Path], *, target_sr: int):
        if isinstance(manifest_path, list):
            paths = [Path(p) for p in manifest_path]
        else:
            paths = [Path(manifest_path)]

        self.manifest_paths = paths
        rows_all: list[dict[str, Any]] = []
        for p in paths:
            rows_all.extend(read_jsonl(p))
        self.rows = [_row_from_dict(r) for r in rows_all]
        self.target_sr = int(target_sr)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.rows[idx]
        wav = load_audio_mono(r.audio_path, target_sr=self.target_sr)
        return {
            "id": r.id,
            "wav": wav,
            "text": r.text,
            "corruption": r.corruption,
        }


def collate_wavs(batch: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [b["id"] for b in batch]
    wavs = [b["wav"] for b in batch]
    texts = [b.get("text", "") for b in batch]
    corruptions = [b.get("corruption") for b in batch]

    lengths = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)
    max_len = int(lengths.max().item())

    x = torch.zeros((len(wavs), max_len), dtype=torch.float32)
    attn = torch.zeros((len(wavs), max_len), dtype=torch.long)
    for i, w in enumerate(wavs):
        L = w.shape[0]
        x[i, :L] = w
        attn[i, :L] = 1

    return {
        "ids": ids,
        "input_values": x,
        "attention_mask": attn,
        "lengths": lengths,
        "texts": texts,
        "corruptions": corruptions,
    }
