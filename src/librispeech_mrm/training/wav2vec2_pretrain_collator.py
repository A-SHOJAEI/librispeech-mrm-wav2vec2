from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class MaskParams:
    time_mask_prob: float
    time_mask_length: int


def _compute_mask_indices(
    *,
    batch_size: int,
    seq_length: int,
    lengths: np.ndarray,
    mask_prob: float,
    mask_length: int,
    rng: np.random.Generator,
    min_masks: int = 1,
) -> np.ndarray:
    if mask_length <= 0:
        raise ValueError("mask_length must be > 0")

    mask = np.zeros((batch_size, seq_length), dtype=np.bool_)

    for i in range(batch_size):
        L = int(lengths[i])
        L = max(1, min(L, seq_length))
        if L - mask_length <= 0:
            continue

        num_spans = int(mask_prob * L / mask_length + rng.random())
        num_spans = max(min_masks, num_spans)
        num_spans = min(num_spans, L // mask_length)
        if num_spans <= 0:
            continue

        start_max = L - mask_length
        # Without replacement for stability.
        starts = rng.choice(start_max + 1, size=num_spans, replace=False)
        for s in starts:
            mask[i, int(s) : int(s) + mask_length] = True

    return mask


def _sample_negative_indices(
    *,
    mask_time_indices: np.ndarray,
    lengths: np.ndarray,
    num_negatives: int,
    rng: np.random.Generator,
) -> np.ndarray:
    bsz, seq_len = mask_time_indices.shape
    out = np.zeros((bsz, seq_len, num_negatives), dtype=np.int64)

    for i in range(bsz):
        L = int(lengths[i])
        L = max(1, min(L, seq_len))
        if L <= 1:
            continue

        valid = np.arange(L, dtype=np.int64)
        for t in range(seq_len):
            if not mask_time_indices[i, t]:
                continue
            # Avoid sampling the same timestep as the positive.
            choices = valid[valid != t]
            if choices.size == 0:
                continue
            out[i, t] = rng.choice(choices, size=num_negatives, replace=True)

    return out


class Wav2Vec2PretrainCollator:
    def __init__(
        self,
        *,
        model,
        seed: int,
        time_mask_prob: float,
        time_mask_length: int,
        num_negatives: int,
    ):
        self.model = model
        self.seed = int(seed)
        self.num_negatives = int(num_negatives)
        self.params = MaskParams(time_mask_prob=float(time_mask_prob), time_mask_length=int(time_mask_length))
        self.step = 0

    def set_mask_params(self, *, time_mask_prob: float, time_mask_length: int) -> None:
        self.params = MaskParams(time_mask_prob=float(time_mask_prob), time_mask_length=int(time_mask_length))

    def set_step(self, step: int) -> None:
        self.step = int(step)

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        # batch items are from ManifestAudioDataset.collate_wavs shape.
        # Here we expect item dicts with "wav".
        wavs = [b["wav"] for b in batch]
        lengths = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)
        max_len = int(lengths.max().item())

        x = torch.zeros((len(wavs), max_len), dtype=torch.float32)
        attn = torch.zeros((len(wavs), max_len), dtype=torch.long)
        for i, w in enumerate(wavs):
            L = w.shape[0]
            x[i, :L] = w
            attn[i, :L] = 1

        with torch.no_grad():
            feat_lengths = self.model._get_feat_extract_output_lengths(lengths)
        feat_lengths_np = feat_lengths.cpu().numpy().astype(np.int64)
        max_feat_len = int(feat_lengths.max().item())

        rng = np.random.default_rng(self.seed + self.step)
        mask_time = _compute_mask_indices(
            batch_size=len(wavs),
            seq_length=max_feat_len,
            lengths=feat_lengths_np,
            mask_prob=self.params.time_mask_prob,
            mask_length=self.params.time_mask_length,
            rng=rng,
        )
        neg_idx = _sample_negative_indices(
            mask_time_indices=mask_time,
            lengths=feat_lengths_np,
            num_negatives=self.num_negatives,
            rng=rng,
        )

        return {
            "input_values": x,
            "attention_mask": attn,
            "mask_time_indices": torch.tensor(mask_time, dtype=torch.bool),
            "sampled_negative_indices": torch.tensor(neg_idx, dtype=torch.long),
        }
