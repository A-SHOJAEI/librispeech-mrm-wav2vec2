from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


@dataclass(frozen=True)
class ConformerCTCConfig:
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400

    conformer_dim: int = 256
    num_layers: int = 12
    num_heads: int = 4
    ffn_dim: int = 1024
    conv_kernel_size: int = 31
    dropout: float = 0.1


class ConformerCTCModel(nn.Module):
    def __init__(self, cfg: ConformerCTCConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            n_mels=cfg.n_mels,
            center=False,
            power=2.0,
        )
        self.amptodb = torchaudio.transforms.AmplitudeToDB(stype="power")

        self.in_proj = nn.Linear(cfg.n_mels, cfg.conformer_dim)
        self.encoder = torchaudio.models.Conformer(
            input_dim=cfg.conformer_dim,
            num_heads=cfg.num_heads,
            ffn_dim=cfg.ffn_dim,
            num_layers=cfg.num_layers,
            depthwise_conv_kernel_size=cfg.conv_kernel_size,
            dropout=cfg.dropout,
        )
        self.out_proj = nn.Linear(cfg.conformer_dim, vocab_size)

    def wav_to_features(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [B, T]
        feats = self.melspec(wav)  # [B, n_mels, frames]
        feats = self.amptodb(feats)
        feats = feats.transpose(1, 2).contiguous()  # [B, frames, n_mels]
        return feats

    def wav_lengths_to_feat_lengths(self, wav_lengths: torch.Tensor) -> torch.Tensor:
        # center=False -> predictable framing.
        n_fft = self.cfg.n_fft
        hop = self.cfg.hop_length
        wl = wav_lengths.to(torch.long)
        fl = torch.where(
            wl >= n_fft,
            1 + (wl - n_fft) // hop,
            torch.ones_like(wl),
        )
        return fl

    def forward(self, wav: torch.Tensor, wav_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.wav_to_features(wav)
        feat_lengths = self.wav_lengths_to_feat_lengths(wav_lengths)

        x = self.in_proj(feats)
        x, out_lengths = self.encoder(x, feat_lengths)
        logits = self.out_proj(x)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, out_lengths


def greedy_ctc_decode(log_probs: torch.Tensor, blank_id: int) -> list[list[int]]:
    # log_probs: [B, T, V]
    ids = torch.argmax(log_probs, dim=-1)  # [B, T]
    hyps: list[list[int]] = []
    for seq in ids.tolist():
        out: list[int] = []
        prev = None
        for t in seq:
            if t == blank_id:
                prev = t
                continue
            if prev is not None and t == prev:
                continue
            out.append(t)
            prev = t
        hyps.append(out)
    return hyps
