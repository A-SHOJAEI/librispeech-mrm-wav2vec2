from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2ForPreTraining


def build_wav2vec2_config(model_cfg: dict[str, Any], masking_cfg: dict[str, Any] | None = None) -> Wav2Vec2Config:
    masking_cfg = masking_cfg or {}

    cfg = Wav2Vec2Config(
        hidden_size=int(model_cfg.get("hidden_size", 768)),
        num_hidden_layers=int(model_cfg.get("num_hidden_layers", 12)),
        num_attention_heads=int(model_cfg.get("num_attention_heads", 12)),
        intermediate_size=int(model_cfg.get("intermediate_size", 3072)),
        conv_dim=list(model_cfg.get("conv_dim", [512, 512, 512, 512, 512, 512, 512])),
        conv_stride=list(model_cfg.get("conv_stride", [5, 2, 2, 2, 2, 2, 2])),
        conv_kernel=list(model_cfg.get("conv_kernel", [10, 3, 3, 3, 3, 2, 2])),
        num_codevectors_per_group=int(model_cfg.get("num_codevectors_per_group", 320)),
        num_codevector_groups=int(model_cfg.get("num_codevector_groups", 2)),
        feat_extract_norm=str(model_cfg.get("feat_extract_norm", "group")),
        feat_extract_activation=str(model_cfg.get("feat_extract_activation", "gelu")),
        mask_time_prob=float(masking_cfg.get("time_mask_prob", 0.65)),
        mask_time_length=int(masking_cfg.get("time_mask_length", 10)),
        mask_feature_prob=float(masking_cfg.get("feature_mask_prob", 0.0)),
        mask_feature_length=int(masking_cfg.get("feature_mask_length", 10)),
    )

    # Defaults aligned with typical wav2vec2 pretraining.
    cfg.num_negatives = int(model_cfg.get("num_negatives", 100))
    cfg.codevector_dim = int(model_cfg.get("codevector_dim", cfg.hidden_size))

    return cfg


def build_pretrain_model(model_cfg: dict[str, Any], masking_cfg: dict[str, Any]) -> Wav2Vec2ForPreTraining:
    cfg = build_wav2vec2_config(model_cfg, masking_cfg)
    return Wav2Vec2ForPreTraining(cfg)


def build_ctc_model(model_cfg: dict[str, Any], vocab_size: int) -> Wav2Vec2ForCTC:
    cfg = build_wav2vec2_config(model_cfg, masking_cfg=None)
    cfg.vocab_size = int(vocab_size)
    # HF uses pad_token_id as the CTC blank for wav2vec2.
    cfg.pad_token_id = 0
    cfg.ctc_loss_reduction = "mean"
    cfg.ctc_zero_infinity = True
    return Wav2Vec2ForCTC(cfg)
