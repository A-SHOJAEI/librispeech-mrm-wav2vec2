from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaskingConfig:
    curriculum: bool

    time_mask_prob: float
    time_mask_length: int

    feature_mask_prob: float
    feature_mask_length: int

    # Curriculum endpoints (optional). If not provided, endpoints equal the base values.
    time_mask_prob_final: float | None = None
    time_mask_length_final: int | None = None
    feature_mask_prob_final: float | None = None
    feature_mask_length_final: int | None = None

    curriculum_steps: int = 0


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_int(a: int, b: int, t: float) -> int:
    return int(round(_lerp(float(a), float(b), t)))


class MaskCurriculum:
    def __init__(self, cfg: MaskingConfig):
        self.cfg = cfg

        self.time_prob0 = float(cfg.time_mask_prob)
        self.time_len0 = int(cfg.time_mask_length)
        self.feat_prob0 = float(cfg.feature_mask_prob)
        self.feat_len0 = int(cfg.feature_mask_length)

        self.time_prob1 = float(cfg.time_mask_prob_final if cfg.time_mask_prob_final is not None else cfg.time_mask_prob)
        self.time_len1 = int(cfg.time_mask_length_final if cfg.time_mask_length_final is not None else cfg.time_mask_length)
        self.feat_prob1 = float(
            cfg.feature_mask_prob_final if cfg.feature_mask_prob_final is not None else cfg.feature_mask_prob
        )
        self.feat_len1 = int(
            cfg.feature_mask_length_final if cfg.feature_mask_length_final is not None else cfg.feature_mask_length
        )

        self.steps = int(cfg.curriculum_steps)

    def params(self, step: int) -> dict[str, float | int]:
        if not self.cfg.curriculum or self.steps <= 0:
            return {
                "time_mask_prob": self.time_prob0,
                "time_mask_length": self.time_len0,
                "feature_mask_prob": self.feat_prob0,
                "feature_mask_length": self.feat_len0,
            }

        t = min(1.0, max(0.0, float(step) / float(self.steps)))
        return {
            "time_mask_prob": float(_lerp(self.time_prob0, self.time_prob1, t)),
            "time_mask_length": int(_lerp_int(self.time_len0, self.time_len1, t)),
            "feature_mask_prob": float(_lerp(self.feat_prob0, self.feat_prob1, t)),
            "feature_mask_length": int(_lerp_int(self.feat_len0, self.feat_len1, t)),
        }
