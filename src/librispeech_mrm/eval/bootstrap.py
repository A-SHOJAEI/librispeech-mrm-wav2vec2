from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from librispeech_mrm.eval.wer import compute_wer


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    p5: float
    p95: float


def bootstrap_wer_ci(
    refs: list[str],
    hyps: list[str],
    *,
    n_samples: int = 200,
    seed: int = 0,
) -> BootstrapCI:
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps must have the same length")
    if len(refs) == 0:
        return BootstrapCI(mean=float("nan"), p5=float("nan"), p95=float("nan"))

    rng = np.random.default_rng(seed)
    N = len(refs)
    wers = []
    for _ in range(int(n_samples)):
        idx = rng.integers(0, N, size=N)
        r = [refs[i] for i in idx]
        h = [hyps[i] for i in idx]
        wers.append(compute_wer(r, h).wer)

    wers = np.asarray(wers, dtype=np.float64)
    return BootstrapCI(mean=float(wers.mean()), p5=float(np.percentile(wers, 5)), p95=float(np.percentile(wers, 95)))
