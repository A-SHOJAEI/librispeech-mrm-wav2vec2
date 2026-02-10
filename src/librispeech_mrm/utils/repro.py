from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ReproConfig:
    seed: int
    deterministic: bool = True
    cudnn_benchmark: bool = False
    # Device intent used to decide whether we should enable CUDA-specific
    # determinism workarounds. If None, we infer from torch.cuda.is_available().
    device: str | None = None


def seed_everything(cfg: ReproConfig) -> None:
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    using_cuda = (str(cfg.device).startswith("cuda")) if cfg.device is not None else torch.cuda.is_available()

    # If we enable deterministic algorithms on CUDA, PyTorch requires a CuBLAS
    # workspace config for deterministic GEMMs (torch.matmul, etc.).
    if cfg.deterministic and using_cuda and torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.manual_seed(cfg.seed)
    if using_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)

    if cfg.deterministic:
        # Can reduce performance; some ops may error if no deterministic implementation exists.
        torch.backends.cudnn.deterministic = True
        # On CUDA, some ops may not have deterministic implementations across
        # all versions/builds; warn instead of crashing.
        torch.use_deterministic_algorithms(True, warn_only=using_cuda)
    else:
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)


def dataloader_worker_init_fn(base_seed: int):
    def _fn(worker_id: int):
        s = base_seed + worker_id
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    return _fn
