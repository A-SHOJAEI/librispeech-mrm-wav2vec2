from __future__ import annotations

from pathlib import Path
from typing import Any

from librispeech_mrm.training.ssl_wav2vec2_finetune_ctc import finetune_ssl_wav2vec2_ctc
from librispeech_mrm.training.ssl_wav2vec2_pretrain import train_ssl_wav2vec2_pretrain
from librispeech_mrm.training.supervised_conformer import train_supervised_conformer_ctc


def run_training(
    cfg: dict[str, Any],
    *,
    artifacts_dir: Path,
    overwrite: bool,
    only_kinds: set[str] | None = None,
) -> None:
    seed = int(cfg.get("seed", 0))
    global_device = cfg.get("device")
    runs = cfg.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise ValueError("Config must include a non-empty 'runs' list")

    for r in runs:
        if not isinstance(r, dict):
            raise TypeError("Each item in runs must be a mapping")
        kind = str(r.get("kind", ""))
        device = r.get("device", global_device)
        device = str(device) if device is not None and str(device).strip() else None
        if only_kinds is not None and kind not in only_kinds:
            continue
        if kind == "supervised_conformer_ctc":
            train_supervised_conformer_ctc(
                r,
                artifacts_dir=artifacts_dir,
                global_seed=seed,
                overwrite=overwrite,
                device=device,
            )
            continue
        if kind == "ssl_wav2vec2_pretrain":
            train_ssl_wav2vec2_pretrain(
                r,
                artifacts_dir=artifacts_dir,
                global_seed=seed,
                overwrite=overwrite,
                device=device,
            )
            continue
        if kind == "ssl_wav2vec2_finetune_ctc":
            finetune_ssl_wav2vec2_ctc(
                r,
                artifacts_dir=artifacts_dir,
                global_seed=seed,
                overwrite=overwrite,
                device=device,
            )
            continue
        raise ValueError(f"Unknown run.kind: {kind} (run={r.get('name')})")
