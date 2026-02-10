from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from librispeech_mrm.models.wav2vec2_factory import build_pretrain_model
from librispeech_mrm.training.datasets import ManifestAudioDataset
from librispeech_mrm.training.masking import MaskCurriculum, MaskingConfig
from librispeech_mrm.training.wav2vec2_pretrain_collator import Wav2Vec2PretrainCollator
from librispeech_mrm.utils.io import ensure_dir, write_json


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_device(device: str | None) -> torch.device:
    if device is None:
        return _device()
    return torch.device(device)


def train_ssl_wav2vec2_pretrain(
    run_cfg: dict[str, Any],
    *,
    artifacts_dir: Path,
    global_seed: int,
    overwrite: bool,
    device: str | None = None,
) -> Path:
    run_name = str(run_cfg["name"])
    run_dir = artifacts_dir / "runs" / run_name
    if run_dir.exists() and not overwrite:
        return run_dir
    if run_dir.exists() and overwrite:
        import shutil

        shutil.rmtree(run_dir)

    ensure_dir(run_dir)

    train_manifest = run_cfg["train_manifest"]
    model_cfg = run_cfg.get("model", {})
    masking_cfg = run_cfg.get("masking", {})

    # Curriculum (MRM schedule) for time masking + feature-channel masking.
    mcfg = MaskingConfig(
        curriculum=bool(masking_cfg.get("curriculum", False)),
        time_mask_prob=float(masking_cfg.get("time_mask_prob", 0.65)),
        time_mask_length=int(masking_cfg.get("time_mask_length", 10)),
        feature_mask_prob=float(masking_cfg.get("feature_mask_prob", 0.0)),
        feature_mask_length=int(masking_cfg.get("feature_mask_length", 10)),
        time_mask_prob_final=masking_cfg.get("time_mask_prob_final"),
        time_mask_length_final=masking_cfg.get("time_mask_length_final"),
        feature_mask_prob_final=masking_cfg.get("feature_mask_prob_final"),
        feature_mask_length_final=masking_cfg.get("feature_mask_length_final"),
        curriculum_steps=int(masking_cfg.get("curriculum_steps", 0)),
    )
    sched = MaskCurriculum(mcfg)

    dev = _resolve_device(device)
    model = build_pretrain_model(model_cfg, masking_cfg).to(dev)

    # Feature masking is controlled inside the model by config.
    model.config.mask_feature_prob = float(mcfg.feature_mask_prob)
    model.config.mask_feature_length = int(mcfg.feature_mask_length)

    ds = ManifestAudioDataset(train_manifest, target_sr=16000)

    batch_size = int(run_cfg.get("batch_size", 2))
    max_steps = int(run_cfg.get("max_steps", 1000))
    lr = float(run_cfg.get("lr", 1e-4))
    grad_clip = float(run_cfg.get("grad_clip_norm", 1.0))

    collator = Wav2Vec2PretrainCollator(
        model=model,
        seed=global_seed,
        time_mask_prob=float(mcfg.time_mask_prob),
        time_mask_length=int(mcfg.time_mask_length),
        num_negatives=int(getattr(model.config, "num_negatives", 100)),
    )

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collator)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    t0 = time.time()
    model.train()
    pbar = tqdm(total=max_steps, desc=f"pretrain:{run_name}")

    it = iter(dl)
    step = 0

    peak_vram_bytes = 0

    while step < max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        params = sched.params(step)
        collator.set_step(step)
        collator.set_mask_params(time_mask_prob=float(params["time_mask_prob"]), time_mask_length=int(params["time_mask_length"]))
        model.config.mask_feature_prob = float(params["feature_mask_prob"])
        model.config.mask_feature_length = int(params["feature_mask_length"])

        x = batch["input_values"].to(dev)
        attn = batch["attention_mask"].to(dev)
        mask_time = batch["mask_time_indices"].to(dev)
        neg_idx = batch["sampled_negative_indices"].to(dev)

        opt.zero_grad(set_to_none=True)
        out = model(
            input_values=x,
            attention_mask=attn,
            mask_time_indices=mask_time,
            sampled_negative_indices=neg_idx,
        )
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        step += 1
        pbar.update(1)
        pbar.set_postfix({"loss": float(loss.item())})

        if torch.cuda.is_available():
            peak_vram_bytes = max(peak_vram_bytes, int(torch.cuda.max_memory_allocated()))

    pbar.close()
    elapsed_s = time.time() - t0

    # Save final checkpoint in HF format.
    hf_dir = run_dir / "hf_pretrain"
    ensure_dir(hf_dir)
    model.save_pretrained(hf_dir)

    summary = {
        "run_name": run_name,
        "kind": "ssl_wav2vec2_pretrain",
        "steps": step,
        "elapsed_s": round(elapsed_s, 3),
        "updates_per_s": round(step / max(1e-9, elapsed_s), 3),
        "peak_vram_gb": round(peak_vram_bytes / (1024**3), 3) if peak_vram_bytes else None,
        "seed": global_seed,
        "masking": {
            "time_mask_prob": mcfg.time_mask_prob,
            "time_mask_length": mcfg.time_mask_length,
            "feature_mask_prob": mcfg.feature_mask_prob,
            "feature_mask_length": mcfg.feature_mask_length,
            "curriculum": mcfg.curriculum,
            "curriculum_steps": mcfg.curriculum_steps,
        },
    }
    write_json(run_dir / "train_summary.json", summary)
    return run_dir
