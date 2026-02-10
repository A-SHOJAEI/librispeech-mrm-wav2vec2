from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from librispeech_mrm.models.conformer_ctc import (
    ConformerCTCConfig,
    ConformerCTCModel,
    greedy_ctc_decode,
)
from librispeech_mrm.models.tokenizer import CharTokenizer
from librispeech_mrm.training.augmentation import NoiseRIRAugment, NoiseRIRAugmentConfig
from librispeech_mrm.training.checkpoint import save_torch
from librispeech_mrm.training.datasets import ManifestAudioDataset, collate_wavs
from librispeech_mrm.utils.io import ensure_dir, write_json


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_device(device: str | None) -> torch.device:
    if device is None:
        return _device()
    return torch.device(device)


def _list_wavs(root: str | Path) -> list[str]:
    p = Path(root)
    return [str(x) for x in sorted(p.rglob("*.wav")) if x.is_file()]


def train_supervised_conformer_ctc(
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
        # Keep it simple; caller opted in.
        import shutil

        shutil.rmtree(run_dir)

    ensure_dir(run_dir)

    train_manifest = run_cfg["train_manifest"]
    dev_manifest = run_cfg["dev_manifest"]
    tokenizer = CharTokenizer.load(run_cfg["tokenizer_path"])

    # Model.
    mc = run_cfg.get("model", {})
    cfg = ConformerCTCConfig(
        sample_rate=16000,
        n_mels=int(mc.get("n_mels", 80)),
        conformer_dim=int(mc.get("conformer_dim", 256)),
        num_layers=int(mc.get("num_layers", 12)),
        num_heads=int(mc.get("num_heads", 4)),
        ffn_dim=int(mc.get("ffn_dim", 1024)),
        conv_kernel_size=int(mc.get("conv_kernel_size", 31)),
        dropout=float(mc.get("dropout", 0.1)),
    )

    dev = _resolve_device(device)
    model = ConformerCTCModel(cfg, vocab_size=len(tokenizer.vocab)).to(dev)

    specaug_cfg = run_cfg.get("specaug", {})
    specaug_enabled = bool(specaug_cfg.get("enabled", False))
    freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=int(specaug_cfg.get("freq_mask_param", 8)))
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param=int(specaug_cfg.get("time_mask_param", 16)))

    # Data.
    ds_train = ManifestAudioDataset(train_manifest, target_sr=cfg.sample_rate)
    ds_dev = ManifestAudioDataset(dev_manifest, target_sr=cfg.sample_rate)

    batch_size = int(run_cfg.get("batch_size", 4))
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_wavs,
    )
    dl_dev = DataLoader(
        ds_dev,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_wavs,
    )

    max_steps = int(run_cfg.get("max_steps", 1000))
    lr = float(run_cfg.get("lr", 1e-3))
    grad_clip = float(run_cfg.get("grad_clip_norm", 1.0))

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ctc = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

    aug_cfg = run_cfg.get("augmentation", {}) if isinstance(run_cfg.get("augmentation", {}), dict) else {}
    augmenter = None
    if bool(aug_cfg.get("enabled", False)):
        noise_dir = aug_cfg.get("noise_dir")
        rir_dir = aug_cfg.get("rir_dir")
        if noise_dir and rir_dir:
            augmenter = NoiseRIRAugment(
                NoiseRIRAugmentConfig(
                    enabled=True,
                    probability=float(aug_cfg.get("probability", 0.5)),
                    snr_db_min=float(aug_cfg.get("snr_db_min", 5.0)),
                    snr_db_max=float(aug_cfg.get("snr_db_max", 20.0)),
                    noise_paths=_list_wavs(noise_dir),
                    rir_paths=_list_wavs(rir_dir),
                ),
                sample_rate=cfg.sample_rate,
                seed=global_seed,
            )

    best_dev_loss = float("inf")
    t0 = time.time()
    step = 0

    it = iter(dl_train)
    pbar = tqdm(total=max_steps, desc=f"train:{run_name}")

    while step < max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl_train)
            batch = next(it)

        model.train()
        opt.zero_grad(set_to_none=True)

        wav = batch["input_values"].to(dev)
        wav_lengths = batch["lengths"].to(dev)
        if augmenter is not None:
            wav = augmenter.apply(wav, wav_lengths, step=step)

        # SpecAugment is applied on log-mel features.
        feats = model.wav_to_features(wav)  # [B, T, n_mels]
        feat_lengths = model.wav_lengths_to_feat_lengths(wav_lengths)
        if specaug_enabled:
            f = feats.transpose(1, 2)  # [B, n_mels, T]
            f = freq_mask(f)
            f = time_mask(f)
            feats = f.transpose(1, 2).contiguous()

        x = model.in_proj(feats)
        x, out_lengths = model.encoder(x, feat_lengths)
        logits = model.out_proj(x)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Targets.
        ys = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in batch["texts"]]
        y_lens = torch.tensor([y.numel() for y in ys], dtype=torch.long, device=dev)
        if int(y_lens.max().item()) == 0:
            # Should not happen for real data; avoid crashing in smoke if a transcript normalizes to empty.
            loss = log_probs.mean() * 0.0
        else:
            y_cat = torch.cat([y.to(dev) for y in ys if y.numel() > 0], dim=0)
            loss = ctc(
                log_probs.transpose(0, 1),
                y_cat,
                out_lengths.to(torch.long),
                y_lens,
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        step += 1
        pbar.update(1)
        pbar.set_postfix({"loss": float(loss.item())})

        # Periodic dev eval.
        if step % 10 == 0 or step == max_steps:
            model.eval()
            dev_losses = []
            with torch.no_grad():
                for b in dl_dev:
                    w = b["input_values"].to(dev)
                    wl = b["lengths"].to(dev)
                    lp, out_l = model(w, wl)
                    ys2 = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in b["texts"]]
                    y2_lens = torch.tensor([y.numel() for y in ys2], dtype=torch.long, device=dev)
                    if int(y2_lens.max().item()) == 0:
                        continue
                    y2_cat = torch.cat([y.to(dev) for y in ys2 if y.numel() > 0], dim=0)
                    dl = ctc(lp.transpose(0, 1), y2_cat, out_l.to(torch.long), y2_lens)
                    dev_losses.append(float(dl.item()))

            dev_loss = float(sum(dev_losses) / max(1, len(dev_losses)))
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                save_torch(run_dir / "best.pt", {"model": model.state_dict(), "cfg": asdict(cfg), "step": step})

    pbar.close()

    elapsed_s = time.time() - t0
    summary = {
        "run_name": run_name,
        "kind": "supervised_conformer_ctc",
        "steps": step,
        "best_dev_loss": best_dev_loss,
        "elapsed_s": round(elapsed_s, 3),
        "seed": global_seed,
    }
    write_json(run_dir / "train_summary.json", summary)
    return run_dir
