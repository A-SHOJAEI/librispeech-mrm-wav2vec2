from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2ForPreTraining

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


def _collate_ctc(batch: list[dict[str, Any]], tokenizer: CharTokenizer) -> dict[str, Any]:
    base = collate_wavs(batch)
    ys = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in base["texts"]]
    y_lens = torch.tensor([y.numel() for y in ys], dtype=torch.long)
    max_y = int(y_lens.max().item()) if ys else 0

    labels = torch.full((len(ys), max_y), fill_value=-100, dtype=torch.long)
    for i, y in enumerate(ys):
        if y.numel() == 0:
            continue
        labels[i, : y.numel()] = y

    base["labels"] = labels
    base["label_lengths"] = y_lens
    return base


def finetune_ssl_wav2vec2_ctc(
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

    init_from_run = str(run_cfg["init_from_run"])
    init_dir = artifacts_dir / "runs" / init_from_run / "hf_pretrain"
    if not init_dir.exists():
        raise FileNotFoundError(f"Missing pretrain checkpoint for init_from_run={init_from_run}: {init_dir}")

    tokenizer = CharTokenizer.load(run_cfg["tokenizer_path"])

    # Load pretraining model and create a CTC model initialized from it.
    pre = Wav2Vec2ForPreTraining.from_pretrained(init_dir)
    base_cfg = Wav2Vec2Config.from_pretrained(init_dir)
    base_cfg.vocab_size = len(tokenizer.vocab)
    base_cfg.pad_token_id = tokenizer.blank_id
    base_cfg.ctc_loss_reduction = "mean"
    base_cfg.ctc_zero_infinity = True

    model = Wav2Vec2ForCTC(base_cfg)
    model.wav2vec2.load_state_dict(pre.wav2vec2.state_dict(), strict=True)

    dev = _resolve_device(device)
    model.to(dev)

    train_manifest = run_cfg["train_manifest"]
    dev_manifest = run_cfg["dev_manifest"]

    ds_train = ManifestAudioDataset(train_manifest, target_sr=16000)
    ds_dev = ManifestAudioDataset(dev_manifest, target_sr=16000)

    batch_size = int(run_cfg.get("batch_size", 2))
    max_steps = int(run_cfg.get("max_steps", 1000))
    lr = float(run_cfg.get("lr", 2e-4))
    grad_clip = float(run_cfg.get("grad_clip_norm", 1.0))

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: _collate_ctc(b, tokenizer),
    )
    dl_dev = DataLoader(
        ds_dev,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: _collate_ctc(b, tokenizer),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

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
                sample_rate=16000,
                seed=global_seed,
            )

    best_dev_loss = float("inf")
    t0 = time.time()
    step = 0

    it = iter(dl_train)
    pbar = tqdm(total=max_steps, desc=f"finetune:{run_name}")

    while step < max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl_train)
            batch = next(it)

        model.train()
        opt.zero_grad(set_to_none=True)

        x = batch["input_values"].to(dev)
        attn = batch["attention_mask"].to(dev)
        labels = batch["labels"].to(dev)

        if augmenter is not None:
            x = augmenter.apply(x, batch["lengths"].to(dev), step=step)

        out = model(input_values=x, attention_mask=attn, labels=labels)
        loss = out.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        step += 1
        pbar.update(1)
        pbar.set_postfix({"loss": float(loss.item())})

        if step % 10 == 0 or step == max_steps:
            model.eval()
            losses = []
            with torch.no_grad():
                for b in dl_dev:
                    x2 = b["input_values"].to(dev)
                    a2 = b["attention_mask"].to(dev)
                    y2 = b["labels"].to(dev)
                    o2 = model(input_values=x2, attention_mask=a2, labels=y2)
                    losses.append(float(o2.loss.item()))
            dev_loss = float(sum(losses) / max(1, len(losses)))
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                hf_dir = run_dir / "hf_ctc"
                ensure_dir(hf_dir)
                model.save_pretrained(hf_dir)
                save_torch(run_dir / "best.pt", {"step": step, "best_dev_loss": best_dev_loss})

    pbar.close()
    elapsed_s = time.time() - t0

    summary = {
        "run_name": run_name,
        "kind": "ssl_wav2vec2_finetune_ctc",
        "init_from_run": init_from_run,
        "steps": step,
        "best_dev_loss": best_dev_loss,
        "elapsed_s": round(elapsed_s, 3),
        "updates_per_s": round(step / max(1e-9, elapsed_s), 3),
        "seed": global_seed,
    }
    write_json(run_dir / "train_summary.json", summary)
    return run_dir
