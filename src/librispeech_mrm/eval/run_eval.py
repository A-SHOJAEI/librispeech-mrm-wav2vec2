from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC

from librispeech_mrm.eval.bootstrap import bootstrap_wer_ci
from librispeech_mrm.eval.corruption import maybe_apply_corruption
from librispeech_mrm.eval.wer import compute_wer
from librispeech_mrm.models.conformer_ctc import ConformerCTCConfig, ConformerCTCModel, greedy_ctc_decode
from librispeech_mrm.models.tokenizer import CharTokenizer, normalize_text
from librispeech_mrm.training.datasets import ManifestAudioDataset
from librispeech_mrm.utils.io import ensure_dir, write_json
from librispeech_mrm.utils.system import system_info


def _device(device: str | None = None) -> torch.device:
    if device is not None and str(device).strip():
        return torch.device(str(device))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _eval_collate(batch: list[dict[str, Any]], *, sample_rate: int) -> dict[str, Any]:
    ids = [b["id"] for b in batch]
    texts = [b.get("text", "") for b in batch]

    wavs = []
    for b in batch:
        w = b["wav"]
        w = maybe_apply_corruption(w, sample_rate=sample_rate, corruption=b.get("corruption"))
        wavs.append(w)

    lengths = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)
    max_len = int(lengths.max().item())

    x = torch.zeros((len(wavs), max_len), dtype=torch.float32)
    attn = torch.zeros((len(wavs), max_len), dtype=torch.long)
    for i, w in enumerate(wavs):
        L = w.shape[0]
        x[i, :L] = w
        attn[i, :L] = 1

    return {"ids": ids, "texts": texts, "input_values": x, "attention_mask": attn, "lengths": lengths}


def _evaluate_conformer_ctc(
    *,
    run_dir: Path,
    run_cfg: dict[str, Any],
    eval_manifest: Path,
    eval_name: str,
    seed: int,
    device: str | None,
) -> dict[str, Any]:
    tokenizer = CharTokenizer.load(run_cfg["tokenizer_path"])

    ckpt = torch.load(run_dir / "best.pt", map_location="cpu")
    cfg = ckpt["cfg"]
    mcfg = ConformerCTCConfig(**cfg)
    model = ConformerCTCModel(mcfg, vocab_size=len(tokenizer.vocab))
    model.load_state_dict(ckpt["model"], strict=True)

    dev = _device(device)
    model.to(dev)
    model.eval()

    ds = ManifestAudioDataset(eval_manifest, target_sr=mcfg.sample_rate)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=lambda b: _eval_collate(b, sample_rate=mcfg.sample_rate))

    refs: list[str] = []
    hyps: list[str] = []

    with torch.no_grad():
        for b in tqdm(dl, desc=f"eval:{run_cfg['name']}:{eval_name}"):
            x = b["input_values"].to(dev)
            L = b["lengths"].to(dev)
            lp, _ = model(x, L)
            hyp_ids = greedy_ctc_decode(lp.cpu(), blank_id=tokenizer.blank_id)
            for t, h in zip(b["texts"], hyp_ids):
                refs.append(normalize_text(t))
                hyps.append(normalize_text(tokenizer.decode(h)))

    wer_res = compute_wer(refs, hyps)
    ci = bootstrap_wer_ci(refs, hyps, n_samples=200, seed=seed)

    return {
        "set": eval_name,
        "manifest": str(eval_manifest),
        "wer": wer_res.wer,
        "wer_ci": {"mean": ci.mean, "p5": ci.p5, "p95": ci.p95},
        "n_utts": len(refs),
    }


def _greedy_ctc_decode_hf(logits: torch.Tensor, blank_id: int) -> list[list[int]]:
    # logits: [B, T, V]
    ids = torch.argmax(logits, dim=-1).tolist()
    hyps: list[list[int]] = []
    for seq in ids:
        out = []
        prev = None
        for t in seq:
            if t == blank_id:
                prev = t
                continue
            if prev is not None and t == prev:
                continue
            out.append(int(t))
            prev = t
        hyps.append(out)
    return hyps


def _evaluate_wav2vec2_ctc(
    *,
    run_dir: Path,
    run_cfg: dict[str, Any],
    eval_manifest: Path,
    eval_name: str,
    seed: int,
    device: str | None,
) -> dict[str, Any]:
    tokenizer = CharTokenizer.load(run_cfg["tokenizer_path"])

    hf_dir = run_dir / "hf_ctc"
    if not hf_dir.exists():
        raise FileNotFoundError(f"Missing hf_ctc for {run_cfg['name']}: {hf_dir}")

    model = Wav2Vec2ForCTC.from_pretrained(hf_dir)
    # Ensure blank id alignment.
    model.config.pad_token_id = tokenizer.blank_id

    dev = _device(device)
    model.to(dev)
    model.eval()

    ds = ManifestAudioDataset(eval_manifest, target_sr=16000)
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=lambda b: _eval_collate(b, sample_rate=16000))

    refs: list[str] = []
    hyps: list[str] = []

    with torch.no_grad():
        for b in tqdm(dl, desc=f"eval:{run_cfg['name']}:{eval_name}"):
            x = b["input_values"].to(dev)
            attn = b["attention_mask"].to(dev)
            out = model(input_values=x, attention_mask=attn)
            hyp_ids = _greedy_ctc_decode_hf(out.logits.cpu(), blank_id=tokenizer.blank_id)
            for t, h in zip(b["texts"], hyp_ids):
                refs.append(normalize_text(t))
                hyps.append(normalize_text(tokenizer.decode(h)))

    wer_res = compute_wer(refs, hyps)
    ci = bootstrap_wer_ci(refs, hyps, n_samples=200, seed=seed)

    return {
        "set": eval_name,
        "manifest": str(eval_manifest),
        "wer": wer_res.wer,
        "wer_ci": {"mean": ci.mean, "p5": ci.p5, "p95": ci.p95},
        "n_utts": len(refs),
    }


def run_evaluation(cfg: dict[str, Any], *, artifacts_dir: Path) -> None:
    seed = int(cfg.get("seed", 0))
    device = cfg.get("device")
    device = str(device) if device is not None and str(device).strip() else None

    eval_cfg = cfg.get("eval", {})
    out_results = Path(eval_cfg.get("output_results_json", artifacts_dir / "results.json"))

    eval_sets = eval_cfg.get("sets", [])
    if not isinstance(eval_sets, list) or not eval_sets:
        raise ValueError("Config must include eval.sets")

    runs_cfg = cfg.get("runs", [])

    results: dict[str, Any] = {
        "system": system_info(),
        "eval": {"sets": eval_sets},
        "runs": {},
    }

    for r in runs_cfg:
        kind = str(r.get("kind", ""))
        run_name = str(r.get("name", ""))
        run_dir = artifacts_dir / "runs" / run_name
        if not run_dir.exists():
            continue
        train_summary_path = run_dir / "train_summary.json"
        train_summary = None
        if train_summary_path.exists():
            try:
                train_summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
            except Exception:
                train_summary = None

        if kind == "supervised_conformer_ctc":
            run_res = []
            for s in eval_sets:
                eval_name = str(s["name"])
                eval_manifest = Path(s["manifest"])
                run_res.append(
                    _evaluate_conformer_ctc(
                        run_dir=run_dir,
                        run_cfg=r,
                        eval_manifest=eval_manifest,
                        eval_name=eval_name,
                        seed=seed,
                        device=device,
                    )
                )
            results["runs"][run_name] = {"kind": kind, "results": run_res, "train_summary": train_summary}
            continue

        if kind == "ssl_wav2vec2_finetune_ctc":
            run_res = []
            for s in eval_sets:
                eval_name = str(s["name"])
                eval_manifest = Path(s["manifest"])
                run_res.append(
                    _evaluate_wav2vec2_ctc(
                        run_dir=run_dir,
                        run_cfg=r,
                        eval_manifest=eval_manifest,
                        eval_name=eval_name,
                        seed=seed,
                        device=device,
                    )
                )
            results["runs"][run_name] = {
                "kind": kind,
                "results": run_res,
                "init_from_run": r.get("init_from_run"),
                "train_summary": train_summary,
            }
            continue

        # Pretrain-only runs are not ASR models.

    ensure_dir(out_results.parent)
    write_json(out_results, results)
