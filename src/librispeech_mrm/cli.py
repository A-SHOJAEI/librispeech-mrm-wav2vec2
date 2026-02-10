from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from librispeech_mrm.data.smoke import SmokeConfig, generate_smoke_dataset
from librispeech_mrm.data.full_prep import OpenSLRConfig, prepare_openslr_datasets
from librispeech_mrm.models.tokenizer import build_char_tokenizer
from librispeech_mrm.reporting.report import generate_report_md
from librispeech_mrm.training.run_train import run_training
from librispeech_mrm.eval.run_eval import run_evaluation
from librispeech_mrm.utils.io import ensure_dir, read_jsonl, read_yaml
from librispeech_mrm.utils.repro import ReproConfig, seed_everything
from librispeech_mrm.utils.system import system_info


def _cfg_get(cfg: dict[str, Any], key: str, default: Any) -> Any:
    v = cfg
    for part in key.split("."):
        if not isinstance(v, dict) or part not in v:
            return default
        v = v[part]
    return v


def cmd_data(cfg: dict[str, Any]) -> None:
    seed = int(cfg.get("seed", 0))
    repro = cfg.get("repro", {})
    device = cfg.get("device")
    seed_everything(
        ReproConfig(
            seed=seed,
            deterministic=bool(repro.get("deterministic", True)),
            cudnn_benchmark=bool(repro.get("cudnn_benchmark", False)),
            device=str(device) if device is not None else None,
        )
    )

    data_profile = _cfg_get(cfg, "data.profile", "smoke")
    data_root = Path(_cfg_get(cfg, "paths.data_root", "data"))
    manifests_dir = Path(_cfg_get(cfg, "paths.manifests_dir", "data/manifests"))
    artifacts_dir = Path(_cfg_get(cfg, "paths.artifacts_dir", "artifacts"))

    ensure_dir(manifests_dir)
    ensure_dir(artifacts_dir / "tokenizers")

    if data_profile == "smoke":
        smoke = cfg.get("data", {}).get("smoke", {})
        corr = cfg.get("data", {}).get("corruption", {})
        scfg = SmokeConfig(
            data_root=data_root,
            manifests_dir=manifests_dir,
            sample_rate=int(cfg.get("data", {}).get("sample_rate", 16000)),
            n_train=int(smoke.get("n_train", 24)),
            n_dev=int(smoke.get("n_dev", 8)),
            n_test=int(smoke.get("n_test", 8)),
            min_utt_seconds=float(smoke.get("min_utt_seconds", 1.0)),
            max_utt_seconds=float(smoke.get("max_utt_seconds", 2.0)),
            corruption_enabled=bool(corr.get("enabled", False)),
            snrs_db=[int(x) for x in corr.get("snrs_db", [])],
            seed=seed,
        )
        generate_smoke_dataset(scfg)

        # Build tokenizer from training manifest transcripts.
        train_manifest = manifests_dir / "smoke_train.jsonl"
        rows = read_jsonl(train_manifest)
        tok = build_char_tokenizer([r["text"] for r in rows])
        tok_path = artifacts_dir / "tokenizers" / "smoke_char.json"
        tok.save(tok_path)
        return

    if data_profile == "openslr":
        openslr = cfg.get("data", {}).get("openslr", {})
        urls = openslr.get("urls", [])
        if not isinstance(urls, list) or not urls:
            raise ValueError("data.openslr.urls must be a non-empty list")

        downloads_dir = Path(openslr.get("downloads_dir", data_root / "downloads"))
        raw_dir = Path(openslr.get("raw_dir", data_root / "raw"))
        checksums_cache = Path(openslr.get("checksums_cache", data_root / "checksums.json"))

        ocfg = OpenSLRConfig(
            urls=[str(u) for u in urls],
            data_root=data_root,
            downloads_dir=downloads_dir,
            raw_dir=raw_dir,
            manifests_dir=manifests_dir,
            checksums_cache=checksums_cache,
            seed=seed,
        )
        prepare_openslr_datasets(ocfg)

        tok_from = openslr.get("tokenizer_from_manifest")
        tok_path = openslr.get("tokenizer_path")
        if tok_from and tok_path:
            rows = read_jsonl(Path(tok_from))
            tok = build_char_tokenizer([r["text"] for r in rows if "text" in r])
            tok.save(Path(tok_path))
        return

    raise ValueError(f"Unknown data.profile: {data_profile}")


def cmd_train(cfg: dict[str, Any], *, overwrite: bool, only_kinds: set[str] | None = None) -> None:
    seed = int(cfg.get("seed", 0))
    repro = cfg.get("repro", {})
    device = cfg.get("device")
    seed_everything(
        ReproConfig(
            seed=seed,
            deterministic=bool(repro.get("deterministic", True)),
            cudnn_benchmark=bool(repro.get("cudnn_benchmark", False)),
            device=str(device) if device is not None else None,
        )
    )

    artifacts_dir = Path(_cfg_get(cfg, "paths.artifacts_dir", "artifacts"))
    ensure_dir(artifacts_dir / "runs")

    # Optional: snapshot system info once for the run group.
    sys = system_info()
    import json

    (artifacts_dir / "system_info.json").write_text(json.dumps(sys, indent=2), encoding="utf-8")

    run_training(cfg, artifacts_dir=artifacts_dir, overwrite=overwrite, only_kinds=only_kinds)


def cmd_eval(cfg: dict[str, Any]) -> None:
    seed = int(cfg.get("seed", 0))
    repro = cfg.get("repro", {})
    device = cfg.get("device")
    seed_everything(
        ReproConfig(
            seed=seed,
            deterministic=bool(repro.get("deterministic", True)),
            cudnn_benchmark=bool(repro.get("cudnn_benchmark", False)),
            device=str(device) if device is not None else None,
        )
    )

    artifacts_dir = Path(_cfg_get(cfg, "paths.artifacts_dir", "artifacts"))
    run_evaluation(cfg, artifacts_dir=artifacts_dir)


def cmd_report(cfg: dict[str, Any]) -> None:
    artifacts_dir = Path(_cfg_get(cfg, "paths.artifacts_dir", "artifacts"))
    eval_cfg = cfg.get("eval", {})
    results_path = Path(eval_cfg.get("output_results_json", artifacts_dir / "results.json"))
    report_path = Path(eval_cfg.get("output_report_md", artifacts_dir / "report.md"))

    generate_report_md(results_json_path=results_path, out_md_path=report_path)


def main() -> None:
    ap = argparse.ArgumentParser(prog="librispeech_mrm")
    ap.add_argument("--config", required=True, help="Path to YAML config")

    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("data", help="Prepare datasets and manifests")

    ap_train = sub.add_parser("train", help="Train models defined in config")
    ap_train.add_argument("--overwrite", action="store_true", help="Overwrite existing run directories")

    ap_pre = sub.add_parser("pretrain", help="Run SSL pretraining runs (wav2vec2-style)")
    ap_pre.add_argument("--overwrite", action="store_true", help="Overwrite existing run directories")

    ap_ft = sub.add_parser("finetune", help="Run fine-tuning runs (CTC)")
    ap_ft.add_argument("--overwrite", action="store_true", help="Overwrite existing run directories")

    sub.add_parser("eval", help="Evaluate trained models and write artifacts/results.json")
    sub.add_parser("report", help="Generate artifacts/report.md from artifacts/results.json")

    args = ap.parse_args()

    cfg = read_yaml(args.config)

    if args.cmd == "data":
        cmd_data(cfg)
        return
    if args.cmd == "train":
        cmd_train(cfg, overwrite=bool(args.overwrite), only_kinds=None)
        return
    if args.cmd == "pretrain":
        cmd_train(cfg, overwrite=bool(args.overwrite), only_kinds={"ssl_wav2vec2_pretrain"})
        return
    if args.cmd == "finetune":
        cmd_train(cfg, overwrite=bool(args.overwrite), only_kinds={"ssl_wav2vec2_finetune_ctc"})
        return
    if args.cmd == "eval":
        cmd_eval(cfg)
        return
    if args.cmd == "report":
        cmd_report(cfg)
        return

    raise RuntimeError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
