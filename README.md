# librispeech-mrm-wav2vec2

This repo tests a specific ASR hypothesis: **wav2vec2-style SSL pretraining becomes more robust and more sample-efficient if masking is “multi-resolution” (time-span + feature-channel) and masking strength is scheduled with a curriculum**, then fine-tuned with CTC and evaluated on clean and corrupted audio.

The implementation is intentionally minimal (single-node loops, greedy CTC decoding) but end-to-end: dataset manifests, supervised baseline, SSL pretrain, SSL fine-tune, corruption evaluation, and a generated report.

## Problem Statement

Pretrain an encoder on unlabeled speech with masked prediction (wav2vec2 pretraining objective), then fine-tune for ASR with CTC. Compare:

- **Supervised baseline**: log-mel frontend + Conformer encoder + CTC (+ SpecAugment).
- **SSL baseline**: wav2vec2 pretraining with **time-span masking only**, then CTC fine-tuning.
- **MRM (proposed)**: wav2vec2 pretraining with **time-span masking + feature-channel masking**, optionally with a **masking curriculum** (linear schedule over steps).
- **Ablations**: remove feature masking; remove curriculum.
- **Robustness ablation**: fine-tune with/without noise+RIR augmentation; evaluate on corrupted test manifests.

## Dataset Provenance

Two data profiles exist and are selected by `data.profile` in the YAML config:

1. `smoke` (default; the committed `artifacts/` were produced from this):
   - Synthetic “speechy” waveforms + fixed phrase transcripts generated locally in `src/librispeech_mrm/data/smoke.py`.
   - Corrupted test manifests are created by attaching a per-utterance `corruption` spec (`noise+rir`, SNR in dB) and applied at eval time.
   - Purpose: validate wiring, training/eval/reporting paths, and determinism.

2. `openslr` (full pipeline; configured in `configs/full.yaml`):
   - Downloads OpenSLR resources sequentially and verifies via MD5 parsed from OpenSLR resource pages (`src/librispeech_mrm/data/openslr.py`).
   - Builds LibriSpeech manifests from extracted `LibriSpeech/<subset>` folders (`src/librispeech_mrm/data/librispeech.py`).
   - Builds deterministic labeled subsets (1h/10h/100h) from `train-clean-100` for sample-efficiency sweeps (`src/librispeech_mrm/data/full_prep.py`).
   - Creates corrupted evaluation manifests by pairing each test utterance with deterministic (seeded) MUSAN noise segments and RIRs (`src/librispeech_mrm/data/corrupted_manifests.py`).

## Methodology (What’s Implemented)

**Manifests**
- JSONL schema includes `id`, `audio_path`, `text`, `duration_s`, `sample_rate`, plus optional `corruption` (`src/librispeech_mrm/training/datasets.py`).
- Audio is loaded mono and resampled at read time (via `ManifestAudioDataset`).

**Supervised baseline**
- Model: `torchaudio.models.Conformer` on log-mel features (`src/librispeech_mrm/models/conformer_ctc.py`).
- Training: AdamW + `torch.nn.CTCLoss`; optional SpecAugment on log-mel (`src/librispeech_mrm/training/supervised_conformer.py`).

**SSL pretraining (wav2vec2-style)**
- Uses HuggingFace `Wav2Vec2ForPreTraining` (`src/librispeech_mrm/models/wav2vec2_factory.py`).
- Time masking indices + negative sampling indices are produced in `src/librispeech_mrm/training/wav2vec2_pretrain_collator.py`.
- **MRM knobs**:
  - Feature-channel masking is controlled via `model.config.mask_feature_prob` / `mask_feature_length`.
  - Curriculum (linear interpolation from initial to final mask params over `curriculum_steps`) is in `src/librispeech_mrm/training/masking.py`.
- Checkpoint output: `artifacts/runs/<run>/hf_pretrain` (`src/librispeech_mrm/training/ssl_wav2vec2_pretrain.py`).

**CTC fine-tuning**
- Builds `Wav2Vec2ForCTC` initialized from a pretraining run (encoder weights copied) (`src/librispeech_mrm/training/ssl_wav2vec2_finetune_ctc.py`).
- Optional on-the-fly noise+RIR augmentation for robustness (`src/librispeech_mrm/training/augmentation.py`).
- Best-dev checkpoint output: `artifacts/runs/<run>/hf_ctc`.

**Evaluation**
- Greedy CTC decoding for both Conformer and wav2vec2 CTC (`src/librispeech_mrm/eval/run_eval.py`).
- WER uses `jiwer.wer` on normalized text (`src/librispeech_mrm/eval/wer.py`, `src/librispeech_mrm/models/tokenizer.py`).
  - WER here is a **ratio**, not a percentage; it can exceed `1.0` when insertions dominate.
- Corruptions (`noise+rir`) are applied at eval time from the manifest `corruption` field (`src/librispeech_mrm/eval/corruption.py`, `src/librispeech_mrm/data/corrupt.py`).
- Confidence intervals: bootstrap over utterances (200 samples) (`src/librispeech_mrm/eval/bootstrap.py`).

## Baselines / Ablations (Configured)

- `configs/smoke.yaml` runs:
  - `supervised_conformer_ctc`
  - `ssl_wav2vec2_baseline_pretrain` -> `finetune_ssl_wav2vec2_baseline_ctc`
  - `ssl_mrm_ablation_no_curriculum_pretrain` -> `finetune_ssl_mrm_ablation_no_curriculum_ctc`

- `configs/full.yaml` additionally defines (not included in the committed `artifacts/` results):
  - Proposed `ssl_mrm_pretrain` (curriculum + feature masking)
  - Ablation `ssl_mrm_ablation_no_feature_mask_pretrain`
  - Robustness fine-tune toggle via `augmentation.enabled` in fine-tune runs

## Exact Results (From `artifacts/report.md`)

The repository includes one completed run group (the `smoke` profile). All numbers below are taken from `artifacts/report.md`, generated from `artifacts/results.json`.

System snapshot (from `artifacts/results.json`):
- `git_commit`: `75f29719680d59290a58feea16d1e0560e7019fe`
- `timestamp_utc`: `2026-02-20T06:29:26+00:00`

WER table (verbatim values from `artifacts/report.md`):

| Run | Set | WER | WER CI (p5..p95) | N |
|---|---|---:|---:|---:|
| supervised_conformer_ctc | clean_test | 1.0000 | 1.0000..1.0000 | 8 |
| supervised_conformer_ctc | noisy_snr5 | 1.0000 | 1.0000..1.0000 | 8 |
| supervised_conformer_ctc | noisy_snr10 | 1.0000 | 1.0000..1.0000 | 8 |
| finetune_ssl_wav2vec2_baseline_ctc | clean_test | 1.0000 | 1.0000..1.0000 | 8 |
| finetune_ssl_wav2vec2_baseline_ctc | noisy_snr5 | 1.0000 | 1.0000..1.0000 | 8 |
| finetune_ssl_wav2vec2_baseline_ctc | noisy_snr10 | 1.0000 | 1.0000..1.0000 | 8 |
| finetune_ssl_mrm_ablation_no_curriculum_ctc | clean_test | 1.0000 | 1.0000..1.0000 | 8 |
| finetune_ssl_mrm_ablation_no_curriculum_ctc | noisy_snr5 | 1.0000 | 1.0000..1.0000 | 8 |
| finetune_ssl_mrm_ablation_no_curriculum_ctc | noisy_snr10 | 1.0000 | 1.0000..1.0000 | 8 |

Training summaries:

| Run | Kind | Steps | Best Dev Loss | Elapsed (s) |
|---|---|---:|---:|---:|
| supervised_conformer_ctc | supervised | 30 | 3.5615 | 0.895 |
| ssl_wav2vec2_baseline_pretrain | pretrain | 20 | - | 3.825 |
| ssl_mrm_ablation_no_curriculum_pretrain | pretrain | 20 | - | 3.670 |
| finetune_ssl_wav2vec2_baseline_ctc | finetune | 30 | 152.5704 | 2.973 |
| finetune_ssl_mrm_ablation_no_curriculum_ctc | finetune | 30 | 157.8058 | 2.966 |

Notes on these checked-in artifacts:
- These results are from the **synthetic smoke dataset**, not LibriSpeech.
- The observed WER values indicate degenerate decoding in the smoke setting; treat them as pipeline validation only.

## Reproduction

### Environment

Pinned deps are in `requirements.txt`. The Makefile bootstraps a venv and installs editable package deps.

```bash
make setup
```

For CUDA wheels, pass a PyTorch wheel index (example for CUDA 12.1):

```bash
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121 make setup
```

### Smoke (Matches The Included Artifacts)

```bash
make clean
make all CONFIG=configs/smoke.yaml
```

Outputs:
- `artifacts/runs/`
- `artifacts/results.json`
- `artifacts/report.md`

### OpenSLR / LibriSpeech (Full Scale)

```bash
.venv/bin/python -m librispeech_mrm.cli --config configs/full.yaml data
.venv/bin/python -m librispeech_mrm.cli --config configs/full.yaml pretrain
.venv/bin/python -m librispeech_mrm.cli --config configs/full.yaml finetune
.venv/bin/python -m librispeech_mrm.cli --config configs/full.yaml eval
.venv/bin/python -m librispeech_mrm.cli --config configs/full.yaml report
```

## Limitations

- No LibriSpeech experiment outputs are committed; only `smoke` artifacts are present.
- Decoding is greedy CTC only (no beam search / LM rescoring).
- Training is single-process and minimal (no resume, no AMP, no DDP); dataloaders use `num_workers=0`.
- Tokenization is character-level; tokenizer is built from the training manifest transcripts.
- Corruption model is limited to `noise+rir` (RIR convolution + SNR mixing).

## Next Research Steps

1. Run `configs/full.yaml` end-to-end and report WER for `ssl_mrm_pretrain` vs baselines/ablations on `test-clean`/`test-other` and corrupted sets.
2. Add beam search decoding (optionally LM rescoring) to avoid conflating model quality with greedy decoding failures.
3. Perform sample-efficiency sweeps using the generated `1h/10h/100h` manifests and report WER vs labeled hours.
4. Ablate the curriculum schedule (linear vs cosine; separate schedules for time vs feature masking; vary `curriculum_steps`).
5. Expand robustness evaluation (noise-only, reverb-only, bandwidth, clipping) and measure train-time augmentation vs test-time corruption mismatch.

