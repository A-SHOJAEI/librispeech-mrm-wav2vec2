`README.md` has been fully rewritten (deleted and recreated) to be specific to this repository’s current code and the exact outputs in `artifacts/results.json` and `artifacts/report.md`.

Key specifics included:
- Problem statement tied to the implemented MRM knobs (time masking + feature masking + curriculum).
- Dataset provenance for both `smoke` (synthetic generator) and `openslr` (OpenSLR download + manifest + corruption pipeline).
- Methodology mapped to the actual modules (`Conformer+CTC`, HF wav2vec2 pretrain, CTC fine-tune, corruption eval, bootstrap CI).
- Baselines/ablations as they exist in `configs/smoke.yaml` and `configs/full.yaml`.
- Exact WER table copied from `artifacts/report.md`, plus the `git_commit` and `timestamp_utc` from `artifacts/results.json`.
- Repro commands that match the Makefile/CLI.
- Concrete limitations and next research steps grounded in what’s missing/next in this codebase.

File: `README.md`