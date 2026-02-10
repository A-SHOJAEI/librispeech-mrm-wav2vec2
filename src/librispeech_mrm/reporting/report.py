from __future__ import annotations

from pathlib import Path
from typing import Any

from librispeech_mrm.utils.io import read_json, write_text


def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def generate_report_md(*, results_json_path: Path, out_md_path: Path) -> None:
    results = read_json(results_json_path)
    runs = results.get("runs", {})

    lines: list[str] = []
    lines.append("# Evaluation Report")
    lines.append("")

    sys = results.get("system", {})
    lines.append("## System")
    lines.append("")
    lines.append("```json")
    import json

    lines.append(json.dumps(sys, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| Run | Kind | Set | WER | WER CI (p5..p95) | N |")
    lines.append("|---|---|---|---:|---:|---:|")

    for run_name, rr in runs.items():
        kind = rr.get("kind", "")
        for s in rr.get("results", []):
            wer = s.get("wer")
            ci = s.get("wer_ci", {})
            lines.append(
                "| {run} | {kind} | {set} | {wer} | {p5}..{p95} | {n} |".format(
                    run=run_name,
                    kind=kind,
                    set=s.get("set"),
                    wer=_fmt(wer),
                    p5=_fmt(ci.get("p5")),
                    p95=_fmt(ci.get("p95")),
                    n=int(s.get("n_utts", 0)),
                )
            )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This report is generated from `artifacts/results.json`. For the full research plan, run the larger configs on "
        "LibriSpeech and evaluate on clean + corrupted test sets (MUSAN SNR 5/10/20 dB; RIRS convolution)."
    )
    lines.append("")

    write_text(out_md_path, "\n".join(lines) + "\n")
