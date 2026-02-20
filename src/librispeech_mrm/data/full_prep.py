"""Full LibriSpeech data preparation via OpenSLR downloads. Placeholder for full implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OpenSLRConfig:
    urls: list[str] = field(default_factory=list)
    data_root: Path = Path("data")
    downloads_dir: Path = Path("data/downloads")
    raw_dir: Path = Path("data/raw")
    manifests_dir: Path = Path("data/manifests")
    checksums_cache: Path = Path("data/checksums.json")
    seed: int = 0


def prepare_openslr_datasets(cfg: OpenSLRConfig) -> None:
    """Download, verify, extract and create JSONL manifests for LibriSpeech splits.

    This is a stub. Use data.profile=smoke for smoke testing.
    """
    raise NotImplementedError(
        "Full OpenSLR / LibriSpeech preparation not implemented. "
        "Use data.profile=smoke for smoke testing."
    )
