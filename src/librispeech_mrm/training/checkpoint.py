from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from librispeech_mrm.utils.io import ensure_dir


def save_torch(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
