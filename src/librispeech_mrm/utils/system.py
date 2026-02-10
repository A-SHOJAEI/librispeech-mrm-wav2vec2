from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any

import psutil
import torch


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_commit_or_none() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp_utc": now_utc_iso(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                }
            )
        info["gpus"] = gpus
    info["git_commit"] = git_commit_or_none()
    return info


def pretty_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True)
