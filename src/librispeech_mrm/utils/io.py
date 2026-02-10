from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: str | os.PathLike) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise TypeError(f"Expected mapping in YAML config: {p}")
    return obj


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def write_json(path: str | os.PathLike, obj: Any, *, indent: int = 2) -> None:
    p = Path(path)
    data = json.dumps(obj, indent=indent, sort_keys=True, ensure_ascii=True).encode("utf-8")
    data += b"\n"
    _atomic_write_bytes(p, data)


def read_json(path: str | os.PathLike) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: str | os.PathLike, text: str) -> None:
    p = Path(path)
    _atomic_write_bytes(p, text.encode("utf-8"))


def write_jsonl(path: str | os.PathLike, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    buf = io.StringIO()
    for r in rows:
        buf.write(json.dumps(r, ensure_ascii=True))
        buf.write("\n")
    _atomic_write_bytes(p, buf.getvalue().encode("utf-8"))


def read_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    p = Path(path)
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"Expected object per line in {p}")
            rows.append(obj)
    return rows
