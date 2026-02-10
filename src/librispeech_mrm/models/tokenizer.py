from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def normalize_text(s: str) -> str:
    # LibriSpeech transcripts are typically uppercase with punctuation stripped; keep it simple.
    s = s.strip().lower()
    s = re.sub(r"[^a-z\s']+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@dataclass(frozen=True)
class CharTokenizer:
    # CTC requires a dedicated blank token.
    blank: str
    pad: str
    vocab: list[str]

    @property
    def blank_id(self) -> int:
        return self.vocab.index(self.blank)

    @property
    def pad_id(self) -> int:
        return self.vocab.index(self.pad)

    def encode(self, text: str) -> list[int]:
        text = normalize_text(text)
        ids = []
        for ch in text:
            if ch == " ":
                ch = "|"  # word separator
            if ch not in self.vocab:
                # Unknown char: drop. For LibriSpeech this shouldn't happen after normalization.
                continue
            ids.append(self.vocab.index(ch))
        return ids

    def decode(self, ids: list[int]) -> str:
        chars = []
        for i in ids:
            if i < 0 or i >= len(self.vocab):
                continue
            ch = self.vocab[i]
            if ch in (self.blank, self.pad):
                continue
            if ch == "|":
                chars.append(" ")
            else:
                chars.append(ch)
        return "".join(chars).strip()

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        obj = {"blank": self.blank, "pad": self.pad, "vocab": self.vocab}
        p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "CharTokenizer":
        p = Path(path)
        obj = json.loads(p.read_text(encoding="utf-8"))
        return CharTokenizer(blank=obj["blank"], pad=obj["pad"], vocab=list(obj["vocab"]))


def build_char_tokenizer(texts: Iterable[str]) -> CharTokenizer:
    chars = set()
    for t in texts:
        t = normalize_text(t)
        for ch in t:
            if ch == " ":
                chars.add("|")
            else:
                chars.add(ch)
    # Stable ordering for reproducibility.
    core = sorted(chars)
    vocab = ["<blank>", "<pad>"] + core
    return CharTokenizer(blank="<blank>", pad="<pad>", vocab=vocab)
