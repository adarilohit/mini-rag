from __future__ import annotations
from dataclasses import dataclass
from typing import List
import re

@dataclass
class Chunk:
    id: str
    text: str

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def clean_text(text: str) -> str:
    # light cleanup: normalize whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[Chunk]:
    """
    Simple, reliable chunker:
    - splits by paragraphs, then packs into approx chunk_size characters
    - overlap is applied at the end of each chunk to preserve continuity
    """
    text = clean_text(text)
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []

    current = ""
    for p in paras:
        if len(current) + len(p) + 2 <= chunk_size:
            current = f"{current}\n\n{p}".strip()
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)

    # apply overlap by characters (safe and simple)
    out: List[Chunk] = []
    for i, c in enumerate(chunks):
        if i == 0:
            merged = c
        else:
            prev = chunks[i - 1]
            overlap_text = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
            merged = (overlap_text + "\n" + c).strip()
        out.append(Chunk(id=f"chunk_{i}", text=merged))
    return out
