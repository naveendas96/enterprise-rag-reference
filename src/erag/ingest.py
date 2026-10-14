"""Load markdown docs with front-matter ACLs and split into heading-aware chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    section: str
    text: str
    roles: list[str] = field(default_factory=list)


def load_docs(folder: Path) -> list[tuple[str, dict, str]]:
    out = []
    for p in sorted(folder.glob("*.md")):
        raw = p.read_text()
        m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
        meta, body = (yaml.safe_load(m.group(1)), m.group(2)) if m else ({}, raw)
        out.append((p.stem, meta, body))
    return out


def chunk_doc(doc_id: str, meta: dict, body: str, max_words: int = 80, overlap: int = 15) -> list[Chunk]:
    """Split on headings first (keeps semantic units), then window long sections."""
    chunks: list[Chunk] = []
    section = meta.get("title", doc_id)
    sections: list[tuple[str, str]] = []
    buf: list[str] = []
    for line in body.splitlines():
        if line.startswith("#"):
            if buf:
                sections.append((section, " ".join(buf)))
            section, buf = line.lstrip("# ").strip(), []
        elif line.strip():
            buf.append(line.strip())
    if buf:
        sections.append((section, " ".join(buf)))
    for sec, text in sections:
        words = text.split()
        step = max_words - overlap
        for i in range(0, max(len(words), 1), step):
            piece = " ".join(words[i:i + max_words])
            chunks.append(Chunk(f"{doc_id}#{len(chunks)}", doc_id, meta.get("title", doc_id),
                                sec, piece, list(meta.get("roles", ["admin"]))))
            if i + max_words >= len(words):
                break
    return chunks


def load_chunks(folder: Path) -> list[Chunk]:
    return [c for d, m, b in load_docs(folder) for c in chunk_doc(d, m, b)]
