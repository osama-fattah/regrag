"""Load documents (Markdown, text, PDF) and split them into section-aware chunks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

SUPPORTED = {".md", ".txt", ".pdf"}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    source: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def retrieval_text(self) -> str:
        # Title and section headings carry a lot of signal in SOPs, so index them with the body.
        return f"{self.title}\n{self.section}\n{self.text}"


def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8")


def _sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (title, [(section_heading, body), ...]) from Markdown-ish text."""
    title = ""
    current, buf, out = "Overview", [], []
    for line in text.splitlines():
        h1 = re.match(r"^#\s+(.*)", line)
        h2 = re.match(r"^#{2,6}\s+(.*)", line)
        if h1 and not title:
            title = h1.group(1).strip()
        elif h2:
            if "".join(buf).strip():
                out.append((current, "\n".join(buf).strip()))
            current, buf = h2.group(1).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        out.append((current, "\n".join(buf).strip()))
    return title, out


def _windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words]
    step = max(1, size - overlap)
    return [words[i : i + size] for i in range(0, max(1, len(words) - overlap), step)]


def chunk_document(path: Path, max_words: int = 160, overlap: int = 30) -> list[Chunk]:
    text = _read(path)
    doc_id = path.stem
    title, sections = _sections(text)
    title = title or doc_id
    chunks: list[Chunk] = []
    for section, body in sections:
        for window in _windows(body.split(), max_words, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{len(chunks):03d}",
                    doc_id=doc_id,
                    title=title,
                    section=section,
                    text=" ".join(window),
                    source=str(path),
                )
            )
    return chunks


def load_corpus(src: str | Path, **kwargs) -> list[Chunk]:
    src = Path(src)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        raise FileNotFoundError(f"No .md, .txt or .pdf files found under {src}")
    return [c for f in files for c in chunk_document(f, **kwargs)]
