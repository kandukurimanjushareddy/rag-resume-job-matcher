"""Load resumes (PDF, DOCX, TXT) and split them into section-aware chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Headings we recognise, mapped to a canonical section name.
SECTION_ALIASES = {
    "summary": "Summary", "profile": "Summary", "objective": "Summary", "about": "Summary",
    "skills": "Skills", "technical skills": "Skills", "core skills": "Skills",
    "projects": "Projects", "academic projects": "Projects",
    "experience": "Experience", "work experience": "Experience", "internships": "Experience",
    "education": "Education", "academics": "Education",
    "certifications": "Certifications", "achievements": "Certifications",
}


@dataclass
class Chunk:
    candidate_id: str
    section: str
    text: str
    chunk_id: str = field(default="")


def read_text(path: str | Path) -> str:
    """Return the plain text of a .pdf, .docx or .txt file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if suffix == ".docx":
        import docx
        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    raise ValueError(f"Unsupported file type: {suffix}")


def _heading(line: str) -> str | None:
    """Return the canonical section name if this line is a section heading."""
    clean = re.sub(r"[^a-z ]", "", line.strip().lower()).strip()
    if not clean or len(clean.split()) > 3:
        return None
    return SECTION_ALIASES.get(clean)


def split_sections(text: str) -> dict[str, str]:
    """Split resume text into {section: text}. Text before the first heading goes to 'Summary'."""
    sections: dict[str, list[str]] = {}
    current = "Summary"
    for line in text.splitlines():
        name = _heading(line)
        if name:
            current = name
            continue
        if line.strip():
            sections.setdefault(current, []).append(line.strip())
    return {name: " ".join(lines) for name, lines in sections.items()}


def chunk_resume(candidate_id: str, text: str, max_words: int = 120) -> list[Chunk]:
    """One chunk per section; long sections are split into windows of max_words words."""
    chunks: list[Chunk] = []
    for section, body in split_sections(text).items():
        words = body.split()
        for start in range(0, len(words), max_words):
            part = " ".join(words[start:start + max_words])
            idx = len(chunks)
            chunks.append(Chunk(candidate_id, section, part, f"{candidate_id}::{section}::{idx}"))
    return chunks


def load_folder(folder: str | Path) -> list[Chunk]:
    """Chunk every supported resume in a folder. The file name (without extension) is the candidate id."""
    chunks: list[Chunk] = []
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() in {".pdf", ".docx", ".txt"}:
            chunks.extend(chunk_resume(path.stem, read_text(path)))
    return chunks
