"""PDF loading + sliding-window chunking.

Uses pypdf for text extraction first (handles designed PDFs better),
falls back to pdfplumber if pypdf fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    """A single chunk of text with provenance metadata."""

    chunk_id: str
    text: str
    source_path: str
    page_number: int | None
    char_start: int
    char_end: int

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_path": self.source_path,
            "page_number": self.page_number,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


def load_pdf(path: str | Path) -> list[dict]:
    """Load a PDF and return one record per page: {page_number, text}.

    Uses pypdf first for clean text extraction (handles designed/scanned PDFs better).
    Falls back to pdfplumber if pypdf fails or is unavailable.

    Args:
        path: Path to the PDF file.

    Returns:
        List of dicts, each with 'page_number' and 'text' keys.

    Raises:
        ImportError: If neither pypdf nor pdfplumber is installed.
        FileNotFoundError: If the PDF file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    # Try pypdf first — cleaner text extraction for designed PDFs
    try:
        import pypdf  # type: ignore
        reader = pypdf.PdfReader(str(path))
        pages: list[dict] = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page_number": i, "text": text})
        return pages
    except Exception:
        pass

    # Fallback to pdfplumber
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(str(path)) as pdf:
            pages: list[dict] = []
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append({"page_number": i, "text": text})
            return pages
    except ImportError as e:
        raise ImportError(
            "Either pypdf or pdfplumber is required for load_pdf. "
            "Install with `pip install pypdf pdfplumber`."
        ) from e
def load_text(path: str | Path) -> list[dict]:
    """Load a plain-text file as a single 'page'. Used for demos + fast iteration.

    The whole file is treated as one document (page_number=1) so the chunker can
    apply sliding-window chunking over it. For large multi-section files, the
    chunker will respect whitespace and produce clean chunks anyway.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    text = path.read_text(encoding="utf-8")
    return [{"page_number": 1, "text": text}]



def _snap_to_sentence_boundary(text: str, max_end: int) -> int:
    """Snap chunk end to nearest period boundary before max_end.

    Prefers period boundaries to avoid fragmenting text that uses
    line breaks as formatting (resumes, etc.). Returns max_end if
    no period boundary found within 150 chars.
    """
    search_start = max(0, max_end - 150)
    slice_text = text[search_start:max_end]
    # Find the last period boundary (not at the very end of the slice)
    idx = slice_text.rfind(".")
    if idx >= 0 and idx > 20 and idx < len(slice_text) - 5:
        return search_start + idx + 1
    return max_end


def chunk_text(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    source_path: str | Path = "",
) -> list[Chunk]:
    """Slide a window of `chunk_size` chars with `chunk_overlap` overlap across the
    concatenated page text, preserving per-page provenance.

    Snaps chunk boundaries to nearest sentence boundary (`.!`) to avoid
    cutting through sentences mid-way.
    """
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            f"Invalid chunk config: size={chunk_size} overlap={chunk_overlap}"
        )

    chunks: list[Chunk] = []
    counter = 0
    for page in pages:
        text = page["text"] or ""
        if not text.strip():
            continue
        start = 0
        while start < len(text):
            raw_end = min(start + chunk_size, len(text))
            # Snap to sentence boundary to avoid cutting mid-sentence
            end = _snap_to_sentence_boundary(text, raw_end)
            # If the snap didn't help (same as raw_end), just use raw_end
            # But don't let end go below start + 50 chars
            if end <= start:
                end = raw_end
            chunk_text_str = text[start:end].strip()
            if chunk_text_str:
                chunks.append(
                    Chunk(
                        chunk_id=f"c{counter:05d}",
                        text=chunk_text_str,
                        source_path=str(source_path),
                        page_number=page["page_number"],
                        char_start=start,
                        char_end=end,
                    )
                )
                counter += 1
            if end >= len(text):
                break
            start = end - chunk_overlap
    return chunks