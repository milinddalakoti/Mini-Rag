"""src.data — PDF + text loading + chunking.

Public API:
    from src.data import load_pdf, load_text, chunk_text, Chunk
"""

from src.data.loader import Chunk, chunk_text, load_pdf, load_text

__all__ = ["Chunk", "chunk_text", "load_pdf", "load_text"]