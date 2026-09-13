"""src.features — embedding + FAISS indexing.

Public API:
    from src.features import Embedder, embed_chunks, build_faiss_index, save_index, load_index
"""

from src.features.embedder import (
    Embedder,
    build_faiss_index,
    embed_chunks,
    load_index,
    save_index,
)

__all__ = [
    "Embedder",
    "embed_chunks",
    "build_faiss_index",
    "save_index",
    "load_index",
]