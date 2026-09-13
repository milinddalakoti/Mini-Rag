"""Embedding + FAISS index helpers.

Wraps sentence-transformers + faiss so the rest of the code never imports them
directly — keeps the dependency on HF hidden behind a thin, swappable interface.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data import Chunk


class Embedder:
    """Wraps a sentence-transformers model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required. `pip install sentence-transformers`."
            ) from e
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = int(self.model.get_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return float32 array shape (N, dim), L2-normalised."""
        vecs = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return vecs.astype(np.float32)


def build_faiss_index(vectors: np.ndarray) -> "faiss.IndexFlatIP":
    """Build an inner-product FAISS index. Vectors must be L2-normalised."""
    try:
        import faiss  # type: ignore
    except ImportError as e:
        raise ImportError("faiss-cpu is required. `pip install faiss-cpu`.") from e

    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_index(index: "faiss.Index", path: str | Path) -> None:
    """Persist a FAISS index to disk."""
    try:
        import faiss  # type: ignore
    except ImportError as e:
        raise ImportError("faiss-cpu is required.") from e

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: str | Path) -> "faiss.Index":
    """Load a FAISS index from disk."""
    try:
        import faiss  # type: ignore
    except ImportError as e:
        raise ImportError("faiss-cpu is required.") from e

    return faiss.read_index(str(path))


def embed_chunks(embedder: Embedder, chunks: list[Chunk]) -> np.ndarray:
    """Convenience: batch-embed a list of Chunks."""
    texts = [c.text for c in chunks]
    return embedder.embed(texts)