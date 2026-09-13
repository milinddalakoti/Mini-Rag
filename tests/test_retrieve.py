"""Smoke test for the retriever — uses random vectors (no model load)."""

import numpy as np


def test_retriever_retrieves_self_first():
    """A query vector equal to chunk k's vector should return k first."""
    from src.data import Chunk
    from src.features import build_faiss_index
    from src.retriever import Retriever

    rng = np.random.default_rng(0)
    n_chunks = 5
    dim = 16
    # Make unit vectors
    vecs = rng.standard_normal((n_chunks, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    chunks = [
        Chunk(
            chunk_id=f"c{i:03d}",
            text=f"chunk {i}",
            source_path="x.pdf",
            page_number=1,
            char_start=0,
            char_end=10,
        )
        for i in range(n_chunks)
    ]

    retriever = Retriever(index=build_faiss_index(vecs), chunks=chunks)
    out = retriever.retrieve(vecs[2], top_k=3)
    assert out[0].chunk.chunk_id == "c002"
    assert out[0].score > out[1].score > out[2].score


def test_retriever_rejects_size_mismatch():
    """Index and chunks must have the same length."""
    import pytest

    from src.data import Chunk
    from src.features import build_faiss_index
    from src.retriever import Retriever

    vecs = np.zeros((2, 4), dtype=np.float32)
    chunks = [
        Chunk(
            chunk_id="c000",
            text="x",
            source_path="x.pdf",
            page_number=1,
            char_start=0,
            char_end=1,
        )
    ]
    with pytest.raises(ValueError, match="Index size"):
        Retriever(index=build_faiss_index(vecs), chunks=chunks)