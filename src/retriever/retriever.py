"""Top-k retrieval wrapper around a FAISS index + the original Chunk list."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # inner product (cosine sim, since vectors are L2-normalised)


class Retriever:
    def __init__(
        self,
        index,
        chunks: list[Chunk],
    ) -> None:
        if index.ntotal != len(chunks):
            raise ValueError(
                f"Index size ({index.ntotal}) != chunks list size ({len(chunks)}). "
                "Did you forget to re-build the index after editing chunks?"
            )
        self.index = index
        self.chunks = chunks

    def retrieve(self, query_vec: np.ndarray, top_k: int = 4) -> list[RetrievedChunk]:
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)
        scores, ids = self.index.search(query_vec, top_k)
        out: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            out.append(RetrievedChunk(chunk=self.chunks[int(idx)], score=float(score)))
        return out