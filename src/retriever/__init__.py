"""src.retriever — top-k retrieval over a FAISS index.

Public API:
    from src.retriever import Retriever, RetrievedChunk
"""

from src.retriever.retriever import RetrievedChunk, Retriever

__all__ = ["Retriever", "RetrievedChunk"]