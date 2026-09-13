"""End-to-end Reader: ties data + features + retriever + generator."""

from __future__ import annotations

from dataclasses import dataclass

from src.data import Chunk, load_pdf, load_text, chunk_text
from src.features import Embedder, build_faiss_index, embed_chunks
from src.retriever import Retriever, RetrievedChunk
from src.generator import ExtractiveGenerator
try:
    from src.generator import Generator
except Exception:
    Generator = None


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[RetrievedChunk]


class Reader:
    """High-level orchestrator. Build once with config, then call `.ask(q)`."""

    def __init__(
        self,
        embedder: Embedder,
        generator: Generator,
        top_k: int = 4,
    ) -> None:
        self.embedder = embedder
        self.generator = generator
        self.top_k = top_k
        self.chunks: list[Chunk] = []
        self.retriever: Retriever | None = None

    def ingest(self, pdf_path: str, is_text: bool = False) -> int:
        """Load + chunk + embed + index the document. Returns # chunks."""
        pages = load_text(pdf_path) if is_text else load_pdf(pdf_path)
        self.chunks = chunk_text(
            pages,
            chunk_size=500,
            chunk_overlap=50,
            source_path=pdf_path,
        )
        vectors = embed_chunks(self.embedder, self.chunks)
        index = build_faiss_index(vectors)
        self.retriever = Retriever(index=index, chunks=self.chunks)
        return len(self.chunks)

    def ask(self, question: str) -> Answer:
        if self.retriever is None:
            raise RuntimeError("Call .ingest(pdf_path) before .ask(question).")
        q_vec = self.embedder.embed([question])
        retrieved = self.retriever.retrieve(q_vec, top_k=self.top_k)
        text = self.generator.generate(question, retrieved)
        return Answer(question=question, answer=text, citations=retrieved)