"""Static unit tests for the prompt builder + reader orchestrator (no model)."""

from src.data import Chunk
from src.generator import SYSTEM_PROMPT, build_prompt
from src.retriever import RetrievedChunk


def _chunk(i: int, page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=f"c{i:05d}",
        text=f"sample text {i}",
        source_path="x.pdf",
        page_number=page,
        char_start=i * 10,
        char_end=(i + 1) * 10,
    )


def test_build_prompt_includes_citations():
    retrieved = [
        RetrievedChunk(chunk=_chunk(0), score=0.9),
        RetrievedChunk(chunk=_chunk(1, page=2), score=0.7),
    ]
    prompt = build_prompt("What is X?", retrieved)
    assert "[c00000]" in prompt
    assert "[c00001]" in prompt
    assert "page 1" in prompt
    assert "page 2" in prompt
    assert "What is X?" in prompt


def test_build_prompt_handles_no_retrieval():
    prompt = build_prompt("Anything?", [])
    assert "Anything?" in prompt
    assert "Context:" in prompt


def test_system_prompt_exists():
    assert "retrieval-augmented" in SYSTEM_PROMPT.lower()
    assert "don't know" in SYSTEM_PROMPT.lower() or "do not know" in SYSTEM_PROMPT.lower()