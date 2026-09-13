"""Unit tests for the chunker."""

from src.data import chunk_text


def test_chunk_text_basic():
    pages = [{"page_number": 1, "text": "a" * 1000}]
    chunks = chunk_text(pages, chunk_size=200, chunk_overlap=50, source_path="x.pdf")
    # 1000 chars / (200-50) windows = ceil(1000/150) ≈ 7 chunks (with overlap)
    assert len(chunks) >= 6
    assert all(c.source_path == "x.pdf" for c in chunks)
    assert all(c.page_number == 1 for c in chunks)


def test_chunk_text_rejects_bad_config():
    pages = [{"page_number": 1, "text": "hello"}]
    # overlap >= size
    try:
        chunk_text(pages, chunk_size=100, chunk_overlap=100, source_path="x.pdf")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for overlap >= size")


def test_chunk_text_skips_blank_pages():
    pages = [
        {"page_number": 1, "text": ""},
        {"page_number": 2, "text": "   \n  "},
        {"page_number": 3, "text": "actual content here"},
    ]
    chunks = chunk_text(pages, chunk_size=100, chunk_overlap=20, source_path="x.pdf")
    assert len(chunks) == 1
    assert chunks[0].text == "actual content here"
    assert chunks[0].page_number == 3


def test_chunk_text_preserves_provenance():
    pages = [
        {"page_number": 1, "text": "page one content"},
        {"page_number": 2, "text": "page two content"},
    ]
    chunks = chunk_text(pages, chunk_size=10, chunk_overlap=0, source_path="doc.pdf")
    page_nums = sorted({c.page_number for c in chunks})
    assert page_nums == [1, 2]


def test_chunk_ids_are_unique():
    pages = [{"page_number": 1, "text": "x" * 100}]
    chunks = chunk_text(pages, chunk_size=20, chunk_overlap=0, source_path="x.pdf")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))