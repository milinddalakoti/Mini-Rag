"""Smoke test for embedder — uses a small model on CPU.

Marked slow; skipped by default if sentence-transformers is not installed.
Enable with: pytest tests/test_embed.py -k embed --no-skip
"""

import numpy as np
import pytest

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


@pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS,
    reason="sentence-transformers not installed",
)
def test_embedder_returns_normalised_vectors():
    from src.features import Embedder

    e = Embedder()
    vecs = e.embed(["hello world", "goodbye world"])
    assert vecs.shape[0] == 2
    assert vecs.shape[1] == e.dim
    # vectors should be L2-normalised → norms ≈ 1.0
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, np.ones(2), atol=1e-5)
    # dtype
    assert vecs.dtype == np.float32


def test_embedder_contract():
    """Static contract: import-time, no model load."""
    from src.features import Embedder

    assert hasattr(Embedder, "embed")
    assert hasattr(Embedder, "__init__")
