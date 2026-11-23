"""embed_query must return the same vector as Chroma's padded embedder,
or retrieval results (and every eval number) would silently change."""
import numpy as np
import pytest

import indexer


@pytest.fixture(scope="module")
def ef():
    try:
        return indexer._get_embed_fn()
    except Exception as exc:  # model not downloadable here
        pytest.skip(f"embedding model unavailable: {exc}")


@pytest.mark.parametrize("text", [
    "How does HTTP Digest authentication retry the request after a 401 challenge?",
    "x",
    "Where is DEFAULT_REDIRECT_LIMIT defined? " * 40,  # longer than 256 tokens: truncated
])
def test_matches_chromas_padded_embedding(ef, text):
    a = np.array(ef([text])[0])
    b = np.array(indexer.embed_query(text))
    assert a.shape == b.shape == (384,)
    assert float(a @ b) > 0.99999
    assert np.abs(a - b).max() < 1e-5
