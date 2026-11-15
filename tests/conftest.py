import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class HashEmbedder:
    """Deterministic bag-of-words embedder so tests run without downloading a model."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim))
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, hash(w.strip(".,")) % self.dim] += 1.0
        return out


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def data_dir():
    return Path(__file__).resolve().parents[1] / "sample_data"
