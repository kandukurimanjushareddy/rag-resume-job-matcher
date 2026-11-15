"""BM25 lexical retrieval, written out so every step can be explained."""
from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = {"a", "an", "and", "the", "of", "to", "in", "for", "on", "with", "is", "are", "be", "as", "at", "by", "or", "we"}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9+#.]+", text.lower()) if t not in STOPWORDS]


class BM25:
    """Okapi BM25.

    score(q, d) = sum over terms t in q of
        idf(t) * f(t, d) * (k1 + 1) / (f(t, d) + k1 * (1 - b + b * |d| / avgdl))
    """

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.freqs = [Counter(t) for t in self.tokens]
        self.lengths = [len(t) for t in self.tokens]
        self.avgdl = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        n = len(docs)
        df = Counter(term for toks in self.tokens for term in set(toks))
        # +1 inside the log keeps idf positive for very common terms
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for freq, length in zip(self.freqs, self.lengths):
            s = 0.0
            for term in q:
                f = freq.get(term, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * length / (self.avgdl or 1))
                    s += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
            out.append(s)
        return out
