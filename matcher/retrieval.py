"""Hybrid retrieval: BM25 + vector search, fused with Reciprocal Rank Fusion, then grouped per candidate."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .bm25 import BM25
from .parsing import Chunk

RRF_K = 60  # standard constant from Cormack et al. (2009); damps the effect of top ranks


@dataclass
class CandidateResult:
    candidate_id: str
    score: float
    chunks: list[tuple[Chunk, float]] = field(default_factory=list)

    def context(self, max_chunks: int = 4) -> str:
        return "\n".join(f"[{c.section}] {c.text}" for c, _ in self.chunks[:max_chunks])


def rrf(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d)), rank starting at 1."""
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] += 1.0 / (k + rank)
    return dict(fused)


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], store):
        self.chunks = {c.chunk_id: c for c in chunks}
        self.order = list(self.chunks)
        self.bm25 = BM25([self.chunks[i].text for i in self.order])
        self.store = store
        self.store.add(list(self.chunks.values()))

    def lexical(self, query: str, k: int) -> list[str]:
        scores = self.bm25.scores(query)
        ranked = sorted(zip(self.order, scores), key=lambda x: -x[1])
        return [cid for cid, s in ranked[:k] if s > 0]

    def semantic(self, query: str, k: int) -> list[str]:
        return [c.chunk_id for c, _ in self.store.search(query, k)]

    def search_chunks(self, query: str, k: int = 20) -> list[tuple[Chunk, float]]:
        fused = rrf([self.lexical(query, k), self.semantic(query, k)])
        ranked = sorted(fused.items(), key=lambda x: -x[1])
        return [(self.chunks[cid], score) for cid, score in ranked]

    def rank_candidates(self, job_description: str, top_n: int = 5, k: int = 20) -> list[CandidateResult]:
        """Aggregate chunk scores per candidate: best chunk + 0.5 x sum of the rest.

        Rewarding several matching sections (skills AND projects) beats one lucky keyword hit,
        while the 0.5 weight stops long resumes winning just by having more chunks.
        """
        per_candidate: dict[str, list[tuple[Chunk, float]]] = defaultdict(list)
        for chunk, score in self.search_chunks(job_description, k):
            per_candidate[chunk.candidate_id].append((chunk, score))
        results = []
        for cid, hits in per_candidate.items():
            hits.sort(key=lambda x: -x[1])
            score = hits[0][1] + 0.5 * sum(s for _, s in hits[1:])
            results.append(CandidateResult(cid, score, hits))
        results.sort(key=lambda r: -r.score)
        return results[:top_n]
