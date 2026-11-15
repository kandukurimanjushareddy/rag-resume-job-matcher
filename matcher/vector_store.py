"""Embeddings and vector stores.

The real pipeline uses sentence-transformers/all-MiniLM-L6-v2 (384-dimensional vectors) and ChromaDB.
Both sit behind small interfaces so tests can swap in a fake embedder and an in-memory store.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .parsing import Chunk

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """all-MiniLM-L6-v2: 384-dim embeddings, normalised so dot product = cosine similarity."""

    def __init__(self, model_name: str = MODEL_NAME):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))


class InMemoryStore:
    """Cosine-similarity search over a numpy matrix. Used in tests and for small runs."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk]) -> None:
        vecs = self.embedder.embed([c.text for c in chunks])
        vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
        self.chunks.extend(chunks)
        self.matrix = vecs if self.matrix is None else np.vstack([self.matrix, vecs])

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if self.matrix is None:
            return []
        q = self.embedder.embed([query])[0]
        q = q / max(np.linalg.norm(q), 1e-12)
        sims = self.matrix @ q
        top = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in top]


class ChromaStore:
    """Persistent ChromaDB collection using cosine distance."""

    def __init__(self, embedder: Embedder, path: str = "chroma_db", collection: str = "resumes"):
        import chromadb
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})
        self.chunks: dict[str, Chunk] = {}

    def add(self, chunks: list[Chunk]) -> None:
        vecs = self.embedder.embed([c.text for c in chunks])
        self.col.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vecs.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{"candidate_id": c.candidate_id, "section": c.section} for c in chunks],
        )
        self.chunks.update({c.chunk_id: c for c in chunks})

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        q = self.embedder.embed([query])[0].tolist()
        res = self.col.query(query_embeddings=[q], n_results=k)
        out = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            chunk = self.chunks.get(cid) or Chunk(meta["candidate_id"], meta["section"], doc, cid)
            out.append((chunk, 1.0 - dist))  # cosine distance -> similarity
        return out
