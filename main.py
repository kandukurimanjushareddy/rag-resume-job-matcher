"""Rank resumes in a folder against a job description.

    python main.py --resumes sample_data/resumes --job sample_data/job_nlp_engineer.txt --top 3
"""
import argparse
import json
from pathlib import Path

from matcher import HybridRetriever, load_folder
from matcher.llm_match import match


def build(resumes: str, store: str):
    from matcher.vector_store import ChromaStore, InMemoryStore, SentenceTransformerEmbedder
    embedder = SentenceTransformerEmbedder()
    vs = ChromaStore(embedder) if store == "chroma" else InMemoryStore(embedder)
    return HybridRetriever(load_folder(resumes), vs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resumes", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--store", choices=["chroma", "memory"], default="chroma")
    args = ap.parse_args()

    job = Path(args.job).read_text(encoding="utf-8")
    retriever = build(args.resumes, args.store)
    for rank, cand in enumerate(retriever.rank_candidates(job, top_n=args.top), start=1):
        print(f"\n#{rank}  {cand.candidate_id}")
        print(json.dumps(match(job, cand), indent=2))


if __name__ == "__main__":
    main()
