"""Measure retrieval quality: Hit@k and MRR on a labelled set of (job file -> correct candidate).

    python evaluate.py --resumes sample_data/resumes --jobs sample_data --labels sample_data/labels.json

Hit@k = share of jobs whose correct candidate is in the top k.
MRR   = mean of 1 / (rank of the correct candidate); 0 if not retrieved.
Report these numbers only from a labelled set you built and checked yourself.
"""
import argparse
import json
from pathlib import Path

from matcher import HybridRetriever, load_folder


def evaluate(retriever, jobs_dir: str, labels: dict, k: int = 3) -> dict:
    hits, rr = 0, 0.0
    for job_file, correct in labels.items():
        job = Path(jobs_dir, job_file).read_text(encoding="utf-8")
        ranked = [c.candidate_id for c in retriever.rank_candidates(job, top_n=50)]
        if correct in ranked:
            pos = ranked.index(correct) + 1
            rr += 1 / pos
            hits += pos <= k
    n = len(labels)
    return {"jobs": n, f"hit@{k}": round(hits / n, 3), "mrr": round(rr / n, 3)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resumes", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    from matcher.vector_store import InMemoryStore, SentenceTransformerEmbedder
    r = HybridRetriever(load_folder(a.resumes), InMemoryStore(SentenceTransformerEmbedder()))
    print(evaluate(r, a.jobs, json.loads(Path(a.labels).read_text()), a.k))
