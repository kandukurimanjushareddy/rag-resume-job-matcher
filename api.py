"""FastAPI service: POST a job description, get ranked candidates.

    uvicorn api:app --reload
"""
from fastapi import FastAPI
from pydantic import BaseModel

from main import build
from matcher.llm_match import match

app = FastAPI(title="RAG Resume & Job Matcher")
retriever = build("sample_data/resumes", "memory")


class Query(BaseModel):
    job_description: str
    top_n: int = 5


@app.post("/match")
def match_job(q: Query):
    return [match(q.job_description, c) for c in retriever.rank_candidates(q.job_description, q.top_n)]
