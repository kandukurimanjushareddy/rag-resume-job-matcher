"""LLM step of the RAG pipeline: turn retrieved resume context into a structured match report."""
from __future__ import annotations

import json
import os
import re

from .retrieval import CandidateResult

PROMPT = """You are screening candidates for a job. Use ONLY the resume excerpts below.
If the excerpts do not show a requirement, count it as missing; do not guess.

JOB DESCRIPTION:
{job}

RESUME EXCERPTS FOR CANDIDATE {cid}:
{context}

Return JSON only, with these keys:
{{"candidate_id": "{cid}", "match_score": <integer 0-100>, "matched_requirements": [..],
  "missing_requirements": [..], "evidence": [<short quotes from the excerpts>], "summary": "<one sentence>"}}"""


def build_prompt(job: str, cand: CandidateResult) -> str:
    return PROMPT.format(job=job.strip(), cid=cand.candidate_id, context=cand.context())


def parse_json(text: str) -> dict:
    """Extract the first JSON object from a model reply (models sometimes wrap it in ```json fences)."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("No JSON object in model reply")
    data = json.loads(match.group(0))
    data["match_score"] = max(0, min(100, int(data.get("match_score", 0))))
    for key in ("matched_requirements", "missing_requirements", "evidence"):
        data.setdefault(key, [])
    return data


def call_gemini(prompt: str, model: str = "gemini-2.0-flash") -> str:
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return client.models.generate_content(model=model, contents=prompt).text


def match(job: str, cand: CandidateResult, llm=None) -> dict:
    """Score one candidate. Without an API key, return the retrieval result only (no invented judgement)."""
    llm = llm or (call_gemini if os.environ.get("GEMINI_API_KEY") else None)
    if llm is None:
        return {"candidate_id": cand.candidate_id, "retrieval_score": round(cand.score, 4),
                "note": "Set GEMINI_API_KEY to get an LLM match report.",
                "top_evidence": [f"[{c.section}] {c.text[:160]}" for c, _ in cand.chunks[:3]]}
    report = parse_json(llm(build_prompt(job, cand)))
    report["retrieval_score"] = round(cand.score, 4)
    return report
