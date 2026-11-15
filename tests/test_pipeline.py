from matcher import HybridRetriever, chunk_resume, load_folder, rrf
from matcher.bm25 import BM25
from matcher.llm_match import match, parse_json
from matcher.parsing import split_sections
from matcher.vector_store import InMemoryStore
from evaluate import evaluate


def test_sections_are_detected():
    text = "Jane\nSummary\nNLP engineer\nSkills\nPython, SQL\nEducation\nB.Tech 2026"
    s = split_sections(text)
    assert s["Skills"] == "Python, SQL"
    assert "B.Tech" in s["Education"]


def test_long_sections_are_windowed():
    text = "Projects\n" + " ".join(f"w{i}" for i in range(250))
    chunks = chunk_resume("c1", text, max_words=100)
    assert [c.section for c in chunks] == ["Projects"] * 3
    assert len({c.chunk_id for c in chunks}) == 3


def test_bm25_prefers_matching_doc():
    bm = BM25(["python fastapi docker", "excel tableau sales"])
    a, b = bm.scores("python docker")
    assert a > b and b == 0


def test_rrf_rewards_agreement():
    fused = rrf([["x", "y"], ["x", "z"]])
    assert fused["x"] > fused["y"] and fused["x"] > fused["z"]


def test_nlp_job_ranks_nlp_candidate_first(embedder, data_dir):
    r = HybridRetriever(load_folder(data_dir / "resumes"), InMemoryStore(embedder))
    job = (data_dir / "job_nlp_engineer.txt").read_text()
    ranked = r.rank_candidates(job, top_n=3)
    assert ranked[0].candidate_id == "rahul_menon"


def test_evaluate_reports_hit_and_mrr(embedder, data_dir):
    r = HybridRetriever(load_folder(data_dir / "resumes"), InMemoryStore(embedder))
    out = evaluate(r, data_dir, {"job_nlp_engineer.txt": "rahul_menon"}, k=1)
    assert out == {"jobs": 1, "hit@1": 1.0, "mrr": 1.0}


def test_llm_report_is_parsed_and_clamped(embedder, data_dir):
    r = HybridRetriever(load_folder(data_dir / "resumes"), InMemoryStore(embedder))
    cand = r.rank_candidates("python embeddings chromadb", top_n=1)[0]
    fake = lambda prompt: '```json\n{"candidate_id": "x", "match_score": 140, "summary": "ok"}\n```'
    report = match("job", cand, llm=fake)
    assert report["match_score"] == 100 and report["missing_requirements"] == []


def test_no_api_key_returns_evidence_not_a_made_up_score(embedder, data_dir, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r = HybridRetriever(load_folder(data_dir / "resumes"), InMemoryStore(embedder))
    cand = r.rank_candidates("python", top_n=1)[0]
    report = match("python", cand)
    assert "match_score" not in report and report["top_evidence"]


def test_parse_json_rejects_non_json():
    import pytest
    with pytest.raises(ValueError):
        parse_json("no json here")
