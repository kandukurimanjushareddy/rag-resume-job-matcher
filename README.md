# RAG-Based Resume & Job Profile Matching System

Keyword screening misses good candidates: a resume that says "sentence embeddings" fails a filter looking for
"semantic search". This project ranks resumes against a job description by **meaning and keywords together**,
then uses an LLM to explain each match, using only evidence found in the resume.

## How it works

```
Resumes (PDF / DOCX / TXT)
   │  parsing.py: extract text, split into sections (Summary, Skills, Projects, Experience, Education)
   ▼
Section-aware chunks ──► BM25 index (bm25.py)            ─┐
                    └──► all-MiniLM-L6-v2 → ChromaDB      ─┤  retrieval.py
                         (384-dim vectors, cosine)         │  Reciprocal Rank Fusion
                                                           ▼
                                           ranked chunks → grouped per candidate
                                                           ▼
                                  llm_match.py: Gemini reads the top chunks → JSON report
                                  (match score, matched / missing requirements, evidence)
```

| Step | Choice | Why |
| --- | --- | --- |
| Chunking | One chunk per resume section, long sections windowed at 120 words | Keeps skills and projects separate, so a match can say *where* the evidence is |
| Lexical search | BM25 (`k1 = 1.5`, `b = 0.75`), written from scratch | Exact tool names (e.g. "ChromaDB") matter in hiring; BM25 catches them |
| Semantic search | `all-MiniLM-L6-v2`, 384-dim, normalised, cosine similarity in ChromaDB | Catches synonyms and paraphrases that BM25 misses |
| Fusion | Reciprocal Rank Fusion, `score = Σ 1 / (60 + rank)` | Combines two rankings without tuning score scales against each other |
| Per-candidate score | best chunk + 0.5 × sum of other chunks | Rewards matches across several sections without favouring long resumes |
| LLM step | Gemini, prompt restricted to retrieved excerpts, JSON output | Missing evidence counts as *missing*; the model is told not to guess |

Without `GEMINI_API_KEY`, the system returns the retrieval score and top evidence only. It never invents a match score.

## Run it

```bash
pip install -r requirements.txt
python main.py --resumes sample_data/resumes --job sample_data/job_nlp_engineer.txt --top 3

export GEMINI_API_KEY=...            # optional: adds the LLM match report
uvicorn api:app --reload             # optional: POST /match {"job_description": "...", "top_n": 5}
```

## Evaluate retrieval

```bash
python evaluate.py --resumes sample_data/resumes --jobs sample_data --labels sample_data/labels.json --k 3
```
Reports **Hit@k** (share of jobs whose correct candidate is in the top k) and **MRR** (mean reciprocal rank).
The sample set has 3 fictional resumes and 1 job, so it checks that the code works. Real numbers need a larger labelled set.

## Tests

```bash
pytest -q
```
Tests use a deterministic hash embedder, so they run offline. They cover section parsing, windowing, BM25,
RRF, end-to-end ranking, evaluation metrics, JSON parsing and the no-API-key path.

## Project layout

```
matcher/parsing.py       load files, detect sections, chunk
matcher/bm25.py          BM25 scoring
matcher/vector_store.py  embedder, ChromaDB store, in-memory store
matcher/retrieval.py     hybrid retrieval, RRF, per-candidate aggregation
matcher/llm_match.py     prompt, Gemini call, JSON validation
main.py / api.py         CLI and FastAPI service
evaluate.py              Hit@k and MRR
```

## Limitations

- Section detection relies on common headings; resumes with unusual layouts fall back to one Summary chunk.
- Scanned (image-only) PDFs need OCR first.
- Fairness: the ranking only sees resume text. Don't use it as the sole basis for a hiring decision.

All sample resumes are fictional.
