# 🎯 Resume ↔ Job-Description Fit Scorer

**A deterministic, recruiter-actionable candidate-fit evaluator that scores a resume against a job description criterion-by-criterion — not one opaque LLM number — built on FastAPI, Pydantic v2, and OpenRouter.**

---

## 🧭 Business Context

Recruiters and hiring managers routinely get a single "match %" from resume-screening tools with no way to see *why*. That number can't be audited, can't be explained to a candidate, and can't be trusted for anything high-stakes.

**🎯 Goal:** decompose JD-vs-resume fit into independently scored, independently weighted, independently explained criteria — so every number on the page has a reason attached to it.

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Language | Python ≥ 3.11 |
| Backend / API | FastAPI, Uvicorn, `python-multipart` |
| Validation | Pydantic v2 |
| LLM Provider | OpenRouter (open-source/openly-licensed models, primary + fallback) |
| HTTP Client | httpx |
| Parsing | PyMuPDF (PDF), native Python (TXT) |
| Config | PyYAML (`config/weights.yaml`), `python-dotenv` (`.env`) |
| Testing | pytest |
| Tooling | uv |

---

## 🔄 Project Workflow

```
Client
  │  POST /score  (JD text + resume file)
  ▼
Resume Parser (PyMuPDF / plain text)
  │  validates extension, %PDF- magic bytes, 5MB limit
  ▼
Cache Check (SHA-256 of JD + resume)
  │  identical pair → return cached ScoreResponse, skip LLM entirely
  ▼
LLM Call #1 — Criteria Extraction        (Primary → Fallback → Fail)
  │  delimited JD → discrete {name, description, weight_hint} list
  ▼
Weight Resolution
  │  config/weights.yaml overrides LLM weight_hint per criterion
  ▼
LLM Call #2 — Batched Criterion Scoring  (Primary → Fallback → Fail)
  │  one call scores ALL criteria against resume, 0–100 rubric
  ▼
Output Sanity Check
  │  score=100 with no evidence tokens → suspicious: true
  ▼
Deterministic Weighted Aggregation
  │  pure Python: Σ(score × weight) / Σ(weight) — no LLM arithmetic
  ▼
Instrumentation + PII-Safe Persistence
  │  latency/tokens/cost → metrics.jsonl · hashes+scores only → runs.jsonl
  ▼
Structured JSON Response (ScoreResponse)
```

---

## ✨ Features / Highlights

- 🧮 **Deterministic aggregation** — the overall score is `Σ(score × weight) / Σ(weight)`, computed in plain Python, never delegated to the LLM. Removes a whole class of arithmetic hallucination.
- 📏 **Fixed scoring rubric + temperature 0** — anchors at `0 = Absent`, `50 = Partial/Adjacent`, `100 = Explicit Match`, so two near-duplicate resumes land close together instead of drifting on model mood.
- 🔁 **Primary → Fallback → Fail resiliency** — one attempt per model, two models max, then a typed `502`. No silent retry storms.
- 🛡️ **Prompt-injection defense** — JD and resume text are wrapped in explicit delimiters (`<<<JD_START>>>`, `<<<RESUME_START>>>`); any `score: 100` with no supporting evidence in the reasoning gets flagged `suspicious: true` rather than silently trusted.
- 🔒 **PII-safe local storage** — `runs.jsonl` stores SHA-256 hashes, timestamps, and scores only. Raw resume text never touches disk.
- ⚡ **Local caching** — identical `(JD, resume)` pairs return instantly from cache, zero repeat LLM cost.
- 📊 **Built-in instrumentation** — every LLM call logs latency, prompt/completion/total tokens, and cost to `metrics.jsonl`.

---

## 📈 Calibration & Metrics Results

A live-LLM calibration run on OpenRouter, two near-duplicate strong resumes (A, B) and one deliberately weak resume (C), same JD:

| Resume | Overall Score | Note |
|---|---|---|
| A — Baseline Strong | **98.18** | |
| B — Near-Duplicate of A | **85.94** | Drift vs A: **12.24 pts** (spec ceiling: ≤ 40 → **PASS**) |
| C — Deliberately Weak | **0.00** | Separation margin from A: **98.18 pts** |

**Tracked across 6 real calls:** 117,070.94 ms total latency (avg 19.51s/call) · 11,688 tokens (4,462 prompt / 7,226 completion) · criterion-scoring calls accounted for **68.03%** of completion tokens — the evidence-reasoning step, not extraction, drives most of the generation cost. Full write-up in [`docs/explanation.md`](docs/explanation.md).

---

## 📁 Project Structure

```
resume-jd-fit-scorer/
├── app/
│   ├── main.py             # FastAPI app, /score /health / routes, pipeline orchestration
│   ├── models.py           # Pydantic schemas: Criterion, CriterionScore, ScoreResponse, ErrorResponse
│   ├── resume_parser.py    # PyMuPDF/TXT extraction, magic-byte + size validation
│   ├── llm_client.py       # OpenRouter wrapper, Primary→Fallback→Fail sequencing
│   ├── extraction.py       # LLM Call #1 — JD to discrete criteria
│   ├── scoring.py          # LLM Call #2 — batched scoring + suspicious-score check
│   ├── config_loader.py    # weights.yaml loading + per-criterion weight resolution
│   ├── metrics.py          # latency/token/cost instrumentation → metrics.jsonl
│   └── storage.py          # PII-safe run persistence + cache lookup → runs.jsonl
├── config/
│   └── weights.yaml        # scoring weights ONLY — no runtime config mixed in
├── docs/
│   └── explanation.md      # design-decision, failure, metric, and next-steps write-up
├── tests/                  # pytest suite — network calls mocked, deterministic
├── .env.example            # OPENROUTER_API_KEY, LLM_MODEL, FALLBACK_MODEL, etc.
├── pyproject.toml
└── requirements.txt
```

---

## ⚙️ Setup

```bash
git clone https://github.com/Tapas2050/resume-jd-fit-scorer.git
cd resume-jd-fit-scorer

# using uv (recommended)
uv sync

# or standard venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

```bash
cp .env.example .env
# edit .env — set OPENROUTER_API_KEY

uv run uvicorn app.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

```bash
uv run pytest -v   # full suite mocks all network calls — no API credits consumed
```

---

## 📚 Concepts Used

- Criterion-decomposed evaluation instead of single blended LLM scoring
- Deterministic post-processing over stochastic LLM output (rubric-anchoring + pure-Python aggregation)
- Resilient external-API integration: bounded retry, primary/fallback provider routing, typed error taxonomy
- Prompt-injection isolation via explicit input delimiting
- PII-minimization by design (hash-only persistence)
- Content-addressed caching (SHA-256 of inputs) for idempotency and cost control
- Config-driven business logic (externalized weights) vs. hardcoded rules
- Async I/O hygiene — offloading blocking PDF parsing to a worker thread

---

## 🔧 Known Gaps / Future Improvements

- **Scanned/image-only PDFs**: no OCR layer — these return a clean `422` instead of a hallucinated read, by design, but OCR fallback is a real next step.
- **English-only**: rubric anchors and extraction prompts aren't validated against non-English JDs/resumes.
- **Free-tier model latency**: `:free` OpenRouter endpoints are subject to provider queueing — timeout is externally configured but not eliminated.
- **No CI pipeline** yet — tests run locally/manually; no GitHub Actions workflow committed.
- **No license file** — add one if this repo is meant for reuse.

---

## 👨‍💻 Author

**Tapas** — Software Engineer, AI/ML & backend/full-stack.
[github.com/Tapas2050](https://github.com/Tapas2050)

---

⭐ If you found this project useful, consider giving it a star!