# Resume ↔ Job-Description Fit Scorer

Deterministic, recruiter-actionable candidate fit evaluation between a Job Description (JD) and a Candidate Resume using FastAPI, Pydantic v2, OpenRouter, and local privacy-preserving storage.

---

## Overview

The **Resume ↔ Job-Description Fit Scorer** evaluates candidate resumes against job descriptions by decomposing the evaluation into discrete, independently scored criteria rather than relying on a single, opaque LLM score. 

### Core Architecture & Pipeline

```
Client POST /score (JD text + Resume PDF/TXT)
  │
  ├── 1. Resume Parser (PyMuPDF / direct text)
  │      └── Validates file extension, %PDF- magic bytes, and 5MB size limit
  │
  ├── 2. Caching & Idempotency Layer
  │      └── Computes SHA-256(JD) and SHA-256(Resume); returns cached ScoreResponse if identical
  │
  ├── 3. LLM Call #1: Criteria Extraction (Primary -> Fallback -> Fail)
  │      └── Extracts discrete criteria {name, description, weight_hint} from delimited JD
  │
  ├── 4. Config-Driven Weights
  │      └── Resolves criterion weights from config/weights.yaml (externalized business logic)
  │
  ├── 5. LLM Call #2: Batched Criterion Scoring (Primary -> Fallback -> Fail)
  │      └── Evaluates all criteria against resume text in ONE batched call with 0-100 rubric
  │
  ├── 6. Security & Output Sanity Check
  │      └── Flags suspicious=true if score is 100 without evidence tokens; enforces prompt isolation
  │
  ├── 7. Deterministic Weighted Aggregation
  │      └── Pure Python calculation: overall = Σ(score × weight) / Σ(weight)
  │
  ├── 8. Instrumentation & PII-Safe Local Persistence
  │      └── Logs latency/tokens/cost to metrics.jsonl; stores hashes & scores to runs.jsonl
  │
  └── 9. Structured JSON Response (ScoreResponse)
```

---

## Key Features

1. **Deterministic Aggregation**: The overall score is calculated purely in Python ($\sum (score \times weight) / \sum weight$), preventing arithmetic hallucinations from the LLM.
2. **Fixed Scoring Rubric**: Prompt anchors at `0 = Absent`, `50 = Partial / Adjacent`, and `100 = Explicit Match` paired with `temperature = 0` ensure consistent scoring across candidates.
3. **Primary → Fallback → Fail Resiliency**: Single attempt per model across primary and fallback endpoints. If both fail, returns a typed HTTP 502 error without runaway retries.
4. **Security & Prompt-Injection Defense**: Inputs are strictly delimited (`<<<JD_START>>>` and `<<<RESUME_START>>>`). An output sanity check flags scores of 100 that lack evidence citations (`suspicious: true`) without silently tampering with scores.
5. **PII-Safe Local Storage**: Resumes are never stored in raw text. Only cryptographic SHA-256 digests, timestamps, run IDs, and score metadata are persisted to `runs.jsonl`.
6. **Local Caching**: Identical `(JD, resume)` pairs return cached results immediately, avoiding redundant LLM latency and cost.
7. **Comprehensive Instrumentation**: Automatically records wall-clock latency (ms), prompt tokens, completion tokens, total tokens, and cost to `metrics.jsonl`.

---

## API Specification

### `POST /score`
Scores a candidate resume against a job description.

* **Content-Type**: `multipart/form-data`
* **Parameters**:
  * `job_description` (string, required): Raw text of the job description.
  * `resume_file` (file upload, required): Candidate resume in `.pdf` or `.txt` format (max 5 MB).

#### Success Response (`HTTP 200 OK`)
```json
{
  "overall_score": 85.94,
  "criteria": [
    {
      "name": "5+ years backend software engineering experience",
      "score": 50.0,
      "weight": 3.0,
      "reasoning": "Employment history shows 4 years of professional backend experience (2020-2022 and 2022-present). Career summary claims 6 years but not verifiable from listed roles.",
      "suspicious": false
    },
    {
      "name": "Relational database expertise (PostgreSQL) and query optimization",
      "score": 100.0,
      "weight": 2.5,
      "reasoning": "Direct experience tuning PostgreSQL queries, managing schema updates, developing custom indexes, and achieving 35% reduction in 99th percentile response times.",
      "suspicious": false
    }
  ],
  "model_used": "nvidia/nemotron-3-ultra-550b-a55b:free",
  "run_id": "bf248055-7aab-4ba6-967b-e009bb36b31b"
}
```

#### Error Responses
* **`HTTP 413 Payload Too Large`**: File exceeds the 5 MB limit.
  ```json
  {"error": "file_too_large", "detail": "File size (6.10 MB) exceeds maximum allowed size (5.00 MB)."}
  ```
* **`HTTP 422 Unprocessable Entity`**: Unreadable, empty, scanned/image-only PDF, or unsupported file type.
  ```json
  {"error": "unparsable_resume", "detail": "PDF contains no extractable text (possibly scanned or image-only)."}
  ```
* **`HTTP 502 Bad Gateway`**: Upstream LLM provider failure across both primary and fallback models.
  ```json
  {"error": "llm_call_failed", "detail": "Both primary and fallback models failed criterion scoring."}
  ```

### Utility Endpoints
* `GET /health`: Health check returning service status (`{"status": "ok", "app": "resume-jd-fit-scorer"}`).
* `GET /`: Root verification endpoint.

---

## Configuration

All runtime settings are managed via environment variables in `.env` (git-ignored, template in `.env.example`):

| Variable | Description | Default / Example |
| :--- | :--- | :--- |
| `OPENROUTER_API_KEY` | OpenRouter authentication key | `sk-or-v1-...` |
| `OPENROUTER_API_URL` | Full Chat Completions API endpoint | `https://openrouter.ai/api/v1/chat/completions` |
| `LLM_BASE_URL` | Base provider URL (optional alternative) | `https://openrouter.ai/api/v1` |
| `LLM_MODEL` | Primary open-source model identifier | `nvidia/nemotron-3-ultra-550b-a55b:free` |
| `FALLBACK_MODEL` | Fallback open-source model identifier | `nvidia/nemotron-3.5-lightning:free` |
| `LLM_TEMPERATURE` | Sampling temperature (`0` for determinism) | `0` |
| `LLM_TIMEOUT_SECONDS` | Network timeout for OpenRouter calls (Master §2.5) | `15` |

### Externalized Scoring Weights (`config/weights.yaml`)
Business weights are separated from code logic:
```yaml
criteria_weights:
  required_skill: 3.0
  years_experience: 2.5
  domain_background: 2.0
  education: 1.0
  nice_to_have: 1.0
fallback_weight: 1.0
```

---

## Installation & Setup

### Prerequisites
* Python `>= 3.11`
* [uv](https://docs.astral.sh/uv/) (recommended) or standard `pip`

### 1. Clone & Install
```bash
git clone https://github.com/Tapas2050/resume-jd-fit-scorer.git
cd resume-jd-fit-scorer

# Install dependencies using uv
uv sync
```

Or using standard Python venv:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and supply your OPENROUTER_API_KEY
```

### 3. Run Development Server
```bash
uv run uvicorn app.main:app --reload --port 8000
```
Interactive Swagger documentation is available at `http://localhost:8000/docs`.

---

## Testing & Verification

Automated testing is executed using `pytest`. Automated test suites mock external network calls to ensure fast, deterministic, and isolated execution without consuming API credits.

```bash
# Run complete test suite (99 tests)
uv run pytest -v
```

### Test Coverage Summary:
* `tests/test_api_integration.py`: End-to-end `/score` pipeline, happy path, weight overrides, caching, and error paths.
* `tests/test_extraction.py`: JD criteria extraction, schema validation, prompt injection defense, and retry sequencing.
* `tests/test_scoring.py`: Batched scoring, 1-to-1 criteria validation, rubric enforcement, and deterministic math.
* `tests/test_hardening.py`: 5MB limits, magic byte validation, PII safety, output sanity checks, and metrics recording.
* `tests/test_metrics.py`: Metrics parsing, derived summaries calculation, missing/null value handling, and calibration metrics verification.
* `tests/test_resume_parser.py`: PDF and TXT text extraction, scanned PDF detection, corruption handling.
* `tests/test_models.py`: Pydantic model boundary constraints (0–100 score bounds, non-negative weights).
* `tests/test_health.py`: Health and root endpoints.

---

## Calibration & Metrics Results

In Phase 10, a controlled live-LLM calibration experiment was performed on OpenRouter with two genuinely near-duplicate strong resumes (Resume A and Resume B) and one deliberately weaker resume (Resume C) against the same Job Description:

* **Resume A (Baseline Strong)**: Overall Score **`98.18`**
* **Resume B (Near-Duplicate A)**: Overall Score **`85.94`**
* **Drift ($|A - B|$)**: **`12.24 points`** (Master Requirement: $\le 40.0$ points $\implies$ **PASS**)
* **Resume C (Deliberately Weaker)**: Overall Score **`0.00`** (Demonstrated clear separation margin of 98.18 points)

### Tracked Metrics Summary (6 Real Calls)
* **Total API Latency**: 117,070.94 ms (~117.07s, avg 19.51s per call)
* **Total Tokens Processed**: 11,688 tokens (4,462 prompt tokens + 7,226 completion tokens)
* **Key Finding**: Criterion-scoring calls generated **68.03%** (approx 68.0%) of all completion tokens (4,912 of 7,226 tokens), confirming that recruiter-actionable evidence reasoning is the primary driver of generation volume and latency.
* Detailed analysis is documented in [`docs/explanation.md`](docs/explanation.md).

---

## Known Limitations

1. **Scanned / Image-Only PDFs**: PyMuPDF extracts embedded digital text. Scanned resumes without an OCR layer return an explicit HTTP 422 error rather than triggering expensive or hallucinated OCR inference.
2. **Language**: Rubric anchors, criteria extraction prompts, and sanity checks are currently optimized for English language resumes and job postings.
3. **Provider Latency & Rate Limits**: Free-tier OpenRouter endpoints (`:free`) are subject to provider queueing delays and upstream rate limits (mitigated by externalized timeout configuration and fallback routing).
