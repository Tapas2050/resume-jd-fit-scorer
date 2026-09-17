# Resume ↔ Job-Description Fit Scorer

Deterministic, recruiter-actionable fit evaluation between a Job Description (JD) and a Candidate Resume.

## Status

**Phase 1: Project Scaffold** (Complete)
- Basic FastAPI application structure
- Dependency management configured via `uv`
- Environment template (`.env.example`) and Git protection (`.gitignore`)
- Health check and root endpoints with automated tests

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (recommended) or standard Python `pip`

## Quick Start

### 1. Clone & Setup

```bash
# Install dependencies with uv
uv sync
```

Or using standard `pip`:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the example environment file:
```bash
cp .env.example .env
```
Note: `.env` is ignored by Git to prevent leaking credentials.

### 3. Run the Development Server

```bash
uv run uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.
- Health check: `http://localhost:8000/health`
- Interactive API docs: `http://localhost:8000/docs`

### 4. Run Tests

```bash
uv run pytest -v
```

## Known Limitations (Phase 1)

- Scoring and LLM criteria extraction endpoints (`POST /score`) are planned for later phases and not yet implemented.
- Resume parsing and storage persistence will be added in subsequent phases.
