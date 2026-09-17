import os
from pathlib import Path
import uuid
import anyio
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from app.config_loader import load_scoring_weights, resolve_criterion_weight
from app.extraction import extract_criteria
from app.llm_client import LLMCallError, get_llm_config
from app.models import CriterionScore, ErrorResponse, ScoreResponse
from app.resume_parser import ResumeParseError, parse_resume
from app.scoring import check_suspicious_score, score_criteria, weighted_overall
from app.storage import find_cached_run, persist_run

# Load environment variables safely if .env exists
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.is_file():
    load_dotenv(dotenv_path=env_path)

app = FastAPI(
    title="Resume JD Fit Scorer API",
    description="Deterministic Resume to Job-Description Fit Scorer API",
    version="0.1.0",
)


@app.get("/")
def read_root() -> dict[str, str]:
    """Root endpoint verifying API is running."""
    return {"message": "Resume JD Fit Scorer API is running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    """Basic health check endpoint."""
    return {
        "status": "ok",
        "app": "resume-jd-fit-scorer",
    }


@app.post(
    "/score",
    response_model=ScoreResponse,
    responses={
        413: {"model": ErrorResponse, "description": "Uploaded file exceeds 5MB limit"},
        422: {"model": ErrorResponse, "description": "Unparsable resume or invalid input"},
        502: {"model": ErrorResponse, "description": "LLM call failed across primary and fallback models"},
    },
)
async def score_resume(
    job_description: str = Form(..., description="Job description text"),
    resume_file: UploadFile = File(..., description="Candidate resume file (.pdf or .txt)"),
):
    """
    End-to-end resume scoring pipeline (Master §2.2):
    1. Parse and validate resume file (PyMuPDF / TXT).
    2. LLM Call #1: Extract discrete criteria from Job Description.
    3. Load external weights from config/weights.yaml and apply to criteria.
    4. LLM Call #2: Batched scoring of criteria against resume text.
    5. Pure Python deterministic weighted overall calculation.
    6. Local PII-safe JSONL run persistence.
    7. Return structured ScoreResponse.
    """
    cleaned_jd = job_description.strip() if job_description else ""
    if not cleaned_jd:
        return JSONResponse(
            status_code=422,
            content={"error": "unparsable_resume", "detail": "Job description text cannot be empty."},
        )

    # 1. Read and parse resume file
    content = await resume_file.read()
    filename = resume_file.filename or ""

    try:
        # Offload synchronous parsing to worker thread to prevent event-loop blocking (Master §5.2)
        resume_text = await anyio.to_thread.run_sync(parse_resume, filename, content)
    except ResumeParseError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.error, "detail": exc.detail},
        )

    # 2. Check local cache / idempotency (Master §5.5)
    cached_record = await anyio.to_thread.run_sync(find_cached_run, cleaned_jd, resume_text)
    if cached_record is not None:
        return ScoreResponse(
            overall_score=cached_record["overall_score"],
            criteria=[CriterionScore(**c) for c in cached_record["criteria"]],
            model_used=cached_record.get("model_used", "cached"),
            run_id=cached_record.get("run_id", str(uuid.uuid4())),
        )

    # 3. LLM Call #1: Extract criteria from Job Description
    try:
        criteria = await extract_criteria(cleaned_jd)
    except LLMCallError as exc:
        return JSONResponse(
            status_code=502,
            content={"error": "llm_call_failed", "detail": f"Criteria extraction failed: {exc.message}"},
        )

    # 4. Load external scoring weights (Master §2.2 Step 4, §2.5, F5)
    weights_map, fallback_weight = await anyio.to_thread.run_sync(load_scoring_weights)
    for c in criteria:
        resolved = resolve_criterion_weight(c.name, c.weight_hint, weights_map, fallback_weight)
        c.weight_hint = resolved

    # 5. LLM Call #2: Batched scoring of resume vs criteria
    try:
        criterion_scores = await score_criteria(resume_text, criteria)
    except LLMCallError as exc:
        return JSONResponse(
            status_code=502,
            content={"error": "llm_call_failed", "detail": f"Resume scoring failed: {exc.message}"},
        )

    # Ensure weights on scored criteria reflect external config and output sanity check is applied
    for cs in criterion_scores:
        cs.weight = resolve_criterion_weight(cs.name, cs.weight, weights_map, fallback_weight)
        if not cs.suspicious:
            cs.suspicious = check_suspicious_score(cs.score, cs.reasoning, resume_text)

    # 6. Deterministic weighted aggregation (pure Python, Master §2.2 Step 6, §4.5)
    overall_score = round(weighted_overall(criterion_scores), 2)

    # 7. Metadata and PII-safe local persistence (Master §4.7, §5.6)
    llm_cfg = get_llm_config()
    model_used = llm_cfg.get("primary_model") or os.getenv("LLM_MODEL", "unknown")
    run_id = str(uuid.uuid4())

    await anyio.to_thread.run_sync(
        persist_run,
        run_id,
        cleaned_jd,
        resume_text,
        overall_score,
        model_used,
        [cs.model_dump() for cs in criterion_scores],
    )

    # 8. Return structured ScoreResponse
    return ScoreResponse(
        overall_score=overall_score,
        criteria=criterion_scores,
        model_used=model_used,
        run_id=run_id,
    )

