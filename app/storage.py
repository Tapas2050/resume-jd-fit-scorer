import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_RUNS_PATH = Path(__file__).resolve().parent.parent / "runs.jsonl"


def compute_sha256(content: str) -> str:
    """Compute SHA-256 hex digest of given content for privacy-preserving storage."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def persist_run(
    run_id: str,
    jd_text: str,
    resume_text: str,
    overall_score: float,
    model_used: str,
    storage_path: Path | None = None,
) -> dict[str, Any]:
    """
    Append run record to local JSONL store (Master §4.7).
    Guarantees PII safety by storing only cryptographic content hashes and score metadata (Master §5.6).
    Never writes raw resume text to disk.
    """
    target_path = storage_path or DEFAULT_RUNS_PATH

    record = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "jd_hash": compute_sha256(jd_text),
        "resume_hash": compute_sha256(resume_text),
        "overall_score": overall_score,
        "model_used": model_used,
    }

    with open(target_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record
