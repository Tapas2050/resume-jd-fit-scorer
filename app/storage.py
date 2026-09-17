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
    criteria: list[dict[str, Any]] | None = None,
    storage_path: Path | None = None,
) -> dict[str, Any]:
    """
    Append run record to local JSONL store (Master §4.7).
    Guarantees PII safety by storing only cryptographic content hashes, score outputs, and metadata (Master §5.6).
    Never writes raw resume text to disk.
    """
    target_path = storage_path or DEFAULT_RUNS_PATH

    record: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "jd_hash": compute_sha256(jd_text),
        "resume_hash": compute_sha256(resume_text),
        "overall_score": overall_score,
        "model_used": model_used,
    }
    if criteria is not None:
        record["criteria"] = criteria

    with open(target_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record


def find_cached_run(
    jd_text: str,
    resume_text: str,
    storage_path: Path | None = None,
) -> dict[str, Any] | None:
    """
    Idempotency cache lookup (Master §5.5).
    Search runs.jsonl for an identical (jd_hash, resume_hash) pair.
    Returns the cached record if found and valid, otherwise None.
    """
    target_path = storage_path or DEFAULT_RUNS_PATH
    if not target_path.is_file():
        return None

    target_jd_hash = compute_sha256(jd_text)
    target_resume_hash = compute_sha256(resume_text)

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if (
                        record.get("jd_hash") == target_jd_hash
                        and record.get("resume_hash") == target_resume_hash
                        and "criteria" in record
                        and "overall_score" in record
                    ):
                        return record
                except json.JSONDecodeError:
                    continue
    except Exception:
        return None

    return None
