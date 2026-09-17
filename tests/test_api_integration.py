import json
from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient

from app.llm_client import LLMCallError
from app.main import app
from app.models import Criterion, CriterionScore

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_api_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-api-key")
    monkeypatch.setenv("LLM_MODEL", "meta-llama/llama-3.1-70b-instruct")
    monkeypatch.setenv("FALLBACK_MODEL", "mistralai/mistral-7b-instruct")

    # Redirect runs.jsonl to temporary path for test isolation
    test_runs_path = tmp_path / "test_runs.jsonl"
    monkeypatch.setattr("app.main.persist_run", lambda **kwargs: _mock_persist_run(test_runs_path, **kwargs))
    return test_runs_path


def _mock_persist_run(storage_path: Path, **kwargs):
    from app.storage import persist_run
    return persist_run(storage_path=storage_path, **kwargs)


def test_score_pipeline_happy_path(mock_api_env):
    """
    Verify complete happy path:
    JD + TXT resume -> criteria extraction -> weights applied -> scoring -> weighted aggregation -> 200 ScoreResponse.
    """
    mock_criteria = [
        Criterion(name="Python required_skill", description="5+ years Python", weight_hint=1.0),
        Criterion(name="education", description="BS in CS", weight_hint=1.0),
    ]

    mock_scores = [
        CriterionScore(name="Python required_skill", score=80.0, weight=3.0, reasoning="6 years Python"),
        CriterionScore(name="education", score=100.0, weight=1.0, reasoning="BS in CS from accredited univ"),
    ]

    with patch("app.main.extract_criteria", new_callable=AsyncMock) as mock_extract, \
         patch("app.main.score_criteria", new_callable=AsyncMock) as mock_score:

        mock_extract.return_value = mock_criteria
        mock_score.return_value = mock_scores

        sample_resume = "Jane Doe\n6 years of Python experience.\nBS in Computer Science."
        files = {
            "resume_file": ("resume.txt", sample_resume.encode("utf-8"), "text/plain")
        }
        data = {
            "job_description": "Looking for Senior Python Developer with BS in Computer Science."
        }

        response = client.post("/score", data=data, files=files)

        assert response.status_code == 200
        resp_json = response.json()

        # Expected overall: (80.0*3.0 + 100.0*1.0) / (3.0 + 1.0) = 340.0 / 4.0 = 85.0
        assert resp_json["overall_score"] == 85.0
        assert len(resp_json["criteria"]) == 2
        assert resp_json["model_used"] == "meta-llama/llama-3.1-70b-instruct"
        assert "run_id" in resp_json
        assert resp_json["criteria"][0]["name"] == "Python required_skill"
        assert resp_json["criteria"][0]["score"] == 80.0

        # Verify JSONL persistence
        assert mock_api_env.is_file()
        lines = mock_api_env.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])

        assert record["run_id"] == resp_json["run_id"]
        assert record["overall_score"] == 85.0
        assert record["model_used"] == "meta-llama/llama-3.1-70b-instruct"
        assert "jd_hash" in record
        assert "resume_hash" in record

        # PII Hygiene check: Raw resume content MUST NOT appear anywhere in the record
        assert "Jane Doe" not in json.dumps(record)
        assert sample_resume not in json.dumps(record)


def test_score_pipeline_unparsable_resume_rejected():
    """Verify empty/unparsable resume returns HTTP 422 with unparsable_resume error."""
    files = {
        "resume_file": ("empty.txt", b"", "text/plain")
    }
    data = {"job_description": "Python developer"}

    response = client.post("/score", data=data, files=files)
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "unparsable_resume"


def test_score_pipeline_unsupported_file_extension_rejected():
    """Verify unsupported file types return HTTP 422 with unsupported_file_type."""
    files = {
        "resume_file": ("resume.docx", b"word doc content", "application/octet-stream")
    }
    data = {"job_description": "Python developer"}

    response = client.post("/score", data=data, files=files)
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "unsupported_file_type"


def test_score_pipeline_file_too_large_rejected():
    """Verify file exceeding 5MB returns HTTP 413."""
    huge_content = b"A" * (5 * 1024 * 1024 + 1024)
    files = {
        "resume_file": ("huge_resume.txt", huge_content, "text/plain")
    }
    data = {"job_description": "Python developer"}

    response = client.post("/score", data=data, files=files)
    assert response.status_code == 413
    body = response.json()
    assert body["error"] == "file_too_large"


def test_score_pipeline_empty_job_description_rejected():
    """Verify empty job description returns HTTP 422."""
    files = {
        "resume_file": ("resume.txt", b"Valid resume text", "text/plain")
    }
    data = {"job_description": "   "}

    response = client.post("/score", data=data, files=files)
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "unparsable_resume"


def test_score_pipeline_llm_extraction_failure_maps_to_502():
    """Verify criteria extraction failure returns HTTP 502 with llm_call_failed."""
    with patch("app.main.extract_criteria", new_callable=AsyncMock) as mock_extract:
        mock_extract.side_effect = LLMCallError("Both primary and fallback models failed")

        files = {
            "resume_file": ("resume.txt", b"Valid resume text", "text/plain")
        }
        data = {"job_description": "Valid job description"}

        response = client.post("/score", data=data, files=files)
        assert response.status_code == 502
        body = response.json()
        assert body["error"] == "llm_call_failed"


def test_score_pipeline_llm_scoring_failure_maps_to_502():
    """Verify criterion scoring failure returns HTTP 502 with llm_call_failed."""
    mock_criteria = [Criterion(name="Python", description="5+ years Python")]

    with patch("app.main.extract_criteria", new_callable=AsyncMock) as mock_extract, \
         patch("app.main.score_criteria", new_callable=AsyncMock) as mock_score:

        mock_extract.return_value = mock_criteria
        mock_score.side_effect = LLMCallError("Both models timed out during scoring")

        files = {
            "resume_file": ("resume.txt", b"Valid resume text", "text/plain")
        }
        data = {"job_description": "Valid job description"}

        response = client.post("/score", data=data, files=files)
        assert response.status_code == 502
        body = response.json()
        assert body["error"] == "llm_call_failed"


def test_weights_yaml_externalization_applied():
    """Verify external weights from config/weights.yaml are loaded and applied to criteria."""
    from app.config_loader import load_scoring_weights, resolve_criterion_weight

    weights_map, fallback_weight = load_scoring_weights()
    assert "required_skill" in weights_map
    assert weights_map["required_skill"] == 3.0
    assert weights_map["years_experience"] == 2.0
    assert fallback_weight == 1.0

    # Test resolution
    assert resolve_criterion_weight("Senior Python required_skill", None, weights_map, fallback_weight) == 3.0
    assert resolve_criterion_weight("Unknown skill", 2.5, weights_map, fallback_weight) == 2.5
    assert resolve_criterion_weight("Unknown skill", None, weights_map, fallback_weight) == 1.0
