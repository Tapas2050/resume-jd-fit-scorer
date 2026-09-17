import pytest
from pydantic import ValidationError
from app.models import Criterion, CriterionScore, ScoreResponse, ErrorResponse


def test_valid_criterion_without_weight_hint():
    """Verify Criterion can be created with default weight_hint=None."""
    criterion = Criterion(name="Python Experience", description="5+ years in Python")
    assert criterion.name == "Python Experience"
    assert criterion.description == "5+ years in Python"
    assert criterion.weight_hint is None


def test_valid_criterion_with_weight_hint():
    """Verify Criterion correctly stores optional weight_hint when provided."""
    criterion = Criterion(
        name="AWS Cloud",
        description="Experience with AWS ECS and S3",
        weight_hint=2.5,
    )
    assert criterion.name == "AWS Cloud"
    assert criterion.description == "Experience with AWS ECS and S3"
    assert criterion.weight_hint == 2.5


def test_valid_criterion_score_at_boundaries():
    """Verify CriterionScore accepts valid scores including 0.0 and 100.0."""
    cs_min = CriterionScore(name="Skill A", score=0.0, weight=1.0, reasoning="Absent")
    assert cs_min.score == 0.0

    cs_max = CriterionScore(name="Skill B", score=100.0, weight=3.0, reasoning="Perfect match")
    assert cs_max.score == 100.0

    cs_mid = CriterionScore(name="Skill C", score=75.5, weight=2.0, reasoning="Strong match")
    assert cs_mid.score == 75.5


def test_criterion_score_below_zero_rejected():
    """Verify CriterionScore rejects score below 0 with ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        CriterionScore(name="Python", score=-0.1, weight=1.0, reasoning="Invalid negative")
    assert "score" in str(exc_info.value)


def test_criterion_score_above_hundred_rejected():
    """Verify CriterionScore rejects score above 100 with ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        CriterionScore(name="Python", score=100.1, weight=1.0, reasoning="Invalid over 100")
    assert "score" in str(exc_info.value)


def test_criterion_score_negative_weight_rejected():
    """Verify CriterionScore rejects negative weight."""
    with pytest.raises(ValidationError) as exc_info:
        CriterionScore(name="Python", score=80.0, weight=-1.0, reasoning="Negative weight")
    assert "weight" in str(exc_info.value)


def test_valid_score_response():
    """Verify ScoreResponse accepts valid top-level and nested criteria structures."""
    criteria = [
        CriterionScore(name="Python", score=85.0, weight=2.0, reasoning="6 years Python"),
        CriterionScore(name="AWS", score=40.0, weight=1.0, reasoning="Brief mention"),
    ]
    response = ScoreResponse(
        overall_score=70.0,
        criteria=criteria,
        model_used="meta-llama/llama-3.1-70b-instruct",
        run_id="run-1234-uuid",
    )
    assert response.overall_score == 70.0
    assert len(response.criteria) == 2
    assert response.criteria[0].name == "Python"
    assert response.model_used == "meta-llama/llama-3.1-70b-instruct"
    assert response.run_id == "run-1234-uuid"


def test_score_response_invalid_nested_criterion_score_rejected():
    """Verify ScoreResponse validation fails if any nested CriterionScore has invalid score."""
    invalid_criteria = [
        {"name": "Python", "score": 150.0, "weight": 2.0, "reasoning": "Out of bounds score"}
    ]
    with pytest.raises(ValidationError) as exc_info:
        ScoreResponse(
            overall_score=75.0,
            criteria=invalid_criteria,  # type: ignore
            model_used="test-model",
            run_id="run-abc",
        )
    assert "criteria" in str(exc_info.value)


def test_score_response_invalid_overall_score_rejected():
    """Verify ScoreResponse rejects overall_score outside 0..100 range."""
    criteria = [CriterionScore(name="Skill", score=50.0, weight=1.0, reasoning="Average")]

    with pytest.raises(ValidationError):
        ScoreResponse(
            overall_score=-1.0,
            criteria=criteria,
            model_used="test-model",
            run_id="run-abc",
        )

    with pytest.raises(ValidationError):
        ScoreResponse(
            overall_score=101.0,
            criteria=criteria,
            model_used="test-model",
            run_id="run-abc",
        )


def test_valid_error_response():
    """Verify ErrorResponse properly validates standard error responses."""
    err = ErrorResponse(
        error="unparsable_resume",
        detail="PDF text extraction returned empty content.",
    )
    assert err.error == "unparsable_resume"
    assert err.detail == "PDF text extraction returned empty content."


def test_error_response_missing_fields_rejected():
    """Verify ErrorResponse rejects payload missing required error or detail fields."""
    with pytest.raises(ValidationError):
        ErrorResponse(error="some_error")  # type: ignore

    with pytest.raises(ValidationError):
        ErrorResponse(detail="some detail")  # type: ignore
