from pydantic import BaseModel, Field


class Criterion(BaseModel):
    """Discrete required criterion extracted from JD."""
    name: str = Field(..., min_length=1, description="Criterion name")
    description: str = Field(..., description="Criterion description/expectation")
    weight_hint: float | None = Field(default=None, description="Optional weight hint from extraction")


class CriterionScore(BaseModel):
    """Evaluation score for a single criterion against resume."""
    name: str = Field(..., min_length=1, description="Criterion name")
    score: float = Field(..., ge=0.0, le=100.0, description="Score bounded between 0 and 100")
    weight: float = Field(..., ge=0.0, description="Criterion scoring weight")
    reasoning: str = Field(..., description="Recruiter-actionable reasoning for score")


class ScoreResponse(BaseModel):
    """Structured response for /score endpoint."""
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Weighted overall score between 0 and 100")
    criteria: list[CriterionScore] = Field(..., description="List of individual criterion scores")
    model_used: str = Field(..., min_length=1, description="Model identifier used for evaluation")
    run_id: str = Field(..., min_length=1, description="Unique run identifier")


class ErrorResponse(BaseModel):
    """Standardized error response payload."""
    error: str = Field(..., min_length=1, description="Error category identifier")
    detail: str = Field(..., description="Detailed explanation of the error")
