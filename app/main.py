from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI

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
