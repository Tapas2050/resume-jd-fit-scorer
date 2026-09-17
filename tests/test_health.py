from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Verify GET /health returns HTTP 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": "resume-jd-fit-scorer",
    }


def test_root_endpoint():
    """Verify GET / returns HTTP 200 and running message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Resume JD Fit Scorer API is running"
    }


def test_nonexistent_endpoint():
    """Verify nonexistent endpoint returns HTTP 404."""
    response = client.get("/nonexistent-endpoint")
    assert response.status_code == 404
