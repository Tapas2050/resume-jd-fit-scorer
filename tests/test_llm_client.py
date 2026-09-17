import json
from unittest.mock import AsyncMock, patch
import httpx
import pytest

from app.llm_client import (
    LLMCallError,
    call_openrouter,
    get_last_call_metrics,
    sanitize_secret,
)


@pytest.fixture(autouse=True)
def mock_openrouter_env(monkeypatch):
    """Ensure consistent test environment variables without real credentials."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret-key-12345")
    monkeypatch.setenv("LLM_MODEL", "meta-llama/llama-3.1-70b-instruct")
    monkeypatch.setenv("FALLBACK_MODEL", "mistralai/mistral-7b-instruct")
    monkeypatch.setenv("LLM_TEMPERATURE", "0")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "15")


def _build_mock_response(
    status_code: int = 200,
    content_obj: dict | None = None,
    raw_text: str | None = None,
    usage: dict | None = None,
) -> httpx.Response:
    """Helper to generate mock httpx.Response."""
    if raw_text is not None:
        body = raw_text
    else:
        payload = {
            "id": "gen-test-123",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(content_obj or {}),
                    }
                }
            ],
            "usage": usage or {
                "prompt_tokens": 50,
                "completion_tokens": 25,
                "total_tokens": 75,
                "cost": 0.00015,
            },
        }
        body = json.dumps(payload)

    return httpx.Response(
        status_code=status_code,
        text=body,
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )


@pytest.mark.anyio
async def test_successful_response():
    """Verify call_openrouter returns parsed JSON and records per-call instrumentation."""
    expected_data = {"criteria": [{"name": "Python", "score": 85}]}
    mock_resp = _build_mock_response(
        content_obj=expected_data,
        usage={"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60, "cost": 0.00012},
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        result = await call_openrouter(
            prompt="Extract criteria",
            schema_hint='{"required": ["criteria"]}',
            model="meta-llama/llama-3.1-70b-instruct",
            temperature=0.0,
        )

        assert result == expected_data
        metrics = get_last_call_metrics()
        assert metrics is not None
        assert metrics.elapsed_ms >= 0
        assert metrics.prompt_tokens == 40
        assert metrics.completion_tokens == 20
        assert metrics.total_tokens == 60
        assert metrics.cost == 0.00012


@pytest.mark.anyio
async def test_correct_model_and_temperature_sent():
    """Verify passed model, temperature, and prompt are properly serialized in the payload."""
    mock_resp = _build_mock_response(content_obj={"status": "ok"})

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        await call_openrouter(
            prompt="Score candidate resume",
            schema_hint="",
            model="mistralai/mistral-7b-instruct",
            temperature=0.5,
        )

        call_kwargs = mock_post.call_args.kwargs
        sent_json = call_kwargs["json"]
        assert sent_json["model"] == "mistralai/mistral-7b-instruct"
        assert sent_json["temperature"] == 0.5
        assert sent_json["messages"] == [{"role": "user", "content": "Score candidate resume"}]


@pytest.mark.anyio
async def test_api_key_from_env_and_in_header():
    """Verify OPENROUTER_API_KEY from environment is passed in Authorization header."""
    mock_resp = _build_mock_response(content_obj={"ok": True})

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        await call_openrouter(
            prompt="Test prompt",
            schema_hint="",
            model="meta-llama/llama-3.1-70b-instruct",
        )

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer test-secret-key-12345"


@pytest.mark.anyio
async def test_missing_api_key_raises_error(monkeypatch):
    """Verify missing OPENROUTER_API_KEY raises LLMCallError without network request."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    with pytest.raises(LLMCallError) as exc_info:
        await call_openrouter(
            prompt="Prompt",
            schema_hint="",
            model="model-1",
        )
    assert "OPENROUTER_API_KEY is not set" in str(exc_info.value)


@pytest.mark.anyio
async def test_timeout_is_configured(monkeypatch):
    """Verify configured timeout in environment is applied to httpx.AsyncClient."""
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    mock_resp = _build_mock_response(content_obj={"ok": True})

    captured_timeouts = []
    original_init = httpx.AsyncClient.__init__

    def spy_init(self, *args, **kwargs):
        captured_timeouts.append(kwargs.get("timeout"))
        original_init(self, *args, **kwargs)

    with patch.object(httpx.AsyncClient, "__init__", spy_init):
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            await call_openrouter(
                prompt="Prompt",
                schema_hint="",
                model="model-1",
            )

    assert len(captured_timeouts) == 1
    timeout = captured_timeouts[0]
    assert timeout is not None
    assert timeout.connect == 30.0
    assert timeout.read == 30.0


@pytest.mark.anyio
async def test_network_failure_raises_llm_call_error():
    """Verify network/connection errors are wrapped in LLMCallError."""
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.ConnectError("Failed to connect to host")

        with pytest.raises(LLMCallError) as exc_info:
            await call_openrouter(
                prompt="Prompt",
                schema_hint="",
                model="model-1",
            )
        assert "network request failed" in str(exc_info.value)


@pytest.mark.anyio
async def test_timeout_raises_llm_call_error():
    """Verify httpx.TimeoutException is wrapped in LLMCallError."""
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.ReadTimeout("Read operation timed out")

        with pytest.raises(LLMCallError) as exc_info:
            await call_openrouter(
                prompt="Prompt",
                schema_hint="",
                model="model-1",
            )
        assert "timed out after" in str(exc_info.value)


@pytest.mark.anyio
async def test_non_2xx_response_raises_llm_call_error():
    """Verify non-2xx status code raises typed LLMCallError with status code."""
    mock_resp = _build_mock_response(
        status_code=429,
        raw_text='{"error": {"message": "Rate limit exceeded"}}',
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        with pytest.raises(LLMCallError) as exc_info:
            await call_openrouter(
                prompt="Prompt",
                schema_hint="",
                model="model-1",
            )
        assert exc_info.value.status_code == 429
        assert "non-200 status 429" in str(exc_info.value)


@pytest.mark.anyio
async def test_malformed_provider_json_raises_llm_call_error():
    """Verify invalid JSON in provider response raises LLMCallError."""
    mock_resp = _build_mock_response(
        status_code=200,
        raw_text="<html>Bad Gateway</html>",
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        with pytest.raises(LLMCallError) as exc_info:
            await call_openrouter(
                prompt="Prompt",
                schema_hint="",
                model="model-1",
            )
        assert "Malformed response JSON" in str(exc_info.value)


@pytest.mark.anyio
async def test_missing_expected_response_content_raises_llm_call_error():
    """Verify responses missing choices or message content raise LLMCallError."""
    # Empty choices
    resp_empty_choices = httpx.Response(
        status_code=200,
        text=json.dumps({"choices": []}),
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = resp_empty_choices
        with pytest.raises(LLMCallError) as exc_info:
            await call_openrouter(prompt="Prompt", schema_hint="", model="model-1")
        assert "Missing or empty 'choices'" in str(exc_info.value)


@pytest.mark.anyio
async def test_invalid_structured_json_raises_llm_call_error():
    """Verify invalid JSON text returned inside model message content raises LLMCallError."""
    # Model returns raw text rather than JSON
    invalid_json_resp = httpx.Response(
        status_code=200,
        text=json.dumps({"choices": [{"message": {"content": "This is raw text, not JSON"}}]}),
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = invalid_json_resp
        with pytest.raises(LLMCallError) as exc_info:
            await call_openrouter(prompt="Prompt", schema_hint="", model="model-1")
        assert "not valid JSON" in str(exc_info.value)


@pytest.mark.anyio
async def test_schema_missing_required_fields_raises_llm_call_error():
    """Verify model output failing schema requirement raises LLMCallError."""
    incomplete_data = {"other_field": 123}
    mock_resp = _build_mock_response(content_obj=incomplete_data)

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        with pytest.raises(LLMCallError) as exc_info:
            await call_openrouter(
                prompt="Prompt",
                schema_hint='{"required": ["criteria"]}',
                model="model-1",
            )
        assert "Missing required fields" in str(exc_info.value)


@pytest.mark.anyio
async def test_no_retry_occurs_inside_call_openrouter_after_failure():
    """Verify call_openrouter executes exactly ONE attempt and does not perform retry on failure."""
    mock_resp = _build_mock_response(
        status_code=500,
        raw_text='{"error": "Internal server error"}',
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        with pytest.raises(LLMCallError):
            await call_openrouter(
                prompt="Prompt",
                schema_hint="",
                model="model-1",
            )

        # Must be called exactly ONCE: single attempt per call_openrouter contract
        assert mock_post.call_count == 1


def test_secret_sanitization():
    """Verify secrets and bearer tokens are properly redacted."""
    text_with_key = "Failed request with test-secret-key-12345 and Bearer sk-or-v1-abcdef123456"
    sanitized = sanitize_secret(text_with_key)
    assert "test-secret-key-12345" not in sanitized
    assert "sk-or-v1-abcdef123456" not in sanitized
    assert "[REDACTED]" in sanitized
