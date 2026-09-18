import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import httpx

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_METRICS_PATH = Path(__file__).resolve().parent.parent / "metrics.jsonl"


def get_openrouter_url() -> str:
    """
    Resolve the OpenRouter chat completions endpoint from environment or default.
    Supports:
    - OPENROUTER_API_URL / LLM_API_URL (full endpoint)
    - LLM_BASE_URL / OPENROUTER_BASE_URL (base URL, appends /chat/completions)
    - Default fallback constant: https://openrouter.ai/api/v1/chat/completions
    """
    full_url = (os.getenv("OPENROUTER_API_URL") or os.getenv("LLM_API_URL") or "").strip().strip('"\'')
    if full_url:
        return full_url

    base_url = (os.getenv("LLM_BASE_URL") or os.getenv("OPENROUTER_BASE_URL") or "").strip().strip('"\'').rstrip("/")
    if base_url:
        return f"{base_url}/chat/completions" if not base_url.endswith("/chat/completions") else base_url

    return OPENROUTER_API_URL




class LLMCallError(Exception):
    """Exception raised when an OpenRouter LLM call fails."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
    ):
        sanitized_msg = sanitize_secret(message)
        super().__init__(sanitized_msg)
        self.message = sanitized_msg
        self.status_code = status_code
        self.response_body = sanitize_secret(response_body) if response_body else None


@dataclass
class CallMetrics:
    """Lightweight per-call instrumentation."""
    elapsed_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None


_last_call_metrics: CallMetrics | None = None


def get_last_call_metrics() -> CallMetrics | None:
    """Retrieve instrumentation captured from the most recent OpenRouter call."""
    return _last_call_metrics


def _set_last_call_metrics(metrics: CallMetrics | None) -> None:
    global _last_call_metrics
    _last_call_metrics = metrics


def persist_call_metrics(
    model: str,
    elapsed_ms: float,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    cost: float | None,
    metrics_path: Path | None = None,
) -> dict[str, Any]:
    """
    Append OpenRouter call metrics to metrics.jsonl (Master §5.4).
    Logs elapsed_ms, actual tokens, and actual cost. Never logs keys, headers, or PII.
    Missing metrics are written as null rather than invented.
    """
    target_path = metrics_path or DEFAULT_METRICS_PATH
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "elapsed_ms": round(elapsed_ms, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost": cost,
    }
    with open(target_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record



def sanitize_secret(text: str) -> str:
    """Remove sensitive keys or tokens from text to prevent secret exposure."""
    if not text:
        return text

    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key and api_key in text:
        text = text.replace(api_key, "[REDACTED]")

    # Mask any Bearer tokens
    text = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer [REDACTED]", text)
    return text


def get_llm_config() -> dict[str, Any]:
    """Read LLM runtime configuration from environment variables."""
    return {
        "api_key": os.getenv("OPENROUTER_API_KEY", "").strip(),
        "primary_model": os.getenv("LLM_MODEL", "").strip(),
        "fallback_model": os.getenv("FALLBACK_MODEL", "").strip(),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0")),
        "timeout_seconds": float(os.getenv("LLM_TIMEOUT_SECONDS", "45")),
        "api_url": get_openrouter_url(),
    }


def _build_response_format(schema_hint: str) -> dict[str, Any] | None:
    """Determine response_format for structured output based on schema_hint."""
    if not schema_hint:
        return {"type": "json_object"}

    try:
        parsed_schema = json.loads(schema_hint) if isinstance(schema_hint, str) else schema_hint
        if isinstance(parsed_schema, dict) and ("properties" in parsed_schema or "type" in parsed_schema):
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": parsed_schema,
                },
            }
    except (json.JSONDecodeError, TypeError):
        pass

    return {"type": "json_object"}


def _validate_schema_hint(data: dict[str, Any], schema_hint: str) -> None:
    """Validate that structured response complies with schema requirements if provided."""
    if not schema_hint:
        return

    try:
        parsed_schema = json.loads(schema_hint) if isinstance(schema_hint, str) else schema_hint
        if isinstance(parsed_schema, dict):
            required_keys = parsed_schema.get("required")
            if isinstance(required_keys, list):
                missing = [k for k in required_keys if k not in data]
                if missing:
                    raise LLMCallError(f"Model output violates schema. Missing required fields: {missing}")
    except json.JSONDecodeError:
        pass


async def call_openrouter(
    prompt: str,
    schema_hint: str,
    model: str,
    temperature: float = 0,
) -> dict:
    """
    Execute ONE attempt against ONE model via OpenRouter's OpenAI-compatible endpoint.
    Caller handles retry / fallback sequencing (Master §2.4, §4.4).
    """
    config = get_llm_config()
    api_key = config["api_key"]
    if not api_key:
        raise LLMCallError("OPENROUTER_API_KEY is not set or empty.")

    timeout_seconds = config["timeout_seconds"]
    timeout = httpx.Timeout(timeout_seconds, connect=timeout_seconds, read=timeout_seconds)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Tapas2050/resume-jd-fit-scorer",
        "X-Title": "Resume JD Fit Scorer",
    }

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
    }

    resp_format = _build_response_format(schema_hint)
    if resp_format:
        payload["response_format"] = resp_format

    start_time = time.perf_counter()
    endpoint_url = get_openrouter_url()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                endpoint_url,
                headers=headers,
                json=payload,
            )
    except httpx.TimeoutException as exc:
        raise LLMCallError(f"OpenRouter request timed out after {timeout_seconds}s: {exc}") from exc
    except httpx.RequestError as exc:
        raise LLMCallError(f"OpenRouter network request failed: {sanitize_secret(str(exc))}") from exc

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    if response.status_code != 200:
        sanitized_body = sanitize_secret(response.text)
        raise LLMCallError(
            f"OpenRouter returned non-200 status {response.status_code}: {sanitized_body}",
            status_code=response.status_code,
            response_body=sanitized_body,
        )

    try:
        raw_json = response.json()
    except Exception as exc:
        raise LLMCallError(f"Malformed response JSON from OpenRouter provider: {exc}") from exc

    choices = raw_json.get("choices")
    if not choices or not isinstance(choices, list):
        raise LLMCallError("Missing or empty 'choices' in OpenRouter response.")

    message = choices[0].get("message")
    if not message or not isinstance(message, dict):
        raise LLMCallError("Missing 'message' in OpenRouter response choice.")

    content = message.get("content")
    if content is None:
        raise LLMCallError("Missing 'content' in OpenRouter response message.")

    try:
        parsed_data = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        raise LLMCallError(f"Model output is not valid JSON: {exc}") from exc

    if not isinstance(parsed_data, dict):
        raise LLMCallError(f"Expected model output to be a JSON object (dict), got {type(parsed_data).__name__}")

    _validate_schema_hint(parsed_data, schema_hint)

    # Lightweight per-call instrumentation (Master §5.4)
    usage = raw_json.get("usage") or {}
    cost_val = usage.get("cost") or usage.get("total_cost")
    metrics = CallMetrics(
        elapsed_ms=round(elapsed_ms, 2),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        cost=float(cost_val) if cost_val is not None else None,
    )
    _set_last_call_metrics(metrics)
    persist_call_metrics(
        model=model,
        elapsed_ms=metrics.elapsed_ms,
        prompt_tokens=metrics.prompt_tokens,
        completion_tokens=metrics.completion_tokens,
        total_tokens=metrics.total_tokens,
        cost=metrics.cost,
    )

    logger.debug(
        "OpenRouter call succeeded (model=%s, elapsed_ms=%.2f, prompt_tokens=%s, completion_tokens=%s)",
        model,
        metrics.elapsed_ms,
        metrics.prompt_tokens,
        metrics.completion_tokens,
    )

    return parsed_data
