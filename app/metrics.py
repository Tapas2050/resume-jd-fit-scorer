import json
from pathlib import Path
from typing import Any

from app.llm_client import DEFAULT_METRICS_PATH


def load_metrics_records(metrics_path: Path | None = None) -> list[dict[str, Any]]:
    """
    Read metrics.jsonl line-by-line and return a list of parsed record dicts.
    Skips empty lines and malformed JSON entries gracefully.
    """
    target = metrics_path or DEFAULT_METRICS_PATH
    if not target.is_file():
        return []

    records: list[dict[str, Any]] = []
    with open(target, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
                if isinstance(data, dict):
                    records.append(data)
            except json.JSONDecodeError:
                continue

    return records


def compute_metrics_summary(
    records: list[dict[str, Any]],
    primary_model: str | None = None,
) -> dict[str, Any]:
    """
    Compute derived factual summaries across OpenRouter metrics records.
    Never invents missing values; preserves None where metrics are absent.
    """
    total_calls = len(records)
    if total_calls == 0:
        return {
            "total_calls": 0,
            "models": {},
            "latency_ms": {"total": 0.0, "avg": None, "min": None, "max": None},
            "prompt_tokens": {"total": None, "avg": None, "available_count": 0},
            "completion_tokens": {"total": None, "avg": None, "available_count": 0},
            "total_tokens": {"total": None, "avg": None, "available_count": 0},
            "cost": {"total": None, "available_count": 0, "has_reported_cost": False},
            "fallback_calls_count": 0,
        }

    models_count: dict[str, int] = {}
    latencies: list[float] = []
    prompt_tokens_list: list[int] = []
    completion_tokens_list: list[int] = []
    total_tokens_list: list[int] = []
    costs_list: list[float] = []

    for r in records:
        model = r.get("model", "unknown")
        models_count[model] = models_count.get(model, 0) + 1

        elapsed = r.get("elapsed_ms")
        if isinstance(elapsed, (int, float)):
            latencies.append(float(elapsed))

        pt = r.get("prompt_tokens")
        if isinstance(pt, (int, float)):
            prompt_tokens_list.append(int(pt))

        ct = r.get("completion_tokens")
        if isinstance(ct, (int, float)):
            completion_tokens_list.append(int(ct))

        tt = r.get("total_tokens")
        if isinstance(tt, (int, float)):
            total_tokens_list.append(int(tt))

        cost = r.get("cost")
        if isinstance(cost, (int, float)):
            costs_list.append(float(cost))

    # Latency summaries
    latency_summary = {
        "total": round(sum(latencies), 2) if latencies else 0.0,
        "avg": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "min": round(min(latencies), 2) if latencies else None,
        "max": round(max(latencies), 2) if latencies else None,
    }

    # Prompt token summaries
    prompt_summary = {
        "total": sum(prompt_tokens_list) if prompt_tokens_list else None,
        "avg": round(sum(prompt_tokens_list) / len(prompt_tokens_list), 2) if prompt_tokens_list else None,
        "available_count": len(prompt_tokens_list),
    }

    # Completion token summaries
    completion_summary = {
        "total": sum(completion_tokens_list) if completion_tokens_list else None,
        "avg": round(sum(completion_tokens_list) / len(completion_tokens_list), 2) if completion_tokens_list else None,
        "available_count": len(completion_tokens_list),
    }

    # Total token summaries
    total_token_summary = {
        "total": sum(total_tokens_list) if total_tokens_list else None,
        "avg": round(sum(total_tokens_list) / len(total_tokens_list), 2) if total_tokens_list else None,
        "available_count": len(total_tokens_list),
    }

    # Cost summaries
    cost_summary = {
        "total": round(sum(costs_list), 6) if costs_list else None,
        "available_count": len(costs_list),
        "has_reported_cost": len(costs_list) > 0,
    }

    # Fallback counts: if primary_model is supplied, count records with different model
    fallback_count = 0
    if primary_model:
        fallback_count = sum(1 for r in records if r.get("model") != primary_model)

    return {
        "total_calls": total_calls,
        "models": models_count,
        "latency_ms": latency_summary,
        "prompt_tokens": prompt_summary,
        "completion_tokens": completion_summary,
        "total_tokens": total_token_summary,
        "cost": cost_summary,
        "fallback_calls_count": fallback_count,
    }


def summarize_metrics_file(
    metrics_path: Path | None = None,
    primary_model: str | None = None,
) -> dict[str, Any]:
    """Convenience helper to load records from file and compute metric summary."""
    records = load_metrics_records(metrics_path)
    return compute_metrics_summary(records, primary_model=primary_model)
