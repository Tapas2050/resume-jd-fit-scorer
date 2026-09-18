import json
from pathlib import Path
import pytest

from app.metrics import (
    compute_metrics_summary,
    load_metrics_records,
    summarize_metrics_file,
)


def test_load_metrics_records_nonexistent_file(tmp_path):
    """Verify loading from a missing file returns an empty list without error."""
    missing_file = tmp_path / "does_not_exist.jsonl"
    assert load_metrics_records(missing_file) == []


def test_load_metrics_records_skips_blank_and_corrupt_lines(tmp_path):
    """Verify load_metrics_records skips empty lines and malformed JSON entries."""
    sample_file = tmp_path / "mixed.jsonl"
    content = (
        "\n"
        '{"model": "test-model", "elapsed_ms": 120.5, "prompt_tokens": 50}\n'
        "   \n"
        "THIS_IS_NOT_JSON\n"
        '{"model": "test-model-2", "elapsed_ms": 250.0, "prompt_tokens": 75}\n'
        "\n"
    )
    sample_file.write_text(content, encoding="utf-8")

    records = load_metrics_records(sample_file)
    assert len(records) == 2
    assert records[0]["model"] == "test-model"
    assert records[1]["model"] == "test-model-2"


def test_compute_metrics_summary_empty():
    """Verify summary of an empty record list produces clean zeros/None without error."""
    summary = compute_metrics_summary([])
    assert summary["total_calls"] == 0
    assert summary["models"] == {}
    assert summary["latency_ms"]["avg"] is None
    assert summary["latency_ms"]["min"] is None
    assert summary["latency_ms"]["max"] is None
    assert summary["prompt_tokens"]["total"] is None
    assert summary["completion_tokens"]["total"] is None
    assert summary["total_tokens"]["total"] is None
    assert summary["cost"]["total"] is None
    assert summary["cost"]["has_reported_cost"] is False
    assert summary["fallback_calls_count"] == 0


def test_compute_metrics_summary_with_complete_data():
    """Verify accurate calculation of derived summaries across realistic synthetic records."""
    records = [
        {
            "model": "primary-model",
            "elapsed_ms": 1000.0,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cost": 0.0001,
        },
        {
            "model": "primary-model",
            "elapsed_ms": 2000.0,
            "prompt_tokens": 200,
            "completion_tokens": 150,
            "total_tokens": 350,
            "cost": 0.0003,
        },
        {
            "model": "fallback-model",
            "elapsed_ms": 3000.0,
            "prompt_tokens": 300,
            "completion_tokens": 100,
            "total_tokens": 400,
            "cost": 0.0002,
        },
    ]

    summary = compute_metrics_summary(records, primary_model="primary-model")

    assert summary["total_calls"] == 3
    assert summary["models"] == {"primary-model": 2, "fallback-model": 1}
    assert summary["fallback_calls_count"] == 1

    # Latency: [1000, 2000, 3000] -> total: 6000, avg: 2000, min: 1000, max: 3000
    assert summary["latency_ms"]["total"] == 6000.0
    assert summary["latency_ms"]["avg"] == 2000.0
    assert summary["latency_ms"]["min"] == 1000.0
    assert summary["latency_ms"]["max"] == 3000.0

    # Prompt tokens: 100 + 200 + 300 = 600, avg: 200
    assert summary["prompt_tokens"]["total"] == 600
    assert summary["prompt_tokens"]["avg"] == 200.0
    assert summary["prompt_tokens"]["available_count"] == 3

    # Completion tokens: 50 + 150 + 100 = 300, avg: 100
    assert summary["completion_tokens"]["total"] == 300
    assert summary["completion_tokens"]["avg"] == 100.0

    # Total tokens: 150 + 350 + 400 = 900, avg: 300
    assert summary["total_tokens"]["total"] == 900
    assert summary["total_tokens"]["avg"] == 300.0

    # Cost: 0.0001 + 0.0003 + 0.0002 = 0.0006
    assert summary["cost"]["total"] == 0.0006
    assert summary["cost"]["has_reported_cost"] is True
    assert summary["cost"]["available_count"] == 3


def test_compute_metrics_summary_with_null_and_missing_values():
    """Verify omitted or null metrics remain None and are never fabricated."""
    records = [
        {
            "model": "free-model",
            "elapsed_ms": 1500.0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cost": None,
        },
        {
            "model": "free-model",
            "elapsed_ms": 2500.0,
            # Keys completely omitted
        },
    ]

    summary = compute_metrics_summary(records)
    assert summary["total_calls"] == 2
    assert summary["latency_ms"]["avg"] == 2000.0
    assert summary["prompt_tokens"]["total"] is None
    assert summary["prompt_tokens"]["avg"] is None
    assert summary["prompt_tokens"]["available_count"] == 0
    assert summary["cost"]["total"] is None
    assert summary["cost"]["has_reported_cost"] is False
    assert summary["cost"]["available_count"] == 0


def test_summarize_metrics_file_convenience(tmp_path):
    """Verify summarize_metrics_file loads and calculates summaries from a file path."""
    metrics_file = tmp_path / "test_summary.jsonl"
    record = {
        "timestamp": "2026-09-18T10:00:00Z",
        "model": "test-model",
        "elapsed_ms": 500.0,
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "cost": 0.00005,
    }
    metrics_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    summary = summarize_metrics_file(metrics_file, primary_model="test-model")
    assert summary["total_calls"] == 1
    assert summary["models"]["test-model"] == 1
    assert summary["fallback_calls_count"] == 0
    assert summary["latency_ms"]["avg"] == 500.0
    assert summary["total_tokens"]["total"] == 30
    assert summary["cost"]["total"] == 0.00005


def test_calibration_metrics_factual_verification():
    """
    Verify the actual Phase 10 calibration metrics file if present:
    - Exactly 6 calls recorded, 0 fallback calls
    - 4,462 total prompt tokens (avg 743.67)
    - 7,226 total completion tokens (avg 1,204.33)
    - 11,688 total tokens (avg 1,948.00)
    - Latency: total 117,070.94 ms, avg 19,511.82 ms, min 8,694.37 ms, max 34,128.71 ms
    - Cost is null (free tier)
    """
    calib_file = Path(__file__).resolve().parent.parent / "calibration_data" / "calibration_metrics.jsonl"
    if not calib_file.is_file():
        pytest.skip("calibration_metrics.jsonl not found in calibration_data/")

    summary = summarize_metrics_file(calib_file, primary_model="nvidia/nemotron-3-ultra-550b-a55b:free")
    assert summary["total_calls"] == 6
    assert summary["fallback_calls_count"] == 0
    assert summary["prompt_tokens"]["total"] == 4462
    assert summary["prompt_tokens"]["avg"] == 743.67
    assert summary["completion_tokens"]["total"] == 7226
    assert summary["completion_tokens"]["avg"] == 1204.33
    assert summary["total_tokens"]["total"] == 11688
    assert summary["total_tokens"]["avg"] == 1948.00
    assert summary["latency_ms"]["total"] == 117070.94
    assert summary["latency_ms"]["avg"] == 19511.82
    assert summary["latency_ms"]["min"] == 8694.37
    assert summary["latency_ms"]["max"] == 34128.71
    assert summary["cost"]["total"] is None
    assert summary["cost"]["has_reported_cost"] is False


def test_calibration_metrics_scoring_completion_share():
    """
    Verify factual token breakdown from Phase 10 calibration:
    - Extraction completion tokens: 741 + 929 + 644 = 2,314 (avg 771.33)
    - Scoring completion tokens: 1,521 + 2,280 + 1,111 = 4,912 (avg 1,637.33)
    - Total completion tokens: 7,226
    - Scoring share: 4,912 / 7,226 * 100 = 68.03% (approx 68.0%)
    """
    calib_file = Path(__file__).resolve().parent.parent / "calibration_data" / "calibration_metrics.jsonl"
    if not calib_file.is_file():
        pytest.skip("calibration_metrics.jsonl not found in calibration_data/")

    records = load_metrics_records(calib_file)
    assert len(records) == 6

    # Extraction calls: records 0, 2, 4
    ext_tokens = sum(records[i]["completion_tokens"] for i in [0, 2, 4])
    # Scoring calls: records 1, 3, 5
    score_tokens = sum(records[i]["completion_tokens"] for i in [1, 3, 5])
    total_tokens = ext_tokens + score_tokens

    assert ext_tokens == 2314
    assert score_tokens == 4912
    assert total_tokens == 7226

    assert round(ext_tokens / 3, 2) == 771.33
    assert round(score_tokens / 3, 2) == 1637.33

    scoring_share_1dec = round((score_tokens / total_tokens) * 100, 1)
    scoring_share_2dec = round((score_tokens / total_tokens) * 100, 2)
    assert scoring_share_1dec == 68.0
    assert scoring_share_2dec == 67.98

