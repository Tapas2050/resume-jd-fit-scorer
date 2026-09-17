from pathlib import Path
from typing import Any
import yaml

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "config" / "weights.yaml"


def load_scoring_weights(config_path: Path | None = None) -> tuple[dict[str, float], float]:
    """
    Load scoring weights from external config file (Master §2.5, F5).
    Returns (weights_mapping, fallback_weight).
    """
    target_path = config_path or DEFAULT_WEIGHTS_PATH
    if not target_path.is_file():
        return {}, 1.0

    with open(target_path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    raw_weights = data.get("criteria_weights", {})
    weights_map = {str(k).lower(): float(v) for k, v in raw_weights.items()}
    fallback_weight = float(data.get("fallback_weight", 1.0))

    return weights_map, fallback_weight


def resolve_criterion_weight(
    criterion_name: str,
    weight_hint: float | None,
    weights_map: dict[str, float],
    fallback_weight: float,
) -> float:
    """
    Resolve weight for a given criterion (Master §2.2 Step 4):
    1. Match in external config weights_map.
    2. Fallback to weight_hint from extraction if not in config.
    3. Fallback to fallback_weight if neither is available.
    """
    norm_name = criterion_name.lower().replace(" ", "_").replace("-", "_")
    for key, weight in weights_map.items():
        if key in norm_name or norm_name in key:
            return weight

    if weight_hint is not None and weight_hint > 0:
        return float(weight_hint)

    return fallback_weight
