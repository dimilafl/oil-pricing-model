from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

DEFAULT_SIGNALS_CONFIG: dict[str, Any] = {
    "risk_premium_alert": {
        "enabled": True,
        "ovx_z_min": 1.0,
        "news_risk_min": 1.0,
    },
    "macro_stress_alert": {
        "enabled": True,
        "vix_z_min": 1.0,
        "oil_return_min": -0.02,
    },
    "correlation_break_alert": {
        "enabled": True,
        "corr_min": -0.3,
        "corr_feature_preference": ["oil_spx_corr_63", "oil_vix_corr_63_proxy"],
    },
    "hedging_pressure_alert": {
        "enabled": True,
        "unusual_put_required": True,
        "vix_z_min": 1.0,
    },
    "tail_risk_alert": {
        "enabled": True,
        "tail_risk_prob_min": 0.5,
    },
    "tuning": {
        "max_trigger_rate": 0.2,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(base_value, value)
        else:
            merged[key] = value
    return merged


def load_signals_config(path: Path | str = Path("configs/signals.json")) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return copy.deepcopy(DEFAULT_SIGNALS_CONFIG)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return copy.deepcopy(DEFAULT_SIGNALS_CONFIG)
    return _deep_merge(DEFAULT_SIGNALS_CONFIG, payload)
