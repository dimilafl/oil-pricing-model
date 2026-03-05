from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _model_feature_set() -> str:
    value = os.getenv("MODEL_FEATURE_SET", "base").strip().lower()
    if value not in {"base", "lagged"}:
        raise ValueError("MODEL_FEATURE_SET must be one of: base, lagged")
    return value


@dataclass(frozen=True)
class Settings:
    base_dir: Path = Path(".")
    data_dir: Path = Path("data")
    cache_dir: Path = Path("data/cache")
    reports_dir: Path = Path("reports")
    models_dir: Path = Path("models")
    db_path: Path = Path("data/oil_risk.db")
    polygon_api_key: str | None = os.getenv("POLYGON_API_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    model_feature_set: str = _model_feature_set()


settings = Settings()
