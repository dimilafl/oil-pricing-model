from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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


settings = Settings()
