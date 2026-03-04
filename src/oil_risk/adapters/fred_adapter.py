from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


class FredAdapter:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def series_to_dataframe(self, series_id: str, force_refresh: bool = False) -> pd.DataFrame:
        cache_file = self.cache_dir / f"fred_{series_id}.parquet"
        if cache_file.exists() and not force_refresh:
            return pd.read_parquet(cache_file)

        url = FRED_CSV.format(series_id=series_id)
        logger.info("Downloading FRED series %s", series_id)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(resp.text))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.date
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["series_id"] = series_id
        df["source"] = "fred"
        df["pulled_at"] = datetime.now(UTC)
        df = df.dropna(subset=["value"])
        df.to_parquet(cache_file, index=False)
        return df
