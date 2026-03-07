from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta

import pandas as pd

from oil_risk.adapters.fred_adapter import FredAdapter
from oil_risk.config import settings
from oil_risk.db.io import delete_by_date, write_dataframe
from oil_risk.db.schema import init_db
from oil_risk.logging_utils import setup_logging
from oil_risk.options_flow import PolygonOptionsFlowProvider

SERIES = ["DCOILWTICO", "DCOILBRENTEU", "VIXCLS", "OVXCLS", "DTWEXBGS", "DGS10", "SP500"]
OPTION_TICKERS = ["USO", "XLE", "SPY"]

_STUB_VALUES: dict[str, float] = {
    "DCOILWTICO": 70.0,
    "DCOILBRENTEU": 74.0,
    "VIXCLS": 20.0,
    "OVXCLS": 35.0,
    "DTWEXBGS": 112.0,
    "DGS10": 4.2,
    "SP500": 5000.0,
}


def _build_stub_market_df(series_id: str, days: int = 90) -> pd.DataFrame:
    today = datetime.now(UTC).date()
    dates = [today - timedelta(days=i) for i in range(days - 1, -1, -1)]
    return pd.DataFrame(
        {
            "date": dates,
            "value": _STUB_VALUES.get(series_id, 1.0),
            "series_id": series_id,
            "source": "stub",
            "pulled_at": datetime.now(UTC),
        }
    )


def _pull_options() -> pd.DataFrame:
    if not settings.polygon_api_key:
        logging.info("POLYGON_API_KEY not set; skipping options flow pull")
        return pd.DataFrame(
            columns=["ticker", "date", "metric_name", "metric_value", "source", "pulled_at"]
        )
    provider = PolygonOptionsFlowProvider(settings.polygon_api_key)
    dt = datetime.now(UTC).date()
    frames: list[pd.DataFrame] = []
    for ticker in OPTION_TICKERS:
        try:
            frames.append(provider.fetch_daily_metrics(ticker=ticker, dt=dt))
        except Exception as exc:  # noqa: BLE001
            logging.warning("Options pull failed for %s: %s", ticker, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run() -> None:
    setup_logging()
    init_db()
    adapter = FredAdapter(settings.cache_dir)
    frames: list[pd.DataFrame] = []
    for sid in SERIES:
        try:
            frames.append(adapter.series_to_dataframe(sid))
        except Exception as exc:  # noqa: BLE001
            logging.warning("FRED fetch failed for %s; will use stub: %s", sid, exc)
    if not frames:
        logging.warning("All FRED series failed; falling back to stub market data")
        lookback = int(os.getenv("LOOKBACK_DAYS", "90"))
        frames = [_build_stub_market_df(sid, lookback) for sid in SERIES]
    merged = pd.concat(frames, ignore_index=True)
    write_dataframe(merged, "market_raw", replace=True)
    logging.info("Wrote %s market rows", len(merged))

    options_df = _pull_options()
    if not options_df.empty:
        delete_by_date("options_raw", datetime.now(UTC).date())
        write_dataframe(options_df, "options_raw", replace=False)
        logging.info("Wrote %s options rows", len(options_df))


def main() -> None:
    run()


if __name__ == "__main__":
    main()
