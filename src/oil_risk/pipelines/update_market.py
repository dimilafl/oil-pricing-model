from __future__ import annotations

import logging
from datetime import UTC, datetime

import pandas as pd

from oil_risk.adapters.fred_adapter import FredAdapter
from oil_risk.config import settings
from oil_risk.db.io import write_dataframe
from oil_risk.db.schema import init_db
from oil_risk.logging_utils import setup_logging
from oil_risk.options_flow import PolygonOptionsFlowProvider

SERIES = ["DCOILWTICO", "DCOILBRENTEU", "VIXCLS", "OVXCLS", "DTWEXBGS", "DGS10"]
OPTION_TICKERS = ["USO", "XLE", "SPY"]


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
        frames.append(adapter.series_to_dataframe(sid))
    merged = pd.concat(frames, ignore_index=True)
    write_dataframe(merged, "market_raw", replace=True)
    logging.info("Wrote %s market rows", len(merged))

    options_df = _pull_options()
    if not options_df.empty:
        write_dataframe(options_df, "options_raw", replace=True)
        logging.info("Wrote %s options rows", len(options_df))


def main() -> None:
    run()


if __name__ == "__main__":
    main()
