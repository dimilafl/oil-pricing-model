from __future__ import annotations

import logging
from datetime import UTC, date, datetime

import pandas as pd
import requests

from oil_risk.options_flow.base import OptionsFlowProvider

logger = logging.getLogger(__name__)
POLYGON_URL = "https://api.polygon.io/v3/snapshot/options/{ticker}"


class PolygonOptionsFlowProvider(OptionsFlowProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_daily_metrics(self, ticker: str, dt: date) -> pd.DataFrame:
        params = {"apiKey": self.api_key, "limit": 250}
        resp = requests.get(POLYGON_URL.format(ticker=ticker), params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("results", [])
        put_volume = 0.0
        call_volume = 0.0
        iv_values: list[float] = []
        for row in results:
            details = row.get("details", {})
            day = row.get("day", {})
            contract_type = details.get("contract_type")
            volume = float(day.get("volume", 0) or 0)
            if contract_type == "put":
                put_volume += volume
            elif contract_type == "call":
                call_volume += volume
            iv = row.get("implied_volatility")
            if iv is not None:
                iv_values.append(float(iv))
        put_call_ratio = put_volume / call_volume if call_volume else None

        rows = [
            {"ticker": ticker, "date": dt, "metric_name": "put_volume", "metric_value": put_volume},
            {
                "ticker": ticker,
                "date": dt,
                "metric_name": "call_volume",
                "metric_value": call_volume,
            },
            {
                "ticker": ticker,
                "date": dt,
                "metric_name": "put_call_ratio",
                "metric_value": put_call_ratio,
            },
        ]
        if iv_values:
            rows.append(
                {
                    "ticker": ticker,
                    "date": dt,
                    "metric_name": "implied_vol_proxy",
                    "metric_value": float(sum(iv_values) / len(iv_values)),
                }
            )
        frame = pd.DataFrame(rows)
        frame["source"] = "polygon"
        frame["pulled_at"] = datetime.now(UTC)
        logger.info("Fetched options metrics for %s (%s rows)", ticker, len(frame))
        return frame
