from __future__ import annotations

import logging

import pandas as pd

from oil_risk.config import settings
from oil_risk.db.io import read_sql, write_dataframe
from oil_risk.features import (
    build_market_features,
    build_news_features,
    build_options_features,
    robust_z,
)
from oil_risk.logging_utils import setup_logging


def run() -> None:
    setup_logging()
    mkt = read_sql("SELECT series_id, date, value FROM market_raw")
    mkt["date"] = pd.to_datetime(mkt["date"])
    wide = mkt.pivot_table(index="date", columns="series_id", values="value").sort_index().ffill()
    ofeatures = pd.DataFrame()
    opt_raw = read_sql("SELECT ticker, date, metric_name, metric_value FROM options_raw")
    if not opt_raw.empty:
        opt_raw["date"] = pd.to_datetime(opt_raw["date"]).dt.date
        ofeatures = build_options_features(opt_raw)
        olong = (
            ofeatures.reset_index()
            .melt(id_vars=["date"], var_name="feature_name", value_name="feature_value")
            .dropna(subset=["feature_value"])
        )
        write_dataframe(olong, "options_features", replace=True)

    mfeatures = build_market_features(wide)

    nfile = settings.cache_dir / "news_normalized.parquet"
    nfeatures = pd.DataFrame()
    if nfile.exists():
        news_norm = pd.read_parquet(nfile)
        news_norm["date"] = pd.to_datetime(news_norm["date"]).dt.date
        llm = read_sql(
            "SELECT r.id, date(r.datetime) AS date, l.category, l.intensity "
            "FROM news_raw r JOIN news_llm l ON r.id = l.id"
        )
        nfeatures = build_news_features(news_norm, llm)
        nlong = (
            nfeatures.reset_index()
            .melt(id_vars=["date"], var_name="feature_name", value_name="feature_value")
            .dropna(subset=["feature_value"])
        )
        write_dataframe(nlong, "news_features", replace=True)
    else:
        logging.warning("No normalized news parquet found, skipping news features")

    lagged_pressure = pd.Series(index=mfeatures.index, dtype=float)
    if "dVIX_lag1" in mfeatures.columns:
        lagged_pressure = lagged_pressure.add(robust_z(mfeatures["dVIX_lag1"]), fill_value=0.0)
    if not nfeatures.empty and "news_risk_score_lag1" in nfeatures.columns:
        news_lag = nfeatures["news_risk_score_lag1"]
        news_lag.index = pd.to_datetime(news_lag.index)
        lagged_pressure = lagged_pressure.add(
            robust_z(news_lag.reindex(mfeatures.index)), fill_value=0.0
        )
    if "spx_return_lag1" in mfeatures.columns:
        lagged_pressure = lagged_pressure - robust_z(mfeatures["spx_return_lag1"])
    else:
        lagged_pressure[:] = pd.NA

    mfeatures["lagged_risk_pressure"] = lagged_pressure
    mlong = (
        mfeatures.reset_index()
        .melt(id_vars=["date"], var_name="feature_name", value_name="feature_value")
        .dropna(subset=["feature_value"])
    )
    mlong["date"] = pd.to_datetime(mlong["date"]).dt.date
    write_dataframe(mlong, "market_features", replace=True)
    logging.info("Wrote market, options, and news features")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
