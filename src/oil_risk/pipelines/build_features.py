from __future__ import annotations

import logging

import pandas as pd

from oil_risk.config import settings
from oil_risk.db.io import read_sql, write_dataframe
from oil_risk.features import build_market_features, build_news_features
from oil_risk.logging_utils import setup_logging


def run() -> None:
    setup_logging()
    mkt = read_sql("SELECT series_id, date, value FROM market_raw")
    mkt["date"] = pd.to_datetime(mkt["date"])
    wide = mkt.pivot_table(index="date", columns="series_id", values="value").sort_index().ffill()
    mfeatures = build_market_features(wide)
    mlong = (
        mfeatures.reset_index()
        .melt(id_vars=["date"], var_name="feature_name", value_name="feature_value")
        .dropna(subset=["feature_value"])
    )
    mlong["date"] = pd.to_datetime(mlong["date"]).dt.date
    write_dataframe(mlong, "market_features", replace=True)

    nfile = settings.cache_dir / "news_normalized.parquet"
    if nfile.exists():
        news_norm = pd.read_parquet(nfile)
        news_norm["date"] = pd.to_datetime(news_norm["date"]).dt.date
        nfeatures = build_news_features(news_norm)
        nlong = (
            nfeatures.reset_index()
            .melt(id_vars=["date"], var_name="feature_name", value_name="feature_value")
            .dropna(subset=["feature_value"])
        )
        write_dataframe(nlong, "news_features", replace=True)
        logging.info("Wrote market and news features")
    else:
        logging.warning("No normalized news parquet found, skipping news features")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
