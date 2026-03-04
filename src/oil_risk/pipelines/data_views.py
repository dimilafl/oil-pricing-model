from __future__ import annotations

import pandas as pd

from oil_risk.db.io import read_sql


def load_feature_frame() -> pd.DataFrame:
    mkt = read_sql("SELECT date, feature_name, feature_value FROM market_features")
    nws = read_sql("SELECT date, feature_name, feature_value FROM news_features")
    opt = read_sql("SELECT date, feature_name, feature_value FROM options_features")

    mkt["date"] = pd.to_datetime(mkt["date"])
    market_wide = mkt.pivot_table(index="date", columns="feature_name", values="feature_value")

    frame = market_wide
    if not nws.empty:
        nws["date"] = pd.to_datetime(nws["date"])
        news_wide = nws.pivot_table(index="date", columns="feature_name", values="feature_value")
        frame = frame.join(news_wide, how="left")
    if not opt.empty:
        opt["date"] = pd.to_datetime(opt["date"])
        opt_wide = opt.pivot_table(index="date", columns="feature_name", values="feature_value")
        frame = frame.join(opt_wide, how="left")
    return frame.sort_index()
