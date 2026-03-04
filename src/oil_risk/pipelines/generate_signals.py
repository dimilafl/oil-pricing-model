from __future__ import annotations

import logging

import pandas as pd

from oil_risk.db.io import read_sql, write_dataframe
from oil_risk.logging_utils import setup_logging


def run() -> None:
    setup_logging()
    mkt = read_sql("SELECT date, feature_name, feature_value FROM market_features")
    nws = read_sql("SELECT date, feature_name, feature_value FROM news_features")
    mkt["date"] = pd.to_datetime(mkt["date"])
    nws["date"] = pd.to_datetime(nws["date"])
    m = mkt.pivot_table(index="date", columns="feature_name", values="feature_value")
    n = nws.pivot_table(index="date", columns="feature_name", values="feature_value")
    df = m.join(n[["geopolitical_risk_score"]], how="left").rename(
        columns={"geopolitical_risk_score": "news_risk_score"}
    )
    latest = df.dropna().iloc[-1]
    dt = df.dropna().index[-1].date()

    sig_rows = [
        {
            "date": dt,
            "signal_name": "risk_premium_alert",
            "signal_value": float((latest["OVX_z_63"] > 1.0) and (latest["news_risk_score"] > 1.0)),
            "metadata_json": {
                "ovx_z_63": latest["OVX_z_63"],
                "news_risk_score": latest["news_risk_score"],
            },
        },
        {
            "date": dt,
            "signal_name": "macro_stress_alert",
            "signal_value": float((latest["VIX_z_63"] > 1.0) and (latest["oil_return"] < -0.02)),
            "metadata_json": {
                "vix_z_63": latest["VIX_z_63"],
                "oil_return": latest["oil_return"],
            },
        },
        {
            "date": dt,
            "signal_name": "correlation_break_alert",
            "signal_value": float(latest["oil_vix_corr_63_proxy"] < -0.3),
            "metadata_json": {"oil_vix_corr_63_proxy": latest["oil_vix_corr_63_proxy"]},
        },
    ]
    out = pd.DataFrame(sig_rows)
    write_dataframe(out, "signals", replace=True)
    logging.info("Signals generated for %s", dt)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
