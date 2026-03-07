from __future__ import annotations

import json
import logging

import pandas as pd

from oil_risk.db.io import read_sql, write_dataframe
from oil_risk.logging_utils import setup_logging
from oil_risk.pipelines.data_views import load_feature_frame
from oil_risk.signals_config import load_signals_config


def _safe_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _pick_corr_feature(row: pd.Series, preference: list[str]) -> tuple[str, float | None]:
    for feature in preference:
        if feature in row.index:
            value = _safe_float(row[feature])
            if value is not None:
                return feature, value
    return "oil_spx_corr_63", None


def run() -> None:
    setup_logging()
    df = load_feature_frame()
    if df.empty:
        raise ValueError("No feature rows available for signal generation")
    work = df.copy()
    work.index = pd.to_datetime(work.index)

    config = load_signals_config()

    risk_cfg = config["risk_premium_alert"]
    macro_cfg = config["macro_stress_alert"]
    corr_cfg = config["correlation_break_alert"]
    hedge_cfg = config["hedging_pressure_alert"]
    tail_cfg = config["tail_risk_alert"]
    lagged_cfg = config["lagged_equity_pressure_alert"]

    tail = read_sql("SELECT date, tail_risk_prob, model_name FROM tail_risk_predictions")
    tail_by_date: pd.DataFrame = pd.DataFrame(columns=["tail_risk_prob", "model_name"])
    if not tail.empty:
        tail_work = tail.copy()
        tail_work["date"] = pd.to_datetime(tail_work["date"]).dt.normalize()
        tail_by_date = tail_work.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        tail_by_date = tail_by_date.set_index("date")[["tail_risk_prob", "model_name"]]

    sig_rows = []
    for dt, row in work.iterrows():
        dt_date = dt.date()
        corr_name, corr_value = _pick_corr_feature(row, corr_cfg["corr_feature_preference"])

        tail_prob = None
        tail_model_name = None
        if dt.normalize() in tail_by_date.index:
            tail_prob = _safe_float(tail_by_date.loc[dt.normalize(), "tail_risk_prob"])
            model_name = tail_by_date.loc[dt.normalize(), "model_name"]
            tail_model_name = None if pd.isna(model_name) else str(model_name)

        unusual_put_activity = _safe_float(row.get("unusual_put_activity"))
        unusual_put_triggered = unusual_put_activity is not None and unusual_put_activity > 0.0

        sig_rows.extend(
            [
                {
                    "date": dt_date,
                    "signal_name": "risk_premium_alert",
                    "signal_value": float(
                        risk_cfg["enabled"]
                        and (row.get("OVX_z_63", float("-inf")) > risk_cfg["ovx_z_min"])
                        and (
                            row.get("geopolitical_risk_score", float("-inf"))
                            > risk_cfg["news_risk_min"]
                        )
                    ),
                    "metadata_json": json.dumps(
                        {
                            "ovx_z_63": _safe_float(row.get("OVX_z_63")),
                            "news_risk_score": _safe_float(row.get("geopolitical_risk_score")),
                            "thresholds": {
                                "ovx_z_min": risk_cfg["ovx_z_min"],
                                "news_risk_min": risk_cfg["news_risk_min"],
                            },
                        }
                    ),
                },
                {
                    "date": dt_date,
                    "signal_name": "macro_stress_alert",
                    "signal_value": float(
                        macro_cfg["enabled"]
                        and (row.get("VIX_z_63", float("-inf")) > macro_cfg["vix_z_min"])
                        and (row.get("oil_return", float("inf")) < macro_cfg["oil_return_min"])
                    ),
                    "metadata_json": json.dumps(
                        {
                            "vix_z_63": _safe_float(row.get("VIX_z_63")),
                            "oil_return": _safe_float(row.get("oil_return")),
                            "thresholds": {
                                "vix_z_min": macro_cfg["vix_z_min"],
                                "oil_return_min": macro_cfg["oil_return_min"],
                            },
                        }
                    ),
                },
                {
                    "date": dt_date,
                    "signal_name": "correlation_break_alert",
                    "signal_value": float(
                        corr_cfg["enabled"]
                        and corr_value is not None
                        and corr_value < corr_cfg["corr_min"]
                    ),
                    "metadata_json": json.dumps(
                        {
                            "corr_feature": corr_name,
                            "corr_feature_used": corr_name,
                            "corr_value": corr_value,
                            "thresholds": {
                                "corr_min": corr_cfg["corr_min"],
                                "corr_feature_preference": corr_cfg["corr_feature_preference"],
                            },
                        }
                    ),
                },
                {
                    "date": dt_date,
                    "signal_name": "hedging_pressure_alert",
                    "signal_value": float(
                        hedge_cfg["enabled"]
                        and (row.get("VIX_z_63", float("-inf")) > hedge_cfg["vix_z_min"])
                        and ((not hedge_cfg["unusual_put_required"]) or unusual_put_triggered)
                    ),
                    "metadata_json": json.dumps(
                        {
                            "unusual_put_activity": unusual_put_activity,
                            "vix_z_63": _safe_float(row.get("VIX_z_63")),
                            "put_call_ratio_mean": _safe_float(row.get("put_call_ratio_mean")),
                            "thresholds": {
                                "unusual_put_required": hedge_cfg["unusual_put_required"],
                                "vix_z_min": hedge_cfg["vix_z_min"],
                            },
                        }
                    ),
                },
                {
                    "date": dt_date,
                    "signal_name": "lagged_equity_pressure_alert",
                    "signal_value": float(
                        lagged_cfg["enabled"]
                        and row.get("lagged_risk_pressure", float("-inf"))
                        > lagged_cfg["lagged_risk_pressure_min"]
                    ),
                    "metadata_json": json.dumps(
                        {
                            "lagged_risk_pressure": _safe_float(row.get("lagged_risk_pressure")),
                            "components": {
                                "dVIX_lag1": _safe_float(row.get("dVIX_lag1")),
                                "news_risk_score_lag1": _safe_float(
                                    row.get("news_risk_score_lag1")
                                ),
                                "spx_return_lag1": _safe_float(row.get("spx_return_lag1")),
                            },
                            "thresholds": {
                                "lagged_risk_pressure_min": lagged_cfg["lagged_risk_pressure_min"],
                            },
                        }
                    ),
                },
                {
                    "date": dt_date,
                    "signal_name": "tail_risk_alert",
                    "signal_value": float(
                        tail_cfg["enabled"]
                        and tail_prob is not None
                        and tail_prob >= tail_cfg["tail_risk_prob_min"]
                    ),
                    "metadata_json": json.dumps(
                        {
                            "tail_risk_prob": tail_prob,
                            "model_name": tail_model_name,
                            "thresholds": {"tail_risk_prob_min": tail_cfg["tail_risk_prob_min"]},
                        }
                    ),
                },
            ]
        )

    out = pd.DataFrame(sig_rows)
    write_dataframe(out, "signals", replace=True)
    logging.info("Signals generated for %s dates", work.index.nunique())


def main() -> None:
    run()


if __name__ == "__main__":
    main()
