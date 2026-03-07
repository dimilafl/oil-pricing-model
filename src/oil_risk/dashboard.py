from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from oil_risk.db.io import read_sql

STATUS_SPECS = [
    ("market_raw", "SELECT date FROM market_raw", "date"),
    ("market_features", "SELECT date FROM market_features", "date"),
    ("news_raw", "SELECT datetime FROM news_raw", "datetime"),
    ("news_features", "SELECT date FROM news_features", "date"),
    ("news_normalized", "SELECT date FROM news_normalized", "date"),
    ("model_state", "SELECT date FROM model_state", "date"),
    ("signals", "SELECT date FROM signals", "date"),
    ("signal_eval", "SELECT date FROM signal_eval", "date"),
    ("reports", "SELECT path, created_at FROM reports", "created_at"),
    ("tail_risk_predictions", "SELECT date FROM tail_risk_predictions", "date"),
]


def _safe_read_sql(query: str, read_sql_fn=read_sql) -> pd.DataFrame:
    try:
        return read_sql_fn(query)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def build_data_status(read_sql_fn=read_sql) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict] = []
    report_file_missing = False

    for name, query, date_col in STATUS_SPECS:
        try:
            df = read_sql_fn(query)
        except Exception:  # noqa: BLE001
            rows.append({"name": name, "row_count": 0, "latest_date": "missing table"})
            if name == "reports":
                report_file_missing = True
            continue

        latest_date = None
        if not df.empty and date_col in df.columns:
            latest_date = str(pd.to_datetime(df[date_col]).max().date())

        if name == "reports" and not df.empty and "path" in df.columns:
            latest_path = Path(str(df.iloc[-1]["path"]))
            if not latest_path.exists():
                report_file_missing = True

        rows.append(
            {
                "name": name,
                "row_count": int(len(df)),
                "latest_date": latest_date,
            }
        )

    status_df = pd.DataFrame(rows)
    missing: list[str] = []
    row_counts = {row["name"]: int(row["row_count"]) for row in rows}

    if row_counts.get("news_features", 0) == 0:
        missing.append(
            "news_features is empty. Only price charts will render. A previous update_news run may have failed."
        )
    if row_counts.get("model_state", 0) == 0:
        missing.append("model_state is empty. Training did not run.")
    if row_counts.get("signals", 0) == 0:
        missing.append("signals is empty. Signal generation did not run.")
    if row_counts.get("reports", 0) == 0 or report_file_missing:
        missing.append("reports are missing or stale. Report generation did not run or report paths are stale.")

    return status_df, missing


def main() -> None:
    st.title("Oil Risk Dashboard")

    status_df, missing = build_data_status()
    st.subheader("Data status")
    st.dataframe(status_df, hide_index=True, use_container_width=True)
    st.markdown("**What is missing**")
    if missing:
        for item in missing:
            st.markdown(f"- {item}")
    else:
        st.markdown("- No critical gaps detected.")

    raw = _safe_read_sql("SELECT date, series_id, value FROM market_raw")
    if not raw.empty and {"date", "series_id", "value"}.issubset(raw.columns):
        raw["date"] = pd.to_datetime(raw["date"])
        piv = raw.pivot_table(index="date", columns="series_id", values="value").reset_index()
        for sid in ["DCOILWTICO", "VIXCLS", "OVXCLS", "SP500"]:
            if sid in piv.columns:
                fig = px.line(piv, x="date", y=sid, title=sid)
                st.plotly_chart(fig, use_container_width=True)

    corr = _safe_read_sql(
        "SELECT date, feature_value FROM market_features WHERE feature_name='oil_spx_corr_63'"
    )
    if not corr.empty:
        corr["date"] = pd.to_datetime(corr["date"])
        fig = px.line(corr, x="date", y="feature_value", title="oil_spx_corr_63")
        st.plotly_chart(fig, use_container_width=True)

    diag_features = _safe_read_sql(
        "SELECT date, feature_name, feature_value FROM market_features "
        "WHERE feature_name IN ('spx_return', 'spx_return_lag1', 'oil_outlier_move_z')"
    )
    if not diag_features.empty:
        diag_features["date"] = pd.to_datetime(diag_features["date"])
        for feature in ["spx_return", "spx_return_lag1", "oil_outlier_move_z"]:
            chunk = diag_features[diag_features["feature_name"] == feature]
            if chunk.empty:
                continue
            fig = px.line(chunk, x="date", y="feature_value", title=feature)
            st.plotly_chart(fig, use_container_width=True)

    nf = _safe_read_sql(
        "SELECT date, feature_name, feature_value FROM news_features "
        "WHERE feature_name IN ('geopolitical_risk_score', 'intensity_sum')"
    )
    if not nf.empty:
        nf["date"] = pd.to_datetime(nf["date"])
        for feature in nf["feature_name"].unique():
            chunk = nf[nf["feature_name"] == feature]
            fig = px.line(chunk, x="date", y="feature_value", title=feature)
            st.plotly_chart(fig, use_container_width=True)

    of = _safe_read_sql(
        "SELECT date, feature_value FROM options_features WHERE feature_name='put_call_ratio_mean'"
    )
    if not of.empty:
        of["date"] = pd.to_datetime(of["date"])
        fig = px.line(of, x="date", y="feature_value", title="options_put_call_ratio_mean")
        st.plotly_chart(fig, use_container_width=True)

    tail = _safe_read_sql("SELECT date, tail_risk_prob FROM tail_risk_predictions ORDER BY date")
    if not tail.empty:
        tail["date"] = pd.to_datetime(tail["date"])
        fig = px.line(tail, x="date", y="tail_risk_prob", title="Tail risk probability")
        st.plotly_chart(fig, use_container_width=True)

    ms = _safe_read_sql("SELECT date, state_id, state_label FROM model_state")
    if not ms.empty:
        ms["date"] = pd.to_datetime(ms["date"])
        fig = px.scatter(ms, x="date", y="state_id", color="state_label", title="Regime timeline")
        st.plotly_chart(fig, use_container_width=True)

    signals = _safe_read_sql("SELECT * FROM signals ORDER BY date DESC")
    if not signals.empty:
        latest_date = str(signals.iloc[0]["date"])
        triggered = signals[
            (signals["date"].astype(str) == latest_date) & (signals["signal_value"] == 1.0)
        ]
        st.subheader(f"Triggered signals ({latest_date})")
        if triggered.empty:
            st.write("No triggered signals on latest date.")
        else:
            st.dataframe(
                triggered[["signal_name", "signal_value", "metadata_json"]], hide_index=True
            )

        evidence_path = Path("artifacts") / f"evidence_{latest_date}.md"
        if evidence_path.exists():
            st.subheader("Latest evidence pack")
            st.markdown(evidence_path.read_text(encoding="utf-8"))

    rep = _safe_read_sql("SELECT path FROM reports ORDER BY created_at DESC LIMIT 1")
    if not rep.empty:
        path = Path(rep.iloc[0]["path"])
        if path.exists():
            st.markdown(path.read_text(encoding="utf-8"))

    eval_rep = _safe_read_sql(
        "SELECT path FROM reports WHERE path LIKE 'reports/eval_%' ORDER BY created_at DESC LIMIT 1"
    )
    if not eval_rep.empty:
        eval_path = Path(eval_rep.iloc[0]["path"])
        if eval_path.exists():
            st.subheader("Latest signal evaluation")
            st.markdown(eval_path.read_text(encoding="utf-8"))

    latest_eval = _safe_read_sql(
        "SELECT eval_name, eval_json, date FROM signal_eval "
        "WHERE eval_name IN ('lag_effect_summary', 'overreaction_fade_summary') "
        "ORDER BY created_at DESC"
    )
    if not latest_eval.empty:
        st.subheader("Latest lag and overreaction diagnostics")
        for eval_name in ["lag_effect_summary", "overreaction_fade_summary"]:
            chunk = latest_eval[latest_eval["eval_name"] == eval_name]
            if chunk.empty:
                continue
            row = chunk.iloc[0]
            parsed = json.loads(row["eval_json"])
            st.caption(f"{eval_name} ({row['date']})")
            st.dataframe(pd.DataFrame(parsed), hide_index=True)


if __name__ == "__main__":
    main()
