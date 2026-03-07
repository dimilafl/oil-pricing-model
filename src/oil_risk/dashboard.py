from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from oil_risk.db.io import read_sql


def _safe_read_sql(query: str) -> pd.DataFrame:
    try:
        return read_sql(query)
    except Exception:
        return pd.DataFrame()


def _latest_date_for_table(table: str, date_col: str) -> str | None:
    latest_df = _safe_read_sql(f"SELECT MAX({date_col}) AS latest_date FROM {table}")
    if latest_df.empty:
        return None
    latest_value = latest_df.iloc[0].get("latest_date")
    if latest_value in {None, "", pd.NaT}:
        return None
    return str(latest_value)


def _build_data_status() -> tuple[pd.DataFrame, list[str]]:
    checks = [
        ("news_normalized", "date"),
        ("news_features", "date"),
        ("model_state", "date"),
        ("signals", "date"),
        ("reports", "created_at"),
        ("signal_eval", "date"),
        ("tail_risk_predictions", "date"),
    ]
    rows: list[dict] = []
    missing: list[str] = []

    for table, date_col in checks:
        count_df = _safe_read_sql(f"SELECT COUNT(*) AS row_count FROM {table}")
        row_count = int(count_df.iloc[0]["row_count"]) if not count_df.empty else 0
        latest_date = _latest_date_for_table(table, date_col) if row_count > 0 else None
        rows.append({"table": table, "row_count": row_count, "latest_date": latest_date})

    status = pd.DataFrame(rows)
    status_map = {row["table"]: row["row_count"] for row in rows}
    if status_map.get("news_features", 0) == 0:
        missing.append("`news_features` is empty: dashboard can show only market/price charts.")
    if status_map.get("model_state", 0) == 0:
        missing.append("`model_state` is empty: train step did not run or failed.")
    if status_map.get("signals", 0) == 0:
        missing.append("`signals` is empty: signal generation did not run.")
    if status_map.get("reports", 0) == 0:
        missing.append("`reports` is empty: report generation did not run.")
    else:
        rep = _safe_read_sql("SELECT path FROM reports ORDER BY created_at DESC LIMIT 1")
        if not rep.empty:
            report_path = Path(rep.iloc[0]["path"])
            if not report_path.exists():
                missing.append("Latest report path is stale/missing on disk.")

    return status, missing


def main() -> None:
    st.title("Oil Risk Dashboard")

    status_df, missing_items = _build_data_status()
    st.subheader("Data status")
    st.dataframe(status_df, hide_index=True, use_container_width=True)
    st.subheader("What is missing")
    if missing_items:
        for item in missing_items:
            st.markdown(f"- {item}")
    else:
        st.markdown("- No missing pipeline artifacts detected.")

    raw = _safe_read_sql("SELECT date, series_id, value FROM market_raw")
    if not raw.empty:
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
