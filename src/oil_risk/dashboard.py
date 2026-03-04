from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from oil_risk.db.io import read_sql


def main() -> None:
    st.title("Oil Risk Dashboard")
    raw = read_sql("SELECT date, series_id, value FROM market_raw")
    raw["date"] = pd.to_datetime(raw["date"])
    piv = raw.pivot_table(index="date", columns="series_id", values="value").reset_index()
    for sid in ["DCOILWTICO", "VIXCLS", "OVXCLS", "SP500"]:
        if sid in piv.columns:
            fig = px.line(piv, x="date", y=sid, title=sid)
            st.plotly_chart(fig, use_container_width=True)

    corr = read_sql(
        "SELECT date, feature_value FROM market_features WHERE feature_name='oil_spx_corr_63'"
    )
    if not corr.empty:
        corr["date"] = pd.to_datetime(corr["date"])
        fig = px.line(corr, x="date", y="feature_value", title="oil_spx_corr_63")
        st.plotly_chart(fig, use_container_width=True)

    nf = read_sql(
        "SELECT date, feature_name, feature_value FROM news_features "
        "WHERE feature_name IN ('geopolitical_risk_score', 'intensity_sum')"
    )
    if not nf.empty:
        nf["date"] = pd.to_datetime(nf["date"])
        for feature in nf["feature_name"].unique():
            chunk = nf[nf["feature_name"] == feature]
            fig = px.line(chunk, x="date", y="feature_value", title=feature)
            st.plotly_chart(fig, use_container_width=True)

    of = read_sql(
        "SELECT date, feature_value FROM options_features WHERE feature_name='put_call_ratio_mean'"
    )
    if not of.empty:
        of["date"] = pd.to_datetime(of["date"])
        fig = px.line(of, x="date", y="feature_value", title="options_put_call_ratio_mean")
        st.plotly_chart(fig, use_container_width=True)

    tail = read_sql("SELECT date, tail_risk_prob FROM tail_risk_predictions ORDER BY date")
    if not tail.empty:
        tail["date"] = pd.to_datetime(tail["date"])
        fig = px.line(tail, x="date", y="tail_risk_prob", title="Tail risk probability")
        st.plotly_chart(fig, use_container_width=True)

    ms = read_sql("SELECT date, state_id, state_label FROM model_state")
    if not ms.empty:
        ms["date"] = pd.to_datetime(ms["date"])
        fig = px.scatter(ms, x="date", y="state_id", color="state_label", title="Regime timeline")
        st.plotly_chart(fig, use_container_width=True)

    signals = read_sql("SELECT * FROM signals ORDER BY date DESC")
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

    rep = read_sql("SELECT path FROM reports ORDER BY created_at DESC LIMIT 1")
    if not rep.empty:
        path = Path(rep.iloc[0]["path"])
        if path.exists():
            st.markdown(path.read_text(encoding="utf-8"))

    eval_rep = read_sql(
        "SELECT path FROM reports WHERE path LIKE 'reports/eval_%' ORDER BY created_at DESC LIMIT 1"
    )
    if not eval_rep.empty:
        eval_path = Path(eval_rep.iloc[0]["path"])
        if eval_path.exists():
            st.subheader("Latest signal evaluation")
            st.markdown(eval_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
