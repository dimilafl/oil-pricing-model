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
    for sid in ["DCOILWTICO", "VIXCLS", "OVXCLS"]:
        fig = px.line(piv, x="date", y=sid, title=sid)
        st.plotly_chart(fig, use_container_width=True)

    nf = read_sql(
        "SELECT date, feature_value FROM news_features WHERE feature_name='geopolitical_risk_score'"
    )
    if not nf.empty:
        nf["date"] = pd.to_datetime(nf["date"])
        fig = px.line(nf, x="date", y="feature_value", title="news_risk_score")
        st.plotly_chart(fig, use_container_width=True)

    ms = read_sql("SELECT date, state_id, state_label FROM model_state")
    if not ms.empty:
        ms["date"] = pd.to_datetime(ms["date"])
        fig = px.scatter(ms, x="date", y="state_id", color="state_label", title="Regime timeline")
        st.plotly_chart(fig, use_container_width=True)

    rep = read_sql("SELECT path FROM reports ORDER BY created_at DESC LIMIT 1")
    if not rep.empty:
        path = Path(rep.iloc[0]["path"])
        if path.exists():
            st.markdown(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
