import pandas as pd

from oil_risk.pipelines import generate_signals


def test_generate_signals_includes_lagged_equity_pressure_alert(monkeypatch):
    idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
    feature_frame = pd.DataFrame(
        {
            "oil_return": [0.01, -0.02],
            "VIX_z_63": [1.5, 1.6],
            "OVX_z_63": [1.5, 1.7],
            "geopolitical_risk_score": [1.0, 1.2],
            "lagged_risk_pressure": [0.0, 2.5],
            "dVIX_lag1": [0.0, 0.5],
            "news_risk_score_lag1": [0.0, 1.1],
            "spx_return_lag1": [0.0, -0.1],
            "oil_spx_corr_63": [-0.1, -0.4],
            "unusual_put_activity": [0.0, 1.0],
            "put_call_ratio_mean": [1.0, 1.2],
        },
        index=idx,
    )

    monkeypatch.setattr(generate_signals, "load_feature_frame", lambda: feature_frame)
    monkeypatch.setattr(generate_signals, "read_sql", lambda _: pd.DataFrame())

    writes = {}
    monkeypatch.setattr(
        generate_signals,
        "write_dataframe",
        lambda df, table_name, replace=False: writes.setdefault(table_name, df.copy()),
    )

    generate_signals.run()

    out = writes["signals"]
    lagged = out[out["signal_name"] == "lagged_equity_pressure_alert"].iloc[0]
    assert lagged["signal_value"] == 1.0
    assert "components" in lagged["metadata_json"]
