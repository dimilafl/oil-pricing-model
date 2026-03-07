import pandas as pd

from oil_risk.pipelines import generate_signals


def test_generate_signals_backfills_history(monkeypatch):
    frame = pd.DataFrame(
        {
            "oil_return": [0.01, -0.03, 0.02, -0.01, 0.03, 0.04],
            "VIX_z_63": [0.8, 1.2, 1.1, 1.3, 0.9, 1.0],
            "OVX_z_63": [0.9, 1.1, 1.3, 1.0, 1.4, 1.2],
            "lagged_risk_pressure": [0.5, 1.7, 2.1, 1.4, 0.9, 1.8],
            "geopolitical_risk_score": [0.7, 1.1, 1.3, 0.8, 1.2, 1.0],
            "oil_spx_corr_63": [-0.1, -0.2, float("nan"), -0.05, -0.12, -0.1],
            "oil_vix_corr_63_proxy": [-0.25, -0.28, -0.45, -0.12, -0.18, -0.2],
            "dVIX_lag1": [0.1, 0.2, 0.3, 0.15, 0.05, 0.2],
            "news_risk_score_lag1": [0.2, 0.4, 0.5, 0.3, 0.2, 0.25],
            "spx_return_lag1": [-0.01, -0.02, -0.03, -0.01, 0.01, 0.0],
        },
        index=pd.to_datetime(
            [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-06",
            ]
        ),
    )

    monkeypatch.setattr(generate_signals, "load_feature_frame", lambda: frame)
    monkeypatch.setattr(
        generate_signals,
        "load_signals_config",
        lambda: {
            "risk_premium_alert": {"enabled": True, "ovx_z_min": 1.0, "news_risk_min": 1.0},
            "macro_stress_alert": {"enabled": True, "vix_z_min": 1.0, "oil_return_min": -0.02},
            "correlation_break_alert": {
                "enabled": True,
                "corr_min": -0.3,
                "corr_feature_preference": ["oil_spx_corr_63", "oil_vix_corr_63_proxy"],
            },
            "hedging_pressure_alert": {
                "enabled": True,
                "unusual_put_required": True,
                "vix_z_min": 1.0,
            },
            "tail_risk_alert": {"enabled": True, "tail_risk_prob_min": 0.5},
            "lagged_equity_pressure_alert": {
                "enabled": True,
                "lagged_risk_pressure_min": 2.0,
            },
            "tuning": {"max_trigger_rate": 0.2},
        },
    )
    monkeypatch.setattr(
        generate_signals,
        "read_sql",
        lambda query: pd.DataFrame(
            [
                {"date": "2024-01-02", "tail_risk_prob": 0.4, "model_name": "tail"},
                {"date": "2024-01-03", "tail_risk_prob": 0.7, "model_name": "tail"},
                {"date": "2024-01-05", "tail_risk_prob": 0.6, "model_name": "tail"},
            ]
        ),
    )

    writes = {}

    def fake_write(df, table_name, replace=False):
        writes[table_name] = (df.copy(), replace)

    monkeypatch.setattr(generate_signals, "write_dataframe", fake_write)

    generate_signals.run()

    out, replace = writes["signals"]
    assert replace is True

    assert set(out["date"]) == set(frame.index.date)

    lagged = out[out["signal_name"] == "lagged_equity_pressure_alert"]
    assert (
        lagged.loc[lagged["date"] == pd.Timestamp("2024-01-03").date(), "signal_value"].iloc[0]
        == 1.0
    )
    assert lagged["signal_value"].sum() == 1.0

    corr = out[out["signal_name"] == "correlation_break_alert"]
    corr_row = corr.loc[corr["date"] == pd.Timestamp("2024-01-03").date()].iloc[0]
    assert corr_row["signal_value"] == 1.0
    assert corr_row["metadata_json"]["corr_feature_used"] == "oil_vix_corr_63_proxy"

    tail = out[out["signal_name"] == "tail_risk_alert"]
    triggered_dates = set(tail.loc[tail["signal_value"] == 1.0, "date"])
    assert triggered_dates == {
        pd.Timestamp("2024-01-03").date(),
        pd.Timestamp("2024-01-05").date(),
    }
    assert tail["metadata_json"].map(lambda m: m is not None).all()
