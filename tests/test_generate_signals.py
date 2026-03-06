import pandas as pd

from oil_risk.pipelines import generate_signals


def test_generate_signals_includes_lagged_equity_pressure_alert(monkeypatch):
    frame = pd.DataFrame(
        {
            "oil_return": [0.01, 0.02],
            "VIX_z_63": [1.2, 1.3],
            "OVX_z_63": [1.1, 1.4],
            "lagged_risk_pressure": [0.5, 1.5],
            "geopolitical_risk_score": [0.8, 1.3],
            "oil_spx_corr_63": [-0.1, -0.2],
            "unusual_put_activity": [0.0, 1.0],
            "put_call_ratio_mean": [1.0, 1.2],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
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
                "corr_feature_preference": ["oil_spx_corr_63"],
            },
            "hedging_pressure_alert": {
                "enabled": True,
                "unusual_put_required": True,
                "vix_z_min": 1.0,
            },
            "tail_risk_alert": {"enabled": True, "tail_risk_prob_min": 0.5},
            "lagged_equity_pressure_alert": {
                "enabled": True,
                "lagged_risk_pressure_min": 1.0,
            },
            "tuning": {"max_trigger_rate": 0.2},
        },
    )
    monkeypatch.setattr(
        generate_signals,
        "read_sql",
        lambda query: pd.DataFrame(
            [{"date": "2024-01-02", "tail_risk_prob": 0.4, "model_name": "tail"}]
        ),
    )

    writes = {}

    def fake_write(df, table_name, replace=False):
        writes[table_name] = df.copy()

    monkeypatch.setattr(generate_signals, "write_dataframe", fake_write)

    generate_signals.run()

    out = writes["signals"]
    lagged = out[out["signal_name"] == "lagged_equity_pressure_alert"].iloc[0]
    assert lagged["signal_value"] == 1.0
