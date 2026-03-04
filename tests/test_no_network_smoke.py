import pandas as pd

from oil_risk.pipelines import evaluate_signals, export_alerts


def test_pipeline_smoke_no_network(monkeypatch, tmp_path):
    market = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=12, freq="D").strftime("%Y-%m-%d"),
            "oil_return": [0.01] * 12,
        }
    )
    state = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=12, freq="D").strftime("%Y-%m-%d"),
            "state_id": [0] * 12,
            "state_label": ["calm"] * 12,
            "state_probabilities_json": ["{}"] * 12,
        }
    )
    signals = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=12, freq="D").strftime("%Y-%m-%d"),
            "signal_name": ["risk_premium_alert"] * 12,
            "signal_value": [1.0] * 12,
            "metadata_json": ["{}"] * 12,
        }
    )

    writes = []

    def fake_eval_read(query: str):
        if "FROM market_features" in query:
            return market[["date", "oil_return"]].copy()
        if "FROM model_state" in query:
            return state[["date", "state_id", "state_label"]].copy()
        if "FROM signals" in query:
            return signals[["date", "signal_name", "signal_value"]].copy()
        raise AssertionError(query)

    monkeypatch.setattr(evaluate_signals, "read_sql", fake_eval_read)
    monkeypatch.setattr(
        evaluate_signals,
        "write_dataframe",
        lambda df, table_name, replace=False: writes.append(table_name),
    )
    monkeypatch.chdir(tmp_path)
    evaluate_signals.run()

    def fake_alerts_read(query: str):
        if "FROM model_state" in query:
            return state.tail(1).copy()
        if "FROM signals" in query:
            return signals.tail(3).copy()
        if "FROM news_features" in query:
            return pd.DataFrame([{"feature_name": "geopolitical_risk_score", "feature_value": 0.0}])
        raise AssertionError(query)

    monkeypatch.setattr(export_alerts, "read_sql", fake_alerts_read)
    monkeypatch.chdir(tmp_path)
    export_alerts.run()

    assert "signal_eval" in writes
    assert (tmp_path / "artifacts" / "alerts.json").exists()
