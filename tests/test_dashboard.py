from __future__ import annotations

import pandas as pd

from oil_risk import dashboard


def test_build_data_status_missing_list(monkeypatch):
    def fake_read_sql(query: str) -> pd.DataFrame:
        if "COUNT(*) AS row_count FROM news_normalized" in query:
            return pd.DataFrame([{"row_count": 10}])
        if "COUNT(*) AS row_count FROM news_features" in query:
            return pd.DataFrame([{"row_count": 0}])
        if "COUNT(*) AS row_count FROM model_state" in query:
            return pd.DataFrame([{"row_count": 0}])
        if "COUNT(*) AS row_count FROM signals" in query:
            return pd.DataFrame([{"row_count": 0}])
        if "COUNT(*) AS row_count FROM reports" in query:
            return pd.DataFrame([{"row_count": 1}])
        if "COUNT(*) AS row_count FROM signal_eval" in query:
            return pd.DataFrame([{"row_count": 3}])
        if "COUNT(*) AS row_count FROM tail_risk_predictions" in query:
            return pd.DataFrame([{"row_count": 4}])

        if "SELECT MAX(date) AS latest_date FROM news_normalized" in query:
            return pd.DataFrame([{"latest_date": "2024-01-02"}])
        if "SELECT MAX(created_at) AS latest_date FROM reports" in query:
            return pd.DataFrame([{"latest_date": "2024-01-02"}])
        if "SELECT MAX(date) AS latest_date FROM signal_eval" in query:
            return pd.DataFrame([{"latest_date": "2024-01-02"}])
        if "SELECT MAX(date) AS latest_date FROM tail_risk_predictions" in query:
            return pd.DataFrame([{"latest_date": "2024-01-02"}])
        if "SELECT path FROM reports ORDER BY created_at DESC LIMIT 1" in query:
            return pd.DataFrame([{"path": "reports/missing_latest.md"}])

        raise RuntimeError("missing table")

    monkeypatch.setattr(dashboard, "read_sql", fake_read_sql)

    status, missing = dashboard._build_data_status()
    assert set(status["table"]) == {
        "news_normalized",
        "news_features",
        "model_state",
        "signals",
        "reports",
        "signal_eval",
        "tail_risk_predictions",
    }
    assert any("news_features" in item for item in missing)
    assert any("model_state" in item for item in missing)
    assert any("signals" in item for item in missing)
    assert any("stale" in item for item in missing)
