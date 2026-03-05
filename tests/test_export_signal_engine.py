import json
from pathlib import Path

import pandas as pd

from oil_risk.pipelines import export_signal_engine


def test_export_signal_engine_writes_payload(monkeypatch, tmp_path):
    model_state = pd.DataFrame(
        [
            {
                "date": "2024-02-03",
                "state_id": 2,
                "state_label": "stress",
                "state_probabilities_json": json.dumps({"p0": 0.1, "p1": 0.2, "p2": 0.7}),
            }
        ]
    )
    signals = pd.DataFrame(
        [
            {"date": "2024-02-03", "signal_name": "risk_premium_alert", "signal_value": 1.0},
            {"date": "2024-02-03", "signal_name": "macro_stress_alert", "signal_value": 0.0},
        ]
    )
    market = pd.DataFrame(
        [
            {"feature_name": "oil_return", "feature_value": -0.02},
            {"feature_name": "VIX_z_63", "feature_value": 1.1},
        ]
    )
    options = pd.DataFrame(
        [
            {"feature_name": "OVX_z_63", "feature_value": 1.9},
            {"feature_name": "put_call_ratio_mean", "feature_value": 0.73},
        ]
    )
    news = pd.DataFrame([{"feature_name": "geopolitical_risk_score", "feature_value": 1.5}])
    reports = pd.DataFrame(
        [
            {"path": "reports/eval_2024-02-03.md"},
            {"path": "reports/report_2024-02-03.md"},
        ]
    )

    def fake_read_sql(query: str):
        if "FROM model_state" in query:
            return model_state.copy()
        if "FROM signals" in query:
            return signals.copy()
        if "FROM market_features" in query:
            return market.copy()
        if "FROM options_features" in query:
            return options.copy()
        if "FROM news_features" in query:
            return news.copy()
        if "FROM reports" in query:
            return reports.copy()
        raise AssertionError(query)

    monkeypatch.setattr(export_signal_engine, "read_sql", fake_read_sql)
    monkeypatch.chdir(tmp_path)
    alerts_path = Path("artifacts/alerts.json")
    alerts_path.parent.mkdir(parents=True, exist_ok=True)
    alerts_path.write_text(json.dumps({"latest_date": "2024-02-03"}), encoding="utf-8")

    out_path = export_signal_engine.run()

    assert out_path == Path("artifacts/signal_engine_signal.json")
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {
        "id",
        "source",
        "timestamp_utc",
        "raw_content",
        "alerts",
        "report_paths",
    }
    assert payload["id"] == "oil-risk:2024-02-03"
    assert payload["source"] == "oil-risk"
    assert payload["alerts"] == {"latest_date": "2024-02-03"}
    assert payload["report_paths"] == {
        "daily_report": "reports/report_2024-02-03.md",
        "eval_report": "reports/eval_2024-02-03.md",
    }

    raw = payload["raw_content"]
    assert "date: 2024-02-03" in raw
    assert "regime_state: stress" in raw
    assert 'state_probabilities: {"p0": 0.1, "p1": 0.2, "p2": 0.7}' in raw
    assert "triggered_signals: risk_premium_alert" in raw
    assert "- oil_return:" in raw
    assert "- VIX_z_63:" in raw
    assert "- OVX_z_63:" in raw
    assert "- news_risk_score:" in raw
    assert "- put_call_ratio_mean:" in raw
    assert "daily_report_path: reports/report_2024-02-03.md" in raw
