import json
from pathlib import Path

import pandas as pd

from oil_risk.pipelines import export_alerts


def test_export_alerts_writes_required_schema(monkeypatch, tmp_path):
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
            {
                "date": "2024-02-03",
                "signal_name": "risk_premium_alert",
                "signal_value": 1.0,
                "metadata_json": json.dumps({"ovx_z_63": 2.0}),
            },
            {
                "date": "2024-02-03",
                "signal_name": "macro_stress_alert",
                "signal_value": 0.0,
                "metadata_json": json.dumps({"vix_z_63": 0.5}),
            },
        ]
    )
    news_features = pd.DataFrame(
        [
            {"feature_name": "geopolitical_risk_score", "feature_value": 1.5},
            {"feature_name": "intensity_sum", "feature_value": 8.0},
        ]
    )
    tail = pd.DataFrame([{"date": "2024-02-03", "tail_risk_prob": 0.42}])

    def fake_read_sql(query: str):
        if "FROM model_state" in query:
            return model_state.copy()
        if "FROM signals" in query:
            return signals.copy()
        if "FROM news_features" in query:
            return news_features.copy()
        if "FROM tail_risk_predictions" in query:
            return tail.copy()
        raise AssertionError(query)

    monkeypatch.setattr(export_alerts, "read_sql", fake_read_sql)
    monkeypatch.setattr(
        export_alerts,
        "build_evidence_pack",
        lambda latest_date: (Path("a"), Path(f"artifacts/evidence_{latest_date}.md")),
    )
    monkeypatch.chdir(tmp_path)

    out_path = export_alerts.run()

    assert out_path == Path("artifacts/alerts.json")
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {
        "latest_date",
        "state",
        "triggered_signals",
        "news",
        "tail_risk_prob",
        "evidence_pack_path",
    }
    assert payload["latest_date"] == "2024-02-03"
    assert payload["state"]["state_id"] == 2
    assert payload["triggered_signals"][0]["signal_name"] == "risk_premium_alert"
    assert payload["news"]["news_risk_score"] == 1.5
    assert payload["tail_risk_prob"] == 0.42
    assert payload["evidence_pack_path"] == "artifacts/evidence_2024-02-03.md"
