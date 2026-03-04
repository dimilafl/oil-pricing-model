import json

import pandas as pd

from oil_risk.pipelines import build_evidence_pack


def test_evidence_pack_writer_produces_required_keys(monkeypatch, tmp_path):
    signals = pd.DataFrame(
        [
            {
                "date": "2024-01-05",
                "signal_name": "tail_risk_alert",
                "signal_value": 1.0,
                "metadata_json": "{}",
            }
        ]
    )
    tail = pd.DataFrame(
        [{"date": "2024-01-05", "tail_risk_prob": 0.7, "model_name": "logistic_regression"}]
    )
    news = pd.DataFrame(
        [
            {
                "id": "1",
                "datetime": "2024-01-04T12:00:00",
                "source": "x",
                "url": "https://x",
                "title": "headline",
                "category": "geopolitics",
                "intensity": 3,
                "summary": "sum",
                "keyword_hits": 2,
            }
        ]
    )

    def fake_read_sql(query: str):
        if "FROM signals" in query:
            return signals.copy()
        if "FROM tail_risk_predictions" in query:
            return tail.copy()
        if "FROM news_raw" in query:
            return news.copy()
        raise AssertionError(query)

    fake_features = pd.DataFrame(
        {
            "oil_return": [0.01, -0.02, 0.03, 0.01, -0.01],
            "VIX_z_63": [0.1, 0.2, 0.3, 0.4, 2.0],
        },
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )

    monkeypatch.setattr(build_evidence_pack, "read_sql", fake_read_sql)
    monkeypatch.setattr(build_evidence_pack, "load_feature_frame", lambda: fake_features)
    monkeypatch.chdir(tmp_path)

    json_path, md_path = build_evidence_pack.run(latest_date="2024-01-05")
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert json_path.exists()
    assert md_path.exists()
    assert {
        "latest_date",
        "triggered_signals",
        "tail_risk",
        "top_feature_contributors",
        "top_news_items",
    }.issubset(payload.keys())
