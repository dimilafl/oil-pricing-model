from __future__ import annotations

from pathlib import Path

import pandas as pd

from oil_risk import dashboard


def test_build_data_status_and_missing_list(tmp_path: Path):
    existing_report = tmp_path / "report.md"
    existing_report.write_text("ok", encoding="utf-8")

    query_map = {
        "SELECT date FROM market_raw": pd.DataFrame({"date": ["2024-01-01", "2024-01-02"]}),
        "SELECT date FROM market_features": pd.DataFrame({"date": ["2024-01-02"]}),
        "SELECT datetime FROM news_raw": pd.DataFrame({"datetime": ["2024-01-02T00:00:00"]}),
        "SELECT date FROM news_features": pd.DataFrame(),
        "SELECT date FROM news_normalized": pd.DataFrame({"date": ["2024-01-02"]}),
        "SELECT date FROM model_state": pd.DataFrame(),
        "SELECT date FROM signals": pd.DataFrame(),
        "SELECT date FROM signal_eval": pd.DataFrame({"date": ["2024-01-02"]}),
        "SELECT path, created_at FROM reports": pd.DataFrame(
            {
                "path": [str(existing_report), "reports/missing.md"],
                "created_at": ["2024-01-02", "2024-01-03"],
            }
        ),
        "SELECT date FROM tail_risk_predictions": pd.DataFrame({"date": ["2024-01-01"]}),
    }

    def fake_read_sql(query: str) -> pd.DataFrame:
        return query_map[query].copy()

    status_df, missing = dashboard.build_data_status(fake_read_sql)
    counts = dict(zip(status_df["name"], status_df["row_count"], strict=False))

    assert counts["market_raw"] == 2
    assert counts["news_features"] == 0
    assert counts["reports"] == 2
    assert any("news_features is empty" in item for item in missing)
    assert any("model_state is empty" in item for item in missing)
    assert any("signals is empty" in item for item in missing)
    assert any("reports are missing or stale" in item for item in missing)
