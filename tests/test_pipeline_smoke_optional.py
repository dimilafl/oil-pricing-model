import pandas as pd

from oil_risk.pipelines import build_features


def test_build_features_smoke_without_optional_keys(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "cache").mkdir(parents=True, exist_ok=True)

    market = pd.DataFrame(
        {
            "series_id": ["DCOILWTICO", "DCOILBRENTEU", "VIXCLS", "OVXCLS", "DTWEXBGS", "DGS10"] * 70,
            "date": [pd.Timestamp("2024-01-01") + pd.Timedelta(days=i // 6) for i in range(420)],
            "value": [100 + i for i in range(420)],
        }
    )

    def fake_read_sql(query: str):
        if "FROM market_raw" in query:
            return market
        return pd.DataFrame()

    writes = []

    def fake_write(df, table_name, replace=False):
        writes.append((table_name, len(df), replace))

    monkeypatch.setattr(build_features, "read_sql", fake_read_sql)
    monkeypatch.setattr(build_features, "write_dataframe", fake_write)
    build_features.run()
    assert any(w[0] == "market_features" for w in writes)
