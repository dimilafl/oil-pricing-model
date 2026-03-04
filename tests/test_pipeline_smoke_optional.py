import pandas as pd

from oil_risk.pipelines import build_features


def test_build_features_smoke_without_optional_keys(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "cache").mkdir(parents=True, exist_ok=True)

    dates = [pd.Timestamp("2024-01-01") + pd.Timedelta(days=i) for i in range(70)]
    market = pd.DataFrame(
        {
            "series_id": [
                sid
                for _ in dates
                for sid in ["DCOILWTICO", "DCOILBRENTEU", "VIXCLS", "OVXCLS", "DTWEXBGS", "DGS10", "SP500"]
            ],
            "date": [d for d in dates for _ in range(7)],
            "value": [100 + i for i in range(len(dates) * 7)],
        }
    )

    news_norm = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-01").date()],
            "article_count": [1, 1],
            "keyword_count": [2, 1],
            "tone": [-1.0, -2.0],
            "source": ["US", "US"],
            "title": ["Iran", "Hormuz"],
            "url": ["https://a", "https://b"],
        }
    )

    llm = pd.DataFrame(
        {
            "id": ["1"],
            "date": ["2024-01-01"],
            "category": ["direct_conflict"],
            "intensity": [2],
        }
    )

    def fake_read_sql(query: str):
        if "FROM market_raw" in query:
            return market
        if "FROM options_raw" in query:
            return pd.DataFrame()
        if "FROM news_raw r JOIN news_llm" in query:
            return llm
        return pd.DataFrame()

    writes = {}

    def fake_write(df, table_name, replace=False):
        writes[table_name] = df.copy()

    monkeypatch.setattr(build_features, "read_sql", fake_read_sql)
    monkeypatch.setattr(build_features, "write_dataframe", fake_write)
    monkeypatch.setattr(build_features.pd, "read_parquet", lambda _: news_norm)
    build_features.settings.cache_dir.mkdir(parents=True, exist_ok=True)
    (build_features.settings.cache_dir / "news_normalized.parquet").write_text("x", encoding="utf-8")

    build_features.run()

    assert "market_features" in writes
    assert "news_features" in writes
    assert "options_features" not in writes
    assert (writes["news_features"]["feature_name"] == "geopolitical_risk_score").any()
    assert (writes["market_features"]["feature_name"] == "oil_spx_corr_63").any()
