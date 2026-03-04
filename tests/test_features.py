import pandas as pd

from oil_risk.features import build_market_features, build_news_features


def test_build_market_features():
    idx = pd.date_range("2024-01-01", periods=70, freq="D")
    df = pd.DataFrame(
        {
            "DCOILWTICO": range(70, 140),
            "DCOILBRENTEU": range(75, 145),
            "VIXCLS": range(10, 80),
            "OVXCLS": range(20, 90),
            "DTWEXBGS": range(100, 170),
            "DGS10": [4 + i * 0.01 for i in range(70)],
        },
        index=idx,
    )
    out = build_market_features(df)
    assert "oil_return" in out.columns
    assert "OVX_z_63" in out.columns


def test_build_news_features():
    news = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-01").date()],
            "article_count": [1, 1],
            "keyword_count": [2, 3],
            "tone": [-5.0, -2.0],
        }
    )
    out = build_news_features(news)
    assert "geopolitical_risk_score" in out.columns
