import pandas as pd

from oil_risk.features import build_market_features, build_news_features, build_options_features


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
            "SP500": range(4000, 4070),
        },
        index=idx,
    )
    out = build_market_features(df)
    assert "oil_return" in out.columns
    assert "OVX_z_63" in out.columns
    assert "spx_return" in out.columns
    assert "oil_spx_corr_63" in out.columns


def test_build_news_features_with_llm():
    news = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-01").date()],
            "article_count": [1, 1],
            "keyword_count": [2, 3],
            "tone": [-5.0, -2.0],
        }
    )
    llm = pd.DataFrame(
        {
            "id": ["1"],
            "date": [pd.Timestamp("2024-01-01").date()],
            "category": ["direct_conflict"],
            "intensity": [2],
        }
    )
    out = build_news_features(news, llm)
    assert "geopolitical_risk_score" in out.columns
    assert "intensity_sum" in out.columns
    assert "category_count_direct_conflict" in out.columns


def test_build_options_features():
    options_raw = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-01").date()],
            "ticker": ["USO", "USO"],
            "metric_name": ["put_volume", "call_volume"],
            "metric_value": [120.0, 100.0],
        }
    )
    ratio = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01").date()],
            "ticker": ["USO"],
            "metric_name": ["put_call_ratio"],
            "metric_value": [1.2],
        }
    )
    out = build_options_features(pd.concat([options_raw, ratio], ignore_index=True), z_threshold=-1)
    assert "put_call_ratio_mean" in out.columns
    assert "unusual_put_activity" in out.columns



def test_build_market_features_lag_and_overreaction_features():
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    df = pd.DataFrame(
        {
            "DCOILWTICO": [100 + i for i in range(40)],
            "DCOILBRENTEU": [105 + i for i in range(40)],
            "VIXCLS": [20 + (i % 5) for i in range(40)],
            "OVXCLS": [25 + (i % 6) for i in range(40)],
            "DTWEXBGS": [110 + i for i in range(40)],
            "DGS10": [4 + i * 0.01 for i in range(40)],
            "SP500": [4000 + i for i in range(40)],
        },
        index=idx,
    )
    out = build_market_features(df)
    for col in [
        "spx_return_lag1",
        "spx_return_lag2",
        "spx_return_lag3",
        "oil_return_lag1",
        "dVIX_lag1",
        "dOVX_lag1",
        "oil_outlier_move_z",
        "oil_overreaction_flag",
    ]:
        assert col in out.columns


def test_build_news_features_adds_lags():
    news = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2024-01-01").date(),
                pd.Timestamp("2024-01-02").date(),
                pd.Timestamp("2024-01-03").date(),
                pd.Timestamp("2024-01-04").date(),
            ],
            "article_count": [1, 2, 3, 4],
            "keyword_count": [1, 2, 3, 4],
            "tone": [-1.0, -2.0, -3.0, -4.0],
        }
    )
    out = build_news_features(news)
    assert out.loc[pd.Timestamp("2024-01-04").date(), "news_risk_score_lag1"] == out.loc[
        pd.Timestamp("2024-01-03").date(), "geopolitical_risk_score"
    ]
    assert "news_risk_score_lag3" in out.columns
