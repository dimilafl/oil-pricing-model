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
    assert out["spx_return_lag1"].equals(out["spx_return"].shift(1))
    assert out["oil_return_lag2"].equals(out["oil_return"].shift(2))
    assert out["dVIX_lag3"].equals(out["dVIX"].shift(3))


def test_oil_overreaction_flag_toggles_from_zscore_threshold():
    idx = pd.date_range("2024-01-01", periods=21, freq="D")
    oil = [100.0 + i * 0.01 for i in range(20)] + [300.0]
    brent = [101.0 + i * 0.01 for i in range(20)] + [300.0]
    df = pd.DataFrame(
        {
            "DCOILWTICO": oil,
            "DCOILBRENTEU": brent,
            "VIXCLS": [20.0 + i * 0.1 for i in range(21)],
            "OVXCLS": [30.0 + i * 0.1 for i in range(21)],
            "DTWEXBGS": [100.0 + i for i in range(21)],
            "DGS10": [4.0 + i * 0.01 for i in range(21)],
            "SP500": [4000.0 + i for i in range(21)],
        },
        index=idx,
    )
    out = build_market_features(df)
    assert out.iloc[-1]["oil_outlier_move_z"] > 2.0
    assert out.iloc[-1]["oil_overreaction_flag"] == 1.0
    assert out.iloc[10]["oil_overreaction_flag"] == 0.0


def test_build_news_features_with_llm():
    news = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2024-01-01").date(),
                pd.Timestamp("2024-01-01").date(),
                pd.Timestamp("2024-01-02").date(),
                pd.Timestamp("2024-01-03").date(),
            ],
            "article_count": [1, 1, 2, 3],
            "keyword_count": [2, 3, 1, 4],
            "tone": [-5.0, -2.0, -1.0, -4.0],
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
    assert "news_risk_score_lag1" in out.columns
    assert out["news_risk_score_lag1"].equals(out["geopolitical_risk_score"].shift(1))


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
