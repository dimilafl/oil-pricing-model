from __future__ import annotations

import numpy as np
import pandas as pd


def robust_z(series: pd.Series) -> pd.Series:
    med = series.median()
    mad = (series - med).abs().median()
    if mad == 0 or pd.isna(mad):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return 0.6745 * (series - med) / mad


def build_market_features(market_wide: pd.DataFrame) -> pd.DataFrame:
    df = market_wide.sort_index().copy()
    df["oil_return"] = np.log(df["DCOILWTICO"]).diff()
    df["brent_return"] = np.log(df["DCOILBRENTEU"]).diff()
    df["dVIX"] = df["VIXCLS"].diff()
    df["dOVX"] = df["OVXCLS"].diff()
    vix_mean = df["VIXCLS"].rolling(63).mean()
    vix_std = df["VIXCLS"].rolling(63).std()
    ovx_mean = df["OVXCLS"].rolling(63).mean()
    ovx_std = df["OVXCLS"].rolling(63).std()
    df["VIX_z_63"] = (df["VIXCLS"] - vix_mean) / vix_std
    df["OVX_z_63"] = (df["OVXCLS"] - ovx_mean) / ovx_std
    df["oil_vix_corr_63_proxy"] = df["oil_return"].rolling(63).corr(df["dVIX"])
    if "SP500" in df.columns:
        df["spx_return"] = np.log(df["SP500"]).diff()
        df["oil_spx_corr_63"] = df["oil_return"].rolling(63).corr(df["spx_return"])
    df["oil_realized_vol_10d"] = df["oil_return"].rolling(10).std()
    df["oil_realized_vol_21d"] = df["oil_return"].rolling(21).std()
    for lag in (1, 2, 3):
        df[f"spx_return_lag{lag}"] = df.get("spx_return", pd.Series(index=df.index)).shift(lag)
        df[f"oil_return_lag{lag}"] = df["oil_return"].shift(lag)
        df[f"dVIX_lag{lag}"] = df["dVIX"].shift(lag)
        df[f"dOVX_lag{lag}"] = df["dOVX"].shift(lag)
    oil_vol = df["oil_realized_vol_21d"].replace(0, np.nan)
    df["oil_outlier_move_z"] = df["oil_return"] / oil_vol
    df["oil_overreaction_flag"] = (df["oil_outlier_move_z"].abs() >= 2.0).astype(float)
    df["usd_level"] = df["DTWEXBGS"]
    df["usd_change"] = df["DTWEXBGS"].diff()
    df["rate_level"] = df["DGS10"]
    df["rate_change"] = df["DGS10"].diff()
    cols = [
        "oil_return",
        "brent_return",
        "dVIX",
        "dOVX",
        "VIX_z_63",
        "OVX_z_63",
        "oil_vix_corr_63_proxy",
        "spx_return",
        "oil_spx_corr_63",
        "oil_realized_vol_10d",
        "oil_realized_vol_21d",
        "spx_return_lag1",
        "spx_return_lag2",
        "spx_return_lag3",
        "oil_return_lag1",
        "oil_return_lag2",
        "oil_return_lag3",
        "dVIX_lag1",
        "dVIX_lag2",
        "dVIX_lag3",
        "dOVX_lag1",
        "dOVX_lag2",
        "dOVX_lag3",
        "oil_outlier_move_z",
        "oil_overreaction_flag",
        "lagged_risk_pressure",
        "usd_level",
        "usd_change",
        "rate_level",
        "rate_change",
    ]
    return df[[c for c in cols if c in df.columns]]


def build_news_features(
    news_norm: pd.DataFrame, news_llm: pd.DataFrame | None = None
) -> pd.DataFrame:
    if news_norm.empty:
        return pd.DataFrame(columns=["date"])
    grouped = news_norm.groupby("date").agg(
        article_count=("article_count", "sum"),
        keyword_count=("keyword_count", "sum"),
        tone_mean=("tone", "mean"),
    )
    grouped["negative_tone_magnitude"] = grouped["tone_mean"].fillna(0).clip(upper=0).abs()
    grouped["z_article_count"] = robust_z(grouped["article_count"])
    grouped["z_keyword_count"] = robust_z(grouped["keyword_count"])
    grouped["z_negative_tone_magnitude"] = robust_z(grouped["negative_tone_magnitude"])
    grouped["geopolitical_risk_score"] = (
        grouped["z_article_count"]
        + grouped["z_keyword_count"]
        + grouped["z_negative_tone_magnitude"]
    )

    if news_llm is not None and not news_llm.empty:
        llm = news_llm.copy()
        llm["date"] = pd.to_datetime(llm["date"]).dt.date
        llm_daily = llm.groupby("date").agg(intensity_sum=("intensity", "sum"))
        cat_counts = llm.pivot_table(
            index="date", columns="category", values="id", aggfunc="count", fill_value=0
        )
        cat_counts.columns = [f"category_count_{c}" for c in cat_counts.columns]
        llm_daily = llm_daily.join(cat_counts, how="left").fillna(0)
        grouped = grouped.join(llm_daily, how="left").fillna(0)
        grouped["geopolitical_risk_score"] = grouped["geopolitical_risk_score"] + 0.5 * robust_z(
            grouped["intensity_sum"]
        )

    for lag in (1, 2, 3):
        grouped[f"news_risk_score_lag{lag}"] = grouped["geopolitical_risk_score"].shift(lag)
    return grouped


def build_options_features(options_raw: pd.DataFrame, z_threshold: float = 1.5) -> pd.DataFrame:
    if options_raw.empty:
        return pd.DataFrame(columns=["date"])
    piv = options_raw.pivot_table(
        index=["date", "ticker"], columns="metric_name", values="metric_value", aggfunc="last"
    ).reset_index()
    piv["put_call_ratio"] = piv.get("put_call_ratio")
    piv["put_call_ratio_z"] = piv.groupby("ticker")["put_call_ratio"].transform(robust_z)
    piv["unusual_put_activity"] = (piv["put_call_ratio_z"] > z_threshold).astype(float)
    by_day = piv.groupby("date").agg(
        put_volume_total=("put_volume", "sum"),
        call_volume_total=("call_volume", "sum"),
        put_call_ratio_mean=("put_call_ratio", "mean"),
        unusual_put_activity=("unusual_put_activity", "max"),
    )
    if "implied_vol_proxy" in piv.columns:
        by_day["implied_vol_proxy_mean"] = piv.groupby("date")["implied_vol_proxy"].mean()
    return by_day
