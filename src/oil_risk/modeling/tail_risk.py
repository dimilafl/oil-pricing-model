from __future__ import annotations

from datetime import UTC, datetime

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

TAIL_RISK_FEATURES = [
    "oil_return",
    "brent_return",
    "dVIX",
    "dOVX",
    "VIX_z_63",
    "OVX_z_63",
    "usd_change",
    "rate_change",
    "spx_return",
    "oil_spx_corr_63",
    "geopolitical_risk_score",
    "intensity_sum",
    "put_call_ratio_mean",
    "unusual_put_activity",
]


def build_tail_risk_dataset(features: pd.DataFrame, threshold: float = 0.02) -> pd.DataFrame:
    dataset = features.copy().sort_index()
    dataset["target"] = dataset["oil_return"].shift(-1).abs().ge(threshold).astype(float)
    return dataset.dropna(subset=["target"])


def _base_estimator(model_name: str):
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=300, random_state=42)
    return LogisticRegression(max_iter=1000)


def train_and_score_tail_risk(
    dataset: pd.DataFrame,
    model_name: str = "logistic_regression",
    n_splits: int = 5,
) -> tuple[CalibratedClassifierCV, pd.DataFrame]:
    available_features = [c for c in TAIL_RISK_FEATURES if c in dataset.columns]
    work = dataset[available_features + ["target"]].dropna().copy()
    if work.empty:
        raise ValueError("No rows available for tail-risk training")
    if work["target"].nunique() < 2:
        raise ValueError("Tail-risk target has fewer than two classes")

    X = work[available_features]
    y = work["target"].astype(int)

    max_splits = max(2, min(n_splits, len(work) - 1))
    cv = TimeSeriesSplit(n_splits=max_splits)
    model = CalibratedClassifierCV(_base_estimator(model_name), cv=cv, method="sigmoid")
    model.fit(X, y)

    scored = work.copy()
    scored["tail_risk_prob"] = model.predict_proba(X)[:, 1]
    scored["target_horizon"] = "1d"
    scored["model_name"] = model_name
    scored["created_at"] = datetime.now(UTC).isoformat()
    return model, scored


def save_tail_risk_model(model: CalibratedClassifierCV, path: str) -> None:
    joblib.dump(model, path)
