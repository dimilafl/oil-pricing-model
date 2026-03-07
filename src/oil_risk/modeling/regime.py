from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass
class RegimeModel:
    scaler: StandardScaler
    model: GaussianMixture
    feature_cols: list[str]


def train_regime_model(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[RegimeModel, pd.DataFrame]:
    train = df[feature_cols].dropna()
    scaler = StandardScaler()
    x = scaler.fit_transform(train)
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(x)
    probs = gmm.predict_proba(x)
    states = gmm.predict(x)
    out = pd.DataFrame(index=train.index)
    out["state_id"] = states
    for i in range(3):
        out[f"p_state_{i}"] = probs[:, i]
    means = train.groupby(states).mean()
    rank = means["news_risk_score"].sort_values().index.tolist()
    _state_labels = ["low_risk", "medium_risk", "high_risk"]
    label_map: dict[int, str] = {rank[i]: _state_labels[i] for i in range(len(rank))}
    for s in range(3):
        if s not in label_map:
            label_map[s] = _state_labels[len(rank) - 1] if rank else "low_risk"
    out["state_label"] = out["state_id"].map(label_map)
    return RegimeModel(scaler=scaler, model=gmm, feature_cols=feature_cols), out


def save_model(model: RegimeModel, path: str) -> None:
    joblib.dump(model, path)
