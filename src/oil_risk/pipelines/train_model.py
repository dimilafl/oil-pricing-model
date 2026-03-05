from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import pandas as pd

from oil_risk.config import settings
from oil_risk.db.io import read_sql, write_dataframe
from oil_risk.logging_utils import setup_logging
from oil_risk.modeling.regime import save_model, train_regime_model
from oil_risk.modeling.tail_risk import (
    build_tail_risk_dataset,
    save_tail_risk_model,
    train_and_score_tail_risk,
)
from oil_risk.pipelines.data_views import load_feature_frame

BASE_FEATURES = ["oil_return", "dVIX", "dOVX", "usd_change", "rate_change", "news_risk_score"]
LAGGED_FEATURES = ["spx_return_lag1", "dVIX_lag1", "news_risk_score_lag1", "lagged_risk_pressure"]


def _model_features() -> list[str]:
    if settings.model_feature_set == "lagged":
        return BASE_FEATURES + LAGGED_FEATURES
    return BASE_FEATURES


def _assemble_training_frame() -> pd.DataFrame:
    mkt = read_sql("SELECT date, feature_name, feature_value FROM market_features")
    nws = read_sql("SELECT date, feature_name, feature_value FROM news_features")
    mkt["date"] = pd.to_datetime(mkt["date"])
    nws["date"] = pd.to_datetime(nws["date"])
    mw = mkt.pivot_table(index="date", columns="feature_name", values="feature_value")
    nw = nws.pivot_table(index="date", columns="feature_name", values="feature_value")
    nw = nw.rename(columns={"geopolitical_risk_score": "news_risk_score"})
    return mw.join(nw, how="inner").sort_index()


def run() -> None:
    setup_logging()
    train_df = _assemble_training_frame()
    features = _model_features()
    missing = [col for col in features if col not in train_df.columns]
    if missing:
        if settings.model_feature_set == "lagged":
            raise ValueError(
                "Missing lagged model features: " + ", ".join(missing) + ". Run oil-build-features to regenerate feature tables."
            )
        raise ValueError("Missing required model features: " + ", ".join(missing))

    model, state_df = train_regime_model(train_df, features)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    model_path = settings.models_dir / f"regime_gmm_{stamp}.joblib"
    save_model(model, str(model_path))

    out = state_df.reset_index().rename(columns={"index": "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["state_probabilities_json"] = out.apply(
        lambda r: json.dumps({"p0": r["p_state_0"], "p1": r["p_state_1"], "p2": r["p_state_2"]}),
        axis=1,
    )
    out = out[["date", "state_id", "state_label", "state_probabilities_json"]]
    write_dataframe(out, "model_state", replace=True)

    try:
        tail_input = load_feature_frame()
        tail_dataset = build_tail_risk_dataset(tail_input)
        tail_model, tail_scored = train_and_score_tail_risk(tail_dataset)
        tail_model_path = settings.models_dir / f"tail_risk_logistic_{stamp}.joblib"
        save_tail_risk_model(tail_model, str(tail_model_path))

        tail_scores = tail_scored.reset_index().rename(columns={"index": "date"})
        tail_scores["date"] = pd.to_datetime(tail_scores["date"]).dt.date
        tail_scores["feature_snapshot_json"] = tail_scores.apply(
            lambda row: json.dumps(
                {
                    key: row[key]
                    for key in tail_scores.columns
                    if key
                    not in {
                        "date",
                        "target",
                        "tail_risk_prob",
                        "target_horizon",
                        "model_name",
                        "created_at",
                    }
                }
            ),
            axis=1,
        )
        write_dataframe(
            tail_scores[
                [
                    "date",
                    "target_horizon",
                    "tail_risk_prob",
                    "model_name",
                    "created_at",
                    "feature_snapshot_json",
                ]
            ],
            "tail_risk_predictions",
            replace=True,
        )
    except ValueError as exc:
        logging.warning("Tail risk training skipped: %s", exc)
    logging.info("Model trained and state table updated")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
