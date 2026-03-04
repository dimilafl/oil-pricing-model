from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import pandas as pd

from oil_risk.config import settings
from oil_risk.db.io import read_sql, write_dataframe
from oil_risk.logging_utils import setup_logging
from oil_risk.modeling.regime import save_model, train_regime_model

FEATURES = ["oil_return", "dVIX", "dOVX", "usd_change", "rate_change", "news_risk_score"]


def _assemble_training_frame() -> pd.DataFrame:
    mkt = read_sql("SELECT date, feature_name, feature_value FROM market_features")
    nws = read_sql("SELECT date, feature_name, feature_value FROM news_features")
    mkt["date"] = pd.to_datetime(mkt["date"])
    nws["date"] = pd.to_datetime(nws["date"])
    mw = mkt.pivot_table(index="date", columns="feature_name", values="feature_value")
    nw = nws.pivot_table(index="date", columns="feature_name", values="feature_value")
    nw = nw.rename(columns={"geopolitical_risk_score": "news_risk_score"})
    return mw.join(nw[["news_risk_score"]], how="inner").sort_index()


def run() -> None:
    setup_logging()
    train_df = _assemble_training_frame()
    model, state_df = train_regime_model(train_df, FEATURES)
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
    logging.info("Model trained and state table updated")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
