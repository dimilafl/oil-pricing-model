from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
)

from oil_risk.config import settings

metadata = MetaData()

market_raw = Table(
    "market_raw",
    metadata,
    Column("series_id", String, nullable=False),
    Column("date", Date, nullable=False),
    Column("value", Float, nullable=True),
    Column("source", String, nullable=False),
    Column("pulled_at", DateTime, nullable=False, default=datetime.utcnow),
)

market_features = Table(
    "market_features",
    metadata,
    Column("date", Date, nullable=False),
    Column("feature_name", String, nullable=False),
    Column("feature_value", Float, nullable=True),
)

options_raw = Table(
    "options_raw",
    metadata,
    Column("ticker", String, nullable=False),
    Column("date", Date, nullable=False),
    Column("metric_name", String, nullable=False),
    Column("metric_value", Float, nullable=True),
    Column("source", String, nullable=False),
    Column("pulled_at", DateTime, nullable=False, default=datetime.utcnow),
)

options_features = Table(
    "options_features",
    metadata,
    Column("date", Date, nullable=False),
    Column("feature_name", String, nullable=False),
    Column("feature_value", Float, nullable=True),
)

news_raw = Table(
    "news_raw",
    metadata,
    Column("id", String, primary_key=True),
    Column("datetime", DateTime, nullable=False),
    Column("source", String, nullable=True),
    Column("url", String, nullable=True),
    Column("title", String, nullable=True),
    Column("raw_record_json", JSON, nullable=False),
    Column("pulled_at", DateTime, nullable=False, default=datetime.utcnow),
)

news_llm = Table(
    "news_llm",
    metadata,
    Column("id", String, primary_key=True),
    Column("relevance_score", Float, nullable=False),
    Column("category", String, nullable=False),
    Column("intensity", Integer, nullable=False),
    Column("entities_json", JSON, nullable=False),
    Column("summary", String, nullable=False),
    Column("model_name", String, nullable=False),
    Column("created_at", DateTime, nullable=False, default=datetime.utcnow),
)

news_features = Table(
    "news_features",
    metadata,
    Column("date", Date, nullable=False),
    Column("feature_name", String, nullable=False),
    Column("feature_value", Float, nullable=True),
)

model_state = Table(
    "model_state",
    metadata,
    Column("date", Date, nullable=False),
    Column("state_id", Integer, nullable=False),
    Column("state_label", String, nullable=False),
    Column("state_probabilities_json", JSON, nullable=False),
)

tail_risk_predictions = Table(
    "tail_risk_predictions",
    metadata,
    Column("date", Date, nullable=False),
    Column("target_horizon", String, nullable=False),
    Column("tail_risk_prob", Float, nullable=False),
    Column("model_name", String, nullable=False),
    Column("created_at", DateTime, nullable=False, default=datetime.utcnow),
    Column("feature_snapshot_json", JSON, nullable=False),
)


signals = Table(
    "signals",
    metadata,
    Column("date", Date, nullable=False),
    Column("signal_name", String, nullable=False),
    Column("signal_value", Float, nullable=False),
    Column("metadata_json", JSON, nullable=False),
)


tuning_runs = Table(
    "tuning_runs",
    metadata,
    Column("run_id", String, nullable=False),
    Column("created_at", DateTime, nullable=False, default=datetime.utcnow),
    Column("metric_name", String, nullable=False),
    Column("best_params_json", JSON, nullable=False),
    Column("leaderboard_json", JSON, nullable=False),
)


signal_eval = Table(
    "signal_eval",
    metadata,
    Column("date", Date, nullable=False),
    Column("eval_name", String, nullable=False),
    Column("eval_json", JSON, nullable=False),
    Column("created_at", DateTime, nullable=False, default=datetime.utcnow),
)

reports = Table(
    "reports",
    metadata,
    Column("date", Date, nullable=False),
    Column("path", String, nullable=False),
    Column("created_at", DateTime, nullable=False, default=datetime.utcnow),
)


def get_engine():
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{settings.db_path}")


def init_db() -> None:
    engine = get_engine()
    metadata.create_all(engine)
