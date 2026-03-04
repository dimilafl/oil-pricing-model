from __future__ import annotations

import json
import logging

import pandas as pd

from oil_risk.adapters.gdelt_adapter import GdeltAdapter
from oil_risk.config import settings
from oil_risk.db.io import read_sql, write_dataframe
from oil_risk.db.schema import get_engine
from oil_risk.db.schema import init_db
from oil_risk.llm.news_classifier import OpenAINewsClassifier
from oil_risk.logging_utils import setup_logging


def _classify_news_once(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty or not settings.openai_api_key:
        return pd.DataFrame()
    existing = read_sql("SELECT id FROM news_llm")
    done = set(existing["id"].tolist()) if not existing.empty else set()
    todo = raw_df[~raw_df["id"].isin(done)]
    if todo.empty:
        return pd.DataFrame()

    classifier = OpenAINewsClassifier(settings.openai_api_key)
    rows: list[dict] = []
    for _, row in todo.iterrows():
        try:
            pred = classifier.classify(row.get("title"), json.dumps(row.get("raw_record_json")))
            rows.append(
                {
                    "id": row["id"],
                    "relevance_score": float(pred.get("relevance_score", 0.0)),
                    "category": pred.get("category", "other"),
                    "intensity": int(pred.get("intensity", 0)),
                    "entities_json": json.dumps(pred.get("entities", {})),
                    "summary": pred.get("short_summary", ""),
                    "model_name": pred.get("model_name", "unknown"),
                    "created_at": pred.get("created_at"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("LLM classification failed for %s: %s", row['id'], exc)
    return pd.DataFrame(rows)


def run() -> None:
    setup_logging()
    init_db()
    adapter = GdeltAdapter(settings.cache_dir)
    raw_df, norm_df = adapter.fetch_and_parse()
    if not raw_df.empty:
        min_dt = pd.to_datetime(raw_df["datetime"]).min()
        with get_engine().begin() as conn:
            conn.exec_driver_sql(
                "DELETE FROM news_raw WHERE datetime >= ?",
                (min_dt.to_pydatetime().replace(tzinfo=None).isoformat(sep=" "),),
            )
        write_dataframe(raw_df, "news_raw", replace=False)
    if not norm_df.empty:
        norm_df.to_parquet(settings.cache_dir / "news_normalized.parquet", index=False)
    llm_df = _classify_news_once(raw_df)
    if not llm_df.empty:
        write_dataframe(llm_df, "news_llm", replace=False)
    logging.info(
        "Wrote %s raw news rows, %s normalized rows, %s llm rows",
        len(raw_df),
        len(norm_df),
        len(llm_df),
    )


def main() -> None:
    run()


if __name__ == "__main__":
    main()
