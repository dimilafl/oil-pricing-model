from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from time import perf_counter

import pandas as pd

from oil_risk.adapters.gdelt_adapter import GdeltAdapter
from oil_risk.config import settings
from oil_risk.db.io import read_sql, write_dataframe
from oil_risk.db.schema import get_engine, init_db
from oil_risk.llm.news_classifier import OpenAINewsClassifier
from oil_risk.logging_utils import setup_logging


def _build_degraded_norm_df(days: int) -> pd.DataFrame:
    today = datetime.now(UTC).date()
    rows = [
        {
            "date": today - timedelta(days=offset),
            "article_count": 0,
            "keyword_count": 0,
            "tone": None,
        }
        for offset in range(days - 1, -1, -1)
    ]
    return pd.DataFrame(rows)


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
            logging.warning("LLM classification failed for %s: %s", row["id"], exc)
    return pd.DataFrame(rows)


def _write_runlog(payload: dict) -> None:
    runlog_dir = settings.data_dir / "runlogs"
    runlog_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    runlog_path = runlog_dir / f"update_news_{stamp}.json"
    runlog_path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")


def run() -> None:
    started = perf_counter()
    setup_logging()
    init_db()
    adapter = GdeltAdapter(settings.cache_dir)
    degraded_mode_used = False
    try:
        raw_df, norm_df = adapter.fetch_and_parse()
        degraded_mode_used = adapter.degraded_mode_used
    except Exception as exc:  # noqa: BLE001
        logging.warning("GDELT fetch failed; entering degraded mode in update_news: %s", exc)
        degraded_mode_used = True
        raw_df = pd.DataFrame()
        norm_df = _build_degraded_norm_df(int(os.getenv("LOOKBACK_DAYS", "90")))

    if not raw_df.empty:
        raw_df = raw_df.drop_duplicates(subset=["id"], keep="last")
        min_dt = pd.to_datetime(raw_df["datetime"]).min()
        with get_engine().begin() as conn:
            conn.exec_driver_sql(
                "DELETE FROM news_raw WHERE datetime >= ?",
                (min_dt.to_pydatetime().replace(tzinfo=None).isoformat(sep=" "),),
            )
        write_dataframe(raw_df, "news_raw", replace=False)

    # Aggregate to one row per date so the parquet accumulates correctly.
    # norm_df has article-level rows; groupby-date collapses them before merge.
    def _agg_to_date(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return df.groupby("date", as_index=False).agg(
            article_count=("article_count", "sum"),
            keyword_count=("keyword_count", "sum"),
            tone=("tone", "mean"),
        )

    norm_df = _agg_to_date(norm_df)
    nfile = settings.cache_dir / "news_normalized.parquet"
    if nfile.exists():
        try:
            existing = _agg_to_date(pd.read_parquet(nfile))
            norm_df = (
                pd.concat([existing, norm_df])
                .drop_duplicates(subset=["date"], keep="last")
                .sort_values("date")
                .reset_index(drop=True)
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("Could not merge existing news_normalized parquet: %s", exc)
    norm_df.to_parquet(nfile, index=False)
    norm_df_db = norm_df.copy()
    for col in ("themes", "persons", "organizations", "locations"):
        if col in norm_df_db.columns:
            norm_df_db[col] = (
                norm_df_db[col]
                .where(norm_df_db[col].notna(), None)
                .apply(lambda x: json.dumps(x) if x is not None else None)
            )
    write_dataframe(norm_df_db, "news_normalized", replace=True)

    llm_df = _classify_news_once(raw_df)
    if not llm_df.empty:
        write_dataframe(llm_df, "news_llm", replace=False)

    runlog = {
        "raw_rows": len(raw_df),
        "normalized_rows": len(norm_df),
        "llm_rows": len(llm_df),
        "min_dt": pd.to_datetime(raw_df["datetime"]).min() if not raw_df.empty else None,
        "max_dt": pd.to_datetime(raw_df["datetime"]).max() if not raw_df.empty else None,
        "duration_seconds": round(perf_counter() - started, 3),
        "cache_hit": adapter.last_cache_hit,
        "fallback_used": adapter.fallback_used,
        "degraded_mode_used": degraded_mode_used,
    }
    _write_runlog(runlog)
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
