"""
Historical GDELT recovery script.

Fetches the last ~84 days of news in 7-day chunks to avoid API timeouts,
then merges the results into data/cache/news_normalized.parquet.

Usage:
    .venv/bin/python scripts/recover_history.py
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from oil_risk.adapters.gdelt_adapter import GdeltAdapter, GdeltQuery

CHUNK_DAYS = 7
TOTAL_WEEKS = 12  # ~84 days
SLEEP_BETWEEN_CHUNKS = 3  # seconds, to avoid rate-limiting

cache_path = Path("data/cache")
cache_path.mkdir(parents=True, exist_ok=True)
adapter = GdeltAdapter(cache_dir=cache_path)

now = datetime.now(UTC)
all_norm: list[pd.DataFrame] = []

print("--- Historical GDELT Recovery ---")
print(f"Fetching {TOTAL_WEEKS} x {CHUNK_DAYS}-day chunks (most recent first)\n")

for week in range(TOTAL_WEEKS):
    end = now - timedelta(weeks=week)
    q = GdeltQuery(days=CHUNK_DAYS, end_date=end)
    label = f"  Week -{week}: {(end - timedelta(days=CHUNK_DAYS)).strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}"
    try:
        _raw, norm_df = adapter.fetch_and_parse(query=q)
        if not norm_df.empty and norm_df["article_count"].sum() > 0:
            all_norm.append(norm_df)
            print(f"{label}  ✓ {len(norm_df)} rows, {norm_df['article_count'].sum()} articles")
        else:
            status = "degraded/empty" if adapter.degraded_mode_used else "0 articles"
            print(f"{label}  ~ {status}")
    except Exception as exc:  # noqa: BLE001
        print(f"{label}  ✗ {exc}")

    if week < TOTAL_WEEKS - 1:
        time.sleep(SLEEP_BETWEEN_CHUNKS)

if not all_norm:
    print("\nNo real data recovered — GDELT API may be unreachable right now.")
    print("Try again later, or just run: make daily")
    raise SystemExit(1)

# Merge with existing parquet
parquet_path = cache_path / "news_normalized.parquet"
new_df = pd.concat(all_norm, ignore_index=True)

# Aggregate article-level rows to date-level (same logic as update_news pipeline)
new_df = (
    new_df.groupby("date", as_index=False)
    .agg(article_count=("article_count", "sum"), keyword_count=("keyword_count", "sum"), tone=("tone", "mean"))
)

if parquet_path.exists():
    existing = pd.read_parquet(parquet_path)
    combined = pd.concat([existing, new_df], ignore_index=True)
else:
    combined = new_df

combined = (
    combined.drop_duplicates(subset=["date"], keep="last")
    .sort_values("date")
    .reset_index(drop=True)
)
combined.to_parquet(parquet_path, index=False)

print(f"\nSaved {len(combined)} date rows to {parquet_path}")
print(f"Date range: {combined['date'].min()} → {combined['date'].max()}")
print(f"Rows with real articles: {(combined['article_count'] > 0).sum()}")
print("\nNext step: run 'make daily' or '.venv/bin/oil-update-news' to sync to the database.")
