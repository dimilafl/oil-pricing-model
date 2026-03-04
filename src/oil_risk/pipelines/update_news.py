from __future__ import annotations

import logging

from oil_risk.adapters.gdelt_adapter import GdeltAdapter
from oil_risk.config import settings
from oil_risk.db.io import write_dataframe
from oil_risk.db.schema import init_db
from oil_risk.logging_utils import setup_logging


def run() -> None:
    setup_logging()
    init_db()
    adapter = GdeltAdapter(settings.cache_dir)
    raw_df, norm_df = adapter.fetch_and_parse()
    if not raw_df.empty:
        write_dataframe(raw_df, "news_raw", replace=True)
    if not norm_df.empty:
        norm_df.to_parquet(settings.cache_dir / "news_normalized.parquet", index=False)
    logging.info("Wrote %s raw news rows and %s normalized rows", len(raw_df), len(norm_df))


def main() -> None:
    run()


if __name__ == "__main__":
    main()
