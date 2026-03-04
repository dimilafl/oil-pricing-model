from __future__ import annotations

import logging

import pandas as pd

from oil_risk.adapters.fred_adapter import FredAdapter
from oil_risk.config import settings
from oil_risk.db.io import write_dataframe
from oil_risk.db.schema import init_db
from oil_risk.logging_utils import setup_logging

SERIES = ["DCOILWTICO", "DCOILBRENTEU", "VIXCLS", "OVXCLS", "DTWEXBGS", "DGS10"]


def run() -> None:
    setup_logging()
    init_db()
    adapter = FredAdapter(settings.cache_dir)
    frames: list[pd.DataFrame] = []
    for sid in SERIES:
        frames.append(adapter.series_to_dataframe(sid))
    merged = pd.concat(frames, ignore_index=True)
    write_dataframe(merged, "market_raw", replace=True)
    logging.info("Wrote %s market rows", len(merged))


def main() -> None:
    run()


if __name__ == "__main__":
    main()
