from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import text

from oil_risk.db.schema import get_engine, init_db


def write_dataframe(df: pd.DataFrame, table_name: str, replace: bool = False) -> None:
    init_db()
    engine = get_engine()
    if_exists = "replace" if replace else "append"
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)


def delete_by_date(table_name: str, dt: date) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {table_name} WHERE date = :d"), {"d": dt.isoformat()})


def read_sql(query: str) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(query, engine)
