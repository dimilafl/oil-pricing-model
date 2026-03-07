from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from oil_risk.pipelines import update_news


class DummyAdapter:
    def __init__(
        self,
        cache_dir: Path,
        raw_df: pd.DataFrame,
        norm_df: pd.DataFrame,
        cache_hit: bool,
        fallback_used: bool = False,
        degraded_mode_used: bool = False,
        raise_exc: Exception | None = None,
    ):
        self.cache_dir = cache_dir
        self._raw = raw_df
        self._norm = norm_df
        self.last_cache_hit = cache_hit
        self.fallback_used = fallback_used
        self.degraded_mode_used = degraded_mode_used
        self._raise_exc = raise_exc

    def fetch_and_parse(self):
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._raw.copy(), self._norm.copy()


def test_update_news_dedup_normalized_and_runlog(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "test.db"
    cache_dir = tmp_path / "cache"
    runlogs = tmp_path / "runlogs"
    cache_dir.mkdir(parents=True)

    raw_batch = pd.DataFrame(
        [
            {
                "id": "a",
                "datetime": "2024-01-01T00:00:00",
                "source": "x",
                "url": "https://a",
                "title": "Iran shipping",
                "raw_record_json": "{}",
                "pulled_at": "2024-01-02T00:00:00",
            },
            {
                "id": "a",
                "datetime": "2024-01-01T00:00:00",
                "source": "x",
                "url": "https://a",
                "title": "Iran shipping",
                "raw_record_json": "{}",
                "pulled_at": "2024-01-02T00:00:00",
            },
            {
                "id": "b",
                "datetime": "2024-01-02T00:00:00",
                "source": "x",
                "url": "https://b",
                "title": "Hormuz tanker",
                "raw_record_json": "{}",
                "pulled_at": "2024-01-02T00:00:00",
            },
        ]
    )
    norm_batch = pd.DataFrame(
        [{"date": "2024-01-01", "article_count": 1, "keyword_count": 1, "tone": -1.0}]
    )

    engine = create_engine(f"sqlite:///{db_path}")
    pd.DataFrame(
        [
            {
                "id": "old",
                "datetime": "2023-12-31 00:00:00",
                "source": "old",
                "url": "https://old",
                "title": "Old",
                "raw_record_json": "{}",
                "pulled_at": "2024-01-01 00:00:00",
            }
        ]
    ).to_sql("news_raw", engine, if_exists="replace", index=False)

    monkeypatch.setattr(update_news, "init_db", lambda: None)
    monkeypatch.setattr(update_news, "get_engine", lambda: engine)

    def fake_write_dataframe(df: pd.DataFrame, table_name: str, replace: bool = False):
        mode = "append"
        if replace:
            mode = "replace"
        df.to_sql(table_name, engine, if_exists=mode, index=False)

    test_settings = type(
        "Settings", (), {"cache_dir": cache_dir, "data_dir": tmp_path, "openai_api_key": None}
    )()
    monkeypatch.setattr(update_news, "settings", test_settings)
    monkeypatch.setattr(update_news, "write_dataframe", fake_write_dataframe)
    monkeypatch.setattr(
        update_news,
        "GdeltAdapter",
        lambda _: DummyAdapter(
            cache_dir,
            raw_batch,
            norm_batch,
            True,
            fallback_used=True,
            degraded_mode_used=False,
        ),
    )

    update_news.run()

    written = pd.read_sql("SELECT id, datetime FROM news_raw ORDER BY datetime", engine)
    assert written["id"].tolist() == ["old", "a", "b"]
    assert written["id"].is_unique

    norm_sql = pd.read_sql("SELECT * FROM news_normalized", engine)
    assert len(norm_sql) == 1
    assert (cache_dir / "news_normalized.parquet").exists()

    runlog_files = sorted(runlogs.glob("update_news_*.json"))
    assert runlog_files
    runlog = json.loads(runlog_files[-1].read_text(encoding="utf-8"))
    assert runlog["raw_rows"] == 2
    assert runlog["cache_hit"] is True
    assert runlog["fallback_used"] is True
    assert runlog["degraded_mode_used"] is False


def test_update_news_fetch_exception_writes_degraded_outputs(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "test.db"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    engine = create_engine(f"sqlite:///{db_path}")

    monkeypatch.setattr(update_news, "init_db", lambda: None)
    monkeypatch.setattr(update_news, "get_engine", lambda: engine)

    def fake_write_dataframe(df: pd.DataFrame, table_name: str, replace: bool = False):
        df.to_sql(table_name, engine, if_exists="replace" if replace else "append", index=False)

    test_settings = type(
        "Settings", (), {"cache_dir": cache_dir, "data_dir": tmp_path, "openai_api_key": None}
    )()
    monkeypatch.setattr(update_news, "settings", test_settings)
    monkeypatch.setattr(update_news, "write_dataframe", fake_write_dataframe)
    monkeypatch.setenv("LOOKBACK_DAYS", "3")
    monkeypatch.setattr(
        update_news,
        "GdeltAdapter",
        lambda _: DummyAdapter(
            cache_dir,
            pd.DataFrame(),
            pd.DataFrame(),
            False,
            raise_exc=RuntimeError("fail"),
        ),
    )

    update_news.run()

    norm_sql = pd.read_sql("SELECT * FROM news_normalized", engine)
    assert len(norm_sql) == 3
    assert norm_sql["article_count"].sum() == 0
    assert (cache_dir / "news_normalized.parquet").exists()
