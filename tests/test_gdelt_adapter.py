from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from oil_risk.adapters.gdelt_adapter import GdeltAdapter, GdeltQuery


class DummyResp:
    def __init__(
        self,
        payload: dict | None = None,
        *,
        status_code: int = 200,
        headers: dict | None = None,
        text: str = "",
        content: bytes = b"",
    ):
        self._payload = payload or {}
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400 and self.status_code != 429:
            raise requests.HTTPError(f"status {self.status_code}")
        return None

    def json(self):
        return self._payload


def test_gdelt_api_pagination_and_dedup(monkeypatch, tmp_path: Path):
    pages = [
        {
            "articles": [
                {
                    "url": "https://a",
                    "title": "Iran tanker update",
                    "seendate": "20240101000000",
                    "sourcecountry": "US",
                    "tone": "-3",
                },
                {
                    "url": "https://b",
                    "title": "Hormuz shipping risk",
                    "seendate": "20240101010000",
                    "sourcecountry": "US",
                    "tone": "-2",
                },
            ]
        },
        {
            "articles": [
                {
                    "url": "https://b",
                    "title": "Hormuz shipping risk",
                    "seendate": "20240101010000",
                    "sourcecountry": "US",
                    "tone": "-2",
                },
                {
                    "url": "https://c",
                    "title": "Sanctions and oil export",
                    "seendate": "20240101020000",
                    "sourcecountry": "US",
                    "tone": "-1",
                },
            ]
        },
        {"articles": []},
    ]
    seen_urls: list[str] = []

    def fake_get(url: str, timeout: int = 60):
        seen_urls.append(url)
        return DummyResp(pages.pop(0))

    monkeypatch.setattr("oil_risk.adapters.gdelt_adapter.requests.get", fake_get)
    monkeypatch.setattr(
        "oil_risk.adapters.gdelt_adapter.datetime",
        type(
            "FixedDateTime",
            (datetime,),
            {
                "now": classmethod(lambda cls, tz=None: datetime(2024, 1, 2, tzinfo=UTC)),
                "strptime": datetime.strptime,
            },
        ),
    )
    adapter = GdeltAdapter(tmp_path)
    raw, norm = adapter.fetch_and_parse(GdeltQuery(days=10, max_records=10, page_size=2))
    assert len(raw) == 3
    assert set(raw["url"]) == {"https://a", "https://b", "https://c"}
    assert len(norm) == 3
    assert "startdatetime=20240101010001" in seen_urls[1]


def test_gdelt_api_cache_ttl(monkeypatch, tmp_path: Path):
    adapter = GdeltAdapter(tmp_path)
    query = GdeltQuery(days=1, max_records=5, page_size=5)
    cache_file = (
        tmp_path / adapter._cache_key(query, datetime(2024, 1, 2, tzinfo=UTC))
    ).with_suffix(".json")
    cache_file.write_text(json.dumps([{"url": "https://cached"}]), encoding="utf-8")

    calls = {"n": 0}

    def fake_get(url: str, timeout: int = 60):
        calls["n"] += 1
        return DummyResp({"articles": [{"url": "https://fresh", "seendate": "20240102000000"}]})

    monkeypatch.setattr("oil_risk.adapters.gdelt_adapter.requests.get", fake_get)
    monkeypatch.setattr(
        "oil_risk.adapters.gdelt_adapter.datetime",
        type(
            "FixedDateTime",
            (datetime,),
            {
                "now": classmethod(lambda cls, tz=None: datetime(2024, 1, 2, tzinfo=UTC)),
                "strptime": datetime.strptime,
            },
        ),
    )

    recent = datetime.now().timestamp() - 100
    monkeypatch.setenv("GDELT_CACHE_TTL_SECONDS", "1800")
    os_utime = __import__("os").utime
    os_utime(cache_file, (recent, recent))
    records = adapter._fetch_api_records(query)
    assert records == [{"url": "https://cached"}]
    assert adapter.last_cache_hit is True

    stale = datetime.now().timestamp() - 3600
    os_utime(cache_file, (stale, stale))
    records = adapter._fetch_api_records(query)
    assert records == [{"url": "https://fresh", "seendate": "20240102000000"}]
    assert calls["n"] == 1
    assert adapter.last_cache_hit is False


def test_gdelt_api_lookback_to_daily_buckets(monkeypatch, tmp_path: Path):
    payload = {
        "articles": [
            {
                "url": "https://in-window-1",
                "title": "Iran tanker",
                "seendate": "20240102000000",
                "sourcecountry": "US",
                "tone": "-1",
            },
            {
                "url": "https://in-window-2",
                "title": "Hormuz blockade",
                "seendate": "20240102120000",
                "sourcecountry": "US",
                "tone": "-2",
            },
            {
                "url": "https://old",
                "title": "Iran old story",
                "seendate": "20231201000000",
                "sourcecountry": "US",
                "tone": "-3",
            },
        ]
    }

    def fake_get(url: str, timeout: int = 60):
        return DummyResp(payload)

    monkeypatch.setattr("oil_risk.adapters.gdelt_adapter.requests.get", fake_get)
    monkeypatch.setattr(
        "oil_risk.adapters.gdelt_adapter.datetime",
        type(
            "FixedDateTime",
            (datetime,),
            {
                "now": classmethod(lambda cls, tz=None: datetime(2024, 1, 3, tzinfo=UTC)),
                "strptime": datetime.strptime,
            },
        ),
    )
    adapter = GdeltAdapter(tmp_path)
    _, norm = adapter.fetch_and_parse(GdeltQuery(days=2, max_records=10, page_size=10))
    assert norm["date"].nunique() == 1
    assert str(norm.iloc[0]["date"]) == "2024-01-02"
    assert norm["article_count"].sum() == 2


def test_gdelt_cache_key_includes_end_date(tmp_path: Path):
    adapter = GdeltAdapter(tmp_path)
    query = GdeltQuery(days=7, max_records=100)
    key = adapter._cache_key(query, datetime(2024, 1, 5, tzinfo=UTC))
    assert key.endswith("_20240105")


def test_429_with_retry_after_then_success(monkeypatch, tmp_path: Path):
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_get(url: str, timeout: int = 60):
        calls["n"] += 1
        if calls["n"] == 1:
            return DummyResp(status_code=429, headers={"Retry-After": "1"})
        return DummyResp(
            {
                "articles": [
                    {
                        "url": "https://ok",
                        "title": "Iran tanker",
                        "seendate": "20240102000000",
                        "sourcecountry": "US",
                        "tone": "-1",
                    }
                ]
            }
        )

    monkeypatch.setattr("oil_risk.adapters.gdelt_adapter.requests.get", fake_get)
    monkeypatch.setattr("oil_risk.adapters.gdelt_adapter.time.sleep", lambda x: sleeps.append(x))
    monkeypatch.setattr(
        "oil_risk.adapters.gdelt_adapter.datetime",
        type(
            "FixedDateTime",
            (datetime,),
            {
                "now": classmethod(lambda cls, tz=None: datetime(2024, 1, 3, tzinfo=UTC)),
                "strptime": datetime.strptime,
            },
        ),
    )

    adapter = GdeltAdapter(tmp_path)
    raw_df, norm_df = adapter.fetch_and_parse(GdeltQuery(days=5, max_records=5, page_size=5))
    assert not raw_df.empty
    assert not norm_df.empty
    assert sleeps == [1.0]
    assert adapter.fallback_used is False
    assert adapter.degraded_mode_used is False


def _create_legacy_zip(path: Path) -> Path:
    row = [
        "DOC1",
        "20240102000000",
        "",
        "",
        "US",
        "https://legacy",
        "Iran shipping update",
        "IRAN;",
        "",
        "IRAN;",
        "",
        "",
        "",
        "",
        "",
        "-2,0,0,0",
    ]
    inner = "sample.gkg.csv"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(inner, "\t".join(row) + "\n")
    return path


def test_persistent_429_triggers_legacy_fallback(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GDELT_MAX_ATTEMPTS", "2")

    def always_429(url: str, timeout: int = 60):
        return DummyResp(status_code=429, headers={"Retry-After": "0"})

    monkeypatch.setattr("oil_risk.adapters.gdelt_adapter.requests.get", always_429)
    monkeypatch.setattr("oil_risk.adapters.gdelt_adapter.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "oil_risk.adapters.gdelt_adapter.datetime",
        type(
            "FixedDateTime",
            (datetime,),
            {
                "now": classmethod(lambda cls, tz=None: datetime(2024, 1, 3, tzinfo=UTC)),
                "strptime": datetime.strptime,
            },
        ),
    )

    zip_path = _create_legacy_zip(tmp_path / "legacy.zip")
    adapter = GdeltAdapter(tmp_path)
    monkeypatch.setattr(adapter, "_download_gkg_files", lambda query: [zip_path])

    raw_df, norm_df = adapter.fetch_and_parse(GdeltQuery(days=5, max_records=5, page_size=5))
    assert not raw_df.empty
    assert not norm_df.empty
    assert adapter.fallback_used is True
    assert adapter.degraded_mode_used is False


def test_doc_and_legacy_fail_triggers_degraded_mode(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("LOOKBACK_DAYS", "4")

    def always_429(url: str, timeout: int = 60):
        return DummyResp(status_code=429, headers={"Retry-After": "0"})

    monkeypatch.setattr("oil_risk.adapters.gdelt_adapter.requests.get", always_429)
    monkeypatch.setattr("oil_risk.adapters.gdelt_adapter.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "oil_risk.adapters.gdelt_adapter.datetime",
        type(
            "FixedDateTime",
            (datetime,),
            {
                "now": classmethod(lambda cls, tz=None: datetime(2024, 1, 5, tzinfo=UTC)),
                "strptime": datetime.strptime,
            },
        ),
    )

    adapter = GdeltAdapter(tmp_path)
    monkeypatch.setattr(adapter, "_download_gkg_files", lambda query: (_ for _ in ()).throw(RuntimeError("down")))

    raw_df, norm_df = adapter.fetch_and_parse(GdeltQuery(days=4, max_records=5, page_size=5))
    assert raw_df.empty
    assert len(norm_df) == 4
    assert norm_df["article_count"].sum() == 0
    assert norm_df["keyword_count"].sum() == 0
    assert norm_df["tone"].isna().all()
    assert adapter.degraded_mode_used is True
