from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from oil_risk.adapters.gdelt_adapter import GdeltAdapter, GdeltQuery


class DummyResp:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
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
    os.environ["GDELT_CACHE_TTL_SECONDS"] = "1800"
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
    os.utime(cache_file, (recent, recent))
    records = adapter._fetch_api_records(query)
    assert records == [{"url": "https://cached"}]
    assert adapter.last_cache_hit is True

    stale = datetime.now().timestamp() - 3600
    os.utime(cache_file, (stale, stale))
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
