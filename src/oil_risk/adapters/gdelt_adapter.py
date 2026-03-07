from __future__ import annotations

import csv
import io
import json
import logging
import os
import time
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from json import JSONDecodeError
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests

logger = logging.getLogger(__name__)
LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
KEYWORDS = [
    "iran",
    "hormuz",
    "strait of hormuz",
    "sanctions",
    "tanker",
    "shipping",
    "missile",
    "drone",
    "oil export",
    "blockade",
]


@dataclass
class GdeltQuery:
    days: int = int(os.getenv("LOOKBACK_DAYS", "90"))
    max_files: int = 24
    max_records: int = 1000
    page_size: int = 250
    use_legacy_lastupdate: bool = False


class GdeltFetchError(RuntimeError):
    """Raised when GDELT data cannot be fetched after retries."""


def _parse_lastupdate(text: str) -> list[str]:
    urls: list[str] = []
    for line in text.strip().splitlines():
        parts = line.split(" ")
        if parts and parts[-1].endswith(".gkg.csv.zip"):
            urls.append(parts[-1])
    return urls


def _split_semicolon_field(value: str) -> list[str]:
    if not value:
        return []
    return [x for x in value.split(";") if x]


def build_query_terms(extra_terms: Iterable[str] | None = None) -> list[str]:
    terms = list(KEYWORDS)
    if extra_terms:
        terms.extend(extra_terms)
    return sorted(set(terms))


class GdeltAdapter:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_cache_hit = False
        self.fallback_used = False
        self.degraded_mode_used = False

    def _cache_key(self, query: GdeltQuery, end_dt: datetime) -> str:
        return f"gdelt_api_{query.days}d_{query.max_records}r_{end_dt.strftime('%Y%m%d')}"

    def _get_json_with_retry(
        self, url: str, timeout: int = 60, max_attempts: int | None = None
    ) -> dict:
        attempts = max_attempts or int(os.getenv("GDELT_MAX_ATTEMPTS", "3"))
        max_backoff_seconds = float(os.getenv("GDELT_MAX_BACKOFF_SECONDS", "30"))
        delay_seconds = 1.0
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                resp = requests.get(url, timeout=timeout)
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    retry_wait = delay_seconds
                    if retry_after:
                        try:
                            retry_wait = max(float(retry_after), retry_wait)
                        except ValueError:
                            pass
                    last_error = GdeltFetchError("GDELT DOC API returned HTTP 429")
                    if attempt == attempts:
                        break
                    time.sleep(min(retry_wait, max_backoff_seconds))
                    delay_seconds = min(delay_seconds * 2, max_backoff_seconds)
                    continue

                resp.raise_for_status()
                if not resp.text.strip():
                    raise GdeltFetchError("GDELT DOC API returned empty body")
                return resp.json()
            except (JSONDecodeError, ValueError) as exc:
                last_error = exc
            except (requests.RequestException, GdeltFetchError) as exc:
                last_error = exc

            if attempt == attempts:
                break
            time.sleep(min(delay_seconds, max_backoff_seconds))
            delay_seconds = min(delay_seconds * 2, max_backoff_seconds)

        raise GdeltFetchError(
            f"Failed to fetch GDELT DOC API after {attempts} attempts"
        ) from last_error

    def _build_degraded_norm_df(self, days: int) -> pd.DataFrame:
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

    def _fetch_api_records(self, query: GdeltQuery) -> list[dict]:
        end_dt = datetime.now(UTC)
        start_dt = end_dt - timedelta(days=query.days)
        cache_ttl_seconds = int(os.getenv("GDELT_CACHE_TTL_SECONDS", "1800"))
        raw_cache = (self.cache_dir / self._cache_key(query, end_dt)).with_suffix(".json")

        if raw_cache.exists():
            cache_age_seconds = time.time() - raw_cache.stat().st_mtime
            if cache_age_seconds <= cache_ttl_seconds:
                self.last_cache_hit = True
                return json.loads(raw_cache.read_text(encoding="utf-8"))

        self.last_cache_hit = False
        records: list[dict] = []
        seen_ids: set[str] = set()
        cursor = start_dt
        while cursor <= end_dt and len(records) < query.max_records:
            page_size = min(query.page_size, query.max_records - len(records))
            params = {
                "query": quote_plus(" OR ".join(build_query_terms())),
                "mode": "ArtList",
                "format": "json",
                "sort": "DateAsc",
                "startdatetime": cursor.strftime("%Y%m%d%H%M%S"),
                "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
                "maxrecords": str(page_size),
            }
            url = DOC_API_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())
            payload = self._get_json_with_retry(url, timeout=60)
            articles = payload.get("articles", [])
            if not articles:
                break

            last_seen = articles[-1].get("seendate")
            for article in articles:
                doc_id = str(
                    article.get("url")
                    or article.get("id")
                    or article.get("seendate")
                    or article.get("title")
                )
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)
                records.append(article)
                if len(records) >= query.max_records:
                    break

            if not last_seen:
                break
            next_cursor = datetime.strptime(last_seen, "%Y%m%d%H%M%S").replace(
                tzinfo=UTC
            ) + timedelta(seconds=1)
            if next_cursor <= cursor:
                break
            cursor = next_cursor

        raw_cache.write_text(json.dumps(records), encoding="utf-8")
        return records

    def _api_to_frames(self, records: list[dict], days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        cutoff = datetime.now(UTC) - timedelta(days=days)
        raw_rows: list[dict] = []
        norm_rows: list[dict] = []
        pulled_at = datetime.now(UTC)

        for article in records:
            seendate = article.get("seendate")
            if not seendate:
                continue
            dt = datetime.strptime(seendate, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
            if dt < cutoff:
                continue
            title = article.get("title")
            url = article.get("url")
            source = article.get("sourcecountry") or article.get("domain")
            text_blob = " ".join([str(title or ""), str(url or "")]).lower()
            keyword_hits = sum(1 for kw in KEYWORDS if kw in text_blob)
            if keyword_hits == 0:
                continue
            tone_value = article.get("tone")
            doc_id = str(url or article.get("id") or article.get("seendate"))
            raw_rows.append(
                {
                    "id": doc_id,
                    "datetime": dt,
                    "source": source,
                    "url": url,
                    "title": title,
                    "raw_record_json": json.dumps(article),
                    "pulled_at": pulled_at,
                }
            )
            norm_rows.append(
                {
                    "date": dt.date(),
                    "article_count": 1,
                    "keyword_count": keyword_hits,
                    "tone": float(tone_value) if tone_value not in {None, ""} else None,
                    "source": source,
                    "title": title,
                    "url": url,
                }
            )

        raw_df = pd.DataFrame(raw_rows)
        if not raw_df.empty:
            raw_df = raw_df.drop_duplicates(subset=["id"], keep="last")
        norm_df = pd.DataFrame(norm_rows)
        return raw_df, norm_df

    def _download_gkg_files(self, query: GdeltQuery) -> list[Path]:
        last = requests.get(LASTUPDATE_URL, timeout=30)
        last.raise_for_status()
        urls = _parse_lastupdate(last.text)[: query.max_files]
        paths: list[Path] = []
        for url in urls:
            fname = url.split("/")[-1]
            target = self.cache_dir / fname
            if not target.exists():
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                target.write_bytes(resp.content)
            paths.append(target)
        return paths

    def fetch_and_parse(self, query: GdeltQuery | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        query = query or GdeltQuery()
        self.fallback_used = False
        self.degraded_mode_used = False

        if not query.use_legacy_lastupdate:
            try:
                records = self._fetch_api_records(query)
                return self._api_to_frames(records, query.days)
            except GdeltFetchError as exc:
                if os.getenv("GDELT_FALLBACK_TO_LEGACY_ON_429", "1") != "1":
                    raise
                logger.warning(
                    "DOC API failed, falling back to legacy lastupdate ingestion: %s", exc
                )
                self.fallback_used = True
                query = GdeltQuery(
                    days=query.days,
                    max_files=query.max_files,
                    max_records=query.max_records,
                    page_size=query.page_size,
                    use_legacy_lastupdate=True,
                )

        try:
            files = self._download_gkg_files(query)
            cutoff = datetime.now(UTC) - timedelta(days=query.days)
            raw_rows: list[dict] = []
            norm_rows: list[dict] = []

            for file in files:
                with zipfile.ZipFile(file) as zf:
                    inner = zf.namelist()[0]
                    with zf.open(inner) as fd:
                        stream = io.TextIOWrapper(fd, encoding="utf-8", errors="ignore")
                        for row in csv.reader(stream, delimiter="\t"):
                            if len(row) < 16:
                                continue
                            dt = datetime.strptime(row[1], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
                            if dt < cutoff:
                                continue
                            themes = _split_semicolon_field(row[7])
                            persons = _split_semicolon_field(row[11])
                            orgs = _split_semicolon_field(row[13])
                            locations = _split_semicolon_field(row[9])
                            tone = row[15].split(",")[0] if row[15] else None
                            src = row[4] if len(row) > 4 else None
                            doc_id = row[0]
                            title = row[6] if len(row) > 6 else None
                            url = row[5] if len(row) > 5 else None
                            text_blob = " ".join(
                                [title or "", " ".join(themes), " ".join(locations)]
                            ).lower()
                            keyword_hits = sum(1 for kw in KEYWORDS if kw in text_blob)
                            if keyword_hits == 0 and "IRAN" not in (row[7] or "").upper():
                                continue
                            raw_rows.append(
                                {
                                    "id": doc_id,
                                    "datetime": dt,
                                    "source": src,
                                    "url": url,
                                    "title": title,
                                    "raw_record_json": json.dumps({"row": row}),
                                    "pulled_at": datetime.now(UTC),
                                }
                            )
                            norm_rows.append(
                                {
                                    "date": dt.date(),
                                    "article_count": 1,
                                    "keyword_count": keyword_hits,
                                    "tone": float(tone) if tone not in {None, ""} else None,
                                    "themes": themes,
                                    "persons": persons,
                                    "organizations": orgs,
                                    "locations": locations,
                                    "source": src,
                                    "title": title,
                                    "url": url,
                                }
                            )

            raw_df = (
                pd.DataFrame(raw_rows).drop_duplicates(subset=["id"], keep="last")
                if raw_rows
                else pd.DataFrame()
            )
            norm_df = pd.DataFrame(norm_rows)
            if norm_df.empty:
                raise GdeltFetchError("Legacy lastupdate ingestion produced no usable rows")
            return raw_df, norm_df
        except Exception as exc:  # noqa: BLE001
            logger.warning("Legacy lastupdate ingestion failed; entering degraded mode: %s", exc)
            self.degraded_mode_used = True
            return pd.DataFrame(), self._build_degraded_norm_df(query.days)
