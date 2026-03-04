from __future__ import annotations

import csv
import io
import json
import logging
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)
LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
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
    days: int = 30
    max_files: int = 24


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


class GdeltAdapter:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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
                        rec = {
                            "id": doc_id,
                            "datetime": dt,
                            "source": src,
                            "url": url,
                            "title": title,
                            "raw_record_json": json.dumps({"row": row}),
                            "pulled_at": datetime.now(UTC),
                        }
                        raw_rows.append(rec)
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
            pd.DataFrame(raw_rows).drop_duplicates(subset=["id"]) if raw_rows else pd.DataFrame()
        )
        norm_df = pd.DataFrame(norm_rows)
        return raw_df, norm_df


def build_query_terms(extra_terms: Iterable[str] | None = None) -> list[str]:
    terms = list(KEYWORDS)
    if extra_terms:
        terms.extend(extra_terms)
    return sorted(set(terms))
