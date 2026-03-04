from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pandas as pd

from oil_risk.db.io import read_sql
from oil_risk.logging_utils import setup_logging
from oil_risk.pipelines.data_views import load_feature_frame


def _top_zscore_features(
    frame: pd.DataFrame, latest_date: pd.Timestamp, limit: int = 10
) -> list[dict[str, object]]:
    hist = frame.loc[:latest_date].copy()
    latest = hist.iloc[-1]
    z_rows: list[dict[str, object]] = []
    for column in hist.columns:
        series = hist[column].dropna()
        if len(series) < 5:
            continue
        std = float(series.std())
        if std == 0.0:
            continue
        z_score = float((latest.get(column) - series.mean()) / std)
        z_rows.append(
            {
                "feature_name": column,
                "feature_value": None if pd.isna(latest.get(column)) else float(latest.get(column)),
                "z_score": z_score,
                "abs_z_score": abs(z_score),
            }
        )
    return sorted(z_rows, key=lambda row: row["abs_z_score"], reverse=True)[:limit]


def _news_window(latest_date: pd.Timestamp) -> list[dict[str, object]]:
    start_ts = latest_date - timedelta(hours=48)
    query = (
        "SELECT r.id, r.datetime, r.source, r.url, r.title, l.category, l.intensity, l.summary, "
        "COALESCE(CAST(json_extract(r.raw_record_json, '$.keyword_count') AS FLOAT), 0.0) AS keyword_hits "
        "FROM news_raw r LEFT JOIN news_llm l ON r.id = l.id "
        f"WHERE r.datetime >= '{start_ts.isoformat()}' AND r.datetime <= '{latest_date.isoformat()}' "
        "ORDER BY keyword_hits DESC, l.intensity DESC"
    )
    news = read_sql(query)
    if news.empty:
        return []
    news["datetime"] = pd.to_datetime(news["datetime"], errors="coerce")
    out = []
    for _, row in news.head(25).iterrows():
        out.append(
            {
                "title": row.get("title"),
                "url": row.get("url"),
                "source": row.get("source"),
                "datetime": None
                if pd.isna(row.get("datetime"))
                else row.get("datetime").isoformat(),
                "keyword_hits": float(row.get("keyword_hits", 0.0) or 0.0),
                "category": row.get("category"),
                "intensity": None if pd.isna(row.get("intensity")) else int(row.get("intensity")),
                "summary": row.get("summary"),
            }
        )
    return out


def run(latest_date: str | None = None) -> tuple[Path, Path]:
    setup_logging()
    signals = read_sql("SELECT * FROM signals ORDER BY date DESC")
    if signals.empty:
        raise ValueError("No signals found for evidence pack")

    if latest_date is None:
        latest_date = str(signals.iloc[0]["date"])
    latest_ts = pd.to_datetime(latest_date)

    triggered_rows = signals[
        (signals["date"].astype(str) == latest_date)
        & (signals["signal_value"].astype(float) == 1.0)
    ]
    triggered = triggered_rows[["signal_name", "signal_value", "metadata_json"]].to_dict(
        orient="records"
    )

    tail = read_sql(
        "SELECT date, tail_risk_prob, model_name FROM tail_risk_predictions "
        f"WHERE date='{latest_date}' ORDER BY created_at DESC LIMIT 1"
    )
    tail_payload = None
    if not tail.empty:
        tail_payload = {
            "tail_risk_prob": float(tail.iloc[0]["tail_risk_prob"]),
            "model_name": str(tail.iloc[0]["model_name"]),
        }

    features = load_feature_frame()
    top_features = _top_zscore_features(features, latest_ts)
    top_news = _news_window(latest_ts)

    payload = {
        "latest_date": latest_date,
        "triggered_signals": triggered,
        "tail_risk": tail_payload,
        "top_feature_contributors": top_features,
        "top_news_items": top_news,
    }

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"evidence_{latest_date}.json"
    md_path = out_dir / f"evidence_{latest_date}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    md_lines = [f"# Evidence Pack ({latest_date})", "", "## Triggered signals"]
    if not triggered:
        md_lines.append("- No triggered signals")
    else:
        for row in triggered:
            md_lines.append(f"- {row['signal_name']}: {row['metadata_json']}")
    md_lines.extend(["", "## Tail risk probability"])
    md_lines.append(f"- {tail_payload}" if tail_payload else "- No tail risk probability available")
    md_lines.extend(["", "## Top feature contributors"])
    for item in top_features:
        md_lines.append(
            f"- {item['feature_name']}: value={item['feature_value']}, z_score={item['z_score']:.4f}"
        )
    md_lines.extend(["", "## Top news items"])
    if not top_news:
        md_lines.append("- No news items in the last 48h")
    else:
        for item in top_news:
            md_lines.append(
                f"- {item['datetime']} | {item['source']} | {item['title']} | "
                f"keyword_hits={item['keyword_hits']} | category={item['category']} | intensity={item['intensity']}"
            )
            if item["url"]:
                md_lines.append(f"  - {item['url']}")
            if item["summary"]:
                md_lines.append(f"  - {item['summary']}")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    run()


if __name__ == "__main__":
    main()
