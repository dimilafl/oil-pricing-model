from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from oil_risk.db.io import read_sql
from oil_risk.logging_utils import setup_logging

REQUIRED_SCALARS = [
    "oil_return",
    "VIX_z_63",
    "OVX_z_63",
    "news_risk_score",
    "put_call_ratio_mean",
]


def _parse_json(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        return json.loads(value)
    return {}


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _format_scalar(name: str, value: float | None) -> str:
    rendered = "null" if value is None else f"{value:.6f}"
    return f"- {name}: {rendered}"


def _load_latest_scalars(latest_date: str) -> dict[str, float | None]:
    market = read_sql(
        "SELECT feature_name, feature_value FROM market_features "
        f"WHERE date='{latest_date}' AND feature_name IN ('oil_return','VIX_z_63')"
    )
    news = read_sql(
        "SELECT feature_name, feature_value FROM news_features "
        f"WHERE date='{latest_date}' AND feature_name='geopolitical_risk_score'"
    )
    options = read_sql(
        "SELECT feature_name, feature_value FROM options_features "
        f"WHERE date='{latest_date}' AND feature_name IN ('OVX_z_63','put_call_ratio_mean')"
    )

    scalars: dict[str, float | None] = {
        "oil_return": None,
        "VIX_z_63": None,
        "OVX_z_63": None,
        "news_risk_score": None,
        "put_call_ratio_mean": None,
    }

    for _, row in market.iterrows():
        name = str(row["feature_name"])
        if name in scalars:
            scalars[name] = _safe_float(row["feature_value"])
    for _, row in options.iterrows():
        name = str(row["feature_name"])
        if name in scalars:
            scalars[name] = _safe_float(row["feature_value"])
    if not news.empty:
        scalars["news_risk_score"] = _safe_float(news.iloc[0]["feature_value"])

    return scalars


def _latest_report_paths() -> dict[str, str | None]:
    reports = read_sql("SELECT path FROM reports ORDER BY created_at DESC")
    daily = None
    eval_report = None
    if not reports.empty:
        for path in reports["path"].astype(str):
            if daily is None and Path(path).name.startswith("report_"):
                daily = path
            if eval_report is None and Path(path).name.startswith("eval_"):
                eval_report = path
            if daily and eval_report:
                break
    return {"daily_report": daily, "eval_report": eval_report}


def run() -> Path:
    setup_logging()
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    alerts_path = out_dir / "alerts.json"
    alerts: dict[str, object] = {}
    if alerts_path.exists():
        alerts = json.loads(alerts_path.read_text(encoding="utf-8"))

    state = read_sql("SELECT * FROM model_state ORDER BY date DESC LIMIT 1")
    signals = read_sql("SELECT * FROM signals ORDER BY date DESC")

    latest_date = None
    regime_label = "unknown"
    regime_probs: dict[str, object] = {}
    if not state.empty:
        latest_date = str(state.iloc[0]["date"])
        regime_label = str(state.iloc[0]["state_label"])
        regime_probs = _parse_json(state.iloc[0]["state_probabilities_json"])
    elif not signals.empty:
        latest_date = str(signals.iloc[0]["date"])

    triggered_signal_names: list[str] = []
    if latest_date is not None and not signals.empty:
        latest_signals = signals[signals["date"].astype(str) == latest_date].copy()
        if not latest_signals.empty:
            latest_signals["signal_value"] = latest_signals["signal_value"].astype(float)
            triggered_signal_names = [
                str(name)
                for name in latest_signals[latest_signals["signal_value"] == 1.0][
                    "signal_name"
                ].tolist()
            ]

    scalars = _load_latest_scalars(latest_date) if latest_date is not None else {}
    report_paths = _latest_report_paths()

    raw_lines = [
        f"date: {latest_date}",
        f"regime_state: {regime_label}",
        f"state_probabilities: {json.dumps(regime_probs, sort_keys=True)}",
        f"triggered_signals: {', '.join(triggered_signal_names) if triggered_signal_names else 'none'}",
        "key_scalars:",
    ]
    for scalar in REQUIRED_SCALARS:
        raw_lines.append(_format_scalar(scalar, scalars.get(scalar)))
    raw_lines.append(f"daily_report_path: {report_paths['daily_report']}")

    payload = {
        "id": f"oil-risk:{latest_date}",
        "source": "oil-risk",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "raw_content": "\n".join(raw_lines),
        "alerts": alerts,
        "report_paths": report_paths,
    }

    out_path = out_dir / "signal_engine_signal.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def main() -> None:
    run()


if __name__ == "__main__":
    main()
