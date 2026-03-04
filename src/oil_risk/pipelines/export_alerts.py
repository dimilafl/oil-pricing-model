from __future__ import annotations

import json
from pathlib import Path

from oil_risk.db.io import read_sql
from oil_risk.logging_utils import setup_logging
from oil_risk.pipelines.build_evidence_pack import run as build_evidence_pack


def _parse_json(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        return json.loads(value)
    return {}


def _latest_news_components(latest_date: str) -> dict[str, object]:
    news = read_sql(
        "SELECT feature_name, feature_value FROM news_features "
        f"WHERE date='{latest_date}' ORDER BY feature_name"
    )
    if news.empty:
        return {"news_risk_score": None, "components": {}}
    component_map = {row["feature_name"]: float(row["feature_value"]) for _, row in news.iterrows()}
    return {
        "news_risk_score": component_map.get("geopolitical_risk_score"),
        "components": component_map,
    }


def run() -> Path:
    setup_logging()
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    state = read_sql("SELECT * FROM model_state ORDER BY date DESC LIMIT 1")
    sig = read_sql("SELECT * FROM signals ORDER BY date DESC")
    tail = read_sql(
        "SELECT date, tail_risk_prob FROM tail_risk_predictions ORDER BY date DESC LIMIT 1"
    )

    latest_date = None
    latest_state: dict[str, object] = {}
    if not state.empty:
        latest_date = str(state.iloc[0]["date"])
        latest_state = {
            "state_id": int(state.iloc[0]["state_id"]),
            "state_label": state.iloc[0]["state_label"],
            "state_probabilities": _parse_json(state.iloc[0]["state_probabilities_json"]),
        }
    elif not sig.empty:
        latest_date = str(sig.iloc[0]["date"])

    triggered_signals: list[dict[str, object]] = []
    if latest_date is not None:
        latest_sig = sig[sig["date"].astype(str) == latest_date].copy()
        if not latest_sig.empty:
            latest_sig["signal_value"] = latest_sig["signal_value"].astype(float)
            for _, row in latest_sig[latest_sig["signal_value"] == 1.0].iterrows():
                triggered_signals.append(
                    {
                        "signal_name": row["signal_name"],
                        "signal_value": float(row["signal_value"]),
                        "metadata": _parse_json(row["metadata_json"]),
                    }
                )

    news_snapshot = (
        _latest_news_components(latest_date)
        if latest_date is not None
        else {
            "news_risk_score": None,
            "components": {},
        }
    )

    evidence_pack_path = None
    if latest_date is not None:
        _, evidence_md = build_evidence_pack(latest_date=latest_date)
        evidence_pack_path = str(evidence_md)

    tail_prob = None
    if not tail.empty:
        tail_prob = float(tail.iloc[0]["tail_risk_prob"])

    alerts = {
        "latest_date": latest_date,
        "state": latest_state,
        "triggered_signals": triggered_signals,
        "news": news_snapshot,
        "tail_risk_prob": tail_prob,
        "evidence_pack_path": evidence_pack_path,
    }

    out_path = out_dir / "alerts.json"
    out_path.write_text(json.dumps(alerts, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def main() -> None:
    run()


if __name__ == "__main__":
    main()
