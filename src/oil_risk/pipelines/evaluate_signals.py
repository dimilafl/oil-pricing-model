from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from oil_risk.config import settings
from oil_risk.db.io import read_sql, write_dataframe
from oil_risk.logging_utils import setup_logging


def _forward_return_sums(returns: pd.Series, horizon: int) -> pd.Series:
    return returns.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))


def _summarize_triggered(signals_with_returns: pd.DataFrame) -> pd.DataFrame:
    triggered = signals_with_returns[signals_with_returns["signal_value"] == 1.0].copy()
    if triggered.empty:
        return pd.DataFrame(
            columns=[
                "signal_name",
                "count",
                "mean_fwd_1d",
                "mean_fwd_5d",
                "mean_fwd_10d",
                "median_fwd_1d",
                "median_fwd_5d",
                "median_fwd_10d",
            ]
        )
    grouped = triggered.groupby("signal_name").agg(
        count=("date", "count"),
        mean_fwd_1d=("fwd_1d", "mean"),
        mean_fwd_5d=("fwd_5d", "mean"),
        mean_fwd_10d=("fwd_10d", "mean"),
        median_fwd_1d=("fwd_1d", "median"),
        median_fwd_5d=("fwd_5d", "median"),
        median_fwd_10d=("fwd_10d", "median"),
    )
    return grouped.reset_index()


def _summarize_by_state(model_states: pd.DataFrame) -> pd.DataFrame:
    if model_states.empty:
        return pd.DataFrame(
            columns=["state_id", "count", "mean_fwd_1d", "mean_fwd_5d", "mean_fwd_10d"]
        )
    grouped = model_states.groupby("state_id").agg(
        count=("date", "count"),
        mean_fwd_1d=("fwd_1d", "mean"),
        mean_fwd_5d=("fwd_5d", "mean"),
        mean_fwd_10d=("fwd_10d", "mean"),
    )
    return grouped.reset_index()


def _summarize_state_signal(signals_with_state: pd.DataFrame) -> pd.DataFrame:
    triggered = signals_with_state[signals_with_state["signal_value"] == 1.0].copy()
    if triggered.empty:
        return pd.DataFrame(
            columns=[
                "state_id",
                "signal_name",
                "count",
                "mean_fwd_1d",
                "mean_fwd_5d",
                "mean_fwd_10d",
            ]
        )
    grouped = triggered.groupby(["state_id", "signal_name"]).agg(
        count=("date", "count"),
        mean_fwd_1d=("fwd_1d", "mean"),
        mean_fwd_5d=("fwd_5d", "mean"),
        mean_fwd_10d=("fwd_10d", "mean"),
    )
    return grouped.reset_index()


def _as_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    divider = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.6f}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, divider, *rows])


def run() -> Path:
    setup_logging()
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    mkt = read_sql(
        "SELECT date, feature_value AS oil_return FROM market_features WHERE feature_name='oil_return'"
    )
    states = read_sql("SELECT date, state_id, state_label FROM model_state")
    sig = read_sql("SELECT date, signal_name, signal_value FROM signals")

    if mkt.empty:
        raise ValueError("market_features is empty; run feature pipeline first")

    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt = mkt.sort_values("date")
    mkt["fwd_1d"] = _forward_return_sums(mkt["oil_return"], 1)
    mkt["fwd_5d"] = _forward_return_sums(mkt["oil_return"], 5)
    mkt["fwd_10d"] = _forward_return_sums(mkt["oil_return"], 10)

    states["date"] = pd.to_datetime(states["date"])
    sig["date"] = pd.to_datetime(sig["date"])
    sig["signal_value"] = sig["signal_value"].astype(float)

    state_with_returns = states.merge(
        mkt[["date", "fwd_1d", "fwd_5d", "fwd_10d"]], on="date", how="left"
    )
    signals_with_returns = sig.merge(
        mkt[["date", "fwd_1d", "fwd_5d", "fwd_10d"]], on="date", how="left"
    )
    signals_with_state = signals_with_returns.merge(
        states[["date", "state_id"]], on="date", how="left"
    )

    by_signal = _summarize_triggered(signals_with_returns)
    by_state = _summarize_by_state(state_with_returns)
    by_state_signal = _summarize_state_signal(signals_with_state)

    latest_date = mkt["date"].max().date()
    created_at = datetime.now(UTC).isoformat()
    eval_records = []
    for eval_name, df in {
        "triggered_signal_summary": by_signal,
        "state_summary": by_state,
        "state_signal_triggered_summary": by_state_signal,
    }.items():
        eval_records.append(
            {
                "date": latest_date,
                "eval_name": eval_name,
                "eval_json": json.dumps(df.to_dict(orient="records")),
                "created_at": created_at,
            }
        )
    write_dataframe(pd.DataFrame(eval_records), "signal_eval", replace=False)

    report_lines = [
        f"# Signal Evaluation ({latest_date.isoformat()})",
        "",
        "## Triggered signal forward-return summary",
        _as_markdown_table(by_signal),
        "",
        "## State forward-return summary",
        _as_markdown_table(by_state),
        "",
        "## Triggered signal summary by state",
        _as_markdown_table(by_state_signal),
    ]
    eval_report = settings.reports_dir / f"eval_{latest_date.isoformat()}.md"
    eval_report.write_text("\n".join(report_lines), encoding="utf-8")

    write_dataframe(
        pd.DataFrame(
            [
                {
                    "date": latest_date,
                    "path": str(eval_report),
                    "created_at": created_at,
                }
            ]
        ),
        "reports",
        replace=False,
    )
    return eval_report


def main() -> None:
    run()


if __name__ == "__main__":
    main()
