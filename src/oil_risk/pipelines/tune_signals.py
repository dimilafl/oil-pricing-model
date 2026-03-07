from __future__ import annotations

import argparse
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from oil_risk.db.io import write_dataframe
from oil_risk.logging_utils import setup_logging
from oil_risk.pipelines.data_views import load_feature_frame
from oil_risk.signals_config import load_signals_config


def _forward_5d_abs_return(series: pd.Series) -> pd.Series:
    return series.shift(-1).rolling(5).sum().shift(-4).abs()


def _score_frame(df: pd.DataFrame, max_trigger_rate: float) -> tuple[float, float]:
    trigger_rate = float(df["triggered"].mean()) if len(df) else 0.0
    if trigger_rate == 0.0 or trigger_rate > max_trigger_rate:
        return float("-inf"), trigger_rate
    trig = df[df["triggered"]]["fwd_5d_abs"]
    non = df[~df["triggered"]]["fwd_5d_abs"]
    if trig.empty or non.empty:
        return float("-inf"), trigger_rate
    return float(trig.mean() - non.mean()), trigger_rate


def run(apply_best: bool = False, metric_name: str = "separation") -> Path:
    setup_logging()
    cfg = load_signals_config()
    max_trigger_rate = float(cfg.get("tuning", {}).get("max_trigger_rate", 0.2))

    frame = load_feature_frame()
    if frame.empty or "oil_return" not in frame.columns:
        raise ValueError("No features for tuning")

    work = frame.copy()
    work["fwd_5d_abs"] = _forward_5d_abs_return(work["oil_return"])
    work = work.dropna(subset=["fwd_5d_abs", "OVX_z_63", "VIX_z_63", "oil_return"]).copy()

    grid = {
        "risk_premium_alert.ovx_z_min": [0.5, 1.0, 1.5],
        "risk_premium_alert.news_risk_min": [0.5, 1.0, 1.5],
        "macro_stress_alert.vix_z_min": [0.5, 1.0, 1.5],
        "macro_stress_alert.oil_return_min": [-0.03, -0.02, -0.01],
        "correlation_break_alert.corr_min": [-0.5, -0.3, -0.1],
        "hedging_pressure_alert.vix_z_min": [0.5, 1.0, 1.5],
        "tail_risk_alert.tail_risk_prob_min": [0.4, 0.5, 0.6],
    }

    rows: list[dict[str, object]] = []
    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values, strict=True))
        triggered = (
            (
                (work["OVX_z_63"] > params["risk_premium_alert.ovx_z_min"])
                & (
                    work.get("geopolitical_risk_score", 0.0)
                    > params["risk_premium_alert.news_risk_min"]
                )
            )
            | (
                (work["VIX_z_63"] > params["macro_stress_alert.vix_z_min"])
                & (work["oil_return"] < params["macro_stress_alert.oil_return_min"])
            )
            | (work.get("oil_spx_corr_63", 0.0) < params["correlation_break_alert.corr_min"])
            | (
                (work["VIX_z_63"] > params["hedging_pressure_alert.vix_z_min"])
                & (work.get("unusual_put_activity", 0.0) > 0.0)
            )
            | (work.get("tail_risk_prob", 0.0) >= params["tail_risk_alert.tail_risk_prob_min"])
        )
        score_frame = work[["fwd_5d_abs"]].copy()
        score_frame["triggered"] = triggered
        score, trigger_rate = _score_frame(score_frame, max_trigger_rate)
        rows.append({"score": score, "trigger_rate": trigger_rate, "params": params})

    leaderboard = pd.DataFrame(rows).sort_values("score", ascending=False)
    best = leaderboard.iloc[0]

    run_id = uuid4().hex[:12]
    created_at = datetime.now(UTC).isoformat()
    write_dataframe(
        pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "created_at": created_at,
                    "metric_name": metric_name,
                    "best_params_json": json.dumps(best["params"]),
                    "leaderboard_json": json.dumps(leaderboard.head(25).to_dict(orient="records")),
                }
            ]
        ),
        "tuning_runs",
        replace=False,
    )

    if apply_best:
        cfg_out = dict(cfg)
        for key, value in best["params"].items():
            section, name = key.split(".", maxsplit=1)
            cfg_out.setdefault(section, {})[name] = value
        Path("configs").mkdir(parents=True, exist_ok=True)
        Path("configs/signals.json").write_text(json.dumps(cfg_out, indent=2), encoding="utf-8")

    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"tuning_{datetime.now(UTC).date().isoformat()}.md"
    report_lines = [
        f"# Signal Tuning Report ({datetime.now(UTC).date().isoformat()})",
        "",
        f"- run_id: {run_id}",
        f"- metric: {metric_name}",
        f"- max_trigger_rate: {max_trigger_rate}",
        "",
        "## Best parameters",
        "```json",
        json.dumps(best["params"], indent=2),
        "```",
        "",
        "## Leaderboard (top 10)",
    ]
    for _, row in leaderboard.head(10).iterrows():
        report_line = (
            f"- score={row['score']:.6f}, trigger_rate={row['trigger_rate']:.4f}, "
            f"params={row['params']}"
        )
        report_lines.append(report_line)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply-best", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(apply_best=args.apply_best)


if __name__ == "__main__":
    main()
