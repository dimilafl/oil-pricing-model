from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from oil_risk.config import settings
from oil_risk.db.io import read_sql, write_dataframe
from oil_risk.logging_utils import setup_logging


def _latest_eval_rows(ev: pd.DataFrame, eval_name: str) -> list[dict]:
    if ev.empty:
        return []
    chunk = ev[ev["eval_name"] == eval_name]
    if chunk.empty:
        return []
    return json.loads(chunk.iloc[0]["eval_json"])


def run() -> Path:
    setup_logging()
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    mkt = read_sql("SELECT date, feature_name, feature_value FROM market_features")
    nws = read_sql("SELECT date, feature_name, feature_value FROM news_features")
    opt = read_sql("SELECT date, feature_name, feature_value FROM options_features")
    st = read_sql("SELECT * FROM model_state ORDER BY date DESC LIMIT 1")
    sg = read_sql("SELECT * FROM signals")
    ev = read_sql("SELECT * FROM signal_eval ORDER BY created_at DESC")
    tail = read_sql(
        "SELECT date, tail_risk_prob, model_name FROM tail_risk_predictions ORDER BY date DESC"
    )

    mkt["date"] = pd.to_datetime(mkt["date"])
    nws["date"] = pd.to_datetime(nws["date"])
    if not opt.empty:
        opt["date"] = pd.to_datetime(opt["date"])
    latest_date = mkt["date"].max().date()
    latest_mkt = mkt[mkt["date"].dt.date == latest_date]
    latest_news = nws[nws["date"].dt.date == latest_date]
    latest_opt = opt[opt["date"].dt.date == latest_date] if not opt.empty else pd.DataFrame()
    latest_signals = sg[sg["date"] == latest_date.isoformat()]

    lines = [
        f"# Oil Risk Daily Report ({latest_date.isoformat()})",
        "",
        "## Latest market snapshot",
    ]
    for _, row in latest_mkt.iterrows():
        lines.append(f"- {row['feature_name']}: {row['feature_value']:.4f}")
    lines.extend(["", "## Hedging proxy snapshot"])
    if latest_opt.empty:
        lines.append("- No options feature data available")
    else:
        for _, row in latest_opt.iterrows():
            lines.append(f"- {row['feature_name']}: {row['feature_value']:.4f}")
    lines.extend(["", "## Latest news snapshot"])
    for _, row in latest_news.iterrows():
        lines.append(f"- {row['feature_name']}: {row['feature_value']:.4f}")
    lines.extend(["", "## LLM category counts"])
    cat_rows = latest_news[latest_news["feature_name"].str.startswith("category_count_")]
    if cat_rows.empty:
        lines.append("- No LLM category count features available")
    else:
        for _, row in cat_rows.iterrows():
            lines.append(f"- {row['feature_name']}: {row['feature_value']:.0f}")
    if not st.empty:
        lines.extend(["", "## Regime state"])
        state_line = (
            f"- state_id: {st.iloc[0]['state_id']}, label: {st.iloc[0]['state_label']}, "
            f"probs: {st.iloc[0]['state_probabilities_json']}"
        )
        lines.append(state_line)
    lines.extend(["", "## Signals"])
    for _, row in latest_signals.iterrows():
        lines.append(
            f"- {row['signal_name']}: {bool(row['signal_value'])}, details: {row['metadata_json']}"
        )

    if not tail.empty:
        latest_tail = tail[tail["date"].astype(str) == latest_date.isoformat()]
        if not latest_tail.empty:
            row = latest_tail.iloc[0]
            lines.extend(["", "## Tail risk probability"])
            lines.append(
                f"- tail_risk_prob: {float(row['tail_risk_prob']):.4f} (model: {row['model_name']})"
            )

    evidence_path = Path("artifacts") / f"evidence_{latest_date.isoformat()}.md"
    lines.extend(["", "## Evidence pack"])
    if evidence_path.exists():
        lines.append(f"- {evidence_path}")
    else:
        lines.append("- Not available")

    if not ev.empty:
        latest_eval = ev.iloc[0]
        lines.extend(["", "## Evaluation snapshot"])
        lines.append(
            f"- {latest_eval['eval_name']} ({latest_eval['date']}): {latest_eval['eval_json']}"
        )

        lag_rows = _latest_eval_rows(ev, "lag_effect_summary")
        overreaction_rows = _latest_eval_rows(ev, "overreaction_fade_summary")
        lines.extend(["", "## Lag and overreaction diagnostics"])
        if lag_rows:
            lines.append("### Lag effect (latest)")
            for row in lag_rows:
                lines.append(
                    "- "
                    f"{row['lag_bin']}: n={int(row['count'])}, "
                    f"mean(fwd_1d)={float(row['mean_fwd_1d']):.4f}, "
                    f"mean(fwd_5d)={float(row['mean_fwd_5d']):.4f}, "
                    f"mean(fwd_10d)={float(row['mean_fwd_10d']):.4f}"
                )
        else:
            lines.append("- Lag effect summary not available")

        if overreaction_rows:
            row = overreaction_rows[0]
            lines.append("### Overreaction fade (latest)")
            lines.append(
                "- "
                f"n={int(row['count'])}, "
                f"mean(fwd_1d)={float(row['mean_fwd_1d']):.4f}, "
                f"mean(fwd_3d)={float(row['mean_fwd_3d']):.4f}, "
                f"mean(fwd_5d)={float(row['mean_fwd_5d']):.4f}, "
                f"reversal_rate_3d={float(row['reversal_rate_3d']):.4f}"
            )
        else:
            lines.append("- Overreaction fade summary not available")

    path = settings.reports_dir / f"report_{latest_date.isoformat()}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    rep = pd.DataFrame(
        [{"date": latest_date, "path": str(path), "created_at": datetime.now(UTC).isoformat()}]
    )
    write_dataframe(rep, "reports", replace=False)
    return path


def main() -> None:
    run()


if __name__ == "__main__":
    main()
