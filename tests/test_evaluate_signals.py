import json

import pandas as pd

from oil_risk.pipelines import evaluate_signals


def test_forward_return_calculation_exact():
    series = pd.Series([0.1, 0.2, -0.1, 0.05, 0.03])
    fwd_1d = evaluate_signals._forward_return_sums(series, 1)
    fwd_5d = evaluate_signals._forward_return_sums(series, 5)

    assert fwd_1d.iloc[:4].tolist() == [0.2, -0.1, 0.05, 0.03]
    assert pd.isna(fwd_1d.iloc[4])
    assert pd.isna(fwd_5d.iloc[0])


def test_grouped_outputs_have_expected_keys_and_counts(monkeypatch, tmp_path):
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    oil_returns = [0.01 * (i + 1) for i in range(12)]

    market_features = pd.DataFrame(
        {
            "date": [d.date().isoformat() for d in dates],
            "oil_return": oil_returns,
        }
    )
    model_state = pd.DataFrame(
        {
            "date": [d.date().isoformat() for d in dates],
            "state_id": [0] * 6 + [1] * 6,
            "state_label": ["calm"] * 6 + ["stress"] * 6,
        }
    )
    signals = pd.DataFrame(
        {
            "date": [d.date().isoformat() for d in dates],
            "signal_name": ["risk_premium_alert"] * 12,
            "signal_value": [1.0 if i in (0, 1, 2, 6, 7) else 0.0 for i in range(12)],
        }
    )

    def fake_read_sql(query: str):
        if "FROM market_features" in query:
            return market_features.copy()
        if "FROM model_state" in query:
            return model_state.copy()
        if "FROM signals" in query:
            return signals.copy()
        raise AssertionError(query)

    writes = []

    def fake_write(df: pd.DataFrame, table_name: str, replace: bool = False):
        writes.append((table_name, replace, df.copy()))

    monkeypatch.setattr(evaluate_signals, "read_sql", fake_read_sql)
    monkeypatch.setattr(evaluate_signals, "write_dataframe", fake_write)
    monkeypatch.chdir(tmp_path)

    out_path = evaluate_signals.run()

    assert out_path.exists()
    signal_eval_writes = [w for w in writes if w[0] == "signal_eval"]
    assert signal_eval_writes
    eval_df = signal_eval_writes[0][2]
    assert set(eval_df["eval_name"]) == {
        "triggered_signal_summary",
        "state_summary",
        "state_signal_triggered_summary",
    }

    trigger_eval_json = eval_df[eval_df["eval_name"] == "triggered_signal_summary"].iloc[0][
        "eval_json"
    ]
    trigger_rows = json.loads(trigger_eval_json)
    assert trigger_rows[0]["signal_name"] == "risk_premium_alert"
    assert trigger_rows[0]["count"] == 5

    by_state_json = eval_df[eval_df["eval_name"] == "state_summary"].iloc[0]["eval_json"]
    by_state_rows = json.loads(by_state_json)
    state_ids = {row["state_id"] for row in by_state_rows}
    assert state_ids == {0, 1}
