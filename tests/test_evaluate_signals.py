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


def test_lag_effect_summary_bins_and_stats():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "spx_return": [0.00, -0.02, -0.001, 0.012, 0.004, -0.013],
            "fwd_1d": [0.01, -0.01, 0.02, -0.02, 0.03, -0.03],
            "fwd_5d": [0.05, -0.05, 0.06, -0.06, 0.07, -0.07],
            "fwd_10d": [0.10, -0.10, 0.11, -0.11, 0.12, -0.12],
        }
    )

    out = evaluate_signals._summarize_lag_effect(frame)

    assert set(out["lag_bin"]) == {"strong_down", "flat", "strong_up"}
    flat = out[out["lag_bin"] == "flat"].iloc[0]
    assert flat["count"] == 2
    assert flat["mean_fwd_1d"] == -0.015


def test_overreaction_fade_summary_uses_internal_zscore():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    oil_returns = [0.005] * 21 + [0.03, -0.028, 0.026, -0.03, 0.004, -0.004, 0.005, -0.005, 0.006]
    frame = pd.DataFrame(
        {
            "date": dates,
            "oil_return": oil_returns,
            "fwd_1d": [0.001] * 30,
            "fwd_3d": [-0.002] * 30,
            "fwd_5d": [-0.003] * 30,
        }
    )

    out = evaluate_signals._summarize_overreaction_fade(frame)

    assert out.iloc[0]["segment"] == "overreaction_days"
    assert out.iloc[0]["count"] > 0
    assert 0.0 <= out.iloc[0]["reversal_rate_3d"] <= 1.0


def test_grouped_outputs_have_expected_keys_and_counts(monkeypatch, tmp_path):
    dates = pd.date_range("2024-01-01", periods=35, freq="D")
    oil_returns = [0.004] * 35
    oil_returns[24] = 0.05
    spx_returns = ([0.0, -0.012, 0.0, 0.011] * 9)[:35]

    records = []
    for d, oil_ret, spx_ret in zip(dates, oil_returns, spx_returns, strict=True):
        records.append(
            {
                "date": d.date().isoformat(),
                "feature_name": "oil_return",
                "feature_value": oil_ret,
            }
        )
        records.append(
            {
                "date": d.date().isoformat(),
                "feature_name": "spx_return",
                "feature_value": spx_ret,
            }
        )
    market_features = pd.DataFrame(records)

    model_state = pd.DataFrame(
        {
            "date": [d.date().isoformat() for d in dates],
            "state_id": [0] * 18 + [1] * 17,
            "state_label": ["calm"] * 18 + ["stress"] * 17,
        }
    )
    signals = pd.DataFrame(
        {
            "date": [d.date().isoformat() for d in dates],
            "signal_name": ["risk_premium_alert"] * 35,
            "signal_value": [1.0 if i in (0, 1, 2, 6, 7, 22) else 0.0 for i in range(35)],
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
        "lag_effect_summary",
        "overreaction_fade_summary",
    }

    trigger_eval_json = eval_df[eval_df["eval_name"] == "triggered_signal_summary"].iloc[0][
        "eval_json"
    ]
    trigger_rows = json.loads(trigger_eval_json)
    assert trigger_rows[0]["signal_name"] == "risk_premium_alert"
    assert trigger_rows[0]["count"] == 6

    by_state_json = eval_df[eval_df["eval_name"] == "state_summary"].iloc[0]["eval_json"]
    by_state_rows = json.loads(by_state_json)
    state_ids = {row["state_id"] for row in by_state_rows}
    assert state_ids == {0, 1}

    lag_json = eval_df[eval_df["eval_name"] == "lag_effect_summary"].iloc[0]["eval_json"]
    lag_rows = json.loads(lag_json)
    assert lag_rows

    overreaction_json = eval_df[eval_df["eval_name"] == "overreaction_fade_summary"].iloc[0][
        "eval_json"
    ]
    overreaction_rows = json.loads(overreaction_json)
    assert overreaction_rows[0]["segment"] == "overreaction_days"
