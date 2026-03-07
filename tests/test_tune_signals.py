import json

import pandas as pd
import pytest

from oil_risk.pipelines import tune_signals
from oil_risk.pipelines.tune_signals import _build_triggered_mask, _score_frame


def test_tuner_respects_max_trigger_rate():
    frame = pd.DataFrame(
        {
            "fwd_5d_abs": [0.1, 0.2, 0.3, 0.4, 0.5],
            "triggered": [True, True, True, False, False],
        }
    )
    score, trigger_rate = _score_frame(frame, max_trigger_rate=0.2)
    assert trigger_rate == 0.6
    assert score == float("-inf")


def test_score_frame_returns_finite_when_triggered_and_non_triggered_exist():
    frame = pd.DataFrame(
        {
            "fwd_5d_abs": [0.9, 0.8, 0.2, 0.1],
            "triggered": [True, True, False, False],
        }
    )
    score, trigger_rate = _score_frame(frame, max_trigger_rate=0.8)
    assert trigger_rate == 0.5
    assert score == pytest.approx(0.7)


def test_tail_risk_join_affects_triggered_calculation(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    features = pd.DataFrame(
        {
            "date": dates,
            "oil_return": [-0.01, 0.01, -0.02, 0.02, -0.01, 0.01, -0.02, 0.01],
            "OVX_z_63": [0.0] * 8,
            "VIX_z_63": [0.0] * 8,
            "geopolitical_risk_score": [0.0] * 8,
            "oil_spx_corr_63": [0.0] * 8,
            "unusual_put_activity": [0.0] * 8,
        }
    ).set_index("date")

    tail = pd.DataFrame(
        {
            "date": [dates[4], dates[5], dates[6]],
            "tail_risk_prob": [0.7, 0.1, 0.2],
        }
    )

    writes = []

    monkeypatch.setattr(tune_signals, "load_feature_frame", lambda: features)
    monkeypatch.setattr(tune_signals, "read_sql", lambda _query: tail)
    monkeypatch.setattr(
        tune_signals,
        "load_signals_config",
        lambda: {
            "tuning": {"max_trigger_rate": 1.0},
            "risk_premium_alert": {},
            "macro_stress_alert": {},
            "correlation_break_alert": {},
            "hedging_pressure_alert": {},
            "tail_risk_alert": {},
        },
    )

    def _capture_write(df, _table_name, replace=False):
        assert replace is False
        writes.append(df.copy())

    monkeypatch.setattr(tune_signals, "write_dataframe", _capture_write)

    tune_signals.run()

    payload = writes[0].iloc[0]
    leaderboard = pd.DataFrame(json.loads(payload["leaderboard_json"]))
    params = leaderboard.iloc[0]["params"]

    work = features.join(tail.set_index("date"), how="left")
    work["tail_risk_prob"] = work["tail_risk_prob"].fillna(0.0)
    triggered = _build_triggered_mask(work, params)
    assert bool(triggered.any())
