import pandas as pd

from oil_risk.pipelines.tune_signals import _score_frame


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
