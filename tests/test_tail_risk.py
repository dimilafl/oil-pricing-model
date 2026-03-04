import pandas as pd

from oil_risk.modeling.tail_risk import build_tail_risk_dataset


def test_build_tail_risk_dataset_labels_on_synthetic_series():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    frame = pd.DataFrame(
        {
            "oil_return": [0.01, 0.03, -0.01, -0.025, 0.0],
            "VIX_z_63": [0, 0, 0, 0, 0],
        },
        index=idx,
    )
    out = build_tail_risk_dataset(frame, threshold=0.02)
    assert list(out["target"]) == [1.0, 0.0, 1.0, 0.0, 0.0]
