from types import SimpleNamespace

import pandas as pd

from oil_risk.pipelines import train_model


def test_run_base_feature_set_uses_existing_features(monkeypatch, tmp_path):
    captured = {}

    monkeypatch.setattr(
        train_model,
        "settings",
        SimpleNamespace(models_dir=tmp_path / "models", model_feature_set="base"),
    )
    monkeypatch.setattr(
        train_model,
        "_assemble_training_frame",
        lambda: pd.DataFrame(
            {
                "oil_return": [0.1, 0.2, 0.15],
                "dVIX": [0.1, 0.2, 0.0],
                "dOVX": [0.1, 0.2, 0.0],
                "usd_change": [0.0, 0.1, 0.1],
                "rate_change": [0.0, 0.1, 0.0],
                "news_risk_score": [0.2, 0.3, 0.4],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        ),
    )

    def fake_train_regime_model(df, features):
        captured["features"] = list(features)
        state_df = pd.DataFrame(
            {
                "state_id": [0, 1, 2],
                "p_state_0": [1.0, 0.0, 0.0],
                "p_state_1": [0.0, 1.0, 0.0],
                "p_state_2": [0.0, 0.0, 1.0],
                "state_label": ["low_risk", "medium_risk", "high_risk"],
            },
            index=df.index,
        )
        return object(), state_df

    monkeypatch.setattr(train_model, "train_regime_model", fake_train_regime_model)
    monkeypatch.setattr(train_model, "save_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_model, "load_feature_frame", lambda: pd.DataFrame())
    monkeypatch.setattr(
        train_model, "build_tail_risk_dataset", lambda _: (_ for _ in ()).throw(ValueError("skip"))
    )
    monkeypatch.setattr(train_model, "write_dataframe", lambda *args, **kwargs: None)

    train_model.run()

    assert captured["features"] == [
        "oil_return",
        "dVIX",
        "dOVX",
        "usd_change",
        "rate_change",
        "news_risk_score",
    ]


def test_run_lagged_feature_set_requires_lagged_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(
        train_model,
        "settings",
        SimpleNamespace(models_dir=tmp_path / "models", model_feature_set="lagged"),
    )
    monkeypatch.setattr(
        train_model,
        "_assemble_training_frame",
        lambda: pd.DataFrame(
            {
                "oil_return": [0.1, 0.2, 0.15],
                "dVIX": [0.1, 0.2, 0.0],
                "dOVX": [0.1, 0.2, 0.0],
                "usd_change": [0.0, 0.1, 0.1],
                "rate_change": [0.0, 0.1, 0.0],
                "news_risk_score": [0.2, 0.3, 0.4],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        ),
    )

    try:
        train_model.run()
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected ValueError for missing lagged columns")

    assert "Run oil-build-features" in message
    assert "spx_return_lag1" in message
