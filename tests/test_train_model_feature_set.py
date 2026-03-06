import pandas as pd
import pytest

from oil_risk.pipelines import train_model


def test_select_feature_columns_base_default(monkeypatch):
    monkeypatch.delenv("MODEL_FEATURE_SET", raising=False)
    cols = train_model._select_feature_columns(pd.DataFrame())
    assert cols == train_model.BASE_FEATURES


def test_select_feature_columns_lagged_requires_columns(monkeypatch):
    monkeypatch.setenv("MODEL_FEATURE_SET", "lagged")
    with pytest.raises(ValueError, match="MODEL_FEATURE_SET=lagged requires columns"):
        train_model._select_feature_columns(pd.DataFrame(columns=train_model.BASE_FEATURES))


def test_select_feature_columns_lagged_ok(monkeypatch):
    monkeypatch.setenv("MODEL_FEATURE_SET", "lagged")
    cols = train_model._select_feature_columns(
        pd.DataFrame(columns=train_model.BASE_FEATURES + train_model.LAGGED_EXTENSION)
    )
    assert cols == train_model.BASE_FEATURES + train_model.LAGGED_EXTENSION
