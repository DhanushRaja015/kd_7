"""
tests/test_features.py
Tests for feature engineering: alignment, no label leakage, shape correctness.
"""
import pytest
import numpy as np
import pandas as pd
from src.data_loader import generate_sample_dataset, AGGREGATE_COL, SUPPORTED_APPLIANCES
from src.features import build_features, align_features_labels, FEATURE_COLUMNS


@pytest.fixture
def df():
    return generate_sample_dataset(n_samples=200, seed=2)


def test_feature_columns_present(df):
    X = build_features(df)
    for col in FEATURE_COLUMNS:
        assert col in X.columns, f"Missing feature: {col}"


def test_feature_index_matches_input(df):
    X = build_features(df)
    assert X.index.equals(df.index)


def test_no_appliance_columns_in_features(df):
    X = build_features(df)
    for app in SUPPORTED_APPLIANCES:
        assert app not in X.columns, f"Appliance column leaked into features: {app}"


def test_feature_shape(df):
    X = build_features(df)
    assert X.shape == (len(df), len(FEATURE_COLUMNS))


def test_align_removes_nan_rows(df):
    X = build_features(df)
    y = df["refrigerator"]
    # Introduce NaN in X at row 0 (lag features produce NaN at start)
    X_orig_nan = X.isna().any(axis=1).sum()
    X_aligned, y_aligned = align_features_labels(X, y)
    assert not X_aligned.isna().any().any()
    assert len(X_aligned) == len(y_aligned)


def test_align_features_labels_same_index(df):
    X = build_features(df)
    y = df["refrigerator"]
    X_a, y_a = align_features_labels(X, y)
    assert X_a.index.equals(y_a.index)


def test_missing_aggregate_raises():
    bad = pd.DataFrame({"not_agg": [1, 2, 3]},
                       index=pd.date_range("2024-01-01", periods=3, freq="min"))
    with pytest.raises(ValueError, match=AGGREGATE_COL):
        build_features(bad)


def test_cyclical_features_bounded(df):
    X = build_features(df)
    for col in ["hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos"]:
        assert X[col].between(-1, 1).all(), f"{col} out of [-1,1]"


def test_is_night_binary(df):
    X = build_features(df)
    assert set(X["is_night"].unique()).issubset({0, 1})
