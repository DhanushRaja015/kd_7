"""
tests/test_preprocessing.py
Tests for cleaning, chronological splitting, and leakage prevention.
"""
import pytest
import numpy as np
import pandas as pd
from src.data_loader import generate_sample_dataset, AGGREGATE_COL
from src.preprocessing import (
    NILMPreprocessor, chronological_split, preprocessing_summary,
)


@pytest.fixture
def df():
    return generate_sample_dataset(n_samples=300, seed=1)


# ── chronological_split ────────────────────────────────────────

def test_split_sizes(df):
    train, val, test = chronological_split(df, 0.70, 0.15)
    n = len(df)
    assert len(train) + len(val) + len(test) == n


def test_split_chronological_order(df):
    train, val, test = chronological_split(df)
    assert train.index[-1] < val.index[0]
    assert val.index[-1]   < test.index[0]


def test_split_no_overlap(df):
    train, val, test = chronological_split(df)
    assert set(train.index).isdisjoint(set(val.index))
    assert set(train.index).isdisjoint(set(test.index))
    assert set(val.index).isdisjoint(set(test.index))


def test_split_invalid_fractions(df):
    with pytest.raises(ValueError):
        chronological_split(df, train_frac=0.9, val_frac=0.2)   # sum > 1


def test_split_train_fraction_boundary(df):
    with pytest.raises(ValueError):
        chronological_split(df, train_frac=0.0)


# ── NILMPreprocessor ──────────────────────────────────────────

def test_preprocessor_clips_negatives(df):
    df_neg = df.copy()
    df_neg.iloc[0, df_neg.columns.get_loc(AGGREGATE_COL)] = -500
    train, _, _ = chronological_split(df_neg)
    prep = NILMPreprocessor()
    cleaned = prep.fit_transform(train)
    assert (cleaned[AGGREGATE_COL] >= 0).all()


def test_preprocessor_fills_nan():
    idx = pd.date_range("2024-01-01", periods=20, freq="min")
    data = {AGGREGATE_COL: [100.0]*20}
    df_nan = pd.DataFrame(data, index=idx)
    df_nan.iloc[5, 0] = np.nan
    prep = NILMPreprocessor()
    out = prep.fit_transform(df_nan)
    assert not out[AGGREGATE_COL].isna().any()


def test_preprocessor_transform_before_fit_raises():
    df_t = generate_sample_dataset(30)
    prep = NILMPreprocessor()
    with pytest.raises(RuntimeError):
        prep.transform(df_t)


def test_preprocessor_fit_only_on_train(df):
    """Scaler must NOT be fitted on val/test data."""
    train, val, test = chronological_split(df)
    prep = NILMPreprocessor(scale_aggregate=True)
    train_c = prep.fit_transform(train)
    val_c   = prep.transform(val)
    test_c  = prep.transform(test)
    # All should transform without error
    assert len(train_c) == len(train)
    assert len(val_c)   == len(val)
    assert len(test_c)  == len(test)


# ── preprocessing_summary ─────────────────────────────────────

def test_preprocessing_summary_keys(df):
    train, val, test = chronological_split(df)
    prep = NILMPreprocessor()
    train_c = prep.fit_transform(train)
    val_c   = prep.transform(val)
    test_c  = prep.transform(test)
    summ = preprocessing_summary(df, train_c, val_c, test_c, ["refrigerator"])
    for k in ["original_rows","train_rows","val_rows","test_rows","labeled"]:
        assert k in summ
