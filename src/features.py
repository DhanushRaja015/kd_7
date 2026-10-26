"""
features.py
-----------
Feature engineering from the aggregate power signal only.

No appliance ground-truth label is ever used as an input feature.
Features are built entirely from the aggregate_w column plus calendar
information derived from the index.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.data_loader import AGGREGATE_COL

logger = logging.getLogger(__name__)

# Canonical feature column order (used to guarantee consistent shapes)
FEATURE_COLUMNS = [
    "agg_w",           # current aggregate reading
    "lag_1",           # 1-step lag
    "lag_2",           # 2-step lag
    "lag_5",           # 5-step lag
    "lag_10",          # 10-step lag
    "rolling_mean_5",  # rolling mean over 5 steps
    "rolling_mean_15", # rolling mean over 15 steps
    "rolling_max_5",   # rolling max over 5 steps
    "rolling_max_15",  # rolling max over 15 steps
    "rolling_std_5",   # rolling std over 5 steps
    "diff_1",          # first difference
    "hour_sin",        # cyclical hour encoding
    "hour_cos",
    "dayofweek_sin",   # cyclical day-of-week encoding
    "dayofweek_cos",
    "is_night",        # binary: 22:00 – 06:00
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix from a DataFrame with a DatetimeIndex and
    an 'aggregate_w' column.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (must have DatetimeIndex and AGGREGATE_COL column).

    Returns
    -------
    pd.DataFrame
        Feature matrix aligned with the input index.  Rows with NaN
        (from lag / rolling operations) are NOT dropped here; the caller
        is responsible for alignment.
    """
    if AGGREGATE_COL not in df.columns:
        raise ValueError(f"Column '{AGGREGATE_COL}' not found in DataFrame.")

    agg = df[AGGREGATE_COL].copy()
    features = pd.DataFrame(index=df.index)

    features["agg_w"] = agg

    # Lag features
    for lag in [1, 2, 5, 10]:
        features[f"lag_{lag}"] = agg.shift(lag)

    # Rolling statistics
    features["rolling_mean_5"] = agg.rolling(5, min_periods=1).mean()
    features["rolling_mean_15"] = agg.rolling(15, min_periods=1).mean()
    features["rolling_max_5"] = agg.rolling(5, min_periods=1).max()
    features["rolling_max_15"] = agg.rolling(15, min_periods=1).max()
    features["rolling_std_5"] = agg.rolling(5, min_periods=1).std().fillna(0)

    # First difference
    features["diff_1"] = agg.diff().fillna(0)

    # Time-of-day (cyclical)
    hour = df.index.hour + df.index.minute / 60.0
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Day-of-week (cyclical)
    dow = df.index.dayofweek.astype(float)
    features["dayofweek_sin"] = np.sin(2 * np.pi * dow / 7)
    features["dayofweek_cos"] = np.cos(2 * np.pi * dow / 7)

    # Night flag
    features["is_night"] = ((df.index.hour >= 22) | (df.index.hour < 6)).astype(int)

    # Ensure column order is canonical
    features = features[FEATURE_COLUMNS]

    return features


def align_features_labels(
    features: pd.DataFrame,
    labels: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Drop rows where either features or labels contain NaN and return
    aligned copies.

    This is the correct place to remove the NaN rows introduced by
    lag features.  It must be called *after* build_features() and
    only on the training / validation set (never on prediction input).
    """
    combined = features.join(labels.rename("__label__"), how="inner")
    combined = combined.dropna()
    X = combined[features.columns]
    y = combined["__label__"]
    return X, y


def features_summary(X: pd.DataFrame) -> dict:
    """Return a lightweight summary of the feature matrix."""
    return {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
        "any_nan": bool(X.isna().any().any()),
        "agg_mean": round(float(X["agg_w"].mean()), 2) if "agg_w" in X.columns else None,
    }
