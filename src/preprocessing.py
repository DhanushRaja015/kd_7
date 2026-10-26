"""
preprocessing.py
----------------
Data cleaning and chronological train/val/test splitting for NILM.

All preprocessing steps are fitted ONLY on the training partition to
prevent data leakage into validation or test sets.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.data_loader import AGGREGATE_COL, SUPPORTED_APPLIANCES

try:
    from sklearn.preprocessing import StandardScaler as _StandardScaler
except ImportError:
    _StandardScaler = None  # type: ignore

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Chronological splitter
# ──────────────────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a sorted DataFrame chronologically into train / val / test.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex sorted ascending.
    train_frac : float
        Fraction of rows for the training set (default 0.70).
    val_frac : float
        Fraction of rows for the validation set (default 0.15).
        The test set receives the remaining fraction.

    Returns
    -------
    (train_df, val_df, test_df)

    Notes
    -----
    - The test partition is NEVER used to fit any preprocessing step.
    - Splits are performed on row indices, not on calendar dates.
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be in (0, 1).")
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be in (0, 1).")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0 to leave room for a test set.")

    n = len(df)
    i_val = int(n * train_frac)
    i_test = int(n * (train_frac + val_frac))

    train = df.iloc[:i_val].copy()
    val = df.iloc[i_val:i_test].copy()
    test = df.iloc[i_test:].copy()

    logger.info(
        "Split: train=%d, val=%d, test=%d rows.", len(train), len(val), len(test)
    )
    return train, val, test


# ──────────────────────────────────────────────────────────────
# Preprocessing pipeline
# ──────────────────────────────────────────────────────────────

class NILMPreprocessor:
    """
    Stateful preprocessor that fits on training data and transforms all splits.

    Operations (applied to aggregate power and appliance power separately):
    1. Clip negative readings to 0.
    2. Fill remaining NaN values by forward-fill then backward-fill.
    3. (Optional) StandardScaler on aggregate column for feature normalisation.
       Appliance columns are NOT normalised here; normalisation for the target
       variables is handled inside the training module if needed.

    Attributes
    ----------
    scaler : StandardScaler | None
        Fitted only after calling fit_transform().
    """

    def __init__(self, scale_aggregate: bool = False) -> None:
        self.scale_aggregate = scale_aggregate
        self.scaler = None   # StandardScaler once fitted
        self._fitted = False

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on and transform the training DataFrame."""
        df = self._clean(train_df)
        if self.scale_aggregate:
            if _StandardScaler is None:
                raise ImportError("scikit-learn is required for scale_aggregate=True. Install it with: pip install scikit-learn")
            self.scaler = _StandardScaler()
            df[AGGREGATE_COL] = self.scaler.fit_transform(df[[AGGREGATE_COL]])
        self._fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform val / test DataFrames using parameters fitted on training data."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() on the training set first.")
        df = self._clean(df)
        if self.scale_aggregate and self.scaler is not None:
            df[AGGREGATE_COL] = self.scaler.transform(df[[AGGREGATE_COL]])
        return df

    def inverse_transform_aggregate(self, values: np.ndarray) -> np.ndarray:
        """Undo scaler on aggregate predictions if scaling was applied."""
        if self.scaler is not None:
            return self.scaler.inverse_transform(values.reshape(-1, 1)).ravel()
        return values

    # ----------------------------------------------------------
    # Private
    # ----------------------------------------------------------

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Clip negatives
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].clip(lower=0)
        # Fill missing values
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        return df


# ──────────────────────────────────────────────────────────────
# Preprocessing summary
# ──────────────────────────────────────────────────────────────

def preprocessing_summary(
    original: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    appliance_keys: list[str],
) -> dict:
    """
    Return a human-readable summary dict of the preprocessing step.
    """
    summary: dict = {
        "original_rows": len(original),
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "train_start": str(train.index[0]) if len(train) else "N/A",
        "train_end": str(train.index[-1]) if len(train) else "N/A",
        "val_start": str(val.index[0]) if len(val) else "N/A",
        "val_end": str(val.index[-1]) if len(val) else "N/A",
        "test_start": str(test.index[0]) if len(test) else "N/A",
        "test_end": str(test.index[-1]) if len(test) else "N/A",
        "appliances": appliance_keys,
        "labeled": len(appliance_keys) > 0,
    }
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        summary[f"{split_name}_agg_mean_w"] = round(float(split_df[AGGREGATE_COL].mean()), 2)
        summary[f"{split_name}_agg_max_w"] = round(float(split_df[AGGREGATE_COL].max()), 2)
    return summary
