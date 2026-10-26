"""
predict.py
----------
Prediction pipeline for NILM.

Accepts a DataFrame with aggregate power (and optionally ground-truth
appliance labels) and returns per-appliance power and ON/OFF predictions
using a trained NILMTrainer.

Predictions are derived exclusively from aggregate features.
Ground-truth labels, when present, are used only for evaluation, never
as model inputs.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.data_loader import AGGREGATE_COL
from src.features import build_features
from src.train import NILMTrainer

logger = logging.getLogger(__name__)


def predict(
    trainer: NILMTrainer,
    df: pd.DataFrame,
    appliance_keys: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Generate appliance-level power and ON/OFF predictions.

    Parameters
    ----------
    trainer : NILMTrainer
        A fitted (or loaded) trainer.
    df : pd.DataFrame
        Input data with DatetimeIndex and AGGREGATE_COL.
        May optionally contain ground-truth appliance columns
        (those columns are NEVER used as features).
    appliance_keys : list[str] | None
        Subset of appliances to predict.  Defaults to all trained appliances.

    Returns
    -------
    pd.DataFrame
        Same index as `df`, with columns:
          - 'aggregate_w'          : original aggregate reading
          - '{app}_pred_w'         : predicted power in watts
          - '{app}_pred_state'     : predicted ON/OFF (1/0)
          - '{app}_true_w'         : ground-truth power if available
    """
    if AGGREGATE_COL not in df.columns:
        raise ValueError(f"Input DataFrame must contain '{AGGREGATE_COL}' column.")

    keys = appliance_keys or trainer.trained_appliances()
    if not keys:
        raise ValueError("No trained appliances available.  Train a model first.")

    X = build_features(df)
    # Do NOT drop NaN rows here; fill lags with bfill so predictions cover the
    # full input time range (first few rows where lags are NaN get filled).
    X = X.bfill().ffill().fillna(0)

    result = pd.DataFrame(index=df.index)
    result[AGGREGATE_COL] = df[AGGREGATE_COL].values

    for app_key in keys:
        if app_key not in trainer.regressors:
            logger.warning("No regressor for '%s'; skipping.", app_key)
            continue

        # --- Regressor ---
        pred_w = trainer.regressors[app_key].predict(X)
        pred_w = np.clip(pred_w, 0, None)   # power cannot be negative
        result[f"{app_key}_pred_w"] = pred_w

        # --- Classifier ---
        if app_key in trainer.classifiers:
            pred_state = trainer.classifiers[app_key].predict(X)
            result[f"{app_key}_pred_state"] = pred_state.astype(int)
        else:
            # Derive state from regressor output
            result[f"{app_key}_pred_state"] = (
                pred_w >= trainer.on_threshold_w
            ).astype(int)

        # --- Ground truth (for evaluation display, NOT used as input) ---
        if app_key in df.columns:
            result[f"{app_key}_true_w"] = df[app_key].values

    return result


def validate_schema_compatibility(
    df: pd.DataFrame,
    trainer: NILMTrainer,
) -> list[str]:
    """
    Check that an unseen DataFrame has the columns needed for prediction.

    Returns a list of warning strings (empty list = OK).
    """
    warnings: list[str] = []
    if AGGREGATE_COL not in df.columns:
        warnings.append(f"Missing required column: '{AGGREGATE_COL}'.")
    if not isinstance(df.index, pd.DatetimeIndex):
        warnings.append("Index is not a DatetimeIndex; time features may be wrong.")
    if len(df) < 15:
        warnings.append(
            "Very few rows (< 15); rolling features will rely on padding and may be unreliable."
        )
    return warnings
