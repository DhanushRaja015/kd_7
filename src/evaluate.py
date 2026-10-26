"""
evaluate.py
-----------
Model evaluation metrics for NILM power regression and ON/OFF classification.

All metrics are computed on held-out test data ONLY.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

logger = logging.getLogger(__name__)

EPS = 1e-9   # Prevents division-by-zero in relative metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE, R² for power regression."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    return {"mae_w": round(mae, 3), "rmse_w": round(rmse, 3), "r2": round(r2, 4)}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Precision, Recall, F1 for ON/OFF classification."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
    }


def energy_error_metrics(
    energy_true_kwh: float,
    energy_pred_kwh: float,
) -> dict:
    """Absolute and relative energy estimation error."""
    abs_err = abs(energy_true_kwh - energy_pred_kwh)
    if abs(energy_true_kwh) < EPS:
        rel_err = float("inf") if abs_err > EPS else 0.0
    else:
        rel_err = abs_err / abs(energy_true_kwh)
    return {
        "energy_true_kwh": round(energy_true_kwh, 4),
        "energy_pred_kwh": round(energy_pred_kwh, 4),
        "absolute_error_kwh": round(abs_err, 4),
        "relative_error_pct": round(rel_err * 100, 2),
    }


def evaluate_appliance(
    pred_df: pd.DataFrame,
    app_key: str,
    baseline_pred_w: Optional[np.ndarray] = None,
    interval_seconds: Optional[float] = None,
) -> dict:
    """
    Full per-appliance evaluation on the test prediction DataFrame.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Output of predict.predict(), must contain '{app_key}_pred_w' and
        optionally '{app_key}_true_w' and '{app_key}_pred_state'.
    app_key : str
    baseline_pred_w : array-like | None
        Baseline predictions aligned with pred_df.
    interval_seconds : float | None
        Sampling interval; if None it is estimated from the index.

    Returns
    -------
    dict with regression, classification, energy, and baseline sub-dicts.
    """
    pred_col = f"{app_key}_pred_w"
    true_col = f"{app_key}_true_w"
    state_col = f"{app_key}_pred_state"

    if pred_col not in pred_df.columns:
        return {"error": f"Prediction column '{pred_col}' not found."}

    pred_w = pred_df[pred_col].values
    result: dict = {"appliance": app_key}

    # Determine interval
    if interval_seconds is None and len(pred_df) > 1:
        diffs = pred_df.index.to_series().diff().dropna()
        interval_seconds = float(diffs.median().total_seconds())
    interval_seconds = interval_seconds or 60.0

    # Energy from predictions
    pred_kwh = float(np.sum(pred_w) * interval_seconds / 3_600_000)
    result["pred_energy_kwh"] = round(pred_kwh, 4)

    if true_col in pred_df.columns:
        true_w = pred_df[true_col].values

        # Regression
        result["regression"] = regression_metrics(true_w, pred_w)

        # ON/OFF classification
        if state_col in pred_df.columns:
            true_state = (true_w >= 10.0).astype(int)
            pred_state = pred_df[state_col].values.astype(int)
            result["classification"] = classification_metrics(true_state, pred_state)

        # Energy error
        true_kwh = float(np.sum(true_w) * interval_seconds / 3_600_000)
        result["energy"] = energy_error_metrics(true_kwh, pred_kwh)

        # Baseline comparison
        if baseline_pred_w is not None:
            bl_pred = np.asarray(baseline_pred_w, dtype=float)
            result["baseline_regression"] = regression_metrics(true_w, bl_pred)
            bl_kwh = float(np.sum(bl_pred) * interval_seconds / 3_600_000)
            result["baseline_energy"] = energy_error_metrics(true_kwh, bl_kwh)

    return result


def evaluate_all(
    pred_df: pd.DataFrame,
    trainer,                       # NILMTrainer (avoid circular import with string hint)
    appliance_keys: list[str],
) -> dict:
    """
    Run evaluate_appliance for every appliance and return a combined report.
    """
    report: dict = {}
    for app_key in appliance_keys:
        bl_pred = None
        if app_key in trainer.baselines:
            bl_pred = trainer.baselines[app_key].predict_power(len(pred_df))
        report[app_key] = evaluate_appliance(pred_df, app_key, bl_pred)
    return report


def metrics_to_csv(report: dict) -> str:
    """Flatten the nested report dict to a CSV string."""
    rows = []
    for app, metrics in report.items():
        if not isinstance(metrics, dict):
            continue
        flat: dict = {"appliance": app}
        for section, vals in metrics.items():
            if isinstance(vals, dict):
                for k, v in vals.items():
                    flat[f"{section}_{k}"] = v
            else:
                flat[section] = vals
        rows.append(flat)
    return pd.DataFrame(rows).to_csv(index=False)
