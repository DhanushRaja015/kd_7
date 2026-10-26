"""
energy.py
---------
Appliance-level energy estimation and electricity cost calculations.

Formula (for each appliance):
    Energy (kWh) = Σ [ power_W × elapsed_seconds ] / 3,600,000

For irregular timestamps, actual elapsed time between readings is used.
Tariffs are user-configurable and never hard-coded.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPS = 1e-12     # Guard against division by zero


# ──────────────────────────────────────────────────────────────
# Core energy calculation
# ──────────────────────────────────────────────────────────────

def compute_energy_kwh(
    power_series: pd.Series,
    interval_seconds: Optional[float] = None,
) -> float:
    """
    Compute energy in kWh from a power series (in watts).

    Parameters
    ----------
    power_series : pd.Series
        Power values in watts, with a DatetimeIndex.
    interval_seconds : float | None
        Fixed sampling interval in seconds.  If None, the actual elapsed
        time between consecutive readings is used (correct for irregular
        sampling).

    Returns
    -------
    float  Energy in kWh (≥ 0).
    """
    power = power_series.dropna().clip(lower=0)
    if len(power) == 0:
        return 0.0

    if interval_seconds is not None:
        # Uniform sampling
        energy_wh = float(power.sum() * interval_seconds / 3600.0)
    else:
        # Irregular sampling: use actual time deltas
        if isinstance(power.index, pd.DatetimeIndex):
            dt = power.index.to_series().diff().dt.total_seconds().fillna(0)
        else:
            dt = pd.Series(np.ones(len(power)))
        energy_wh = float((power.values * dt.values).sum() / 3600.0)

    return max(0.0, energy_wh / 1000.0)   # Wh → kWh


def compute_cost(energy_kwh: float, tariff_per_kwh: float) -> float:
    """
    Estimate electricity cost.

    Parameters
    ----------
    energy_kwh : float
    tariff_per_kwh : float
        User-supplied tariff in currency per kWh.

    Returns
    -------
    float  Estimated cost (same currency as tariff).
    """
    if tariff_per_kwh < 0:
        raise ValueError("Tariff cannot be negative.")
    return round(energy_kwh * tariff_per_kwh, 4)


# ──────────────────────────────────────────────────────────────
# Appliance-level summary
# ──────────────────────────────────────────────────────────────

def appliance_energy_summary(
    pred_df: pd.DataFrame,
    appliance_keys: list[str],
    tariff_per_kwh: float = 0.12,
    interval_seconds: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute per-appliance energy, cost, peak power, and energy share.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Must contain '{app_key}_pred_w' columns (from predict.predict()).
    appliance_keys : list[str]
    tariff_per_kwh : float
        Electricity tariff in currency per kWh.
    interval_seconds : float | None
        Sampling interval; None = derived from index.

    Returns
    -------
    pd.DataFrame  One row per appliance with columns:
        appliance, pred_energy_kwh, peak_pred_w, estimated_cost,
        energy_share_pct, [true_energy_kwh, true_peak_w]  if labels exist.
    """
    rows = []
    total_pred_kwh = 0.0

    for app_key in appliance_keys:
        pred_col = f"{app_key}_pred_w"
        true_col = f"{app_key}_true_w"

        if pred_col not in pred_df.columns:
            continue

        pred_w = pred_df[pred_col].clip(lower=0)
        pred_kwh = compute_energy_kwh(pred_w, interval_seconds)
        peak_w = float(pred_w.max())
        cost = compute_cost(pred_kwh, tariff_per_kwh)

        row: dict = {
            "appliance": app_key,
            "pred_energy_kwh": round(pred_kwh, 4),
            "peak_pred_w": round(peak_w, 2),
            "estimated_cost": round(cost, 4),
        }

        if true_col in pred_df.columns:
            true_w = pred_df[true_col].clip(lower=0)
            true_kwh = compute_energy_kwh(true_w, interval_seconds)
            row["true_energy_kwh"] = round(true_kwh, 4)
            row["true_peak_w"] = round(float(true_w.max()), 2)

        rows.append(row)
        total_pred_kwh += pred_kwh

    df_out = pd.DataFrame(rows)
    if len(df_out) == 0:
        return df_out

    # Energy share
    df_out["energy_share_pct"] = df_out["pred_energy_kwh"].apply(
        lambda x: round(x / (total_pred_kwh + EPS) * 100, 2)
    )
    return df_out


# ──────────────────────────────────────────────────────────────
# Daily energy breakdown
# ──────────────────────────────────────────────────────────────

def daily_energy_breakdown(
    pred_df: pd.DataFrame,
    appliance_keys: list[str],
) -> pd.DataFrame:
    """
    Resample predictions to daily totals (kWh) per appliance.

    Returns a DataFrame indexed by date with one column per appliance.
    """
    if not isinstance(pred_df.index, pd.DatetimeIndex):
        raise ValueError("pred_df must have a DatetimeIndex.")

    daily: dict = {}
    for app_key in appliance_keys:
        col = f"{app_key}_pred_w"
        if col not in pred_df.columns:
            continue
        # Daily mean power → energy (mean_W × 86400 / 3,600,000 = kWh/day)
        daily_mean = pred_df[col].resample("D").mean()
        daily_kwh = daily_mean * 86400 / 3_600_000
        daily[app_key] = daily_kwh

    if not daily:
        return pd.DataFrame()

    df_daily = pd.DataFrame(daily).fillna(0)
    df_daily.index.name = "date"
    return df_daily


# ──────────────────────────────────────────────────────────────
# Aggregate discrepancy
# ──────────────────────────────────────────────────────────────

def aggregate_discrepancy(
    pred_df: pd.DataFrame,
    appliance_keys: list[str],
    aggregate_col: str = "aggregate_w",
) -> pd.Series:
    """
    Compute the per-row discrepancy between aggregate measured power and
    the sum of all appliance predictions.

    discrepancy = aggregate_w - Σ(app_pred_w)

    A positive value means the predictions do not fully account for the
    measured load; a negative value means predictions over-estimate.
    """
    pred_cols = [f"{k}_pred_w" for k in appliance_keys if f"{k}_pred_w" in pred_df.columns]
    if not pred_cols or aggregate_col not in pred_df.columns:
        return pd.Series(dtype=float)

    pred_sum = pred_df[pred_cols].sum(axis=1)
    discrepancy = pred_df[aggregate_col] - pred_sum
    return discrepancy
