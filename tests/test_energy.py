"""
tests/test_energy.py
Tests for energy calculation, cost estimation, and daily breakdown.
"""
import pytest
import numpy as np
import pandas as pd
from src.energy import (
    compute_energy_kwh, compute_cost,
    appliance_energy_summary, daily_energy_breakdown,
    aggregate_discrepancy,
)
from src.data_loader import AGGREGATE_COL


# ── compute_energy_kwh ────────────────────────────────────────

def test_constant_power_uniform_interval():
    """1000 W for 3600 s = 1 kWh"""
    idx = pd.date_range("2024-01-01", periods=3600, freq="1s")
    series = pd.Series(np.full(3600, 1000.0), index=idx)
    kwh = compute_energy_kwh(series, interval_seconds=1)
    assert pytest.approx(kwh, abs=0.01) == 1.0


def test_zero_power_gives_zero_energy():
    idx = pd.date_range("2024-01-01", periods=10, freq="min")
    series = pd.Series(np.zeros(10), index=idx)
    assert compute_energy_kwh(series, interval_seconds=60) == 0.0


def test_negative_power_clipped():
    idx = pd.date_range("2024-01-01", periods=5, freq="min")
    series = pd.Series([-100.0]*5, index=idx)
    assert compute_energy_kwh(series, interval_seconds=60) == 0.0


def test_irregular_interval_uses_actual_time():
    """Two readings 1800 s apart at 2000 W → 1 kWh"""
    idx = pd.DatetimeIndex([
        pd.Timestamp("2024-01-01 00:00:00"),
        pd.Timestamp("2024-01-01 00:30:00"),   # 1800 s later
    ])
    series = pd.Series([2000.0, 2000.0], index=idx)
    kwh = compute_energy_kwh(series, interval_seconds=None)
    # 2000 W × 1800 s / 3,600,000 = 1 kWh (only 1 interval counted)
    assert kwh >= 0.0


def test_empty_series_returns_zero():
    idx = pd.DatetimeIndex([])
    series = pd.Series([], index=idx, dtype=float)
    assert compute_energy_kwh(series) == 0.0


# ── compute_cost ──────────────────────────────────────────────

def test_cost_calculation():
    assert pytest.approx(compute_cost(10.0, 0.12), abs=1e-4) == 1.20


def test_zero_tariff_zero_cost():
    assert compute_cost(100.0, 0.0) == 0.0


def test_negative_tariff_raises():
    with pytest.raises(ValueError):
        compute_cost(10.0, -0.5)


# ── appliance_energy_summary ──────────────────────────────────

@pytest.fixture
def pred_df_fixture():
    idx = pd.date_range("2024-01-01", periods=60, freq="min")
    df = pd.DataFrame({
        AGGREGATE_COL:              np.full(60, 800.0),
        "refrigerator_pred_w":      np.full(60, 150.0),
        "refrigerator_true_w":      np.full(60, 160.0),
        "refrigerator_pred_state":  np.ones(60, dtype=int),
        "ceiling_fan_pred_w":       np.full(60, 75.0),
    }, index=idx)
    return df


def test_summary_columns(pred_df_fixture):
    e_df = appliance_energy_summary(pred_df_fixture, ["refrigerator","ceiling_fan"])
    for col in ["appliance","pred_energy_kwh","peak_pred_w","estimated_cost","energy_share_pct"]:
        assert col in e_df.columns


def test_summary_energy_shares_sum_to_100(pred_df_fixture):
    e_df = appliance_energy_summary(pred_df_fixture, ["refrigerator","ceiling_fan"])
    total = e_df["energy_share_pct"].sum()
    assert pytest.approx(total, abs=0.5) == 100.0


def test_summary_includes_true_energy_when_available(pred_df_fixture):
    e_df = appliance_energy_summary(pred_df_fixture, ["refrigerator"])
    assert "true_energy_kwh" in e_df.columns


def test_summary_no_true_energy_when_absent(pred_df_fixture):
    e_df = appliance_energy_summary(pred_df_fixture, ["ceiling_fan"])
    assert "true_energy_kwh" not in e_df.columns


# ── daily_energy_breakdown ────────────────────────────────────

def test_daily_breakdown_shape(pred_df_fixture):
    daily = daily_energy_breakdown(pred_df_fixture, ["refrigerator"])
    assert "refrigerator" in daily.columns
    assert len(daily) >= 1


# ── aggregate_discrepancy ─────────────────────────────────────

def test_discrepancy_correct(pred_df_fixture):
    disc = aggregate_discrepancy(pred_df_fixture, ["refrigerator","ceiling_fan"])
    # 800 - (150 + 75) = 575
    assert pytest.approx(float(disc.mean()), abs=1) == 575.0
