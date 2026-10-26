"""
tests/test_data_loader.py
Tests for CSV validation, loading, column mapping, and quality reporting.
"""
import io
import pytest
import pandas as pd
import numpy as np
from src.data_loader import (
    DataLoader, validate_csv_bytes, generate_sample_dataset,
    sample_csv_bytes, AGGREGATE_COL, SUPPORTED_APPLIANCES,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    return generate_sample_dataset(n_samples=120, seed=0)


@pytest.fixture
def sample_csv(sample_df):
    buf = io.StringIO()
    sample_df.reset_index().rename(columns={AGGREGATE_COL: "aggregate_w"}).to_csv(buf, index=False)
    return buf.getvalue().encode()


# ── validate_csv_bytes ────────────────────────────────────────

def test_empty_bytes_rejected():
    ok, msg = validate_csv_bytes(b"")
    assert not ok
    assert "empty" in msg.lower()


def test_single_row_rejected():
    ok, msg = validate_csv_bytes(b"timestamp,aggregate_w\n")
    assert not ok


def test_valid_bytes_accepted(sample_csv):
    ok, msg = validate_csv_bytes(sample_csv)
    assert ok


# ── DataLoader.load ───────────────────────────────────────────

def test_load_aggregate_only():
    csv = b"timestamp,aggregate_w\n2024-01-01 00:00:00,100\n2024-01-01 00:01:00,120\n"
    loader = DataLoader(timestamp_col="timestamp", aggregate_col="aggregate_w")
    df = loader.load(csv)
    assert AGGREGATE_COL in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == 2


def test_load_with_appliance_columns(sample_csv):
    app_map = {k: k for k in SUPPORTED_APPLIANCES}
    loader = DataLoader(
        timestamp_col="timestamp",
        aggregate_col="aggregate_w",
        appliance_col_map=app_map,
    )
    df = loader.load(sample_csv)
    for k in SUPPORTED_APPLIANCES:
        assert k in df.columns, f"Missing appliance column: {k}"


def test_missing_required_column_raises():
    csv = b"time,power\n2024-01-01,100\n2024-01-02,110\n"
    loader = DataLoader(timestamp_col="timestamp", aggregate_col="aggregate_w")
    with pytest.raises(ValueError, match="Required columns"):
        loader.load(csv)


def test_invalid_timestamp_raises():
    csv = b"timestamp,aggregate_w\nnot-a-date,100\nalso-bad,200\n"
    loader = DataLoader()
    with pytest.raises(ValueError, match="timestamp"):
        loader.load(csv)


def test_duplicate_timestamps_removed():
    csv = (
        b"timestamp,aggregate_w\n"
        b"2024-01-01 00:00:00,100\n"
        b"2024-01-01 00:00:00,200\n"  # duplicate
        b"2024-01-01 00:01:00,150\n"
    )
    loader = DataLoader(timestamp_col="timestamp", aggregate_col="aggregate_w")
    df = loader.load(csv)
    assert len(df) == 2
    assert df.index.is_unique


def test_kw_unit_conversion():
    csv = b"timestamp,aggregate_kw\n2024-01-01 00:00:00,1.5\n2024-01-01 00:01:00,2.0\n"
    loader = DataLoader(timestamp_col="timestamp", aggregate_col="aggregate_kw", power_unit="kW")
    df = loader.load(csv)
    assert pytest.approx(df[AGGREGATE_COL].iloc[0], abs=1) == 1500.0


def test_non_numeric_power_raises():
    csv = b"timestamp,aggregate_w\n2024-01-01 00:00,not_a_number\n"
    loader = DataLoader()
    with pytest.raises((ValueError, Exception)):
        loader.load(csv)


def test_quality_report_structure(sample_csv):
    loader = DataLoader(
        timestamp_col="timestamp", aggregate_col="aggregate_w",
        appliance_col_map={k: k for k in SUPPORTED_APPLIANCES},
    )
    df = loader.load(sample_csv)
    qr = loader.quality_report(df)
    for key in ["total_rows","duration","missing_aggregate","negative_aggregate"]:
        assert key in qr


def test_sample_csv_bytes_downloadable():
    raw = sample_csv_bytes()
    assert isinstance(raw, bytes)
    assert len(raw) > 0
    df = pd.read_csv(io.BytesIO(raw))
    assert "aggregate_w" in df.columns
