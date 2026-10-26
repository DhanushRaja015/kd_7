"""
data_loader.py
--------------
Handles CSV ingestion, schema validation, column mapping, and basic
quality checks for NILM datasets.

Supported schemas
-----------------
- Aggregate-only  : timestamp + aggregate power column
- Labeled         : timestamp + aggregate power + one or more appliance columns
- UK-DALE style   : documented adapters (manual import)

All loaded data is returned as a pandas DataFrame with:
  - A DatetimeIndex named 'timestamp'
  - An 'aggregate_w' column (watts)
  - Optional appliance columns named by appliance key
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Public constants
# ──────────────────────────────────────────────────────────────
AGGREGATE_COL = "aggregate_w"

# Canonical appliance keys used throughout the project.
SUPPORTED_APPLIANCES = [
    "refrigerator",
    "ceiling_fan",
    "air_conditioner",
    "washing_machine",
    "electric_kettle",
]

APPLIANCE_DISPLAY_NAMES = {
    "refrigerator": "Refrigerator",
    "ceiling_fan": "Ceiling Fan",
    "air_conditioner": "Air Conditioner",
    "washing_machine": "Washing Machine",
    "electric_kettle": "Electric Kettle",
}


# ──────────────────────────────────────────────────────────────
# CSV validation helpers
# ──────────────────────────────────────────────────────────────

def validate_csv_bytes(raw: bytes) -> tuple[bool, str]:
    """
    Quick sanity check on raw CSV bytes before full parsing.

    Returns (ok: bool, message: str).
    """
    if len(raw) == 0:
        return False, "Uploaded file is empty."
    if len(raw) > 2 * 1024 * 1024 * 1024:      # 2 GB hard limit
        return False, "File exceeds the 2 GB size limit."
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception as exc:
        return False, f"Could not decode file as UTF-8: {exc}"
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return False, "File has fewer than 2 non-empty rows (header + at least one data row required)."
    return True, "OK"


def detect_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """Heuristic: return the first column that can be parsed as datetime."""
    for col in df.columns:
        sample = df[col].dropna().head(20).astype(str)
        try:
            pd.to_datetime(sample, infer_datetime_format=True)
            return col
        except Exception:
            continue
    return None


# ──────────────────────────────────────────────────────────────
# Core loader
# ──────────────────────────────────────────────────────────────

class DataLoader:
    """
    Parses a CSV file (path or in-memory bytes) into a clean DataFrame.

    Parameters
    ----------
    timestamp_col : str
        Name of the timestamp column in the source CSV.
    aggregate_col : str
        Name of the aggregate power column in the source CSV.
    appliance_col_map : dict[str, str]
        Mapping from canonical appliance key → CSV column name.
        Example: {"refrigerator": "fridge_w", "ceiling_fan": "fan_w"}
    power_unit : str
        One of "W" (watts, default) or "kW" (kilowatts, converted to W).
    resample_interval : str | None
        pandas offset alias for resampling, e.g. "1min", "30s".
        None means no resampling.
    """

    def __init__(
        self,
        timestamp_col: str = "timestamp",
        aggregate_col: str = "aggregate_w",
        appliance_col_map: Optional[dict] = None,
        power_unit: str = "W",
        resample_interval: Optional[str] = None,
    ) -> None:
        self.timestamp_col = timestamp_col
        self.aggregate_col = aggregate_col
        self.appliance_col_map = appliance_col_map or {}
        self.power_unit = power_unit.upper()
        self.resample_interval = resample_interval
        self._df: Optional[pd.DataFrame] = None

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def load(
        self,
        source: Union[str, Path, bytes, io.BytesIO],
    ) -> pd.DataFrame:
        """
        Load and validate data from a CSV source.

        Parameters
        ----------
        source : path-like, bytes, or BytesIO

        Returns
        -------
        pd.DataFrame  with DatetimeIndex, 'aggregate_w', and appliance cols.

        Raises
        ------
        ValueError  on schema or content problems.
        """
        raw_df = self._read_csv(source)
        df = self._map_columns(raw_df)
        df = self._parse_timestamps(df)
        df = self._convert_units(df)
        df = self._sort_and_dedup(df)
        if self.resample_interval:
            df = self._resample(df)
        self._validate_numeric(df)
        self._df = df
        return df

    def quality_report(self, df: Optional[pd.DataFrame] = None) -> dict:
        """Return a quality summary for the loaded DataFrame."""
        df = df if df is not None else self._df
        if df is None:
            raise RuntimeError("No data loaded yet. Call load() first.")

        n = len(df)
        duration = (df.index[-1] - df.index[0]) if n > 1 else pd.Timedelta(0)
        # Sampling interval stats
        if n > 1:
            diffs = df.index.to_series().diff().dropna()
            median_interval = diffs.median()
            irregular = int((diffs != median_interval).sum())
        else:
            median_interval = pd.Timedelta(0)
            irregular = 0

        report = {
            "total_rows": n,
            "duration": str(duration),
            "start": str(df.index[0]) if n else "N/A",
            "end": str(df.index[-1]) if n else "N/A",
            "median_interval_s": int(median_interval.total_seconds()) if n > 1 else 0,
            "irregular_intervals": irregular,
            "missing_aggregate": int(df[AGGREGATE_COL].isna().sum()),
            "negative_aggregate": int((df[AGGREGATE_COL] < 0).sum()),
            "appliances_available": [k for k in self.appliance_col_map],
        }
        for app_key in self.appliance_col_map:
            if app_key in df.columns:
                report[f"missing_{app_key}"] = int(df[app_key].isna().sum())
        return report

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _read_csv(self, source) -> pd.DataFrame:
        if isinstance(source, bytes):
            source = io.BytesIO(source)
        try:
            df = pd.read_csv(source, low_memory=False)
        except Exception as exc:
            raise ValueError(f"Could not parse CSV: {exc}") from exc
        if df.empty:
            raise ValueError("CSV parsed to an empty DataFrame.")
        return df

    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {self.timestamp_col, self.aggregate_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Required columns not found in CSV: {missing}. "
                f"Available columns: {list(df.columns)}"
            )
        keep = {self.timestamp_col: "timestamp", self.aggregate_col: AGGREGATE_COL}
        for app_key, csv_col in self.appliance_col_map.items():
            if csv_col in df.columns:
                keep[csv_col] = app_key
            else:
                logger.warning(
                    "Appliance column '%s' for '%s' not found; skipping.", csv_col, app_key
                )
        return df[list(keep.keys())].rename(columns=keep)

    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], infer_datetime_format=True, utc=False)
        except Exception as exc:
            raise ValueError(f"Could not parse timestamp column: {exc}") from exc
        df = df.set_index("timestamp")
        df.index.name = "timestamp"
        return df

    def _convert_units(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.power_unit == "KW":
            for col in [AGGREGATE_COL] + list(self.appliance_col_map.keys()):
                if col in df.columns:
                    df[col] = df[col] * 1000.0
        return df

    def _sort_and_dedup(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_index()
        n_before = len(df)
        # Keep the first occurrence for duplicate timestamps
        df = df[~df.index.duplicated(keep="first")]
        n_after = len(df)
        if n_before != n_after:
            logger.info("Removed %d duplicate timestamp(s).", n_before - n_after)
        return df

    def _resample(self, df: pd.DataFrame) -> pd.DataFrame:
        # Mean for power columns; forward-fill then drop remaining NaN
        df = df.resample(self.resample_interval).mean()
        df = df.ffill().dropna(subset=[AGGREGATE_COL])
        return df

    def _validate_numeric(self, df: pd.DataFrame) -> None:
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(
                    f"Column '{col}' is not numeric. Power readings must be numbers."
                )
        neg = (df[AGGREGATE_COL] < 0).sum()
        if neg:
            logger.warning(
                "%d negative aggregate readings found. These may indicate metering errors.", neg
            )


# ──────────────────────────────────────────────────────────────
# Sample dataset generator
# ──────────────────────────────────────────────────────────────

def generate_sample_dataset(
    n_samples: int = 1440,
    interval_seconds: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a deterministic synthetic labeled dataset for testing/demo.

    This is NOT real-world data.  Labels are generated from a simple
    rule-based schedule with added Gaussian noise.

    Returns a DataFrame with DatetimeIndex and columns:
      aggregate_w, refrigerator, ceiling_fan, air_conditioner,
      washing_machine, electric_kettle
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-15 00:00:00")
    idx = pd.date_range(start, periods=n_samples, freq=f"{interval_seconds}s")
    hours = idx.hour + idx.minute / 60.0

    def on_prob(start_h, end_h, prob=0.95):
        return np.where((hours >= start_h) & (hours < end_h), prob, 0.05)

    # Refrigerator: always cycling, ~150W average
    fridge_on = rng.random(n_samples) < 0.6
    fridge_w = np.where(fridge_on, rng.normal(150, 10, n_samples), rng.normal(5, 2, n_samples))
    fridge_w = np.clip(fridge_w, 0, None)

    # Ceiling fan: daytime use
    fan_on = rng.random(n_samples) < on_prob(7, 23)
    fan_w = np.where(fan_on, rng.normal(75, 5, n_samples), rng.normal(0, 1, n_samples))
    fan_w = np.clip(fan_w, 0, None)

    # Air conditioner: afternoon peak
    ac_on = rng.random(n_samples) < on_prob(12, 21, 0.7)
    ac_w = np.where(ac_on, rng.normal(1500, 80, n_samples), rng.normal(0, 5, n_samples))
    ac_w = np.clip(ac_w, 0, None)

    # Washing machine: morning use
    wm_on = rng.random(n_samples) < on_prob(8, 11, 0.4)
    wm_w = np.where(wm_on, rng.normal(500, 50, n_samples), rng.normal(0, 2, n_samples))
    wm_w = np.clip(wm_w, 0, None)

    # Electric kettle: breakfast and evening
    ek_prob = on_prob(7, 8.5, 0.6) + on_prob(17, 18, 0.5)
    ek_prob = np.clip(ek_prob, 0, 0.9)
    ek_on = rng.random(n_samples) < ek_prob
    ek_w = np.where(ek_on, rng.normal(2000, 100, n_samples), rng.normal(0, 1, n_samples))
    ek_w = np.clip(ek_w, 0, None)

    aggregate = fridge_w + fan_w + ac_w + wm_w + ek_w + rng.normal(30, 5, n_samples)
    aggregate = np.clip(aggregate, 0, None)

    df = pd.DataFrame(
        {
            AGGREGATE_COL: aggregate,
            "refrigerator": fridge_w,
            "ceiling_fan": fan_w,
            "air_conditioner": ac_w,
            "washing_machine": wm_w,
            "electric_kettle": ek_w,
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


def sample_csv_bytes() -> bytes:
    """Return sample dataset as UTF-8 CSV bytes for download."""
    df = generate_sample_dataset(n_samples=60)
    df_export = df.reset_index()
    df_export.rename(
        columns={AGGREGATE_COL: "aggregate_w"},
        inplace=True,
    )
    return df_export.to_csv(index=False).encode("utf-8")
