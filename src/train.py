"""
train.py
--------
Model training, persistence, and baseline implementation for NILM.

Architecture
------------
For each appliance two separate models are trained:
  1. A Random Forest regressor  → estimates power in watts.
  2. A Random Forest classifier → predicts ON/OFF state (derived from
     a threshold applied to the ground-truth label).

A baseline model (median-power estimator) is also fitted for comparison.

No ground-truth appliance column is used as a feature.
Only features derived from the aggregate signal are used as inputs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    RandomForestClassifier = None  # type: ignore
    RandomForestRegressor = None   # type: ignore

from src.data_loader import AGGREGATE_COL
from src.features import build_features, align_features_labels, FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
ON_THRESHOLD_W = 10.0          # Watts above which an appliance is considered ON
DEFAULT_MODEL_DIR = Path("models")


# ──────────────────────────────────────────────────────────────
# Baseline
# ──────────────────────────────────────────────────────────────

class MedianBaseline:
    """
    Trivial baseline: always predicts the training-set median power.

    For ON/OFF it uses a threshold on the median prediction.
    """

    def __init__(self) -> None:
        self.median_power: Optional[float] = None

    def fit(self, y_train: pd.Series) -> "MedianBaseline":
        self.median_power = float(np.median(y_train.dropna()))
        return self

    def predict_power(self, n: int) -> np.ndarray:
        if self.median_power is None:
            raise RuntimeError("Baseline not fitted.")
        return np.full(n, self.median_power)

    def predict_state(self, n: int) -> np.ndarray:
        if self.median_power is None:
            raise RuntimeError("Baseline not fitted.")
        state = 1 if self.median_power >= ON_THRESHOLD_W else 0
        return np.full(n, state, dtype=int)


# ──────────────────────────────────────────────────────────────
# NILM Trainer
# ──────────────────────────────────────────────────────────────

class NILMTrainer:
    """
    Trains per-appliance Random Forest regressor and classifier models.

    Parameters
    ----------
    appliance_keys : list[str]
        Canonical appliance keys to train (must appear as columns in data).
    n_estimators : int
        Number of trees in each Random Forest.
    max_depth : int | None
        Maximum tree depth.  None = unlimited.
    on_threshold_w : float
        Watts threshold for binarising labels into ON/OFF states.
    random_state : int
    """

    def __init__(
        self,
        appliance_keys: list[str],
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        on_threshold_w: float = ON_THRESHOLD_W,
        random_state: int = 42,
    ) -> None:
        self.appliance_keys = appliance_keys
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.on_threshold_w = on_threshold_w
        self.random_state = random_state

        # Fitted objects
        self.regressors: dict[str, RandomForestRegressor] = {}
        self.classifiers: dict[str, RandomForestClassifier] = {}
        self.baselines: dict[str, MedianBaseline] = {}
        self.training_summary: dict = {}
        self._feature_columns = FEATURE_COLUMNS

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "NILMTrainer":
        """
        Fit models for each configured appliance.

        Parameters
        ----------
        train_df : pd.DataFrame
            Must have DatetimeIndex, AGGREGATE_COL, and appliance columns.
        val_df : pd.DataFrame | None
            Optional validation set for logging (not used to fit).
        """
        X_train_full = build_features(train_df)

        for app_key in self.appliance_keys:
            if app_key not in train_df.columns:
                logger.warning(
                    "Appliance '%s' not found in training data; skipping.", app_key
                )
                continue

            logger.info("Training models for appliance: %s", app_key)

            y_power = train_df[app_key]
            y_state = (y_power >= self.on_threshold_w).astype(int)

            X, y_p = align_features_labels(X_train_full, y_power)
            _, y_s = align_features_labels(X_train_full, y_state)

            if len(X) < 10:
                logger.warning(
                    "Too few samples (%d) for appliance '%s'; skipping.", len(X), app_key
                )
                continue

            # Regressor
            reg = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
            reg.fit(X, y_p)
            self.regressors[app_key] = reg

            # Classifier
            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
            clf.fit(X, y_s)
            self.classifiers[app_key] = clf

            # Baseline
            bl = MedianBaseline().fit(y_p)
            self.baselines[app_key] = bl

            self.training_summary[app_key] = {
                "train_samples": len(X),
                "power_mean_train": round(float(y_p.mean()), 2),
                "on_fraction_train": round(float(y_s.mean()), 3),
                "baseline_median_w": round(bl.median_power, 2),
            }

        logger.info("Training complete.  Appliances: %s", list(self.regressors.keys()))
        return self

    def save(self, model_dir: Path = DEFAULT_MODEL_DIR) -> dict[str, Path]:
        """Persist all fitted models to disk."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        saved: dict[str, Path] = {}

        for app_key, reg in self.regressors.items():
            p = model_dir / f"{app_key}_regressor.joblib"
            joblib.dump(reg, p)
            saved[f"{app_key}_reg"] = p

        for app_key, clf in self.classifiers.items():
            p = model_dir / f"{app_key}_classifier.joblib"
            joblib.dump(clf, p)
            saved[f"{app_key}_clf"] = p

        for app_key, bl in self.baselines.items():
            p = model_dir / f"{app_key}_baseline.joblib"
            joblib.dump(bl, p)
            saved[f"{app_key}_baseline"] = p

        # Summary JSON
        summary_path = model_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "appliance_keys": self.appliance_keys,
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "on_threshold_w": self.on_threshold_w,
                    "random_state": self.random_state,
                    "per_appliance": self.training_summary,
                },
                f,
                indent=2,
            )
        saved["summary"] = summary_path
        return saved

    @classmethod
    def load(cls, model_dir: Path = DEFAULT_MODEL_DIR) -> "NILMTrainer":
        """Load a previously saved NILMTrainer from disk."""
        model_dir = Path(model_dir)
        summary_path = model_dir / "training_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"No training_summary.json found in {model_dir}")

        with open(summary_path) as f:
            meta = json.load(f)

        trainer = cls(
            appliance_keys=meta["appliance_keys"],
            n_estimators=meta.get("n_estimators", 100),
            max_depth=meta.get("max_depth"),
            on_threshold_w=meta.get("on_threshold_w", ON_THRESHOLD_W),
            random_state=meta.get("random_state", 42),
        )
        trainer.training_summary = meta.get("per_appliance", {})

        for app_key in meta["appliance_keys"]:
            reg_p = model_dir / f"{app_key}_regressor.joblib"
            clf_p = model_dir / f"{app_key}_classifier.joblib"
            bl_p = model_dir / f"{app_key}_baseline.joblib"

            if reg_p.exists():
                trainer.regressors[app_key] = joblib.load(reg_p)
            if clf_p.exists():
                trainer.classifiers[app_key] = joblib.load(clf_p)
            if bl_p.exists():
                trainer.baselines[app_key] = joblib.load(bl_p)

        return trainer

    def trained_appliances(self) -> list[str]:
        return list(self.regressors.keys())
