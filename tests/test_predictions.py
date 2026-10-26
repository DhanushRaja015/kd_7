"""
tests/test_predictions.py
Tests for training pipeline, prediction correctness, model save/load,
and no-label-leakage guarantee.
"""
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import generate_sample_dataset, AGGREGATE_COL, SUPPORTED_APPLIANCES
from src.features import FEATURE_COLUMNS, build_features
from src.predict import predict, validate_schema_compatibility
from src.preprocessing import NILMPreprocessor, chronological_split
from src.train import NILMTrainer, MedianBaseline


# ── Shared fixture ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def splits():
    df = generate_sample_dataset(n_samples=600, seed=99)
    train, val, test = chronological_split(df, 0.70, 0.15)
    prep = NILMPreprocessor()
    train_c = prep.fit_transform(train)
    val_c   = prep.transform(val)
    test_c  = prep.transform(test)
    return train_c, val_c, test_c


@pytest.fixture(scope="module")
def trained_trainer(splits):
    train, val, _ = splits
    trainer = NILMTrainer(
        appliance_keys=["refrigerator", "ceiling_fan"],
        n_estimators=10,
        random_state=0,
    )
    trainer.fit(train, val)
    return trainer


# ── NILMTrainer training ──────────────────────────────────────

def test_trainer_fits_without_error(splits):
    train, val, _ = splits
    trainer = NILMTrainer(["refrigerator"], n_estimators=5, random_state=1)
    trainer.fit(train)
    assert "refrigerator" in trainer.trained_appliances()


def test_trainer_skips_missing_appliance(splits):
    train, val, _ = splits
    trainer = NILMTrainer(["nonexistent_appliance"], n_estimators=5, random_state=1)
    trainer.fit(train)
    assert len(trainer.trained_appliances()) == 0


def test_training_summary_populated(trained_trainer):
    summ = trained_trainer.training_summary
    assert "refrigerator" in summ
    assert "train_samples" in summ["refrigerator"]


# ── No target label leakage ────────────────────────────────────

def test_appliance_column_not_in_feature_columns():
    """Ground-truth appliance columns must never appear in FEATURE_COLUMNS."""
    for app in SUPPORTED_APPLIANCES:
        assert app not in FEATURE_COLUMNS, (
            f"Appliance '{app}' found in FEATURE_COLUMNS — label leakage!"
        )


def test_features_use_only_aggregate(splits):
    train, _, _ = splits
    X = build_features(train)
    for app in SUPPORTED_APPLIANCES:
        assert app not in X.columns


# ── Prediction ────────────────────────────────────────────────

def test_predict_output_shape(trained_trainer, splits):
    _, _, test = splits
    pred = predict(trained_trainer, test, ["refrigerator"])
    assert f"refrigerator_pred_w" in pred.columns
    assert len(pred) == len(test)


def test_predict_no_negative_power(trained_trainer, splits):
    _, _, test = splits
    pred = predict(trained_trainer, test)
    for k in trained_trainer.trained_appliances():
        col = f"{k}_pred_w"
        if col in pred.columns:
            assert (pred[col] >= 0).all(), f"Negative predictions for {k}"


def test_predict_state_binary(trained_trainer, splits):
    _, _, test = splits
    pred = predict(trained_trainer, test)
    for k in trained_trainer.trained_appliances():
        state_col = f"{k}_pred_state"
        if state_col in pred.columns:
            assert set(pred[state_col].unique()).issubset({0, 1})


def test_predict_true_col_present_when_labels_exist(trained_trainer, splits):
    _, _, test = splits
    pred = predict(trained_trainer, test, ["refrigerator"])
    # Refrigerator labels exist in the test set
    assert "refrigerator_true_w" in pred.columns


def test_predict_true_col_absent_when_no_labels(trained_trainer, splits):
    _, _, test = splits
    # Drop all appliance columns
    test_agg_only = test[[AGGREGATE_COL]]
    pred = predict(trained_trainer, test_agg_only, ["refrigerator"])
    assert "refrigerator_true_w" not in pred.columns


def test_predict_requires_aggregate_col(trained_trainer, splits):
    _, _, test = splits
    bad = test.drop(columns=[AGGREGATE_COL])
    with pytest.raises(ValueError):
        predict(trained_trainer, bad)


# ── Model save / load ─────────────────────────────────────────

def test_save_and_load(trained_trainer, splits):
    _, _, test = splits
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "models"
        trained_trainer.save(model_dir)
        assert (model_dir / "training_summary.json").exists()

        loaded = NILMTrainer.load(model_dir)
        assert set(loaded.trained_appliances()) == set(trained_trainer.trained_appliances())

        # Predictions from loaded model must match
        pred_orig   = predict(trained_trainer, test)
        pred_loaded = predict(loaded, test)
        for k in trained_trainer.trained_appliances():
            col = f"{k}_pred_w"
            if col in pred_orig.columns and col in pred_loaded.columns:
                np.testing.assert_allclose(
                    pred_orig[col].values, pred_loaded[col].values, rtol=1e-5
                )


def test_load_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        NILMTrainer.load(Path("/nonexistent/path"))


# ── Baseline ──────────────────────────────────────────────────

def test_baseline_predict_power_shape():
    bl = MedianBaseline()
    y = pd.Series([100.0, 200.0, 150.0])
    bl.fit(y)
    out = bl.predict_power(10)
    assert len(out) == 10
    assert pytest.approx(out[0], abs=1e-4) == 150.0


def test_baseline_predict_state_binary():
    bl = MedianBaseline()
    bl.fit(pd.Series([5.0, 8.0, 200.0]))
    states = bl.predict_state(5)
    assert set(states).issubset({0, 1})


# ── Schema compatibility ──────────────────────────────────────

def test_schema_warns_missing_aggregate(trained_trainer):
    bad = pd.DataFrame({"other_col": [1, 2, 3]},
                       index=pd.date_range("2024-01-01", periods=3, freq="min"))
    warns = validate_schema_compatibility(bad, trained_trainer)
    assert any("aggregate" in w.lower() for w in warns)


def test_schema_ok_for_valid_data(trained_trainer, splits):
    _, _, test = splits
    warns = validate_schema_compatibility(test, trained_trainer)
    assert len(warns) == 0
