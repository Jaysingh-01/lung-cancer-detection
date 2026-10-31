"""Tests for lcrisk.model: pipeline construction, calibration, bundle I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from lcrisk.data import load_survey_csv
from lcrisk.model import (
    ModelBundle,
    build_calibrated_estimator,
    build_estimator,
    build_provenance,
    dataset_fingerprint,
)
from lcrisk.schema import FEATURE_NAMES


class TestBuildEstimator:
    def test_returns_pipeline_with_scaler_first(self) -> None:
        pipeline = build_estimator("logistic_regression")
        assert isinstance(pipeline, Pipeline)
        assert pipeline.steps[0][0] == "scaler"

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(KeyError):
            build_estimator("not_a_real_model")

    def test_scaler_is_refit_per_fold_not_leaked(self, learnable_csv: Path) -> None:
        # The scaler lives inside the pipeline, so fitting on a subset must not
        # see statistics from rows outside that subset.
        features, target, _ = load_survey_csv(learnable_csv)
        half = len(features) // 2
        pipeline = build_estimator("logistic_regression")
        pipeline.fit(features.iloc[:half], target.iloc[:half])
        scaler_mean_subset = pipeline.named_steps["scaler"].mean_.copy()

        full_pipeline = build_estimator("logistic_regression")
        full_pipeline.fit(features, target)
        scaler_mean_full = full_pipeline.named_steps["scaler"].mean_

        assert not np.allclose(scaler_mean_subset, scaler_mean_full)


class TestBuildCalibratedEstimator:
    def test_none_returns_uncalibrated_pipeline(self) -> None:
        estimator = build_calibrated_estimator("logistic_regression", method="none")
        assert isinstance(estimator, Pipeline)

    def test_sigmoid_wraps_in_calibration(self) -> None:
        estimator = build_calibrated_estimator("logistic_regression", method="sigmoid")
        assert type(estimator).__name__ == "CalibratedClassifierCV"

    def test_rejects_unknown_method(self) -> None:
        with pytest.raises(ValueError):
            build_calibrated_estimator("logistic_regression", method="bogus")  # type: ignore[arg-type]

    def test_fits_and_predicts_probabilities_in_valid_range(
        self, learnable_csv: Path
    ) -> None:
        features, target, _ = load_survey_csv(learnable_csv)
        estimator = build_calibrated_estimator(
            "logistic_regression", method="sigmoid", cv=3
        )
        estimator.fit(features, target)
        proba = estimator.predict_proba(features)
        assert proba.shape == (len(features), 2)
        assert np.all((proba >= 0) & (proba <= 1))
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestModelBundlePersistence:
    def test_round_trips_through_save_and_load(
        self, tmp_path: Path, learnable_csv: Path
    ) -> None:
        features, target, _ = load_survey_csv(learnable_csv)
        estimator = build_calibrated_estimator("logistic_regression", method="sigmoid")
        estimator.fit(features, target)

        bundle = ModelBundle(
            estimator=estimator,
            feature_names=list(features.columns),
            threshold=0.6,
            model_name="logistic_regression",
            calibration_method="sigmoid",
            metrics={"cross_validated": {"point": {"sensitivity": 0.9}}},
        )
        out_dir = bundle.save(tmp_path / "bundle")
        loaded = ModelBundle.load(out_dir)

        assert loaded.model_name == "logistic_regression"
        assert loaded.threshold == pytest.approx(0.6)
        assert loaded.is_calibrated
        original_proba = bundle.predict_proba(features)
        loaded_proba = loaded.predict_proba(features)
        assert np.allclose(original_proba, loaded_proba)

    def test_load_missing_estimator_raises(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            ModelBundle.load(empty_dir)

    def test_load_rejects_schema_mismatch(
        self, tmp_path: Path, learnable_csv: Path
    ) -> None:
        features, target, _ = load_survey_csv(learnable_csv)
        estimator = build_estimator("logistic_regression")
        estimator.fit(features, target)
        bundle = ModelBundle(
            estimator=estimator,
            feature_names=list(features.columns)[:-1],  # drop one feature: mismatch
            threshold=0.5,
            model_name="logistic_regression",
            calibration_method="none",
        )
        out_dir = bundle.save(tmp_path / "mismatched")
        with pytest.raises(ValueError, match="does not match"):
            ModelBundle.load(out_dir)

    def test_predict_proba_requires_all_feature_columns(
        self, learnable_csv: Path
    ) -> None:
        features, target, _ = load_survey_csv(learnable_csv)
        estimator = build_estimator("logistic_regression")
        estimator.fit(features, target)
        bundle = ModelBundle(
            estimator=estimator,
            feature_names=list(features.columns),
            threshold=0.5,
            model_name="logistic_regression",
            calibration_method="none",
        )
        with pytest.raises(ValueError, match="missing"):
            bundle.predict_proba(features.drop(columns=["AGE"]))


class TestProvenance:
    def test_fingerprint_is_stable_for_identical_data(self, learnable_csv: Path) -> None:
        features, target, _ = load_survey_csv(learnable_csv)
        fp1 = dataset_fingerprint(features, target)
        fp2 = dataset_fingerprint(features.copy(), target.copy())
        assert fp1 == fp2

    def test_fingerprint_changes_with_data(self, learnable_csv: Path) -> None:
        features, target, _ = load_survey_csv(learnable_csv)
        fp_original = dataset_fingerprint(features, target)
        mutated = features.copy()
        mutated.loc[0, "AGE"] = mutated.loc[0, "AGE"] + 1
        fp_mutated = dataset_fingerprint(mutated, target)
        assert fp_original != fp_mutated

    def test_provenance_records_versions(self, learnable_csv: Path) -> None:
        features, target, _ = load_survey_csv(learnable_csv)
        record = build_provenance(
            data_path=learnable_csv, features=features, target=target, random_state=42
        )
        assert record["scikit_learn_version"]
        assert record["n_training_rows"] == len(features)
        assert record["random_state"] == 42
