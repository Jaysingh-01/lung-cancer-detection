"""Tests for lcrisk.predict: the RiskAssessor service layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from lcrisk.data import load_survey_csv, reference_patient
from lcrisk.model import ModelBundle, build_calibrated_estimator
from lcrisk.predict import (
    BAND_ELEVATED,
    BAND_HIGH,
    BAND_LOW,
    RiskAssessor,
    blank_patient,
    load_assessor,
)
from lcrisk.schema import EncodingError, FEATURE_NAMES
from lcrisk.threshold import select_threshold


@pytest.fixture
def trained_assessor(learnable_csv: Path) -> RiskAssessor:
    features, target, _ = load_survey_csv(learnable_csv)
    estimator = build_calibrated_estimator("logistic_regression", method="sigmoid", cv=3)
    estimator.fit(features, target)
    proba = estimator.predict_proba(features)[:, 1]
    point = select_threshold(target.to_numpy(), proba, strategy="youden")
    bundle = ModelBundle(
        estimator=estimator,
        feature_names=list(features.columns),
        threshold=point.threshold,
        model_name="logistic_regression",
        calibration_method="sigmoid",
        metrics={
            "cross_validated": {
                "point": {
                    "sensitivity": point.achieved_sensitivity,
                    "specificity": point.achieved_specificity,
                    "expected_calibration_error": 0.05,
                }
            }
        },
        reference_patient=reference_patient(features),
    )
    return RiskAssessor(bundle)


class TestBlankPatient:
    def test_contains_all_features(self) -> None:
        patient = blank_patient()
        assert set(patient) == set(FEATURE_NAMES)

    def test_is_encodable(self) -> None:
        from lcrisk.schema import encode_patient

        encode_patient(blank_patient())  # must not raise


class TestRiskAssessorAssess:
    def test_symptomatic_patient_scores_higher_than_healthy(
        self, trained_assessor: RiskAssessor, healthy_patient, symptomatic_patient
    ) -> None:
        healthy_result = trained_assessor.assess(healthy_patient)
        sick_result = trained_assessor.assess(symptomatic_patient)
        assert sick_result.probability > healthy_result.probability

    def test_no_and_yes_smoking_produce_different_scores(
        self, trained_assessor: RiskAssessor, healthy_patient
    ) -> None:
        no_smoking = dict(healthy_patient, SMOKING="No")
        yes_smoking = dict(healthy_patient, SMOKING="Yes")
        result_no = trained_assessor.assess(no_smoking)
        result_yes = trained_assessor.assess(yes_smoking)
        assert result_no.probability != result_yes.probability

    def test_flagged_matches_threshold_comparison(
        self, trained_assessor: RiskAssessor, symptomatic_patient
    ) -> None:
        result = trained_assessor.assess(symptomatic_patient)
        assert result.flagged == (result.probability >= result.threshold)

    def test_band_is_one_of_the_known_bands(
        self, trained_assessor: RiskAssessor, healthy_patient
    ) -> None:
        result = trained_assessor.assess(healthy_patient)
        assert result.band in (BAND_LOW, BAND_ELEVATED, BAND_HIGH, "borderline")

    def test_invalid_answer_raises_encoding_error(
        self, trained_assessor: RiskAssessor, healthy_patient
    ) -> None:
        bad = dict(healthy_patient, SMOKING="unsure")
        with pytest.raises(EncodingError):
            trained_assessor.assess(bad)

    def test_explain_false_skips_contributions(
        self, trained_assessor: RiskAssessor, symptomatic_patient
    ) -> None:
        result = trained_assessor.assess(symptomatic_patient, explain=False)
        assert result.contributions == []

    def test_explain_true_returns_bounded_contributions(
        self, trained_assessor: RiskAssessor, symptomatic_patient
    ) -> None:
        result = trained_assessor.assess(symptomatic_patient, max_reasons=3)
        assert len(result.contributions) <= 3

    def test_uncertainty_note_mentions_calibration_error(
        self, trained_assessor: RiskAssessor, healthy_patient
    ) -> None:
        result = trained_assessor.assess(healthy_patient)
        assert "calibration error" in result.uncertainty_note().lower()


class TestPredictiveValuesAt:
    def test_returns_ppv_and_npv(self, trained_assessor: RiskAssessor) -> None:
        values = trained_assessor.predictive_values_at(0.01)
        assert values is not None
        assert 0.0 <= values["ppv"] <= 1.0
        assert 0.0 <= values["npv"] <= 1.0

    def test_lower_prevalence_lowers_ppv(self, trained_assessor: RiskAssessor) -> None:
        low = trained_assessor.predictive_values_at(0.01)
        high = trained_assessor.predictive_values_at(0.5)
        assert low is not None and high is not None
        assert low["ppv"] <= high["ppv"]


class TestLoadAssessor:
    def test_loads_and_caches_from_directory(
        self, tmp_path: Path, learnable_csv: Path
    ) -> None:
        features, target, _ = load_survey_csv(learnable_csv)
        estimator = build_calibrated_estimator("logistic_regression", method="sigmoid", cv=3)
        estimator.fit(features, target)
        bundle = ModelBundle(
            estimator=estimator,
            feature_names=list(features.columns),
            threshold=0.5,
            model_name="logistic_regression",
            calibration_method="sigmoid",
        )
        out_dir = bundle.save(tmp_path / "bundle_for_loading")

        load_assessor.cache_clear()
        assessor_a = load_assessor(str(out_dir))
        assessor_b = load_assessor(str(out_dir))
        assert assessor_a is assessor_b  # cached, not reloaded
