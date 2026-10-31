"""Tests for lcrisk.evaluate: clinical metrics, calibration, PPV/NPV transport."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lcrisk.evaluate import (
    bootstrap_ci,
    calibration_curve_points,
    clinical_metrics,
    confusion_counts,
    expected_calibration_error,
    npv_at_prevalence,
    ppv_at_prevalence,
)


class TestConfusionCounts:
    def test_perfect_predictions(self) -> None:
        counts = confusion_counts([0, 0, 1, 1], [0, 0, 1, 1])
        assert (counts.tn, counts.fp, counts.fn, counts.tp) == (2, 0, 0, 2)

    def test_all_wrong(self) -> None:
        counts = confusion_counts([0, 0, 1, 1], [1, 1, 0, 0])
        assert (counts.tn, counts.fp, counts.fn, counts.tp) == (0, 2, 2, 0)

    def test_total_matches_input_length(self) -> None:
        counts = confusion_counts([0, 1, 1, 0, 1], [0, 1, 0, 0, 1])
        assert counts.n == 5


class TestClinicalMetrics:
    def test_perfect_classifier(self) -> None:
        y_true = [0] * 10 + [1] * 10
        y_prob = [0.05] * 10 + [0.95] * 10
        metrics = clinical_metrics(y_true, y_prob, threshold=0.5)
        assert metrics.sensitivity == 1.0
        assert metrics.specificity == 1.0
        assert metrics.roc_auc == 1.0
        assert metrics.accuracy == 1.0

    def test_threshold_shifts_sensitivity_specificity_tradeoff(self) -> None:
        y_true = [0, 0, 0, 1, 1, 1]
        y_prob = [0.1, 0.4, 0.6, 0.5, 0.7, 0.9]
        low = clinical_metrics(y_true, y_prob, threshold=0.2)
        high = clinical_metrics(y_true, y_prob, threshold=0.8)
        assert low.sensitivity >= high.sensitivity
        assert low.specificity <= high.specificity

    def test_rejects_invalid_threshold(self) -> None:
        with pytest.raises(ValueError):
            clinical_metrics([0, 1], [0.1, 0.9], threshold=1.5)

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError):
            clinical_metrics([0, 1, 1], [0.1, 0.9], threshold=0.5)

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError):
            clinical_metrics([], [], threshold=0.5)

    def test_single_class_yields_nan_discrimination_not_crash(self) -> None:
        metrics = clinical_metrics([1, 1, 1], [0.6, 0.7, 0.8], threshold=0.5)
        assert math.isnan(metrics.roc_auc)
        assert math.isnan(metrics.specificity)  # no negatives to compute it from

    def test_ppv_npv_adjusted_differ_from_raw_at_different_prevalence(self) -> None:
        # Imperfect classifier: some false positives and false negatives, so
        # PPV genuinely depends on prevalence rather than being trivially 1.0.
        y_true = [0] * 5 + [1] * 45  # 90% prevalence sample
        y_prob = [0.2, 0.2, 0.2, 0.2, 0.8] + [0.8] * 40 + [0.2] * 5
        metrics = clinical_metrics(
            y_true, y_prob, threshold=0.5, reference_prevalence=0.01
        )
        assert metrics.ppv > metrics.ppv_adjusted  # raw PPV inflated by high prevalence
        assert metrics.npv_adjusted > 0.9  # NPV stays high in a rare-disease setting


class TestPredictiveValueTransport:
    def test_ppv_increases_with_prevalence(self) -> None:
        low = ppv_at_prevalence(0.9, 0.9, prevalence=0.01)
        high = ppv_at_prevalence(0.9, 0.9, prevalence=0.5)
        assert low < high

    def test_npv_decreases_with_prevalence(self) -> None:
        low_prev_npv = npv_at_prevalence(0.9, 0.9, prevalence=0.01)
        high_prev_npv = npv_at_prevalence(0.9, 0.9, prevalence=0.5)
        assert low_prev_npv > high_prev_npv

    def test_perfect_test_has_ppv_and_npv_of_one(self) -> None:
        assert ppv_at_prevalence(1.0, 1.0, prevalence=0.3) == pytest.approx(1.0)
        assert npv_at_prevalence(1.0, 1.0, prevalence=0.3) == pytest.approx(1.0)

    def test_rejects_invalid_prevalence(self) -> None:
        with pytest.raises(ValueError):
            ppv_at_prevalence(0.9, 0.9, prevalence=1.5)
        with pytest.raises(ValueError):
            npv_at_prevalence(0.9, 0.9, prevalence=-0.1)


class TestCalibration:
    def test_perfectly_calibrated_predictions_have_zero_ece(self) -> None:
        rng = np.random.default_rng(0)
        prob = rng.uniform(0, 1, size=20000)
        truth = (rng.uniform(0, 1, size=20000) < prob).astype(int)
        ece = expected_calibration_error(truth, prob, n_bins=10)
        assert ece < 0.02

    def test_badly_calibrated_predictions_have_high_ece(self) -> None:
        # Always predicts 0.99 regardless of true rate of 0.1.
        truth = np.array([0] * 90 + [1] * 10)
        prob = np.full(100, 0.99)
        ece = expected_calibration_error(truth, prob, n_bins=10)
        assert ece > 0.5

    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError):
            expected_calibration_error([0, 1], [0.5, 0.5, 0.5])

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            expected_calibration_error([], [])

    def test_calibration_curve_points_cover_provided_bins_only(self) -> None:
        truth = [0, 1, 0, 1]
        prob = [0.1, 0.9, 0.1, 0.9]
        points = calibration_curve_points(truth, prob, n_bins=10)
        assert len(points) == 2  # only two occupied bins
        assert all(0.0 <= p["mean_predicted"] <= 1.0 for p in points)


class TestBootstrapCi:
    def test_interval_contains_point_estimate(self) -> None:
        rng = np.random.default_rng(1)
        truth = rng.integers(0, 2, size=200)
        prob = np.clip(truth * 0.6 + rng.normal(0, 0.2, size=200) + 0.2, 0, 1)
        point = clinical_metrics(truth, prob, threshold=0.5).sensitivity
        low, high = bootstrap_ci(
            truth, prob, metric="sensitivity", threshold=0.5, n_resamples=300
        )
        assert low <= point + 1e-9
        assert high >= point - 1e-9

    def test_rejects_unknown_metric(self) -> None:
        with pytest.raises(ValueError):
            bootstrap_ci([0, 1], [0.1, 0.9], metric="not_a_metric", threshold=0.5)

    def test_single_class_returns_nan_interval(self) -> None:
        low, high = bootstrap_ci(
            [1, 1, 1], [0.6, 0.7, 0.8], metric="sensitivity", threshold=0.5
        )
        assert math.isnan(low)
        assert math.isnan(high)
