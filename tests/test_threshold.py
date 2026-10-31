"""Tests for lcrisk.threshold: operating-point selection."""

from __future__ import annotations

import numpy as np
import pytest

from lcrisk.threshold import select_threshold, threshold_sweep


def _separable_scores(n: int = 200, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Labels and scores with real separation but some overlap."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 2, size=n)
    prob = np.clip(truth * 0.5 + rng.normal(0.25, 0.2, size=n), 0, 1)
    return truth, prob


class TestSelectThreshold:
    def test_target_sensitivity_meets_or_exceeds_target(self) -> None:
        truth, prob = _separable_scores()
        point = select_threshold(
            truth, prob, strategy="target_sensitivity", target=0.9
        )
        assert point.achieved_sensitivity >= 0.9 - 1e-9

    def test_target_sensitivity_maximises_specificity_among_eligible(self) -> None:
        truth, prob = _separable_scores()
        loose = select_threshold(truth, prob, strategy="target_sensitivity", target=0.5)
        strict = select_threshold(truth, prob, strategy="target_sensitivity", target=0.95)
        # A higher sensitivity floor cannot achieve better specificity.
        assert strict.achieved_specificity <= loose.achieved_specificity + 1e-9

    def test_target_specificity_meets_or_exceeds_target(self) -> None:
        truth, prob = _separable_scores()
        point = select_threshold(
            truth, prob, strategy="target_specificity", target=0.9
        )
        assert point.achieved_specificity >= 0.9 - 1e-9

    def test_youden_returns_valid_point(self) -> None:
        truth, prob = _separable_scores()
        point = select_threshold(truth, prob, strategy="youden")
        assert 0.0 <= point.threshold <= 1.0
        assert point.target is None

    def test_f1_returns_valid_point(self) -> None:
        truth, prob = _separable_scores()
        point = select_threshold(truth, prob, strategy="f1")
        assert 0.0 <= point.threshold <= 1.0

    def test_unreachable_target_falls_back_gracefully(self) -> None:
        # A weak classifier cannot reach 99.9% sensitivity at any threshold
        # short of flagging nearly everyone; the function must not raise.
        truth = np.array([0, 0, 0, 0, 1, 1])
        prob = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        point = select_threshold(
            truth, prob, strategy="target_sensitivity", target=0.999
        )
        assert 0.0 <= point.threshold <= 1.0

    def test_rejects_single_class_input(self) -> None:
        with pytest.raises(ValueError):
            select_threshold([1, 1, 1], [0.5, 0.6, 0.7], strategy="youden")

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError):
            select_threshold([], [], strategy="youden")

    def test_rejects_bad_target(self) -> None:
        truth, prob = _separable_scores()
        with pytest.raises(ValueError):
            select_threshold(truth, prob, strategy="target_sensitivity", target=1.5)

    def test_rejects_unknown_strategy(self) -> None:
        truth, prob = _separable_scores()
        with pytest.raises(ValueError):
            select_threshold(truth, prob, strategy="nonsense")  # type: ignore[arg-type]


class TestThresholdSweep:
    def test_returns_requested_number_of_points(self) -> None:
        truth, prob = _separable_scores()
        rows = threshold_sweep(truth, prob, n_points=11)
        assert len(rows) == 11

    def test_sensitivity_decreases_as_threshold_rises(self) -> None:
        truth, prob = _separable_scores()
        rows = threshold_sweep(truth, prob, n_points=21)
        sensitivities = [r["sensitivity"] for r in rows if r["sensitivity"] == r["sensitivity"]]
        assert sensitivities == sorted(sensitivities, reverse=True)
