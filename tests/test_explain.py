"""Tests for lcrisk.explain: permutation importance and counterfactual attribution."""

from __future__ import annotations

from pathlib import Path

import pytest

from lcrisk.data import load_survey_csv
from lcrisk.explain import describe_contribution, explain_patient, population_importance, top_contributions
from lcrisk.model import build_estimator
from lcrisk.schema import FEATURE_NAMES, NO, YES, encode_patient


@pytest.fixture
def fitted_estimator(learnable_csv: Path):
    features, target, _ = load_survey_csv(learnable_csv)
    estimator = build_estimator("logistic_regression")
    estimator.fit(features, target)
    return estimator, features, target


class TestPopulationImportance:
    def test_returns_one_record_per_feature(self, fitted_estimator) -> None:
        estimator, features, target = fitted_estimator
        records = population_importance(estimator, features, target, n_repeats=5)
        assert {r["feature"] for r in records} == set(FEATURE_NAMES)

    def test_sorted_descending_by_importance(self, fitted_estimator) -> None:
        estimator, features, target = fitted_estimator
        records = population_importance(estimator, features, target, n_repeats=5)
        values = [r["importance_mean"] for r in records]
        assert values == sorted(values, reverse=True)

    def test_known_driver_ranks_above_irrelevant_noise_feature(
        self, fitted_estimator
    ) -> None:
        # SMOKING is one of the three true drivers baked into the synthetic
        # dataset (see conftest.make_learnable_frame); PEER_PRESSURE carries no
        # signal there and should rank lower.
        estimator, features, target = fitted_estimator
        records = population_importance(estimator, features, target, n_repeats=20)
        by_name = {r["feature"]: r["importance_mean"] for r in records}
        assert by_name["SMOKING"] > by_name["PEER_PRESSURE"]


class TestExplainPatient:
    def test_returns_one_contribution_per_feature(self, symptomatic_patient) -> None:
        estimator, features, target = build_and_fit()
        encoded = encode_patient(symptomatic_patient)
        contributions = explain_patient(
            lambda frame: estimator.predict_proba(frame)[:, 1], encoded
        )
        assert {c.feature for c in contributions} == set(FEATURE_NAMES)

    def test_flipping_a_true_driver_changes_risk(self, symptomatic_patient) -> None:
        estimator, features, target = build_and_fit()
        encoded = encode_patient(symptomatic_patient)
        contributions = explain_patient(
            lambda frame: estimator.predict_proba(frame)[:, 1], encoded
        )
        smoking = next(c for c in contributions if c.feature == "SMOKING")
        assert smoking.delta != 0.0
        assert smoking.counterfactual_value != smoking.patient_value

    def test_sorted_by_absolute_effect_descending(self, symptomatic_patient) -> None:
        estimator, features, target = build_and_fit()
        encoded = encode_patient(symptomatic_patient)
        contributions = explain_patient(
            lambda frame: estimator.predict_proba(frame)[:, 1], encoded
        )
        deltas = [abs(c.delta) for c in contributions]
        assert deltas == sorted(deltas, reverse=True)

    def test_missing_feature_raises(self, symptomatic_patient) -> None:
        estimator, _, _ = build_and_fit()
        encoded = encode_patient(symptomatic_patient)
        del encoded["AGE"]
        with pytest.raises(ValueError):
            explain_patient(lambda frame: estimator.predict_proba(frame)[:, 1], encoded)


class TestTopContributions:
    def test_respects_limit(self, symptomatic_patient) -> None:
        estimator, _, _ = build_and_fit()
        encoded = encode_patient(symptomatic_patient)
        contributions = explain_patient(
            lambda frame: estimator.predict_proba(frame)[:, 1], encoded
        )
        top = top_contributions(contributions, limit=3, min_abs_delta=0.0)
        assert len(top) <= 3

    def test_filters_negligible_effects(self, symptomatic_patient) -> None:
        estimator, _, _ = build_and_fit()
        encoded = encode_patient(symptomatic_patient)
        contributions = explain_patient(
            lambda frame: estimator.predict_proba(frame)[:, 1], encoded
        )
        top = top_contributions(contributions, limit=20, min_abs_delta=0.9)
        assert all(abs(c.delta) >= 0.9 for c in top)


class TestDescribeContribution:
    def test_produces_readable_sentence_for_binary_feature(
        self, symptomatic_patient
    ) -> None:
        estimator, _, _ = build_and_fit()
        encoded = encode_patient(symptomatic_patient)
        contributions = explain_patient(
            lambda frame: estimator.predict_proba(frame)[:, 1], encoded
        )
        smoking = next(c for c in contributions if c.feature == "SMOKING")
        sentence = describe_contribution(smoking)
        assert "Smoking history" in sentence
        assert "percentage points" in sentence


def build_and_fit():
    """Helper: build and fit an estimator on module-level synthetic data."""
    import tempfile
    from pathlib import Path as _Path

    from conftest import make_learnable_frame

    frame = make_learnable_frame(n=300, seed=5)
    with tempfile.TemporaryDirectory() as tmp:
        path = _Path(tmp) / "data.csv"
        frame.to_csv(path, index=False)
        features, target, _ = load_survey_csv(path)
        estimator = build_estimator("logistic_regression")
        estimator.fit(features, target)
        return estimator, features, target
