"""Model explanation: population-level importance and per-patient attribution.

Design choices
--------------
*Permutation importance, not ``feature_importances_``.* A tree ensemble's
built-in impurity importance is biased toward high-cardinality features -- here
it would systematically overrate ``AGE`` (67 distinct values) against the binary
symptoms (2 values each) regardless of predictive value. Permutation importance
measures the actual drop in a scoring metric when a column is shuffled.

*Counterfactual attribution, not SHAP.* Per-patient reasons are computed by
flipping one answer at a time and measuring the change in predicted risk. For
binary features this is exact and needs no extra dependency, and it maps
directly onto a question a clinician actually asks: "how much of this score is
driven by the cough?"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance

from .schema import FEATURE_NAMES, FEATURES_BY_NAME, NO, YES


@dataclass(frozen=True, slots=True)
class FeatureContribution:
    """How one answer moved a single patient's predicted risk."""

    feature: str
    label: str
    group: str
    patient_value: int
    counterfactual_value: int
    risk_with_patient_value: float
    risk_with_counterfactual: float

    @property
    def delta(self) -> float:
        """Change in absolute risk attributable to the patient's answer.

        Positive means this answer raised the patient's risk relative to the
        counterfactual.
        """
        return self.risk_with_patient_value - self.risk_with_counterfactual

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view."""
        return {
            "feature": self.feature,
            "label": self.label,
            "group": self.group,
            "patient_value": self.patient_value,
            "counterfactual_value": self.counterfactual_value,
            "risk_with_patient_value": round(self.risk_with_patient_value, 6),
            "risk_with_counterfactual": round(self.risk_with_counterfactual, 6),
            "delta": round(self.delta, 6),
        }


def population_importance(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series | np.ndarray,
    *,
    scoring: str = "roc_auc",
    n_repeats: int = 30,
    random_state: int = 42,
) -> list[dict[str, object]]:
    """Rank features by permutation importance.

    Args:
        estimator: Fitted estimator exposing ``predict_proba``.
        features: Held-out or full feature frame in canonical column order.
        target: Matching labels.
        scoring: Any scikit-learn scorer name; ROC-AUC by default because it is
            insensitive to the dataset's extreme class imbalance.
        n_repeats: Shuffles per feature. More repeats, tighter interval.
        random_state: Seed for reproducibility.

    Returns:
        Records sorted by mean importance descending, each with the mean and
        standard deviation of the score drop.
    """
    result = permutation_importance(
        estimator,
        features[list(FEATURE_NAMES)],
        np.asarray(target),
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    records = [
        {
            "feature": name,
            "label": FEATURES_BY_NAME[name].label,
            "importance_mean": round(float(mean), 6),
            "importance_std": round(float(std), 6),
            "scoring": scoring,
        }
        for name, mean, std in zip(
            FEATURE_NAMES, result.importances_mean, result.importances_std
        )
    ]
    records.sort(key=lambda r: r["importance_mean"], reverse=True)  # type: ignore[arg-type,return-value]
    return records


def explain_patient(
    predict_proba: "callable[[pd.DataFrame], np.ndarray]",
    patient: dict[str, int],
    *,
    reference: dict[str, int] | None = None,
    age_step: int = 10,
) -> list[FeatureContribution]:
    """Attribute a patient's predicted risk to individual answers.

    Each binary answer is flipped to its opposite and the model re-queried; age
    is shifted by ``age_step`` years. The resulting deltas show which answers
    are driving this particular prediction.

    Args:
        predict_proba: Callable mapping a feature frame to positive-class
            probabilities, typically ``ModelBundle.predict_proba``.
        patient: Encoded patient features, all of :data:`FEATURE_NAMES`.
        reference: Unused for binary flips; reserved for future baseline
            comparison. Accepted so callers can pass a bundle's reference
            patient without special-casing.
        age_step: Years to subtract from age for the age counterfactual.

    Returns:
        Contributions sorted by absolute effect, largest first.

    Raises:
        ValueError: If ``patient`` is missing a required feature.
    """
    del reference  # accepted for call-site symmetry; not needed for flips
    missing = [n for n in FEATURE_NAMES if n not in patient]
    if missing:
        raise ValueError(f"patient is missing features: {missing}")

    base_row = {name: int(patient[name]) for name in FEATURE_NAMES}
    variants: list[tuple[str, int, dict[str, int]]] = []
    for name in FEATURE_NAMES:
        spec = FEATURES_BY_NAME[name]
        counterfactual = dict(base_row)
        if spec.kind == "binary":
            flipped = NO if base_row[name] == YES else YES
        elif spec.kind == "sex":
            flipped = 1 - base_row[name]
        else:  # age
            flipped = max(18, base_row[name] - age_step)
        counterfactual[name] = flipped
        variants.append((name, flipped, counterfactual))

    # One batched call: the patient row followed by every counterfactual.
    frame = pd.DataFrame(
        [ [base_row[n] for n in FEATURE_NAMES] ]
        + [ [cf[n] for n in FEATURE_NAMES] for _, _, cf in variants ],
        columns=list(FEATURE_NAMES),
    )
    probabilities = np.asarray(predict_proba(frame), dtype=float)
    patient_risk = float(probabilities[0])

    contributions = [
        FeatureContribution(
            feature=name,
            label=FEATURES_BY_NAME[name].label,
            group=FEATURES_BY_NAME[name].group,
            patient_value=base_row[name],
            counterfactual_value=flipped,
            risk_with_patient_value=patient_risk,
            risk_with_counterfactual=float(probabilities[i + 1]),
        )
        for i, (name, flipped, _) in enumerate(variants)
    ]
    contributions.sort(key=lambda c: abs(c.delta), reverse=True)
    return contributions


def describe_contribution(contribution: FeatureContribution) -> str:
    """Render one contribution as a clinician-readable sentence.

    Args:
        contribution: The contribution to describe.

    Returns:
        A sentence describing the direction and size of the effect.
    """
    spec = FEATURES_BY_NAME[contribution.feature]
    delta_pp = contribution.delta * 100.0
    direction = "raises" if delta_pp >= 0 else "lowers"

    if spec.kind == "binary":
        answer = "yes" if contribution.patient_value == YES else "no"
        return (
            f"{spec.label} = {answer} {direction} this risk estimate by "
            f"{abs(delta_pp):.1f} percentage points versus the opposite answer."
        )
    if spec.kind == "sex":
        answer = "male" if contribution.patient_value == 1 else "female"
        other = "female" if contribution.patient_value == 1 else "male"
        return (
            f"Sex = {answer} {direction} this estimate by {abs(delta_pp):.1f} "
            f"percentage points versus {other}."
        )
    return (
        f"Age {contribution.patient_value} {direction} this estimate by "
        f"{abs(delta_pp):.1f} percentage points versus age "
        f"{contribution.counterfactual_value}."
    )


def top_contributions(
    contributions: Sequence[FeatureContribution],
    *,
    limit: int = 5,
    min_abs_delta: float = 0.005,
) -> list[FeatureContribution]:
    """Filter contributions down to the ones worth showing a user.

    Args:
        contributions: Output of :func:`explain_patient`.
        limit: Maximum number to return.
        min_abs_delta: Discard effects smaller than this in absolute risk;
            sub-half-a-percentage-point movements are noise, not explanation.

    Returns:
        Up to ``limit`` contributions exceeding ``min_abs_delta``.
    """
    return [c for c in contributions if abs(c.delta) >= min_abs_delta][:limit]
