"""Decision-threshold selection for a screening instrument.

The original app hard-coded 0.5. That is defensible only when false positives
and false negatives cost the same amount, which is never true in cancer
screening: a missed cancer is far worse than a follow-up test. The threshold is
therefore a clinical policy choice, selected on data and recorded with the
model so the operating point is auditable rather than implicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from sklearn.metrics import roc_curve

Strategy = Literal["target_sensitivity", "target_specificity", "youden", "f1"]


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """A selected decision threshold and the trade-off it implies."""

    threshold: float
    strategy: str
    achieved_sensitivity: float
    achieved_specificity: float
    target: float | None
    rationale: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view."""
        return {
            "threshold": round(self.threshold, 6),
            "strategy": self.strategy,
            "achieved_sensitivity": round(self.achieved_sensitivity, 6),
            "achieved_specificity": round(self.achieved_specificity, 6),
            "target": self.target,
            "rationale": self.rationale,
        }


def select_threshold(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    strategy: Strategy = "target_sensitivity",
    target: float = 0.95,
) -> OperatingPoint:
    """Choose a decision threshold from out-of-fold predictions.

    Strategies:
        ``target_sensitivity``
            Lowest threshold achieving at least ``target`` sensitivity, then the
            most specific among those. The screening default: fix the miss rate
            you are willing to tolerate, then minimise false alarms.
        ``target_specificity``
            Mirror image, for confirmatory rather than screening use.
        ``youden``
            Maximise Youden's J (sensitivity + specificity - 1), the point
            furthest from chance. Weights both errors equally.
        ``f1``
            Maximise F1. Included for comparison with the ML literature; it
            ignores true negatives and is a poor fit for screening.

    Args:
        y_true: Ground-truth 0/1 labels, ideally out-of-fold.
        y_prob: Predicted probabilities from the same out-of-fold procedure.
        strategy: Selection rule, see above.
        target: Required sensitivity or specificity for the ``target_*``
            strategies. Ignored otherwise.

    Returns:
        The selected :class:`OperatingPoint`.

    Raises:
        ValueError: If inputs are degenerate, ``target`` is outside [0, 1], or
            ``strategy`` is unknown.
    """
    truth = np.asarray(y_true).astype(int)
    prob = np.asarray(y_prob, dtype=float)
    if truth.size == 0:
        raise ValueError("cannot select a threshold from an empty sample")
    if truth.shape != prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if not 0 < truth.sum() < truth.size:
        raise ValueError("threshold selection requires both classes to be present")
    if not 0.0 <= target <= 1.0:
        raise ValueError(f"target must be in [0, 1], got {target}")

    fpr, tpr, thresholds = roc_curve(truth, prob)
    specificity = 1.0 - fpr
    # roc_curve prepends an inf threshold for the "flag nothing" corner.
    thresholds = np.where(np.isinf(thresholds), 1.0, thresholds)

    if strategy == "target_sensitivity":
        eligible = np.flatnonzero(tpr >= target)
        if eligible.size == 0:
            best = int(np.argmax(tpr))
            rationale = (
                f"No threshold reaches {target:.0%} sensitivity on this sample; "
                f"selected the maximum achievable ({tpr[best]:.1%})."
            )
        else:
            best = int(eligible[np.argmax(specificity[eligible])])
            rationale = (
                f"Lowest-alarm threshold that still detects at least "
                f"{target:.0%} of cases. Chosen because a missed cancer costs "
                "far more than a follow-up test."
            )
    elif strategy == "target_specificity":
        eligible = np.flatnonzero(specificity >= target)
        if eligible.size == 0:
            best = int(np.argmax(specificity))
            rationale = (
                f"No threshold reaches {target:.0%} specificity; selected the "
                f"maximum achievable ({specificity[best]:.1%})."
            )
        else:
            best = int(eligible[np.argmax(tpr[eligible])])
            rationale = (
                f"Most sensitive threshold holding specificity at or above "
                f"{target:.0%}."
            )
    elif strategy == "youden":
        best = int(np.argmax(tpr - fpr))
        rationale = (
            "Maximises Youden's J (sensitivity + specificity - 1), weighting "
            "false negatives and false positives equally."
        )
    elif strategy == "f1":
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = [
                _f1_at(truth, prob, t) for t in thresholds
            ]
        best = int(np.nanargmax(scores))
        rationale = "Maximises F1; ignores true negatives, shown for comparison."
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    return OperatingPoint(
        threshold=float(thresholds[best]),
        strategy=strategy,
        achieved_sensitivity=float(tpr[best]),
        achieved_specificity=float(specificity[best]),
        target=float(target) if strategy.startswith("target_") else None,
        rationale=rationale,
    )


def _f1_at(truth: np.ndarray, prob: np.ndarray, threshold: float) -> float:
    """Compute F1 at one threshold, returning NaN when undefined."""
    pred = (prob >= threshold).astype(int)
    tp = int(np.sum((truth == 1) & (pred == 1)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))
    denominator = 2 * tp + fp + fn
    return float("nan") if denominator == 0 else 2 * tp / denominator


def threshold_sweep(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    n_points: int = 21,
) -> list[dict[str, float]]:
    """Tabulate the sensitivity/specificity trade-off across thresholds.

    Publishing the whole curve lets a reader pick a different operating point
    for their own cost trade-off instead of inheriting this project's.

    Args:
        y_true: Ground-truth 0/1 labels.
        y_prob: Predicted probabilities.
        n_points: Number of evenly spaced thresholds in [0, 1].

    Returns:
        One record per threshold with sensitivity, specificity, and counts.
    """
    truth = np.asarray(y_true).astype(int)
    prob = np.asarray(y_prob, dtype=float)
    rows: list[dict[str, float]] = []
    for t in np.linspace(0.0, 1.0, n_points):
        pred = (prob >= t).astype(int)
        tp = int(np.sum((truth == 1) & (pred == 1)))
        fp = int(np.sum((truth == 0) & (pred == 1)))
        fn = int(np.sum((truth == 1) & (pred == 0)))
        tn = int(np.sum((truth == 0) & (pred == 0)))
        rows.append(
            {
                "threshold": round(float(t), 4),
                "sensitivity": round(tp / (tp + fn), 4) if tp + fn else float("nan"),
                "specificity": round(tn / (tn + fp), 4) if tn + fp else float("nan"),
                "flagged": tp + fp,
                "missed_cases": fn,
            }
        )
    return rows
