"""Clinical evaluation metrics for a binary risk model.

Why this module is not just ``sklearn.metrics.accuracy_score``
--------------------------------------------------------------
Accuracy is close to meaningless for this task. In a dataset that is 86%
positive, predicting "cancer" for everyone scores 86% accuracy while providing
zero clinical information. The metrics that matter for a screening instrument
are:

* **Sensitivity** (recall) -- the fraction of true cases detected. Missing a
  cancer is the costly error, so this is the primary metric.
* **Specificity** -- the fraction of healthy people correctly cleared, which
  determines the false-alarm burden.
* **PPV / NPV** -- what a positive or negative result actually means for a
  patient. Both depend on prevalence, so they are reported *adjusted* to a
  realistic prevalence rather than the dataset's own.
* **Likelihood ratios** -- prevalence-independent, so they transfer across
  populations.
* **Discrimination** (ROC-AUC, PR-AUC) -- ranking quality, threshold-free.
* **Calibration** (Brier, ECE) -- whether a stated "70% risk" corresponds to
  70% of such patients having the disease. An uncalibrated probability shown
  as a percentage is a misleading number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)

#: Lung cancer prevalence in a high-risk screening population (heavy smokers
#: aged 55-74), used as the default reference for prevalence adjustment.
#: Consistent with the ~1% baseline cancer detection rate reported by the
#: National Lung Screening Trial.
SCREENING_PREVALENCE: float = 0.01


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    """Raw confusion-matrix counts at a fixed decision threshold."""

    tn: int
    fp: int
    fn: int
    tp: int

    @property
    def n(self) -> int:
        """Total number of observations."""
        return self.tn + self.fp + self.fn + self.tp


@dataclass(frozen=True, slots=True)
class ClinicalMetrics:
    """Threshold-dependent and threshold-free performance of a risk model.

    ``ppv_adjusted``/``npv_adjusted`` are the predictive values transported to
    ``reference_prevalence``; ``ppv``/``npv`` are the raw values observed in the
    evaluation sample and are only valid for a population with the same
    prevalence.
    """

    threshold: float
    counts: ConfusionCounts
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    ppv_adjusted: float
    npv_adjusted: float
    reference_prevalence: float
    positive_likelihood_ratio: float
    negative_likelihood_ratio: float
    balanced_accuracy: float
    f1: float
    mcc: float
    accuracy: float
    prevalence: float
    roc_auc: float
    pr_auc: float
    brier: float
    expected_calibration_error: float

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view, rounding floats for readability."""
        out: dict[str, object] = {}
        for key, value in asdict(self).items():
            if key == "counts":
                out[key] = dict(value)  # type: ignore[arg-type]
            elif isinstance(value, float):
                out[key] = round(value, 6)
            else:
                out[key] = value
        return out


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide, returning NaN when the denominator is zero.

    A zero denominator means the metric is undefined for this sample (e.g. no
    negative cases means specificity cannot be estimated). NaN records that
    honestly instead of substituting a made-up 0.0.
    """
    return float("nan") if denominator == 0 else numerator / denominator


def confusion_counts(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> ConfusionCounts:
    """Compute confusion-matrix counts for binary 0/1 labels.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        The four confusion-matrix counts.
    """
    truth = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)
    return ConfusionCounts(
        tn=int(np.sum((truth == 0) & (pred == 0))),
        fp=int(np.sum((truth == 0) & (pred == 1))),
        fn=int(np.sum((truth == 1) & (pred == 0))),
        tp=int(np.sum((truth == 1) & (pred == 1))),
    )


def ppv_at_prevalence(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Transport PPV to a target prevalence via Bayes' theorem.

    A model's PPV measured on an 86%-positive research sample says nothing
    about its PPV in a screening clinic where 1% of attendees have cancer.
    Sensitivity and specificity are (approximately) properties of the test, so
    predictive values are recomputed from them at the prevalence of interest.

    Args:
        sensitivity: True positive rate.
        specificity: True negative rate.
        prevalence: Disease prevalence in the target population, in [0, 1].

    Returns:
        Positive predictive value at ``prevalence``, or NaN if undefined.

    Raises:
        ValueError: If ``prevalence`` is outside [0, 1].
    """
    if not 0.0 <= prevalence <= 1.0:
        raise ValueError(f"prevalence must be in [0, 1], got {prevalence}")
    tp = sensitivity * prevalence
    fp = (1.0 - specificity) * (1.0 - prevalence)
    return _safe_divide(tp, tp + fp)


def npv_at_prevalence(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Transport NPV to a target prevalence via Bayes' theorem.

    Args:
        sensitivity: True positive rate.
        specificity: True negative rate.
        prevalence: Disease prevalence in the target population, in [0, 1].

    Returns:
        Negative predictive value at ``prevalence``, or NaN if undefined.

    Raises:
        ValueError: If ``prevalence`` is outside [0, 1].
    """
    if not 0.0 <= prevalence <= 1.0:
        raise ValueError(f"prevalence must be in [0, 1], got {prevalence}")
    tn = specificity * (1.0 - prevalence)
    fn = (1.0 - sensitivity) * prevalence
    return _safe_divide(tn, tn + fn)


def expected_calibration_error(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Compute the expected calibration error (ECE).

    Predictions are grouped into equal-width probability bins; ECE is the
    sample-weighted mean absolute gap between predicted confidence and observed
    frequency within each bin. 0 is perfect calibration.

    Args:
        y_true: Ground-truth 0/1 labels.
        y_prob: Predicted probabilities of the positive class.
        n_bins: Number of equal-width bins.

    Returns:
        The ECE in [0, 1].

    Raises:
        ValueError: If ``n_bins`` < 1 or inputs are empty/mismatched.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    truth = np.asarray(y_true, dtype=float)
    prob = np.asarray(y_prob, dtype=float)
    if truth.shape != prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if truth.size == 0:
        raise ValueError("cannot compute calibration error on an empty sample")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize with right=True puts p in bin i when edges[i-1] < p <= edges[i].
    bin_index = np.clip(np.digitize(prob, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bin_index == b
        count = int(mask.sum())
        if count == 0:
            continue
        total += (count / truth.size) * abs(truth[mask].mean() - prob[mask].mean())
    return float(total)


def calibration_curve_points(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Return per-bin calibration curve data.

    Args:
        y_true: Ground-truth 0/1 labels.
        y_prob: Predicted probabilities.
        n_bins: Number of equal-width bins.

    Returns:
        One record per non-empty bin with mean predicted probability, observed
        event rate, and bin count.
    """
    truth = np.asarray(y_true, dtype=float)
    prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(prob, edges[1:-1], right=True), 0, n_bins - 1)
    points: list[dict[str, float]] = []
    for b in range(n_bins):
        mask = bin_index == b
        count = int(mask.sum())
        if count == 0:
            continue
        points.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "mean_predicted": float(prob[mask].mean()),
                "observed_rate": float(truth[mask].mean()),
                "count": count,
            }
        )
    return points


def clinical_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    reference_prevalence: float = SCREENING_PREVALENCE,
    n_calibration_bins: int = 10,
) -> ClinicalMetrics:
    """Compute the full clinical metric set at a given decision threshold.

    Args:
        y_true: Ground-truth 0/1 labels.
        y_prob: Predicted probabilities of the positive class.
        threshold: Probability at or above which a case is flagged positive.
        reference_prevalence: Prevalence used for PPV/NPV adjustment.
        n_calibration_bins: Bins for the calibration error.

    Returns:
        Populated :class:`ClinicalMetrics`.

    Raises:
        ValueError: If inputs are empty, mismatched, or the threshold is not in
            [0, 1].
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    truth = np.asarray(y_true).astype(int)
    prob = np.asarray(y_prob, dtype=float)
    if truth.shape != prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if truth.size == 0:
        raise ValueError("cannot evaluate an empty sample")

    predicted = (prob >= threshold).astype(int)
    counts = confusion_counts(truth, predicted)

    sensitivity = _safe_divide(counts.tp, counts.tp + counts.fn)
    specificity = _safe_divide(counts.tn, counts.tn + counts.fp)
    ppv = _safe_divide(counts.tp, counts.tp + counts.fp)
    npv = _safe_divide(counts.tn, counts.tn + counts.fn)
    precision = ppv

    # Discrimination metrics need both classes present.
    both_classes = 0 < truth.sum() < truth.size
    roc = float(roc_auc_score(truth, prob)) if both_classes else float("nan")
    pr = float(average_precision_score(truth, prob)) if both_classes else float("nan")
    mcc = float(matthews_corrcoef(truth, predicted)) if both_classes else float("nan")

    return ClinicalMetrics(
        threshold=float(threshold),
        counts=counts,
        sensitivity=sensitivity,
        specificity=specificity,
        ppv=ppv,
        npv=npv,
        ppv_adjusted=ppv_at_prevalence(sensitivity, specificity, reference_prevalence),
        npv_adjusted=npv_at_prevalence(sensitivity, specificity, reference_prevalence),
        reference_prevalence=float(reference_prevalence),
        positive_likelihood_ratio=_safe_divide(sensitivity, 1.0 - specificity),
        negative_likelihood_ratio=_safe_divide(1.0 - sensitivity, specificity),
        balanced_accuracy=(sensitivity + specificity) / 2.0,
        f1=_safe_divide(2 * precision * sensitivity, precision + sensitivity),
        mcc=mcc,
        accuracy=_safe_divide(counts.tp + counts.tn, counts.n),
        prevalence=float(truth.mean()),
        roc_auc=roc,
        pr_auc=pr,
        brier=float(brier_score_loss(truth, prob)),
        expected_calibration_error=expected_calibration_error(
            truth, prob, n_bins=n_calibration_bins
        ),
    )


def bootstrap_ci(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    metric: str,
    threshold: float,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Estimate a percentile bootstrap confidence interval for one metric.

    With a few hundred rows, point estimates are unstable; publishing an
    interval alongside them is the difference between a measurement and a
    guess. Resampling is stratified within class to keep both classes present.

    Args:
        y_true: Ground-truth 0/1 labels.
        y_prob: Predicted probabilities.
        metric: Attribute name on :class:`ClinicalMetrics` to bootstrap.
        threshold: Decision threshold used for threshold-dependent metrics.
        n_resamples: Number of bootstrap resamples.
        confidence: Central mass of the interval, e.g. 0.95.
        random_state: Seed for reproducibility.

    Returns:
        ``(lower, upper)`` percentile bounds, NaN if not estimable.

    Raises:
        ValueError: If ``metric`` is not a float field of ClinicalMetrics.
    """
    if metric not in ClinicalMetrics.__annotations__ or metric == "counts":
        raise ValueError(f"{metric!r} is not a bootstrappable ClinicalMetrics field")
    truth = np.asarray(y_true).astype(int)
    prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(random_state)

    pos_idx = np.flatnonzero(truth == 1)
    neg_idx = np.flatnonzero(truth == 0)
    if pos_idx.size == 0 or neg_idx.size == 0:
        return (float("nan"), float("nan"))

    samples: list[float] = []
    for _ in range(n_resamples):
        idx = np.concatenate(
            (
                rng.choice(pos_idx, size=pos_idx.size, replace=True),
                rng.choice(neg_idx, size=neg_idx.size, replace=True),
            )
        )
        value = getattr(
            clinical_metrics(truth[idx], prob[idx], threshold=threshold), metric
        )
        if not np.isnan(value):
            samples.append(float(value))

    if not samples:
        return (float("nan"), float("nan"))
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(samples, alpha)),
        float(np.quantile(samples, 1.0 - alpha)),
    )


def metrics_with_intervals(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    metrics: Iterable[str] = ("sensitivity", "specificity", "roc_auc", "pr_auc"),
    n_resamples: int = 2000,
    reference_prevalence: float = SCREENING_PREVALENCE,
) -> dict[str, object]:
    """Compute point estimates plus bootstrap intervals for selected metrics.

    Args:
        y_true: Ground-truth 0/1 labels.
        y_prob: Predicted probabilities.
        threshold: Decision threshold.
        metrics: Names of metrics to bootstrap.
        n_resamples: Bootstrap resamples per metric.
        reference_prevalence: Prevalence for PPV/NPV adjustment.

    Returns:
        Dict with all point estimates under ``"point"`` and intervals under
        ``"ci95"``.
    """
    point = clinical_metrics(
        y_true, y_prob, threshold=threshold, reference_prevalence=reference_prevalence
    )
    intervals = {
        name: bootstrap_ci(
            y_true, y_prob, metric=name, threshold=threshold, n_resamples=n_resamples
        )
        for name in metrics
    }
    return {
        "point": point.as_dict(),
        "ci95": {k: [round(v[0], 6), round(v[1], 6)] for k, v in intervals.items()},
    }
