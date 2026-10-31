"""Training pipeline: model selection, calibration, evaluation, packaging.

Run as a module::

    python -m lcrisk.train --data data/survey_lung_cancer.csv --out models/current

Every reported number comes from out-of-fold predictions produced by repeated
stratified cross-validation. Nothing is scored on data used to fit the model
that scored it, and the decision threshold is selected on the same out-of-fold
predictions rather than tuned against a final test set.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
)

from .data import load_survey_csv, reference_patient
from .evaluate import (
    SCREENING_PREVALENCE,
    calibration_curve_points,
    clinical_metrics,
    metrics_with_intervals,
)
from .explain import population_importance
from .model import (
    MODEL_ZOO,
    ModelBundle,
    build_calibrated_estimator,
    build_provenance,
)
from .threshold import select_threshold, threshold_sweep

logger = logging.getLogger(__name__)

DEFAULT_DATA = "data/survey_lung_cancer.csv"
DEFAULT_OUT = "models/current"


def out_of_fold_probabilities(
    estimator: Any,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Compute out-of-fold positive-class probabilities.

    Args:
        estimator: Unfitted estimator; cloned per fold by scikit-learn.
        features: Encoded features.
        target: Encoded 0/1 target.
        n_splits: Stratified folds.
        random_state: Seed for the fold split.

    Returns:
        Array of out-of-fold probabilities aligned with ``features``.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    probabilities = cross_val_predict(
        estimator, features, target, cv=cv, method="predict_proba", n_jobs=1
    )
    return np.asarray(probabilities)[:, 1]


def compare_models(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    calibration_methods: tuple[str, ...] = ("none", "sigmoid", "isotonic"),
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Score every model/calibration combination with repeated CV.

    Selection uses mean ROC-AUC for discrimination and mean Brier score for
    calibration quality, averaged over repeats so a single lucky fold split
    cannot decide the winner.

    Args:
        features: Encoded features.
        target: Encoded 0/1 target.
        calibration_methods: Calibration variants to try per model.
        n_splits: Folds per repeat.
        n_repeats: Number of independent fold splits.
        random_state: Base seed.

    Returns:
        Records sorted by mean ROC-AUC descending.
    """
    results: list[dict[str, Any]] = []
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    # Group the repeated splits so each repeat yields one full set of OOF preds.
    all_splits = list(splitter.split(features, target))
    per_repeat = [all_splits[i * n_splits : (i + 1) * n_splits] for i in range(n_repeats)]

    for model_name in MODEL_ZOO:
        for method in calibration_methods:
            aucs: list[float] = []
            briers: list[float] = []
            eces: list[float] = []
            for repeat_splits in per_repeat:
                oof = np.full(len(features), np.nan)
                for train_idx, test_idx in repeat_splits:
                    estimator = build_calibrated_estimator(
                        model_name, method=method, cv=3, random_state=random_state  # type: ignore[arg-type]
                    )
                    estimator.fit(features.iloc[train_idx], target.iloc[train_idx])
                    oof[test_idx] = estimator.predict_proba(features.iloc[test_idx])[:, 1]
                metrics = clinical_metrics(target.to_numpy(), oof, threshold=0.5)
                aucs.append(metrics.roc_auc)
                briers.append(metrics.brier)
                eces.append(metrics.expected_calibration_error)
            results.append(
                {
                    "model": model_name,
                    "calibration": method,
                    "roc_auc_mean": round(float(np.mean(aucs)), 6),
                    "roc_auc_std": round(float(np.std(aucs)), 6),
                    "brier_mean": round(float(np.mean(briers)), 6),
                    "ece_mean": round(float(np.mean(eces)), 6),
                    "note": MODEL_ZOO[model_name]["note"],
                }
            )
            logger.info(
                "%-20s %-9s AUC=%.4f+-%.4f Brier=%.4f ECE=%.4f",
                model_name,
                method,
                results[-1]["roc_auc_mean"],
                results[-1]["roc_auc_std"],
                results[-1]["brier_mean"],
                results[-1]["ece_mean"],
            )
    results.sort(key=lambda r: r["roc_auc_mean"], reverse=True)
    return results


def train(
    *,
    data_path: str | Path = DEFAULT_DATA,
    out_dir: str | Path = DEFAULT_OUT,
    threshold_strategy: str = "target_sensitivity",
    target_sensitivity: float = 0.95,
    reference_prevalence: float = SCREENING_PREVALENCE,
    n_splits: int = 5,
    n_repeats: int = 5,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> ModelBundle:
    """Run the full training pipeline and write a model bundle.

    Steps: load and validate data, compare models under repeated CV, refit the
    winner, derive out-of-fold predictions, select an operating threshold,
    compute clinical metrics with bootstrap intervals, compute permutation
    importance, then persist everything with provenance.

    Args:
        data_path: Source CSV.
        out_dir: Bundle output directory.
        threshold_strategy: Passed to
            :func:`~lcrisk.threshold.select_threshold`.
        target_sensitivity: Sensitivity floor for the screening threshold.
        reference_prevalence: Prevalence for PPV/NPV adjustment.
        n_splits: CV folds.
        n_repeats: CV repeats for model selection.
        n_bootstrap: Bootstrap resamples for confidence intervals.
        random_state: Global seed.

    Returns:
        The saved :class:`~lcrisk.model.ModelBundle`.
    """
    features, target, report = load_survey_csv(data_path)
    logger.info(
        "loaded %d rows (%d positive / %d negative, prevalence %.1f%%)",
        report.n_rows_used,
        report.n_positive,
        report.n_negative,
        report.prevalence * 100,
    )

    logger.info("comparing %d models x calibration variants", len(MODEL_ZOO))
    comparison = compare_models(
        features,
        target,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    winner = comparison[0]
    model_name, calibration = str(winner["model"]), str(winner["calibration"])
    logger.info("selected %s + %s calibration", model_name, calibration)

    # Out-of-fold predictions from the winning configuration, used for both
    # threshold selection and reported metrics.
    estimator_proto = build_calibrated_estimator(
        model_name, method=calibration, cv=3, random_state=random_state  # type: ignore[arg-type]
    )
    oof = out_of_fold_probabilities(
        estimator_proto,
        features,
        target,
        n_splits=n_splits,
        random_state=random_state,
    )

    operating_point = select_threshold(
        target.to_numpy(),
        oof,
        strategy=threshold_strategy,  # type: ignore[arg-type]
        target=target_sensitivity,
    )
    logger.info(
        "threshold %.4f -> sensitivity %.3f, specificity %.3f",
        operating_point.threshold,
        operating_point.achieved_sensitivity,
        operating_point.achieved_specificity,
    )

    cv_metrics = metrics_with_intervals(
        target.to_numpy(),
        oof,
        threshold=operating_point.threshold,
        n_resamples=n_bootstrap,
        reference_prevalence=reference_prevalence,
    )
    metrics_at_half = clinical_metrics(
        target.to_numpy(),
        oof,
        threshold=0.5,
        reference_prevalence=reference_prevalence,
    ).as_dict()

    # Fit the final model on all available data.
    final_estimator = build_calibrated_estimator(
        model_name, method=calibration, cv=n_splits, random_state=random_state  # type: ignore[arg-type]
    )
    final_estimator.fit(features, target)

    logger.info("computing permutation importance")
    importance = population_importance(
        final_estimator, features, target, n_repeats=30, random_state=random_state
    )

    metrics: dict[str, Any] = {
        "cross_validated": cv_metrics,
        "cross_validated_at_threshold_0.5": metrics_at_half,
        "calibration_curve": calibration_curve_points(target.to_numpy(), oof),
        "threshold_sweep": threshold_sweep(target.to_numpy(), oof),
        "evaluation_protocol": (
            f"{n_splits}-fold stratified cross-validation, out-of-fold "
            f"predictions; model selected over {n_repeats} repeated splits; "
            f"{n_bootstrap} stratified bootstrap resamples for intervals."
        ),
    }

    bundle = ModelBundle(
        estimator=final_estimator,
        feature_names=list(features.columns),
        threshold=operating_point.threshold,
        model_name=model_name,
        calibration_method=calibration,
        metrics=metrics,
        operating_point=operating_point.as_dict(),
        data_report=report.as_dict(),
        model_comparison=comparison,
        feature_importance=importance,
        reference_patient=reference_patient(features),
        provenance=build_provenance(
            data_path=data_path,
            features=features,
            target=target,
            random_state=random_state,
            extra={"selection_metric": "mean cross-validated ROC-AUC"},
        ),
    )
    saved = bundle.save(out_dir)
    logger.info("bundle written to %s", saved)
    return bundle


def _print_summary(bundle: ModelBundle) -> None:
    """Print a concise human-readable training summary."""
    point = bundle.metrics["cross_validated"]["point"]
    ci = bundle.metrics["cross_validated"]["ci95"]
    print("\n" + "=" * 68)
    print(f"  MODEL: {bundle.model_name}  |  calibration: {bundle.calibration_method}")
    print("=" * 68)
    print(f"  Operating threshold : {bundle.threshold:.4f}")
    print(f"  {bundle.operating_point['rationale']}")
    print("-" * 68)
    for name in ("sensitivity", "specificity", "roc_auc", "pr_auc"):
        low, high = ci[name]
        print(f"  {name:<22} {point[name]:.4f}   95% CI [{low:.4f}, {high:.4f}]")
    for name in ("brier", "expected_calibration_error", "balanced_accuracy"):
        print(f"  {name:<22} {point[name]:.4f}")
    print("-" * 68)
    prevalence = point["reference_prevalence"]
    print(f"  At a realistic screening prevalence of {prevalence:.1%}:")
    print(f"    PPV {point['ppv_adjusted']:.4f}   NPV {point['npv_adjusted']:.4f}")
    print(f"  (In-sample prevalence is {point['prevalence']:.1%}, so the raw")
    print(f"    PPV {point['ppv']:.4f} / NPV {point['npv']:.4f} do not transfer.)")
    print("-" * 68)
    print("  Top features by permutation importance:")
    for record in bundle.feature_importance[:5]:
        print(f"    {record['label']:<26} {record['importance_mean']:+.4f}")
    print("=" * 68 + "\n")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument list, defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(
        prog="python -m lcrisk.train",
        description="Train, calibrate, and evaluate the lung-cancer risk model.",
    )
    parser.add_argument("--data", default=DEFAULT_DATA, help="path to the survey CSV")
    parser.add_argument("--out", default=DEFAULT_OUT, help="bundle output directory")
    parser.add_argument(
        "--threshold-strategy",
        default="target_sensitivity",
        choices=("target_sensitivity", "target_specificity", "youden", "f1"),
    )
    parser.add_argument("--target-sensitivity", type=float, default=0.95)
    parser.add_argument(
        "--reference-prevalence",
        type=float,
        default=SCREENING_PREVALENCE,
        help="prevalence used to adjust PPV/NPV",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    bundle = train(
        data_path=args.data,
        out_dir=args.out,
        threshold_strategy=args.threshold_strategy,
        target_sensitivity=args.target_sensitivity,
        reference_prevalence=args.reference_prevalence,
        n_splits=args.folds,
        n_repeats=args.repeats,
        n_bootstrap=args.bootstrap,
        random_state=args.seed,
    )
    if not args.quiet:
        _print_summary(bundle)

    metrics_path = Path(args.out) / "metrics.json"
    metrics_path.write_text(
        json.dumps(bundle.metrics, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
