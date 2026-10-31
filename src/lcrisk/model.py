"""Model construction, probability calibration, and artefact persistence.

Two structural decisions here fix defects in the original project:

1. **The scaler lives inside the pipeline.** Shipping a separately-fitted
   ``StandardScaler`` pickle invites two failures: the scaler gets fitted on the
   full dataset before splitting (leaking test-set statistics into training),
   and the scaler and model can drift out of sync. As a single
   :class:`~sklearn.pipeline.Pipeline`, the scaler is refitted inside every
   cross-validation fold and the pair cannot be mismatched.

2. **Probabilities are calibrated.** A random forest's ``predict_proba`` is a
   vote share, not a probability -- it is typically over-confident near 0.5 and
   under-confident at the extremes. Displaying an uncalibrated vote share as
   "68.4% risk" is a false precision claim. Every shipped model is wrapped in
   :class:`~sklearn.calibration.CalibratedClassifierCV` and its calibration is
   measured and published.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .schema import FEATURE_NAMES

#: Filename of the estimator inside a bundle directory.
ESTIMATOR_FILE = "estimator.joblib"
#: Filename of the human-readable metadata inside a bundle directory.
METADATA_FILE = "metadata.json"

CalibrationMethod = Literal["sigmoid", "isotonic", "none"]
ModelName = Literal["logistic_regression", "random_forest", "gradient_boosting"]

#: Model zoo. Small tabular datasets rarely justify a complex model, so a
#: regularised linear baseline is included and allowed to win -- it usually
#: calibrates better and yields interpretable coefficients.
MODEL_ZOO: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "factory": lambda seed: LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        ),
        "note": "L2-regularised linear baseline; coefficients are log-odds ratios.",
    },
    "random_forest": {
        "factory": lambda seed: RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "note": (
            "Depth and leaf size constrained deliberately: with a few hundred "
            "rows an unconstrained forest memorises the sample."
        ),
    },
    "gradient_boosting": {
        "factory": lambda seed: GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.8,
            random_state=seed,
        ),
        "note": "Shallow boosted stumps; captures mild feature interactions.",
    },
}


def build_estimator(name: ModelName | str, *, random_state: int = 42) -> Pipeline:
    """Build an unfitted scaling + classification pipeline.

    Args:
        name: Key into :data:`MODEL_ZOO`.
        random_state: Seed passed to the classifier.

    Returns:
        A :class:`~sklearn.pipeline.Pipeline` of ``StandardScaler`` then the
        chosen classifier.

    Raises:
        KeyError: If ``name`` is not in the model zoo.
    """
    if name not in MODEL_ZOO:
        raise KeyError(
            f"unknown model {name!r}; choose from {sorted(MODEL_ZOO)}"
        )
    classifier = MODEL_ZOO[name]["factory"](random_state)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def build_calibrated_estimator(
    name: ModelName | str,
    *,
    method: CalibrationMethod = "sigmoid",
    cv: int = 5,
    random_state: int = 42,
) -> BaseEstimator:
    """Build an unfitted, calibration-wrapped pipeline.

    Args:
        name: Key into :data:`MODEL_ZOO`.
        method: ``"sigmoid"`` (Platt scaling, robust on small samples),
            ``"isotonic"`` (flexible, needs more data), or ``"none"`` to skip
            calibration.
        cv: Internal cross-validation folds used to fit the calibrator.
        random_state: Seed passed to the classifier.

    Returns:
        The estimator, calibration-wrapped unless ``method="none"``.

    Raises:
        ValueError: If ``method`` is not a recognised calibration method.
    """
    base = build_estimator(name, random_state=random_state)
    if method == "none":
        return base
    if method not in ("sigmoid", "isotonic"):
        raise ValueError(f"unknown calibration method {method!r}")
    return CalibratedClassifierCV(base, method=method, cv=cv)


@dataclass(slots=True)
class ModelBundle:
    """A fitted estimator packaged with everything needed to use it honestly.

    A bare pickle is not a deployable model: it carries no record of what data
    it saw, how well it scored, what threshold it should run at, or which
    library versions can load it. Those live here and are surfaced in the UI.
    """

    estimator: BaseEstimator
    feature_names: list[str]
    threshold: float
    model_name: str
    calibration_method: str
    metrics: dict[str, Any] = field(default_factory=dict)
    operating_point: dict[str, Any] = field(default_factory=dict)
    data_report: dict[str, Any] = field(default_factory=dict)
    model_comparison: list[dict[str, Any]] = field(default_factory=list)
    feature_importance: list[dict[str, Any]] = field(default_factory=list)
    reference_patient: dict[str, int] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        """Return the JSON-serialisable metadata (everything but the estimator)."""
        return {
            "model_name": self.model_name,
            "calibration_method": self.calibration_method,
            "feature_names": list(self.feature_names),
            "threshold": self.threshold,
            "operating_point": self.operating_point,
            "metrics": self.metrics,
            "model_comparison": self.model_comparison,
            "feature_importance": self.feature_importance,
            "reference_patient": self.reference_patient,
            "data_report": self.data_report,
            "provenance": self.provenance,
        }

    @property
    def is_calibrated(self) -> bool:
        """True when the stored probabilities passed through a calibrator."""
        return self.calibration_method in ("sigmoid", "isotonic")

    def save(self, directory: str | Path) -> Path:
        """Persist the bundle to a directory.

        Args:
            directory: Destination directory, created if absent.

        Returns:
            The directory the bundle was written to.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, out / ESTIMATOR_FILE)
        (out / METADATA_FILE).write_text(
            json.dumps(self.metadata(), indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        return out

    @classmethod
    def load(cls, directory: str | Path) -> "ModelBundle":
        """Load a bundle previously written by :meth:`save`.

        Args:
            directory: Bundle directory.

        Returns:
            The reconstructed bundle.

        Raises:
            FileNotFoundError: If the estimator or metadata file is absent.
            ValueError: If the stored feature list does not match the current
                schema, which would mean silently feeding the model columns it
                was not trained on.
        """
        src = Path(directory)
        estimator_path = src / ESTIMATOR_FILE
        metadata_path = src / METADATA_FILE
        if not estimator_path.is_file():
            raise FileNotFoundError(f"No estimator at {estimator_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"No metadata at {metadata_path}")

        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        stored_features = list(meta.get("feature_names", []))
        if stored_features != list(FEATURE_NAMES):
            raise ValueError(
                "Bundle feature list does not match the current schema.\n"
                f"  bundle: {stored_features}\n  schema: {list(FEATURE_NAMES)}\n"
                "Retrain the model after a schema change."
            )
        return cls(
            estimator=joblib.load(estimator_path),
            feature_names=stored_features,
            threshold=float(meta["threshold"]),
            model_name=meta.get("model_name", "unknown"),
            calibration_method=meta.get("calibration_method", "none"),
            metrics=meta.get("metrics", {}),
            operating_point=meta.get("operating_point", {}),
            data_report=meta.get("data_report", {}),
            model_comparison=meta.get("model_comparison", []),
            feature_importance=meta.get("feature_importance", []),
            reference_patient=meta.get("reference_patient", {}),
            provenance=meta.get("provenance", {}),
        )

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict positive-class probability for each row.

        Args:
            features: DataFrame containing at least the bundle's features.

        Returns:
            1-D array of positive-class probabilities.

        Raises:
            ValueError: If required feature columns are missing.
        """
        missing = [c for c in self.feature_names if c not in features.columns]
        if missing:
            raise ValueError(f"Input is missing feature columns: {missing}")
        ordered = features[self.feature_names]
        return self.estimator.predict_proba(ordered)[:, 1]  # type: ignore[attr-defined]


def dataset_fingerprint(features: pd.DataFrame, target: pd.Series) -> str:
    """Compute a stable content hash of the training data.

    Recorded in provenance so a model can be tied to the exact data that
    produced it, and so an accidental data change is detectable.

    Args:
        features: Encoded feature frame.
        target: Encoded target series.

    Returns:
        Hex SHA-256 digest, truncated to 16 characters.
    """
    digest = hashlib.sha256()
    digest.update(
        pd.concat([features.reset_index(drop=True), target.reset_index(drop=True)], axis=1)
        .to_csv(index=False)
        .encode("utf-8")
    )
    return digest.hexdigest()[:16]


def build_provenance(
    *,
    data_path: str | Path,
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the provenance record stored with a model.

    Args:
        data_path: Source dataset path.
        features: Encoded training features.
        target: Encoded training target.
        random_state: Global seed used for training.
        extra: Additional keys to merge in.

    Returns:
        Provenance dict recording data identity, environment, and seed.
    """
    record: dict[str, Any] = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_path": str(data_path),
        "data_fingerprint_sha256_16": dataset_fingerprint(features, target),
        "n_training_rows": int(len(features)),
        "random_state": random_state,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "joblib_version": joblib.__version__,
    }
    if extra:
        record.update(extra)
    return record
