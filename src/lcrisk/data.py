"""Dataset loading, schema validation, and data-quality reporting.

Loading is strict: unknown value domains, missing columns, and impossible ages
are errors, not warnings. Quality problems that are *properties of the data*
rather than bugs (small sample, class imbalance, duplicate rows) are reported
so they can be published alongside the model's metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (
    AGE_MAX,
    AGE_MIN,
    BINARY_FEATURE_NAMES,
    FEATURE_NAMES,
    NO,
    TARGET_COLUMN,
    YES,
    canonical_column,
    encode_sex,
    encode_target,
)

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when a dataset does not conform to the expected schema."""


@dataclass(slots=True)
class DataQualityReport:
    """Findings about a loaded dataset that must be disclosed with any metric.

    These are characteristics of the source data, not defects introduced by
    this codebase. They bound how much any model trained here can be trusted.
    """

    n_rows_raw: int
    n_rows_used: int
    n_duplicates_removed: int
    n_positive: int
    n_negative: int
    prevalence: float
    age_min: int
    age_max: int
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view of the report."""
        return {
            "n_rows_raw": self.n_rows_raw,
            "n_rows_used": self.n_rows_used,
            "n_duplicates_removed": self.n_duplicates_removed,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "prevalence": round(self.prevalence, 6),
            "age_min": self.age_min,
            "age_max": self.age_max,
            "warnings": list(self.warnings),
        }


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names, failing on collisions."""
    renamed = {col: canonical_column(col) for col in frame.columns}
    out = frame.rename(columns=renamed)
    duplicated = out.columns[out.columns.duplicated()].tolist()
    if duplicated:
        raise DataValidationError(
            f"Column names collide after normalisation: {duplicated}. "
            "The source file has two columns meaning the same thing."
        )
    return out


def _validate_binary_domains(frame: pd.DataFrame) -> None:
    """Ensure every binary feature contains only the survey's 1/2 encoding."""
    for name in BINARY_FEATURE_NAMES:
        values = set(pd.unique(frame[name].dropna()))
        unexpected = values - {NO, YES}
        if unexpected:
            raise DataValidationError(
                f"Column {name} contains {sorted(unexpected)}; this schema "
                f"requires the survey encoding {NO}=no / {YES}=yes. If your "
                "source uses 0/1, remap it before loading rather than letting "
                "the mismatch reach the model."
            )


def load_survey_csv(
    path: str | Path,
    *,
    drop_duplicates: bool = True,
) -> tuple[pd.DataFrame, pd.Series, DataQualityReport]:
    """Load and validate the lung-cancer survey CSV.

    Args:
        path: Path to the CSV file.
        drop_duplicates: Remove exact duplicate rows. Default ``True``. This
            matters for honest evaluation: duplicate rows split across
            cross-validation folds put identical records in both train and
            test, which inflates every score.

    Returns:
        Tuple of ``(features, target, report)`` where ``features`` has columns
        in :data:`~lcrisk.schema.FEATURE_NAMES` order, ``target`` is 0/1, and
        ``report`` documents data-quality findings.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        DataValidationError: If required columns are missing or values fall
            outside their permitted domain.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. See README for how to fetch it."
        )

    raw = pd.read_csv(csv_path)
    n_raw = len(raw)
    frame = _normalise_columns(raw)

    required = [*FEATURE_NAMES, TARGET_COLUMN]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise DataValidationError(
            f"Dataset is missing required columns: {', '.join(missing)}. "
            f"Found: {', '.join(map(str, frame.columns))}"
        )
    frame = frame[required].copy()

    if frame.isna().any().any():
        na_cols = frame.columns[frame.isna().any()].tolist()
        raise DataValidationError(
            f"Missing values present in {na_cols}. This pipeline has no "
            "imputation strategy; decide one explicitly before training."
        )

    # Encode sex and target before domain checks so both accept raw strings.
    frame["GENDER"] = frame["GENDER"].map(encode_sex)
    target_encoded = frame[TARGET_COLUMN].map(encode_target)

    warnings: list[str] = []
    n_duplicates = 0
    if drop_duplicates:
        combined = frame.assign(**{TARGET_COLUMN: target_encoded})
        mask = ~combined.duplicated()
        n_duplicates = int((~mask).sum())
        if n_duplicates:
            frame = frame.loc[mask].reset_index(drop=True)
            target_encoded = target_encoded.loc[mask].reset_index(drop=True)
            warnings.append(
                f"Removed {n_duplicates} exact duplicate rows. Keeping them "
                "would leak identical records across CV folds."
            )

    _validate_binary_domains(frame)

    ages = frame["AGE"]
    if not np.issubdtype(ages.dtype, np.number):
        raise DataValidationError("AGE must be numeric.")
    out_of_range = ages[(ages < AGE_MIN) | (ages > AGE_MAX)]
    if not out_of_range.empty:
        raise DataValidationError(
            f"{len(out_of_range)} AGE values outside [{AGE_MIN}, {AGE_MAX}]: "
            f"{sorted(out_of_range.unique())[:5]}"
        )
    frame["AGE"] = ages.astype(int)

    features = frame[list(FEATURE_NAMES)].astype(int)
    target = target_encoded.astype(int).rename(TARGET_COLUMN)

    n_pos = int(target.sum())
    n_neg = int(len(target) - n_pos)
    prevalence = float(target.mean())

    if len(features) < 500:
        warnings.append(
            f"Only {len(features)} rows available. Every metric below carries "
            "a wide confidence interval; treat point estimates as indicative."
        )
    if n_neg < 50:
        warnings.append(
            f"Only {n_neg} negative cases. Specificity is estimated from a very "
            "small denominator and is the least reliable metric reported."
        )
    if not 0.2 <= prevalence <= 0.8:
        warnings.append(
            f"Dataset prevalence is {prevalence:.1%}, far from any realistic "
            "screening population (~1-2%). PPV/NPV computed directly on this "
            "data do not transfer; use prevalence-adjusted values."
        )

    report = DataQualityReport(
        n_rows_raw=n_raw,
        n_rows_used=len(features),
        n_duplicates_removed=n_duplicates,
        n_positive=n_pos,
        n_negative=n_neg,
        prevalence=prevalence,
        age_min=int(features["AGE"].min()),
        age_max=int(features["AGE"].max()),
        warnings=warnings,
    )
    for message in warnings:
        logger.warning("data quality: %s", message)

    return features, target, report


def patient_frame(encoded: dict[str, int]) -> pd.DataFrame:
    """Build a single-row model input frame from encoded patient features.

    Args:
        encoded: Output of :func:`~lcrisk.schema.encode_patient`.

    Returns:
        One-row DataFrame with columns in canonical model order.

    Raises:
        DataValidationError: If any canonical feature is absent.
    """
    missing = [name for name in FEATURE_NAMES if name not in encoded]
    if missing:
        raise DataValidationError(f"Missing encoded features: {missing}")
    return pd.DataFrame([[encoded[n] for n in FEATURE_NAMES]], columns=list(FEATURE_NAMES))


def reference_patient(features: pd.DataFrame) -> dict[str, int]:
    """Compute a population reference patient used as an explanation baseline.

    Binary features take their modal value and age takes the median, giving a
    "typical respondent" against which an individual's answers are contrasted.

    Args:
        features: Encoded training features.

    Returns:
        Encoded feature mapping for the reference patient.
    """
    reference: dict[str, int] = {}
    for name in FEATURE_NAMES:
        column = features[name]
        reference[name] = int(column.median() if name == "AGE" else column.mode().iat[0])
    return reference
