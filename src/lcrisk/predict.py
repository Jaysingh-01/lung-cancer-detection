"""Inference service: raw user answers in, explained risk assessment out.

This is the only module the user interface talks to. Encoding, ordering,
thresholding, banding, and explanation all happen here so that no caller can
reintroduce the class of bug that broke the original app -- a UI that quietly
built its own feature vector.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .data import patient_frame
from .evaluate import npv_at_prevalence, ppv_at_prevalence
from .explain import FeatureContribution, explain_patient, top_contributions
from .model import ModelBundle
from .schema import FEATURE_NAMES, encode_patient

#: Risk bands are defined *relative to the model's own operating threshold*
#: rather than at a round number like 50%. ``BORDERLINE_FRACTION`` sets where
#: the low/borderline boundary sits as a fraction of that threshold.
BORDERLINE_FRACTION: float = 0.5

BAND_LOW = "low"
BAND_BORDERLINE = "borderline"
BAND_ELEVATED = "elevated"
BAND_HIGH = "high"

#: Ordered bands with the guidance shown to a user. Wording deliberately avoids
#: diagnostic language: this instrument cannot diagnose or rule out cancer.
BAND_GUIDANCE: dict[str, str] = {
    BAND_LOW: (
        "Below the model's flagging threshold. This is not a negative cancer "
        "result and does not rule anything out."
    ),
    BAND_BORDERLINE: (
        "Approaching but below the flagging threshold. Clinical judgement "
        "should take precedence over this number."
    ),
    BAND_ELEVATED: (
        "At or above the flagging threshold. Indicates the answer pattern "
        "resembles cases in the training data; warrants clinical discussion."
    ),
    BAND_HIGH: (
        "Well above the flagging threshold. Warrants clinical discussion and "
        "appropriate diagnostic work-up decided by a clinician."
    ),
}


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    """The complete result of assessing one patient."""

    probability: float
    threshold: float
    band: str
    guidance: str
    flagged: bool
    is_calibrated: bool
    calibration_error: float | None
    encoded_features: dict[str, int]
    contributions: list[FeatureContribution] = field(default_factory=list)
    model_name: str = "unknown"

    @property
    def percent(self) -> float:
        """The risk estimate as a percentage."""
        return self.probability * 100.0

    def uncertainty_note(self) -> str:
        """Describe the reliability of the stated probability.

        Returns:
            A sentence quantifying the expected gap between the stated
            probability and observed frequency, based on the model's measured
            calibration error.
        """
        if not self.is_calibrated:
            return (
                "This model's outputs are NOT calibrated: the number is a "
                "classifier score, not a probability. Do not read it as a "
                "percentage chance."
            )
        if self.calibration_error is None:
            return "Calibration was not measured for this model."
        return (
            f"Measured calibration error is {self.calibration_error * 100:.1f} "
            "percentage points, so treat this estimate as approximate rather "
            "than exact."
        )


def _band_for(probability: float, threshold: float) -> str:
    """Assign a risk band relative to the operating threshold."""
    if probability >= threshold + (1.0 - threshold) / 2.0:
        return BAND_HIGH
    if probability >= threshold:
        return BAND_ELEVATED
    if probability >= threshold * BORDERLINE_FRACTION:
        return BAND_BORDERLINE
    return BAND_LOW


class RiskAssessor:
    """Loads a model bundle and turns raw answers into explained assessments."""

    def __init__(self, bundle: ModelBundle) -> None:
        """Initialise from an already-loaded bundle.

        Args:
            bundle: The fitted model bundle to serve.
        """
        self._bundle = bundle

    @classmethod
    def from_directory(cls, directory: str | Path) -> "RiskAssessor":
        """Load a bundle from disk and wrap it.

        Args:
            directory: Bundle directory written by :meth:`ModelBundle.save`.

        Returns:
            A ready assessor.

        Raises:
            FileNotFoundError: If the bundle files are absent.
            ValueError: If the bundle schema does not match the code.
        """
        return cls(ModelBundle.load(directory))

    @property
    def bundle(self) -> ModelBundle:
        """The underlying model bundle, for metadata display."""
        return self._bundle

    @property
    def threshold(self) -> float:
        """The model's operating threshold."""
        return self._bundle.threshold

    def _calibration_error(self) -> float | None:
        """Pull the measured calibration error out of the bundle metrics."""
        point = self._bundle.metrics.get("cross_validated", {}).get("point", {})
        value = point.get("expected_calibration_error")
        return float(value) if isinstance(value, (int, float)) else None

    def assess(
        self,
        raw: Mapping[str, Any],
        *,
        explain: bool = True,
        max_reasons: int = 5,
    ) -> RiskAssessment:
        """Assess one patient from raw, unencoded answers.

        Args:
            raw: Mapping of canonical feature name to a raw answer -- booleans,
                ``"yes"``/``"no"``, ``"M"``/``"F"``, or an age number.
            explain: Whether to compute per-answer contributions.
            max_reasons: Maximum contributions to retain.

        Returns:
            The completed :class:`RiskAssessment`.

        Raises:
            EncodingError: If any answer cannot be encoded unambiguously.
        """
        encoded = encode_patient(raw)
        frame = patient_frame(encoded)
        probability = float(self._bundle.predict_proba(frame)[0])

        contributions: list[FeatureContribution] = []
        if explain:
            contributions = top_contributions(
                explain_patient(
                    self._bundle.predict_proba,
                    encoded,
                    reference=self._bundle.reference_patient or None,
                ),
                limit=max_reasons,
            )

        band = _band_for(probability, self._bundle.threshold)
        return RiskAssessment(
            probability=probability,
            threshold=self._bundle.threshold,
            band=band,
            guidance=BAND_GUIDANCE[band],
            flagged=probability >= self._bundle.threshold,
            is_calibrated=self._bundle.is_calibrated,
            calibration_error=self._calibration_error(),
            encoded_features=encoded,
            contributions=contributions,
            model_name=self._bundle.model_name,
        )

    def predictive_values_at(self, prevalence: float) -> dict[str, float] | None:
        """Recompute PPV/NPV at a caller-supplied prevalence.

        Uses the bundle's cross-validated sensitivity and specificity, letting a
        user see what a flag would mean in *their* population rather than in the
        research sample.

        Args:
            prevalence: Assumed prevalence in [0, 1].

        Returns:
            ``{"ppv": ..., "npv": ..., "prevalence": ...}``, or ``None`` when the
            bundle has no cross-validated metrics to transport.

        Raises:
            ValueError: If ``prevalence`` is outside [0, 1].
        """
        point = self._bundle.metrics.get("cross_validated", {}).get("point", {})
        sensitivity = point.get("sensitivity")
        specificity = point.get("specificity")
        if not isinstance(sensitivity, (int, float)) or not isinstance(
            specificity, (int, float)
        ):
            return None
        return {
            "prevalence": prevalence,
            "ppv": ppv_at_prevalence(float(sensitivity), float(specificity), prevalence),
            "npv": npv_at_prevalence(float(sensitivity), float(specificity), prevalence),
        }


@functools.lru_cache(maxsize=4)
def load_assessor(directory: str | Path) -> RiskAssessor:
    """Load and cache an assessor by bundle directory.

    Deserialising the estimator is expensive relative to a prediction, so it is
    cached. The original app reloaded a 4.7 MB pickle on every widget change.

    Args:
        directory: Bundle directory.

    Returns:
        A cached assessor instance.
    """
    return RiskAssessor.from_directory(directory)


def blank_patient() -> dict[str, Any]:
    """Build a default answer set: female, age 60, every symptom absent.

    Returns:
        Raw (unencoded) answers for every feature in the schema.
    """
    answers: dict[str, Any] = {}
    for name in FEATURE_NAMES:
        if name == "GENDER":
            answers[name] = "F"
        elif name == "AGE":
            answers[name] = 60
        else:
            answers[name] = False
    return answers
