"""``lcrisk`` -- a lung-cancer risk model with an auditable evaluation trail.

Intended use
------------
Research and educational use only. This package implements a risk *stratifier*
trained on a small self-reported symptom survey. It is not a diagnostic device,
it has no regulatory clearance, and it must not be used to make care decisions.
See ``README.md`` for the measured performance and the full limitations list.

Public API
----------
:func:`~lcrisk.predict.load_assessor`
    Load a trained model bundle and serve explained risk assessments.
:func:`~lcrisk.train.train`
    Run the training, calibration, and evaluation pipeline.
:mod:`~lcrisk.schema`
    Feature definitions and the strict input encoders.
"""

from __future__ import annotations

from .data import DataQualityReport, DataValidationError, load_survey_csv
from .evaluate import (
    ClinicalMetrics,
    clinical_metrics,
    npv_at_prevalence,
    ppv_at_prevalence,
)
from .explain import explain_patient, population_importance
from .model import ModelBundle, build_calibrated_estimator, build_estimator
from .predict import RiskAssessment, RiskAssessor, blank_patient, load_assessor
from .schema import (
    FEATURE_NAMES,
    FEATURES,
    EncodingError,
    encode_patient,
    encode_yes_no,
)
from .threshold import OperatingPoint, select_threshold

__version__ = "1.0.0"

#: Shown wherever a risk number is displayed.
DISCLAIMER = (
    "Research and educational use only. This is not a medical device, is not "
    "a diagnostic test, and has no regulatory clearance. It cannot rule lung "
    "cancer in or out. Any health decision must be made by a qualified "
    "clinician."
)

__all__ = [
    "DISCLAIMER",
    "__version__",
    "ClinicalMetrics",
    "DataQualityReport",
    "DataValidationError",
    "EncodingError",
    "FEATURES",
    "FEATURE_NAMES",
    "ModelBundle",
    "OperatingPoint",
    "RiskAssessment",
    "RiskAssessor",
    "blank_patient",
    "build_calibrated_estimator",
    "build_estimator",
    "clinical_metrics",
    "encode_patient",
    "encode_yes_no",
    "explain_patient",
    "load_assessor",
    "load_survey_csv",
    "npv_at_prevalence",
    "population_importance",
    "ppv_at_prevalence",
    "select_threshold",
]
