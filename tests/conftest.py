"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lcrisk.schema import FEATURE_NAMES, MALE, NO, TARGET_COLUMN, YES

REAL_DATA = Path(__file__).resolve().parents[1] / "data" / "survey_lung_cancer.csv"
BUNDLE_DIR = Path(__file__).resolve().parents[1] / "models" / "current"


@pytest.fixture(scope="session")
def real_data_path() -> Path:
    """Path to the real survey CSV, skipping the test if it is absent."""
    if not REAL_DATA.is_file():
        pytest.skip(f"dataset not present at {REAL_DATA}")
    return REAL_DATA


@pytest.fixture(scope="session")
def bundle_dir() -> Path:
    """Path to the trained bundle, skipping the test if it is absent."""
    if not (BUNDLE_DIR / "metadata.json").is_file():
        pytest.skip(f"no trained bundle at {BUNDLE_DIR}; run python -m lcrisk.train")
    return BUNDLE_DIR


def make_learnable_frame(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Build a synthetic dataset with genuine signal, in raw CSV form.

    Used so tests can exercise the training pipeline quickly without depending
    on the real dataset. Signal is deliberately strong and simple so that a
    correct pipeline is expected to score well above chance.

    Args:
        n: Number of rows.
        seed: RNG seed.

    Returns:
        DataFrame with raw-style columns, including the string target.
    """
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for name in FEATURE_NAMES:
        if name == "AGE":
            data[name] = rng.integers(35, 85, size=n)
        elif name == "GENDER":
            data[name] = rng.choice(["M", "F"], size=n)
        else:
            data[name] = rng.choice([NO, YES], size=n)

    frame = pd.DataFrame(data)
    # Three drivers plus age, then a logistic link -> real, learnable signal.
    logit = (
        1.6 * (frame["SMOKING"] == YES)
        + 1.3 * (frame["COUGHING"] == YES)
        + 1.1 * (frame["SHORTNESS_OF_BREATH"] == YES)
        + 0.05 * (frame["AGE"] - 60)
        - 2.2
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    frame[TARGET_COLUMN] = np.where(
        rng.random(n) < probability, "YES", "NO"
    )
    return frame


@pytest.fixture
def learnable_csv(tmp_path: Path) -> Path:
    """Write a synthetic learnable dataset to a temporary CSV."""
    path = tmp_path / "synthetic.csv"
    make_learnable_frame().to_csv(path, index=False)
    return path


@pytest.fixture
def messy_csv(tmp_path: Path) -> Path:
    """Write a dataset with the real file's messy headers and duplicate rows."""
    frame = make_learnable_frame(n=120, seed=11)
    frame = pd.concat([frame, frame.head(10)], ignore_index=True)  # 10 duplicates
    renames = {
        "CHRONIC_DISEASE": "CHRONIC DISEASE",
        "FATIGUE": "FATIGUE ",
        "ALLERGY": "ALLERGY ",
        "ALCOHOL_CONSUMING": "ALCOHOL CONSUMING",
        "SHORTNESS_OF_BREATH": "SHORTNESS OF BREATH",
        "SWALLOWING_DIFFICULTY": "SWALLOWING DIFFICULTY",
        "CHEST_PAIN": "CHEST PAIN",
    }
    path = tmp_path / "messy.csv"
    frame.rename(columns=renames).to_csv(path, index=False)
    return path


@pytest.fixture
def healthy_patient() -> dict[str, object]:
    """A patient with no reported symptoms."""
    answers: dict[str, object] = {
        name: False for name in FEATURE_NAMES if name not in ("AGE", "GENDER")
    }
    answers["AGE"] = 45
    answers["GENDER"] = "F"
    return answers


@pytest.fixture
def symptomatic_patient() -> dict[str, object]:
    """A patient reporting every symptom."""
    answers: dict[str, object] = {
        name: True for name in FEATURE_NAMES if name not in ("AGE", "GENDER")
    }
    answers["AGE"] = 70
    answers["GENDER"] = MALE
    return answers
