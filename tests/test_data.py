"""Tests for lcrisk.data: loading, validation, and quality reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lcrisk.data import DataValidationError, load_survey_csv, patient_frame, reference_patient
from lcrisk.schema import FEATURE_NAMES


class TestLoadSurveyCsv:
    def test_loads_expected_shape(self, learnable_csv: Path) -> None:
        features, target, _ = load_survey_csv(learnable_csv)
        assert list(features.columns) == list(FEATURE_NAMES)
        assert len(features) == len(target)
        assert set(target.unique()) <= {0, 1}

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_survey_csv(tmp_path / "does_not_exist.csv")

    def test_missing_column_raises(self, tmp_path: Path, learnable_csv: Path) -> None:
        frame = pd.read_csv(learnable_csv).drop(columns=["SMOKING"])
        path = tmp_path / "missing_col.csv"
        frame.to_csv(path, index=False)
        with pytest.raises(DataValidationError, match="SMOKING"):
            load_survey_csv(path)

    def test_out_of_domain_binary_value_raises(
        self, tmp_path: Path, learnable_csv: Path
    ) -> None:
        frame = pd.read_csv(learnable_csv)
        frame.loc[0, "SMOKING"] = 9  # invalid: schema only permits 1/2
        path = tmp_path / "bad_domain.csv"
        frame.to_csv(path, index=False)
        with pytest.raises(DataValidationError, match="SMOKING"):
            load_survey_csv(path)

    def test_out_of_range_age_raises(self, tmp_path: Path, learnable_csv: Path) -> None:
        frame = pd.read_csv(learnable_csv)
        frame.loc[0, "AGE"] = 200
        path = tmp_path / "bad_age.csv"
        frame.to_csv(path, index=False)
        with pytest.raises(DataValidationError, match="AGE"):
            load_survey_csv(path)

    def test_missing_values_raise(self, tmp_path: Path, learnable_csv: Path) -> None:
        frame = pd.read_csv(learnable_csv)
        frame.loc[0, "AGE"] = None
        path = tmp_path / "has_na.csv"
        frame.to_csv(path, index=False)
        with pytest.raises(DataValidationError, match="Missing values"):
            load_survey_csv(path)

    def test_messy_headers_are_normalised(self, messy_csv: Path) -> None:
        features, target, _ = load_survey_csv(messy_csv)
        assert list(features.columns) == list(FEATURE_NAMES)
        assert len(target) > 0

    def test_duplicates_are_removed_by_default(self, messy_csv: Path) -> None:
        features, _, report = load_survey_csv(messy_csv)
        raw_rows = report.n_rows_raw
        assert report.n_duplicates_removed == 10
        assert len(features) == raw_rows - 10

    def test_duplicates_kept_when_disabled(self, messy_csv: Path) -> None:
        features, _, report = load_survey_csv(messy_csv, drop_duplicates=False)
        assert report.n_duplicates_removed == 0
        assert len(features) == report.n_rows_raw

    def test_report_warns_on_small_sample(self, learnable_csv: Path) -> None:
        _, _, report = load_survey_csv(learnable_csv)
        assert any("rows available" in w for w in report.warnings)

    def test_report_records_prevalence(self, learnable_csv: Path) -> None:
        features, target, report = load_survey_csv(learnable_csv)
        assert report.prevalence == pytest.approx(target.mean())
        assert report.n_positive + report.n_negative == len(features)


class TestPatientFrame:
    def test_builds_single_row_in_canonical_order(self) -> None:
        encoded = {name: 1 for name in FEATURE_NAMES}
        encoded["AGE"] = 55
        frame = patient_frame(encoded)
        assert list(frame.columns) == list(FEATURE_NAMES)
        assert len(frame) == 1
        assert frame.loc[0, "AGE"] == 55

    def test_missing_feature_raises(self) -> None:
        encoded = {name: 1 for name in FEATURE_NAMES}
        del encoded["CHEST_PAIN"]
        with pytest.raises(DataValidationError, match="CHEST_PAIN"):
            patient_frame(encoded)


class TestReferencePatient:
    def test_returns_all_features(self, learnable_csv: Path) -> None:
        features, _, _ = load_survey_csv(learnable_csv)
        reference = reference_patient(features)
        assert set(reference) == set(FEATURE_NAMES)

    def test_age_is_within_observed_range(self, learnable_csv: Path) -> None:
        features, _, _ = load_survey_csv(learnable_csv)
        reference = reference_patient(features)
        assert features["AGE"].min() <= reference["AGE"] <= features["AGE"].max()
