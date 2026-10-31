"""End-to-end pipeline tests and a regression guard against the legacy model."""

from __future__ import annotations

from pathlib import Path

import pytest

from lcrisk.model import ModelBundle
from lcrisk.predict import RiskAssessor
from lcrisk.train import train


class TestTrainPredictEndToEnd:
    """Train on a small synthetic dataset and confirm the whole path works."""

    def test_train_writes_a_loadable_bundle(
        self, tmp_path: Path, learnable_csv: Path
    ) -> None:
        out_dir = tmp_path / "bundle"
        train(
            data_path=learnable_csv,
            out_dir=out_dir,
            n_splits=3,
            n_repeats=2,
            n_bootstrap=50,
        )
        bundle = ModelBundle.load(out_dir)
        assert bundle.model_name in ("logistic_regression", "random_forest", "gradient_boosting")
        assert 0.0 <= bundle.threshold <= 1.0

    def test_trained_model_beats_chance_on_learnable_data(
        self, tmp_path: Path, learnable_csv: Path
    ) -> None:
        out_dir = tmp_path / "bundle"
        bundle = train(
            data_path=learnable_csv,
            out_dir=out_dir,
            n_splits=3,
            n_repeats=2,
            n_bootstrap=50,
        )
        auc = bundle.metrics["cross_validated"]["point"]["roc_auc"]
        assert auc > 0.65, (
            f"ROC-AUC {auc:.3f} on data with known injected signal; the "
            "pipeline should clearly beat chance (0.5)."
        )

    def test_full_pipeline_from_csv_to_patient_assessment(
        self, tmp_path: Path, learnable_csv: Path
    ) -> None:
        out_dir = tmp_path / "bundle"
        train(data_path=learnable_csv, out_dir=out_dir, n_splits=3, n_repeats=2, n_bootstrap=50)
        assessor = RiskAssessor.from_directory(out_dir)

        healthy = {name: False for name in assessor.bundle.feature_names if name not in ("AGE", "GENDER")}
        healthy["AGE"] = 40
        healthy["GENDER"] = "F"
        sick = {name: True for name in assessor.bundle.feature_names if name not in ("AGE", "GENDER")}
        sick["AGE"] = 75
        sick["GENDER"] = "M"

        healthy_result = assessor.assess(healthy)
        sick_result = assessor.assess(sick)
        assert 0.0 <= healthy_result.probability <= 1.0
        assert 0.0 <= sick_result.probability <= 1.0
        # SMOKING/COUGHING/SHORTNESS_OF_BREATH are the injected drivers.
        assert sick_result.probability > healthy_result.probability

    def test_metadata_json_written_alongside_bundle(
        self, tmp_path: Path, learnable_csv: Path
    ) -> None:
        out_dir = tmp_path / "bundle"
        train(data_path=learnable_csv, out_dir=out_dir, n_splits=3, n_repeats=2, n_bootstrap=50)
        assert (out_dir / "metadata.json").is_file()
        assert (out_dir / "estimator.joblib").is_file()


class TestLegacyModelIsRetired:
    """The originally shipped artefacts must not be reachable from the package.

    They were trained on data whose statistics do not match the real survey
    (mean/std profile of uniformly random draws, and a recorded training
    sample size of 1000 versus the real dataset's few hundred rows), and they
    score worse than chance when evaluated on real data. This test guards
    against accidentally wiring them back in.
    """

    def test_lcrisk_does_not_import_the_legacy_pickle_paths(self) -> None:
        import lcrisk
        import lcrisk.model
        import lcrisk.predict
        import lcrisk.train

        for module in (lcrisk, lcrisk.model, lcrisk.predict, lcrisk.train):
            source = Path(module.__file__).read_text(encoding="utf-8")  # type: ignore[arg-type]
            assert "lung_cancer_random_forest_model.pkl" not in source
            assert "data_scaler_transform.pkl" not in source

    def test_current_bundle_is_not_the_legacy_random_forest(self) -> None:
        bundle_dir = Path(__file__).resolve().parents[1] / "models" / "current"
        if not (bundle_dir / "metadata.json").is_file():
            pytest.skip("no trained bundle present")
        bundle = ModelBundle.load(bundle_dir)
        # The legacy model was an uncalibrated RandomForestClassifier trained
        # on synthetic noise; the retrained bundle must be calibrated.
        assert bundle.is_calibrated
