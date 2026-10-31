# Legacy quarantine

These files are the **original** artifacts that this project shipped before the
rebuild. They are kept only for reference and for the audit trail. **Nothing in
the `lcrisk` package or the Streamlit app loads them**, and they must not be
wired back in. An integration test
(`tests/test_integration.py::test_lcrisk_does_not_import_the_legacy_pickle_paths`)
enforces this.

## Files

- `lung_cancer_random_forest_model.pkl` — an uncalibrated `RandomForestClassifier`.
- `data_scaler_transform.pkl` — the `StandardScaler` that was bundled with it.

## Why they were removed

When these artifacts were evaluated against the real survey data
(`data/survey_lung_cancer.csv`) during the rebuild, they failed catastrophically:

- **ROC-AUC ≈ 0.29 — worse than a coin flip (0.5).** The classifier ranked
  patients in roughly the *wrong* order.
- It **missed 237 of 270 real cancer cases** at the shipped threshold and
  assigned *lower* risk to smokers.
- The bundled scaler recorded `n_samples_seen_ = 1000`, but the real dataset has
  only 309 rows. Every symptom column had mean ≈ 1.50 and std ≈ 0.50 — the exact
  statistical signature of uniformly random 1/2 draws.

Taken together, the evidence indicates the original model was trained on
**randomly generated data**, not on the survey. Any number it produced was
meaningless.

## What replaced them

A reproducible, cross-validated, calibrated pipeline under `src/lcrisk/`, with
the selected model saved as a versioned bundle in `models/current/`. See the
top-level `README.md` for measured performance and the full limitations list.
