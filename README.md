# lcrisk — Lung Cancer Risk Stratification

A calibrated risk-stratification model built on a small, self-reported lung
cancer symptom survey, with an auditable evaluation trail and an interface that
refuses to overstate what the model can do.

> **Research and educational use only.** This is **not** a medical device, not a
> diagnostic test, and has no regulatory clearance. It cannot rule lung cancer
> in or out. Any health decision must be made by a qualified clinician. See
> [Limitations](#limitations).

---

## Why this project exists

The version originally shipped here loaded a random-forest pickle that, when
evaluated against the real survey, scored **ROC-AUC ≈ 0.29 — worse than a coin
flip** and missed 237 of 270 cancer cases. Its bundled scaler had been fit on
1000 samples of what were, statistically, uniformly random draws: it had been
trained on synthetic noise, not on the survey.

This rebuild replaces that with a reproducible pipeline: strict input encoding,
honest cross-validated evaluation with confidence intervals, probability
calibration, an explicitly chosen operating threshold, and a UI that shows the
model card and prevalence-adjusted predictive values next to every score. The
old artifacts are retained, disabled, under [`legacy_quarantine/`](legacy_quarantine/).

## What it does

- Encodes 15 survey inputs (sex, age, and 13 yes/no symptom/behaviour items)
  through a single strict schema (`lcrisk.schema`) shared by the model and the UI.
- Serves a **calibrated probability**, a risk band, and a per-patient
  explanation of which answers drove the score.
- Recomputes **PPV/NPV at a user-chosen prevalence**, because predictive values
  measured on this 86%-positive sample do not transfer to a real ~1% screening
  population.

## Measured performance

Selected model: **L2-regularised logistic regression with sigmoid (Platt)
calibration**, chosen on mean cross-validated ROC-AUC over the model families in
`models/current/metadata.json`.

Protocol: 5-fold stratified cross-validation on out-of-fold predictions; 2000
stratified bootstrap resamples for 95% confidence intervals.

| Metric | Estimate | 95% CI |
|---|---|---|
| ROC-AUC | 0.944 | [0.913, 0.971] |
| PR-AUC | 0.991 | [0.986, 0.995] |
| Sensitivity (at operating threshold) | 95.4% | [92.4%, 97.9%] |
| Specificity (at operating threshold) | 76.3% | [63.2%, 89.5%] |
| Brier score | 0.066 | — |
| Expected calibration error | 6.3% | — |

**Operating threshold: 0.671** (strategy: `target_sensitivity`). It is the
lowest-alarm threshold that still detects ≥95% of cases, chosen because a missed
cancer costs far more than a follow-up test. The full sensitivity/specificity
trade-off curve is in `models/current/metrics.json` (`threshold_sweep`) and is
shown in the app so a different cost balance can be selected.

Accuracy is deliberately **not** a headline metric: the training sample is 86.2%
positive, so a "predict cancer for everyone" baseline would score ~86% accuracy
while carrying no information.

## Data

`data/survey_lung_cancer.csv` — 309 raw rows, 276 used after removing 33 exact
duplicates (which would otherwise leak across CV folds). Of those, **238
positive / 38 negative** (86.2% prevalence). Ages 21–87.

## Installation

Requires Python ≥ 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[app,dev]"   # runtime + Streamlit app + test tools
```

Pinned runtime dependencies are declared in `pyproject.toml` (scikit-learn
1.7.2, pandas 2.3.3, numpy 2.5.2, joblib 1.5.3).

## Usage

### Run the app

```bash
streamlit run app.py
```

The app loads the trained bundle from `models/current/`. Inputs are held in
memory for the session only — nothing is stored, logged, or transmitted.

### Retrain / reproduce the bundle

```bash
python -m lcrisk.train --data data/survey_lung_cancer.csv --out models/current
```

Training is seeded (`random_state=42`) and records provenance — data
fingerprint, library versions, and timestamp — into `models/current/metadata.json`.

### Use the library

```python
from lcrisk import load_assessor, blank_patient

assessor = load_assessor("models/current")
patient = blank_patient()  # raw answers: sex "F"/"M", age int, symptoms bool
patient.update({"AGE": 62, "SMOKING": True, "COUGHING": True, "FATIGUE": True})

assessment = assessor.assess(patient)
print(assessment.percent, assessment.band)  # e.g. 47.4 borderline
```

## Deployment

The app is a Streamlit server: it needs a long-running Python process and a
WebSocket connection. Static hosts such as Netlify, GitHub Pages and Vercel
cannot run it.

All options deploy from the GitHub repo, so `src/`, `models/current/`, `app.py`,
`requirements.txt` and `.streamlit/` must be committed and pushed.
`requirements.txt` pins scikit-learn 1.7.2 because the bundled estimator only
loads under that version.

**Streamlit Community Cloud (free, no Docker).** At
[share.streamlit.io](https://share.streamlit.io), create an app from this repo,
main file `app.py`. Under *Advanced settings* choose Python 3.14 (the version
the bundle was trained with).

**Any Docker host** (Render, Hugging Face Spaces, Railway, Fly.io, Cloud Run,
Azure Web App for Containers). The image listens on `$PORT` (default 8501) and
exposes a health check at `/_stcore/health`.

```bash
docker build -t lcrisk .
docker run --rm -p 8501:8501 lcrisk   # http://localhost:8501
```

- *Render:* New → Web Service → this repo; Render detects the `Dockerfile`.
- *Hugging Face Spaces:* create a Space with the Docker SDK and add
  `app_port: 8501` to the Space README's front matter.

Streamlit usage telemetry is switched off in `.streamlit/config.toml`, so the
deployed app sends nothing to Streamlit.

## Project layout

```
app.py                     Streamlit UI (schema-driven; cannot disagree with the model)
src/lcrisk/
  schema.py                Feature definitions + strict input encoders
  data.py                  CSV loading, dedup, and data-quality report
  model.py                 Estimator construction, calibration, ModelBundle
  threshold.py             Operating-point selection strategies
  evaluate.py              Clinical metrics, bootstrap CIs, prevalence adjustment
  explain.py               Per-patient and population-level explanations
  predict.py               RiskAssessor: load a bundle, serve assessments
  train.py                 End-to-end training/calibration/evaluation pipeline
models/current/            Trained bundle: estimator.joblib, metadata.json, metrics.json
data/                      survey_lung_cancer.csv
tests/                     pytest suite (131 tests)
legacy_quarantine/         The original broken artifacts, disabled — see its README
```

## Testing

```bash
pytest            # 131 tests
pytest --cov      # with coverage (lcrisk package)
```

`InconsistentVersionWarning` from scikit-learn is promoted to an error in
`pyproject.toml`, so a bundle trained under a mismatched library version fails
loudly rather than silently.

## Limitations

Read these before interpreting any score:

1. **It is a symptom questionnaire, not a diagnostic study.** Several of the
   strongest predictors (swallowing difficulty, chest pain, persistent cough)
   are symptoms of *advanced* disease. The model detects people who are already
   symptomatic; it does not find early, curable disease — the actual point of
   screening.
2. **Tiny, severely imbalanced sample.** 276 rows, only 38 negatives.
   Specificity rests on that small denominator, hence its wide CI.
3. **86.2% prevalence matches no real population.** A screening cohort is ~1%.
   Predictive values measured here do not transfer; the app recomputes PPV/NPV
   at a prevalence you set.
4. **No external validation.** Every number is cross-validation on a single
   dataset. Real-world performance is unknown and usually substantially worse.
5. **Association, not causation.** Feature importance reflects correlations in
   one survey, not causal effects.
6. **No fairness audit.** The data carries only sex and age — no ethnicity,
   socioeconomic, or comorbidity fields — so subgroup performance is unknown.
7. **Not a regulated device.** No clinical validation, no regulatory clearance,
   no quality management system.

For real screening guidance, consult your national programme (e.g. the USPSTF
recommendation of annual low-dose CT for adults 50–80 with a 20 pack-year
smoking history).

## License

MIT (see `pyproject.toml`).
