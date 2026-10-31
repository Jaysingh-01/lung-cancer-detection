"""Streamlit interface for the lung-cancer risk model.

Run with::

    streamlit run app.py

The interface is deliberately built so that it cannot disagree with the model.
Every input widget is generated from :data:`lcrisk.schema.FEATURES`, and the
feature vector is assembled by :func:`lcrisk.schema.encode_patient` rather than
by this file. The original version of this app hand-built its own feature dict
and silently inverted two of them; removing that duplication removes the whole
class of bug.

It also refuses to present the model as better than it is: the model card, the
measured operating point, and the prevalence-adjusted predictive values are all
shown next to the risk number, not hidden behind a link.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Support running from a source checkout without installing the package.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lcrisk import DISCLAIMER, __version__  # noqa: E402
from lcrisk.explain import describe_contribution  # noqa: E402
from lcrisk.predict import (  # noqa: E402
    BAND_ELEVATED,
    BAND_HIGH,
    BAND_LOW,
    RiskAssessor,
)
from lcrisk.schema import FEATURES, EncodingError  # noqa: E402

BUNDLE_DIR = Path(__file__).resolve().parent / "models" / "current"

st.set_page_config(
    page_title="Lung Cancer Risk Stratification (Research Tool)",
    page_icon="🫁",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading model bundle...")
def get_assessor(bundle_dir: str) -> RiskAssessor:
    """Load and cache the risk assessor.

    Cached because deserialising the estimator is far more expensive than a
    prediction, and Streamlit re-executes this script on every widget change.

    Args:
        bundle_dir: Path to the model bundle directory.

    Returns:
        The loaded assessor.
    """
    return RiskAssessor.from_directory(bundle_dir)


def render_model_card(assessor: RiskAssessor) -> None:
    """Render the model card: what this model is and how well it measured.

    Args:
        assessor: The loaded assessor whose bundle metadata is displayed.
    """
    bundle = assessor.bundle
    point = bundle.metrics.get("cross_validated", {}).get("point", {})
    ci = bundle.metrics.get("cross_validated", {}).get("ci95", {})
    data_report = bundle.data_report
    provenance = bundle.provenance

    st.subheader("Model card")

    if not bundle.is_calibrated:
        st.error(
            "This model's probabilities are **not calibrated**. Numbers shown "
            "are classifier scores and must not be read as percentages."
        )

    cols = st.columns(4)
    cols[0].metric(
        "Sensitivity",
        f"{point.get('sensitivity', float('nan')):.1%}",
        help="Share of true cases detected at the operating threshold. "
        "The primary metric for a screening instrument.",
    )
    cols[1].metric(
        "Specificity",
        f"{point.get('specificity', float('nan')):.1%}",
        help="Share of non-cases correctly cleared. Estimated from only "
        f"{data_report.get('n_negative', '?')} negative cases here.",
    )
    cols[2].metric(
        "ROC-AUC",
        f"{point.get('roc_auc', float('nan')):.3f}",
        help="Ranking quality, independent of threshold. 0.5 is chance.",
    )
    cols[3].metric(
        "Calibration error",
        f"{point.get('expected_calibration_error', float('nan')):.1%}",
        help="Mean gap between stated probability and observed frequency. "
        "Lower is better; 0 is perfect.",
    )

    with st.expander("Full performance, with confidence intervals"):
        rows = []
        for name in ("sensitivity", "specificity", "roc_auc", "pr_auc"):
            interval = ci.get(name, [float("nan"), float("nan")])
            rows.append(
                {
                    "Metric": name.replace("_", " ").title(),
                    "Estimate": f"{point.get(name, float('nan')):.4f}",
                    "95% CI": f"[{interval[0]:.4f}, {interval[1]:.4f}]",
                }
            )
        for name in ("brier", "expected_calibration_error", "balanced_accuracy", "mcc"):
            rows.append(
                {
                    "Metric": name.replace("_", " ").title(),
                    "Estimate": f"{point.get(name, float('nan')):.4f}",
                    "95% CI": "not bootstrapped",
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(bundle.metrics.get("evaluation_protocol", ""))

        st.markdown("**Why accuracy is not shown as a headline metric**")
        st.caption(
            f"The training data is {data_report.get('prevalence', 0):.1%} positive. "
            "Predicting 'cancer' for every patient would score that same "
            "accuracy while carrying no information, so accuracy is omitted in "
            "favour of sensitivity, specificity, and calibration."
        )

    with st.expander("Operating point and threshold"):
        operating = bundle.operating_point
        st.write(
            f"Decision threshold **{bundle.threshold:.4f}** "
            f"(strategy: `{operating.get('strategy', 'unknown')}`)"
        )
        st.caption(operating.get("rationale", ""))
        sweep = bundle.metrics.get("threshold_sweep", [])
        if sweep:
            st.caption(
                "The full trade-off curve, so a different cost balance can be "
                "chosen rather than inheriting this one:"
            )
            st.dataframe(pd.DataFrame(sweep), hide_index=True, height=240)

    with st.expander("Training data and provenance"):
        st.write(
            f"**{data_report.get('n_rows_used', '?')} rows used** "
            f"({data_report.get('n_positive', '?')} positive / "
            f"{data_report.get('n_negative', '?')} negative), from "
            f"{data_report.get('n_rows_raw', '?')} raw rows with "
            f"{data_report.get('n_duplicates_removed', 0)} exact duplicates removed."
        )
        for warning in data_report.get("warnings", []):
            st.warning(warning, icon="⚠️")
        st.json(
            {
                "model": bundle.model_name,
                "calibration": bundle.calibration_method,
                "selected_by": provenance.get("selection_metric"),
                "trained_at_utc": provenance.get("trained_at_utc"),
                "data_fingerprint": provenance.get("data_fingerprint_sha256_16"),
                "random_state": provenance.get("random_state"),
                "scikit_learn": provenance.get("scikit_learn_version"),
                "python": provenance.get("python_version"),
            },
            expanded=False,
        )

    with st.expander("Model selection: what else was tried"):
        comparison = bundle.model_comparison
        if comparison:
            st.dataframe(
                pd.DataFrame(comparison)[
                    ["model", "calibration", "roc_auc_mean", "roc_auc_std",
                     "brier_mean", "ece_mean"]
                ],
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Selected on mean cross-validated ROC-AUC. Note how much "
                "calibration improves the Brier score and calibration error "
                "for every model family."
            )

    with st.expander("Which inputs the model relies on (population level)"):
        importance = bundle.feature_importance
        if importance:
            frame = pd.DataFrame(importance)[["label", "importance_mean", "importance_std"]]
            frame.columns = ["Feature", "Importance (AUC drop)", "Std"]
            st.dataframe(frame, hide_index=True, width="stretch")
            st.caption(
                "Permutation importance: the drop in ROC-AUC when a column is "
                "shuffled. Used instead of a tree's built-in importance, which "
                "is biased toward many-valued features like age."
            )


def render_inputs() -> dict[str, object]:
    """Render the patient input form, generated from the feature schema.

    Returns:
        Mapping of canonical feature name to the raw widget value.
    """
    answers: dict[str, object] = {}
    groups: dict[str, list] = {}
    for spec in FEATURES:
        groups.setdefault(spec.group, []).append(spec)

    for group_name, specs in groups.items():
        st.markdown(f"**{group_name}**")
        columns = st.columns(min(len(specs), 3))
        for index, spec in enumerate(specs):
            column = columns[index % len(columns)]
            if spec.kind == "sex":
                answers[spec.name] = column.selectbox(
                    spec.label,
                    options=["F", "M"],
                    format_func=lambda v: "Female" if v == "F" else "Male",
                    help=spec.help or None,
                    key=f"in_{spec.name}",
                )
            elif spec.kind == "numeric":
                answers[spec.name] = column.number_input(
                    spec.label,
                    min_value=18,
                    max_value=100,
                    value=60,
                    step=1,
                    help=spec.help or None,
                    key=f"in_{spec.name}",
                )
            else:
                answers[spec.name] = column.toggle(
                    spec.label,
                    value=False,
                    help=spec.help or None,
                    key=f"in_{spec.name}",
                )
    return answers


def render_result(assessor: RiskAssessor, answers: dict[str, object]) -> None:
    """Compute and render the risk assessment for the given answers.

    Args:
        assessor: The loaded assessor.
        answers: Raw answers from :func:`render_inputs`.
    """
    try:
        assessment = assessor.assess(answers)
    except EncodingError as exc:
        st.error(f"Could not encode these inputs: {exc}")
        return

    st.subheader("Result")

    banner = {
        BAND_LOW: st.info,
        BAND_ELEVATED: st.warning,
        BAND_HIGH: st.error,
    }.get(assessment.band, st.warning)

    banner(
        f"**Model risk score: {assessment.percent:.1f}%** "
        f"({assessment.band.upper()}) — threshold for flagging is "
        f"{assessment.threshold:.1%}"
    )
    st.caption(assessment.guidance)
    st.caption(assessment.uncertainty_note())

    st.progress(min(max(assessment.probability, 0.0), 1.0))

    st.markdown("---")
    st.markdown("#### What does this score actually mean for a patient?")
    st.caption(
        "A risk score is not a probability of disease until it is combined with "
        "how common the disease is in the population being tested. Set the "
        "prevalence for your population to see what a flag would really imply."
    )
    prevalence_percent = st.slider(
        "Assumed lung cancer prevalence in the tested population (%)",
        min_value=0.1,
        max_value=90.0,
        value=1.0,
        step=0.1,
        help="~1% is typical of a high-risk screening cohort. The training "
        "sample was 86% positive, which is not a screening population.",
    )
    predictive = assessor.predictive_values_at(prevalence_percent / 100.0)
    if predictive:
        left, right = st.columns(2)
        left.metric(
            "PPV — a flagged patient has cancer with probability",
            f"{predictive['ppv']:.1%}",
            help="Positive predictive value at the chosen prevalence.",
        )
        right.metric(
            "NPV — a non-flagged patient is cancer-free with probability",
            f"{predictive['npv']:.1%}",
            help="Negative predictive value at the chosen prevalence.",
        )
        if predictive["ppv"] < 0.2:
            st.warning(
                f"At {prevalence_percent:.1f}% prevalence, most flagged patients "
                f"would **not** have lung cancer (PPV {predictive['ppv']:.1%}). "
                "This is a mathematical consequence of screening for a rare "
                "disease and is why this tool cannot be used to screen.",
                icon="⚠️",
            )

    if assessment.contributions:
        st.markdown("---")
        st.markdown("#### Which answers drove this score")
        st.caption(
            "Each row re-runs the model with that single answer reversed, so the "
            "effect is specific to this patient rather than a population average."
        )
        for contribution in assessment.contributions:
            st.write(f"- {describe_contribution(contribution)}")

    with st.expander("Exact feature vector sent to the model"):
        st.caption(
            "Binary answers use the source survey's encoding (1 = no, 2 = yes); "
            "sex is 0 = female, 1 = male."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {"Feature": k, "Encoded value": v}
                    for k, v in assessment.encoded_features.items()
                ]
            ),
            hide_index=True,
            height=300,
        )


def main() -> None:
    """Render the whole application."""
    st.title("🫁 Lung Cancer Risk Stratification")
    st.caption(f"lcrisk v{__version__} — research instrument, not a medical device")

    st.error(DISCLAIMER, icon="🚫")

    if not BUNDLE_DIR.is_dir():
        st.warning(
            f"No trained model found at `{BUNDLE_DIR}`.\n\n"
            "Train one first:\n\n"
            "```bash\npython -m lcrisk.train --data data/survey_lung_cancer.csv "
            "--out models/current\n```"
        )
        st.stop()

    try:
        assessor = get_assessor(str(BUNDLE_DIR))
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Could not load the model bundle: {exc}")
        st.stop()

    assess_tab, card_tab, limits_tab = st.tabs(
        ["Assess a patient", "Model card", "Limitations"]
    )

    with assess_tab:
        st.caption(
            "Answers are held in memory for this session only. Nothing is "
            "stored, logged, or transmitted."
        )
        with st.form("patient_form"):
            answers = render_inputs()
            submitted = st.form_submit_button(
                "Calculate risk score", type="primary", width="stretch"
            )
        if submitted:
            render_result(assessor, answers)
        else:
            st.info("Enter the patient's answers above, then calculate.")

    with card_tab:
        render_model_card(assessor)

    with limits_tab:
        render_limitations(assessor)


def render_limitations(assessor: RiskAssessor) -> None:
    """Render the limitations tab.

    Args:
        assessor: The loaded assessor, used for concrete data figures.
    """
    report = assessor.bundle.data_report
    st.subheader("Limitations you must read before interpreting any score")

    st.markdown(
        f"""
**1. The training data is a symptom questionnaire, not a diagnostic study.**
Inputs are self-reported answers. Several of the strongest predictors
(swallowing difficulty, chest pain, persistent cough) are symptoms of
*advanced* disease. A model built on them detects people who are already
symptomatic; it does not find early, curable disease, which is the entire
purpose of screening.

**2. The sample is tiny and severely imbalanced.**
{report.get('n_rows_used', '?')} rows, of which only
{report.get('n_negative', '?')} are negative. Specificity rests on that small
denominator, which is why its confidence interval is wide.

**3. Prevalence is {report.get('prevalence', 0):.1%}, which no real population matches.**
A screening cohort runs about 1%. Predictive values measured on this sample do
not transfer, which is why the app recomputes PPV/NPV at a prevalence you set.

**4. No external validation.**
Every number comes from cross-validation on a single dataset. Performance on a
different population, country, or care setting is unknown and is normally
substantially worse.

**5. Association, not causation.**
The model learned correlations in one survey. A feature's importance here is not
evidence that it causes or prevents cancer.

**6. No demographic fairness audit.**
The dataset carries only sex and age, with no ethnicity, socioeconomic, or
comorbidity data, so subgroup performance cannot be assessed. Do not assume it
performs equally across groups.

**7. Not a regulated device.**
No clinical validation, no regulatory clearance, no quality management system.
Using it to guide real care would be both unsafe and, in most jurisdictions,
unlawful.
"""
    )

    st.info(
        "For actual lung cancer screening guidance, refer to your national "
        "programme — for example the USPSTF recommendation of annual low-dose "
        "CT for adults 50-80 with a 20 pack-year smoking history.",
        icon="ℹ️",
    )

    with st.expander("Why the previously shipped model was removed"):
        st.markdown(
            """
The model this project originally shipped
(`lung_cancer_random_forest_model.pkl`) was evaluated against the real survey
data during this rebuild and scored **ROC-AUC 0.286 — worse than a coin flip**.
It missed 237 of 270 real cancer cases, and predicted *lower* risk for smokers.

Its bundled scaler recorded `n_samples_seen_ = 1000` while the real dataset has
309 rows, and every symptom's mean was 1.50 with standard deviation 0.50 — the
exact signature of uniformly random 1/2 draws. The evidence indicates it was
trained on randomly generated data. Those files are retained under
`legacy_quarantine/` for reference and are no longer loadable by this app.
"""
        )


if __name__ == "__main__":
    main()
