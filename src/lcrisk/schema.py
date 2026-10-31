"""Single source of truth for the feature schema and its encoding.

Every other module derives column names, ordering, and value encodings from
here. Nothing else in the codebase is allowed to hard-code a feature list --
that duplication is what let the original app drift away from its own model.

Encoding notes
--------------
The source survey encodes every binary symptom as ``1 = no`` / ``2 = yes``
(not 0/1), and sex as ``F``/``M``. Those conventions are preserved so the
numbers fed to the model match the numbers the model was fitted on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, Mapping

# --- Encoding constants -----------------------------------------------------

#: Survey encoding for a negative/absent binary response.
NO: Final[int] = 1
#: Survey encoding for a positive/present binary response.
YES: Final[int] = 2

#: Sex encoding, matching a LabelEncoder fitted on the sorted values ``F``,``M``.
FEMALE: Final[int] = 0
MALE: Final[int] = 1

#: Raw target values in the source CSV.
TARGET_COLUMN: Final[str] = "LUNG_CANCER"
TARGET_POSITIVE: Final[str] = "YES"
TARGET_NEGATIVE: Final[str] = "NO"

#: Plausible age bounds. Values outside this range are rejected as data errors
#: rather than silently clipped.
AGE_MIN: Final[int] = 18
AGE_MAX: Final[int] = 100


FeatureKind = Literal["binary", "numeric", "sex"]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Describes one model input feature.

    Attributes:
        name: Canonical column name used by the model and all dataframes.
        label: Human-readable label for the user interface.
        kind: Value domain -- ``binary`` (1/2), ``numeric``, or ``sex`` (0/1).
        group: UI grouping, also used to organise explanation output.
        help: Short clarifying text shown to the user.
    """

    name: str
    label: str
    kind: FeatureKind
    group: str
    help: str = ""

    @property
    def is_modifiable_binary(self) -> bool:
        """True when the feature can be flipped for counterfactual explanation."""
        return self.kind == "binary"


# --- Feature definitions ----------------------------------------------------
# Order is significant: it defines the column order passed to the model.

FEATURES: Final[tuple[FeatureSpec, ...]] = (
    FeatureSpec("GENDER", "Sex", "sex", "Demographics",
                "Biological sex as recorded in the source survey."),
    FeatureSpec("AGE", "Age (years)", "numeric", "Demographics"),
    FeatureSpec("SMOKING", "Smoking history", "binary", "Lifestyle",
                "Any history of tobacco smoking."),
    FeatureSpec("ALCOHOL_CONSUMING", "Alcohol consumption", "binary", "Lifestyle"),
    FeatureSpec("PEER_PRESSURE", "Peer pressure", "binary", "Lifestyle",
                "Social pressure toward smoking or drinking."),
    FeatureSpec("YELLOW_FINGERS", "Yellow fingers", "binary", "Signs",
                "Tar staining, a proxy for heavy smoking."),
    FeatureSpec("ANXIETY", "Anxiety", "binary", "Signs"),
    FeatureSpec("CHRONIC_DISEASE", "Chronic disease history", "binary", "History"),
    FeatureSpec("ALLERGY", "Allergic history", "binary", "History"),
    FeatureSpec("FATIGUE", "Persistent fatigue", "binary", "Symptoms"),
    FeatureSpec("WHEEZING", "Wheezing", "binary", "Symptoms"),
    FeatureSpec("COUGHING", "Persistent cough", "binary", "Symptoms"),
    FeatureSpec("SHORTNESS_OF_BREATH", "Shortness of breath", "binary", "Symptoms"),
    FeatureSpec("SWALLOWING_DIFFICULTY", "Swallowing difficulty", "binary", "Symptoms"),
    FeatureSpec("CHEST_PAIN", "Chest pain", "binary", "Symptoms"),
)

#: Canonical model input order.
FEATURE_NAMES: Final[tuple[str, ...]] = tuple(f.name for f in FEATURES)

#: Lookup by canonical name.
FEATURES_BY_NAME: Final[Mapping[str, FeatureSpec]] = {f.name: f for f in FEATURES}

#: Binary symptom/lifestyle features (everything except sex and age).
BINARY_FEATURE_NAMES: Final[tuple[str, ...]] = tuple(
    f.name for f in FEATURES if f.kind == "binary"
)

#: Aliases mapping messy source-CSV headers onto canonical names. The public
#: survey CSV ships headers with trailing spaces and spaces instead of
#: underscores (e.g. ``"FATIGUE "``, ``"CHRONIC DISEASE"``).
_EXTRA_ALIASES: Final[Mapping[str, str]] = {
    "CHRONICDISEASE": "CHRONIC_DISEASE",
    "ALCOHOLCONSUMING": "ALCOHOL_CONSUMING",
    "SHORTNESSOFBREATH": "SHORTNESS_OF_BREATH",
    "SWALLOWINGDIFFICULTY": "SWALLOWING_DIFFICULTY",
    "CHESTPAIN": "CHEST_PAIN",
    "YELLOWFINGERS": "YELLOW_FINGERS",
    "PEERPRESSURE": "PEER_PRESSURE",
    "SEX": "GENDER",
    "LUNGCANCER": "LUNG_CANCER",
}


def canonical_column(raw: str) -> str:
    """Normalise one raw CSV header to its canonical name.

    Handles surrounding whitespace, internal spaces, hyphens, and case.

    Args:
        raw: Header string exactly as it appears in the source file.

    Returns:
        The canonical column name, or the normalised name unchanged when the
        column is not part of the known schema.

    Examples:
        >>> canonical_column("FATIGUE ")
        'FATIGUE'
        >>> canonical_column("CHRONIC DISEASE")
        'CHRONIC_DISEASE'
    """
    squashed = "".join(ch for ch in raw.upper() if ch.isalnum())
    if squashed in _EXTRA_ALIASES:
        return _EXTRA_ALIASES[squashed]
    for known in (*FEATURE_NAMES, TARGET_COLUMN):
        if "".join(ch for ch in known if ch.isalnum()) == squashed:
            return known
    return raw.strip().upper().replace(" ", "_").replace("-", "_")


class EncodingError(ValueError):
    """Raised when an input value cannot be encoded unambiguously."""


def encode_yes_no(value: bool | str | int) -> int:
    """Encode a yes/no answer as the survey's ``1 = no`` / ``2 = yes``.

    This function is deliberately strict. It accepts booleans, the exact
    strings ``"yes"``/``"no"`` (any case, surrounding whitespace ignored), and
    the already-encoded integers ``1``/``2``. Anything else raises.

    The strictness is the point. The original implementation was::

        return 2 if val or val == "Yes" else 1

    Because ``val`` was evaluated for truthiness first and the non-empty string
    ``"No"`` is truthy, ``"No"`` encoded to ``2`` -- silently inverting every
    dropdown answer. Guessing at ambiguous input is how that class of bug
    survives; refusing to guess is how it gets caught.

    Args:
        value: The answer to encode.

    Returns:
        :data:`YES` (2) for an affirmative answer, :data:`NO` (1) otherwise.

    Raises:
        EncodingError: If ``value`` is not an unambiguous yes/no answer.

    Examples:
        >>> encode_yes_no(True), encode_yes_no(False)
        (2, 1)
        >>> encode_yes_no("No"), encode_yes_no("yes")
        (1, 2)
    """
    # bool must be checked before int: bool is a subclass of int.
    if isinstance(value, bool):
        return YES if value else NO
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in ("yes", "y", "true"):
            return YES
        if text in ("no", "n", "false"):
            return NO
        raise EncodingError(
            f"Ambiguous yes/no value {value!r}. Expected 'yes' or 'no'."
        )
    if isinstance(value, int):
        if value in (NO, YES):
            return value
        raise EncodingError(
            f"Integer yes/no value must be {NO} (no) or {YES} (yes), got {value!r}."
        )
    raise EncodingError(
        f"Cannot encode {value!r} of type {type(value).__name__} as yes/no."
    )


def encode_sex(value: str | int) -> int:
    """Encode sex as ``0 = female`` / ``1 = male``.

    Args:
        value: ``"M"``/``"Male"``/``"F"``/``"Female"`` (any case), or 0/1.

    Returns:
        :data:`MALE` (1) or :data:`FEMALE` (0).

    Raises:
        EncodingError: If ``value`` is not a recognised sex value.
    """
    if isinstance(value, bool):
        raise EncodingError("Refusing to encode a bool as sex; pass 'M'/'F'.")
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in ("m", "male"):
            return MALE
        if text in ("f", "female"):
            return FEMALE
        raise EncodingError(f"Unrecognised sex value {value!r}. Expected 'M' or 'F'.")
    if isinstance(value, int):
        if value in (FEMALE, MALE):
            return value
        raise EncodingError(f"Integer sex must be 0 (female) or 1 (male), got {value!r}.")
    raise EncodingError(f"Cannot encode {value!r} as sex.")


def encode_age(value: float | int) -> int:
    """Validate and encode an age in years.

    Args:
        value: Age in years.

    Returns:
        The age as an ``int``.

    Raises:
        EncodingError: If the age is not a finite number within
            [:data:`AGE_MIN`, :data:`AGE_MAX`].
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EncodingError(f"Age must be a number, got {value!r}.")
    if value != value:  # NaN
        raise EncodingError("Age must not be NaN.")
    if not AGE_MIN <= value <= AGE_MAX:
        raise EncodingError(
            f"Age {value} is outside the supported range "
            f"[{AGE_MIN}, {AGE_MAX}]. This model was not validated there."
        )
    return int(value)


def encode_target(value: str | int | bool) -> int:
    """Encode the outcome label as ``1 = lung cancer`` / ``0 = no lung cancer``.

    Args:
        value: ``"YES"``/``"NO"`` (any case) or 0/1.

    Returns:
        1 for the positive class, 0 for the negative class.

    Raises:
        EncodingError: If the label is unrecognised.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in ("yes", "y", "1", "true"):
            return 1
        if text in ("no", "n", "0", "false"):
            return 0
        raise EncodingError(f"Unrecognised target label {value!r}.")
    if isinstance(value, int) and value in (0, 1):
        return value
    raise EncodingError(f"Cannot encode {value!r} as a target label.")


def encode_patient(raw: Mapping[str, bool | str | int | float]) -> dict[str, int]:
    """Encode one patient's raw answers into model-ready features.

    Args:
        raw: Mapping of canonical feature name to raw answer. Must contain
            every name in :data:`FEATURE_NAMES`.

    Returns:
        Dict keyed by canonical feature name in :data:`FEATURE_NAMES` order,
        with values in the model's expected encoding.

    Raises:
        EncodingError: If a feature is missing or a value cannot be encoded.
    """
    missing = [name for name in FEATURE_NAMES if name not in raw]
    if missing:
        raise EncodingError(f"Missing required features: {', '.join(missing)}")

    encoded: dict[str, int] = {}
    for spec in FEATURES:
        value = raw[spec.name]
        try:
            if spec.kind == "sex":
                encoded[spec.name] = encode_sex(value)  # type: ignore[arg-type]
            elif spec.kind == "numeric":
                encoded[spec.name] = encode_age(value)  # type: ignore[arg-type]
            else:
                encoded[spec.name] = encode_yes_no(value)  # type: ignore[arg-type]
        except EncodingError as exc:
            raise EncodingError(f"{spec.name}: {exc}") from exc
    return encoded
