"""Tests for lcrisk.schema: encoding correctness and the original app bug."""

from __future__ import annotations

import pytest

from lcrisk.schema import (
    FEATURE_NAMES,
    FEMALE,
    MALE,
    NO,
    YES,
    EncodingError,
    encode_age,
    encode_patient,
    encode_sex,
    encode_target,
    encode_yes_no,
)


class TestEncodeYesNoRegression:
    """Regression test for the original app's truthiness bug.

    The original ``map_binary``::

        def map_binary(val):
            return 2 if val or val == "Yes" else 1

    evaluates ``val`` for truthiness before comparing to ``"Yes"``. Since the
    non-empty string ``"No"`` is truthy, ``map_binary("No")`` returned 2 (the
    "yes" code) -- identical to ``map_binary("Yes")``. Every dropdown answer of
    "No" was silently treated as "Yes". These tests must fail if that logic
    ever reappears.
    """

    def test_no_string_encodes_to_no_not_yes(self) -> None:
        assert encode_yes_no("No") == NO

    def test_yes_and_no_strings_encode_differently(self) -> None:
        assert encode_yes_no("Yes") != encode_yes_no("No")

    def test_no_is_case_and_whitespace_insensitive(self) -> None:
        for text in ("no", "NO", " No ", "N", "n"):
            assert encode_yes_no(text) == NO, f"{text!r} should encode to NO"

    def test_yes_is_case_and_whitespace_insensitive(self) -> None:
        for text in ("yes", "YES", " Yes ", "Y", "y"):
            assert encode_yes_no(text) == YES, f"{text!r} should encode to YES"

    def test_booleans_are_not_affected_by_the_bug(self) -> None:
        # Checkboxes passed real booleans and were unaffected by the original
        # bug; this test just documents that boundary.
        assert encode_yes_no(True) == YES
        assert encode_yes_no(False) == NO

    def test_falsy_but_wrong_type_string_is_not_silently_accepted(self) -> None:
        # An empty string is falsy in Python but is not a valid answer, and
        # must raise rather than silently resolve to NO via truthiness.
        with pytest.raises(EncodingError):
            encode_yes_no("")


class TestEncodeYesNoStrictness:
    def test_rejects_ambiguous_strings(self) -> None:
        with pytest.raises(EncodingError):
            encode_yes_no("maybe")

    def test_rejects_out_of_domain_integers(self) -> None:
        with pytest.raises(EncodingError):
            encode_yes_no(0)
        with pytest.raises(EncodingError):
            encode_yes_no(3)

    def test_accepts_already_encoded_integers(self) -> None:
        assert encode_yes_no(NO) == NO
        assert encode_yes_no(YES) == YES

    def test_rejects_unsupported_type(self) -> None:
        with pytest.raises(EncodingError):
            encode_yes_no(None)  # type: ignore[arg-type]
        with pytest.raises(EncodingError):
            encode_yes_no(1.5)  # type: ignore[arg-type]


class TestEncodeSex:
    @pytest.mark.parametrize("text", ["M", "m", "Male", "male", " M "])
    def test_male_variants(self, text: str) -> None:
        assert encode_sex(text) == MALE

    @pytest.mark.parametrize("text", ["F", "f", "Female", "female"])
    def test_female_variants(self, text: str) -> None:
        assert encode_sex(text) == FEMALE

    def test_rejects_unknown_string(self) -> None:
        with pytest.raises(EncodingError):
            encode_sex("other")

    def test_rejects_bool(self) -> None:
        with pytest.raises(EncodingError):
            encode_sex(True)  # type: ignore[arg-type]

    def test_accepts_encoded_ints(self) -> None:
        assert encode_sex(0) == FEMALE
        assert encode_sex(1) == MALE

    def test_rejects_out_of_domain_int(self) -> None:
        with pytest.raises(EncodingError):
            encode_sex(2)


class TestEncodeAge:
    def test_accepts_valid_age(self) -> None:
        assert encode_age(60) == 60
        assert encode_age(60.0) == 60

    def test_rejects_below_minimum(self) -> None:
        with pytest.raises(EncodingError):
            encode_age(5)

    def test_rejects_above_maximum(self) -> None:
        with pytest.raises(EncodingError):
            encode_age(150)

    def test_rejects_nan(self) -> None:
        with pytest.raises(EncodingError):
            encode_age(float("nan"))

    def test_rejects_bool(self) -> None:
        with pytest.raises(EncodingError):
            encode_age(True)  # type: ignore[arg-type]

    def test_rejects_non_numeric(self) -> None:
        with pytest.raises(EncodingError):
            encode_age("60")  # type: ignore[arg-type]


class TestEncodeTarget:
    @pytest.mark.parametrize("value,expected", [("YES", 1), ("no", 0), (1, 1), (0, 0)])
    def test_encodes(self, value: object, expected: int) -> None:
        assert encode_target(value) == expected  # type: ignore[arg-type]

    def test_rejects_unknown_label(self) -> None:
        with pytest.raises(EncodingError):
            encode_target("maybe")


class TestEncodePatient:
    def test_produces_all_canonical_features_in_order(
        self, healthy_patient: dict[str, object]
    ) -> None:
        encoded = encode_patient(healthy_patient)
        assert list(encoded.keys()) == list(FEATURE_NAMES)

    def test_healthy_patient_is_all_no(self, healthy_patient: dict[str, object]) -> None:
        encoded = encode_patient(healthy_patient)
        for name in FEATURE_NAMES:
            if name not in ("AGE", "GENDER"):
                assert encoded[name] == NO

    def test_missing_feature_raises(self, healthy_patient: dict[str, object]) -> None:
        incomplete = dict(healthy_patient)
        del incomplete["SMOKING"]
        with pytest.raises(EncodingError, match="SMOKING"):
            encode_patient(incomplete)

    def test_bad_value_error_names_the_feature(
        self, healthy_patient: dict[str, object]
    ) -> None:
        bad = dict(healthy_patient)
        bad["SMOKING"] = "unsure"
        with pytest.raises(EncodingError, match="SMOKING"):
            encode_patient(bad)

    def test_mixed_no_and_yes_answers_do_not_collapse_together(self) -> None:
        # A direct end-to-end check that "No" and "Yes" answers passed through
        # the same dict produce different encoded values for each feature.
        base = {name: "No" for name in FEATURE_NAMES if name not in ("AGE", "GENDER")}
        base["AGE"] = 50
        base["GENDER"] = "F"

        all_no = encode_patient(base)
        one_yes = dict(base)
        one_yes["SMOKING"] = "Yes"
        with_smoking = encode_patient(one_yes)

        assert all_no["SMOKING"] == NO
        assert with_smoking["SMOKING"] == YES
        # every other feature must be unaffected
        for name in FEATURE_NAMES:
            if name != "SMOKING":
                assert all_no[name] == with_smoking[name]
