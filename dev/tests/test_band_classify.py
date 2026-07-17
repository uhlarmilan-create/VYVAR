"""Unit tests for band_classify (Band-Detect Step 2)."""
from __future__ import annotations

import pytest

from band_classify import (
    FILTER_SYNONYM_TO_CANONICAL,
    PhotometricBand,
    band_failsafe_clear,
    classify_photometric_band,
    color_term_auto_from_band,
    compare_legacy_color_term_auto,
    effective_band_for_extinction,
    guess_aavso_code_from_obs_group,
    normalize_filter_synonym,
    normalize_fits_filter,
    obs_group_first_token,
)


@pytest.mark.parametrize(
    ("obs_group", "fits_filter", "aavso", "expected"),
    [
        ("NoFilter_60_2", "", None, PhotometricBand.CLEAR_UNFILTERED),
        ("NoFilter_60_2", None, "CV", PhotometricBand.CLEAR_UNFILTERED),
        ("B_20_2", None, None, PhotometricBand.STANDARD_FILTER),
        ("V_20_2", None, None, PhotometricBand.STANDARD_FILTER),
        ("R_20_2", None, None, PhotometricBand.STANDARD_FILTER),
        ("L_20_2", None, None, PhotometricBand.LUMINANCE),
        ("r_60_4", None, None, PhotometricBand.STANDARD_FILTER),
        ("Blue_20_2", None, None, PhotometricBand.STANDARD_FILTER),
        ("Green_20_2", None, None, PhotometricBand.STANDARD_FILTER),
        ("Red_20_2", None, None, PhotometricBand.STANDARD_FILTER),
        ("Luminance_20_2", None, None, PhotometricBand.LUMINANCE),
        ("Lum_20_2", None, None, PhotometricBand.LUMINANCE),
        ("CV_20_2", None, None, PhotometricBand.CLEAR_UNFILTERED),
        ("CR_20_2", None, None, PhotometricBand.CLEAR_UNFILTERED),
        ("", None, None, PhotometricBand.CLEAR_UNFILTERED),
        ("foo_20_2", None, None, PhotometricBand.UNKNOWN),
        ("foo_20_2", "V", None, PhotometricBand.STANDARD_FILTER),
        ("unknown_20_2", "unknown", None, PhotometricBand.CLEAR_UNFILTERED),
        ("unknown_20_2", "nan", None, PhotometricBand.CLEAR_UNFILTERED),
        ("SG_30_1", None, None, PhotometricBand.STANDARD_FILTER),
    ],
)
def test_classify_obs_groups(
    obs_group: str,
    fits_filter: str | None,
    aavso: str | None,
    expected: PhotometricBand,
) -> None:
    assert classify_photometric_band(obs_group, fits_filter=fits_filter, aavso_code=aavso) is expected


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("Johnson V", "V"),
        ("johnson v", "V"),
        ("Bessell B", "B"),
        ("BEssell B", "B"),
        ("Gaia G", "G"),
        ("gaia g", "G"),
        ("Sloan G", "SG"),
        ("No Filter", "NOFILTER"),
        ("Clear", "CLEAR"),
        ("Luminance", "LUMINANCE"),
        ("g'", "G'"),
        ("G'", "G'"),
    ],
)
def test_synonym_normalization(raw: str, canonical: str) -> None:
    assert normalize_filter_synonym(raw) == canonical


@pytest.mark.parametrize(
    ("fits", "norm"),
    [
        ("", ""),
        ("  ", ""),
        ("unknown", ""),
        ("none", ""),
        ("nan", ""),
        ("  V  ", "V"),
        ("Johnson V", "V"),
    ],
)
def test_normalize_fits_filter(fits: str, norm: str) -> None:
    assert normalize_fits_filter(fits) == norm


def test_obs_group_first_token() -> None:
    assert obs_group_first_token("NoFilter_60_2") == "NoFilter"
    assert obs_group_first_token("B_20_2") == "B"
    assert obs_group_first_token("Johnson_V_20_2") == "Johnson_V"


def test_aavso_code_fallback() -> None:
    assert (
        classify_photometric_band("foo_20_2", aavso_code="V")
        is PhotometricBand.STANDARD_FILTER
    )
    assert (
        classify_photometric_band("foo_20_2", aavso_code="CV")
        is PhotometricBand.CLEAR_UNFILTERED
    )
    assert classify_photometric_band("foo_20_2", aavso_code="UNKN") is PhotometricBand.UNKNOWN


def test_failsafe_extinction() -> None:
    assert effective_band_for_extinction(PhotometricBand.UNKNOWN) is PhotometricBand.CLEAR_UNFILTERED
    assert effective_band_for_extinction(PhotometricBand.STANDARD_FILTER) is PhotometricBand.STANDARD_FILTER
    assert band_failsafe_clear(PhotometricBand.UNKNOWN) is True
    assert band_failsafe_clear(PhotometricBand.STANDARD_FILTER) is False
    assert band_failsafe_clear(PhotometricBand.LUMINANCE) is True


def test_synonym_table_nonempty() -> None:
    assert "JOHNSON V" in {k.upper() for k in FILTER_SYNONYM_TO_CANONICAL}


@pytest.mark.parametrize(
    ("obs_group", "legacy_ct", "new_ct"),
    [
        ("NoFilter_60_2", False, False),
        ("B_20_2", True, True),
        ("V_20_2", True, True),
        ("R_20_2", True, True),
        ("L_20_2", False, False),
        ("CV_20_2", False, False),  # band_classify wired (legacy broadband True retired)
        ("CR_20_2", False, False),
    ],
)
def test_documented_flip_cases(obs_group: str, legacy_ct: bool, new_ct: bool) -> None:
    band = classify_photometric_band(obs_group)
    assert color_term_auto_from_band(band) is new_ct
    assert compare_legacy_color_term_auto(obs_group) is legacy_ct


def test_guess_aavso_nofilter() -> None:
    code, warn = guess_aavso_code_from_obs_group("NoFilter_60_2")
    assert code == "CV"
    assert warn is None


def test_whitespace_case_variants() -> None:
    assert classify_photometric_band("  v_20_2  ") is PhotometricBand.STANDARD_FILTER
    assert classify_photometric_band("CLEAR_60_2") is PhotometricBand.CLEAR_UNFILTERED
    assert classify_photometric_band("nofilter_60_2") is PhotometricBand.CLEAR_UNFILTERED
