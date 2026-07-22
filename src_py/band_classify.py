"""Photometric band classification - single source of truth (Band-Detect Step 2).

Consolidates fragmented filter/band logic used today by color-term gating and AAVSO export.
Production callers are **not** rewired here; use :func:`classify_photometric_band` in new code
and compare against legacy gates before switching behavior.

Conservative default for extinction / k'' callers: :data:`PhotometricBand.UNKNOWN` must be treated
as :data:`PhotometricBand.CLEAR_UNFILTERED` (fail-safe: tight colour, no unreliable k'').
Use :func:`band_failsafe_clear` or :func:`effective_band_for_extinction` for that mapping.
"""
from __future__ import annotations

import enum
import re
from typing import Final

# ---------------------------------------------------------------------------
# Public enum
# ---------------------------------------------------------------------------


class PhotometricBand(enum.Enum):
    """Broad photometric band class for pipeline policy (colour term, k'', export hints)."""

    STANDARD_FILTER = "standard_filter"
    LUMINANCE = "luminance"
    CLEAR_UNFILTERED = "clear_unfiltered"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Synonym table - multi-token / vendor strings -> canonical token (applied before class lookup)
# Keys are uppercase, whitespace-collapsed, punctuation-normalized via :func:`_synonym_lookup_key`.
# ---------------------------------------------------------------------------

FILTER_SYNONYM_TO_CANONICAL: Final[dict[str, str]] = {
    # Johnson / Bessell / Cousins (multi-word)
    "JOHNSON U": "U",
    "JOHNSON B": "B",
    "JOHNSON V": "V",
    "JOHNSON R": "R",
    "JOHNSON I": "I",
    "BESSELL U": "U",
    "BESSELL B": "B",
    "BESSELL V": "V",
    "BESSELL R": "R",
    "BESSELL I": "I",
    "COUSINS R": "RC",
    "COUSINS I": "IC",
    "COUSINS RC": "RC",
    "COUSINS IC": "IC",
    # Sloan / SDSS
    "SLOAN U": "SU",
    "SLOAN G": "SG",
    "SLOAN R": "SR",
    "SLOAN I": "SI",
    "SLOAN Z": "SZ",
    "SDSS U": "SU",
    "SDSS G": "SG",
    "SDSS R": "SR",
    "SDSS I": "SI",
    "SDSS Z": "SZ",
    # Gaia passband (distinct from Sloan g in policy - both STANDARD_FILTER)
    "GAIA G": "G",
    "GAIA_G": "G",
    # Clear / unfiltered variants
    "NO FILTER": "NOFILTER",
    "NO_FILTER": "NOFILTER",
    "NOFILTER": "NOFILTER",
    "CLEAR": "CLEAR",
    "UNFILTERED": "NOFILTER",
    "NONE": "NONE",
    # Luminance
    "LUMINANCE": "LUMINANCE",
    "LUM": "LUM",
    # OSC tri-colour labels (M67 scripts)
    "BLUE": "BLUE",
    "GREEN": "GREEN",
    "RED": "RED",
    # Clear-transformed (AAVSO codes as input tokens)
    "CLEAR V": "CV",
    "CLEAR R": "CR",
}

# Canonical tokens -> band class (uppercase keys)
_STANDARD: Final[frozenset[str]] = frozenset(
    {
        "U",
        "B",
        "V",
        "R",
        "I",
        "RJ",
        "IJ",
        "RC",
        "IC",
        "VC",
        "BC",
        "G",
        "Z",
        "GP",
        "RP",
        "IP",
        "UP",
        "ZP",
        "G'",
        "R'",
        "I'",
        "U'",
        "Z'",
        "SG",
        "SR",
        "SI",
        "SU",
        "SZ",
        "TG",
        "TB",
        "TR",
        "SLOAN_G",
        "SLOAN_R",
        "SLOAN_I",
        "SLOAN_U",
        "SLOAN_Z",
        "JOHNSON_B",
        "JOHNSON_V",
        "JOHNSON_R",
        "JOHNSON_I",
        "COUSINS_R",
        "COUSINS_I",
        "BLUE",
        "GREEN",
        "RED",
        "J",
        "H",
        "K",
        "Y",
    }
)

_LUMINANCE: Final[frozenset[str]] = frozenset({"L", "LUM", "LUMINANCE"})

_CLEAR: Final[frozenset[str]] = frozenset(
    {
        "",
        "NOFILTER",
        "NO_FILTER",
        "CLEAR",
        "C",
        "NONE",
        "CV",
        "CR",
        "CLR",
        "CL",
        "UNKNOWN",
        "NAN",
    }
)

# AAVSO FILT codes -> band (when passed as ``aavso_code``)
_AAVSO_TO_BAND: Final[dict[str, PhotometricBand]] = {
    "U": PhotometricBand.STANDARD_FILTER,
    "B": PhotometricBand.STANDARD_FILTER,
    "V": PhotometricBand.STANDARD_FILTER,
    "R": PhotometricBand.STANDARD_FILTER,
    "I": PhotometricBand.STANDARD_FILTER,
    "RJ": PhotometricBand.STANDARD_FILTER,
    "IJ": PhotometricBand.STANDARD_FILTER,
    "RC": PhotometricBand.STANDARD_FILTER,
    "IC": PhotometricBand.STANDARD_FILTER,
    "SU": PhotometricBand.STANDARD_FILTER,
    "SG": PhotometricBand.STANDARD_FILTER,
    "SR": PhotometricBand.STANDARD_FILTER,
    "SI": PhotometricBand.STANDARD_FILTER,
    "SZ": PhotometricBand.STANDARD_FILTER,
    "TB": PhotometricBand.STANDARD_FILTER,
    "TG": PhotometricBand.STANDARD_FILTER,
    "TR": PhotometricBand.STANDARD_FILTER,
    "J": PhotometricBand.STANDARD_FILTER,
    "H": PhotometricBand.STANDARD_FILTER,
    "K": PhotometricBand.STANDARD_FILTER,
    "Y": PhotometricBand.STANDARD_FILTER,
    "CV": PhotometricBand.CLEAR_UNFILTERED,
    "CR": PhotometricBand.CLEAR_UNFILTERED,
    "UNKN": PhotometricBand.UNKNOWN,
}


def _synonym_lookup_key(text: str) -> str:
    s = str(text or "").strip().upper()
    s = re.sub(r"[\s\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_FILTER_SYNONYMS: Final[dict[str, str]] = {
    _synonym_lookup_key(k): v for k, v in FILTER_SYNONYM_TO_CANONICAL.items()
}


def _canonical_lookup_key(text: str) -> str:
    """Normalize a single-token or post-synonym token for set lookup."""
    s = str(text or "").strip().upper()
    s = s.replace("''", "'").replace("''", "'")
    s = re.sub(r"[\s\-]+", "_", s)
    return s


def normalize_filter_synonym(raw: str) -> str:
    """Map vendor / multi-token filter string to a canonical single token."""
    key = _synonym_lookup_key(raw)
    if not key:
        return ""
    if key in _FILTER_SYNONYMS:
        return _FILTER_SYNONYMS[key]
    # Single-token pass-through with punctuation normalization
    return _canonical_lookup_key(raw)


def normalize_fits_filter(raw: str | None) -> str:
    """FITS FILTER normalization (matches import ``_safe_filter_token`` semantics)."""
    t = str(raw or "").strip()
    if not t or t.lower() in {"unknown", "none", "nan"}:
        return ""
    return normalize_filter_synonym(t) or t


def obs_group_first_token(obs_group: str) -> str:
    """First segment of obs_group folder name (before ``_exptime_bin`` suffix)."""
    raw = str(obs_group or "").split("|")[0].strip()
    if not raw:
        return ""
    parts = [p for p in raw.split("_") if p]
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-2])
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return parts[0] if parts else raw


def _classify_canonical_token(token: str) -> PhotometricBand:
    key = _canonical_lookup_key(token)
    if not key:
        return PhotometricBand.CLEAR_UNFILTERED
    if key in _CLEAR:
        return PhotometricBand.CLEAR_UNFILTERED
    if key in _LUMINANCE:
        return PhotometricBand.LUMINANCE
    if key in _STANDARD:
        return PhotometricBand.STANDARD_FILTER
    # Legacy broadband prefix heuristic (matches old ``_is_broadband_photometric_filter``)
    for prefix in ("B", "V", "R", "I", "U", "G", "Z"):
        if key.startswith(prefix) and len(key) <= 8:
            return PhotometricBand.STANDARD_FILTER
    return PhotometricBand.UNKNOWN


def _classify_raw_string(raw: str) -> PhotometricBand:
    if not str(raw or "").strip():
        return PhotometricBand.CLEAR_UNFILTERED
    canonical = normalize_filter_synonym(raw)
    return _classify_canonical_token(canonical)


def _classify_aavso_code(code: str | None) -> PhotometricBand | None:
    if code is None:
        return None
    c = str(code).strip().upper()
    if not c:
        return None
    if c in _AAVSO_TO_BAND:
        return _AAVSO_TO_BAND[c]
    # e.g. "Rc" from export builtin
    c_norm = c.replace(" ", "")
    if c_norm in _AAVSO_TO_BAND:
        return _AAVSO_TO_BAND[c_norm]
    if c in ("RC", "IC"):
        return PhotometricBand.STANDARD_FILTER
    return PhotometricBand.UNKNOWN


def classify_photometric_band(
    obs_group: str,
    *,
    fits_filter: str | None = None,
    aavso_code: str | None = None,
) -> PhotometricBand:
    """Classify observation band for policy (colour term, k'', export hints).

    Source priority (first definitive result wins):

    1. ``obs_group`` first token (folder name, e.g. ``NoFilter_60_2`` -> ``NoFilter``)
    2. Normalized FITS ``FILTER`` string
    3. AAVSO FILT code

    Returns :data:`PhotometricBand.UNKNOWN` when no source recognizes the band.
    **Callers doing extinction / k'' must treat UNKNOWN as CLEAR_UNFILTERED** - see
    :func:`effective_band_for_extinction`.

    Does not read config or FITS files; pass ``fits_filter`` / ``aavso_code`` explicitly.
    """
    for source in (
        obs_group_first_token(obs_group),
        normalize_fits_filter(fits_filter) if fits_filter is not None else None,
    ):
        if source is None:
            continue
        band = _classify_raw_string(source)
        if band is not PhotometricBand.UNKNOWN:
            return band

    aavso_band = _classify_aavso_code(aavso_code)
    if aavso_band is not None and aavso_band is not PhotometricBand.UNKNOWN:
        return aavso_band

    # Re-try FITS empty string explicitly (clear) after obs_group unknown
    if fits_filter is not None and not str(fits_filter).strip():
        return PhotometricBand.CLEAR_UNFILTERED

    if aavso_band is PhotometricBand.UNKNOWN:
        return PhotometricBand.UNKNOWN

    return PhotometricBand.UNKNOWN


def band_failsafe_clear(band: PhotometricBand) -> bool:
    """True when k'' / second-order extinction should use the conservative (clear) branch."""
    return band in (PhotometricBand.CLEAR_UNFILTERED, PhotometricBand.UNKNOWN, PhotometricBand.LUMINANCE)


def effective_band_for_extinction(band: PhotometricBand) -> PhotometricBand:
    """Map UNKNOWN -> CLEAR_UNFILTERED for fail-safe extinction policy."""
    if band is PhotometricBand.UNKNOWN:
        return PhotometricBand.CLEAR_UNFILTERED
    return band


def color_term_auto_from_band(band: PhotometricBand) -> bool:
    """Whether ``apply_color_term=auto`` *should* enable CT given band class (new policy)."""
    return band is PhotometricBand.STANDARD_FILTER


def compare_legacy_color_term_auto(obs_group: str) -> bool:
    """Legacy ``resolve_apply_color_term(cfg, obs_group)`` with ``apply_color_term='auto'``."""
    from config import AppConfig  # noqa: PLC0415
    from photometry_core import resolve_apply_color_term  # noqa: PLC0415

    cfg = AppConfig()
    cfg.apply_color_term = "auto"
    return bool(resolve_apply_color_term(cfg, obs_group))


def guess_aavso_code_from_obs_group(obs_group: str) -> tuple[str, str | None]:
    """Best-effort AAVSO FILT via export layer (OSC-aware)."""
    from export_reports import resolve_aavso_filt_from_obs_group  # noqa: PLC0415

    return resolve_aavso_filt_from_obs_group(obs_group, None)
