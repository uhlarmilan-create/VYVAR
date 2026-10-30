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
from typing import Any, Final


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
    # OSC tri-colour labels
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

# G05-13: broadband luminance-type filters by vendor name, matched on the compact key (upper case,
# only letters and digits: "UV/IR Cut" -> "UVIRCUT"). IR-cut / UV-IR-cut / L filters pass the whole
# visible band like a luminance filter: no colour term, reported like clear (CV). Light-pollution
# broadband filters (CLS, L-Pro) are in the same class. Narrowband filters are not listed (UNKNOWN).
_LUMINANCE_COMPACT: Final[frozenset[str]] = frozenset(
    {
        "L", "LUM", "LUMINANCE", "LRGBL", "L1", "L2", "L3",
        "IR", "IRCUT", "IRBLOCK", "IRBLOCKING", "UVIR", "UVIRCUT", "UVIRBLOCK", "IRUVCUT", "UVIRL",
        "BAADERL", "BAADERUVIRCUT", "BAADERLUVIRCUT", "ASTRONOMIKL", "ASTRONOMIKL1", "ASTRONOMIKL2",
        "ASTRONOMIKL3", "ASTRONOMIKUVIR", "ZWOL", "ZWOIRCUT", "ZWOUVIRCUT", "CHROMAL", "CHROMALUM",
        "OPTOLONGL", "OPTOLONGUVIRCUT", "OPTOLONGLPRO", "LPRO", "CLS", "ASTRONOMIKCLS", "IDAS",
        "IDASLPS", "LPS",
    }
)
_CLEAR_COMPACT: Final[frozenset[str]] = frozenset(
    {"", "NOFILTER", "CLEAR", "C", "NONE", "CLR", "CL", "UNFILTERED", "OPEN", "EMPTY", "BAADERCLEAR",
     "ZWOCLEAR", "UNKNOWN", "NAN"}
)


def _compact_key(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())


# ---------------------------------------------------------------------------
# G08-13: ONE table "filter name -> standard band" for band_classify, gaia_bands and k2_extinction.
# Values: Sloan "u" "g" "r" "i" "z"; Johnson-Cousins "U" "B" "V" "Rc" "Ic"; Johnson "Rj" "Ij".
# Single letters are case sensitive (lower case = Sloan, upper case = Johnson-Cousins; a bare R / I
# is Cousins). Upper-case "G" and "Z" have no Johnson-Cousins meaning and are read as Sloan
# (G08-14: an LRGB green "G" is open - see VYVAR_DECISIONS HDR-ONE-2026-10).
# ---------------------------------------------------------------------------
_SINGLE_LETTER_BAND: Final[dict[str, str]] = {
    "u": "u", "g": "g", "r": "r", "i": "i", "z": "z",
    "U": "U", "B": "B", "V": "V", "R": "Rc", "I": "Ic",
    "G": "g", "Z": "z", "v": "V", "b": "B",
}


def _build_band_names() -> dict[str, str]:
    out: dict[str, str] = {}
    for x in "ugriz":
        for name in (f"{x}p", f"{x}'", f"s{x}", f"sloan{x}", f"sloan{x}'", f"sdss{x}", f"sdss{x}'",
                     f"sloan{x}p"):
            out[name] = x
    for x, band in (("u", "U"), ("b", "B"), ("v", "V")):
        for name in (f"johnson{x}", f"bessell{x}", f"{x}j", f"{x}c" if x != "u" else "uj"):
            out[name] = band
    out.update({
        "rc": "Rc", "ic": "Ic", "rj": "Rj", "ij": "Ij",
        "cousinsr": "Rc", "cousinsi": "Ic", "cousinsrc": "Rc", "cousinsic": "Ic",
        "bessellr": "Rc", "besselli": "Ic", "johnsonr": "Rj", "johnsoni": "Ij",
    })
    return out


_BAND_NAMES: Final[dict[str, str]] = _build_band_names()


def standard_band_of_filter(filter_token: str | None) -> str | None:
    """Standard photometric band of a filter name (the one table, G08-13); None when not a standard band.

    Tri-colour (TG/TB/TR, Red/Green/Blue), clear, luminance and unknown names return None.
    """
    raw = str(filter_token or "").strip()
    if not raw:
        return None
    if len(raw) == 1:
        return _SINGLE_LETTER_BAND.get(raw)
    key = re.sub(r"[^a-z0-9']", "", raw.lower().replace("''", "'"))
    if len(key) == 1:
        return _SINGLE_LETTER_BAND.get(key if raw.strip(" _-").islower() else key.upper())
    return _BAND_NAMES.get(key)

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
    """FITS FILTER normalization for band lookup (empty / unknown -> "", synonyms applied)."""
    t = str(raw or "").strip()
    if not t or t.lower() in {"unknown", "none", "nan"}:
        return ""
    return normalize_filter_synonym(t) or t


# folder form of an observation group: <filter>_<exposure s, may be fractional>_<binning>[_<OSC channel>]
_FOLDER_GROUP_RE = re.compile(r"^(?P<flt>.*?)_(?P<exp>\d+(?:\.\d+)?)_(?P<bin>\d+)(?:_(?:oneRGGB|R|G|B))?$")


def obs_group_first_token(obs_group: str) -> str:
    """The filter token of an observation group - the ONE parser (G05-14, G08-13).

    Pipe form (``VY_OBSG``, ``fits_meta.observation_group_key``) ``<filter>|<exp>|<bin>``: the whole
    first field (``Sloan_r|120|2`` -> ``Sloan_r``). Folder form ``<filter>_<exp>_<bin>`` (exposure may
    be fractional, an OSC channel suffix may follow): everything before the exposure field
    (``Sloan_r_2.5_1`` -> ``Sloan_r``, ``NoFilter_60_2_G`` -> ``NoFilter``). A bare name is returned
    unchanged.
    """
    text = str(obs_group or "").strip()
    if not text:
        return ""
    if "|" in text:
        return text.split("|")[0].strip()
    m = _FOLDER_GROUP_RE.match(text)
    if m is not None and m.group("flt").strip("_"):
        return m.group("flt").strip("_")
    return text


def _classify_canonical_token(token: str) -> PhotometricBand:
    key = _canonical_lookup_key(token)
    if not key:
        return PhotometricBand.CLEAR_UNFILTERED
    if key in _CLEAR:
        return PhotometricBand.CLEAR_UNFILTERED
    if key in _LUMINANCE:
        return PhotometricBand.LUMINANCE
    if key in _STANDARD or standard_band_of_filter(token) is not None:
        return PhotometricBand.STANDARD_FILTER
    compact = _compact_key(token)
    if compact in _CLEAR_COMPACT:
        return PhotometricBand.CLEAR_UNFILTERED
    if compact in _LUMINANCE_COMPACT:
        return PhotometricBand.LUMINANCE
    # G05-13: no prefix heuristic - an unknown name is UNKNOWN (fail-safe: treated as clear)
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

    1. ``obs_group`` filter token (:func:`obs_group_first_token`, pipe or folder form)
    2. Normalized FITS ``FILTER`` string
    3. AAVSO FILT code

    Returns :data:`PhotometricBand.UNKNOWN` when no source recognizes the band.
    **Callers doing extinction / k'' must treat UNKNOWN as CLEAR_UNFILTERED** - see
    :func:`effective_band_for_extinction`.

    Does not read config or FITS files; pass ``fits_filter`` / ``aavso_code`` explicitly.
    """
    og_token = obs_group_first_token(obs_group)
    if not og_token and fits_filter is None and aavso_code is None:
        return PhotometricBand.CLEAR_UNFILTERED
    for source in (
        og_token or None,
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


def color_term_auto_from_band(band: PhotometricBand) -> bool:
    """Whether ``apply_color_term=auto`` *should* enable CT given band class (new policy)."""
    return band is PhotometricBand.STANDARD_FILTER


# --- moved from aperture_photometry (LAYER-2026-10) ---


def resolve_apply_color_term(
    cfg: Any | None,
    obs_group: str,
    *,
    fits_filter: str | None = None,
    aavso_code: str | None = None,
) -> bool:
    """User/config toggle: CT applies correction only - never limits the target set."""
    mode = str(getattr(cfg, "apply_color_term", "auto") or "auto").strip().lower()
    if mode in ("0", "false", "no", "off"):
        return False
    if mode in ("1", "true", "yes", "on"):
        return True
    band = classify_photometric_band(
        obs_group,
        fits_filter=fits_filter,
        aavso_code=aavso_code,
    )
    return bool(color_term_auto_from_band(band))
