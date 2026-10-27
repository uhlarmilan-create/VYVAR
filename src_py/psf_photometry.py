"""Effective PSF (ePSF): build-star selection and ePSF builder helpers.

``select_epsf_build_stars`` picks the build stars on the MASTERSTAR catalogue
(``masterstar_epsf_build_stars.csv``); RUN ePSF builds every frame's ePSF from them and fits the
frame with it (``epsf_frame.py``, EPSF-SPATIAL-2026-10).

Uses Photutils EPSFBuilder / PSFPhotometry. Does not import ``pipeline`` (avoid cycles).
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table
from masterstar_context import header_core_fwhm_px
from photutils import __version__ as _photutils_version
from photutils.psf import EPSFBuilder, EPSFStar, EPSFStars, extract_stars

from database import VyvarDatabase
from gaia_catalog_id import normalize_gaia_source_id as _norm_catalog_id
from infolog import log_event

LOGGER = logging.getLogger(__name__)

# EPSF-FRAME-2026-10: build stars of the MASTERSTAR ePSF; every per-frame ePSF is built from them.
EPSF_BUILD_STARS_NAME = "masterstar_epsf_build_stars.csv"
EPSF_BUILD_STARS_META_NAME = "epsf_build_stars_meta.json"
EPSF_SPATIAL_DELTA_NAME = "epsf_spatial_delta.fits"
EPSF_FRAMES_DIR = "epsf_frames"
EPSF_FRAMES_MANIFEST = "epsf_frames_manifest.json"
#: Manifest keys that describe the build, not the models (left out of the model hash).
_MANIFEST_VOLATILE_KEYS = frozenset({"created_utc"})


def epsf_models_sha256(manifest: dict[str, Any]) -> str:
    """Hash of the ePSF model set of a night (EPSF-HASH-2026-10).

    The manifest without its build time, as canonical JSON: every per-frame model sha256, the
    night spatial perturbation sha256 and the build parameters. The same models give the same
    hash in every run (the file hash changed with ``created_utc``)."""
    import hashlib  # noqa: PLC0415

    content = {k: v for k, v in manifest.items() if k not in _MANIFEST_VOLATILE_KEYS}
    blob = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


# Brightest isolated build stars kept (photutils 3.1: "a few hundred"; 600 fill a 3x3 grid with
# >= 50 stars per cell on draft 535; more stars no longer change the ePSF - EPSF-SPATIAL-2026-10).
_EPSF_BUILD_MAX_STARS = 600
# Cutout / model size in FWHM. Draft 535: the growth curve of isolated bright stars reaches 0.94 at
# 7 px and levels off only at ~12 px (6 FWHM); a 15 px model left ~5 % of a bright neighbour's light
# unmodelled next to a faint star (EPSF-BLEND-2026-10).
_EPSF_CUTOUT_FWHM = 12.0
# Iteration cap only: the build stops when 95 % of the stars' centres move < 0.001 px between
# iterations (photutils 3.1 convergence). The 49-star 535 MASTERSTAR build converges at
# iteration 18, so the photutils default of 10 would stop it unconverged.
_EPSF_BUILD_MAXITERS = 50
_EPSF_BUILD_CLEAN_SOURCE_STATES = frozenset({"DETECTED_P1", "DETECTED_P2", "FORCED_SEED"})


def _epsf_apply_build_selection_gates(
    df: pd.DataFrame,
    *,
    platesolve_dir: Path,
    cutout_size: int,
    image_shape: tuple[int, int],
    funnel: dict[str, Any],
) -> pd.DataFrame:
    """Build-star gates: linear zone, clean source_state, edge-safe cutout.

    EPSF-SPATIAL-2026-10: the build stars cover the whole field (isolated, unsaturated, not
    variable), not only the comparison stars of the science set (49 stars on draft 535): the
    per-frame ePSF and its 3x3 spatial perturbation need >= 50 stars per cell.
    """

    work = df.copy()
    funnel["n_build_input"] = int(len(work))

    if "zone" in work.columns:
        work = work[work["zone"].astype(str).str.strip().str.lower() == "linear"]
    funnel["n_after_zone_linear"] = int(len(work))

    if "source_state" in work.columns:
        ss = work["source_state"].astype(str).str.strip().str.upper()
        work = work[ss.isin(_EPSF_BUILD_CLEAN_SOURCE_STATES)]
    funnel["n_after_clean_source_state"] = int(len(work))

    h, w = image_shape
    half = int(cutout_size) // 2
    margin = half + 1
    xs = pd.to_numeric(work["x"], errors="coerce")
    ys = pd.to_numeric(work["y"], errors="coerce")
    edge_ok = (
        xs.notna()
        & ys.notna()
        & (xs >= margin)
        & (xs < w - margin)
        & (ys >= margin)
        & (ys < h - margin)
    )
    work = work.loc[edge_ok]
    funnel["n_after_edge_safe_cutout"] = int(len(work))

    return work


def _clamp_fwhm_px(v: float) -> float:
    return float(max(2.0, min(12.0, v)))


def _median_fwhm_obs_files(db: VyvarDatabase, draft_id: int) -> float | None:
    rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
    vals: list[float] = []
    for row in rows:
        try:
            x = float(row.get("FWHM"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and x > 0.0:
            vals.append(x)
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def get_epsf_fwhm_from_context(
    masterstar_fits_path: Path,
    db: VyvarDatabase,
    draft_id: int,
) -> float:
    """Return FWHM in pixels for EPSFBuilder (MASTERSTAR VY_FWHM
    -> manifest files[] median -> 4.5), clamped to [2, 12]."""
    fwhm: float | None = None
    p = Path(masterstar_fits_path)
    try:
        if p.is_file():
            with fits.open(p, memmap=True) as hdul:
                fwhm = header_core_fwhm_px(hdul[0].header)
    except Exception:  # noqa: BLE001
        fwhm = None

    if fwhm is None:
        fwhm = _median_fwhm_obs_files(db, draft_id)

    if fwhm is None:
        fwhm = 4.5

    return _clamp_fwhm_px(fwhm)


def _to_odd_cutout(n: int) -> int:
    n = max(15, int(n))
    if n % 2 == 0:
        n += 1
    return n


def _scalar_is_explicit_false(v: Any) -> bool:
    """True only for explicit false (not unknown / empty)."""
    if isinstance(v, np.bool_):
        return not bool(v)
    if v is False:
        return True
    if v is True or v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    s = str(v).strip().lower()
    if s == "":
        return False
    return s in ("false", "0", "no")


def _scalar_is_explicit_true(v: Any) -> bool:
    """True only for explicit true (not unknown / empty)."""
    if isinstance(v, np.bool_):
        return bool(v)
    if v is True:
        return True
    if v is False or v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    s = str(v).strip().lower()
    if s == "":
        return False
    return s in ("true", "1", "yes", "y")


def _read_plate_scale_arcsec_px_from_fits(fits_path: Path) -> float | None:
    """Read plate scale in arcsec/px from FITS header, CD/WCS-FIRST.

    Priority: (1) solved WCS CD matrix / CDELT; (2) VY_PLTS - ONLY if it agrees with
    the CD value within 5% (else ignored); (3) SECPIX/PIXSCALE/SCALE only when no CD.
    Returns None if no reliable value. Sanity range: 0.1-30.0 arcsec/px (wide-field safe).
    """
    _sane_min = 0.1
    _sane_max = 30.0
    try:
        with fits.open(Path(fits_path), memmap=False) as hdul:
            hdr = hdul[0].header

            # (1) Authoritative: solved WCS (handles CD, PC+CDELT, SIP), then raw CD, then CDELT.
            cd_scale: float | None = None
            try:
                import warnings  # noqa: PLC0415

                import numpy as _np  # noqa: PLC0415
                from astropy.wcs import WCS as _WCS  # noqa: PLC0415
                from astropy.wcs import FITSFixedWarning as _FFW  # noqa: PLC0415
                from astropy.wcs.utils import proj_plane_pixel_scales as _pps  # noqa: PLC0415

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", _FFW)
                    _w = _WCS(hdr)
                if _w.has_celestial:
                    v = float(_np.mean(_pps(_w))) * 3600.0
                    if _sane_min <= v <= _sane_max:
                        cd_scale = v
            except Exception:  # noqa: BLE001
                cd_scale = None
            if cd_scale is None:
                cd11 = hdr.get("CD1_1")
                cd12 = hdr.get("CD1_2", 0.0)
                if cd11 is not None:
                    try:
                        v = math.sqrt(float(cd11) ** 2 + float(cd12) ** 2) * 3600.0
                        if _sane_min <= v <= _sane_max:
                            cd_scale = v
                    except (TypeError, ValueError):
                        pass
            if cd_scale is None:
                cdelt1 = hdr.get("CDELT1")
                if cdelt1 is not None:
                    try:
                        v = abs(float(cdelt1)) * 3600.0
                        if _sane_min <= v <= _sane_max:
                            cd_scale = v
                    except (TypeError, ValueError):
                        pass

            if cd_scale is not None:
                # (2) VY_PLTS cross-check: warn and ignore if it disagrees > 5%.
                vy = hdr.get("VY_PLTS")
                if vy is not None:
                    try:
                        vyf = float(vy)
                        if vyf > 0 and abs(vyf - cd_scale) / cd_scale > 0.05:
                            logging.warning(
                                "[ePSF] VY_PLTS=%.3f disagrees with CD-derived %.3f arcsec/px (>5%%) - using CD.",
                                vyf,
                                cd_scale,
                            )
                    except (TypeError, ValueError):
                        pass
                return cd_scale

            # (3) No usable CD/WCS - header keyword fallbacks (incl. VY_PLTS).
            vy = hdr.get("VY_PLTS")
            if vy is not None:
                try:
                    v = float(vy)
                    if _sane_min <= v <= _sane_max:
                        return v
                except (TypeError, ValueError):
                    pass

            for key in ("SECPIX", "PIXSCALE", "PIXSCAL1", "PLTSCALE"):
                val = hdr.get(key)
                if val is not None:
                    try:
                        v = float(val)
                        if _sane_min <= v <= _sane_max:
                            return v
                    except (TypeError, ValueError):
                        pass

            scale = hdr.get("SCALE")
            if scale is not None:
                try:
                    v = float(scale)
                    if _sane_min <= v <= _sane_max:
                        return v
                except (TypeError, ValueError):
                    pass

            return None
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0447] Plate scale from the MASTERSTAR header unreadable: %s', exc)
        return None


# ePSF/input FWHM ratio warning band (diagnostic only; set from V3e harness scatter).
_EPSF_FWHM_RATIO_WARN_LO = 0.80
_EPSF_FWHM_RATIO_WARN_HI = 1.25


def _epsf_fwhm_native_from_profile(epsf_data: np.ndarray, *, osamp: int) -> float:
    """Azimuthally-binned radial profile half-max FWHM (native px)."""
    z = np.asarray(epsf_data, dtype=np.float64)
    cy, cx = np.array(z.shape) // 2
    yy, xx = np.indices(z.shape)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).ravel()
    v = z.ravel()
    ok = np.isfinite(v) & np.isfinite(r)
    r = r[ok]
    v = v[ok]
    if r.size < 10:
        return float("nan")
    peak = float(np.max(v))
    if peak <= 0:
        return float("nan")
    v = v / peak
    h, w = z.shape
    rmax = min(h, w) * 0.45
    bin_w = 0.5  # oversampled px
    edges = np.arange(0.0, rmax + bin_w * 0.5, bin_w)
    centers: list[float] = []
    means: list[float] = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        sel = (r >= lo) & (r < hi)
        if int(sel.sum()) < 3:
            continue
        centers.append(0.5 * (lo + hi))
        means.append(float(np.mean(v[sel])))
    if len(centers) < 4:
        return float("nan")
    c_arr = np.asarray(centers, dtype=np.float64)
    m_arr = np.asarray(means, dtype=np.float64)
    cross = None
    for i in range(len(c_arr) - 1):
        a, b = m_arr[i], m_arr[i + 1]
        if a >= 0.5 >= b and a != b:
            frac = (a - 0.5) / (a - b)
            cross = float(c_arr[i] + frac * (c_arr[i + 1] - c_arr[i]))
            break
    if cross is None:
        return float("nan")
    return float(2.0 * cross / max(1, int(osamp)))


def _epsf_build_imagepsf_from_stars(
    stars: Any,
    *,
    osamp: int,
    fwhm_px: float,
    cutout_size: int,
) -> dict[str, Any]:
    """Run EPSFBuilder on a (sky-subtracted) EPSFStars set; return normalized array + QC.

    Shared by the single global build and the spatially-varying grid build so the ePSF
    construction, QC, and ImagePSF flux normalization are identical in both paths.

    EPSF-UPGRADE-2026-10: photutils >= 3.1.1 defaults (Anderson & King 2000; Anderson 2016):
    ``smoothing_kernel='auto'`` (quartic kernel scaled to the ePSF FWHM, none when undersampled),
    ``fit_shape='auto'`` (2 x FWHM), ``recentering_func=centroid_symmetry``, the 0.375 px
    residual deposit with the alias passband, and convergence on relative star-centre changes.
    photutils 3.0 renormalized the stars every iteration with the first-iteration flux
    (photutils #2386), which made the MASTERSTAR ePSF diverge after ~5 iterations.
    """
    osamp = max(1, int(osamp))
    # EPSF-NOCLIP-2026-10 (Milan 2026-10-26, D1): no sigma clipping (INV-NOCLIP-01) - photutils
    # combines the residuals of every grid point with a plain median when ``sigma_clip=None``.
    # BO CVn, 12 frames, build/test split: same fit residual on held-out stars (1.574 vs 1.573 %),
    # PSF flux 1.4 % closer to the cutout sum, ePSF peak 1.5 % lower.
    result = EPSFBuilder(oversampling=osamp, maxiters=_EPSF_BUILD_MAXITERS, progress_bar=False,
                         sigma_clip=None)(stars)
    epsf_model = result.epsf
    info = result.iteration_info
    build_iterations = (
        [
            {
                "iteration": int(r["iteration"]),
                "stage": str(r["stage"]),
                "converged": bool(r["converged"]),
                "converged_fraction": round(float(r["converged_fraction"]), 4),
                "max_center_shift_px": round(float(r["max_center_shift"]), 5),
                "common_center_shift_px": round(float(r["common_center_shift"]), 5),
                "n_fit_failed": int(r["n_fit_failed"]),
                "max_epsf_change": round(float(r["max_epsf_change"]), 5),
            }
            for r in info
        ]
        if info is not None
        else []
    )
    kernel = result.smoothing_kernel
    _smoothing = "auto:none" if kernel is None else f"auto:{kernel.shape[1]}x{kernel.shape[0]}"
    if not result.converged:
        log_event(
            f"ePSF QC WARNING: EPSFBuilder did not converge in {result.iterations} iterations "
            f"(converged fraction {result.final_converged_fraction})"
        )
    epsf_data = epsf_model.data

    nan_frac = float(np.sum(~np.isfinite(epsf_data)) / epsf_data.size)

    # ePSF FWHM via azimuthally-binned radial profile (oversampled -> native)
    try:
        epsf_fwhm_native = _epsf_fwhm_native_from_profile(epsf_data, osamp=osamp)
    except Exception:  # noqa: BLE001
        epsf_fwhm_native = float("nan")

    # symmetry / asymmetry
    try:
        cy, cx = np.array(epsf_data.shape) // 2
        q1 = epsf_data[:cy, :cx]
        q2 = epsf_data[:cy, cx + (epsf_data.shape[1] % 2):]
        q3 = epsf_data[cy + (epsf_data.shape[0] % 2):, :cx]
        q4 = epsf_data[cy + (epsf_data.shape[0] % 2):, cx + (epsf_data.shape[1] % 2):]
        min_r = min(q1.shape[0], q2.shape[0], q3.shape[0], q4.shape[0])
        min_c = min(q1.shape[1], q2.shape[1], q3.shape[1], q4.shape[1])
        quads = np.stack([q[:min_r, :min_c] for q in [q1, q2[:, ::-1], q3[::-1, :], q4[::-1, ::-1]]])
        finite_q = np.all(np.isfinite(quads), axis=0)
        peak = float(np.nanmax(epsf_data))
        asymmetry = (
            float(np.nanstd(quads[:, finite_q], axis=0).mean()) / peak
            if (finite_q.any() and peak > 0)
            else float("nan")
        )
    except Exception:  # noqa: BLE001
        asymmetry = float("nan")

    _qc = {
        "epsf_fwhm_native_px": round(epsf_fwhm_native, 3) if math.isfinite(epsf_fwhm_native) else None,
        "epsf_vs_input_fwhm_ratio": (
            round(epsf_fwhm_native / fwhm_px, 3)
            if (math.isfinite(epsf_fwhm_native) and fwhm_px > 0)
            else None
        ),
        "epsf_nan_fraction": round(nan_frac, 4),
        "epsf_asymmetry": round(asymmetry, 4) if math.isfinite(asymmetry) else None,
    }

    _epsf_raw = np.asarray(epsf_model.data, dtype=np.float64).copy()
    _norm_factor = float(_epsf_raw.sum() / (osamp**2))
    if _norm_factor > 0 and np.isfinite(_norm_factor):
        _epsf_normalized = _epsf_raw / _norm_factor
    else:
        _epsf_normalized = _epsf_raw
    arr = np.asarray(_epsf_normalized, dtype=np.float32)
    return {
        "arr": arr,
        "qc": _qc,
        "norm_factor": _norm_factor,
        "smoothing": _smoothing,
        "epsf_sum_native": float(_epsf_normalized.sum() / osamp**2),
        "build_iterations": build_iterations,
        "build_converged": bool(result.converged),
        "build_n_excluded_stars": int(result.n_excluded_stars),
        "build_fit_shape": list(result.fit_shape) if result.fit_shape is not None else None,
        "build_center_asymmetry_px": (
            [round(float(v), 4) for v in result.center_asymmetry] if result.center_asymmetry is not None else None
        ),
    }


_CONE_CATALOG_NAME = "field_catalog_cone.csv"


def frame_mag_limit(platesolve_dir: Path) -> float:
    """Frame limit G (``frame_depth.g_lim_frame`` in photometry/pipeline_meta.json, FRAME-DEPTH-2026-10).

    Stars fainter than the frame limit are part of the sky; raises when the night has no measured depth.
    """
    pm = Path(platesolve_dir) / "photometry" / "pipeline_meta.json"
    fd = (json.loads(pm.read_text(encoding="utf-8")).get("frame_depth") or {}) if pm.is_file() else {}
    g_lim = fd.get("g_lim_frame")
    if g_lim is None or not math.isfinite(float(g_lim)):
        raise ValueError(f"frame_depth.g_lim_frame missing in {pm}")
    return float(g_lim)


def _load_cone_catalog(epsf_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load the FULL Gaia cone catalogue (ra_deg, dec_deg, mag) for ePSF isolation.

    Returns ``None`` if the cone CSV is absent/unreadable so the caller can fall back to
    the legacy candidate-vs-candidate test.
    """
    p = Path(epsf_dir) / _CONE_CATALOG_NAME
    if not p.is_file():
        return None
    c = pd.read_csv(p, low_memory=False, usecols=lambda col: col in ("ra_deg", "dec_deg", "mag"))  # unreadable: error
    ra = pd.to_numeric(c["ra_deg"], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(c["dec_deg"], errors="coerce").to_numpy(dtype=float)
    mag = pd.to_numeric(c.get("mag"), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    return ra[ok], dec[ok], mag[ok]


def _epsf_sky_subtracted_stars(data: np.ndarray, cat: Table, cutout_size: int) -> EPSFStars:
    """Cut out the build stars and subtract each cutout's sky (median of its 2-px border).

    EPSFBuilder does not subtract sky. Each ``EPSFStar`` is created from the sky-subtracted
    cutout, so its initial flux (the cutout sum the builder normalizes the star by) is the star
    flux without the sky pedestal. Until EPSF-UPGRADE-2026-10 the cutouts were extracted from the
    raw frame and their data replaced afterwards, so every star kept a flux that included the
    sky; with photutils 3.0 never updating that normalization (photutils #2386) the 535
    MASTERSTAR ePSF diverged (edge/peak 1.9 after 15 iterations).
    """
    raw = extract_stars(NDData(np.asarray(data, dtype=np.float32)), cat, size=cutout_size)
    out = []
    for star in raw:
        d = np.asarray(star.data, dtype=np.float64)
        border = np.ones(d.shape, dtype=bool)
        border[2:-2, 2:-2] = False
        vals = d[border & np.isfinite(d)]
        sky = float(np.median(vals)) if vals.size >= 4 else 0.0
        out.append(EPSFStar((d - sky).astype(np.float32), cutout_center=star.cutout_center, origin=star.origin))
    return EPSFStars(out)


def _epsf_prepare_stars(
    masterstar_fits_path: Path,
    masterstars_csv_path: Path,
    db: VyvarDatabase,
    draft_id: int,
    *,
    min_stars: int,
) -> dict[str, Any]:
    """Select clean, isolated ePSF candidate stars and return sky-subtracted EPSFStars + meta.

    Candidate positions come only from ``masterstars_full_match.csv`` (quality-filtered
    rows with finite ``x``/``y``). The ``db`` parameter is used for FWHM manifest fallback and
    gain/read-noise resolution elsewhere in the ePSF pipeline - not for star selection.

    Conscious widening (MS-SOURCES-RETIRE, 2026-08-21): production ePSF uses the full
    CSV-quality pool instead of the retired MASTER_SOURCES safe-comp subset, targeting, and
    broad-pool augment. PSF-star selection refinement belongs to ePSF-VALID-01.
    """
    mpath = Path(masterstar_fits_path)
    csvpath = Path(masterstars_csv_path)
    if not mpath.is_file():
        raise FileNotFoundError(f"MASTERSTAR FITS not found: {mpath}")
    if not csvpath.is_file():
        raise FileNotFoundError(f"masterstars_full_match.csv not found: {csvpath}")

    min_stars = max(10, int(min_stars))

    fwhm_px = get_epsf_fwhm_from_context(mpath, db, draft_id)
    cutout_size = _to_odd_cutout(int(round(fwhm_px * _EPSF_CUTOUT_FWHM)))
    log_event(f"PSF ePSF: FWHM={fwhm_px:.3f} px (clamped context), cutout_size={cutout_size}")

    def _csv_catalog_id_cell(raw: Any) -> str:
        if raw is None:
            return ""
        s = str(raw).strip()
        if not s or s.lower() in ("nan", "none"):
            return ""
        return _norm_catalog_id(raw)

    df = pd.read_csv(csvpath, low_memory=False, converters={"catalog_id": _csv_catalog_id_cell})
    funnel: dict[str, Any] = {"n_csv_input": int(len(df))}
    if "catalog_known_variable" not in df.columns:
        _vsx = (
            df["vsx_known_variable"].map(_scalar_is_explicit_true)
            if "vsx_known_variable" in df.columns
            else pd.Series(False, index=df.index)
        )
        _gvar = (
            df["gaia_dr3_variable_catalog"].map(_scalar_is_explicit_true)
            if "gaia_dr3_variable_catalog" in df.columns
            else pd.Series(False, index=df.index)
        )
        df = df.copy()
        df["catalog_known_variable"] = _vsx | _gvar

    req = ("catalog_id", "catalog_known_variable", "likely_saturated", "photometry_ok", "x", "y")
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(
            f"masterstars_full_match.csv missing required columns {missing} in {csvpath}"
        )

    _funnel_mask = pd.Series(True, index=df.index)
    _funnel_mask &= df["catalog_known_variable"].map(_scalar_is_explicit_false)
    funnel["n_after_variable_excluded"] = int(_funnel_mask.sum())
    _funnel_mask &= df["likely_saturated"].map(_scalar_is_explicit_false)
    funnel["n_after_peak_sat_likely"] = int(_funnel_mask.sum())
    if "is_saturated" in df.columns:
        _funnel_mask &= ~df["is_saturated"].fillna(False).astype(bool)
        funnel["n_after_peak_sat_is_saturated"] = int(_funnel_mask.sum())
    else:
        funnel["n_after_peak_sat_is_saturated"] = funnel["n_after_peak_sat_likely"]
    _funnel_mask &= df["photometry_ok"].map(_scalar_is_explicit_true)
    funnel["n_after_photometry_ok"] = int(_funnel_mask.sum())
    if "is_noisy" in df.columns:
        _funnel_mask &= ~df["is_noisy"].fillna(False).astype(bool)
        funnel["n_after_not_noisy"] = int(_funnel_mask.sum())
    else:
        funnel["n_after_not_noisy"] = funnel["n_after_photometry_ok"]
    if "is_usable" in df.columns:
        _funnel_mask &= df["is_usable"].fillna(False).astype(bool)
        funnel["n_after_usable"] = int(_funnel_mask.sum())
    else:
        funnel["n_after_usable"] = funnel["n_after_not_noisy"]
    funnel["n_after_snr_cut"] = None  # SNR not filtered in _epsf_prepare_stars
    funnel["n_after_snr_cut_reason"] = "not applied in this function"

    csv_mask = (
        df["catalog_known_variable"].map(_scalar_is_explicit_false)
        & df["likely_saturated"].map(_scalar_is_explicit_false)
        & df["photometry_ok"].map(_scalar_is_explicit_true)
    )
    if "is_saturated" in df.columns:
        csv_mask &= ~df["is_saturated"].fillna(False).astype(bool)
    if "is_noisy" in df.columns:
        csv_mask &= ~df["is_noisy"].fillna(False).astype(bool)
    if "is_usable" in df.columns:
        csv_mask &= df["is_usable"].fillna(False).astype(bool)
    csv_ok = df.loc[csv_mask].copy()
    with fits.open(mpath, memmap=True) as _hd_gate:
        _gate_shape = np.asarray(_hd_gate[0].data).shape
    csv_ok = _epsf_apply_build_selection_gates(
        csv_ok,
        platesolve_dir=mpath.parent,
        cutout_size=cutout_size,
        image_shape=(int(_gate_shape[0]), int(_gate_shape[1])),
        funnel=funnel,
    )
    csv_ok["_cid"] = csv_ok["catalog_id"].map(_norm_catalog_id)
    csv_ok = csv_ok[csv_ok["_cid"] != ""]
    funnel["n_csv_with_catalog_id"] = int(len(csv_ok))
    log_event(f"PSF ePSF: CSV filter -> {len(csv_ok)} rows with non-empty catalog_id")
    LOGGER.info(
        "[ePSF funnel] after CSV quality: n_input=%s n_after_sat=%s n_after_photometry_ok=%s n_csv_with_id=%s",
        funnel.get("n_csv_input"),
        funnel.get("n_after_peak_sat_is_saturated"),
        funnel.get("n_after_photometry_ok"),
        funnel.get("n_csv_with_catalog_id"),
    )

    _ra_by_cid: dict[str, float] = {}
    _dec_by_cid: dict[str, float] = {}
    _mag_by_cid: dict[str, float] = {}
    _mag_col = next((c for c in ("mag", "catalog_mag", "phot_g_mean_mag") if c in csv_ok.columns), None)
    if "ra_deg" in csv_ok.columns and "dec_deg" in csv_ok.columns:
        for _, r in csv_ok.iterrows():
            cid = str(r.get("_cid", "") or "").strip()
            if not cid:
                continue
            try:
                ra_v = float(pd.to_numeric(r.get("ra_deg"), errors="coerce"))
                de_v = float(pd.to_numeric(r.get("dec_deg"), errors="coerce"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(ra_v) and math.isfinite(de_v):
                _ra_by_cid[cid] = ra_v
                _dec_by_cid[cid] = de_v
            if _mag_col is not None:
                mg = float(pd.to_numeric(r.get(_mag_col), errors="coerce"))
                if math.isfinite(mg):
                    _mag_by_cid[cid] = mg

    star_rows: list[dict[str, Any]] = []
    for _, r in csv_ok.iterrows():
        cid = str(r.get("_cid", "") or "").strip()
        if not cid:
            continue
        try:
            x_f = float(pd.to_numeric(r.get("x"), errors="coerce"))
            y_f = float(pd.to_numeric(r.get("y"), errors="coerce"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(x_f) and math.isfinite(y_f)):
            continue
        star_rows.append(
            {
                "catalog_id": cid,
                "x": x_f,
                "y": y_f,
                "ra_deg": _ra_by_cid.get(cid, float("nan")),
                "dec_deg": _dec_by_cid.get(cid, float("nan")),
                "mag": _mag_by_cid.get(cid, float("nan")),
            }
        )

    funnel["used_broad_pool"] = False
    funnel["n_safe_comp_csv_join"] = int(len(star_rows))

    if len(star_rows) == 0:
        funnel["n_after_csv_select"] = 0
        funnel["n_final"] = 0
        _fmsg = (
            "ePSF candidate funnel: "
            + ", ".join(f"{k}={v}" for k, v in funnel.items())
            + f" - no ePSF candidates after CSV quality filter ({csvpath.name})"
        )
        LOGGER.info("[ePSF funnel] %s", _fmsg)
        raise ValueError(_fmsg)

    psf_stars_df = pd.DataFrame(star_rows)
    n_join = int(len(psf_stars_df))
    funnel["n_after_csv_select"] = n_join
    funnel["n_after_db_join"] = n_join
    funnel["n_after_targeting"] = n_join
    log_event(
        f"PSF ePSF: masterstars_full_match.csv candidates -> {n_join} star positions "
        f"(file: {csvpath.name})"
    )
    LOGGER.info("[ePSF funnel] csv path: n_after_csv_select=%s", n_join)

    if n_join < min_stars:
        funnel["n_final"] = n_join
        _fmsg = (
            f"EPSF build needs at least {min_stars} clean stars from {csvpath.name}; found {n_join}. "
            f"Funnel: {funnel}"
        )
        LOGGER.info("[ePSF funnel] insufficient after CSV select: %s", _fmsg)
        raise ValueError(_fmsg)

    # Isolation: reject any candidate with a Gaia *cone-catalogue* neighbour within
    # NxFWHM (correct plate scale) that is within ~Deltamag (brighter, or up to Deltamag fainter).
    # Compared against the FULL cone - not just the candidate set - so a bright NON-candidate
    # neighbour correctly disqualifies a candidate (the previous candidate-vs-candidate test
    # let such stars pass). Falls back to candidate-vs-candidate if the cone CSV is missing.
    _isolation_fwhm_mult = 3.0
    _isolation_delta_mag = 2.5
    _isolation_radius_px = float(fwhm_px) * _isolation_fwhm_mult
    plate_scale_arcsec_px = _read_plate_scale_arcsec_px_from_fits(mpath)
    n_before_iso = int(len(psf_stars_df))
    funnel["n_before_isolation"] = n_before_iso
    if plate_scale_arcsec_px is None or not math.isfinite(plate_scale_arcsec_px) or plate_scale_arcsec_px <= 0:
        # FAIL-LOUD-2026-10 (code review 2 G09-10): build stars without the isolation filter would put bright
        # neighbours into every frame ePSF; the solved MASTERSTAR always has a plate scale
        raise RuntimeError(f"ePSF build stars: MASTERSTAR plate scale unavailable ({mpath})")
    if n_join > 0 and "ra_deg" in psf_stars_df.columns and "dec_deg" in psf_stars_df.columns:
        from sky_separation import angular_distance_deg_vectorized as _angular_distance_deg_vectorized

        radius_deg = _isolation_radius_px * float(plate_scale_arcsec_px) / 3600.0
        self_deg = 0.5 * float(plate_scale_arcsec_px) / 3600.0  # <0.5px counts as the star itself
        _cand_mag = (
            pd.to_numeric(psf_stars_df.get("mag"), errors="coerce").to_numpy(dtype=float)
            if "mag" in psf_stars_df.columns
            else np.full(len(psf_stars_df), np.nan)
        )
        _cone = _load_cone_catalog(mpath.parent)
        _isolated: list[bool] = []
        if _cone is not None:
            cone_ra, cone_dec, cone_mag = _cone
            _n_rej = 0
            for i, row in psf_stars_df.iterrows():
                ra_i = float(row["ra_deg"])
                de_i = float(row["dec_deg"])
                if not (math.isfinite(ra_i) and math.isfinite(de_i)):
                    _isolated.append(False)
                    continue
                cosd = max(math.cos(math.radians(de_i)), 0.2)
                box = (np.abs(cone_dec - de_i) <= radius_deg * 1.5) & (
                    np.abs(cone_ra - ra_i) <= radius_deg * 1.5 / cosd
                )
                if not box.any():
                    _isolated.append(True)
                    continue
                d_deg = _angular_distance_deg_vectorized(ra_i, de_i, cone_ra[box], cone_dec[box])
                m_box = cone_mag[box]
                cand_m = float(_cand_mag[i]) if (i < len(_cand_mag) and math.isfinite(_cand_mag[i])) else float("nan")
                near = (d_deg > self_deg) & (d_deg <= radius_deg)
                if math.isfinite(cand_m):
                    contaminating = near & ((m_box - cand_m) <= _isolation_delta_mag)
                else:
                    contaminating = near  # no candidate mag -> any neighbour disqualifies (conservative)
                iso = not bool(np.any(contaminating))
                _isolated.append(iso)
                if not iso:
                    _n_rej += 1
            psf_stars_df = psf_stars_df.loc[_isolated].reset_index(drop=True)
            funnel["n_after_isolation"] = int(len(psf_stars_df))
            log_event(
                f"[ePSF] Cone isolation ({_isolation_fwhm_mult}xFWHM={_isolation_radius_px:.1f}px, "
                f"Deltamag<={_isolation_delta_mag}): rejected {_n_rej}/{n_before_iso}, kept {len(psf_stars_df)}"
            )
            LOGGER.info(
                "[ePSF funnel] after isolation: n_before=%s n_after=%s rejected=%s",
                n_before_iso,
                funnel["n_after_isolation"],
                _n_rej,
            )
        else:
            _ra_arr = psf_stars_df["ra_deg"].to_numpy(dtype=float)
            _dec_arr = psf_stars_df["dec_deg"].to_numpy(dtype=float)
            for _, row in psf_stars_df.iterrows():
                ra_i = float(row["ra_deg"])
                de_i = float(row["dec_deg"])
                if not (math.isfinite(ra_i) and math.isfinite(de_i)):
                    _isolated.append(False)
                    continue
                dists_px = (
                    _angular_distance_deg_vectorized(ra_i, de_i, _ra_arr, _dec_arr)
                    * 3600.0
                    / float(plate_scale_arcsec_px)
                )
                neighbor_dists = dists_px[dists_px > 0.01]
                _isolated.append(
                    len(neighbor_dists) == 0 or float(neighbor_dists.min()) > _isolation_radius_px
                )
            psf_stars_df = psf_stars_df.loc[_isolated].reset_index(drop=True)
            funnel["n_after_isolation"] = int(len(psf_stars_df))
            LOGGER.info(
                "[ePSF] Isolation (candidate-set fallback, %.1fxFWHM=%.1fpx): %d PSF stars",
                _isolation_fwhm_mult,
                _isolation_radius_px,
                len(psf_stars_df),
            )
            LOGGER.info(
                "[ePSF funnel] after isolation (fallback): n_before=%s n_after=%s",
                n_before_iso,
                funnel["n_after_isolation"],
            )
        if len(psf_stars_df) < min_stars:
            funnel["n_final"] = int(len(psf_stars_df))
            _fmsg = (
                f"EPSF build needs at least {min_stars} isolated stars; found {len(psf_stars_df)}. "
                f"Funnel: {funnel}"
            )
            LOGGER.info("[ePSF funnel] insufficient after isolation: %s", _fmsg)
            raise ValueError(_fmsg)

    if len(psf_stars_df) > _EPSF_BUILD_MAX_STARS:
        psf_stars_df = (
            psf_stars_df.assign(_m=pd.to_numeric(psf_stars_df["mag"], errors="coerce"))
            .sort_values("_m", na_position="last", kind="mergesort")
            .head(_EPSF_BUILD_MAX_STARS)
            .drop(columns="_m")
            .reset_index(drop=True)
        )
    funnel["n_after_brightest_cap"] = int(len(psf_stars_df))
    funnel["build_max_stars"] = int(_EPSF_BUILD_MAX_STARS)

    xs = psf_stars_df["x"].to_numpy(dtype=float).tolist()
    ys = psf_stars_df["y"].to_numpy(dtype=float).tolist()
    cat = Table([xs, ys], names=("x", "y"))
    with fits.open(mpath, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)

    stars = _epsf_sky_subtracted_stars(data, cat, cutout_size)
    n_ext = int(getattr(stars, "n_stars", 0))
    funnel["n_after_extract_stars"] = n_ext
    funnel["n_final"] = n_ext
    log_event(f"PSF ePSF: extract_stars retained {n_ext} cutouts (size={cutout_size})")
    LOGGER.info("[ePSF funnel] final: %s", funnel)
    if n_ext < min_stars:
        _fmsg = (
            f"EPSF build needs at least {min_stars} extractable star cutouts; got {n_ext} "
            f"(many positions may lie outside the frame or overlap borders). Funnel: {funnel}"
        )
        LOGGER.info("[ePSF funnel] insufficient extract_stars: %s", _fmsg)
        raise ValueError(_fmsg)

    return {
        "mpath": mpath,
        "min_stars": int(min_stars),
        "fwhm_px": float(fwhm_px),
        "cutout_size": int(cutout_size),
        "data": data,
        "stars": stars,
        "n_ext": int(n_ext),
        "n_join": int(n_join),
        "n_before_isolation": int(n_before_iso),
        "n_after_isolation": int(len(psf_stars_df)),
        "isolation_radius_px": float(_isolation_radius_px),
        "plate_scale_arcsec_px": plate_scale_arcsec_px,
        "funnel": funnel,
        "psf_stars_df": psf_stars_df,
    }


def select_epsf_build_stars(
    masterstar_fits_path: Path,
    masterstars_csv_path: Path,
    db: VyvarDatabase,
    draft_id: int,
    *,
    min_stars: int,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Select the ePSF build stars on the MASTERSTAR catalogue (EPSF-SPATIAL-2026-10).

    MASTERSTAR serves aperture photometry; the ePSF is built on every frame (Milan 2026-10-18), so
    only the star selection comes from the MASTERSTAR catalogue: isolated, unsaturated, not variable,
    clean source state, edge-safe, the brightest ``_EPSF_BUILD_MAX_STARS``. Writes
    ``masterstar_epsf_build_stars.csv`` and ``epsf_build_stars_meta.json`` (FWHM, cutout size,
    selection funnel) next to the MASTERSTAR (or into ``out_dir``) and returns the meta.
    """
    prep = _epsf_prepare_stars(masterstar_fits_path, masterstars_csv_path, db, draft_id, min_stars=min_stars)
    out_dir = Path(out_dir) if out_dir is not None else Path(prep["mpath"]).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stars_df = prep["psf_stars_df"][["catalog_id", "x", "y", "mag"]]
    stars_df.to_csv(out_dir / EPSF_BUILD_STARS_NAME, index=False)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    meta = {
        "fwhm_px": float(prep["fwhm_px"]),
        "cutout_size": int(prep["cutout_size"]),
        "n_stars": int(len(stars_df)),
        "isolation_radius_px": float(prep["isolation_radius_px"]),
        "plate_scale_arcsec_px": prep["plate_scale_arcsec_px"],
        "build_funnel": dict(prep.get("funnel") or {}),
        "build_max_stars": int(_EPSF_BUILD_MAX_STARS),
        "draft_id": int(draft_id),
        "photutils_version": _photutils_version,
        "created_utc": created,
    }
    (out_dir / EPSF_BUILD_STARS_META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log_event(f"PSF ePSF: {len(stars_df)} build stars selected on MASTERSTAR -> {EPSF_BUILD_STARS_NAME}")
    return meta


def _psf_resolve_gain_read_noise(
    frame_hdr: Any,
    *,
    photometry_dir: Path,
    cfg: Any,
) -> tuple[float, float]:
    """Gain (e-/ADU container) and read noise (e-) for the PSF fit weights: the values the run's
    aperture photometry used (GAIN-ONE-2026-10, aperture_per_target.json); the frame header / config through
    the same resolver only when that file is missing."""
    from aperture_pertarget import load_photometric_gain  # noqa: PLC0415

    gain, rn = load_photometric_gain(photometry_dir)
    if not (math.isfinite(gain) and gain > 0 and math.isfinite(rn) and rn >= 0):
        from rig_photometry import photometric_gain_read_noise  # noqa: PLC0415

        pg = photometric_gain_read_noise(frame_hdr, cfg, None)
        if not (math.isfinite(gain) and gain > 0):
            gain = float(pg.gain)
        if not (math.isfinite(rn) and rn >= 0):
            rn = float(pg.read_noise)
    return float(gain), float(rn)



__all__ = [
    "select_epsf_build_stars",
    "get_epsf_fwhm_from_context",
]
