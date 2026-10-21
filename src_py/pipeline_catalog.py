"""Moved from pipeline.py (CONSOLIDATE-01E6a). Facade re-exports these names.

Catalog queries, field-catalog cone cache, per-frame export workers,
DAO-to-catalog match, saturation zones, airmass fills.
Spawn-worker global `_EXPORT_PER_FRAME_WORKER_STATE` lives here so
`global` binds one namespace.
The four giants stay in pipeline.py this wave (E6b).
"""
from __future__ import annotations

import astro_env  # noqa: F401  (IERS-OFF-2026-10: no IERS downloads)

import json
import logging
import pickle
import math
import warnings
from pathlib import Path
from typing import Any, Sequence
import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning
import pandas as pd
from config import AppConfig
from catalog_match_trust import export_catalog_match_mode_from_internal
from catalog_db import query_local_exoplanet, query_local_gaia, query_local_vsx
from database import VyvarDatabase
from time_utils import _header_float as _header_float_tu
from photometry import enhance_catalog_dataframe_aperture
from infolog import log_event
from proc_frame_store import proc_csv_path_for_aligned_fits
from plain_stats import plain_mean_med_std
from utils import DAO_STAR_FINDER_NO_ROUNDNESS_FILTER, dao_detection_fwhm_pixels
from masterstar_gaia_accounting import _dao_xy_binned_to_full
from pipeline_calibrate import (
    _effective_saturation_limit,
    _has_valid_wcs,
)
from pipeline_preprocess import (
    _load_raw_for_frame,
    _load_raw_hdr_for_frame,
)
from pipeline_astrometry import (
    _finite_positive_adu,
    _vyvar_df_to_csv,
)
from pipeline_constants import (
    LINEARITY_FRAC,
    SAT_LIMIT_CONTAINER_CLIP_ADU,
    SAT_LIMIT_NO_KNEE_FRAC,
    SAT_LIMIT_PEAK_TEST_SOURCE,
    _EXO_HOST_ANNOTATION_COLUMNS,
)

# Same named logger as pipeline.LOGGER (logging.getLogger singleton).
# Avoids pipeline -> pipeline_catalog -> pipeline at module load.
LOGGER = logging.getLogger("pipeline")

def _apply_aperture_catalog_enhancements_from_st(
    df: pd.DataFrame,
    data: Any,
    hdr: fits.Header,
    st: dict[str, Any],
) -> pd.DataFrame:
    """Aperture photometry + linearity flags for per-frame catalog DataFrames."""

    try:
        _go = st.get("gaussian_fwhm_px_override")
        try:
            _go_f = float(_go) if _go is not None else None
        except (TypeError, ValueError):
            _go_f = None
        return enhance_catalog_dataframe_aperture(
            df,
            data,
            hdr,
            aperture_fwhm_factor=float(st.get("aperture_fwhm_factor", 1.7)),
            annulus_inner_fwhm=float(st.get("annulus_inner_fwhm", 4.0)),
            annulus_outer_fwhm=float(st.get("annulus_outer_fwhm", 6.0)),
            gaussian_fwhm_px_override=_go_f,
            err_empty_apertures_n=int(st.get("err_empty_apertures_n", 64)),
            err_empty_apertures_min=int(st.get("err_empty_apertures_min", 16)),
            fwhm_night_median_px=st.get("fwhm_night_median_px"),
            qc_fwhm_by_name=st.get("qc_fwhm_by_name"),
            frame_name=st.get("current_frame_name"),
        )
    except Exception as exc:  # noqa: BLE001
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().catalog_aperture_enhance_fail += 1
        LOGGER.error("[CATALOG] aperture enhancement failed: %s", exc)
        return df


def _fits_header_vy_algn_aligned(hdr: fits.Header) -> bool:
    """True when frame pixels are on the MASTERSTAR reference alignment grid (VY_ALGN)."""
    try:
        val = hdr.get("VY_ALGN")
        if val is None:
            return True  # legacy frames without tag - keep pixel-fallback behaviour
        if isinstance(val, tuple):
            val = val[0]
        if isinstance(val, bool):
            return bool(val)
        s = str(val).strip().lower()
        if s in ("true", "1", "t", "yes"):
            return True
        if s in ("false", "0", "f", "no"):
            return False
    except Exception:  # noqa: BLE001
        pass
    return True


def _sat_ctx_from_worker(st: dict[str, Any]) -> Any | None:
    raw = st.get("sat_diag_ctx_dict")
    if not raw:
        return None
    try:
        from sat_diag import PileupResult, SatDiagContext  # noqa: PLC0415

        pileup_raw = raw.get("pileup")
        pileup = PileupResult(**pileup_raw) if isinstance(pileup_raw, dict) else None
        d = dict(raw)
        d.pop("pileup", None)
        d["pileup"] = pileup
        return SatDiagContext(**{k: v for k, v in d.items() if k in SatDiagContext.__dataclass_fields__})
    except Exception:  # noqa: BLE001
        return None


def find_qc_metrics_csv(
    archive_path: Path | str,
    *,
    draft_id: int | None = None,
    db: VyvarDatabase | None = None,
) -> Path | None:
    """Find ``qc_metrics.csv`` under draft lights root (legacy fallback: processed/lights)."""
    from draft_provenance import draft_archive_root, resolve_draft_lights_root

    ap = draft_archive_root(Path(archive_path).expanduser())
    candidates: list[Path] = [
        resolve_draft_lights_root(ap, draft_id=draft_id, db=db) / "qc_metrics.csv",
        ap / "processed" / "lights" / "qc_metrics.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _bin2d_mean(arr: "np.ndarray", factor: int) -> "np.ndarray":
    """Mean-bin a 2D array by integer ``factor`` (>=1). Used to speed up reference scoring."""
    import numpy as np

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or int(factor) < 2:
        return arr
    f = int(factor)
    h, w = arr.shape
    h2, w2 = (h // f) * f, (w // f) * f
    if h2 < f or w2 < f:
        return arr
    a = arr[:h2, :w2].reshape(h2 // f, f, w2 // f, f)
    return np.mean(a, axis=(1, 3)).astype(np.float32)


def _dao_star_count_from_array(arr: "np.ndarray", *, fwhm_px: float = 3.0) -> int:
    """Count DAOStarFinder sources (same recipe as alignment star detection)."""
    import numpy as np
    from photutils.detection import DAOStarFinder

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        return 0
    finite = np.isfinite(arr)
    if not np.any(finite):
        return 0
    _, med, std = plain_mean_med_std(arr[finite], sigma=3.0, maxiters=5)
    std = float(std) if np.isfinite(std) else 0.0
    if std <= 0:
        return 0
    img2 = np.nan_to_num((arr - float(med)).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        finder = DAOStarFinder(
            fwhm=float(fwhm_px),
            threshold=max(3.0 * std, 1e-6),
            **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
        )
        tbl = finder(img2)
    except Exception:  # noqa: BLE001
        return 0
    if tbl is None or len(tbl) == 0:
        return 0
    return int(len(tbl))


def _pick_reference_frame_by_star_count(files: list[Path]) -> tuple[Path, dict[str, int]]:
    """Choose the FITS with the most detected stars as alignment / plate-solve reference.

    Uses 2x2 binning + slightly smaller DAO FWHM for speed; preserves relative rankings.
    """
    import numpy as np

    scores: dict[str, int] = {}
    for fp in files:
        try:
            with fits.open(fp, memmap=False) as hdul:
                data = np.array(hdul[0].data, dtype=np.float32, copy=True)
            data_b = _bin2d_mean(data, 2)
            # 2x2 binning: FWHM v px ~ sips_dao_fwhm_px/2 (sirsie okno pre komu v rohoch).
            scores[str(fp)] = _dao_star_count_from_array(data_b, fwhm_px=2.5)
        except Exception:  # noqa: BLE001
            scores[str(fp)] = 0
    if not files:
        raise ValueError("no FITS files")
    if not scores or max(scores.values(), default=0) <= 0:
        return files[0], scores
    best_n = max(scores.values())
    for fp in files:
        if scores.get(str(fp), 0) == best_n:
            return fp, scores
    return files[0], scores


def cone_sky_box(center: SkyCoord, radius_deg: float) -> tuple[float, float, float, float]:
    """RA/Dec box [deg] that contains the great-circle cone (CATALOG-FOOTPRINT-2026-10).

    RA half-width is ``r / cos(dec)`` at the box edge nearest the pole (a box of +-r in RA covers
    only r*cos(dec) on the sky: at Dec 61 a 4.7 deg cone lost the chip edges). A cone that reaches
    a pole takes every RA. RA may lie outside [0, 360); ``query_local_*`` splits the wrap."""
    ra0 = float(center.icrs.ra.deg)
    de0 = float(center.icrs.dec.deg)
    r = max(0.0, float(radius_deg))
    de_min = max(-90.0, de0 - r)
    de_max = min(90.0, de0 + r)
    d_edge = max(abs(de_min), abs(de_max))
    if d_edge >= 89.999:
        return 0.0, 360.0, de_min, de_max
    half = r / max(math.cos(math.radians(d_edge)), 1e-6)
    if half >= 180.0:
        return 0.0, 360.0, de_min, de_max
    return ra0 - half, ra0 + half, de_min, de_max


def _query_gaia_local(
    *,
    center: SkyCoord,
    radius_deg: float,
    gaia_db_path: Path | None,
    max_mag: float | None = None,
) -> pd.DataFrame:
    """Every local Gaia DR3 star in the great-circle cone (ICRS deg), optionally G <= ``max_mag``.

    CATALOG-FOOTPRINT-2026-10: no brightest-first row cap. The cone follows the solved chip
    footprint, so the catalogue is complete to the DB depth on the chip (draft 532: a 100 000-row
    brightest cap in an oversized cone cut the identity catalogue at G 13.46 while DAO detected
    stars to G ~15.5)."""
    try:
        _ra_l = float(center.icrs.ra.deg)
        _de_l = float(center.icrs.dec.deg)
    except Exception:  # noqa: BLE001
        _ra_l, _de_l = float("nan"), float("nan")
    log_event(f"CATALOG SEARCH (GAIA local): Ra={_ra_l}, Dec={_de_l}, Radius={float(radius_deg):.2f} deg")
    if gaia_db_path is None:
        return pd.DataFrame()
    gp = Path(gaia_db_path).expanduser().resolve()
    if not gp.is_file():
        return pd.DataFrame()

    ra_min, ra_max, de_min, de_max = cone_sky_box(center, float(radius_deg))
    _ql_kw: dict[str, Any] = {"ra_min": ra_min, "ra_max": ra_max, "dec_min": de_min, "dec_max": de_max}
    if max_mag is not None:
        try:
            _mm = float(max_mag)
            if math.isfinite(_mm) and _mm > 0:
                _ql_kw["mag_limit"] = _mm
        except (TypeError, ValueError):
            pass
    rows = query_local_gaia(gp, **_ql_kw)
    if not rows:
        return pd.DataFrame()
    df0 = pd.DataFrame(rows)
    if "bp_rp" not in df0.columns and "bp_mag" in df0.columns and "rp_mag" in df0.columns:
        df0["bp_rp"] = pd.to_numeric(df0["bp_mag"], errors="coerce") - pd.to_numeric(
            df0["rp_mag"], errors="coerce"
        )
    df = df0.rename(
        columns={"source_id": "catalog_id", "ra": "ra_deg", "dec": "dec_deg", "g_mag": "mag", "bp_rp": "bp_rp"}
    )
    df["catalog"] = "GAIA_DR3"
    # Great-circle cone cut (the SQL query is a box).
    _raq = pd.to_numeric(df["ra_deg"], errors="coerce")
    _deq = pd.to_numeric(df["dec_deg"], errors="coerce")
    _okq = _raq.notna() & _deq.notna()
    if bool(_okq.any()):
        sub = df.loc[_okq].copy()
        _coo_q = SkyCoord(
            ra=pd.to_numeric(sub["ra_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
            dec=pd.to_numeric(sub["dec_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
            frame="icrs",
        )
        _inner = center.separation(_coo_q).deg <= float(radius_deg) + 1e-9
        df = sub.loc[_inner].reset_index(drop=True)
    if max_mag is not None and "mag" in df.columns:
        m = pd.to_numeric(df["mag"], errors="coerce")
        df = df[(m.notna()) & (m <= float(max_mag))].copy()
    df = df.reset_index(drop=True)
    # Gaia provides BP-RP; do not map it into B-V (different color index).
    if "b_v" not in df.columns:
        df = df.copy()
        df["b_v"] = np.nan
    if "bp_rp" in df.columns:
        df["bp_rp"] = pd.to_numeric(df["bp_rp"], errors="coerce")
    _gm = pd.to_numeric(df.get("mag"), errors="coerce")
    log_event(
        f"GAIA cone: {len(df)} stars in r={float(radius_deg):.3f} deg"
        + (f", G {float(_gm.min()):.2f}-{float(_gm.max()):.2f}" if len(df) and _gm.notna().any() else "")
    )
    return df


def _query_vsx_local(
    *,
    center: SkyCoord,
    radius_deg: float,
    vsx_db_path: Path | None,
    require_db: bool = False,
) -> pd.DataFrame:
    """Query **local VSX SQLite** for the field; same kuzel ako Gaia (najprv obdlznik, potom great-circle orez)."""
    from catalog_db import count_vsx_rows, require_vsx_db

    if require_db:
        vp = require_vsx_db(vsx_db_path)
        n_total = count_vsx_rows(vp)
    elif vsx_db_path is None:
        return pd.DataFrame()
    else:
        vp = Path(vsx_db_path).expanduser().resolve()
        if not vp.is_file():
            return pd.DataFrame()
        n_total = count_vsx_rows(vp)
    try:
        _ra_l = float(center.icrs.ra.deg)
        _de_l = float(center.icrs.dec.deg)
    except Exception:  # noqa: BLE001
        if require_db:
            raise
        return pd.DataFrame()
    # CATALOG-FOOTPRINT-2026-10: cone box (cos dec), no row cap (a LIMIT without ORDER BY
    # dropped a contiguous slice of the field).
    ra_min, ra_max, de_min, de_max = cone_sky_box(center, float(radius_deg))
    rows = query_local_vsx(vp, ra_min=ra_min, ra_max=ra_max, dec_min=de_min, dec_max=de_max)
    if not rows:
        if require_db:
            log_event(
                f"VSX cone=0 on {vp} ({n_total} total rows) - field genuinely empty "
                f"(r~{float(radius_deg):.3f} deg, Ra={_ra_l:.4f}, Dec={_de_l:.4f})"
            )
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "ra_deg" not in df.columns or "dec_deg" not in df.columns:
        if require_db:
            from catalog_db import VSXCatalogError

            raise VSXCatalogError(f"VSX query returned unexpected columns from {vp}.")
        return pd.DataFrame()
    _raq = pd.to_numeric(df["ra_deg"], errors="coerce")
    _deq = pd.to_numeric(df["dec_deg"], errors="coerce")
    _okq = _raq.notna() & _deq.notna()
    if not bool(_okq.any()):
        if require_db:
            log_event(
                f"VSX cone=0 on {vp} ({n_total} total rows) - field genuinely empty "
                f"(r~{float(radius_deg):.3f} deg, Ra={_ra_l:.4f}, Dec={_de_l:.4f})"
            )
        return pd.DataFrame()
    sub = df.loc[_okq].copy()
    _coo_q = SkyCoord(
        ra=pd.to_numeric(sub["ra_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
        dec=pd.to_numeric(sub["dec_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
        frame="icrs",
    )
    _inner = center.separation(_coo_q).deg <= float(radius_deg) + 1e-9
    out = sub.loc[_inner].reset_index(drop=True)
    if require_db and out.empty:
        log_event(
            f"VSX cone=0 on {vp} ({n_total} total rows) - field genuinely empty "
            f"(r~{float(radius_deg):.3f} deg, Ra={_ra_l:.4f}, Dec={_de_l:.4f})"
        )
    elif not out.empty:
        try:
            log_event(
                f"CATALOG SEARCH (VSX local): {len(out)} zdrojov v kuzeli r~{float(radius_deg):.3f} deg "
                f"(Ra={_ra_l:.4f}, Dec={_de_l:.4f})"
            )
        except Exception:  # noqa: BLE001
            pass
    return out


def _query_exoplanet_local(
    *,
    center: SkyCoord,
    radius_deg: float,
    exoplanet_db_path: Path | None,
) -> pd.DataFrame:
    """Query local exoplanet host SQLite for the field; box query + great-circle cone filter."""
    if exoplanet_db_path is None:
        return pd.DataFrame()
    from catalog_db import require_exoplanet_db  # noqa: PLC0415

    ep = require_exoplanet_db(exoplanet_db_path)
    try:
        _ra_l = float(center.icrs.ra.deg)
        _de_l = float(center.icrs.dec.deg)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    ra_min, ra_max, de_min, de_max = cone_sky_box(center, float(radius_deg))
    rows = query_local_exoplanet(ep, ra_min=ra_min, ra_max=ra_max, dec_min=de_min, dec_max=de_max)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "ra_deg" not in df.columns or "dec_deg" not in df.columns:
        return pd.DataFrame()
    _raq = pd.to_numeric(df["ra_deg"], errors="coerce")
    _deq = pd.to_numeric(df["dec_deg"], errors="coerce")
    _okq = _raq.notna() & _deq.notna()
    if not bool(_okq.any()):
        return pd.DataFrame()
    sub = df.loc[_okq].copy()
    _coo_q = SkyCoord(
        ra=pd.to_numeric(sub["ra_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
        dec=pd.to_numeric(sub["dec_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
        frame="icrs",
    )
    _inner = center.separation(_coo_q).deg <= float(radius_deg) + 1e-9
    out = sub.loc[_inner].reset_index(drop=True)
    try:
        log_event(
            f"CATALOG SEARCH (exoplanet local): {len(out)} hostov v kuzeli r~{float(radius_deg):.3f} deg "
            f"(Ra={_ra_l:.4f}, Dec={_de_l:.4f})"
        )
    except Exception:  # noqa: BLE001
        pass
    return out


def _exo_host_annotation_arrays(
    det_coords: SkyCoord,
    exo_df: pd.DataFrame,
    max_sep_arcsec: float,
    *,
    frame_name: str = "",
) -> tuple[dict[str, Any], list[str]]:
    """Nearest exoplanet host per detection within ``max_sep_arcsec`` (informational only)."""
    import numpy as np
    from astropy.coordinates import search_around_sky

    n = len(det_coords)
    out: dict[str, Any] = {
        "exo_host_obj_id": np.array([""] * n, dtype=object),
        "exo_host_name": np.array([""] * n, dtype=object),
        "exo_cat_source": np.array([""] * n, dtype=object),
        "exo_disposition": np.array([""] * n, dtype=object),
        "exo_match_sep_arcsec": np.full(n, np.nan, dtype=np.float64),
    }
    warnings_out: list[str] = []
    if exo_df is None or exo_df.empty or n == 0:
        return out, warnings_out

    exc = SkyCoord(
        ra=np.asarray(exo_df["ra_deg"], dtype=float) * u.deg,
        dec=np.asarray(exo_df["dec_deg"], dtype=float) * u.deg,
        frame="icrs",
    )
    max_sep = float(max_sep_arcsec)
    idx_nearest, sep2d, _ = det_coords.match_to_catalog_sky(exc)
    sep_arc = np.asarray(sep2d.to(u.arcsec).value, dtype=np.float64)
    hit = np.isfinite(sep_arc) & (sep_arc <= max_sep)

    try:
        idx_d, idx_e, sep_a, _ = search_around_sky(det_coords, exc, max_sep * u.arcsec)
        counts: dict[int, int] = {}
        for i_d in np.asarray(idx_d, dtype=np.int64):
            counts[int(i_d)] = counts.get(int(i_d), 0) + 1
        for i_d, cnt in counts.items():
            if cnt > 1:
                msg = (
                    f"[EXO MATCH] {cnt} exoplanet hosts within {max_sep:g} arcsec of detection "
                    f"index {i_d}"
                    + (f" ({frame_name})" if frame_name else "")
                )
                warnings_out.append(msg)
                LOGGER.warning(msg)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[EXO MATCH] ambiguity search skipped: %s", exc)

    for i in np.flatnonzero(hit):
        j = int(idx_nearest[i])
        if j < 0 or j >= len(exo_df):
            continue
        row = exo_df.iloc[j]
        out["exo_host_obj_id"][i] = str(row.get("obj_id", "") or "").strip()
        out["exo_host_name"][i] = str(row.get("host_name", "") or "").strip()
        out["exo_cat_source"][i] = str(row.get("cat_source", "") or "").strip()
        out["exo_disposition"][i] = str(row.get("disposition", "") or "").strip()
        out["exo_match_sep_arcsec"][i] = float(sep_arc[i])

    return out, warnings_out


def _slice_exo_annotation(exo_ann: dict[str, Any], keep: Any) -> dict[str, Any]:
    import numpy as np

    k = np.asarray(keep, dtype=bool)
    return {col: np.asarray(exo_ann[col])[k] for col in _EXO_HOST_ANNOTATION_COLUMNS}


def _build_exoplanet_promotion_rows_from_masterstars(
    master_df: pd.DataFrame,
    hdr: fits.Header,
    cfg: AppConfig,
    *,
    frame_w_px: int,
    frame_h_px: int,
    margin_px: float = 50.0,
) -> pd.DataFrame:
    """Promote masterstars rows with exoplanet host match within configured separation."""
    _exs = str(cfg.objects_db_path or "").strip()
    if not _exs:
        return pd.DataFrame()
    from catalog_db import require_exoplanet_db  # noqa: PLC0415

    try:
        _exo_path = require_exoplanet_db(_exs)
    except Exception:
        raise
    if master_df is None or master_df.empty:
        return pd.DataFrame()
    if "catalog_id" not in master_df.columns or "ra_deg" not in master_df.columns or "dec_deg" not in master_df.columns:
        return pd.DataFrame()

    import numpy as np

    from gaia_catalog_id import catalog_id_series_for_masterstars_export, masterstar_row_gaia_key

    m = master_df.copy()
    if "catalog_id" in m.columns:
        m["catalog_id"] = catalog_id_series_for_masterstars_export(m)
    cid_s = m["catalog_id"].fillna("").astype(str).str.strip()
    ok_cid = cid_s.ne("") & ~cid_s.str.lower().isin({"nan", "none"})
    ra = pd.to_numeric(m["ra_deg"], errors="coerce")
    de = pd.to_numeric(m["dec_deg"], errors="coerce")
    ok_sky = ra.notna() & de.notna()
    m = m.loc[ok_cid & ok_sky].copy()
    if m.empty:
        return pd.DataFrame()

    if "x" in m.columns and "y" in m.columns:
        xn = pd.to_numeric(m["x"], errors="coerce")
        yn = pd.to_numeric(m["y"], errors="coerce")
        in_frame = xn.between(-float(margin_px), float(frame_w_px) + float(margin_px)) & yn.between(
            -float(margin_px), float(frame_h_px) + float(margin_px)
        )
        m = m.loc[in_frame].copy()
    if m.empty:
        return pd.DataFrame()

    exo_max = 3.0
    try:
        exo_max = float(cfg.exoplanet_match_max_sep_arcsec)
        if not math.isfinite(exo_max):
            exo_max = 3.0
    except (TypeError, ValueError):
        exo_max = 3.0
    exo_max = max(0.5, min(30.0, float(exo_max)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs_obj = WCS(hdr)
    if not wcs_obj.has_celestial:
        return pd.DataFrame()
    center, radius_deg = _effective_field_catalog_cone_radius_deg(wcs_obj, int(frame_h_px), int(frame_w_px))
    exo_df = _query_exoplanet_local(
        center=center,
        radius_deg=radius_deg,
        exoplanet_db_path=_exo_path,
    )
    n_hosts_in_field = int(len(exo_df)) if exo_df is not None and not exo_df.empty else 0

    ra_arr = pd.to_numeric(m["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
    de_arr = pd.to_numeric(m["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
    det_coords = SkyCoord(ra=ra_arr * u.deg, dec=de_arr * u.deg, frame="icrs")
    exo_ann, _ = _exo_host_annotation_arrays(
        det_coords,
        exo_df if exo_df is not None else pd.DataFrame(),
        exo_max,
    )

    exo_ids = np.asarray(exo_ann["exo_host_obj_id"], dtype=object)
    exo_sep = np.asarray(exo_ann["exo_match_sep_arcsec"], dtype=np.float64)
    promote_mask = np.zeros(len(m), dtype=bool)
    for i in range(len(m)):
        oid = str(exo_ids[i] or "").strip()
        sep = float(exo_sep[i]) if math.isfinite(float(exo_sep[i])) else float("nan")
        promote_mask[i] = bool(oid) and math.isfinite(sep) and sep <= exo_max
    if not bool(np.any(promote_mask)):
        log_event(
            f"[EXO TARGET] funnel: hosts_in_field={n_hosts_in_field} masterstars_in_frame={len(m)} "
            f"promoted=0 sep_max={exo_max:g} arcsec"
        )
        return pd.DataFrame()

    sub = m.loc[promote_mask].copy()
    prom_idx = np.where(promote_mask)[0]

    rows: list[dict[str, Any]] = []
    for j, (_, row) in enumerate(sub.iterrows()):
        ii = int(prom_idx[j])
        cid_norm = masterstar_row_gaia_key(row)
        if not cid_norm:
            continue
        host_name = str(exo_ann["exo_host_name"][ii] or "").strip()
        obj_id = str(exo_ann["exo_host_obj_id"][ii] or "").strip()
        disp = str(exo_ann["exo_disposition"][ii] or "").strip()
        src = str(exo_ann["exo_cat_source"][ii] or "").strip()
        try:
            mag_v = float(pd.to_numeric(row.get("mag", row.get("phot_g_mean_mag")), errors="coerce"))
        except (TypeError, ValueError):
            mag_v = float("nan")
        if not math.isfinite(mag_v):
            mag_v = float("nan")
        rows.append(
            {
                "name": host_name or obj_id,
                "catalog_id": cid_norm,
                "catalog": "EXOPLANET",
                "ra_deg": float(row["ra_deg"]),
                "dec_deg": float(row["dec_deg"]),
                "priority": 2,
                "notes": f"{src} {disp}".strip(),
                "vsx_name": "",
                "vsx_type": "",
                "vsx_period": np.nan,
                "x": pd.to_numeric(row.get("x"), errors="coerce"),
                "y": pd.to_numeric(row.get("y"), errors="coerce"),
                "mag": mag_v,
                "zone": str(row.get("zone", "") or "").strip().lower(),
                "gaia_match_arcsec": float(exo_sep[ii]),
                "gaia_match_quality": "good",
                "gaia_match_source": "masterstars_exo",
                "vsx_mag_max": np.nan,
                "exo_host_obj_id": obj_id,
                "exo_host_name": host_name,
                "exo_cat_source": src,
                "exo_disposition": disp,
                "exo_match_sep_arcsec": float(exo_sep[ii]),
                "target_origin": "EXOPLANET",
            }
        )
    if not rows:
        log_event(
            f"[EXO TARGET] funnel: hosts_in_field={n_hosts_in_field} masterstars_in_frame={len(m)} "
            f"promoted=0 sep_max={exo_max:g} arcsec (no Gaia catalog_id on matched rows)"
        )
        return pd.DataFrame()
    log_event(
        f"[EXO TARGET] funnel: hosts_in_field={n_hosts_in_field} masterstars_in_frame={len(m)} "
        f"promoted={len(rows)} sep_max={exo_max:g} arcsec"
    )
    log_event(f"[EXO TARGET] {len(rows)} exoplanet host(s) promoted from masterstars (<={exo_max:g} arcsec)")
    return pd.DataFrame(rows)


def inv_sat_limit_peak_test_adu() -> tuple[float, str]:
    """Catalog-zone peak-test (INV-SAT-LIMIT). One authority for zone and per-frame.

    Hard container clip is ``SAT_LIMIT_CONTAINER_CLIP_ADU`` (named separately;
    not used as the clean-frame test). This returns the peak-test used for
    ``zone==saturated``: currently 0.80 x 65535 = 52428 ADU when the D1-2 knee
    is unmeasured. n_stack is not applied here: per-frame CSVs are one exposure.
    """
    peak = float(SAT_LIMIT_CONTAINER_CLIP_ADU) * float(SAT_LIMIT_NO_KNEE_FRAC)
    return peak, SAT_LIMIT_PEAK_TEST_SOURCE


def _box_peak_max_adu(data: "np.ndarray", x: float, y: float, half: int = 3) -> float:
    """Maximum pixel value in ``(2*half+1)^2`` box around ``(x,y)`` on the **original** image (linear units)."""
    import numpy as np

    arr = np.asarray(data)
    if arr.ndim != 2:
        return float("nan")
    h, w = arr.shape
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    y0, y1 = max(0, yi - half), min(h, yi + half + 1)
    x0, x1 = max(0, xi - half), min(w, xi + half + 1)
    if y0 >= y1 or x0 >= x1:
        return float("nan")
    return float(np.nanmax(arr[y0:y1, x0:x1]))


def _box_peaks_at_centroids(
    arr: "np.ndarray",
    x: "np.ndarray",
    y: "np.ndarray",
    *,
    half: int = 3,
) -> "np.ndarray":
    """Maximum ADU in each ``(2*half+1)^2`` box centred on ``round(x), round(y)`` (vectorized).

    Used for per-star saturation on thousands of DAO detections; matches ``_box_peak_max_adu`` on
    interior pixels. Falls back to a Python loop if SciPy is unavailable.
    """
    import numpy as np

    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(xa.size)
    if n == 0:
        return np.zeros([], dtype=np.float64)
    if n != int(ya.size):
        raise ValueError("x and y must have the same length")

    try:
        from scipy.ndimage import maximum_filter  # type: ignore
    except ImportError:
        return np.asarray(
            [_box_peak_max_adu(arr, float(xa[i]), float(ya[i]), half=half) for i in range(n)],
            dtype=np.float64,
        )

    a = np.asarray(np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0))
    size = int(2 * int(half) + 1)
    mf = maximum_filter(a, size=size, mode="nearest")
    xi = np.rint(xa).astype(np.intp, copy=False)
    yi = np.rint(ya).astype(np.intp, copy=False)
    h, w = mf.shape
    xi = np.clip(xi, 0, w - 1)
    yi = np.clip(yi, 0, h - 1)
    return np.asarray(mf[yi, xi], dtype=np.float64)


def _icrs_deg_to_unitxyz(ra_deg: "np.ndarray", dec_deg: "np.ndarray") -> "np.ndarray":
    """ICRS degrees -> unit direction vectors on the celestial sphere (N,3)."""
    import numpy as np

    ra = np.radians(np.asarray(ra_deg, dtype=np.float64).ravel())
    de = np.radians(np.asarray(dec_deg, dtype=np.float64).ravel())
    cd = np.cos(de)
    return np.column_stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(de)])


def _chord_to_arcsec(dist_chord: "np.ndarray") -> "np.ndarray":
    """Chord length between unit sphere points (0...2) -> great-circle separation in arcseconds."""
    import numpy as np

    d = np.asarray(dist_chord, dtype=np.float64)
    half = np.clip(d * 0.5, 0.0, 1.0)
    return np.degrees(2.0 * np.arcsin(half)) * 3600.0


def build_ucac_catalog_kdtree(cat_df: pd.DataFrame) -> tuple[Any, "np.ndarray"] | None:
    """Build a SciPy ``cKDTree`` on finite ``ra_deg``/``dec_deg`` rows and row indices into ``cat_df``.

    Read-only ``query`` calls are thread-safe across workers sharing the same tree.
    """
    import numpy as np

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    if cat_df is None or cat_df.empty or "ra_deg" not in cat_df.columns or "dec_deg" not in cat_df.columns:
        return None
    ra = np.asarray(pd.to_numeric(cat_df["ra_deg"], errors="coerce"), dtype=np.float64)
    de = np.asarray(pd.to_numeric(cat_df["dec_deg"], errors="coerce"), dtype=np.float64)
    m = np.isfinite(ra) & np.isfinite(de)
    if not np.any(m):
        return None
    orig_idx = np.nonzero(m)[0].astype(np.int64)
    xyz = _icrs_deg_to_unitxyz(ra[m], de[m])
    return cKDTree(xyz), orig_idx


def _all_pix2world_icrs_deg(wcs_obj: WCS, x: "np.ndarray", y: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    """Vectorized pixel -> world (degrees) for celestial axes; single WCS call (no per-star Python loops)."""
    import numpy as np

    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        world0, world1 = wcs_obj.all_pix2world(xa, ya, 0)
    return (
        np.asarray(world0, dtype=np.float64).ravel(),
        np.asarray(world1, dtype=np.float64).ravel(),
    )


def _saturated_core_plateau_vectorized(
    data: "np.ndarray",
    x: "np.ndarray",
    y: "np.ndarray",
    *,
    half_inner: int = 1,
    plateau_rel: float = 0.996,
    min_plateau_pixels: int = 5,
) -> "np.ndarray":
    """Vectorized saturated-core plateau over ``(x,y)`` centroids (3x3 patch per star)."""
    import numpy as np

    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        return np.zeros(len(np.asarray(x).ravel()), dtype=bool)
    h, w = arr.shape
    n = int(np.asarray(x, dtype=np.float64).size)
    yi = np.rint(np.asarray(y, dtype=np.float64).ravel()).astype(np.int32, copy=False)
    xi = np.rint(np.asarray(x, dtype=np.float64).ravel()).astype(np.int32, copy=False)
    hi = max(0, int(half_inner))
    offs = np.arange(-hi, hi + 1, dtype=np.int32)
    yy = yi.reshape(-1, 1, 1) + offs.reshape(1, -1, 1)
    xx = xi.reshape(-1, 1, 1) + offs.reshape(1, 1, -1)
    yy = np.clip(yy, 0, h - 1)
    xx = np.clip(xx, 0, w - 1)
    patches = arr[yy, xx]
    pmax = np.nanmax(patches.reshape(n, -1), axis=1)
    thr = pmax * float(plateau_rel)
    thr3 = thr.reshape(-1, 1, 1)
    n_high = np.sum(np.isfinite(patches) & (patches >= thr3), axis=(1, 2))
    return (n_high >= int(min_plateau_pixels)) & np.isfinite(pmax) & (pmax > 0)


def _vectorized_star_saturation_columns(
    arr: "np.ndarray",
    x: "np.ndarray",
    y: "np.ndarray",
    *,
    sat_limit: float | None,
    sat_frac: float,
    peak_dao: "np.ndarray",
    peak_max_adu: "np.ndarray",
) -> dict[str, "np.ndarray"]:
    """Per-star saturation flags as column arrays (vectorized over all centroids)."""
    import numpy as np

    n = int(np.asarray(x, dtype=np.float64).size)
    pmax = np.asarray(peak_max_adu, dtype=np.float64).reshape(-1)
    lim = sat_limit
    sf = float(sat_frac)
    pdv = np.asarray(peak_dao, dtype=np.float64).reshape(-1)

    # SAT-LIN-2026-10: ``sat_limit`` is already the peak-test level (fraction applied
    # once by _resolve_peak_saturation_limit_adu); ``sat_frac`` is not applied again.
    _ = sf
    if lim is not None:
        sat_peak = np.isfinite(pmax) & (pmax >= float(lim))
        plateau_skip_thr = float(lim) * 0.55
        need_plateau = np.isfinite(pmax) & (pmax >= plateau_skip_thr)
    else:
        sat_peak = np.zeros(n, dtype=bool)
        need_plateau = np.ones(n, dtype=bool)

    pl_full = _saturated_core_plateau_vectorized(arr, x, y, half_inner=1, plateau_rel=0.996, min_plateau_pixels=5)
    if lim is None:
        sat_plateau = pl_full
    else:
        sat_plateau = np.where(need_plateau, pl_full, False)

    likely = sat_peak | sat_plateau
    peak_dao_col = pdv.copy()
    peak_dao_col[~np.isfinite(peak_dao_col)] = np.nan
    peak_max_col = pmax.copy()
    peak_max_col[~np.isfinite(peak_max_col)] = np.nan
    sl = np.full(n, np.nan, dtype=np.float64)
    if lim is not None:
        sl[:] = float(lim)
    return {
        "peak_dao": peak_dao_col,
        "peak_max_adu": peak_max_col,
        "saturate_limit_adu": sl,
        "saturated_from_peak": sat_peak,
        "saturated_plateau": sat_plateau,
        "likely_saturated": likely,
        "photometry_ok": ~likely,
    }


def _proc_sat_block_for_csv(sat_block: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    """Omit obsolete proc columns; return (csv_block, n_saturated_from_peak, n_saturated_plateau) for meta."""
    import numpy as np

    pk = sat_block.get("saturated_from_peak")
    pl = sat_block.get("saturated_plateau")
    n_pk = int(np.asarray(pk, dtype=bool).sum()) if pk is not None else 0
    n_pl = int(np.asarray(pl, dtype=bool).sum()) if pl is not None else 0
    csv_block = {
        k: v for k, v in sat_block.items() if k not in ("saturated_from_peak", "saturated_plateau")
    }
    return csv_block, n_pk, n_pl


def _field_center_and_radius_from_wcs(w: WCS, h: int, wpx: int) -> tuple[SkyCoord, float]:
    """Footprint for **one** Vizier/Gaia cone query tied to the detector - not 'cela obloha'.

    Uses the geometric pixel centre, then the **maximum** great-circle separation from that centre to a
    dense sample of the **full rectangle border** (not only corners). With sip / distortion, the farthest
    sky point from the centre is not always a corner - undersized cones produced a visible circular
    'matched only in the middle' QA overlay. Adds multiplicative + absolute margin for edge stars and
    small plate-solve errors.
    """
    center = SkyCoord.from_pixel((wpx - 1) / 2.0, (h - 1) / 2.0, wcs=w, origin=0)
    max_sep_deg = 0.0
    step = max(1, int(round(float(max(wpx, h)) / 120.0)))
    pixels: set[tuple[int, int]] = set()
    for cx, cy in ((0, 0), (wpx - 1, 0), (0, h - 1), (wpx - 1, h - 1)):
        pixels.add((int(cx), int(cy)))
    for x in range(0, wpx, step):
        pixels.add((x, 0))
        pixels.add((x, h - 1))
    for y in range(0, h, step):
        pixels.add((0, y))
        pixels.add((wpx - 1, y))
    for cx, cy in pixels:
        cc = SkyCoord.from_pixel(float(cx), float(cy), wcs=w, origin=0)
        max_sep_deg = max(max_sep_deg, float(center.separation(cc).deg))
    # Margin: WCS at edges, rectangle vs cone on sphere, plate-solve error. Larger margin because
    # ``field_catalog_cone.csv`` built from a cropped MASTERSTAR must not underserve full-chip frames.
    radius_deg = max(max_sep_deg * 1.38, max_sep_deg + 45.0 / 3600.0)
    # If CD scale is wrong (too 'zoomed' on sky), spherical sampling collapses - blend in tangent-plane
    # half-diagonal from pixel scales so the cone stays physically plausible.
    try:
        scales = w.proj_plane_pixel_scales()
        sx_deg = abs(float(scales[0].to(u.deg).value))
        sy_deg = abs(float(scales[1].to(u.deg).value))
        r_cd = float(math.hypot(0.5 * (wpx - 1) * sx_deg, 0.5 * (h - 1) * sy_deg))
        radius_deg = max(radius_deg, r_cd * 1.22)
    except Exception:  # noqa: BLE001
        pass
    # F3 (RUN522-CRASH-01): no global 3.5 deg floor. The cone follows the solved
    # chip footprint (+ margins above). A fixed floor made the cone ~100x the chip
    # area on long-focus rigs; the brightest-first row cap then truncated the
    # field catalogue at G~15.8 in dense fields (522: 936 of 2930 G<=17.5 on chip).
    # Bad-WCS protection stays in the CD half-diagonal term (CATALOG-FOOTPRINT-2026-10:
    # there is no optics floor any more; the WCS is verified by astrosolve).
    return center, float(radius_deg)


def _effective_field_catalog_cone_radius_deg(w: WCS, h: int, wpx: int) -> tuple[SkyCoord, float]:
    """Catalogue cone of the solved chip footprint (CATALOG-FOOTPRINT-2026-10).

    The solved WCS is the only authority: header optics (XPIXSZ is already the binned pixel in
    ASCOM / N.I.N.A. / MaxIm headers and was multiplied by the binning again) and a UI FOV made
    the draft-532 cone 13.64 deg instead of 4.68 deg."""
    return _field_center_and_radius_from_wcs(w, h, wpx)


_FIELD_CATALOG_SPEC = "CATALOG-FOOTPRINT-2026-10"


def _invalidate_field_catalog_cone_cache_if_needed(field_catalog_csv: Path, *, effective_radius_deg: float) -> None:
    """Remove ``field_catalog_cone.csv`` + meta when it was built for another cone or by an older
    catalogue rule (a brightest-first capped catalogue must not be reused)."""
    p_csv = Path(field_catalog_csv)
    meta_p = _field_catalog_cone_meta_path(p_csv)
    if not p_csv.is_file() and not meta_p.is_file():
        return
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
    except Exception:  # noqa: BLE001
        meta = {}
    reasons: list[str] = []
    if str(meta.get("spec") or "") != _FIELD_CATALOG_SPEC:
        reasons.append("katalog vytvoreny starsim pravidlom (orez najjasnejsich)")
    r_stored = float(meta.get("cone_radius_deg") or 0.0)
    r_eff = float(effective_radius_deg)
    if abs(r_eff - r_stored) > 0.02 * max(r_eff, 1e-6) + 45.0 / 3600.0:
        reasons.append(f"kuzel r~{r_eff:.4f} deg != ulozene {r_stored:.4f} deg")
    if not reasons:
        return
    p_csv.unlink(missing_ok=True)
    meta_p.unlink(missing_ok=True)
    log_event("Katalog: field_catalog_cone cache vymazana - " + "; ".join(reasons) + " (nacitam nanovo).")


def _field_catalog_cone_meta_path(field_catalog_csv: Path) -> Path:
    return field_catalog_csv.parent / "field_catalog_cone_meta.json"


def _write_field_catalog_cone_meta(
    field_catalog_csv: Path,
    *,
    center: SkyCoord,
    radius_deg: float,
    naxis1: int,
    naxis2: int,
) -> None:
    """Persist cone parameters used to build ``field_catalog_cone.csv`` (cache invalidation)."""
    p = _field_catalog_cone_meta_path(field_catalog_csv)
    rec = {
        "spec": _FIELD_CATALOG_SPEC,
        "cone_radius_deg": float(radius_deg),
        "center_ra_icrs_deg": float(center.ra.deg),
        "center_dec_icrs_deg": float(center.dec.deg),
        "naxis1": int(naxis1),
        "naxis2": int(naxis2),
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1), encoding="utf-8")


_MASTERSTAR_ZONE_LOG_ONCE: set[str] = set()


def _masterstar_zone_linear_threshold(
    dao_detection_n_equiv: Any,
) -> float | None:
    """Linear-zone lower bound from DAO detection significance (peak_dao/bg_sigma >= N_equiv)."""
    try:
        t1 = float(dao_detection_n_equiv)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(t1) or t1 <= 0:
        return None
    return t1


def _detect_empirical_clip_level_adu(data: "np.ndarray") -> float | None:
    """Return the ADU level of frame truncation pile-up, or None if the frame is not clipped."""
    import numpy as np

    arr = np.asarray(data, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    vmax = float(np.max(finite))
    if not math.isfinite(vmax) or vmax <= 0:
        return None
    at_max = np.abs(finite - vmax) <= 0.5
    if int(np.count_nonzero(at_max)) >= 8:
        return vmax
    # SAT-PRECAL-2026-10: no guessing of 65535 / 32767 from pixels that merely lie near those
    # values (draft 532, float frames with a +32768 offset: star wings near 65535 ADU were
    # taken for a clip -> stars at 66 % of the real saturation flagged saturated).
    return None


def _resolve_peak_saturation_limit_adu(
    *,
    camera_sat_limit_adu: float | None,
    saturate_fraction: float,
    n_stack: int = 1,
    sky_median_adu: float | None = None,
    frame_max_adu: float | None = None,
    empirical_clip_adu: float | None = None,
    measured_sat_adu: float | None = None,
) -> float | None:
    """Peak-test saturation ceiling: the level SAT-DIAG measured on this draft's frames first,
    then the clip of this image, then a stated camera constant (only on raw-scale data)."""
    ns = max(1, int(n_stack))
    frac = float(saturate_fraction)
    if measured_sat_adu is not None:
        try:
            ms = float(measured_sat_adu)
            if math.isfinite(ms) and ms > 0:
                return ms * frac / float(ns)
        except (TypeError, ValueError):
            pass
    if empirical_clip_adu is not None:
        try:
            ec = float(empirical_clip_adu)
            if math.isfinite(ec) and ec > 0:
                return ec * frac / float(ns)
        except (TypeError, ValueError):
            pass
    if camera_sat_limit_adu is None:
        return None
    try:
        raw_cam = float(camera_sat_limit_adu)
        cam_lim = raw_cam * frac / float(ns)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cam_lim) or cam_lim <= 0:
        return None
    sky = float("nan")
    fmax = float("nan")
    try:
        if sky_median_adu is not None:
            sky = float(sky_median_adu)
        if frame_max_adu is not None:
            fmax = float(frame_max_adu)
    except (TypeError, ValueError):
        pass
    # Float MASTERSTAR stacks can overshoot the 16-bit clip by a few percent
    # (interpolation). Veto only when the image is clearly a different unit scale
    # (SAT-LIMIT-01: 515 stack max 68429 vs clip 65535 is not a unit change).
    if math.isfinite(fmax) and math.isfinite(raw_cam) and fmax > raw_cam * 1.20:
        return None
    if math.isfinite(sky) and sky > 0.20 * cam_lim:
        return None
    return cam_lim


def _masterstar_zone_log_once(key: str, msg: str) -> None:
    if key in _MASTERSTAR_ZONE_LOG_ONCE:
        return
    _MASTERSTAR_ZONE_LOG_ONCE.add(key)
    logging.warning(msg)


def _resolve_masterstar_bg_sigma_adu(*, sigma_px: Any) -> float | None:
    """Per-pixel background sigma of the MASTERSTAR (``det_meta['bg_sigma_adu']``), or None."""
    try:
        sp = float(sigma_px)
    except (TypeError, ValueError):
        return None
    return sp if math.isfinite(sp) and sp > 0 else None


def _annotate_masterstars_flux_zones(
    df: pd.DataFrame,
    *,
    equipment_saturate_adu: float | None,
    saturate_limit_adu_fallback: Any = None,
    n_stack: int | None = None,
    saturate_limit_fraction: float = SAT_LIMIT_NO_KNEE_FRAC,
    sigma_px: Any = None,
    sky_median_adu: Any = None,
    frame_max_adu: Any = None,
    empirical_clip_adu: Any = None,
    dao_detection_n_equiv: Any = None,
    measured_sat_adu: float | None = None,
) -> pd.DataFrame:
    """Tag MASTERSTAR catalog rows by peak significance (peak_dao/bg_sigma) and saturation.

    Linear/usability boundary T1 = ``dao_detection_n_equiv`` (same N-sigma as DAOFIND threshold).
    Noisy sub-bands step below in whole sigma (``_MASTERSTAR_ZONE_SIGMA_STEP``).

    ``sigma_px`` is the per-pixel sky sigma of the MASTERSTAR (``det_meta["bg_sigma_adu"]``).
    """
    import numpy as np

    out = df.copy()
    if "flux" not in out.columns:
        return out
    flux_s = pd.to_numeric(out["flux"], errors="coerce")


    ns = int(n_stack) if n_stack is not None else 1
    ns = max(1, ns)

    # RAW camera/container clip (do not pre-scale; _resolve_peak_saturation_limit_adu applies frac).
    sat_lim_raw = _finite_positive_adu(equipment_saturate_adu)
    peak_frac = float(saturate_limit_fraction)
    unresolved_clip = sat_lim_raw is None
    if unresolved_clip:
        sat_lim_raw = _finite_positive_adu(saturate_limit_adu_fallback)
        unresolved_clip = sat_lim_raw is None
    if unresolved_clip:
        sat_lim_raw = float(SAT_LIMIT_CONTAINER_CLIP_ADU)
        peak_frac = float(SAT_LIMIT_NO_KNEE_FRAC)
        _masterstar_zone_log_once(
            "sat_limit_unresolved",
            "[INV-SAT-LIMIT] header/equipment/sat_diag clip unresolved; "
            f"using clip={SAT_LIMIT_CONTAINER_CLIP_ADU:.0f} ADU (GAIN-DOMAIN-01 container), "
            f"peak-test frac={SAT_LIMIT_NO_KNEE_FRAC:.2f} "
            f"(peak-test={SAT_LIMIT_CONTAINER_CLIP_ADU * SAT_LIMIT_NO_KNEE_FRAC:.1f} ADU). "
            "Never silently admit.",
        )

    out["saturate_limit_adu"] = float(sat_lim_raw) if sat_lim_raw is not None else np.nan

    if "peak_max_adu" in out.columns:
        peak_col_used = "peak_max_adu"
        peak_s = pd.to_numeric(out["peak_max_adu"], errors="coerce")
    elif "peak_dao" in out.columns:
        peak_col_used = "peak_dao"
        peak_s = pd.to_numeric(out["peak_dao"], errors="coerce")
    else:
        peak_col_used = "flux"
        peak_s = flux_s

    peak_sat_lim = _resolve_peak_saturation_limit_adu(
        camera_sat_limit_adu=sat_lim_raw,
        saturate_fraction=peak_frac,
        n_stack=ns,
        sky_median_adu=sky_median_adu,
        frame_max_adu=frame_max_adu,
        empirical_clip_adu=empirical_clip_adu,
        measured_sat_adu=measured_sat_adu,
    )
    if peak_sat_lim is None and unresolved_clip:
        # Scale veto rejected even the conservative default. INV-SAT-LIMIT: still apply peak test.
        peak_sat_lim = float(sat_lim_raw) * float(peak_frac) / float(ns)
        _masterstar_zone_log_once(
            "sat_limit_unresolved_scale_veto",
            "[INV-SAT-LIMIT] peak-test scale veto ignored for unresolved clip; "
            f"applying peak-test={float(peak_sat_lim):.1f} ADU.",
        )
    if peak_sat_lim is None:
        peak_sat_lim = float(sat_lim_raw) if sat_lim_raw is not None else None
        _masterstar_zone_log_once(
            "sat_limit_85pct_nan",
            "[ZONE-SAT-01] saturate_limit_adu_85pct unresolved; peak test uses saturate_limit_adu "
            f"({peak_sat_lim}).",
        )
    out["saturate_limit_adu_85pct"] = peak_sat_lim if peak_sat_lim is not None else np.nan
    out["zone_peak_column"] = peak_col_used
    out["zone_sat_limit_used"] = float(peak_sat_lim) if peak_sat_lim is not None else np.nan

    out["zone"] = ""
    if peak_sat_lim is not None:
        out.loc[peak_s > float(peak_sat_lim), "zone"] = "saturated"

    bg_sigma = _resolve_masterstar_bg_sigma_adu(sigma_px=sigma_px)
    if bg_sigma is None:
        _masterstar_zone_log_once(
            "sigma_unresolvable",
            "[MASTERSTAR zone] bg_sigma_adu unresolvable - "
            "leaving zone empty (maps to neznama_zona downstream).",
        )
    else:
        t1 = _masterstar_zone_linear_threshold(dao_detection_n_equiv)
        if t1 is None:
            _masterstar_zone_log_once(
                "n_equiv_missing",
                "[MASTERSTAR zone] dao_detection_n_equiv unresolvable - "
                "leaving zone empty (maps to neznama_zona downstream).",
            )
        else:
            if "peak_dao" in out.columns:
                peak_dao_s = pd.to_numeric(out["peak_dao"], errors="coerce")
            else:
                peak_dao_s = pd.Series(np.nan, index=out.index, dtype=float)
            peak_sig = peak_dao_s / float(bg_sigma)
            unsat = out["zone"].eq("")
            sig_ok = unsat & peak_sig.notna()
            sig_miss = unsat & peak_sig.isna()
            out.loc[sig_miss, "zone"] = "unknown"
            out.loc[sig_ok & (peak_sig >= t1), "zone"] = "linear"
            out.loc[sig_ok & (peak_sig < t1), "zone"] = "noise"
            if sig_miss.any():
                _masterstar_zone_log_once(
                    "peak_dao_missing",
                    "[MASTERSTAR zone] peak_dao missing/NaN rows "
                    "marked zone=unknown (not flux fallback).",
                )

    if peak_sat_lim is not None:
        out["is_saturated"] = (peak_s > float(peak_sat_lim)).fillna(False)
    else:
        # Camera clip was supplied but rejected as wrong-scale (precalibrated stack).
        out["is_saturated"] = False

    out["is_noisy"] = out["zone"].eq("noise")
    out["is_usable"] = out["zone"].eq("linear") & flux_s.notna()
    return out


def _dao_auto_binning_factor(h: int, w: int) -> int:
    """2x2 mean binning for DAO on large chips (~4x fewer pixels); skipped below ~5 MP."""
    mp = float(int(h) * int(w)) / 1_000_000.0
    if mp < 5.0:
        return 1
    return 2


def _pixel_noise_sigma_pp_adu(arr: Any) -> float:
    """Gradient-immune pixel noise from adjacent differences (MAD estimator).

    For smooth large-scale structure, ``I[i+1]-I[i]`` is nearly constant so MAD tracks
    pixel-to-pixel noise, not sky gradient. Validated on BO CVn MASTERSTAR (~46 ADU stable
    across draft 435/450 while ``plain_mean_med_std`` moved 26%; see SIGMA-ESTIMATOR-VERIFY).
    """
    import numpy as np

    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2 or a.size < 64:
        return float("nan")
    finite = np.isfinite(a)
    chunks: list[np.ndarray] = []
    dh = a[:, 1:] - a[:, :-1]
    mh = finite[:, 1:] & finite[:, :-1]
    if int(np.count_nonzero(mh)) >= 32:
        chunks.append(dh[mh].ravel())
    dv = a[1:, :] - a[:-1, :]
    mv = finite[1:, :] & finite[:-1, :]
    if int(np.count_nonzero(mv)) >= 32:
        chunks.append(dv[mv].ravel())
    if not chunks:
        return float("nan")
    diffs = np.concatenate(chunks)
    if diffs.size < 64:
        return float("nan")
    med = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - med)))
    if not math.isfinite(mad) or mad <= 0:
        return float("nan")
    return float(mad / 0.674489750196082 / math.sqrt(2.0))


def _dao_noise_sigma_adu(
    arr: Any,
    *,
    bfac: int,
    fallback_std: float,
    data_dao: Any | None = None,
) -> float:
    """Legacy per-pixel noise scale (diagnostic only; not used for DAO threshold under option B)."""

    sigma_pp = _pixel_noise_sigma_pp_adu(arr)
    if math.isfinite(float(sigma_pp)) and float(sigma_pp) > 0:
        return float(sigma_pp) / math.sqrt(float(max(1, bfac)))
    if int(bfac) > 1 and data_dao is not None:
        _, _, std_dao = plain_mean_med_std(data_dao, sigma=3.0, maxiters=3)
        logging.warning("[DAO] sigma_pp unavailable; falling back to plain_mean_med_std on binned frame")
        return float(std_dao)
    logging.warning("[DAO] sigma_pp unavailable; falling back to global plain_mean_med_std")
    return float(fallback_std)


def _dao_convolved_background_rms_adu(
    data_dao: Any,
    *,
    fwhm_px: float,
    sigma_radius: float = 1.5,
) -> tuple[float, float]:
    """Noise of the DAO convolved detection image (threshold basis, DAO-NOISE-2026-10).

    Builds the same kernel DAOStarFinder uses, convolves ``data_dao`` and returns
    (sigma_convolved, kernel.rel_err) with sigma = 1.4826 * MAD of the convolved
    image. Stars are a minority of pixels, so the MAD measures the sky noise
    (including real confusion noise) without any clipping. The former sample std
    included the stars and was 40x too large on 523 (63 vs 1.5 ADU).
    Caller uses ``scale_threshold=False`` and ``threshold = N * sigma_convolved``.
    """
    import numpy as np
    from photutils.detection.core import _StarFinderKernel
    from scipy.ndimage import convolve

    arr = np.asarray(data_dao, dtype=np.float32)
    if arr.ndim != 2 or arr.size < 64:
        return float("nan"), float("nan")
    fwhm = max(1.2, float(fwhm_px))
    kernel = _StarFinderKernel(fwhm=fwhm, sigma_radius=float(sigma_radius))
    conv = convolve(arr, kernel.data, mode="nearest")
    v = conv[np.isfinite(conv)]
    if v.size < 64:
        return float("nan"), float(kernel.rel_err)
    med = float(np.median(v))
    rms = 1.4826 * float(np.median(np.abs(v - med)))
    return float(rms), float(kernel.rel_err)


#: MASTERSTAR detections whose false fraction (negated-image test) exceeds this get a warning
DAO_FALSE_FRACTION_WARN = 0.05


#: negated-image detections closer than this x FWHM to a real detection are side lobes of the
#: zero-sum DAO kernel around a (negated) star, not noise; they and that area are excluded
DAO_FALSE_EXCLUDE_FWHM = 2.5


def dao_false_detection_fraction(
    data_dao: Any,
    *,
    fwhm_px: float,
    threshold_adu: float,
    star_xy: Any,
) -> dict[str, Any]:
    """False-detection fraction of a DAO detection run (DAO-FALSE-2026-10).

    The same DAOStarFinder (FWHM, threshold, no roundness cut) runs on the negated,
    sky-subtracted image: for symmetric noise every detection there is false (inverted-map
    test of survey catalogs; cf. False Discovery Rate, Hopkins et al. 2002). The zero-sum DAO
    kernel turns every negated star into a ring of positive side lobes, so negated-image
    detections within ``DAO_FALSE_EXCLUDE_FWHM`` x FWHM of a real detection (``star_xy``, in
    ``data_dao`` pixels) are dropped and the count is scaled up by the excluded area.
    """
    import numpy as np
    from photutils.detection import DAOStarFinder
    from scipy.spatial import cKDTree

    from utils import DAO_STAR_FINDER_NO_ROUNDNESS_FILTER

    arr = np.asarray(data_dao, dtype=np.float32)
    neg = -np.nan_to_num(arr - float(np.nanmedian(arr)), nan=0.0)
    tbl = DAOStarFinder(
        fwhm=float(fwhm_px), threshold=float(threshold_adu), scale_threshold=False, n_brightest=None,
        **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
    )(neg)
    xy = np.asarray(star_xy, dtype=np.float64).reshape(-1, 2)
    n_pos = int(len(xy))
    r_ex = DAO_FALSE_EXCLUDE_FWHM * float(fwhm_px)
    n_neg_all = 0 if tbl is None else int(len(tbl))
    n_neg = n_neg_all
    free = 1.0
    if n_pos and n_neg_all:
        tree = cKDTree(xy)
        d, _ = tree.query(np.c_[np.asarray(tbl["x_centroid"]), np.asarray(tbl["y_centroid"])])
        n_neg = int(np.count_nonzero(d > r_ex))
    if n_pos:
        # free sky fraction: a coarse grid (step ~ FWHM) tested against the exclusion disks
        h, w = arr.shape
        step = max(1.0, float(fwhm_px))
        gy, gx = np.mgrid[0:h:step, 0:w:step]
        dg, _ = cKDTree(xy).query(np.c_[gx.ravel(), gy.ravel()])
        free = float(np.count_nonzero(dg > r_ex)) / float(dg.size)
    n_false = float(n_neg) / free if free > 0 else float("nan")
    frac = n_false / float(n_pos) if n_pos > 0 else float("nan")
    return {
        "n_detected": n_pos,
        "n_negative_all": n_neg_all,
        "n_negative_far": n_neg,
        "free_sky_fraction": round(free, 4),
        "n_false_estimate": round(n_false, 1) if n_pos else 0.0,
        "false_fraction": frac,
        "warn_above": DAO_FALSE_FRACTION_WARN,
        "method": "DAOStarFinder on the negated sky-subtracted detection image, same FWHM and threshold; "
                  f"detections within {DAO_FALSE_EXCLUDE_FWHM} FWHM of a real detection dropped, count scaled by the free sky",
    }


_BATCH_E_N_EQUIV_LOGGED = False


def _dao_detection_threshold_adu(
    rms_conv: float,
    *,
    cfg: Any | None,
    dao_threshold_sigma: float,
) -> tuple[float, float]:
    """T4-1 Option B: threshold = N_equiv * rms_conv (measured N from Part 2b)."""
    global _BATCH_E_N_EQUIV_LOGGED  # noqa: PLW0603
    rms = float(rms_conv)
    if not math.isfinite(rms) or rms <= 0:
        return 1e-6, float("nan")
    n_eff = float(getattr(cfg, "detection_threshold_sigma", dao_threshold_sigma)) if cfg is not None else float(
        dao_threshold_sigma
    )
    if not math.isfinite(n_eff) or n_eff <= 0:
        n_eff = float(dao_threshold_sigma)
    if cfg is not None and not _BATCH_E_N_EQUIV_LOGGED:
        LOGGER.info("[BATCH-E E.4] N_equiv=%.2f applied for DAO detection threshold", n_eff)
        _BATCH_E_N_EQUIV_LOGGED = True
    return max(n_eff * rms, 1e-6), n_eff


def _apply_dao_centroid_wcs_guard(
    x: "np.ndarray",
    y: "np.ndarray",
    *,
    matched: "np.ndarray",
    safe: "np.ndarray",
    master_df: pd.DataFrame,
    fwhm_px: float,
    max_shift_fwhm: float,
) -> tuple["np.ndarray", "np.ndarray", int]:
    """Replace DAO centroids with MASTERSTAR reference pixels when shift exceeds guard."""
    import numpy as np

    xo = np.asarray(x, dtype=np.float64).copy()
    yo = np.asarray(y, dtype=np.float64).copy()
    if master_df is None or master_df.empty or "x" not in master_df.columns or "y" not in master_df.columns:
        return xo, yo, 0
    mx = pd.to_numeric(master_df["x"], errors="coerce").to_numpy(dtype=np.float64)
    my = pd.to_numeric(master_df["y"], errors="coerce").to_numpy(dtype=np.float64)
    m = np.asarray(matched, dtype=bool)
    if not m.any():
        return xo, yo, 0
    s = np.clip(np.asarray(safe, dtype=np.int64), 0, max(len(mx) - 1, 0))
    x_ref = mx[s]
    y_ref = my[s]
    ok = m & np.isfinite(x_ref) & np.isfinite(y_ref) & np.isfinite(xo) & np.isfinite(yo)
    if not ok.any():
        return xo, yo, 0
    max_px = float(min(1.5, float(max(0.1, max_shift_fwhm)) * float(max(1.2, fwhm_px))))
    # IDENT-JUMP-01 (unaligned): never keep a DAO centroid that landed on a
    # neighbour. Cap the allowed shift at 1.5 px absolute vs WCS/master ref.
    shift = np.hypot(xo - x_ref, yo - y_ref)
    use_wcs = ok & (shift > max_px)
    xo[use_wcs] = x_ref[use_wcs]
    yo[use_wcs] = y_ref[use_wcs]
    return xo, yo, int(np.count_nonzero(use_wcs))


def _lock_matched_centroids_to_master_grid(
    x: "np.ndarray",
    y: "np.ndarray",
    *,
    matched: "np.ndarray",
    safe: "np.ndarray",
    master_df: pd.DataFrame,
) -> tuple["np.ndarray", "np.ndarray", int]:
    """Lock matched catalog stars to the MASTERSTAR sub-pixel (x, y) on aligned frames.

    POS-SUBPIX-2026-10: the frame is registered to MASTERSTAR, so the photometric
    centre is the reference coordinate itself. The former bounded argmax "refine"
    snapped centres to integer pixels (78 % of centres on draft 523) and moved the
    aperture by up to 1 px on noise peaks; it is removed. IDENT-JUMP-01 holds
    trivially (no search, no neighbour can be picked).

    ``arr``, ``fwhm_px``, ``search_fwhm`` and ``refine_bound_px`` are kept for
    call-site compatibility and are not used.
    """
    import numpy as np

    xo = np.asarray(x, dtype=np.float64).copy()
    yo = np.asarray(y, dtype=np.float64).copy()
    if master_df is None or master_df.empty or "x" not in master_df.columns or "y" not in master_df.columns:
        return xo, yo, 0
    mx = pd.to_numeric(master_df["x"], errors="coerce").to_numpy(dtype=np.float64)
    my = pd.to_numeric(master_df["y"], errors="coerce").to_numpy(dtype=np.float64)
    m = np.asarray(matched, dtype=bool)
    if not m.any() or len(mx) == 0:
        return xo, yo, 0
    s = np.clip(np.asarray(safe, dtype=np.int64), 0, len(mx) - 1)
    idx = np.nonzero(m)[0]
    xr = mx[s[idx]]
    yr = my[s[idx]]
    ok = np.isfinite(xr) & np.isfinite(yr)
    xo[idx[ok]] = xr[ok]
    yo[idx[ok]] = yr[ok]
    return xo, yo, int(ok.sum())


def _proc_deduplicate_matched_catalog_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per non-empty ``catalog_id`` (highest peak / flux wins).

    Multiple DAO detections can match the same Gaia source in one frame; duplicates
    inflate Phase-1 comp RMS and can select faint spurious matches.
    """
    if df is None or len(df) == 0 or "catalog_id" not in df.columns:
        return df
    out = df.copy()
    cid = out["catalog_id"].fillna("").astype(str).str.strip()
    matched = cid.ne("") & ~cid.str.lower().isin({"nan", "none"})
    if not bool(matched.any()):
        return out
    # Always a Series: out.get("flux") is None when the column is absent (empty-DAO
    # frames after forced-phot inject - draft 515 crash), and pd.to_numeric(None)
    # collapses to numpy.float64 without .fillna.
    if "peak_max_adu" in out.columns:
        score = pd.to_numeric(out["peak_max_adu"], errors="coerce")
    elif "dao_flux" in out.columns:
        score = pd.to_numeric(out["dao_flux"], errors="coerce")
    elif "flux" in out.columns:
        score = pd.to_numeric(out["flux"], errors="coerce")
    else:
        score = pd.Series(float("nan"), index=out.index, dtype="float64")
    score = pd.Series(pd.to_numeric(score, errors="coerce"), index=out.index, dtype="float64")
    out["_dedupe_score"] = score.fillna(-1.0)
    keep_idx = (
        out.loc[matched]
        .sort_values("_dedupe_score", ascending=False, kind="mergesort")
        .drop_duplicates(subset=["catalog_id"], keep="first")
        .index
    )
    keep_mask = out.index.isin(keep_idx) | ~matched
    out = out.loc[keep_mask].drop(columns=["_dedupe_score"]).reset_index(drop=True)
    n_dropped = int(matched.sum()) - int(len(keep_idx))
    if n_dropped > 0:
        LOGGER.debug("[PROC] deduplicated %d duplicate catalog_id rows in per-frame CSV", n_dropped)
    return out


def _mean_bin2d_for_dao(data0: "np.ndarray", factor: int) -> tuple["np.ndarray", int]:
    import numpy as np

    f = int(factor)
    a = np.asarray(data0, dtype=np.float32)
    if f <= 1:
        return a, 1
    h, w = a.shape
    h2, w2 = (h // f) * f, (w // f) * f
    if h2 < f or w2 < f:
        return a, 1
    b = a[:h2, :w2].reshape(h2 // f, f, w2 // f, f).mean(axis=(1, 3)).astype(np.float32)
    return b, f


def _gaia_chip_xy_from_catalog(
    cat_df: pd.DataFrame | None,
    wcs_obj: Any,
    *,
    wpx: int,
    h: int,
) -> pd.DataFrame:
    """On-chip Gaia rows with x_gaia/y_gaia for star-mask and born-owned pass2."""
    import numpy as np

    if cat_df is None or cat_df.empty or "ra_deg" not in cat_df.columns or "dec_deg" not in cat_df.columns:
        return pd.DataFrame()
    ra = pd.to_numeric(cat_df["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
    de = pd.to_numeric(cat_df["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(ra) & np.isfinite(de)
    if not bool(ok.any()):
        return pd.DataFrame()
    gx, gy = wcs_obj.world_to_pixel_values(ra[ok], de[ok])
    sub = cat_df.loc[ok].copy().reset_index(drop=True)
    sub["x_gaia"] = gx
    sub["y_gaia"] = gy
    sub["g_mag"] = pd.to_numeric(sub.get("mag"), errors="coerce")
    inb = (
        (sub["x_gaia"] >= 0)
        & (sub["x_gaia"] < float(wpx))
        & (sub["y_gaia"] >= 0)
        & (sub["y_gaia"] < float(h))
    )
    return sub.loc[inb].reset_index(drop=True)


def _proc_rename_det_names_to_catalog_id(df: pd.DataFrame) -> pd.DataFrame:
    """Matched rows still named DET_* get ``name`` = ``catalog_id`` (stale master names)."""
    if df is None or df.empty or "catalog_id" not in df.columns or "name" not in df.columns:
        return df
    _matched_mask = (
        df["catalog_id"].notna()
        & (df["catalog_id"].astype(str).str.strip() != "")
        & (df["name"].astype(str).str.startswith("DET_"))
    )
    if not _matched_mask.any():
        return df
    out = df.copy()
    out.loc[_matched_mask, "name"] = out.loc[_matched_mask, "catalog_id"].astype(str)
    return out


def _proc_drop_unmatched_dao_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unmatched DAO_ONLY detections (no catalog_id) BEFORE expensive per-star work.

    GAIA_MATCHED rows carry a catalog_id and are kept; DAO_ONLY rows
    (empty catalog_id) are discarded by the final keep-matched filter anyway. Keying on
    catalog_id (not source_type) is correct in both export paths, including the in-process path
    where source_type=GAIA_MATCHED is assigned only later.
    """
    if df is None or len(df) == 0 or "catalog_id" not in df.columns:
        return df
    cid = df["catalog_id"].fillna("").astype(str).str.strip()
    keep = cid.ne("") & ~cid.str.lower().isin({"nan", "none"})
    return df.loc[keep].reset_index(drop=True)


def _proc_catalog_keep_matched_rows_only(df: pd.DataFrame) -> pd.DataFrame:
    """TODO-13: keep catalog rows with valid ``catalog_id`` and ``source_type``."""
    if df is None or len(df) == 0 or "catalog_id" not in df.columns:
        return df
    cid = df["catalog_id"].fillna("").astype(str).str.strip()
    _valid_cid = cid.ne("") & ~cid.str.lower().isin({"nan", "none"})
    if "source_type" in df.columns:
        _valid_source = (
            df["source_type"].fillna("").astype(str).str.strip().str.upper().eq("GAIA_MATCHED")
        )
        keep = _valid_cid & _valid_source
    else:
        keep = _valid_cid
    return df.loc[keep].reset_index(drop=True)


def _prefilter_dao_table_brightest(tbl: Any, keep_top: int) -> Any:
    """Cap DAO rows before sorting when the finder returns an enormous table (slow + useless)."""
    import numpy as np

    if tbl is None or len(tbl) <= int(keep_top):
        return tbl
    flux_np = np.asarray(tbl["flux"], dtype=np.float64)
    k = int(keep_top)
    take = np.argpartition(flux_np, -k)[-k:]
    return tbl[take]


def _dao_spatial_flux_cap_row_indices(
    tbl: Any,
    *,
    max_n: int,
    width_px: float,
    height_px: float,
) -> "np.ndarray":
    """Row indices into ``tbl`` for up to ``max_n`` sources, spread across the detector.

    Sorting globally by flux and truncating biases toward the frame centre when vignetting or
    gradients make edge stars fainter in ADU - QA then falsely looks like a ``catalog disc``. This
    fills a coarse grid brightest-first per cell, then tops up by global flux (same cap).
    """
    import numpy as np

    n = len(tbl)
    m = int(max_n)
    if n <= m:
        return np.arange(n, dtype=np.int64)
    x = np.asarray(tbl["x_centroid"], dtype=np.float64)
    y = np.asarray(tbl["y_centroid"], dtype=np.float64)
    flux = np.asarray(tbl["flux"], dtype=np.float64)
    w = max(float(width_px), 1.0)
    h = max(float(height_px), 1.0)
    # ~25 sources per cell on average, bounded grid
    ncell_target = max(48, min(512, max(1, m // 25)))
    aspect = w / h
    ny = max(4, int(round((ncell_target / aspect) ** 0.5)))
    nx = max(4, int(round(ncell_target / float(ny))))
    ncells = nx * ny
    per_cell = max(1, m // ncells)
    ix = np.clip((x / w * nx).astype(np.int64), 0, nx - 1)
    iy = np.clip((y / h * ny).astype(np.int64), 0, ny - 1)
    cell_id = ix + iy * nx
    order = np.argsort(-flux)
    taken = np.zeros(n, dtype=bool)
    picked: list[int] = []
    counts = np.zeros(ncells, dtype=np.int32)
    for idx in order:
        c = int(cell_id[idx])
        if counts[c] < per_cell and len(picked) < m:
            picked.append(int(idx))
            counts[c] += 1
            taken[idx] = True
    for idx in order:
        if len(picked) >= m:
            break
        if not taken[idx]:
            picked.append(int(idx))
            taken[idx] = True
    return np.asarray(sorted(picked), dtype=np.int64)


def detect_stars_match_master_reference(
    data: "np.ndarray",
    hdr: fits.Header,
    master_df: pd.DataFrame,
    *,
    max_catalog_rows: int = 12000,
    match_sep_arcsec: float = 8.0,
    dao_threshold_sigma: float = 3.5,
    dao_fwhm_px: float | None = None,
    fallback_saturate_adu: float | None = None,
    equipment_saturate_adu: float | None = None,
    frame_name: str = "",
    sat_diag_ctx: Any | None = None,
    raw_data: "np.ndarray | None" = None,
    raw_hdr: fits.Header | None = None,
    ref_ra_deg: float | None = None,
    ref_dec_deg: float | None = None,
    drift_ref_catalog_id: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """DAO on this frame + nearest-neighbor match to ``masterstars.csv`` (no Vizier / cone).

    Catalog IDs and static columns come from the master row; ``x``, ``y``, ``flux``, ``peak_*`` and
    saturation flags are per-frame. Intended for ``detrended_aligned`` data whose WCS matches
    ``MASTERSTAR.fits`` astrometry.
    """
    from pipeline import _dao_targeted_pass2_unmatched_gaia  # noqa: PLC0415

    import numpy as np

    m = master_df
    # Primary mode: sky match using per-frame WCS (DAO x/y -> RA/Dec), then NN match against MASTERSTAR ra/dec.
    # Fallback: if frame has no usable celestial WCS, do a pixel NN match (15 px) against MASTERSTAR x/y.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs_obj = WCS(hdr)

    # MASTERSTAR sky catalog
    if "ra_deg" not in m.columns or "dec_deg" not in m.columns:
        raise ValueError("masterstars table must contain ra_deg, dec_deg")
    _ra = pd.to_numeric(m["ra_deg"], errors="coerce")
    _de = pd.to_numeric(m["dec_deg"], errors="coerce")
    okm_sky = _ra.notna() & _de.notna()
    if not bool(okm_sky.any()):
        raise ValueError("No valid ra_deg/dec_deg rows in masterstars")
    m_valid = m.loc[okm_sky].reset_index(drop=True)
    ra_m = pd.to_numeric(m_valid["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
    de_m = pd.to_numeric(m_valid["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
    master_coords = SkyCoord(ra=ra_m * u.deg, dec=de_m * u.deg, frame="icrs")

    match_mode = "sky"
    plate_scale_arcsec_per_px = None

    arr = np.asarray(data, dtype=np.float32)
    mean, med, std = plain_mean_med_std(arr, sigma=3.0, maxiters=3)
    data0 = np.nan_to_num((arr - med).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    bfac = 1
    _cfg_dao = AppConfig()
    _fb_m = float(dao_fwhm_px) if dao_fwhm_px is not None else float(_cfg_dao.sips_dao_fwhm_px)
    _base_fw_m = dao_detection_fwhm_pixels(hdr, configured_fallback=_fb_m)
    try:
        from photutils.detection import DAOStarFinder  # type: ignore

        thr_s = float(dao_threshold_sigma)
        thr_s = max(0.5, min(20.0, thr_s))
        dao_scale = _dao_auto_binning_factor(*data0.shape)
        data_dao, bfac = _mean_bin2d_for_dao(data0, dao_scale)
        fwhm_eff = max(1.2, _base_fw_m / float(bfac))
        rms_conv, _dao_rel_err = _dao_convolved_background_rms_adu(data_dao, fwhm_px=fwhm_eff)
        sigma_pp_diag = _dao_noise_sigma_adu(arr, bfac=bfac, fallback_std=float(std), data_dao=data_dao)
        std_dao = rms_conv
        if std_dao is None or (not np.isfinite(float(std_dao))) or float(std_dao) <= 0:
            try:
                std_dao = float(np.nanstd(arr))
            except Exception:  # noqa: BLE001
                std_dao = float("nan")
        if std_dao is None or (not np.isfinite(float(std_dao))) or float(std_dao) <= 0:
            _nm = str(frame_name or hdr.get("FILENAME") or "").strip() or "frame"
            try:
                finite = np.isfinite(arr)
                n_finite = int(np.count_nonzero(finite))
                if n_finite > 0:
                    vals = arr[finite]
                    n_unique = int(len(np.unique(vals)))
                    mn = float(np.nanmin(vals))
                    mx = float(np.nanmax(vals))
                else:
                    n_unique, mn, mx = 0, float("nan"), float("nan")
                print(
                    f"DEBUG std=0: {_nm} n_unique={n_unique} n_finite={n_finite} "
                    f"min={mn:.1f} max={mx:.1f}"
                )
            except Exception:  # noqa: BLE001
                pass
            # If the frame isn't constant (min != max), try std from non-zero finite pixels.
            try:
                nonzero_mask = (arr != 0) & np.isfinite(arr)
                if int(np.count_nonzero(nonzero_mask)) > 100:
                    std_dao = float(np.std(arr[nonzero_mask]))
            except Exception:  # noqa: BLE001
                pass
        if std_dao is None or (not np.isfinite(float(std_dao))) or float(std_dao) <= 0:
            _nm = str(frame_name or hdr.get("FILENAME") or "").strip() or "frame"
            print(f"WARNING: {_nm} std=0 aj po fallback, preskakujem")
            empty_meta: dict[str, Any] = {"status": "std_zero"}
            return pd.DataFrame(), empty_meta
        _thr, _n_equiv_used = _dao_detection_threshold_adu(
            float(std_dao), cfg=_cfg_dao, dao_threshold_sigma=float(thr_s),
        )
        try:
            _nm = str(frame_name or hdr.get("FILENAME") or "").strip() or "frame"
            logging.getLogger(__name__).debug(
                f"DAO input: {_nm} mean={float(np.nanmean(arr)):.1f} std={float(np.nanstd(arr)):.1f} "
                f"threshold={float(_thr):.1f} rms_conv={float(rms_conv):.2f} n_equiv={float(_n_equiv_used):.2f} "
                f"sigma_pp_diag={float(sigma_pp_diag):.2f} "
                f"fwhm={float(fwhm_eff):.2f}"
            )
        except Exception:  # noqa: BLE001
            pass
        finder = DAOStarFinder(
            fwhm=float(fwhm_eff),
            threshold=float(_thr),
            scale_threshold=False,
            n_brightest=None,
            **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
        )
        tbl = finder(data_dao)
        if tbl is not None and len(tbl) > 0:
            tbl = _prefilter_dao_table_brightest(tbl, max(int(max_catalog_rows) * 12, 36_000))
            from astropy.table import Table as _AstropyTable

            tbl = _AstropyTable(tbl, copy=True)
            tbl["vy_dao_pass"] = np.ones(len(tbl), dtype=np.int16)
        n_pass1_dao_m = int(len(tbl)) if tbl is not None else 0
        _h_m, _wpx_m = int(data0.shape[0]), int(data0.shape[1])
        try:
            _sigma_p2_m = float(_cfg_dao.masterstar_dao_pass2_sigma)
        except (TypeError, ValueError):
            _sigma_p2_m = 1.9
        try:
            _ctol_p2_m = float(_cfg_dao.masterstar_dao_pass2_center_tol_px)
        except (TypeError, ValueError):
            _ctol_p2_m = 5.0
        tbl, _n_unmatched_master, _n_pass2_raw_m = _dao_targeted_pass2_unmatched_gaia(
            data0,
            tbl,
            cat_df=m_valid,
            wcs_obj=wcs_obj,
            bfac=int(bfac),
            fwhm_px=float(max(1.2, _base_fw_m)),
            pass2_sigma=float(_sigma_p2_m),
            pass2_center_tol_px=float(_ctol_p2_m),
            match_sep_arcsec=float(match_sep_arcsec),
            wpx=int(_wpx_m),
            h=int(_h_m),
        )
        n_merged_dao_m = int(len(tbl)) if tbl is not None else 0
        LOGGER.info(
            "[DAO pass 1 master] %d detections, %d MASTERSTAR unmatched",
            int(n_pass1_dao_m),
            int(_n_unmatched_master),
        )
        LOGGER.info(
            "[DAO pass 2 master] %d additional detections from %d targeted positions",
            int(_n_pass2_raw_m),
            int(_n_unmatched_master),
        )
        LOGGER.info("[DAO total master] %d detections after merge", int(n_merged_dao_m))
    except Exception:  # noqa: BLE001
        tbl = None
        bfac = 1

    _fb_sat = fallback_saturate_adu

    if sat_diag_ctx is not None and getattr(sat_diag_ctx, "sat_adu", None) is not None:
        sat_limit = float(sat_diag_ctx.sat_adu)
        sat_limit_src = str(getattr(sat_diag_ctx, "sat_source", "sat_diag"))
    else:
        sat_limit, sat_limit_src = _effective_saturation_limit(
            hdr, fallback_adu=_fb_sat, equipment_saturate_adu=equipment_saturate_adu
        )
    sat_frac = float(LINEARITY_FRAC)

    foot_meta = {
        "catalog_footprint": {"method": "master_reference_only"},
        "saturation": {
            "effective_limit_adu": float(sat_limit) if sat_limit is not None else None,
            "limit_source": sat_limit_src,
            "plateau_half_inner_px": 1,
            "plateau_rel": 0.996,
            "plateau_min_pixels": 5,
        },
    }

    empty_meta = {
        "n_detected": 0,
        "n_detected_dao": 0,
        "n_matched": 0,
        "catalog_rows": int(len(master_df)),
        "catalog_match_mode": export_catalog_match_mode_from_internal(match_mode),
        "n_likely_saturated": 0,
        "n_saturated_from_peak": 0,
        "n_saturated_plateau": 0,
        "saturate_limit_adu": float(sat_limit) if sat_limit is not None else None,
        "saturate_limit_source": sat_limit_src,
        "n_vsx_in_field": 0,
        "n_gaia_variable_in_field": 0,
        "field_catalog_cone_csv": None,
        "dao_threshold_sigma": float(dao_threshold_sigma),
        "dao_fwhm_px": float(max(1.2, _base_fw_m)),
        "dao_detect_binning": 1,
        "match_sep_arcsec_requested": float(match_sep_arcsec),
        "match_sep_arcsec_effective": float(match_sep_arcsec),
        "plate_scale_arcsec_per_px": (
            float(plate_scale_arcsec_per_px) if plate_scale_arcsec_per_px is not None else None
        ),
        **foot_meta,
    }


    if tbl is None or len(tbl) == 0:
        df_out = pd.DataFrame()
        df_out = _proc_rename_det_names_to_catalog_id(df_out)
        empty_meta["n_detected"] = int(len(df_out))
        empty_meta["n_matched"] = int(
            df_out["catalog_id"].fillna("").astype(str).str.strip().ne("").sum()
        ) if len(df_out) and "catalog_id" in df_out.columns else 0
        return df_out, empty_meta

    _fwhm_used_m = float(max(1.2, _base_fw_m / float(bfac)))
    _d_h, _d_w = int(data_dao.shape[0]), int(data_dao.shape[1])
    _keep = _dao_spatial_flux_cap_row_indices(
        tbl, max_n=int(max_catalog_rows), width_px=float(_d_w), height_px=float(_d_h)
    )
    tbl = tbl[_keep]
    tbl.sort("flux")
    tbl = tbl[::-1]
    n = len(tbl)
    xb = np.asarray(tbl["x_centroid"], dtype=np.float64)
    yb = np.asarray(tbl["y_centroid"], dtype=np.float64)
    x, y = _dao_xy_binned_to_full(xb, yb, bfac)
    flux = np.asarray(tbl["flux"], dtype=np.float64)
    peak_dao = np.asarray(tbl["peak"], dtype=np.float64) if "peak" in tbl.colnames else np.full(n, np.nan)

    match_thr = float(match_sep_arcsec)

    def _pixel_nn_match(*, dist_thr_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Pixel-space NN match against MASTERSTAR x/y.

        Returns: (icomp, sep_arr, oix)
        - icomp: indices into the *filtered* master_xy array (as from KDTree query)
        - sep_arr: distance in px (NaN when unmatched)
        - oix: mapping from filtered master_xy indices back to m_valid row indices
        """
        ic: np.ndarray
        sep: np.ndarray
        ox: np.ndarray | None
        if "x" not in m_valid.columns or "y" not in m_valid.columns:
            ic = np.zeros(n, dtype=np.int64)
            sep = np.full(n, np.nan, dtype=np.float64)
            ox = None
            return ic, sep, ox
        try:
            from scipy.spatial import cKDTree  # type: ignore

            mx = pd.to_numeric(m_valid["x"], errors="coerce").to_numpy(dtype=np.float64)
            my = pd.to_numeric(m_valid["y"], errors="coerce").to_numpy(dtype=np.float64)
            okxy = np.isfinite(mx) & np.isfinite(my)
            mx2 = mx[okxy]
            my2 = my[okxy]
            ox = np.nonzero(okxy)[0].astype(np.int64)
            if mx2.size == 0:
                ic = np.zeros(n, dtype=np.int64)
                sep = np.full(n, np.nan, dtype=np.float64)
                return ic, sep, ox
            master_xy = np.column_stack([mx2, my2])
            det_xy = np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)])
            tree = cKDTree(master_xy)
            dist_px, idx = tree.query(det_xy, distance_upper_bound=float(dist_thr_px))
            ic = np.asarray(idx, dtype=np.int64)
            sep = np.asarray(dist_px, dtype=np.float64)  # px
            sep[~np.isfinite(sep)] = np.nan
            return ic, sep, ox
        except Exception:  # noqa: BLE001
            ic = np.zeros(n, dtype=np.int64)
            sep = np.full(n, np.nan, dtype=np.float64)
            ox = None
            return ic, sep, ox

    icomp: np.ndarray | None = None
    sep_arcsec_arr: np.ndarray
    oix: np.ndarray | None = None
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    _frame_on_ref_grid = _fits_header_vy_algn_aligned(hdr)

    # Robust strategy:
    # - If celestial WCS exists, try sky-match first (arcsec threshold).
    # - If sky-match looks suspiciously bad (e.g. WCS offset), fall back to pixel match
    #   only when the frame is on the reference alignment grid (VY_ALGN=True).
    # Matching unaligned DAO xy to the reference pixel grid is invalid by construction.
    if getattr(wcs_obj, "has_celestial", False):
        ra_deg, dec_deg = _all_pix2world_icrs_deg(wcs_obj, x, y)
        det_coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        idx_sky, sep2d, _ = det_coords.match_to_catalog_sky(master_coords)
        sep_sky = np.asarray(sep2d.to_value(u.arcsec), dtype=np.float64)
        ic_sky = np.asarray(idx_sky, dtype=np.int64)

        # Heuristic: if typical nearest-neighbor sep is far above the threshold,
        # the WCS is likely offset (e.g. wrong reference grid / flip / stale solve).
        med_sep = float(np.nanmedian(sep_sky)) if np.isfinite(np.nanmedian(sep_sky)) else float("inf")
        if (not np.isfinite(med_sep)) or (med_sep > max(30.0, match_thr * 3.0)):
            if not _frame_on_ref_grid:
                icomp = ic_sky
                sep_arcsec_arr = sep_sky
                oix = None
                match_mode = "sky_unaligned_no_pixel_fallback"
            else:
                ic_px, sep_px, oix_px = _pixel_nn_match(dist_thr_px=float(6.0))
                # Prefer pixel match if it yields any finite distances.
                if int(np.count_nonzero(np.isfinite(sep_px))) > 0:
                    icomp = ic_px
                    sep_arcsec_arr = sep_px  # px (kept numeric for mask)
                    oix = oix_px
                    match_mode = "pixel_fallback_bad_wcs"
                    ra_deg = np.full(n, np.nan, dtype=np.float64)
                    dec_deg = np.full(n, np.nan, dtype=np.float64)
                else:
                    icomp = ic_sky
                    sep_arcsec_arr = sep_sky
                    oix = None
                    match_mode = "sky"
        else:
            icomp = ic_sky
            sep_arcsec_arr = sep_sky
            oix = None
            match_mode = "sky"
    else:
        ra_deg = np.full(n, np.nan, dtype=np.float64)
        dec_deg = np.full(n, np.nan, dtype=np.float64)
        if not _frame_on_ref_grid:
            match_mode = "nondet_unaligned_no_wcs"
            icomp = np.zeros(n, dtype=np.int64)
            sep_arcsec_arr = np.full(n, np.nan, dtype=np.float64)
            oix = None
        else:
            match_mode = "pixel_fallback_no_wcs"
            ic_px, sep_px, oix_px = _pixel_nn_match(dist_thr_px=float(15.0))
            icomp = ic_px
            sep_arcsec_arr = sep_px  # px (kept numeric for mask)
            oix = oix_px

    nm_m = len(m_valid)
    idx_det = np.arange(1, n + 1, dtype=np.int32)
    det_str = np.array([f"DET_{i:04d}" for i in idx_det], dtype=object)
    finite_sep = np.isfinite(sep_arcsec_arr)
    icomp_a = icomp if icomp is not None else np.zeros(n, dtype=np.int64)
    if match_mode in ("pixel_fallback_no_wcs", "pixel_fallback_bad_wcs") and oix is not None:
        ok_ix = (icomp_a >= 0) & (icomp_a < int(len(oix)))
        cat_row = np.full(n, -1, dtype=np.int64)
        cat_row[ok_ix] = oix[icomp_a[ok_ix]]
        thr_px = float(15.0) if match_mode == "pixel_fallback_no_wcs" else float(6.0)
        matched = finite_sep & ok_ix & (sep_arcsec_arr <= thr_px) & (cat_row >= 0)
        # For output semantics, keep match_sep_arcsec as NaN when unmatched (already handled).
    else:
        cat_row = np.clip(icomp_a.astype(np.int64, copy=False), 0, max(nm_m - 1, 0))
        matched = finite_sep & (sep_arcsec_arr <= match_thr) & (nm_m > 0)

    safe = np.clip(np.where(matched, cat_row, 0), 0, max(nm_m - 1, 0))
    cat_s_m = (
        m_valid["catalog"].fillna("").astype(str).str.strip().to_numpy(dtype=object)
        if "catalog" in m_valid.columns
        else np.array([""] * nm_m, dtype=object)
    )
    cid_m = (
        m_valid["catalog_id"].fillna("").astype(str).str.strip().to_numpy(dtype=object)
        if "catalog_id" in m_valid.columns
        else np.array([""] * nm_m, dtype=object)
    )
    nm_c = (
        m_valid["name"].fillna("").astype(str).str.strip().to_numpy(dtype=object)
        if "name" in m_valid.columns
        else np.array([""] * nm_m, dtype=object)
    )
    mag_m = (
        pd.to_numeric(m_valid["mag"], errors="coerce").to_numpy(dtype=np.float64)
        if "mag" in m_valid.columns
        else np.full(nm_m, np.nan, dtype=np.float64)
    )
    vx_m = (
        m_valid["vsx_known_variable"].fillna(False).astype(bool).to_numpy()
        if "vsx_known_variable" in m_valid.columns
        else np.zeros(nm_m, dtype=bool)
    )
    gv_m = (
        m_valid["gaia_dr3_variable_catalog"].fillna(False).astype(bool).to_numpy()
        if "gaia_dr3_variable_catalog" in m_valid.columns
        else np.zeros(nm_m, dtype=bool)
    )
    if "catalog_known_variable" in m_valid.columns:
        ck_m = m_valid["catalog_known_variable"].fillna(False).astype(bool).to_numpy()
    else:
        ck_m = vx_m | gv_m

    _fwhm_cent = float(max(1.2, _base_fw_m / float(bfac)))
    if _frame_on_ref_grid:
        x, y, _n_dao_wcs_fallback = _lock_matched_centroids_to_master_grid(
            x,
            y,
            matched=matched,
            safe=safe,
            master_df=m_valid,
        )
    else:
        x, y, _n_dao_wcs_fallback = _apply_dao_centroid_wcs_guard(
            x,
            y,
            matched=matched,
            safe=safe,
            master_df=m_valid,
            fwhm_px=_fwhm_cent,
            max_shift_fwhm=float(getattr(_cfg_dao, "dao_centroid_max_shift_fwhm", 1.0)),
        )

    pmax_arr = _box_peaks_at_centroids(arr, x, y)
    _frame_max_adu = float(np.nanmax(arr))
    _empirical_clip_adu = _detect_empirical_clip_level_adu(arr)
    _peak_sat_lim = _resolve_peak_saturation_limit_adu(
        camera_sat_limit_adu=sat_limit,
        saturate_fraction=sat_frac,
        sky_median_adu=float(med),
        frame_max_adu=_frame_max_adu,
        empirical_clip_adu=_empirical_clip_adu,
        measured_sat_adu=(sat_diag_ctx.measured_sat_adu() if sat_diag_ctx is not None else None),
    )
    _sat_block = _vectorized_star_saturation_columns(
        arr,
        x,
        y,
        sat_limit=_peak_sat_lim,
        sat_frac=sat_frac,
        peak_dao=peak_dao,
        peak_max_adu=pmax_arr,
    )
    _sat_csv, n_sat_pk, n_sat_pl = _proc_sat_block_for_csv(_sat_block)

    cat_sel = cat_s_m[safe]
    cid_sel = cid_m[safe]
    nm_sel = nm_c[safe]
    name_cand = np.where(
        nm_sel != "",
        nm_sel,
        np.where(cid_sel != "", cid_sel, np.where(cat_sel != "", cat_sel, det_str)),
    )
    name_out = np.where(matched, name_cand, det_str)

    mag_out = np.full(n, np.nan, dtype=np.float64)
    mag_out[matched] = mag_m[safe[matched]]

    vx_out = np.zeros(n, dtype=bool)
    gv_out = np.zeros(n, dtype=bool)
    ck_out = np.zeros(n, dtype=bool)
    vx_out[matched] = vx_m[safe[matched]]
    gv_out[matched] = gv_m[safe[matched]]
    ck_out[matched] = ck_m[safe[matched]]

    cid_out = np.array([""] * n, dtype=object)
    cid_out[matched] = cid_m[safe[matched]]


    df_out = pd.DataFrame(
        {
            "name": name_out,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "mag": mag_out,
            "catalog_id": cid_out,
            "x": x,
            "y": y,
            "flux": flux,
            "vsx_known_variable": vx_out,
            "gaia_dr3_variable_catalog": gv_out,
            **_sat_csv,
        }
    )
    n_detected_dao = int(n)

    df_out = _proc_rename_det_names_to_catalog_id(df_out)

    n_sat = int(df_out["likely_saturated"].sum()) if len(df_out) and "likely_saturated" in df_out.columns else 0
    cat_nonempty = (
        df_out["catalog_id"].fillna("").astype(str).str.strip().ne("")
        if len(df_out) and "catalog_id" in df_out.columns
        else pd.Series([], dtype=bool)
    )
    n_matched_final = int(cat_nonempty.sum()) if len(df_out) else 0
    meta = {
        "n_detected_dao": n_detected_dao,
        "n_detected": int(len(df_out)),
        "n_matched": n_matched_final,
        "catalog_rows": int(len(master_df)),
        "catalog_match_mode": export_catalog_match_mode_from_internal(match_mode),
        "n_likely_saturated": n_sat,
        # FORCED-SAT-2026-10: the aligned-frame peak-test level of this frame, for the forced rows
        "peak_sat_limit_adu": float(_peak_sat_lim) if _peak_sat_lim is not None else None,
        "n_saturated_from_peak": n_sat_pk,
        "n_saturated_plateau": n_sat_pl,
        "n_dao_wcs_centroid_fallback": int(_n_dao_wcs_fallback),
        "saturate_limit_adu": float(sat_limit) if sat_limit is not None else None,
        "saturate_limit_source": sat_limit_src,
        "n_vsx_in_field": 0,
        "n_gaia_variable_in_field": 0,
        **foot_meta,
        "field_catalog_cone_csv": None,
        "dao_threshold_sigma": float(dao_threshold_sigma),
        "dao_fwhm_px": _fwhm_used_m,
        "dao_detect_binning": int(bfac),
        "match_sep_arcsec_requested": float(match_sep_arcsec),
        "match_sep_arcsec_effective": float(match_thr),
        "plate_scale_arcsec_per_px": (
            float(plate_scale_arcsec_per_px) if plate_scale_arcsec_per_px is not None else None
        ),
    }
    if int(_n_dao_wcs_fallback) > 0:
        LOGGER.info(
            "[BATCH-E E.2] centroid WCS fallback triggered on %d star-frames",
            int(_n_dao_wcs_fallback),
        )
    if (
        sat_diag_ctx is not None
        and raw_data is not None
        and raw_hdr is not None
        and len(df_out) > 0
    ):
        try:
            from sat_diag import apply_raw_peaks_to_proc_df  # noqa: PLC0415

            apply_raw_peaks_to_proc_df(
                df_out,
                np.asarray(raw_data),
                raw_hdr,
                sat_diag_ctx,
                ref_ra=ref_ra_deg,
                ref_dec=ref_dec_deg,
                drift_ref_catalog_id=drift_ref_catalog_id,
                aligned_hdr=hdr,
            )
            n_sat = (
                int(df_out["likely_saturated_raw"].sum())
                if "likely_saturated_raw" in df_out.columns
                else n_sat
            )
            meta["sat_limit_source"] = str(getattr(sat_diag_ctx, "sat_source", sat_limit_src))
            meta["sat_peak_source"] = str(
                getattr(sat_diag_ctx, "sat_peak_source", "PLACED_APERTURE")
            )
            meta["raw_peaks_used"] = True
        except Exception as _sat_exc:  # noqa: BLE001
            LOGGER.warning("[SAT-DIAG] raw peak merge failed for %s: %s", frame_name, _sat_exc)
    return df_out, meta


def stamp_forced_rows_saturation(
    df: pd.DataFrame,
    data: "np.ndarray",
    hdr: fits.Header,
    *,
    peak_sat_limit_adu: float | None,
    sat_diag_ctx: Any | None = None,
    raw_data: "np.ndarray | None" = None,
    raw_hdr: fits.Header | None = None,
    ref_ra_deg: float | None = None,
    ref_dec_deg: float | None = None,
    drift_ref_catalog_id: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-frame saturation of the forced rows, by the rule of the DAO rows (FORCED-SAT-2026-10).

    Forced rows (``forced_photometry``) are injected after the DAO detection, which measures the
    peaks; before this they carried no ``peak_max_adu`` / ``likely_saturated`` and counted as clean
    (537: 1013 forced rows of 13 faint stars inside a saturated neighbour's core). Same steps as
    ``detect_stars_match_master_reference``: aligned-frame box peak + saturated-core plateau
    (``_vectorized_star_saturation_columns`` at the frame's peak-test level), then the raw-frame
    placed-aperture peak (``sat_diag.apply_raw_peaks_to_proc_df``). DAO rows are not touched.
    """
    meta: dict[str, Any] = {"n_forced": 0, "n_likely_saturated": 0}
    if df is None or not len(df) or "forced_photometry" not in df.columns:
        return df, meta
    forced = df["forced_photometry"].map(lambda v: str(v).strip().lower() in ("true", "1", "1.0")).to_numpy(bool)
    xs = pd.to_numeric(df["x"], errors="coerce").to_numpy(np.float64)
    ys = pd.to_numeric(df["y"], errors="coerce").to_numpy(np.float64)
    fm = forced & np.isfinite(xs) & np.isfinite(ys)
    meta["n_forced"] = int(fm.sum())
    if not fm.any():
        return df, meta
    arr = np.asarray(data, dtype=np.float32)
    x, y = xs[fm], ys[fm]
    lim = float(peak_sat_limit_adu) if peak_sat_limit_adu is not None and math.isfinite(float(peak_sat_limit_adu)) else None
    block = _vectorized_star_saturation_columns(
        arr, x, y,
        sat_limit=lim,
        sat_frac=float(LINEARITY_FRAC),
        peak_dao=np.full(x.size, np.nan),
        peak_max_adu=_box_peaks_at_centroids(arr, x, y),
    )
    csv_block, _n_pk, _n_pl = _proc_sat_block_for_csv(block)
    sub = df.loc[fm].copy()
    for k, v in csv_block.items():
        if k == "peak_dao":  # no DAO detection on this frame
            continue
        sub[k] = v
    if sat_diag_ctx is not None and raw_data is not None and raw_hdr is not None:
        from sat_diag import apply_raw_peaks_to_proc_df  # noqa: PLC0415

        apply_raw_peaks_to_proc_df(
            sub,
            np.asarray(raw_data),
            raw_hdr,
            sat_diag_ctx,
            ref_ra=ref_ra_deg,
            ref_dec=ref_dec_deg,
            drift_ref_catalog_id=drift_ref_catalog_id,
            aligned_hdr=hdr,
        )
        meta["raw_peaks_used"] = True
    out = df.copy()
    for k in sub.columns:
        if k not in out.columns:
            out[k] = pd.Series([np.nan] * len(out), index=out.index, dtype=object)
        if out[k].dtype != sub[k].dtype:
            out[k] = out[k].astype(object)
        out.loc[fm, k] = sub[k].to_numpy()
    meta["n_likely_saturated"] = int(
        sub["likely_saturated"].map(lambda v: str(v).strip().lower() in ("true", "1", "1.0")).sum()
    ) if "likely_saturated" in sub.columns else 0
    return out, meta


_EXPORT_PER_FRAME_WORKER_STATE: dict[str, Any] = {}


def _init_export_per_frame_worker(state: dict[str, Any]) -> None:
    global _EXPORT_PER_FRAME_WORKER_STATE
    _EXPORT_PER_FRAME_WORKER_STATE = state


def _airmass_from_altitude_deg(alt_deg: float) -> float:
    """Kasten & Young (1989) airmass from altitude in degrees."""
    alt_rad = math.radians(float(alt_deg))
    return 1.0 / (
        math.sin(alt_rad) + 0.50572 * (float(alt_deg) + 6.07995) ** (-1.6364)
    )


def _compute_airmass_from_altaz(
    hdr: fits.Header,
    cfg: AppConfig | None = None,
    *,
    db: VyvarDatabase | None = None,
    draft_id: int | None = None,
) -> float:
    """Compute airmass from field center RA/Dec + observer site + mid-exposure JD (AltAz).

    Site is resolved via the unified resolver (per-draft ID_LOCATION -> header
    SITELAT -> flagged config) so airmass uses the same location as BJD/HJD and is
    no longer tied to ``cfg.observer_*`` alone (config-drift trap).
    """
    from time_utils import resolve_observer_location  # noqa: PLC0415

    lat_r, lon_r, alt_r = resolve_observer_location(hdr, db, draft_id, cfg=cfg)
    if lat_r is None or lon_r is None:
        return float("nan")
    lat = float(lat_r)
    lon = float(lon_r)
    alt_m = float(alt_r) if alt_r is not None else 0.0
    if lat == 0.0 and lon == 0.0:
        return float("nan")

    from time_utils import mid_exposure_jd  # noqa: PLC0415

    jd = mid_exposure_jd(hdr)
    if jd is None:
        return float("nan")

    # Safe: airmass from CRVAL1/2 or OBJCTRA/DEC; returns nan if missing - handled downstream.
    ra = _header_float_tu(hdr, "CRVAL1")
    dec = _header_float_tu(hdr, "CRVAL2")
    if ra is None or dec is None:
        if "OBJCTRA" in hdr and "OBJCTDEC" in hdr:
            try:
                from time_utils import _parse_objctradec  # noqa: PLC0415

                ra, dec = _parse_objctradec(str(hdr["OBJCTRA"]), str(hdr["OBJCTDEC"]))
            except Exception:  # noqa: BLE001
                ra, dec = None, None
    if ra is None or dec is None:
        return float("nan")

    try:
        from astropy.coordinates import AltAz, EarthLocation, SkyCoord  # noqa: PLC0415
        from astropy.time import Time  # noqa: PLC0415

        location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt_m * u.m)
        t = Time(float(jd), format="jd", scale="utc", location=location)
        target = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg, frame="icrs")
        altaz = target.transform_to(AltAz(obstime=t, location=location))
        alt_deg = float(altaz.alt.deg)
        if not math.isfinite(alt_deg) or alt_deg < 5.0 or alt_deg > 90.0:
            return float("nan")
        am = _airmass_from_altitude_deg(alt_deg)
        if 0.9 <= am <= 10.0:
            LOGGER.info(
                "Airmass from AltAz fallback: %.4f (alt=%.2f deg)",
                am,
                alt_deg,
            )
            return round(am, 5)
    except Exception:  # noqa: BLE001
        return float("nan")
    return float("nan")


def _extract_airmass_from_header(
    hdr: fits.Header,
    cfg: AppConfig | None = None,
    *,
    db: VyvarDatabase | None = None,
    draft_id: int | None = None,
) -> float:
    """Extract airmass from a FITS header for per-frame catalog metadata.

    Priority order:
    1. ``AIRMASS`` / ``AIRMAS`` / ``SECZ`` header keywords when in the physical range.
    2. ``ALT_OBJ`` / ``OBJCTALT`` / ``ALTITUDE`` / ``TELALT`` converted via
       Kasten & Young (1989) (:func:`_airmass_from_altitude_deg`).
    3. Field-center AltAz computed from WCS/object coordinates, observer site, and
       mid-exposure JD (:func:`_compute_airmass_from_altaz`), also using Kasten & Young (1989).

    Returns ``float('nan')`` when no path yields a finite value.
    """
    # Priamy keyword
    for kw in ("AIRMASS", "AIRMAS", "SECZ"):
        val = hdr.get(kw)
        if val is not None:
            try:
                v = float(val)
                if 0.9 <= v <= 10.0:  # fyzikalny rozsah
                    return v
            except (TypeError, ValueError):
                pass

    # Fallback: altitude -> airmass cez aproximaciu Kasten & Young (1989)
    for kw in ("ALT_OBJ", "OBJCTALT", "ALTITUDE", "TELALT"):
        val = hdr.get(kw)
        if val is not None:
            try:
                alt_deg = float(val)
                if 5.0 <= alt_deg <= 90.0:
                    am = _airmass_from_altitude_deg(alt_deg)
                    if 0.9 <= am <= 10.0:
                        return round(am, 5)
            except (TypeError, ValueError):
                pass

    am_altaz = _compute_airmass_from_altaz(hdr, cfg, db=db, draft_id=draft_id)
    if math.isfinite(am_altaz):
        return am_altaz

    return float("nan")


def _cfg_from_export_worker_state(st: dict[str, Any]) -> AppConfig:
    """Rebuild minimal AppConfig observer fields from per-frame worker state dict."""
    c = AppConfig()
    c.observer_lat = float(st.get("observer_lat", 0.0) or 0.0)
    c.observer_lon = float(st.get("observer_lon", 0.0) or 0.0)
    c.observer_alt_m = float(st.get("observer_alt_m", 0.0) or 0.0)
    return c


# OUTPUT-REVIEW-2026-10 step 4: columns the per-frame export uses in memory (identity, the
# catalogue-only filter, the forced-row and saturation steps) or that duplicate a kept column,
# but that no reader takes from a proc file (column audit 2026-10-21). Not written.
PROC_NOT_WRITTEN: tuple[str, ...] = (
    "name",  # = catalog_id
    "flux",  # = dao_flux
    "catalog",
    "b_v",
    "source_type",
    "forced_photometry",
    "peak_dao",
    "saturate_limit_adu",
    "likely_saturated_raw",
    "zone",
    "is_usable",
    "geometry_ok",
    "geometry_miss_reason",
    "non_single_star",
    "exo_host_obj_id",
    "exo_host_name",
    "exo_cat_source",
    "exo_disposition",
    "exo_match_sep_arcsec",
)


def proc_columns_for_write(df: pd.DataFrame) -> pd.DataFrame:
    """The proc CSV as written: ``df`` without :data:`PROC_NOT_WRITTEN`."""
    return df.drop(columns=[c for c in PROC_NOT_WRITTEN if c in df.columns])


def _export_per_frame_run_catalog_core(
    base_path: Path,
    hdr: fits.Header,
    data: Any,
    st: dict[str, Any],
) -> dict[str, Any]:
    log_event("DEBUG: per-frame worker entry point called")
    fname = base_path.name
    debug_pixel_match: dict[str, Any] = {
        "file": fname,
        "master_cols": None,
        "have_x": None,
        "have_y": None,
        "match_mode": None,
        "plate_scale_arcsec_per_px": None,
        "n_matched": None,
    }
    deferred_writes: list[tuple[str, pd.DataFrame]] = []
    if not _has_valid_wcs(hdr):
        return {
            "file": fname,
            "status": "no_wcs",
            "csv": "",
            "deferred_writes": deferred_writes,
            "infolog_messages": [],
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        _ = WCS(hdr)

    master_tab = st.get("master_tab")
    # MASTER-LOCK-2026-10: identities come only from the MASTERSTAR catalogue.
    if master_tab is None or getattr(master_tab, "empty", True):
        return {
            "file": fname,
            "status": "error: masterstars_full_match.csv missing",
            "csv": "",
            "deferred_writes": deferred_writes,
            "infolog_messages": [
                f"Per-frame catalog {fname}: masterstars_full_match.csv missing (MASTERSTAR lock)",
            ],
        }

    df: pd.DataFrame
    meta: dict[str, Any]
    try:
        # One-time debug: verify MASTERSTAR pixel columns used for KD-tree matching.
        try:
            if not bool(st.get("_debug_logged_master_xy")) and master_tab is not None:
                st["_debug_logged_master_xy"] = True
                cols = [c for c in ("x", "y", "ra_deg", "dec_deg") if c in master_tab.columns]
                if cols:
                    log_event(
                        "DEBUG: masterstars_full_match.csv sample columns "
                        + ",".join(cols)
                        + ":\n"
                        + master_tab.loc[:, cols].head(5).to_string(index=False)
                    )
        except Exception:  # noqa: BLE001
            pass
        try:
            if master_tab is not None and "catalog_id" in master_tab.columns and "ra_deg" in master_tab.columns:
                logging.getLogger(__name__).debug(
                    "SKY-MATCH: "
                    f"master_tab rows={len(master_tab)}, "
                    f"catalog_id notna={int(master_tab['catalog_id'].notna().sum())}, "
                    f"ra_deg notna={int(master_tab['ra_deg'].notna().sum())}"
                )
        except Exception:  # noqa: BLE001
            pass
        try:
            if master_tab is not None:
                debug_pixel_match["master_cols"] = list(master_tab.columns[:10])
                debug_pixel_match["have_x"] = bool("x" in master_tab.columns)
                debug_pixel_match["have_y"] = bool("y" in master_tab.columns)
        except Exception:  # noqa: BLE001
            pass
        _sat_ctx = _sat_ctx_from_worker(st)
        _raw_arr = _load_raw_for_frame(st, fname)
        _raw_hdr = _load_raw_hdr_for_frame(st, fname)
        df, meta = detect_stars_match_master_reference(
            data,
            hdr,
            master_tab,
            max_catalog_rows=int(st["max_catalog_rows"]),
            match_sep_arcsec=float(st["catalog_match_max_sep_arcsec"]),
            dao_threshold_sigma=float(st["dao_threshold_sigma"]),
            dao_fwhm_px=float(st.get("dao_fwhm_px", 2.5)),
            equipment_saturate_adu=st.get("equipment_saturate_adu"),
            frame_name=fname,
            sat_diag_ctx=_sat_ctx,
            raw_data=_raw_arr,
            raw_hdr=_raw_hdr,
            ref_ra_deg=st.get("sat_diag_ref_ra"),
            ref_dec_deg=st.get("sat_diag_ref_dec"),
            drift_ref_catalog_id=st.get("sat_diag_ref_catalog_id"),
        )
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0373] detect_stars_and_match_catalog exception returns status error with empty csv for that f...: %s', exc)
        return {
            "file": fname,
            "status": f"error: {exc}",
            "csv": "",
            "deferred_writes": deferred_writes,
            "infolog_messages": [f"Per-frame catalog {fname}: {exc}"],
            "debug_pixel_match": debug_pixel_match,
        }


    _before_dao = len(df)
    df = _proc_drop_unmatched_dao_rows(df)
    LOGGER.debug("[TODO-13] catalog-only pre-filter (detect): %d -> %d rows", _before_dao, len(df))

    # --- Join MASTERSTAR annotations into per-frame catalog (via catalog_id) ---
    # For matched rows (catalog_id non-empty), bring stable MASTERSTAR columns (bp_rp, G, saturation).
    try:
        if master_tab is not None and "catalog_id" in df.columns and "catalog_id" in master_tab.columns:
            _JOIN_COLS = [
                "is_saturated",
                "bp_rp",
                "phot_g_mean_mag",
                "saturate_limit_adu_85pct",
                "source_type",
            ]
            join_cols = [c for c in _JOIN_COLS if c in master_tab.columns]
            if join_cols:
                master_lookup = master_tab[["catalog_id"] + join_cols].copy()
                master_lookup["catalog_id"] = (
                    master_lookup["catalog_id"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
                )
                master_lookup = master_lookup.dropna(subset=["catalog_id"])
                master_lookup = master_lookup.drop_duplicates(subset=["catalog_id"], keep="first")

                df = df.copy()
                df["catalog_id"] = df["catalog_id"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

                # Avoid column collisions; MASTERSTAR values should win for these columns.
                df = df.drop(columns=[c for c in join_cols if c in df.columns], errors="ignore")
                df = df.merge(master_lookup, on="catalog_id", how="left")
    except Exception:  # noqa: BLE001
        pass

    # FORCED-PHOT-01: inject force-eligible MASTERSTAR members missing from DAO match.
    # DAO path above is unchanged (n_raw_dao / match counts preserved). Forced rows are
    # measured at locked MASTERSTAR (x,y) with bounded peak refine.
    try:
        if master_tab is not None and bool(st.get("forced_photometry_enabled", True)):
            from forced_photometry import inject_forced_masterstar_rows  # noqa: PLC0415

            _force_ids = st.get("forced_photometry_catalog_ids")
            _force_set = (
                {str(x).strip() for x in _force_ids if str(x).strip()}
                if _force_ids
                else None
            )
            _margin = float(st.get("forced_photometry_margin_px", 0.0) or 0.0)
            df, _fp_meta = inject_forced_masterstar_rows(
                df,
                master_tab,
                image=np.asarray(data) if data is not None else None,
                margin_px=_margin,
                force_ids=_force_set,
            )
            try:
                meta["forced_photometry"] = dict(_fp_meta)
                debug_pixel_match["forced_photometry"] = dict(_fp_meta)
            except Exception:  # noqa: BLE001
                pass
            LOGGER.debug(
                "[FORCED-PHOT] %s injected=%s geometry_miss=%s",
                fname,
                _fp_meta.get("n_injected"),
                _fp_meta.get("n_geometry_miss"),
            )
    except Exception as _fp_exc:  # noqa: BLE001
        LOGGER.error("[FORCED-PHOT] inject failed (DAO path kept): %s", _fp_exc)

    # FORCED-SAT-2026-10: forced rows get the per-frame saturation test of the DAO rows
    # (aligned-frame peak + core plateau, then the raw-frame placed-aperture peak).
    try:
        df, _fs_meta = stamp_forced_rows_saturation(
            df,
            data,
            hdr,
            peak_sat_limit_adu=meta.get("peak_sat_limit_adu"),
            sat_diag_ctx=_sat_ctx,
            raw_data=_raw_arr,
            raw_hdr=_raw_hdr,
            ref_ra_deg=st.get("sat_diag_ref_ra"),
            ref_dec_deg=st.get("sat_diag_ref_dec"),
            drift_ref_catalog_id=st.get("sat_diag_ref_catalog_id"),
        )
        meta["forced_saturation"] = _fs_meta
    except Exception as _fs_exc:  # noqa: BLE001
        LOGGER.error("[FORCED-SAT] %s: saturation test of forced rows failed: %s", fname, _fs_exc)
        meta["forced_saturation"] = {"status": f"failed: {_fs_exc}"}

    try:
        debug_pixel_match["match_mode"] = meta.get("catalog_match_mode")
        debug_pixel_match["plate_scale_arcsec_per_px"] = meta.get("plate_scale_arcsec_per_px")
        debug_pixel_match["n_matched"] = meta.get("n_matched")
    except Exception:  # noqa: BLE001
        pass

    st["current_frame_name"] = fname
    df = _apply_aperture_catalog_enhancements_from_st(df, data, hdr, st)

    # --- Time columns (JD / HJD / BJD) ---
    _db_tc = None
    try:
        from time_utils import compute_time_columns

        _dbp = str(st.get("database_path") or "").strip()
        _did_tc = st.get("draft_id")
        if _dbp and _did_tc is not None:
            try:
                _db_tc = VyvarDatabase(Path(_dbp))
            except Exception:  # noqa: BLE001
                _db_tc = None
        _geo_cfg = _cfg_from_export_worker_state(st)
        _time_cols = compute_time_columns(
            hdr,
            db=_db_tc,
            draft_id=int(_did_tc) if _did_tc is not None else None,
            cfg=_geo_cfg,
        )
        _tk = ("jd_mid", "hjd_mid", "bjd_tdb_mid")
        _cols_base = list(df.columns)
        _anchors = [c for c in ("jd", "inspection_jd") if c in _cols_base]
        if _anchors:
            _pos = max(_cols_base.index(c) for c in _anchors) + 1
        else:
            _flux_first = next((c for c in _cols_base if c in ("dao_flux", "flux")), None)
            _pos = _cols_base.index(_flux_first) if _flux_first is not None else len(_cols_base)
        for _i, _nm in enumerate(_tk):
            df.insert(_pos + _i, _nm, _time_cols[_nm])

        # Airmass - rovnaka hodnota pre vsetky hviezdy v snimke (frame-level)
        _am_val = _extract_airmass_from_header(
            hdr,
            cfg=_geo_cfg,
            db=_db_tc,
            draft_id=int(_did_tc) if _did_tc is not None else None,
        )
        if "airmass" not in df.columns:
            _am_insert_pos = _pos + len(_tk)
            df.insert(_am_insert_pos, "airmass", _am_val)
    except Exception as _tc_exc:  # noqa: BLE001
        log_event(f"Time columns skipped: {_tc_exc}")
        for _tc in ("jd_mid", "hjd_mid", "bjd_tdb_mid"):
            if _tc not in df.columns:
                df[_tc] = None
        if "airmass" not in df.columns:
            df["airmass"] = float("nan")
    finally:
        if _db_tc is not None:
            try:
                _db_tc.conn.close()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("[PIPELINE] Cleanup step failed (non-critical): %s", exc)

    df2 = df.copy()
    _cmm_export = str(meta.get("catalog_match_mode") or "").strip()
    if _cmm_export:
        df2["catalog_match_mode"] = _cmm_export

    _before_cat = len(df2)
    df2 = _proc_catalog_keep_matched_rows_only(df2)
    LOGGER.debug("[TODO-13] catalog-only filter: %d -> %d rows", _before_cat, len(df2))

    # FORCED-PHOT / DRAFT-514-TRIAGE A2: multiple DAO detections can match the same
    # Gaia id; dedupe before write. Alternate export path already called this;
    # production ``_export_per_frame_run_catalog_core`` did not (root cause of
    # 127 duplicate rows in proc_BO_CVn_Light_001 on draft 514).
    _before_dedupe = len(df2)
    df2 = _proc_deduplicate_matched_catalog_rows(df2)
    if len(df2) != _before_dedupe:
        LOGGER.debug(
            "[PROC] dedupe catalog_id: %d -> %d rows",
            _before_dedupe,
            len(df2),
        )

    from proc_frame_store import PROC_FRAME_COLS, frame_values_of, round_proc_columns  # noqa: PLC0415

    df2 = round_proc_columns(proc_columns_for_write(df2))
    # Per-frame values go to frames.csv (written by the collector once all frames are done).
    frame_values = frame_values_of(df2)
    df2 = df2.drop(columns=[c for c in PROC_FRAME_COLS if c in df2.columns])

    csv_paths: list[str] = []
    write_sidecar = bool(st.get("write_sidecar_csv_next_to_fits"))
    mirror_flat = bool(st.get("mirror_flat_platesolve_folder"))
    defer = bool(st.get("defer_disk_writes"))
    out_flat = Path(str(st.get("out_flat") or "."))

    if write_sidecar:
        sidecar = proc_csv_path_for_aligned_fits(base_path)
        if defer:
            deferred_writes.append((str(sidecar), df2.copy()))
        else:
            _vyvar_df_to_csv(df2, sidecar)
        csv_paths.append(str(sidecar))

    if mirror_flat:
        stem = Path(fname).stem
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)[:120]
        flat_path = out_flat / f"{safe}_catalog.csv"
        if defer:
            deferred_writes.append((str(flat_path), df2.copy()))
        else:
            _vyvar_df_to_csv(df2, flat_path)
        csv_paths.append(str(flat_path))

    primary_csv = csv_paths[0] if csv_paths else ""
    return {
        "file": fname,
        "status": "ok",
        "csv": primary_csv,
        "n_detected": meta.get("n_detected"),
        "n_matched": meta.get("n_matched"),
        "raw_peaks_used": bool(meta.get("raw_peaks_used")),
        "frame_values": frame_values,
        "deferred_writes": deferred_writes,
        "infolog_messages": [],
        "debug_pixel_match": debug_pixel_match,
    }


def _export_per_frame_disk_worker_task(fp_str: str) -> dict[str, Any]:
    import numpy as np

    st = _EXPORT_PER_FRAME_WORKER_STATE
    fp = Path(fp_str)
    try:
        with fits.open(fp, memmap=False) as hdul:
            hdr = hdul[0].header.copy()
            data = np.array(hdul[0].data, dtype=np.float32, copy=True)
    except Exception as exc:  # noqa: BLE001
        return {
            "file": fp.name,
            "status": f"read_error: {exc}",
            "csv": "",
            "deferred_writes": [],
            "infolog_messages": [f"Per-frame catalog {fp.name}: read_error: {exc}"],
        }
    return _export_per_frame_run_catalog_core(fp, hdr, data, st)


def _export_per_frame_ram_worker_task(
    packed: tuple[str, bytes, bytes, int, int],
) -> dict[str, Any]:
    """Picklable worker: (fits_path_str, pickled_fits_Header, float32_bytes, ny, nx)."""
    import numpy as np

    st = _EXPORT_PER_FRAME_WORKER_STATE
    base_str, hdr_pkl, raw, ny, nx = packed
    fname = Path(base_str).name
    empty_def: list[tuple[str, pd.DataFrame]] = []
    try:
        hdr = pickle.loads(hdr_pkl)
    except Exception as exc:  # noqa: BLE001
        return {
            "file": fname,
            "status": f"header_error: {exc}",
            "csv": "",
            "deferred_writes": empty_def,
            "infolog_messages": [f"Per-frame catalog {fname}: header_error: {exc}"],
        }
    try:
        data = np.frombuffer(raw, dtype=np.float32, count=ny * nx).reshape((ny, nx)).copy()
    except Exception as exc:  # noqa: BLE001
        return {
            "file": fname,
            "status": f"buffer_error: {exc}",
            "csv": "",
            "deferred_writes": empty_def,
            "infolog_messages": [f"Per-frame catalog {fname}: buffer_error: {exc}"],
        }
    return _export_per_frame_run_catalog_core(Path(base_str), hdr, data, st)


def _finalize_hybrid_bkg_fallback_sidecar(
    proc_dir: Path,
    *,
    write_sidecar: bool,
    gain: float,
    read_noise: float,
    setup_label: str,
) -> dict[str, Any]:
    """Post-export hybrid Howell fallback scaling when sidecar proc CSVs exist."""
    if not write_sidecar:
        return {}
    try:
        from photometry_core import finalize_hybrid_bkg_fallback_proc_dir

        return finalize_hybrid_bkg_fallback_proc_dir(
            proc_dir,
            gain=float(gain),
            read_noise=float(read_noise),
            setup_label=str(setup_label),
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[PHOT] hybrid bkg fallback finalize skipped: %s", exc)
        return {}


def _estimate_catalog_frame_hw(
    work_ram: Sequence[tuple[str, Any, Any]] | None,
    files: list[Path],
) -> tuple[int, int]:
    """Rough (ny, nx) for RAM cap heuristics (per-frame catalog export)."""
    import numpy as np

    if work_ram:
        d = np.asarray(work_ram[0][2])
        if d.ndim == 2:
            return int(d.shape[0]), int(d.shape[1])
    for fp in files[:1]:
        try:
            with fits.open(fp, memmap=False) as h:
                nh = h[0].header
                ny = int(nh.get("NAXIS2", 0) or 0)
                nx = int(nh.get("NAXIS1", 0) or 0)
                if ny > 0 and nx > 0:
                    return ny, nx
        except Exception:  # noqa: BLE001
            continue
    return 2048, 2048


def _vyvar_cap_mp_workers_for_catalog(
    n_workers: int,
    frame_hw: tuple[int, int],
    *,
    reserve_gb: float,
) -> int:
    """Cap process pool size using available RAM (rough float32 frame footprint per worker)."""
    h, w = frame_hw
    if h <= 0 or w <= 0:
        h, w = 2048, 2048
    per_worker = max(int(h * w * 4 * 3), 1)
    try:
        import psutil
    except ImportError:
        return max(1, n_workers)
    reserve = int(max(0.0, float(reserve_gb)) * (1024**3))
    avail = int(psutil.virtual_memory().available) - reserve
    if avail <= 0:
        return 1
    mx = max(1, avail // per_worker)
    return max(1, min(int(n_workers), mx))


