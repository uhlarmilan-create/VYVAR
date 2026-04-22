"""MASTERSTAR astrometry optimizer for broader Gaia matching.

Workflow:
- Load ``masterstars.csv``.
- Build central-region displacement model from already matched stars.
- Apply a radial polynomial correction in pixel space.
- Re-match unmatched stars with adaptive sky radius (1.0" -> 5.0" toward edges).
- Save ``masterstars_full_match.csv``.

Non-negotiable rules (Zeiss wide-field / MASTERSTAR):
- SIP distortion refit order is ``_SIP_MIN_ORDER`` (default 3 — stabilejšie pri horšom počiatočnom matchi ako SIP5).
- **The Grip**: **5** iterative rematch+refit+WCS cycles; each cycle tightens sky radius and refits SIP.
- **Pass 0** sky gate starts at **5.0 arcsec**; after the first coarse refit, subsequent
  grip iterations use a **tighter** center radius.
- **Bright anchors** (Gaia ``g_mag`` < 10.5): never dropped from the displacement model for RMS alone;
  matching tolerates up to **5 px** for anchors.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS

from database import get_gaia_db_max_g_mag, query_local_gaia
from infolog import log_event
from utils import strip_celestial_wcs_keys

# --- Non-negotiable optimizer constants (see module docstring) ---
_SIP_MIN_ORDER: int = 3
_GRIP_ITERATIONS: int = 5
_PASS0_CENTER_ARCSEC: float = 5.0
_BRIGHT_MAG_ANCHOR: float = 10.5  # g_mag < 10.5 = bright anchor (Zeiss wide-field)
_BRIGHT_MAX_MATCH_DPX: float = 5.0
# Post PC parity flip: only small matched-only pixel shift (RA/Dec v CSV sú pred flipom — NN jump na všetky detekcie klamal ~600px).
_POST_FLIP_JUMP_MAX_ABS_PX: float = 25.0
_POST_FLIP_JUMP_MAX_HYPOT_PX: float = 35.0
_POST_FLIP_MATCHED_MIN: int = 20
_FITS_KEY_OPT_PARITY_FLIP: str = "VY_OPTPF"


def _optimizer_parity_flip_already_in_fits(fits_path: Path) -> bool:
    """True if optimizer už raz flipol PC na tomto FITS (druhý beh nesmie znova prepínať)."""
    try:
        with fits.open(fits_path, memmap=False) as hdul:
            h = hdul[0].header
            v = h.get(_FITS_KEY_OPT_PARITY_FLIP)
            if v is None:
                return False
            if isinstance(v, (bool, np.bool_)):
                return bool(v)
            try:
                return int(v) != 0
            except (TypeError, ValueError):
                return str(v).strip().lower() in ("1", "t", "true", "yes")
    except Exception:  # noqa: BLE001
        return False


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _apply_wcs_pc_parity_flip_to_primary(fits_path: Path, *, set_vy_optpf: bool = True) -> bool:
    """Negate PC matrix column 0 in primary HDU WCS (toggle handedness), flush FITS.

    ``set_vy_optpf=True`` značí úspešný *zámerný* flip optimizéra; pri revert-e daj ``False`` a kľúč sa nepridá.
    """
    try:
        with fits.open(fits_path, mode="update", memmap=False) as hdul:
            hh = hdul[0].header
            ww = WCS(hh)
            if not getattr(ww, "has_celestial", False):
                return False
            pc = np.asarray(ww.wcs.get_pc(), dtype=np.float64).copy()
            if pc.shape != (2, 2):
                return False
            pc[:, 0] *= -1.0
            ww.wcs.pc = pc
            wh = ww.to_header(relax=True)
            strip_celestial_wcs_keys(hh)
            for key in wh:
                if key in ("", "COMMENT", "HISTORY", "SIMPLE", "BITPIX", "NAXIS", "EXTEND"):
                    continue
                if key.startswith("NAXIS") and key != "NAXIS":
                    continue
                try:
                    hh[key] = wh[key]
                except Exception:  # noqa: BLE001
                    pass
            note = "VYVAR: astrometry_optimizer parity flip (PC[:,0] negated)"
            if set_vy_optpf:
                hh.add_history(note)
                hh[_FITS_KEY_OPT_PARITY_FLIP] = (True, "VYVAR optimizer applied one PC parity flip")
            else:
                hh.add_history(note + " [revert]")
                if _FITS_KEY_OPT_PARITY_FLIP in hh:
                    del hh[_FITS_KEY_OPT_PARITY_FLIP]
            hdul.flush()
        return True
    except Exception as exc:  # noqa: BLE001
        log_event(f"Astrometry optimizer: parity flip write failed: {exc!s}")
        return False


def _norm_id(v: Any) -> str:
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""
    try:
        fv = float(s)
        if np.isfinite(fv) and abs(fv - round(fv)) < 1e-9:
            return str(int(round(fv)))
    except Exception:  # noqa: BLE001
        pass
    return s


def _poly_features(xn: np.ndarray, yn: np.ndarray) -> np.ndarray:
    r2 = xn * xn + yn * yn
    r4 = r2 * r2
    return np.column_stack([np.ones_like(xn), xn, yn, r2, xn * yn, xn * r2, yn * r2, r4])


def _fit_poly_model(x: np.ndarray, y: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x0 = float(np.nanmedian(x))
    y0 = float(np.nanmedian(y))
    sx = float(max(1.0, np.nanstd(x)))
    sy = float(max(1.0, np.nanstd(y)))
    xn = (x - x0) / sx
    yn = (y - y0) / sy
    a = _poly_features(xn, yn)
    cx, *_ = np.linalg.lstsq(a, dx, rcond=None)
    cy, *_ = np.linalg.lstsq(a, dy, rcond=None)
    # Pack normalizer into first 4 values for reuse.
    norm = np.array([x0, y0, sx, sy], dtype=np.float64)
    return np.concatenate([norm, cx]), np.concatenate([norm, cy])


def _eval_poly(model: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0, y0, sx, sy = model[:4]
    coeff = model[4:]
    xn = (x - x0) / sx
    yn = (y - y0) / sy
    return _poly_features(xn, yn) @ coeff


def _gaia_for_field(
    df: pd.DataFrame,
    gaia_db_path: str | Path,
    *,
    mag_limit: float | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    ra_col = _first_existing_col(df, ["ra_deg", "RA_DEG", "ra", "RA"])
    de_col = _first_existing_col(df, ["dec_deg", "DEC_DEG", "dec", "DEC"])
    if ra_col is None or de_col is None:
        return pd.DataFrame()
    ra = pd.to_numeric(df.get(ra_col), errors="coerce")
    de = pd.to_numeric(df.get(de_col), errors="coerce")
    ok = ra.notna() & de.notna()
    if not bool(ok.any()):
        return pd.DataFrame()
    margin = 0.03
    _ml = mag_limit
    if _ml is None or (not math.isfinite(float(_ml))) or float(_ml) <= 0:
        _gmx = float(get_gaia_db_max_g_mag(Path(gaia_db_path)))
        _ml = _gmx if _gmx > 0.0 else 20.0
    _mr = None
    if max_rows is not None:
        try:
            _mri = int(max_rows)
            if _mri > 0:
                _mr = _mri
        except (TypeError, ValueError):
            _mr = None
    rows = query_local_gaia(
        gaia_db_path,
        ra_min=float(ra[ok].min()) - margin,
        ra_max=float(ra[ok].max()) + margin,
        dec_min=float(de[ok].min()) - margin,
        dec_max=float(de[ok].max()) + margin,
        mag_limit=float(_ml),
        max_rows=_mr,
    )
    if not rows:
        return pd.DataFrame()
    gdf = pd.DataFrame(rows)
    gdf["source_id"] = gdf.get("source_id").astype(str)
    return gdf


def optimize_masterstar_matches(
    *,
    masterstars_csv: str | Path,
    masterstar_fits: str | Path,
    gaia_db_path: str | Path,
    output_csv: str | Path | None = None,
    gaia_mag_limit: float | None = None,
    gaia_max_catalog_rows: int | None = None,
    mirror_orientation_extra_log: bool = True,
    sip_force_rms_guard_ratio: float | None = 1.15,
) -> Path:
    """Optimize MASTERSTAR matches and write ``masterstars_full_match.csv``.

    ``gaia_mag_limit`` / ``gaia_max_catalog_rows`` should match the main field catalog query
    (defaults: DB max g_mag and no SQL row cap) so pass3/pass4 edge rematch sees faint Gaia stars.
    """
    csv_path = Path(masterstars_csv).resolve()
    fits_path = Path(masterstar_fits).resolve()
    out_path = (
        Path(output_csv).resolve()
        if output_csv
        else (Path(masterstar_fits).resolve().parent / "masterstars_full_match.csv")
    )

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Empty CSV: {csv_path}")

    with fits.open(fits_path, memmap=False) as hdul:
        hdr = hdul[0].header.copy()
        data = np.asarray(hdul[0].data, dtype=np.float32)
    w = WCS(hdr)
    if not getattr(w, "has_celestial", False):
        raise ValueError(f"MASTERSTAR FITS has no usable WCS: {fits_path}")

    gdf = _gaia_for_field(
        df,
        gaia_db_path,
        mag_limit=gaia_mag_limit,
        max_rows=gaia_max_catalog_rows,
    )
    if gdf.empty:
        raise ValueError("Gaia query returned no rows for field bounds.")
    _ra_q = pd.to_numeric(gdf["ra"], errors="coerce")
    _de_q = pd.to_numeric(gdf["dec"], errors="coerce")
    gdf = gdf[np.isfinite(_ra_q) & np.isfinite(_de_q)].reset_index(drop=True)
    if gdf.empty:
        raise ValueError("Gaia query returned no finite RA/Dec rows.")
    gcoo = SkyCoord(
        ra=pd.to_numeric(gdf["ra"], errors="coerce").to_numpy(dtype=np.float64) * u.deg,
        dec=pd.to_numeric(gdf["dec"], errors="coerce").to_numpy(dtype=np.float64) * u.deg,
        frame="icrs",
    )
    gx, gy = w.world_to_pixel(gcoo)
    gdf["x_wcs"] = np.asarray(gx, dtype=np.float64)
    gdf["y_wcs"] = np.asarray(gy, dtype=np.float64)
    gdf = gdf[np.isfinite(gdf["x_wcs"]) & np.isfinite(gdf["y_wcs"])].reset_index(drop=True)
    if gdf.empty:
        raise ValueError("Gaia query: no stars project into the image with current WCS.")
    gcoo = SkyCoord(
        ra=pd.to_numeric(gdf["ra"], errors="coerce").to_numpy(dtype=np.float64) * u.deg,
        dec=pd.to_numeric(gdf["dec"], errors="coerce").to_numpy(dtype=np.float64) * u.deg,
        frame="icrs",
    )

    x = pd.to_numeric(df.get("x"), errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(df.get("y"), errors="coerce").to_numpy(dtype=np.float64)
    id_col = _first_existing_col(df, ["catalog_id", "source_id", "gaia_id", "CATALOG_ID", "GAIA_ID"])
    sep_col = _first_existing_col(df, ["match_sep_arcsec", "sep_arcsec", "gaia_sep_arcsec", "MATCH_SEP_ARCSEC"])
    ra_col = _first_existing_col(df, ["ra_deg", "RA_DEG", "ra", "RA"])
    de_col = _first_existing_col(df, ["dec_deg", "DEC_DEG", "dec", "DEC"])
    try:
        _dra = pd.to_numeric(df.get(ra_col), errors="coerce") if ra_col is not None else pd.Series(np.nan, index=df.index)
        _dde = pd.to_numeric(df.get(de_col), errors="coerce") if de_col is not None else pd.Series(np.nan, index=df.index)
        _gra = pd.to_numeric(gdf.get("ra"), errors="coerce")
        _gde = pd.to_numeric(gdf.get("dec"), errors="coerce")
        log_event(
            "Astrometry optimizer BBOX: "
            f"DAO RA[{float(_dra.min()):.6f},{float(_dra.max()):.6f}] "
            f"Dec[{float(_dde.min()):.6f},{float(_dde.max()):.6f}] | "
            f"Gaia RA[{float(_gra.min()):.6f},{float(_gra.max()):.6f}] "
            f"Dec[{float(_gde.min()):.6f},{float(_gde.max()):.6f}]"
        )
    except Exception:  # noqa: BLE001
        pass
    if id_col is None:
        cid = pd.Series([""] * len(df))
    else:
        cid = df.get(id_col, pd.Series([""] * len(df))).fillna("").astype(str)
        if "catalog_id" not in df.columns:
            df["catalog_id"] = cid
    sep = pd.to_numeric(df.get(sep_col), errors="coerce") if sep_col is not None else pd.Series(np.nan, index=df.index)
    log_event(
        f"Astrometry optimizer input columns: id_col={id_col or 'NONE'}, sep_col={sep_col or 'NONE'}, "
        f"matched_nonempty={int(cid.astype(str).str.strip().ne('').sum())}/{len(df)}"
    )

    gmap = {_norm_id(sid): i for i, sid in enumerate(gdf["source_id"].astype(str).tolist())}
    matched_idx: list[int] = []
    gm_idx: list[int] = []
    for i, sid in enumerate(cid.tolist()):
        sid_n = _norm_id(sid)
        if not sid_n or sid_n not in gmap:
            continue
        if not (np.isfinite(x[i]) and np.isfinite(y[i])):
            continue
        # Align with MASTERSTAR detect_stars match tolerance (~8″); 2″ was too strict after WCS handoff.
        if pd.notna(sep.iloc[i]) and float(sep.iloc[i]) >= 8.0:
            continue
        matched_idx.append(i)
        gm_idx.append(gmap[sid_n])
    if len(matched_idx) < 50:
        raise ValueError(f"Not enough matched stars to build displacement model (need >=50, got {len(matched_idx)}).")

    h, wpx = data.shape
    cx, cy = float(wpx / 2.0), float(h / 2.0)
    rr = np.hypot(x[matched_idx] - cx, y[matched_idx] - cy)
    rmax = float(max(1.0, np.nanmax(rr)))
    central = rr <= (0.45 * rmax)
    if int(np.count_nonzero(central)) < 6:
        central = np.ones_like(rr, dtype=bool)

    mm = np.asarray(matched_idx, dtype=int)[central]
    gg = np.asarray(gm_idx, dtype=int)[central]
    dx = x[mm] - gdf["x_wcs"].to_numpy(dtype=np.float64)[gg]
    dy = y[mm] - gdf["y_wcs"].to_numpy(dtype=np.float64)[gg]
    g_mag_c = pd.to_numeric(gdf.get("g_mag"), errors="coerce").to_numpy(dtype=np.float64)[gg]
    # Center model build: allow higher residuals in the first round (strict 1px cuts are too aggressive).
    # Rule: never drop bright anchors (g_mag < 10) for RMS — keep them even if resid > 3px.
    mdl_x0, mdl_y0 = _fit_poly_model(x[mm], y[mm], dx, dy)
    try:
        rx = dx - _eval_poly(mdl_x0, x[mm], y[mm])
        ry = dy - _eval_poly(mdl_y0, x[mm], y[mm])
        r = np.hypot(rx, ry)
        bright_anchor = np.isfinite(g_mag_c) & (g_mag_c < float(_BRIGHT_MAG_ANCHOR))
        keep = np.isfinite(r) & ((r <= 3.0) | bright_anchor)
        if int(np.count_nonzero(keep)) >= max(12, int(0.5 * len(mm))):
            mdl_x, mdl_y = _fit_poly_model(x[mm][keep], y[mm][keep], dx[keep], dy[keep])
            log_event(
                f"Astrometry optimizer: model built from {int(np.count_nonzero(keep))}/{len(mm)} central stars "
                f"(resid<=3px OR g_mag<{_BRIGHT_MAG_ANCHOR})."
            )
        else:
            mdl_x, mdl_y = mdl_x0, mdl_y0
            log_event(f"Astrometry optimizer: model built from {len(mm)} central matched stars (no resid clip).")
    except Exception:  # noqa: BLE001
        mdl_x, mdl_y = mdl_x0, mdl_y0
        log_event(f"Astrometry optimizer: model built from {len(mm)} central matched stars (no resid clip).")

    if ra_col is None or de_col is None:
        raise ValueError("Optimizer needs RA/Dec columns (ra_deg/dec_deg or aliases).")
    dcoo = SkyCoord(
        ra=pd.to_numeric(df.get(ra_col), errors="coerce").to_numpy(dtype=np.float64) * u.deg,
        dec=pd.to_numeric(df.get(de_col), errors="coerce").to_numpy(dtype=np.float64) * u.deg,
        frame="icrs",
    )
    used_ids = set([_norm_id(s) for s in cid.tolist() if _norm_id(s)])
    g_ra = pd.to_numeric(gdf["ra"], errors="coerce").to_numpy(dtype=np.float64)
    g_de = pd.to_numeric(gdf["dec"], errors="coerce").to_numpy(dtype=np.float64)
    g_mag = pd.to_numeric(gdf.get("g_mag"), errors="coerce").to_numpy(dtype=np.float64)
    g_bprp = pd.to_numeric(gdf.get("bp_rp"), errors="coerce").to_numpy(dtype=np.float64)
    gx0 = gdf["x_wcs"].to_numpy(dtype=np.float64)
    gy0 = gdf["y_wcs"].to_numpy(dtype=np.float64)
    gx_corr = gx0 + _eval_poly(mdl_x, gx0, gy0)
    gy_corr = gy0 + _eval_poly(mdl_y, gx0, gy0)
    # Initial jump (before any Grip rematch): median dx/dy over ALL detection↔nearest-Gaia candidates.
    # Without this global shift the center hole persists. Kept additive across displacement refits.
    mdx0_pass0 = 0.0
    mdy0_pass0 = 0.0
    try:
        idx_all = np.where(np.isfinite(x) & np.isfinite(y))[0]
        if idx_all.size > 0:
            gcoo_all = SkyCoord(ra=g_ra * u.deg, dec=g_de * u.deg, frame="icrs")
            idx_nn, sep_nn, _ = dcoo[idx_all].match_to_catalog_sky(gcoo_all)
            keep0 = np.isfinite(sep_nn.arcsecond) & (sep_nn.arcsecond <= 180.0)
            if int(np.count_nonzero(keep0)) >= 10:
                i_det = idx_all[keep0]
                i_cat = np.asarray(idx_nn[keep0], dtype=int)
                dx0 = x[i_det] - gx_corr[i_cat]
                dy0 = y[i_det] - gy_corr[i_cat]
                mdx0 = float(np.nanmedian(dx0))
                mdy0 = float(np.nanmedian(dy0))
                if math.isfinite(mdx0) and math.isfinite(mdy0):
                    mdx0_pass0 = float(mdx0)
                    mdy0_pass0 = float(mdy0)
                    gx_corr = gx_corr + mdx0_pass0
                    gy_corr = gy_corr + mdy0_pass0
                    log_event(
                        f"Astrometry optimizer initial jump: median dx={mdx0:.2f}px dy={mdy0:.2f}px "
                        f"from {int(np.count_nonzero(keep0))} candidates (NN sky <=180\")."
                    )
    except Exception as exc:  # noqa: BLE001
        log_event(f"Astrometry optimizer initial jump skipped: {exc!s}")
    log_event("Astrometry optimizer: adaptive rematch radius enabled (center=5.0\", edge>800px => 10–15\").")
    cid_series = df.get("catalog_id", pd.Series([""] * len(df), index=df.index)).copy()

    def _is_unmatched_row(row_idx: int) -> bool:
        return _norm_id(cid_series.iloc[row_idx]) == ""

    def _write_match(row_idx: int, picked_idx: int, sep_arcsec: float) -> None:
        sid = _norm_id(gdf.iloc[picked_idx]["source_id"])
        used_ids.add(sid)
        cid_series.iloc[row_idx] = sid
        df.at[row_idx, "catalog"] = "GAIA_DR3"
        df.at[row_idx, "catalog_id"] = sid
        gm = float(g_mag[picked_idx]) if np.isfinite(g_mag[picked_idx]) else np.nan
        df.at[row_idx, "mag"] = gm
        df.at[row_idx, "catalog_mag"] = gm
        df.at[row_idx, "phot_g_mean_mag"] = gm
        _bpr = float(g_bprp[picked_idx]) if np.isfinite(g_bprp[picked_idx]) else np.nan
        df.at[row_idx, "b_v"] = _bpr
        df.at[row_idx, "bp_rp"] = _bpr
        df.at[row_idx, "match_sep_arcsec"] = float(sep_arcsec)

    def _refit_displacement_from_all_matches() -> None:
        """Re-fit pixel-space polynomial from every star that already has a catalog_id (The Grip)."""
        nonlocal mdl_x, mdl_y, gx_corr, gy_corr
        mi: list[int] = []
        gj: list[int] = []
        for ii in range(len(df)):
            sid = _norm_id(cid_series.iloc[ii])
            if not sid or sid not in gmap:
                continue
            if not (np.isfinite(x[ii]) and np.isfinite(y[ii])):
                continue
            mi.append(ii)
            gj.append(int(gmap[sid]))
        if len(mi) < 20:
            return
        mi_a = np.asarray(mi, dtype=int)
        gj_a = np.asarray(gj, dtype=int)
        xs_ = x[mi_a]
        ys_ = y[mi_a]
        gxj = gx0[gj_a]
        gyj = gy0[gj_a]
        dx_ = xs_ - gxj
        dy_ = ys_ - gyj
        mdl_x, mdl_y = _fit_poly_model(xs_, ys_, dx_, dy_)
        gx_corr = gx0 + _eval_poly(mdl_x, gx0, gy0) + float(mdx0_pass0)
        gy_corr = gy0 + _eval_poly(mdl_y, gx0, gy0) + float(mdy0_pass0)

    def _match_bright_anchor_stars() -> int:
        """Bright anchors (g_mag < 10.5): match within pixel tolerance even at high RMS."""
        added_b = 0
        search_radius_arcsec = 45.0
        for i in range(len(df)):
            if not _is_unmatched_row(i):
                continue
            if not (np.isfinite(x[i]) and np.isfinite(y[i])):
                continue
            sep_arc = dcoo[i].separation(SkyCoord(ra=g_ra * u.deg, dec=g_de * u.deg, frame="icrs")).arcsecond
            cand = np.where(np.isfinite(sep_arc) & (sep_arc <= float(search_radius_arcsec)))[0]
            if cand.size == 0:
                continue
            dpx = np.hypot(gx_corr[cand] - x[i], gy_corr[cand] - y[i])
            order = np.argsort(dpx)
            picked = None
            for j_idx in order:
                j = int(cand[j_idx])
                if not (np.isfinite(g_mag[j]) and float(g_mag[j]) < float(_BRIGHT_MAG_ANCHOR)):
                    continue
                if float(dpx[j_idx]) > float(_BRIGHT_MAX_MATCH_DPX):
                    break
                sid = _norm_id(gdf.iloc[int(j)]["source_id"])
                if sid in used_ids:
                    continue
                picked = int(j)
                break
            if picked is None:
                continue
            _write_match(i, int(picked), float(sep_arc[picked]))
            added_b += 1
        return int(added_b)

    # Two-pass rematch per grip iteration:
    # 1) conservative center-oriented pass (stable, low false positives)
    # 2) adaptive edge pass with larger radius in corners (up to >=10")
    def _rematch_pass(*, adaptive_edges: bool, center_base_arcsec: float) -> int:
        added = 0
        base = float(max(0.5, center_base_arcsec))
        for i in range(len(df)):
            if not _is_unmatched_row(i):
                continue
            if not (np.isfinite(x[i]) and np.isfinite(y[i])):
                continue
            r = float(np.hypot(x[i] - cx, y[i] - cy))
            if adaptive_edges:
                # Required adaptive form: R = R_center + dist_from_center_px / factor
                # Choose factor so corners reach at least ~10".
                factor = max(120.0, rmax / 8.5)
                rad_arcsec = base + (r / factor)
                if r > 800.0:
                    rr = min(1.0, max(0.0, (r - 800.0) / max(1.0, rmax - 800.0)))
                    rad_arcsec = max(rad_arcsec, 10.0 + 5.0 * rr)
                rad_arcsec = max(rad_arcsec, 10.0 if r >= 0.90 * rmax else rad_arcsec)
            else:
                # First pass: center lock with tighter envelope.
                rad_arcsec = base + min(2.5, r / max(300.0, rmax / 5.0))
            sep_arc = dcoo[i].separation(SkyCoord(ra=g_ra * u.deg, dec=g_de * u.deg, frame="icrs")).arcsecond
            cand = np.where(np.isfinite(sep_arc) & (sep_arc <= rad_arcsec))[0]
            if cand.size == 0:
                continue
            dpx = np.hypot(gx_corr[cand] - x[i], gy_corr[cand] - y[i])
            order = np.argsort(dpx)
            picked = None
            for j in cand[order]:
                sid = _norm_id(gdf.iloc[int(j)]["source_id"])
                if sid in used_ids:
                    continue
                picked = int(j)
                break
            if picked is None:
                continue
            _write_match(i, int(picked), float(sep_arc[picked]))
            added += 1
        return int(added)

    def _refit_wcs_sip5_once(*, tag: str) -> dict[str, Any]:
        """SIP WCS (order ``_SIP_MIN_ORDER``) from current matches; reproject Gaia to pixels and refit displacement poly."""
        nonlocal gx0, gy0, gx_corr, gy_corr, mdl_x, mdl_y
        try:
            import astropy.units as _u
            from astropy.coordinates import SkyCoord as _SkyCoord
            from astropy.wcs import WCS as _WCS
            from astropy.wcs.utils import fit_wcs_from_points as _fit_wcs_from_points

            from vyvar_platesolver import _fit_sip_on_matches  # type: ignore
            from utils import strip_celestial_wcs_keys

            cid2 = df.get("catalog_id", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
            okm = cid2.ne("") & pd.to_numeric(df.get("ra_deg"), errors="coerce").notna() & pd.to_numeric(
                df.get("dec_deg"), errors="coerce"
            ).notna()
            okm &= pd.to_numeric(df.get("x"), errors="coerce").notna() & pd.to_numeric(df.get("y"), errors="coerce").notna()
            mdf = df.loc[okm].copy()
            n_pairs = int(len(mdf))
            if n_pairs < 25:
                log_event(f"Astrometry optimizer {tag}: WCS refit skipped (need >=25 pairs, got {n_pairs}).")
                return {}
            x_obs = pd.to_numeric(mdf["x"], errors="coerce").to_numpy(dtype=np.float64)
            y_obs = pd.to_numeric(mdf["y"], errors="coerce").to_numpy(dtype=np.float64)
            ra_obs = pd.to_numeric(mdf["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
            de_obs = pd.to_numeric(mdf["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
            world = _SkyCoord(ra=ra_obs * _u.deg, dec=de_obs * _u.deg, frame="icrs")
            sip_order = max(2, int(_SIP_MIN_ORDER))
            w_lin = _fit_wcs_from_points((x_obs, y_obs), world, projection="TAN")
            w_sip, meta = _fit_sip_on_matches(
                w_lin,
                x_obs,
                y_obs,
                world,
                max_order=int(sip_order),
                sip_force_rms_guard_ratio=sip_force_rms_guard_ratio,
            )
            w_out = w_sip if w_sip is not None else w_lin
            wh = w_out.to_header(relax=True)
            with fits.open(fits_path, mode="update", memmap=False) as hdul2:
                hh = hdul2[0].header
                strip_celestial_wcs_keys(hh)
                for k in wh:
                    if k in ("", "COMMENT", "HISTORY", "SIMPLE", "BITPIX", "NAXIS", "EXTEND"):
                        continue
                    if k.startswith("NAXIS") and k != "NAXIS":
                        continue
                    try:
                        hh[k] = wh[k]
                    except Exception:  # noqa: BLE001
                        pass
                hh["VY_SIPRF"] = (True, f"WCS refit ({tag}, SIP{int(sip_order)})")
                if meta.get("rms_linear_px") is not None and meta.get("rms_sip_px") is not None:
                    hh.add_history(
                        f"VYVAR: {tag} SIP{int(sip_order)} rms_px lin={float(meta['rms_linear_px']):.3f} -> sip={float(meta['rms_sip_px']):.3f}"
                    )
                hdul2.flush()
            log_event(
                f"Astrometry optimizer {tag}: SIP{int(sip_order)} n_pairs={n_pairs} "
                f"(rms_lin={meta.get('rms_linear_px')}, rms_sip={meta.get('rms_sip_px')})"
            )
            with fits.open(fits_path, memmap=False) as _h:
                w_new = _WCS(_h[0].header)
            if not getattr(w_new, "has_celestial", False):
                return {}
            gxn, gyn = w_new.world_to_pixel(gcoo)
            gx0 = np.asarray(gxn, dtype=np.float64)
            gy0 = np.asarray(gyn, dtype=np.float64)
            gdf["x_wcs"] = gx0
            gdf["y_wcs"] = gy0
            _refit_displacement_from_all_matches()
            return dict(meta)
        except Exception as exc:  # noqa: BLE001
            log_event(f"Astrometry optimizer {tag}: WCS refit skipped: {exc!s}")
            return {}

    # The Grip: 5 iterations; Pass 0 uses 5" center gate, then tighten each iteration after WCS refit.
    grip_sum_a = 0
    grip_sum_b = 0
    grip_sum_c = 0
    grip_no_progress = 0
    _grip_regression_break = False
    with fits.open(fits_path, memmap=False) as _hg0:
        _best_grip_hdr = _hg0[0].header.copy()
    _best_grip_rms_lin = float("inf")
    for grip_it in range(int(_GRIP_ITERATIONS)):
        center_r = max(0.5, float(_PASS0_CENTER_ARCSEC) - float(grip_it) * 1.0)
        n_a = _rematch_pass(adaptive_edges=False, center_base_arcsec=center_r)
        n_b = _rematch_pass(adaptive_edges=True, center_base_arcsec=center_r)
        n_c = _match_bright_anchor_stars()
        _refit_displacement_from_all_matches()
        meta_grip = _refit_wcs_sip5_once(tag=f"Grip{grip_it + 1}")
        _rms_lin = float(meta_grip.get("rms_linear_px") or 999.0)
        _rms_sip = float(meta_grip.get("rms_sip_px") or 999.0)
        _guard = float(sip_force_rms_guard_ratio) if sip_force_rms_guard_ratio is not None else None
        if _guard is not None and _rms_lin > 0 and _rms_sip > _rms_lin * _guard:
            log_event(
                f"VYVAR optimizer: Grip{grip_it + 1} SIP regression "
                f"(lin={_rms_lin:.3f} sip={_rms_sip:.3f} ratio={_rms_sip / _rms_lin:.3f}) "
                f"— stopping Grip loop, keeping best linear WCS."
            )
            _grip_regression_break = True
            break
        if meta_grip and math.isfinite(_rms_lin) and _rms_lin < _best_grip_rms_lin:
            with fits.open(fits_path, memmap=False) as _hcur:
                _best_grip_hdr = _hcur[0].header.copy()
            _best_grip_rms_lin = _rms_lin
        grip_sum_a += n_a
        grip_sum_b += n_b
        grip_sum_c += n_c
        log_event(
            f"Astrometry optimizer Grip {grip_it + 1}/{int(_GRIP_ITERATIONS)}: "
            f"center_r={center_r:.2f}\" passA=+{n_a} passB=+{n_b} bright_anchors=+{n_c} (displacement+SIP)."
        )
        if int(n_a) == 0 and int(n_b) == 0 and int(n_c) == 0:
            grip_no_progress += 1
            if grip_no_progress >= 2:
                log_event(
                    "Astrometry optimizer: predčasný koniec Grip (2× po sebe 0 nových zhôd) — šetrim čas."
                )
                break
        else:
            grip_no_progress = 0
    if _grip_regression_break:
        try:
            w_best = WCS(_best_grip_hdr)
            wh_r = w_best.to_header(relax=True)
        except Exception as exc:  # noqa: BLE001
            log_event(f"Astrometry optimizer: Grip regression restore skipped (bad best header): {exc!s}")
            wh_r = None
        if wh_r is not None:
            try:
                with fits.open(fits_path, mode="update", memmap=False) as hdul_r:
                    hh = hdul_r[0].header
                    strip_celestial_wcs_keys(hh)
                    for k in wh_r:
                        if k in ("", "COMMENT", "HISTORY", "SIMPLE", "BITPIX", "NAXIS", "EXTEND"):
                            continue
                        if k.startswith("NAXIS") and k != "NAXIS":
                            continue
                        try:
                            hh[k] = wh_r[k]
                        except Exception:  # noqa: BLE001
                            pass
                    if "VY_SIPRF" in _best_grip_hdr:
                        try:
                            hh["VY_SIPRF"] = _best_grip_hdr["VY_SIPRF"]
                        except Exception:  # noqa: BLE001
                            pass
                    hdul_r.flush()
                with fits.open(fits_path, memmap=False) as _hf:
                    w_rf = WCS(_hf[0].header)
                if getattr(w_rf, "has_celestial", False):
                    gxn, gyn = w_rf.world_to_pixel(gcoo)
                    gx0 = np.asarray(gxn, dtype=np.float64)
                    gy0 = np.asarray(gyn, dtype=np.float64)
                    gdf["x_wcs"] = gx0
                    gdf["y_wcs"] = gy0
                    _refit_displacement_from_all_matches()
            except Exception as exc:  # noqa: BLE001
                log_event(f"Astrometry optimizer: Grip regression FITS restore failed: {exc!s}")
    n_pass1, n_pass2, n_bright_grip = int(grip_sum_a), int(grip_sum_b), int(grip_sum_c)
    do_parity_flip = False
    # Orientation sanity check: mirrored sky coords for bright unmatched corner stars → optional WCS parity fix.
    try:
        flux_col = pd.to_numeric(df.get("flux"), errors="coerce") if "flux" in df.columns else pd.Series(np.nan, index=df.index)
        cid_dbg = df.get("catalog_id", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
        r_all = np.hypot(x - cx, y - cy)
        corner_mask = (
            cid_dbg.eq("")
            & np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(r_all)
            & (
                (x < 220.0)
                | (x > float(wpx) - 220.0)
                | (y < 180.0)
                | (y > float(h) - 180.0)
                | (r_all > 900.0)
            )
        )
        cand_dbg = np.where(corner_mask.to_numpy() if hasattr(corner_mask, "to_numpy") else np.asarray(corner_mask))[0]
        if cand_dbg.size > 0:
            if flux_col.notna().any():
                fvals = flux_col.to_numpy(dtype=np.float64)
                cand_dbg = cand_dbg[np.argsort(-np.nan_to_num(fvals[cand_dbg], nan=-np.inf))]
            cand_dbg = cand_dbg[:10]
            if cand_dbg.size > 0 and np.any(np.isfinite(g_ra)) and np.any(np.isfinite(g_de)):
                ra0 = float(np.nanmedian(g_ra))
                de0 = float(np.nanmedian(g_de))
                gcoo_dbg = SkyCoord(ra=g_ra * u.deg, dec=g_de * u.deg, frame="icrs")
                d_dbg = dcoo[cand_dbg]
                _idx0, sep_base, _ = d_dbg.match_to_catalog_sky(gcoo_dbg)
                base_rate = float(np.mean(sep_base.arcsecond <= 25.0))
                m_ra = 2.0 * ra0 - d_dbg.ra.deg
                m_de = 2.0 * de0 - d_dbg.dec.deg
                mcoo = SkyCoord(ra=m_ra * u.deg, dec=np.clip(m_de, -89.999999, 89.999999) * u.deg, frame="icrs")
                _idx1, sep_mir, _ = mcoo.match_to_catalog_sky(gcoo_dbg)
                mir_rate = float(np.mean(sep_mir.arcsecond <= 25.0))
                log_event(
                    f"Astrometry optimizer orientation test: corner unmatched={int(cand_dbg.size)} "
                    f"base@25\"={base_rate*100.0:.1f}% mirror@25\"={mir_rate*100.0:.1f}%."
                )
                # Prísnejšie: slabý base_rate + výraznejší mirror, aby sa neflipovalo pri šume (648px jump bol falošný pozitív).
                if mir_rate > base_rate + 0.25 and (base_rate < 0.38 or mir_rate >= 0.42):
                    do_parity_flip = True
                    log_event("Astrometry optimizer orientation warning: mirrored RA/Dec matches markedly better (possible WCS parity issue).")
                    if mirror_orientation_extra_log:
                        log_event(
                            "MASTERSTAR: skontrolujte parity / zrkadlenie v optike alebo znovu plate solve s konzistentným RA/Dec hintom."
                        )
    except Exception as exc:  # noqa: BLE001
        log_event(f"Astrometry optimizer orientation test skipped: {exc!s}")

    if do_parity_flip and _optimizer_parity_flip_already_in_fits(fits_path):
        log_event(
            "Astrometry optimizer: parity flip preskočený (VY_OPTPF v FITS — už jedna úprava PC na tomto MASTERSTAR)."
        )
        do_parity_flip = False

    if do_parity_flip:
        _pf_snap = (
            np.asarray(gx0, dtype=np.float64, copy=True),
            np.asarray(gy0, dtype=np.float64, copy=True),
            np.array(mdl_x, dtype=np.float64, copy=True),
            np.array(mdl_y, dtype=np.float64, copy=True),
            np.asarray(gx_corr, dtype=np.float64, copy=True),
            np.asarray(gy_corr, dtype=np.float64, copy=True),
            float(mdx0_pass0),
            float(mdy0_pass0),
        )
        if not _apply_wcs_pc_parity_flip_to_primary(fits_path, set_vy_optpf=True):
            log_event("Astrometry optimizer: zápis parity flip na disk zlyhal.")
            do_parity_flip = False

    if do_parity_flip:
        log_event(
            "Astrometry optimizer: aplikovaný WCS parity flip na MASTERSTAR.fits — obnova Gaia→px a korekcia len zo spárovaných."
        )
        try:
            with fits.open(fits_path, memmap=False) as _hf:
                w_pf = WCS(_hf[0].header)
            if getattr(w_pf, "has_celestial", False):
                gxn, gyn = w_pf.world_to_pixel(gcoo)
                gx0 = np.asarray(gxn, dtype=np.float64)
                gy0 = np.asarray(gyn, dtype=np.float64)
                gdf["x_wcs"] = gx0
                gdf["y_wcs"] = gy0
                mi_a = np.asarray(matched_idx, dtype=int)
                gm_a = np.asarray(gm_idx, dtype=int)
                rr_pf = np.hypot(x[mi_a] - cx, y[mi_a] - cy)
                rmax_pf = float(max(1.0, np.nanmax(rr_pf)))
                central_pf = rr_pf <= (0.45 * rmax_pf)
                if int(np.count_nonzero(central_pf)) < 6:
                    central_pf = np.ones_like(rr_pf, dtype=bool)
                mm_pf = mi_a[central_pf]
                gg_pf = gm_a[central_pf]
                dx_pf = x[mm_pf] - gx0[gg_pf]
                dy_pf = y[mm_pf] - gy0[gg_pf]
                mdl_x0_pf, mdl_y0_pf = _fit_poly_model(x[mm_pf], y[mm_pf], dx_pf, dy_pf)
                mdl_x, mdl_y = mdl_x0_pf, mdl_y0_pf
                try:
                    rx_p = dx_pf - _eval_poly(mdl_x0_pf, x[mm_pf], y[mm_pf])
                    ry_p = dy_pf - _eval_poly(mdl_y0_pf, x[mm_pf], y[mm_pf])
                    r_p = np.hypot(rx_p, ry_p)
                    g_mag_pf = pd.to_numeric(gdf.get("g_mag"), errors="coerce").to_numpy(dtype=np.float64)[gg_pf]
                    bright_pf = np.isfinite(g_mag_pf) & (g_mag_pf < float(_BRIGHT_MAG_ANCHOR))
                    keep_pf = np.isfinite(r_p) & ((r_p <= 3.0) | bright_pf)
                    if int(np.count_nonzero(keep_pf)) >= max(12, int(0.5 * len(mm_pf))):
                        mdl_x, mdl_y = _fit_poly_model(
                            x[mm_pf][keep_pf], y[mm_pf][keep_pf], dx_pf[keep_pf], dy_pf[keep_pf]
                        )
                except Exception:  # noqa: BLE001
                    pass
                gx_base = gx0 + _eval_poly(mdl_x, gx0, gy0)
                gy_base = gy0 + _eval_poly(mdl_y, gx0, gy0)
                mdx_pf = 0.0
                mdy_pf = 0.0
                dxm: list[float] = []
                dym: list[float] = []
                for ii in range(len(df)):
                    sid = _norm_id(cid_series.iloc[ii])
                    if not sid or sid not in gmap:
                        continue
                    jj = int(gmap[sid])
                    if not (np.isfinite(x[ii]) and np.isfinite(y[ii])):
                        continue
                    if not (np.isfinite(gx_base[jj]) and np.isfinite(gy_base[jj])):
                        continue
                    dxm.append(float(x[ii] - gx_base[jj]))
                    dym.append(float(y[ii] - gy_base[jj]))
                n_m = int(len(dxm))
                if n_m >= _POST_FLIP_MATCHED_MIN:
                    mdx_pf = float(np.nanmedian(np.asarray(dxm, dtype=np.float64)))
                    mdy_pf = float(np.nanmedian(np.asarray(dym, dtype=np.float64)))
                jmp_ok = (
                    n_m >= _POST_FLIP_MATCHED_MIN
                    and math.isfinite(mdx_pf)
                    and math.isfinite(mdy_pf)
                    and abs(mdx_pf) <= float(_POST_FLIP_JUMP_MAX_ABS_PX)
                    and abs(mdy_pf) <= float(_POST_FLIP_JUMP_MAX_ABS_PX)
                    and math.hypot(mdx_pf, mdy_pf) <= float(_POST_FLIP_JUMP_MAX_HYPOT_PX)
                )
                if not jmp_ok:
                    log_event(
                        f"Astrometry optimizer post-flip: matched-only offset neakceptovateľný "
                        f"(n={n_m}, dx={mdx_pf:.2f}, dy={mdy_pf:.2f} px; "
                        f"max |.|≤{_POST_FLIP_JUMP_MAX_ABS_PX}, hypot≤{_POST_FLIP_JUMP_MAX_HYPOT_PX}) — REVERT parity."
                    )
                    _rev_ok = _apply_wcs_pc_parity_flip_to_primary(fits_path, set_vy_optpf=False)
                    if not _rev_ok:
                        log_event(
                            "Astrometry optimizer: varovanie — revert parity na disk zlyhal; obnovujem stav pred flipom v RAM."
                        )
                    (
                        gx0,
                        gy0,
                        mdl_x,
                        mdl_y,
                        gx_corr,
                        gy_corr,
                        mdx0_pass0,
                        mdy0_pass0,
                    ) = _pf_snap
                    gdf["x_wcs"] = gx0
                    gdf["y_wcs"] = gy0
                else:
                    mdx0_pass0 = float(mdx_pf)
                    mdy0_pass0 = float(mdy_pf)
                    gx_corr = gx_base + mdx0_pass0
                    gy_corr = gy_base + mdy0_pass0
                    log_event(
                        f"Astrometry optimizer post-flip offset (medián z {n_m} spárovaných): "
                        f"dx={mdx_pf:.2f}px dy={mdy_pf:.2f}px."
                    )
                    _refit_displacement_from_all_matches()
                    ex_a = _rematch_pass(adaptive_edges=False, center_base_arcsec=float(_PASS0_CENTER_ARCSEC))
                    ex_b = _rematch_pass(adaptive_edges=True, center_base_arcsec=float(_PASS0_CENTER_ARCSEC))
                    ex_c = _match_bright_anchor_stars()
                    _refit_displacement_from_all_matches()
                    log_event(
                        f"Astrometry optimizer post-parity-flip re-match: passA=+{ex_a} passB=+{ex_b} bright=+{ex_c}."
                    )
        except Exception as exc_pf2:  # noqa: BLE001
            log_event(f"Astrometry optimizer post-parity-flip rebuild failed: {exc_pf2!s}")

    # Final edge-only pass after displacement / pre-SIP correction: no NAXIS clipping before matching.
    # Keep stars even when projected outside nominal image bounds.
    n_pass3 = 0
    n_pass3_iter = 2
    log_event(
        f"Astrometry optimizer pass3 debug: rematching edge detections with {int(len(gdf))} available catalog stars."
    )
    for _it in range(n_pass3_iter):
        added_iter = 0
        for i in range(len(df)):
            if not _is_unmatched_row(i):
                continue
            if not (np.isfinite(x[i]) and np.isfinite(y[i])):
                continue
            r_det = float(np.hypot(x[i] - cx, y[i] - cy))
            if not (float(x[i]) < 200.0 or float(x[i]) > 1800.0 or r_det > 800.0):
                continue
            # Adaptive pass3 radius: corners/edges get wider sky gate.
            search_radius = 60.0 if r_det > 900.0 else (40.0 if r_det > 800.0 else 25.0)
            sep_arc = dcoo[i].separation(SkyCoord(ra=g_ra * u.deg, dec=g_de * u.deg, frame="icrs")).arcsecond
            cand = np.where(np.isfinite(sep_arc) & (sep_arc <= float(search_radius)))[0]
            if cand.size == 0:
                continue
            dpx = np.hypot(gx_corr[cand] - x[i], gy_corr[cand] - y[i])
            order = np.argsort(dpx)
            picked = None
            for j in cand[order]:
                sid = _norm_id(gdf.iloc[int(j)]["source_id"])
                if sid in used_ids:
                    continue
                picked = int(j)
                break
            if picked is None:
                continue
            _write_match(i, int(picked), float(sep_arc[picked]))
            n_pass3 += 1
            added_iter += 1
        log_event(f"Astrometry optimizer pass3 iteration {_it + 1}/{n_pass3_iter}: +{added_iter} matches.")
    log_event(f"DEBUG: pass3 actually SAVED [{int(n_pass3)}] new matches to dataframe.")
    log_event(
        f"Astrometry optimizer re-match: Grip(passA+passB+bright) totals +{n_pass1}/+{n_pass2}/+{n_bright_grip}, "
        f"pass3(edge@25/60\") +{n_pass3}."
    )

    # Keep all detections (including DAO-only) for downstream discovery tests.
    # We still compute edge diagnostics, but do not drop rows from CSV.
    x_col = pd.to_numeric(df.get("x"), errors="coerce")
    y_col = pd.to_numeric(df.get("y"), errors="coerce")
    edge_safe = (
        x_col.notna()
        & y_col.notna()
        & (x_col >= 10.0)
        & (x_col < float(wpx) - 10.0)
        & (y_col >= 10.0)
        & (y_col < float(h) - 10.0)
    )
    df["edge_safe_10px"] = edge_safe.astype(bool)
    n_edge_unsafe = int((~edge_safe).sum())
    if n_edge_unsafe > 0:
        log_event(
            f"Astrometry optimizer edge diagnostics: {n_edge_unsafe} stars outside 10px safe area "
            f"(kept for DAO-only/discovery analysis)."
        )

    # Final diagnostics-only block: keep all rows; annotate SNR/saturation for QA.
    df["catalog_id"] = df.get("catalog_id", pd.Series([""] * len(df))).fillna("").astype(str)
    is_gaia = df["catalog_id"].str.strip().ne("")
    _, bg_med, bg_std = sigma_clipped_stats(data[np.isfinite(data)], sigma=3.0, maxiters=5)
    bg_std = float(bg_std) if np.isfinite(bg_std) and float(bg_std) > 0 else float(np.nanstd(data[np.isfinite(data)]))
    if not math.isfinite(bg_std) or bg_std <= 0:
        bg_std = 1.0

    sat_ceiling = float(np.nanmax(data[np.isfinite(data)])) if np.any(np.isfinite(data)) else 65535.0
    sat_thr = 0.95 * sat_ceiling
    peak_col = pd.to_numeric(df.get("peak_max_adu"), errors="coerce") if "peak_max_adu" in df.columns else pd.Series(np.nan, index=df.index)
    sat_flag = (
        (df.get("likely_saturated", pd.Series(False, index=df.index)).fillna(False).astype(bool))
        | (peak_col.fillna(-np.inf) >= sat_thr)
    )

    flux_col = pd.to_numeric(df.get("flux"), errors="coerce") if "flux" in df.columns else pd.Series(np.nan, index=df.index)
    snr_ok = flux_col.fillna(0.0) >= (10.0 * float(bg_std))
    snr50_ok = flux_col.fillna(0.0) >= (50.0 * float(bg_std))
    bright_thr = float(np.nanpercentile(flux_col.dropna().to_numpy(), 99.0)) if flux_col.notna().any() else 10.0 * float(bg_std)
    is_discovery = (~is_gaia) & snr50_ok & (flux_col.fillna(0.0) >= bright_thr)
    df["snr10_ok"] = snr_ok.astype(bool)
    df["snr50_ok"] = snr50_ok.astype(bool)
    df["is_discovery_candidate"] = is_discovery.astype(bool)
    df["is_saturated_flagged"] = sat_flag.astype(bool)
    n_red_noise = int((~is_gaia & ~snr50_ok).sum())
    log_event(
        f"Astrometry optimizer diagnostics: {n_red_noise} DAO-only stars below 50σ (kept in CSV by design)."
    )

    # Final SIP refit after pass3 edge recovery (Grip already ran SIP each iteration).
    _refit_wcs_sip5_once(tag="post-pass3")

    # Pass 4: final full-field rematch after SIP WCS refit (wide sky radius to fill gaps).
    try:
        with fits.open(fits_path, memmap=False) as _hd:
            w2 = WCS(_hd[0].header)
        if getattr(w2, "has_celestial", False):
            gcoo2 = SkyCoord(ra=g_ra * u.deg, dec=g_de * u.deg, frame="icrs")
            gx2, gy2 = w2.world_to_pixel(gcoo2)
            gx2 = np.asarray(gx2, dtype=np.float64)
            gy2 = np.asarray(gy2, dtype=np.float64)
            added4 = 0
            search_radius4 = 30.0  # arcsec (20–30")
            for i in range(len(df)):
                if not _is_unmatched_row(i):
                    continue
                if not (np.isfinite(x[i]) and np.isfinite(y[i])):
                    continue
                sep_arc = dcoo[i].separation(SkyCoord(ra=g_ra * u.deg, dec=g_de * u.deg, frame="icrs")).arcsecond
                cand = np.where(np.isfinite(sep_arc) & (sep_arc <= float(search_radius4)))[0]
                if cand.size == 0:
                    continue
                dpx = np.hypot(gx2[cand] - x[i], gy2[cand] - y[i])
                order = np.argsort(dpx)
                picked = None
                for j in cand[order]:
                    sid = _norm_id(gdf.iloc[int(j)]["source_id"])
                    if sid in used_ids:
                        continue
                    picked = int(j)
                    break
                if picked is None:
                    continue
                _write_match(i, int(picked), float(sep_arc[picked]))
                added4 += 1
            log_event(f"Astrometry optimizer pass4 full-field@30\": +{int(added4)} matches after SIP refit.")
    except Exception as exc:  # noqa: BLE001
        log_event(f"Astrometry optimizer pass4 skipped: {exc!s}")

    # Final global shift correction (post parity flip + post pass3/pass4):
    # If the remaining Gaia→pixel residual median is large, fix CRPIX and re-match once.
    try:
        sep_col2 = _first_existing_col(df, ["match_sep_arcsec", "sep_arcsec", "gaia_sep_arcsec", "MATCH_SEP_ARCSEC"])
        sep_s = pd.to_numeric(df.get(sep_col2), errors="coerce") if sep_col2 is not None else pd.Series(np.nan, index=df.index)
        cid_s = df.get("catalog_id", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
        good = cid_s.str.strip().ne("") & sep_s.notna() & (sep_s.astype(float) <= 5.0)
        idx_g = np.where(good.to_numpy() if hasattr(good, "to_numpy") else np.asarray(good))[0]
        if idx_g.size >= 20:
            with fits.open(fits_path, memmap=False) as _hdg:
                w_g = WCS(_hdg[0].header)
            if getattr(w_g, "has_celestial", False):
                # Map Gaia coords for each matched row via catalog_id -> gdf index.
                gcoo_all = SkyCoord(ra=g_ra * u.deg, dec=g_de * u.deg, frame="icrs")
                gxg, gyg = w_g.world_to_pixel(gcoo_all)
                gxg = np.asarray(gxg, dtype=np.float64)
                gyg = np.asarray(gyg, dtype=np.float64)

                dxs: list[float] = []
                dys: list[float] = []
                for ii in idx_g.tolist():
                    sid = _norm_id(cid_s.iloc[int(ii)])
                    if not sid or sid not in gmap:
                        continue
                    jj = int(gmap[sid])
                    if not (np.isfinite(x[int(ii)]) and np.isfinite(y[int(ii)])):
                        continue
                    if not (np.isfinite(gxg[jj]) and np.isfinite(gyg[jj])):
                        continue
                    dxs.append(float(x[int(ii)] - gxg[jj]))
                    dys.append(float(y[int(ii)] - gyg[jj]))
                if len(dxs) >= 15:
                    mdx = float(np.nanmedian(np.asarray(dxs, dtype=np.float64)))
                    mdy = float(np.nanmedian(np.asarray(dys, dtype=np.float64)))
                    if math.isfinite(mdx) and math.isfinite(mdy):
                        log_event(
                            f"Astrometry optimizer final shift probe: median dx={mdx:.2f}px dy={mdy:.2f}px "
                            f"from {len(dxs)} pairs (match_sep<=5\")."
                        )
                        if abs(mdx) > 2.0 or abs(mdy) > 2.0:
                            # Apply CRPIX correction (simple global translation in pixel space).
                            try:
                                with fits.open(fits_path, mode="update", memmap=False) as _hdu_up:
                                    hh = _hdu_up[0].header
                                    cr1 = float(hh.get("CRPIX1", 0.0) or 0.0)
                                    cr2 = float(hh.get("CRPIX2", 0.0) or 0.0)
                                    hh["CRPIX1"] = (cr1 + float(mdx), "VYVAR optimizer global shift correction")
                                    hh["CRPIX2"] = (cr2 + float(mdy), "VYVAR optimizer global shift correction")
                                    hh.add_history(
                                        f"VYVAR: optimizer global shift correction (CRPIX += [{mdx:.3f},{mdy:.3f}] px)"
                                    )
                                    _hdu_up.flush()
                                log_event(
                                    f"Astrometry optimizer global shift applied: CRPIX1+= {mdx:.2f}px CRPIX2+= {mdy:.2f}px — re-matching @15\"."
                                )
                            except Exception as exc:  # noqa: BLE001
                                log_event(f"Astrometry optimizer global shift write failed: {exc!s}")
                                raise

                            # One re-match pass (15") to refresh catalog assignments under corrected WCS.
                            used_ids = set([_norm_id(s) for s in cid_s.tolist() if _norm_id(s)])
                            idx_nn2, sep2d2, _ = dcoo.match_to_catalog_sky(gcoo_all)
                            sep_arc2 = np.asarray(sep2d2.arcsecond, dtype=np.float64)
                            rad15 = 15.0
                            added15 = 0
                            for i in range(len(df)):
                                if not (np.isfinite(x[i]) and np.isfinite(y[i])):
                                    continue
                                j = int(idx_nn2[i])
                                if not np.isfinite(sep_arc2[i]) or float(sep_arc2[i]) > rad15:
                                    continue
                                sid = _norm_id(gdf.iloc[j]["source_id"])
                                if not sid:
                                    continue
                                if sid in used_ids and _norm_id(cid_s.iloc[i]) != sid:
                                    continue
                                if _norm_id(cid_s.iloc[i]) == "":
                                    _write_match(i, int(j), float(sep_arc2[i]))
                                    used_ids.add(sid)
                                    added15 += 1
                                else:
                                    # Update separation for already-matched rows (leave ID).
                                    if sep_col2 is not None:
                                        df.at[i, "match_sep_arcsec"] = float(sep_arc2[i])
                            log_event(f"Astrometry optimizer global shift re-match@15\": +{int(added15)} new matches.")
    except Exception as exc:  # noqa: BLE001
        log_event(f"Astrometry optimizer global shift correction skipped: {exc!s}")

    if "phot_g_mean_mag" not in df.columns:
        if "mag" in df.columns:
            df["phot_g_mean_mag"] = pd.to_numeric(df["mag"], errors="coerce")
        else:
            df["phot_g_mean_mag"] = np.nan
    if "bp_rp" not in df.columns:
        if "b_v" in df.columns:
            df["bp_rp"] = pd.to_numeric(df["b_v"], errors="coerce")
        else:
            df["bp_rp"] = np.nan

    df.to_csv(out_path, index=False)
    n_all = int(len(df))
    n_match = int(df.get("catalog_id", pd.Series([""] * len(df))).fillna("").astype(str).str.strip().ne("").sum())
    log_event(f"Astrometry optimizer: wrote {out_path} ({n_match}/{n_all} catalog-matched).")
    return out_path
