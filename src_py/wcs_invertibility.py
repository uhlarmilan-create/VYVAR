"""WCS round-trip invertibility QA (F-428-WCS-INV FIX 1 + FIX 2 helpers)."""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.wcs import Sip
import warnings

logger = logging.getLogger(__name__)

DEFAULT_GRID = 9
DEFAULT_P99_THRESHOLD_PX = 0.2


def wcs_roundtrip_grid_residuals(
    wcs_obj: WCS,
    *,
    naxis1: int,
    naxis2: int,
    grid: int = DEFAULT_GRID,
) -> np.ndarray:
    """|pix - world2pix(pix2world(pix))| on a ``grid`` x ``grid`` frame sample (incl. corners)."""
    if grid < 2:
        grid = 2
    w = int(naxis1)
    h = int(naxis2)
    xs = np.linspace(0.5, max(0.5, w - 0.5), int(grid), dtype=np.float64)
    ys = np.linspace(0.5, max(0.5, h - 0.5), int(grid), dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    px = xx.ravel()
    py = yy.ravel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        ra, de = wcs_obj.all_pix2world(px, py, 0)
        px2, py2 = wcs_obj.all_world2pix(ra, de, 0, tolerance=1e-6, maxiter=50)
    return np.hypot(np.asarray(px2, float) - px, np.asarray(py2, float) - py)


def evaluate_wcs_roundtrip(
    wcs_obj: WCS,
    *,
    naxis1: int,
    naxis2: int,
    grid: int = DEFAULT_GRID,
    p99_threshold_px: float = DEFAULT_P99_THRESHOLD_PX,
) -> dict[str, Any]:
    """Evaluate round-trip residual stats; ``pass`` when p99 < threshold."""
    res = wcs_roundtrip_grid_residuals(wcs_obj, naxis1=naxis1, naxis2=naxis2, grid=grid)
    finite = res[np.isfinite(res)]
    if finite.size == 0:
        return {
            "pass": False,
            "wcs_roundtrip_p99_px": float("nan"),
            "wcs_roundtrip_p50_px": float("nan"),
            "wcs_roundtrip_max_px": float("nan"),
            "grid": int(grid),
            "n_samples": 0,
            "p99_threshold_px": float(p99_threshold_px),
        }
    p50 = float(np.percentile(finite, 50))
    p99 = float(np.percentile(finite, 99))
    mx = float(np.max(finite))
    return {
        "pass": bool(p99 < float(p99_threshold_px)),
        "wcs_roundtrip_p99_px": p99,
        "wcs_roundtrip_p50_px": p50,
        "wcs_roundtrip_max_px": mx,
        "grid": int(grid),
        "n_samples": int(finite.size),
        "p99_threshold_px": float(p99_threshold_px),
    }


def ensure_sip_inverse_coefficients(
    wcs_obj: WCS,
    *,
    fit_grid: int = 12,
    naxis1: int | None = None,
    naxis2: int | None = None,
) -> WCS:
    """Attach SIP inverse (AP/BP) consistent with forward A/B via grid fit (FIX 2)."""
    if wcs_obj.sip is None:
        return wcs_obj
    sip = wcs_obj.sip
    if sip.ap is not None and sip.bp is not None:
        return wcs_obj
    order = int(max(sip.a.shape[0], sip.b.shape[0])) - 1
    if order < 2:
        return wcs_obj
    crpix1, crpix2 = (float(sip.crpix[0]), float(sip.crpix[1]))
    if naxis1 is None or naxis2 is None:
        shape = getattr(wcs_obj, "pixel_shape", None)
        if shape and len(shape) >= 2:
            naxis1 = int(shape[0]) if naxis1 is None else int(naxis1)
            naxis2 = int(shape[1]) if naxis2 is None else int(naxis2)
        else:
            naxis1 = int(naxis1 or 512)
            naxis2 = int(naxis2 or 512)
    xs = np.linspace(0.5, max(0.5, naxis1 - 0.5), int(fit_grid), dtype=np.float64)
    ys = np.linspace(0.5, max(0.5, naxis2 - 0.5), int(fit_grid), dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    px = xx.ravel()
    py = yy.ravel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        ra, de = wcs_obj.all_pix2world(px, py, 0)
        px_back, py_back = wcs_obj.all_world2pix(ra, de, 0, tolerance=1e-6, maxiter=50)
    # Distortion-only offset from linear TAN at (px_back, py_back).
    w_lin = wcs_obj.deepcopy()
    w_lin.sip = None
    w_lin.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        ra_lin, de_lin = w_lin.all_pix2world(px_back, py_back, 0)
        px_lin, py_lin = w_lin.all_world2pix(ra_lin, de_lin, 0)
    du = px - px_lin
    dv = py - py_lin
    u = px_back - crpix1
    v = py_back - crpix2
    idxs = _sip_uv_indices(order)
    M = np.column_stack([(u**i) * (v**j) for i, j in idxs])
    reg = 1e-9 * np.eye(M.shape[1], dtype=np.float64)
    try:
        ap_coef = np.linalg.solve(M.T @ M + reg, M.T @ du)
        bp_coef = np.linalg.solve(M.T @ M + reg, M.T @ dv)
    except np.linalg.LinAlgError:
        ap_coef, _, _, _ = np.linalg.lstsq(M, du, rcond=None)
        bp_coef, _, _, _ = np.linalg.lstsq(M, dv, rcond=None)
    ap, bp = _fill_sip_mats(ap_coef, bp_coef, idxs, order)
    w_out = wcs_obj.deepcopy()
    w_out.sip = Sip(sip.a, sip.b, ap, bp, (crpix1, crpix2))
    return w_out


def _sip_uv_indices(max_order: int) -> list[tuple[int, int]]:
    idxs: list[tuple[int, int]] = []
    for i in range(max_order + 1):
        for j in range(max_order + 1):
            if i + j >= 2:
                idxs.append((i, j))
    return idxs


def _fill_sip_mats(
    ap_coef: np.ndarray,
    bp_coef: np.ndarray,
    idxs: list[tuple[int, int]],
    max_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    dim = max_order + 1
    ap = np.zeros((dim, dim), dtype=np.float64)
    bp = np.zeros((dim, dim), dtype=np.float64)
    for k, (i, j) in enumerate(idxs):
        ap[i, j] = float(ap_coef[k])
        bp[i, j] = float(bp_coef[k])
    return ap, bp


def apply_wcs_refit_with_invertibility_gate(
    *,
    fits_path: str,
    wcs_candidate: WCS,
    previous_header: fits.Header,
    context: str,
    naxis1: int,
    naxis2: int,
    log_fn: Any | None = None,
    p99_threshold_px: float = DEFAULT_P99_THRESHOLD_PX,
) -> tuple[bool, dict[str, Any]]:
    """Write ``wcs_candidate`` to FITS if round-trip gate passes; else restore ``previous_header``.

    Returns (applied, gate_meta).
    """
    _log = log_fn or logger.info
    w_try = ensure_sip_inverse_coefficients(wcs_candidate, naxis1=naxis1, naxis2=naxis2)
    gate = evaluate_wcs_roundtrip(w_try, naxis1=naxis1, naxis2=naxis2, p99_threshold_px=p99_threshold_px)
    if not gate.get("pass"):
        _log(
            f"WCS invertibility gate FAIL ({context}): p99={gate.get('wcs_roundtrip_p99_px'):.4f}px "
            f">= {p99_threshold_px}px — keeping previous WCS."
        )
        return False, gate
    from utils import strip_celestial_wcs_keys
    from wcs_header_io import copy_wcs_header_keys

    wh = w_try.to_header(relax=True)
    if copy_wcs_header_keys(fits.Header(), wh, context=f"{context} pre-write"):
        _log(f"WCS invertibility gate FAIL ({context}): header copy probe failed — keeping previous WCS.")
        gate["pass"] = False
        return False, gate
    with fits.open(fits_path, mode="update", memmap=False) as hdul:
        hh = hdul[0].header
        strip_celestial_wcs_keys(hh)
        failed = copy_wcs_header_keys(hh, wh, context=context)
        if failed:
            _log(f"WCS invertibility gate FAIL ({context}): core keys {failed} — keeping previous WCS.")
            gate["pass"] = False
            return False, gate
        hh["VY_WCSRT"] = (True, f"Round-trip p99={gate.get('wcs_roundtrip_p99_px', float('nan')):.4f}px")
        hdul.flush()
    _log(
        f"WCS invertibility gate PASS ({context}): p99={gate.get('wcs_roundtrip_p99_px'):.4f}px "
        f"(threshold {p99_threshold_px}px)."
    )
    return True, gate


def post_match_pixel_sep(
    x: float,
    y: float,
    gaia_ra: float,
    gaia_dec: float,
    wcs_obj: WCS,
    *,
    fwhm_px: float,
    warn_factor: float = 1.5,
    fail_factor: float = 3.0,
) -> tuple[str, float]:
    """Classify post-match identity in pixel space: ok | warn | fail."""
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(gaia_ra) and math.isfinite(gaia_dec)):
        return "fail", float("nan")
    try:
        gx, gy = wcs_obj.world_to_pixel_values(float(gaia_ra), float(gaia_dec))
    except Exception:  # noqa: BLE001
        return "fail", float("nan")
    dpx = float(math.hypot(float(gx) - float(x), float(gy) - float(y)))
    fwhm = max(0.5, float(fwhm_px))
    if dpx > fail_factor * fwhm:
        return "fail", dpx
    if dpx > warn_factor * fwhm:
        return "warn", dpx
    return "ok", dpx


def finalize_masterstar_sky_coords(
    df: Any,
    wcs_obj: WCS,
    *,
    gaia_db_path: str | None = None,
    log_fn: Any | None = None,
) -> Any:
    """Set ra_deg/dec_deg from Gaia (matched) or final WCS pix2world (unmatched); add coord_source."""
    import pandas as pd

    from gaia_catalog_id import normalize_gaia_source_id

    out = df.copy()
    n = len(out)
    x = pd.to_numeric(out.get("x"), errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(out.get("y"), errors="coerce").to_numpy(dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        ra_w, dec_w = wcs_obj.all_pix2world(x, y, 0)
    out["ra_deg"] = np.asarray(ra_w, dtype=np.float64)
    out["dec_deg"] = np.asarray(dec_w, dtype=np.float64)
    out["coord_source"] = np.array(["final_wcs"] * n, dtype=object)
    cid_raw = out.get("catalog_id", pd.Series([""] * n, index=out.index))
    cid_norm = cid_raw.map(normalize_gaia_source_id).astype(str).str.strip()
    matched = cid_norm.ne("") & cid_norm.ne("nan")
    if not bool(matched.any()):
        return out
    gaia_coords: dict[str, tuple[float, float]] = {}
    ids = sorted({c for c in cid_norm[matched].tolist() if c})
    gdb = str(gaia_db_path or "").strip()
    if gdb and ids:
        try:
            import sqlite3

            from repair_catalog_ids import _pick_gaia_table

            con = sqlite3.connect(gdb)
            table = _pick_gaia_table(con)
            for i0 in range(0, len(ids), 400):
                part = ids[i0 : i0 + 400]
                ph = ",".join("?" * len(part))
                for sid, ra, de in con.execute(
                    f"SELECT source_id, ra, dec FROM {table} WHERE source_id IN ({ph})",
                    part,
                ):
                    key = normalize_gaia_source_id(str(sid))
                    if key:
                        gaia_coords[key] = (float(ra), float(de))
            con.close()
        except Exception as exc:  # noqa: BLE001
            if log_fn:
                log_fn(f"finalize_masterstar_sky_coords: Gaia DB lookup skipped ({exc!s})")
    n_gaia = 0
    for idx in out.index[matched]:
        key = str(cid_norm.loc[idx])
        g = gaia_coords.get(key)
        if g is None:
            continue
        out.at[idx, "ra_deg"] = g[0]
        out.at[idx, "dec_deg"] = g[1]
        out.at[idx, "coord_source"] = "gaia_catalog"
        n_gaia += 1
    if log_fn and n_gaia:
        log_fn(
            f"finalize_masterstar_sky_coords: gaia_catalog={n_gaia} final_wcs={int(n - n_gaia)} "
            f"(matched rows with DB hit={n_gaia}/{int(matched.sum())})"
        )
    return out


def apply_post_match_identity_gate_df(
    df: Any,
    wcs_obj: WCS,
    *,
    gaia_ra_dec_by_cid: dict[str, tuple[float, float]],
    fwhm_px: float = 3.5,
    log_fn: Any | None = None,
) -> tuple[Any, dict[str, int]]:
    """Drop catalog assignments where sep(world2pix(Gaia), x/y) exceeds fail threshold."""
    import pandas as pd

    from gaia_catalog_id import normalize_gaia_source_id

    out = df.copy()
    counts: dict[str, int] = {"warn": 0, "fail": 0, "ok": 0}
    cid = out.get("catalog_id", pd.Series([""] * len(out), index=out.index)).map(normalize_gaia_source_id)
    for idx in out.index:
        key = str(cid.loc[idx] or "").strip()
        if not key or key == "nan":
            continue
        g = gaia_ra_dec_by_cid.get(key)
        if g is None:
            continue
        xv = float(pd.to_numeric(out.at[idx, "x"], errors="coerce"))
        yv = float(pd.to_numeric(out.at[idx, "y"], errors="coerce"))
        verdict, _dpx = post_match_pixel_sep(xv, yv, g[0], g[1], wcs_obj, fwhm_px=fwhm_px)
        counts[verdict] = counts.get(verdict, 0) + 1
        if verdict == "fail":
            for col, val in (
                ("catalog_id", ""),
                ("catalog", ""),
                ("match_sep_arcsec", float("nan")),
            ):
                if col in out.columns:
                    out.at[idx, col] = val
    if log_fn and (counts.get("warn") or counts.get("fail")):
        log_fn(
            f"post_match_identity_gate: ok={counts.get('ok', 0)} warn={counts.get('warn', 0)} "
            f"fail={counts.get('fail', 0)} (FWHM={fwhm_px:.2f}px)"
        )
    return out, counts


def evaluate_matched_world2pix_identity_px(
    df: Any,
    wcs_obj: WCS,
    *,
    gaia_ra_dec_by_cid: dict[str, tuple[float, float]] | None = None,
    gaia_db_path: str | None = None,
    log_fn: Any | None = None,
) -> dict[str, Any]:
    """Distribution of sep(world2pix(Gaia[cid]), x/y) over matched masterstar rows (px)."""
    import pandas as pd

    from gaia_catalog_id import normalize_gaia_source_id

    empty: dict[str, Any] = {
        "matched_world2pix_identity_n": 0,
        "matched_world2pix_identity_p50_px": float("nan"),
        "matched_world2pix_identity_p95_px": float("nan"),
        "matched_world2pix_identity_p99_px": float("nan"),
        "matched_world2pix_identity_max_px": float("nan"),
    }
    if df is None or getattr(df, "empty", True):
        return empty

    cid_raw = df.get("catalog_id", pd.Series([""] * len(df), index=df.index))
    cid_norm = cid_raw.map(normalize_gaia_source_id).astype(str).str.strip()
    matched = cid_norm.ne("") & cid_norm.ne("nan")
    if not bool(matched.any()):
        return empty

    coords = dict(gaia_ra_dec_by_cid or {})
    ids = sorted({c for c in cid_norm[matched].tolist() if c})
    gdb = str(gaia_db_path or "").strip()
    if gdb and ids:
        missing = [i for i in ids if i not in coords]
        if missing:
            try:
                import sqlite3

                from repair_catalog_ids import _pick_gaia_table

                con = sqlite3.connect(gdb)
                table = _pick_gaia_table(con)
                for i0 in range(0, len(missing), 400):
                    part = missing[i0 : i0 + 400]
                    ph = ",".join("?" * len(part))
                    for sid, ra, de in con.execute(
                        f"SELECT source_id, ra, dec FROM {table} WHERE source_id IN ({ph})",
                        part,
                    ):
                        key = normalize_gaia_source_id(str(sid))
                        if key:
                            coords[key] = (float(ra), float(de))
                con.close()
            except Exception as exc:  # noqa: BLE001
                if log_fn:
                    log_fn(f"matched_world2pix_identity_px: Gaia DB lookup skipped ({exc!s})")

    seps: list[float] = []
    for idx in df.index[matched]:
        key = str(cid_norm.loc[idx] or "").strip()
        g = coords.get(key)
        if g is None:
            continue
        xv = float(pd.to_numeric(df.at[idx, "x"], errors="coerce"))
        yv = float(pd.to_numeric(df.at[idx, "y"], errors="coerce"))
        _verdict, dpx = post_match_pixel_sep(xv, yv, g[0], g[1], wcs_obj, fwhm_px=3.5)
        if math.isfinite(dpx):
            seps.append(float(dpx))

    if not seps:
        return empty

    arr = np.asarray(seps, dtype=np.float64)
    out = {
        "matched_world2pix_identity_n": int(arr.size),
        "matched_world2pix_identity_p50_px": float(np.percentile(arr, 50)),
        "matched_world2pix_identity_p95_px": float(np.percentile(arr, 95)),
        "matched_world2pix_identity_p99_px": float(np.percentile(arr, 99)),
        "matched_world2pix_identity_max_px": float(np.max(arr)),
    }
    if log_fn:
        log_fn(
            "matched_world2pix_identity_px: "
            f"n={out['matched_world2pix_identity_n']} "
            f"p95={out['matched_world2pix_identity_p95_px']:.3f} "
            f"p99={out['matched_world2pix_identity_p99_px']:.3f}"
        )
    return out
