"""Effective PSF (ePSF) construction on MASTERSTAR and per-star PSF photometry.

Uses Photutils EPSFBuilder / PSFPhotometry. Does not import ``pipeline`` (avoid cycles).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table
from photutils.psf import EPSFBuilder, ImagePSF, PSFPhotometry, extract_stars

from database import VyvarDatabase
from gaia_catalog_id import normalize_gaia_source_id as _norm_catalog_id
from infolog import log_event

_MASTERSTAR_EPSF_NAME = "masterstar_epsf.fits"
_MASTERSTAR_EPSF_META_NAME = "masterstar_epsf_meta.json"


def _clamp_fwhm_px(v: float) -> float:
    return float(max(2.0, min(12.0, v)))


def _median_fwhm_obs_files(db: VyvarDatabase, draft_id: int) -> float | None:
    cur = db.conn.execute(
        """
        SELECT FWHM FROM OBS_FILES
        WHERE DRAFT_ID = ? AND FWHM IS NOT NULL;
        """,
        (int(draft_id),),
    )
    vals: list[float] = []
    for row in cur.fetchall():
        try:
            x = float(row[0])
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
    """Return FWHM in pixels for EPSFBuilder (VY_FWHM header → OBS_FILES median → 4.5), clamped to [2, 12]."""
    fwhm: float | None = None
    p = Path(masterstar_fits_path)
    try:
        if p.is_file():
            with fits.open(p, memmap=True) as hdul:
                hdr = hdul[0].header
                if "VY_FWHM" in hdr:
                    raw = float(hdr["VY_FWHM"])
                    if math.isfinite(raw) and raw > 0.0:
                        fwhm = raw
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


def _fit_shape_for_cutout(cutout_size: int) -> tuple[int, int]:
    fs = max(3, cutout_size - 4)
    if fs % 2 == 0:
        fs -= 1
    fs = max(3, fs)
    return (fs, fs)


def build_epsf_model(
    masterstar_fits_path: Path,
    masterstars_csv_path: Path,
    db: VyvarDatabase,
    draft_id: int,
    *,
    oversampling: int = 2,
    min_stars: int = 15,
) -> Path:
    """Build ePSF from clean comparison stars and write ``masterstar_epsf.fits`` + meta JSON."""
    mpath = Path(masterstar_fits_path)
    csvpath = Path(masterstars_csv_path)
    if not mpath.is_file():
        raise FileNotFoundError(f"MASTERSTAR FITS not found: {mpath}")
    if not csvpath.is_file():
        raise FileNotFoundError(f"masterstars CSV not found: {csvpath}")

    fwhm_px = get_epsf_fwhm_from_context(mpath, db, draft_id)
    cutout_size = _to_odd_cutout(int(fwhm_px * 5))
    log_event(
        f"PSF ePSF: FWHM={fwhm_px:.3f} px (clamped context), cutout_size={cutout_size}, oversampling={oversampling}"
    )

    def _csv_catalog_id_cell(raw: Any) -> str:
        if raw is None:
            return ""
        s = str(raw).strip()
        if not s or s.lower() in ("nan", "none"):
            return ""
        return _norm_catalog_id(raw)

    df = pd.read_csv(csvpath, low_memory=False, converters={"catalog_id": _csv_catalog_id_cell})
    req = ("catalog_id", "catalog_known_variable", "likely_saturated", "photometry_ok")
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"masterstars CSV missing required columns: {missing}")

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
    csv_ok["_cid"] = csv_ok["catalog_id"].map(_norm_catalog_id)
    csv_ok = csv_ok[csv_ok["_cid"] != ""]
    allowed = set(csv_ok["_cid"].unique())
    log_event(f"PSF ePSF: CSV filter → {len(csv_ok)} rows with non-empty catalog_id")

    cur = db.conn.execute(
        """
        SELECT SOURCE_ID_GAIA, X_MASTER, Y_MASTER
        FROM MASTER_SOURCES
        WHERE DRAFT_ID = ?
          AND COALESCE(IS_SAFE_COMP, 0) = 1
          AND (EXCLUSION_REASON IS NULL OR TRIM(EXCLUSION_REASON) = '');
        """,
        (int(draft_id),),
    )
    xs: list[float] = []
    ys: list[float] = []
    for row in cur.fetchall():
        gid = _norm_catalog_id(row["SOURCE_ID_GAIA"])
        if not gid or gid not in allowed:
            continue
        xm, ym = row["X_MASTER"], row["Y_MASTER"]
        if xm is None or ym is None:
            continue
        try:
            xs.append(float(xm))
            ys.append(float(ym))
        except (TypeError, ValueError):
            continue

    n_join = len(xs)
    log_event(f"PSF ePSF: joined MASTER_SOURCES ∩ CSV → {n_join} star positions")
    if n_join < min_stars:
        raise ValueError(
            f"EPSF build needs at least {min_stars} clean stars after DB+CSV join; found {n_join}."
        )

    cat = Table([xs, ys], names=("x", "y"))
    with fits.open(mpath, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)

    nd = NDData(data)
    stars = extract_stars(nd, cat, size=cutout_size)
    n_ext = int(getattr(stars, "n_stars", 0))
    log_event(f"PSF ePSF: extract_stars retained {n_ext} cutouts (size={cutout_size})")
    if n_ext < min_stars:
        raise ValueError(
            f"EPSF build needs at least {min_stars} extractable star cutouts; got {n_ext} "
            f"(many positions may lie outside the frame or overlap borders)."
        )

    osamp = max(1, int(oversampling))
    builder = EPSFBuilder(oversampling=osamp, maxiters=15, progress_bar=False)
    epsf_model, _fitted = builder(stars)
    arr = np.asarray(epsf_model.data, dtype=np.float32)

    out_dir = mpath.parent
    epsf_path = out_dir / _MASTERSTAR_EPSF_NAME
    meta_path = out_dir / _MASTERSTAR_EPSF_META_NAME

    fits.PrimaryHDU(data=arr).writeto(epsf_path, overwrite=True)

    created = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    meta = {
        "fwhm_px": float(fwhm_px),
        "cutout_size": int(cutout_size),
        "oversampling": int(osamp),
        "n_stars_used": int(n_ext),
        "draft_id": int(draft_id),
        "created_utc": created,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log_event(
        f"PSF ePSF: saved {epsf_path.name} shape={arr.shape}, n_stars_used={n_ext}, meta → {meta_path.name}"
    )
    return epsf_path


def psf_photometry_stars(
    frame_data: np.ndarray,
    frame_hdr: fits.Header,
    star_positions: pd.DataFrame,
    epsf_model_path: Path,
    *,
    cutout_size: int | None = None,
    error: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run PSFPhotometry on cutouts per star; never fails per row (exceptions → NaN, fit_ok False).

    If ``error`` is provided (same shape as ``frame_data``), it is passed to ``PSFPhotometry`` for
    per-pixel uncertainties (enables finite reduced χ² when the model fits).
    """
    _ = frame_hdr  # reserved for future metadata (WCS, gain, …)

    cols_req = ("x", "y", "catalog_id", "name")
    for c in cols_req:
        if c not in star_positions.columns:
            raise ValueError(f"star_positions missing required column: {c!r}")

    ep = Path(epsf_model_path)
    if not ep.is_file():
        raise FileNotFoundError(f"EPSF FITS not found: {ep}")

    if cutout_size is None:
        meta_fp = ep.parent / _MASTERSTAR_EPSF_META_NAME
        if not meta_fp.is_file():
            raise FileNotFoundError(f"cutout_size=None requires {meta_fp}")
        meta = json.loads(meta_fp.read_text(encoding="utf-8"))
        cutout_size = int(meta["cutout_size"])
        os_meta = meta.get("oversampling", 2)
        if isinstance(os_meta, list):
            osamp: Any = int(os_meta[0]) if len(os_meta) else 2
        else:
            osamp = int(os_meta)
    else:
        osamp = 2
        try:
            meta_fp = ep.parent / _MASTERSTAR_EPSF_META_NAME
            if meta_fp.is_file():
                meta = json.loads(meta_fp.read_text(encoding="utf-8"))
                om = meta.get("oversampling", 2)
                osamp = int(om[0]) if isinstance(om, list) and len(om) else int(om)
        except Exception:  # noqa: BLE001
            pass

    cutout_size = int(cutout_size)
    if cutout_size % 2 == 0 or cutout_size < 3:
        raise ValueError(f"cutout_size must be odd and >= 3, got {cutout_size}")

    err_full: np.ndarray | None = None
    if error is not None:
        err_full = np.asarray(error, dtype=np.float64)
        if err_full.shape != frame_data.shape:
            raise ValueError(
                f"error map shape {err_full.shape} != frame_data shape {frame_data.shape}"
            )

    psf_data = np.asarray(fits.getdata(ep), dtype=np.float64)
    psf_model = ImagePSF(psf_data, oversampling=osamp)
    fit_shape = _fit_shape_for_cutout(cutout_size)
    phot = PSFPhotometry(psf_model, fit_shape=fit_shape, progress_bar=False)

    h, w = frame_data.shape
    half = cutout_size // 2

    out_rows: list[dict[str, Any]] = []
    _cols = [
        "catalog_id",
        "name",
        "x",
        "y",
        "psf_flux",
        "psf_flux_err",
        "psf_chi2",
        "psf_fit_ok",
    ]
    if star_positions.empty:
        return pd.DataFrame(columns=_cols)

    for _, row in star_positions.iterrows():
        cid = row["catalog_id"]
        name = row["name"]
        try:
            x = float(row["x"])
            y = float(row["y"])
        except (TypeError, ValueError):
            out_rows.append(
                {
                    "catalog_id": cid,
                    "name": name,
                    "x": row["x"],
                    "y": row["y"],
                    "psf_flux": float("nan"),
                    "psf_flux_err": float("nan"),
                    "psf_chi2": float("nan"),
                    "psf_fit_ok": False,
                }
            )
            continue

        base = {
            "catalog_id": cid,
            "name": name,
            "x": x,
            "y": y,
            "psf_flux": float("nan"),
            "psf_flux_err": float("nan"),
            "psf_chi2": float("nan"),
            "psf_fit_ok": False,
        }

        xi, yi = int(round(x)), int(round(y))
        if xi < half or yi < half or xi >= w - half or yi >= h - half:
            out_rows.append(base)
            continue

        x1 = xi - half
        y1 = yi - half
        x2 = x1 + cutout_size
        y2 = y1 + cutout_size

        try:
            cut = np.asarray(frame_data[y1:y2, x1:x2], dtype=np.float64)
            if cut.shape != (cutout_size, cutout_size):
                out_rows.append(base)
                continue

            xc = x - x1
            yc = y - y1
            flux_guess = float(np.nansum(cut))
            if not math.isfinite(flux_guess) or flux_guess <= 0.0:
                flux_guess = float(np.nanmax(cut)) * 0.5 * cutout_size * cutout_size
                if not math.isfinite(flux_guess) or flux_guess <= 0.0:
                    flux_guess = 1.0

            init = Table([[xc], [yc], [flux_guess]], names=("x_0", "y_0", "flux_0"))
            err_cut = None
            if err_full is not None:
                err_cut = np.asarray(err_full[y1:y2, x1:x2], dtype=np.float64)
                if err_cut.shape != cut.shape:
                    raise ValueError("error cutout shape mismatch")
            res = phot(cut, init_params=init, error=err_cut)

            flux_fit = float(res["flux_fit"][0])
            flux_err = float(res["flux_err"][0])
            chi2 = float(res["reduced_chi2"][0])
            flags = int(res["flags"][0])
            converged = (flags & 8) == 0
            chi2_ok = math.isfinite(chi2) and chi2 < 5.0
            fit_ok = bool(converged and chi2_ok)

            out_rows.append(
                {
                    "catalog_id": cid,
                    "name": name,
                    "x": x,
                    "y": y,
                    "psf_flux": flux_fit,
                    "psf_flux_err": flux_err,
                    "psf_chi2": chi2,
                    "psf_fit_ok": fit_ok,
                }
            )
        except Exception:  # noqa: BLE001
            out_rows.append(base)

    return pd.DataFrame(out_rows, columns=_cols)


__all__ = [
    "get_epsf_fwhm_from_context",
    "build_epsf_model",
    "psf_photometry_stars",
]
