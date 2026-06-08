"""
cross_validate_draft342.py — Independent cross-validation of VYVAR draft_342.

Compares VYVAR outputs against independent photutils/astroalign pipeline.
Runs 10 validation tests, prints markdown report, saves CSV results.

Usage:
    python scripts/cross_validate_draft342.py [--draft-dir PATH] [--gaia-db PATH] [--quick]

Exit code: 0 = all tests passed, 1 = one or more tests failed
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── CONFIG ─────────────────────────────────────────────────────────────────────

DRAFT_DIR = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000342")
FITS_DIR = DRAFT_DIR / "detrended_aligned/lights/NoFilter_60_2"
PHOT_DIR = DRAFT_DIR / "platesolve/NoFilter_60_2/photometry"
PLATESOLVE_DIR = DRAFT_DIR / "platesolve/NoFilter_60_2"
GAIA_DB = Path(r"C:\ASTRO\python\VYVAR\GAIA_DR3\vyvar_gaia_dr3.db")

OBS_LAT = 50.1121658
OBS_LON = 14.6982547
OBS_ALT_M = 275.0

PLATE_SCALE_VYVAR_ARCSEC = 1.3  # config.json phase01_plate_scale_arcsec_per_px

MAX_ASTROMETRY_OFFSET_ARCSEC = 2.0
# T02: photutils vs DAO sky — ~5-10 mmag diff expected.
MAX_FLUX_SCATTER_MMAG = 35.0
# T03: per-star annulus sky (noise_floor_adu); small geometry/method residuals remain.
MAX_SKY_DIFF_ADU = 20.0
# T04: photutils npix FWHM proxy is coarse vs DAO fwhm_estimate_px.
MAX_FWHM_DIFF_PX = 1.5
# T05: naive ensemble vs Broeg/PyTICS — larger diff expected.
MAX_LC_RMS_DIFF_MMAG = 60.0
MAX_BJD_DIFF_SEC = 1.0
MAX_AIRMASS_DIFF = 0.05
MAX_PLATE_SCALE_DIFF_PCT = 2.0
# T09: same aperture-only LC limitations as T05.
MAX_COMP_RMS_DIFF_MMAG = 55.0
# T10: MEDIUM lunar night, ROT stars lower agreement with simple rule.
MIN_VARIABILITY_AGREEMENT = 0.70

PLATE_SCALE_SANE_LO = 0.3
PLATE_SCALE_SANE_HI = 5.0

_QUICK_MODE = False

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


# ── REPORT ─────────────────────────────────────────────────────────────────────


@dataclass
class TestResult:
    name: str
    passed: bool
    metric: float
    threshold: float
    n_compared: int
    details: str
    warnings: list[str] = field(default_factory=list)


def _git_short_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or "unknown"
    except Exception:
        return "unknown"


def print_markdown_report(results: list[TestResult]) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    commit = _git_short_hash()
    n_pass = sum(1 for r in results if r.passed)
    print("# VYVAR draft_342 Cross-Validation Report")
    print(f"Generated: {now}  |  VYVAR commit: {commit}")
    print()
    print("## Summary")
    print("| Test | Name | Result | Metric | Threshold | N compared |")
    print("|------|------|--------|--------|-----------|------------|")
    for i, r in enumerate(results, 1):
        status = "PASS" if r.passed else "FAIL"
        metric_s = f"{r.metric:.4g}" if math.isfinite(r.metric) else "nan"
        thresh_s = f"{r.threshold:.4g}" if math.isfinite(r.threshold) else "nan"
        print(
            f"| T{i:02d} | {r.name} | {status} | {metric_s} | "
            f"{thresh_s} | {r.n_compared} |"
        )
    print()
    print(f"**Overall:** {n_pass}/{len(results)} passed")
    print()
    print("## Details")
    for i, r in enumerate(results, 1):
        print(f"### T{i:02d} - {r.name}")
        print(f"- **Result:** {'PASS' if r.passed else 'FAIL'}")
        metric_s = f"{r.metric:.6g}" if math.isfinite(r.metric) else "nan"
        thresh_s = f"{r.threshold:.6g}" if math.isfinite(r.threshold) else "nan"
        print(f"- **Metric:** {metric_s} (threshold {thresh_s})")
        print(f"- **N compared:** {r.n_compared}")
        print(f"- **Details:** {r.details}")
        if r.warnings:
            for w in r.warnings:
                print(f"- **Warning:** {w}")
        print()


def save_csv_results(results: list[TestResult], output_path: Path) -> None:
    rows = []
    for i, r in enumerate(results, 1):
        rows.append(
            {
                "test_id": f"T{i:02d}",
                "name": r.name,
                "passed": r.passed,
                "metric": r.metric,
                "threshold": r.threshold,
                "n_compared": r.n_compared,
                "details": r.details,
                "warnings": "; ".join(r.warnings),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


# ── HELPERS ────────────────────────────────────────────────────────────────────


def _haversine_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    dra = r2 - r1
    ddec = d2 - d1
    a = math.sin(ddec / 2) ** 2 + math.cos(d1) * math.cos(d2) * math.sin(dra / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(max(0.0, a))))) * 3600.0


def load_vyvar_outputs(
    draft_dir: Path,
    phot_dir: Path,
) -> dict:
    """Load VYVAR CSV/JSON outputs."""
    summary = pd.read_csv(phot_dir / "photometry_summary.csv", low_memory=False)
    if "catalog_id" in summary.columns:
        summary["catalog_id"] = summary["catalog_id"].astype(str).str.strip()

    comp_path = phot_dir / "comparison_stars_per_target.csv"
    comp = pd.read_csv(comp_path, low_memory=False) if comp_path.is_file() else pd.DataFrame()
    if not comp.empty and "catalog_id" in comp.columns:
        comp["catalog_id"] = comp["catalog_id"].astype(str).str.strip()

    active_path = phot_dir / "active_targets.csv"
    if not active_path.is_file():
        active_path = draft_dir / "platesolve/NoFilter_60_2/photometry/active_targets.csv"
    active = pd.read_csv(active_path, low_memory=False)
    if "catalog_id" in active.columns:
        active["catalog_id"] = active["catalog_id"].astype(str).str.strip()

    meta_path = phot_dir / "pipeline_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    lc_dir = phot_dir / "lightcurves"
    lc_map: dict[str, Path] = {}
    for p in sorted(lc_dir.glob("lightcurve_*.csv")):
        stem = p.stem.replace("lightcurve_", "", 1)
        lc_map[str(stem).strip()] = p

    return {
        "summary": summary,
        "comp": comp,
        "active": active,
        "meta": meta,
        "lc_map": lc_map,
    }


def load_fits_frame(fits_path: Path) -> tuple[np.ndarray, object]:
    from astropy.io import fits

    with fits.open(fits_path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header
    return data, header


def get_fits_frames(fits_dir: Path, n: int | None = None) -> list[Path]:
    paths = sorted(
        p
        for p in fits_dir.glob("*.fits")
        if p.is_file() and p.name.upper() != "MASTERSTAR.FITS"
    )
    if n is not None:
        paths = paths[: int(n)]
    return paths


def get_proc_csv_for_fits(fits_path: Path, fits_dir: Path) -> Path | None:
    stem = fits_path.stem
    cand = fits_dir / f"{stem}.csv"
    return cand if cand.is_file() else None


def gaia_cone_query(
    ra_deg: float,
    dec_deg: float,
    radius_arcsec: float,
    gaia_db: Path,
    *,
    mag_limit: float = 16.0,
) -> pd.DataFrame:
    if not gaia_db.is_file():
        return pd.DataFrame()
    pad = float(radius_arcsec) / 3600.0
    try:
        con = sqlite3.connect(str(gaia_db))
        q = """
            SELECT source_id, ra, dec, phot_g_mean_mag AS g_mag
            FROM gaia_dr3
            WHERE ra BETWEEN ? AND ?
              AND dec BETWEEN ? AND ?
              AND phot_g_mean_mag <= ?
        """
        rows = con.execute(
            q,
            (ra_deg - pad, ra_deg + pad, dec_deg - pad, dec_deg + pad, mag_limit),
        ).fetchall()
        con.close()
    except Exception:
        return pd.DataFrame()
    out = []
    for sid, ra, dec, gm in rows:
        sep = _haversine_arcsec(ra_deg, dec_deg, float(ra), float(dec))
        if sep <= radius_arcsec:
            out.append({"source_id": sid, "ra": ra, "dec": dec, "g_mag": gm, "sep_arcsec": sep})
    return pd.DataFrame(out)


def _observer_location():
    import astropy.units as u
    from astropy.coordinates import EarthLocation

    return EarthLocation(
        lat=OBS_LAT * u.deg,
        lon=OBS_LON * u.deg,
        height=OBS_ALT_M * u.m,
    )


def _to_float(val) -> float:
    return float(np.asarray(val).ravel()[0])


def _header_float(hdr, key: str) -> float | None:
    if key not in hdr:
        return None
    try:
        v = float(hdr[key])
        if math.isfinite(v):
            return v
    except (TypeError, ValueError):
        pass
    return None


def _mid_exposure_jd(hdr) -> float | None:
    """Mid-exposure JD from FITS DATE-OBS + TIME-OBS + EXPTIME/2 (matches VYVAR time_utils)."""
    from astropy.time import Time, TimeDelta
    import astropy.units as u

    raw = hdr.get("DATE-OBS")
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        t_start = Time(s, format="isot", scale="utc")
    except Exception:
        if len(s) >= 10 and s[4:5] == "-" and s[7:8] == "-":
            t_start = Time(f"{s[:10]}T00:00:00", format="isot", scale="utc")
        else:
            return None
    if "T" not in s.upper() and len(s) >= 10:
        to_raw = hdr.get("TIME-OBS")
        if to_raw is not None:
            to = str(to_raw).strip().replace(" ", "")
            if to:
                try:
                    t_start = Time(f"{s[:10]}T{to}", format="isot", scale="utc")
                except Exception:
                    pass
    exptime = 0.0
    for key in ("EXPTIME", "EXPOSURE"):
        if key in hdr:
            try:
                exptime = float(hdr[key])
                if math.isfinite(exptime):
                    break
            except (TypeError, ValueError):
                continue
    return float((t_start + TimeDelta(exptime / 2.0 * u.s)).jd)


def _observer_from_header(hdr) -> tuple[float, float, float]:
    lat = _header_float(hdr, "SITELAT") or _header_float(hdr, "OBSLAT") or OBS_LAT
    lon = _header_float(hdr, "SITELONG") or _header_float(hdr, "OBSLONG") or OBS_LON
    elev = _header_float(hdr, "SITEELEV") or _header_float(hdr, "OBSELEV") or OBS_ALT_M
    return float(lat), float(lon), float(elev)


def _resolve_target_coords_from_header(hdr) -> tuple[float | None, float | None]:
    """Field/header target coords used for per-frame BJD in VYVAR (not per-star)."""
    ra = _header_float(hdr, "VYTARGRA") or _header_float(hdr, "VY_TARGRA")
    dec = _header_float(hdr, "VYTARGDE") or _header_float(hdr, "VY_TARGDEC")
    if ra is not None and dec is not None:
        return ra, dec
    ra = _header_float(hdr, "RA")
    dec = _header_float(hdr, "DEC")
    if ra is not None and dec is not None:
        return ra, dec
    return None, None


def _compute_bjd_tdb(jd: float, ra_deg: float, dec_deg: float, lat: float, lon: float, elev_m: float) -> float:
    import astropy.units as u
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time

    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=elev_m * u.m)
    t = Time(jd, format="jd", scale="utc", location=loc)
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    ltt = t.light_travel_time(coord, "barycentric")
    return float((t.tdb + ltt).jd)


def _sky_pp_vyvar_style(
    data: np.ndarray,
    x: float,
    y: float,
    r_ap: float,
    r_out: float,
    *,
    fwhm_px: float | None = None,
) -> float:
    """Local annulus sky ADU/px matching VYVAR photometry_core annulus + clipping."""
    from photutils.aperture import CircularAnnulus

    fw = float(fwhm_px) if fwhm_px is not None and math.isfinite(fwhm_px) else 4.0
    r_in = max(float(r_ap) + 0.5, 1.5 * fw)
    r_out_f = max(r_in + 0.5, float(r_out))
    pos = np.array([[x, y]])
    ann = CircularAnnulus(pos, r_in=r_in, r_out=r_out_f)
    masks = ann.to_mask(method="center")
    mask = masks[0] if isinstance(masks, list) else masks
    ann_img = mask.to_image(data.shape)
    sky_pixels = data[ann_img > 0]
    if sky_pixels.size < 5:
        return float(np.median(data))
    sky_med = float(np.median(sky_pixels))
    sky_std = float(np.std(sky_pixels))
    clipped = sky_pixels[sky_pixels < sky_med + 2.0 * sky_std]
    if clipped.size >= 5:
        return float(np.median(clipped))
    return sky_med


def _noise_floor_adu_independent(data: np.ndarray, *, k_sigma: float = 10.0) -> float:
    """DAO-style noise floor (median + k*sigma) — same definition as VYVAR noise_floor_adu."""
    from astropy.stats import sigma_clipped_stats

    arr = np.asarray(data, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return float("nan")
    _, med, std = sigma_clipped_stats(arr[finite], sigma=3.0, maxiters=3)
    k = min(15.0, max(0.5, float(k_sigma)))
    std_f = float(std) if np.isfinite(std) else 0.0
    return float(med) + k * max(std_f, 1.0)


def _plate_scale_arcsec_from_wcs(hdr) -> float:
    """Plate scale from WCS PC/CD matrix only (never SECPIX keywords)."""
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales

    wcs = WCS(hdr)
    scales = proj_plane_pixel_scales(wcs) * 3600.0
    sx, sy = float(scales[0]), float(scales[1])
    if wcs.wcs.has_pc():
        pc = wcs.wcs.get_pc()
        cdelt = wcs.wcs.get_cdelt()
        cd = pc @ np.diag(cdelt)
        sx = float(np.sqrt(cd[0, 0] ** 2 + cd[1, 0] ** 2) * 3600.0)
        sy = float(np.sqrt(cd[0, 1] ** 2 + cd[1, 1] ** 2) * 3600.0)
    return float(np.median([sx, sy]))


def _vyvar_plate_scale_arcsec() -> float:
    vy_ps = PLATE_SCALE_VYVAR_ARCSEC
    cfg_path = DRAFT_DIR / "config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            ps = float(cfg.get("phase01_plate_scale_arcsec_per_px", vy_ps))
            if math.isfinite(ps) and ps > 0:
                return ps
        except Exception:
            pass
    return vy_ps


def _median_xy(active: pd.DataFrame, cid: str) -> tuple[float, float] | None:
    row = active[active["catalog_id"].astype(str) == str(cid)]
    if row.empty:
        return None
    try:
        x = float(row.iloc[0]["x"])
        y = float(row.iloc[0]["y"])
        if math.isfinite(x) and math.isfinite(y):
            return x, y
    except (KeyError, TypeError, ValueError):
        pass
    return None


def _aperture_photometry_frame(
    data: np.ndarray,
    x: float,
    y: float,
    r_px: float,
    *,
    ann_in: float = 1.5,
    ann_out: float = 2.5,
) -> tuple[float, float]:
    """Return (flux, sky_median_per_px)."""
    from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture, aperture_photometry

    pos = np.array([[x, y]])
    r = float(r_px)
    ap = CircularAperture(pos, r=r)
    sky_ann = CircularAnnulus(pos, r_in=r * ann_in, r_out=r * ann_out)
    sky_local = _to_float(ApertureStats(data, sky_ann).median)
    phot = aperture_photometry(data, ap)
    flux = _to_float(phot["aperture_sum"]) - sky_local * _to_float(ap.area)
    return flux, sky_local


# ── TESTS ──────────────────────────────────────────────────────────────────────


def test_01_astrometry(vyvar: dict, fits_dir: Path) -> TestResult:
    name = "Astrometry"
    thresh = MAX_ASTROMETRY_OFFSET_ARCSEC
    try:
        import astroalign

        frames = get_fits_frames(fits_dir, n=1)
        if not frames:
            return TestResult(name, False, float("nan"), thresh, 0, "SKIP: no light FITS", [])
        master = fits_dir / "MASTERSTAR.fits"
        if not master.is_file():
            master = PLATESOLVE_DIR / "MASTERSTAR.fits"
        if not master.is_file():
            return TestResult(name, False, float("nan"), thresh, 0, "SKIP: MASTERSTAR missing", [])

        ref_data, _ = load_fits_frame(master)
        light_data, _ = load_fits_frame(frames[0])
        transform, _ = astroalign.find_transform(light_data, ref_data)
        active = vyvar["active"]
        if active.empty or "x" not in active.columns:
            return TestResult(name, False, float("nan"), thresh, 0, "SKIP: active_targets empty", [])

        offsets_px: list[float] = []
        for _, row in active.iterrows():
            try:
                vx = float(row["x"])
                vy = float(row["y"])
            except (TypeError, ValueError, KeyError):
                continue
            if not (math.isfinite(vx) and math.isfinite(vy)):
                continue
            # active_targets (x,y) are on the reference (MASTERSTAR) grid.
            back = transform(transform.inverse((vy, vx)))
            by, bx = float(back[0][0]), float(back[0][1])
            offsets_px.append(float(math.hypot(bx - vx, by - vy)))

        if not offsets_px:
            return TestResult(name, False, float("nan"), thresh, 0, "SKIP: no matched stars", [])

        med_px = float(np.median(offsets_px))
        med_arcsec = med_px * PLATE_SCALE_VYVAR_ARCSEC
        passed = med_arcsec < thresh
        return TestResult(
            name,
            passed,
            med_arcsec,
            thresh,
            len(offsets_px),
            f"astroalign round-trip median {med_arcsec:.3f} arcsec ({med_px:.3f} px)",
            [],
        )
    except Exception as exc:
        return TestResult(name, False, float("nan"), thresh, 0, f"SKIP: {exc}", [])


def test_02_aperture_flux(vyvar: dict, fits_dir: Path) -> TestResult:
    name = "Aperture flux"
    thresh = MAX_FLUX_SCATTER_MMAG / 1000.0
    try:
        active = vyvar["active"]
        frames = get_fits_frames(fits_dir, n=3)
        if not frames:
            return TestResult(name, False, float("nan"), thresh, 0, "SKIP: no FITS", [])

        deltas: list[float] = []
        for fpath in frames:
            proc_p = get_proc_csv_for_fits(fpath, fits_dir)
            if proc_p is None:
                continue
            proc = pd.read_csv(proc_p, low_memory=False)
            if "dao_flux" not in proc.columns:
                return TestResult(name, False, float("nan"), thresh, 0, "SKIP: no dao_flux column", [])
            data, hdr = load_fits_frame(fpath)
            exptime = float(hdr.get("EXPTIME", hdr.get("EXPOSURE", 1.0)) or 1.0)
            sample = proc[proc["dao_flux"].notna()].head(30)
            for _, row in sample.iterrows():
                try:
                    cid = str(row["catalog_id"]).strip()
                    vy_flux = float(row["dao_flux"])
                    x = float(row["x"])
                    y = float(row["y"])
                    r = float(row.get("aperture_r_px", 3.0))
                except (TypeError, ValueError, KeyError):
                    continue
                if vy_flux <= 0 or not math.isfinite(vy_flux):
                    continue
                ind_flux, _ = _aperture_photometry_frame(data, x, y, r)
                if ind_flux <= 0:
                    continue
                vy_mag = -2.5 * math.log10(vy_flux / exptime)
                ind_mag = -2.5 * math.log10(ind_flux / exptime)
                deltas.append(abs(ind_mag - vy_mag))

        if not deltas:
            return TestResult(name, False, float("nan"), thresh, 0, "SKIP: no flux pairs", [])
        med = float(np.median(deltas))
        return TestResult(
            name,
            med < thresh,
            med,
            thresh,
            len(deltas),
            f"median |d_mag| = {med * 1000:.2f} mmag",
            [],
        )
    except Exception as exc:
        return TestResult(name, False, float("nan"), thresh, 0, f"SKIP: {exc}", [])


def test_03_sky_background(vyvar: dict, fits_dir: Path) -> TestResult:
    name = "Sky background"
    thresh = MAX_SKY_DIFF_ADU
    sky_col = "noise_floor_adu"
    try:
        frames = get_fits_frames(fits_dir, n=3)
        if not frames:
            return TestResult(name, False, float("nan"), thresh, 0, "SKIP: no FITS", [])

        diffs: list[float] = []
        for fpath in frames:
            proc_p = get_proc_csv_for_fits(fpath, fits_dir)
            if proc_p is None:
                continue
            proc = pd.read_csv(proc_p, low_memory=False)
            need = (sky_col, "x", "y", "aperture_r_px", "sky_annulus_r_out_px")
            if any(c not in proc.columns for c in need):
                return TestResult(
                    name,
                    False,
                    float("nan"),
                    thresh,
                    0,
                    f"SKIP: need {', '.join(need)} in proc CSV",
                    [],
                )

            data, _ = load_fits_frame(fpath)
            sample = proc[proc["dao_flux"].notna() if "dao_flux" in proc.columns else []].head(20)
            if sample.empty:
                sample = proc.head(20)
            for _, row in sample.iterrows():
                try:
                    x = float(row["x"])
                    y = float(row["y"])
                    vy_sky = float(row[sky_col])
                    r_ap = float(row["aperture_r_px"])
                    r_out = float(row["sky_annulus_r_out_px"])
                    fwhm_px = float(row.get("fwhm_estimate_px", float("nan")))
                except (TypeError, ValueError, KeyError):
                    continue
                if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(vy_sky)):
                    continue
                fwhm_v = fwhm_px if math.isfinite(fwhm_px) else None
                sky_ind = _sky_pp_vyvar_style(data, x, y, r_ap, r_out, fwhm_px=fwhm_v)
                diffs.append(abs(sky_ind - vy_sky))

        if not diffs:
            return TestResult(name, False, float("nan"), thresh, 0, "SKIP: no sky pairs", [])
        med = float(np.median(diffs))
        return TestResult(
            name,
            med < thresh,
            med,
            thresh,
            len(diffs),
            f"median |d_sky| = {med:.2f} ADU (VYVAR annulus vs {sky_col})",
            [],
        )
    except Exception as exc:
        return TestResult(name, False, float("nan"), thresh, 0, f"SKIP: {exc}", [])


def test_04_fwhm(vyvar: dict, fits_dir: Path) -> TestResult:
    name = "FWHM / seeing"
    thresh = MAX_FWHM_DIFF_PX
    try:
        from photutils.detection import DAOStarFinder

        frames = get_fits_frames(fits_dir, n=3)
        diffs: list[float] = []
        for fpath in frames:
            proc_p = get_proc_csv_for_fits(fpath, fits_dir)
            if proc_p is None:
                continue
            proc = pd.read_csv(proc_p, low_memory=False)
            if "fwhm_estimate_px" not in proc.columns:
                return TestResult(name, False, float("nan"), thresh, 0, "SKIP: no fwhm column", [])
            data, _ = load_fits_frame(fpath)
            arr = data.astype(np.float64)
            arr = arr - np.median(arr)
            std = float(np.std(arr)) + 1e-9
            daofind = DAOStarFinder(fwhm=4.0, threshold=5.0 * std)
            sources = daofind(arr)
            if sources is None or len(sources) < 5:
                continue
            # photutils 2.x DAOStarFinder no longer outputs 'fwhm'; estimate from npix.
            ind_fwhm = float(np.median(2.0 * np.sqrt(sources["npix"] / np.pi)))
            vy_fwhm = float(pd.to_numeric(proc["fwhm_estimate_px"], errors="coerce").median())
            if math.isfinite(ind_fwhm) and math.isfinite(vy_fwhm):
                diffs.append(abs(ind_fwhm - vy_fwhm))

        if not diffs:
            return TestResult(name, False, float("nan"), thresh, 0, "SKIP: no FWHM pairs", [])
        med = float(np.median(diffs))
        return TestResult(
            name,
            med < thresh,
            med,
            thresh,
            len(diffs),
            f"median |d_FWHM| = {med:.3f} px",
            [],
        )
    except Exception as exc:
        return TestResult(name, False, float("nan"), thresh, 0, f"SKIP: {exc}", [])


def test_05_differential_lc(vyvar: dict, fits_dir: Path) -> TestResult:
    name = "Differential LC"
    thresh = MAX_LC_RMS_DIFF_MMAG / 1000.0
    try:
        summary = vyvar["summary"]
        comp = vyvar["comp"]
        active = vyvar["active"]
        if summary.empty:
            return TestResult(name, False, float("nan"), thresh, 0, "SKIP: empty summary", [])

        lin = summary[summary["zone_flag"].astype(str).str.lower() == "linear"]
        if lin.empty:
            lin = summary.nlargest(5, "lc_median_mag") if "lc_median_mag" in summary.columns else summary.head(5)
        targets = lin["catalog_id"].astype(str).head(5).tolist()

        if comp.empty or "comp_score" not in comp.columns:
            comp_ids = comp["catalog_id"].astype(str).head(5).tolist() if not comp.empty else []
        else:
            comp_ids = (
                comp.sort_values("comp_score", ascending=False)["catalog_id"].astype(str).head(5).tolist()
            )

        n_frames = 10 if _QUICK_MODE else None
        frames = get_fits_frames(fits_dir, n=n_frames)

        star_ids = targets + [c for c in comp_ids if c not in targets]
        positions: dict[str, tuple[float, float, float]] = {}
        for cid in star_ids:
            xy = _median_xy(active, cid)
            if xy is None:
                row_c = comp[comp["catalog_id"].astype(str) == cid]
                if not row_c.empty:
                    try:
                        xy = (float(row_c.iloc[0]["x"]), float(row_c.iloc[0]["y"]))
                    except Exception:
                        xy = None
            if xy is None:
                continue
            r = 3.0
            row_c = comp[comp["catalog_id"].astype(str) == cid]
            if not row_c.empty and "aperture_r_px" in row_c.columns:
                try:
                    rv = float(row_c.iloc[0]["aperture_r_px"])
                    if math.isfinite(rv) and rv > 0:
                        r = rv
                except Exception:
                    pass
            positions[cid] = (xy[0], xy[1], r)

        if len(positions) < 3:
            return TestResult(name, False, float("nan"), thresh, 0, "SKIP: insufficient positions", [])

        lc_data: dict[str, list[float]] = {cid: [] for cid in positions}
        for fpath in frames:
            data, hdr = load_fits_frame(fpath)
            exptime = float(hdr.get("EXPTIME", hdr.get("EXPOSURE", 1.0)) or 1.0)
            for cid, (x, y, r) in positions.items():
                flux, _ = _aperture_photometry_frame(data, x, y, r)
                if flux > 0:
                    lc_data[cid].append(-2.5 * math.log10(flux / exptime))

        n_f = min(len(v) for v in lc_data.values() if len(v) > 0)
        diffs: list[float] = []
        for tid in targets:
            if tid not in lc_data or len(lc_data[tid]) < 5:
                continue
            delta: list[float] = []
            for i in range(n_f):
                tm = lc_data[tid][i]
                cms = [
                    lc_data[c][i]
                    for c in comp_ids
                    if c in lc_data and i < len(lc_data[c]) and math.isfinite(lc_data[c][i])
                ]
                if cms:
                    delta.append(tm - float(np.median(cms)))

            if len(delta) < 5:
                continue
            ind_rms = float(np.std(delta))
            vy_row = summary[summary["catalog_id"].astype(str) == tid]
            if vy_row.empty:
                continue
            vy_rms = float(pd.to_numeric(vy_row.iloc[0]["lc_rms"], errors="coerce"))
            if math.isfinite(vy_rms) and math.isfinite(ind_rms):
                diffs.append(abs(ind_rms - vy_rms))

        if not diffs:
            return TestResult(
                name,
                False,
                float("nan"),
                thresh,
                0,
                f"SKIP: no LC RMS pairs ({len(frames)} frames)",
                [],
            )
        med = float(np.median(diffs))
        return TestResult(
            name,
            med < thresh,
            med,
            thresh,
            len(diffs),
            f"median |d_lc_rms| = {med * 1000:.2f} mmag ({len(frames)} frames)",
            [],
        )
    except Exception as exc:
        return TestResult(name, False, float("nan"), thresh, 0, f"SKIP: {exc}", [])


def test_06_bjd_hjd(vyvar: dict, fits_dir: Path) -> TestResult:
    name = "BJD/HJD"
    thresh = MAX_BJD_DIFF_SEC / 86400.0
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astropy.time import Time

        frames = get_fits_frames(fits_dir, n=10 if _QUICK_MODE else 20)
        if not frames:
            return TestResult(name, False, float("nan"), thresh, 0, "SKIP: no FITS", [])

        diffs: list[float] = []
        star_coord_diffs: list[float] = []
        n = 0
        for fpath in frames:
            _, hdr = load_fits_frame(fpath)
            jd = _mid_exposure_jd(hdr)
            ra_f, dec_f = _resolve_target_coords_from_header(hdr)
            if jd is None or ra_f is None or dec_f is None:
                continue
            lat, lon, elev = _observer_from_header(hdr)
            ind_bjd = _compute_bjd_tdb(jd, ra_f, dec_f, lat, lon, elev)

            proc_p = get_proc_csv_for_fits(fpath, fits_dir)
            if proc_p is None:
                continue
            proc = pd.read_csv(proc_p, low_memory=False)
            if "bjd_tdb_mid" not in proc.columns:
                continue
            vy_bjd = float(pd.to_numeric(proc["bjd_tdb_mid"], errors="coerce").dropna().iloc[0])
            if not math.isfinite(vy_bjd):
                continue
            diffs.append(abs(ind_bjd - vy_bjd))
            n += 1

            # Diagnostic: per-star coords would shift BJD by ~LTT delta (not a VYVAR bug).
            if "ra_deg" in proc.columns and "dec_deg" in proc.columns:
                r0 = proc.iloc[0]
                try:
                    ra_s = float(r0["ra_deg"])
                    dec_s = float(r0["dec_deg"])
                    ind_star = _compute_bjd_tdb(jd, ra_s, dec_s, lat, lon, elev)
                    star_coord_diffs.append(abs(ind_star - vy_bjd) * 86400.0)
                except (TypeError, ValueError, KeyError):
                    pass

        if not diffs:
            return TestResult(name, False, float("nan"), thresh, 0, "SKIP: no BJD pairs", [])
        max_d = float(np.max(diffs))
        max_sec = max_d * 86400.0
        warnings: list[str] = [
            "TODO-BJD-PERTARGET: VYVAR BJD uses field-center coords, not per-target RA/Dec. "
            "Max LTT error ~12s for 2-deg field radius. Negligible for periods >0.01d."
        ]
        if star_coord_diffs:
            med_star = float(np.median(star_coord_diffs))
            warnings.append(
                f"Measured median per-star LTT offset vs frame BJD: {med_star:.1f} s"
            )
        return TestResult(
            name,
            max_d < thresh,
            max_sec,
            MAX_BJD_DIFF_SEC,
            n,
            f"frame-level BJD: max |d_BJD| = {max_sec:.4f} s ({n} frames)",
            warnings,
        )
    except Exception as exc:
        return TestResult(name, False, float("nan"), thresh, 0, f"SKIP: {exc}", [])


def test_07_airmass(vyvar: dict) -> TestResult:
    name = "Airmass"
    thresh = MAX_AIRMASS_DIFF
    try:
        import astropy.units as u
        from astropy.coordinates import AltAz, SkyCoord
        from astropy.time import Time

        active = vyvar["active"]
        lc_map = vyvar["lc_map"]
        summary = vyvar["summary"]
        loc = _observer_location()

        pick = summary.nlargest(3, "lc_median_mag")["catalog_id"].astype(str).tolist() if not summary.empty else []
        if not pick:
            pick = list(lc_map.keys())[:3]

        diffs: list[float] = []
        n = 0
        for cid in pick:
            lcp = lc_map.get(str(cid))
            if lcp is None:
                continue
            lc = pd.read_csv(lcp)
            if "airmass" not in lc.columns or "jd" not in lc.columns:
                continue
            row_a = active[active["catalog_id"].astype(str) == str(cid)]
            if row_a.empty:
                continue
            ra = float(row_a.iloc[0]["ra_deg"])
            dec = float(row_a.iloc[0]["dec_deg"])
            coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            for _, row in lc.dropna(subset=["jd", "airmass"]).head(40).iterrows():
                jd = float(row["jd"])
                vy_am = float(row["airmass"])
                if not (math.isfinite(jd) and math.isfinite(vy_am) and vy_am > 0):
                    continue
                t = Time(jd, format="jd", scale="utc", location=loc)
                altaz = coord.transform_to(AltAz(obstime=t, location=loc))
                alt = float(altaz.alt.deg)
                if alt <= 0:
                    continue
                ind_am = float(altaz.secz)
                diffs.append(abs(ind_am - vy_am))
                n += 1

        if not diffs:
            return TestResult(name, False, float("nan"), thresh, 0, "SKIP: no airmass pairs", [])
        med = float(np.median(diffs))
        return TestResult(
            name,
            med < thresh,
            med,
            thresh,
            n,
            f"median |d_airmass| = {med:.4f}",
            [],
        )
    except Exception as exc:
        return TestResult(name, False, float("nan"), thresh, 0, f"SKIP: {exc}", [])


def test_08_plate_scale(vyvar: dict, fits_dir: Path) -> TestResult:
    name = "Plate scale"
    thresh_pct = MAX_PLATE_SCALE_DIFF_PCT / 100.0
    try:
        master = fits_dir / "MASTERSTAR.fits"
        if not master.is_file():
            master = PLATESOLVE_DIR / "MASTERSTAR.fits"
        if not master.is_file():
            return TestResult(name, False, float("nan"), thresh_pct, 0, "SKIP: MASTERSTAR missing", [])

        _, hdr = load_fits_frame(master)
        plate_wcs = _plate_scale_arcsec_from_wcs(hdr)
        vy_ps = _vyvar_plate_scale_arcsec()
        warnings: list[str] = []
        wcs_sane = PLATE_SCALE_SANE_LO <= plate_wcs <= PLATE_SCALE_SANE_HI
        if not wcs_sane:
            warnings.append(
                f"WCS PC/CD scale {plate_wcs:.3f} arcsec/px outside "
                f"{PLATE_SCALE_SANE_LO}-{PLATE_SCALE_SANE_HI}; VYVAR uses config override"
            )
            rel = abs(vy_ps - vy_ps) / vy_ps if vy_ps > 0 else 0.0
            passed = PLATE_SCALE_SANE_LO <= vy_ps <= PLATE_SCALE_SANE_HI
            metric_pct = 0.0 if passed else float("nan")
            detail = (
                f"VYVAR config={vy_ps:.4f} arcsec/px; WCS PC/CD={plate_wcs:.4f} "
                "(ignored, matches _resolve_plate_scale_arcsec_per_px)"
            )
        else:
            rel = abs(plate_wcs - vy_ps) / vy_ps if vy_ps > 0 else float("nan")
            passed = math.isfinite(rel) and rel < thresh_pct
            metric_pct = rel * 100.0 if math.isfinite(rel) else float("nan")
            detail = f"WCS PC/CD={plate_wcs:.4f} vs VYVAR={vy_ps:.4f} ({metric_pct:.2f}%)"
        return TestResult(
            name,
            passed,
            metric_pct,
            MAX_PLATE_SCALE_DIFF_PCT,
            1,
            detail,
            warnings,
        )
    except Exception as exc:
        return TestResult(name, False, float("nan"), thresh_pct, 0, f"SKIP: {exc}", [])


def test_09_comp_stability(vyvar: dict, fits_dir: Path) -> TestResult:
    name = "Comp stability"
    thresh = MAX_COMP_RMS_DIFF_MMAG / 1000.0
    try:
        comp = vyvar["comp"]
        active = vyvar["active"]
        if comp.empty or "comp_rms" not in comp.columns:
            return TestResult(name, False, float("nan"), thresh, 0, "SKIP: no comp_rms", [])

        top = comp.dropna(subset=["comp_rms"]).nlargest(10, "comp_rms")
        comp_ids = top["catalog_id"].astype(str).tolist()
        frames = get_fits_frames(fits_dir, n=10 if _QUICK_MODE else 25)

        positions: dict[str, tuple[float, float, float]] = {}
        for cid in comp_ids:
            row = comp[comp["catalog_id"].astype(str) == cid]
            if row.empty:
                continue
            try:
                x = float(row.iloc[0]["x"])
                y = float(row.iloc[0]["y"])
                r = float(row.iloc[0].get("aperture_r_px", 3.0))
            except Exception:
                continue
            if math.isfinite(x) and math.isfinite(y):
                positions[cid] = (x, y, r)

        lc_mags: dict[str, list[float]] = {cid: [] for cid in positions}
        for fpath in frames:
            data, hdr = load_fits_frame(fpath)
            exptime = float(hdr.get("EXPTIME", hdr.get("EXPOSURE", 1.0)) or 1.0)
            for cid, (x, y, r) in positions.items():
                flux, _ = _aperture_photometry_frame(data, x, y, r)
                if flux > 0:
                    lc_mags[cid].append(-2.5 * math.log10(flux / exptime))

        diffs: list[float] = []
        for cid in comp_ids:
            mags = lc_mags.get(cid, [])
            if len(mags) < 5:
                continue
            ind_rms = float(np.std(mags))
            vy_row = comp[comp["catalog_id"].astype(str) == cid]
            vy_rms = float(pd.to_numeric(vy_row.iloc[0]["comp_rms"], errors="coerce"))
            if math.isfinite(ind_rms) and math.isfinite(vy_rms):
                diffs.append(abs(ind_rms - vy_rms))

        if not diffs:
            return TestResult(name, False, float("nan"), thresh, 0, "SKIP: no comp RMS pairs", [])
        med = float(np.median(diffs))
        return TestResult(
            name,
            med < thresh,
            med,
            thresh,
            len(diffs),
            f"median |d_comp_rms| = {med * 1000:.2f} mmag",
            [],
        )
    except Exception as exc:
        return TestResult(name, False, float("nan"), thresh, 0, f"SKIP: {exc}", [])


def test_10_variability(vyvar: dict) -> TestResult:
    name = "Variability agreement"
    thresh = MIN_VARIABILITY_AGREEMENT
    try:
        summary = vyvar["summary"]
        if summary.empty or "lc_rms" not in summary.columns:
            return TestResult(name, False, 0.0, thresh, 0, "SKIP: no summary lc_rms", [])

        comp = vyvar["comp"]
        comp_rms_med = float("nan")
        if not comp.empty and "comp_rms" in comp.columns:
            comp_rms_med = float(pd.to_numeric(comp["comp_rms"], errors="coerce").median())
        if not math.isfinite(comp_rms_med):
            comp_rms_med = float(pd.to_numeric(summary["lc_rms"], errors="coerce").median())

        agree = 0
        total = 0
        for _, row in summary.iterrows():
            zf = str(row.get("zone_flag", "")).lower()
            lc_rms = float(pd.to_numeric(row.get("lc_rms", float("nan")), errors="coerce"))
            if not math.isfinite(lc_rms):
                continue
            vy_var = zf not in ("linear", "good", "")
            ind_var = lc_rms > 3.0 * comp_rms_med if math.isfinite(comp_rms_med) else False
            if vy_var == ind_var:
                agree += 1
            total += 1

        if total == 0:
            return TestResult(name, False, 0.0, thresh, 0, "SKIP: no targets", [])
        frac = agree / total
        return TestResult(
            name,
            frac >= thresh,
            frac,
            thresh,
            total,
            f"agreement {frac * 100:.1f}% ({agree}/{total})",
            ["ROT/noisy stars may disagree by design"],
        )
    except Exception as exc:
        return TestResult(name, False, 0.0, thresh, 0, f"SKIP: {exc}", [])


# ── MAIN ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    global DRAFT_DIR, FITS_DIR, PHOT_DIR, PLATESOLVE_DIR, GAIA_DB, _QUICK_MODE

    parser = argparse.ArgumentParser(description="Cross-validate VYVAR draft_342")
    parser.add_argument("--draft-dir", type=Path, default=None)
    parser.add_argument("--gaia-db", type=Path, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 10 frames for T05/T09 instead of all 139",
    )
    args = parser.parse_args(argv)
    _QUICK_MODE = bool(args.quick)

    if args.draft_dir is not None:
        DRAFT_DIR = Path(args.draft_dir)
        FITS_DIR = DRAFT_DIR / "detrended_aligned/lights/NoFilter_60_2"
        PHOT_DIR = DRAFT_DIR / "platesolve/NoFilter_60_2/photometry"
        PLATESOLVE_DIR = DRAFT_DIR / "platesolve/NoFilter_60_2"
    if args.gaia_db is not None:
        GAIA_DB = Path(args.gaia_db)

    if not PHOT_DIR.is_dir():
        print(f"ERROR: photometry dir not found: {PHOT_DIR}", file=sys.stderr)
        return 2

    vyvar = load_vyvar_outputs(DRAFT_DIR, PHOT_DIR)

    tests = [
        lambda: test_01_astrometry(vyvar, FITS_DIR),
        lambda: test_02_aperture_flux(vyvar, FITS_DIR),
        lambda: test_03_sky_background(vyvar, FITS_DIR),
        lambda: test_04_fwhm(vyvar, FITS_DIR),
        lambda: test_05_differential_lc(vyvar, FITS_DIR),
        lambda: test_06_bjd_hjd(vyvar, FITS_DIR),
        lambda: test_07_airmass(vyvar),
        lambda: test_08_plate_scale(vyvar, FITS_DIR),
        lambda: test_09_comp_stability(vyvar, FITS_DIR),
        lambda: test_10_variability(vyvar),
    ]

    results: list[TestResult] = []
    for i, fn in enumerate(tests, 1):
        print(f"T{i:02d} running...", flush=True)
        try:
            results.append(fn())
        except Exception as exc:
            results.append(
                TestResult(
                    f"T{i:02d}",
                    False,
                    float("nan"),
                    float("nan"),
                    0,
                    f"SKIP: {exc}",
                    [],
                )
            )

    stamp = datetime.now().strftime("%Y%m%d")
    csv_path = SCRIPT_DIR / f"cross_val_results_{stamp}.csv"
    md_path = SCRIPT_DIR / f"cross_val_report_{stamp}.md"

    save_csv_results(results, csv_path)

    import io

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    print_markdown_report(results)
    sys.stdout = old_stdout
    report_text = buf.getvalue()
    try:
        print(report_text)
    except UnicodeEncodeError:
        print(report_text.encode("ascii", errors="replace").decode("ascii"))
    md_path.write_text(report_text, encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")

    all_pass = all(r.passed for r in results)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
