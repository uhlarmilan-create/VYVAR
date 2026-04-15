"""Photometry helpers: stress-tests, QA summaries, aperture photometry, BPM flags.

Sidecar CSVs under ``platesolve/per_frame_catalogs`` may use DAO flux or aperture-subtracted flux
when :class:`config.AppConfig`.aperture_photometry_enabled is true.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils import fits_binning_xy_from_header

# Stĺpce načítavané z per-frame CSV pre bootstrap (78 % úspora pamäte)
_PHASE_USECOLS_PERFRAME: list[str] = [
    "name",
    "catalog_id",
    "bjd_tdb_mid",
    "flux",
    "dao_flux",
    "noise_floor_adu",
    "aperture_r_px",
    "is_usable",
    "is_saturated",
    "is_noisy",
    "snr50_ok",
    "vsx_known_variable",
    "likely_saturated",
]


def _angular_distance_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Uhlová vzdialenosť v stupňoch (haversine)."""
    r1, d1, r2, d2 = map(math.radians, [ra1, dec1, ra2, dec2])
    a = (
        math.sin((d2 - d1) / 2) ** 2
        + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2) ** 2
    )
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(a))))


def _normalize_id_value(x: Any) -> str:
    """Normalize Gaia-like IDs loaded as floats; keep non-numeric strings."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    try:
        return str(int(float(s)))
    except Exception:  # noqa: BLE001
        return s


def _normalize_id_series(s: pd.Series) -> pd.Series:
    return s.apply(_normalize_id_value)


def _bool_col(series: pd.Series) -> pd.Series:
    """Normalizuje stĺpec na bool bez ohľadu na True/False/'true'/'false'/1/0."""
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes", "y"))


@dataclass(frozen=True)
class StressTestResult:
    per_source_rms: dict[str, float]
    frames_sampled: int
    frames_used: int


def _iter_fits_recursive(root: Path) -> list[Path]:
    root = Path(root)
    out: list[Path] = []
    for p in root.rglob("*.fits"):
        if p.is_file():
            out.append(p)
    for p in root.rglob("*.fit"):
        if p.is_file():
            out.append(p)
    out.sort()
    return out


def stress_test_relative_rms_from_sidecars(
    *,
    frames_root: Path,
    source_ids: list[str],
    sample_frac: float = 0.10,
    seed: int = 42,
    flux_col: str = "flux",
    name_col: str = "name",
    min_stars_per_frame: int = 3,
) -> StressTestResult:
    """Compute relative RMS for many sources on a random frame sample.

    For each sampled frame with a sidecar CSV, compute per-frame ensemble median among present sources,
    then record relative flux for each star: f_i / median(f_all). Returns RMS over time for each star.
    """
    root = Path(frames_root)
    files = [fp for fp in _iter_fits_recursive(root) if fp.with_suffix(".csv").is_file()]
    if not files or not source_ids:
        return StressTestResult(per_source_rms={}, frames_sampled=0, frames_used=0)

    frac = float(sample_frac)
    frac = 0.10 if not math.isfinite(frac) else max(0.01, min(1.0, frac))
    k = max(1, int(round(len(files) * frac)))
    rnd = random.Random(int(seed))
    sample = rnd.sample(files, k=min(k, len(files)))

    want = [str(x).strip() for x in source_ids if str(x).strip()]
    want_set = set(want)
    rel_lists: dict[str, list[float]] = {nm: [] for nm in want}

    frames_used = 0
    for fp in sample:
        sidecar = fp.with_suffix(".csv")
        try:
            dff = pd.read_csv(sidecar)
        except Exception:
            continue
        if name_col not in dff.columns or flux_col not in dff.columns:
            continue
        names = dff[name_col].astype(str).str.strip()
        flux = pd.to_numeric(dff[flux_col], errors="coerce")
        mask = names.isin(want_set) & flux.notna() & (flux.astype(float) > 0)
        if not bool(mask.any()):
            continue
        sub = dff.loc[mask, [name_col, flux_col]].copy()
        sub[name_col] = sub[name_col].astype(str).str.strip()
        sub[flux_col] = pd.to_numeric(sub[flux_col], errors="coerce").astype(float)
        sub = sub.dropna()
        if len(sub) < int(min_stars_per_frame):
            continue
        med = float(sub[flux_col].median())
        if not math.isfinite(med) or med <= 0:
            continue
        frames_used += 1
        for _, row in sub.iterrows():
            nm = str(row[name_col]).strip()
            if nm in rel_lists:
                rel_lists[nm].append(float(row[flux_col]) / med)

    out: dict[str, float] = {}
    for nm, arr in rel_lists.items():
        if len(arr) < 3:
            continue
        mu = 1.0
        rms = math.sqrt(sum((x - mu) ** 2 for x in arr) / float(len(arr)))
        if math.isfinite(rms):
            out[nm] = float(rms)
    return StressTestResult(per_source_rms=out, frames_sampled=int(len(sample)), frames_used=int(frames_used))


def vsx_is_known_variable_top3_per_bin(
    *,
    rows: list[dict[str, Any]],
    phot_category_key: str = "phot_category",
    rms_key: str = "stress_rms",
    ra_key: str = "ra",
    dec_key: str = "dec",
    max_per_bin: int = 3,
    radius_arcsec: float = 2.0,
) -> set[str]:
    """Return set of Gaia source_id strings that are present in VSX near the best (lowest RMS) stars per bin."""
    try:
        from astroquery.vizier import Vizier  # type: ignore
        import astropy.units as u
        from astropy.coordinates import SkyCoord
    except Exception:
        return set()

    by_bin: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        b = str(r.get(phot_category_key) or "").strip()
        sid = str(r.get("source_id_gaia") or "").strip()
        if not b or not sid:
            continue
        v = r.get(rms_key)
        try:
            rms = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(rms):
            continue
        by_bin.setdefault(b, []).append(r)

    viz = Vizier(row_limit=50)
    flagged: set[str] = set()
    for b, items in by_bin.items():
        items_sorted = sorted(items, key=lambda x: float(x.get(rms_key)))
        for r in items_sorted[: int(max_per_bin)]:
            sid = str(r.get("source_id_gaia") or "").strip()
            try:
                ra = float(r.get(ra_key))
                de = float(r.get(dec_key))
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(ra) and math.isfinite(de)):
                continue
            c = SkyCoord(ra=ra * u.deg, dec=de * u.deg, frame="icrs")
            try:
                t = viz.query_region(c, radius=float(radius_arcsec) * u.arcsec, catalog="B/vsx")
            except Exception:
                continue
            if t and len(t) > 0 and len(t[0]) > 0:
                flagged.add(sid)
    return flagged


def common_field_intersection_bbox_px(
    *,
    frame_paths: list[Path],
    finite_stride: int = 16,
) -> tuple[float, float, float, float] | None:
    """Compute intersection bbox of finite pixels across frames (x0,y0,x1,y1).

    Intended for WCS-reprojected aligned frames where uncovered regions are NaN.
    Uses strided sampling for speed.
    """
    try:
        import numpy as np
        from astropy.io import fits
    except Exception:  # noqa: BLE001
        return None

    fps = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(fps) < 2:
        return None

    x0_i, y0_i = 0.0, 0.0
    x1_i, y1_i = float("inf"), float("inf")
    stride = max(1, int(finite_stride))

    for fp in fps:
        try:
            with fits.open(fp, memmap=False) as hdul:
                data = np.asarray(hdul[0].data, dtype=np.float32)
        except Exception:  # noqa: BLE001
            continue
        if data.ndim != 2:
            continue
        samp = data[::stride, ::stride]
        fin = np.isfinite(samp)
        if not bool(fin.any()):
            continue
        ys, xs = np.where(fin)
        x0 = float(xs.min() * stride)
        y0 = float(ys.min() * stride)
        x1 = float(min(data.shape[1] - 1, xs.max() * stride + (stride - 1)))
        y1 = float(min(data.shape[0] - 1, ys.max() * stride + (stride - 1)))
        x0_i = max(x0_i, x0)
        y0_i = max(y0_i, y0)
        x1_i = min(x1_i, x1)
        y1_i = min(y1_i, y1)

    if not (math.isfinite(x0_i) and math.isfinite(y0_i) and math.isfinite(x1_i) and math.isfinite(y1_i)):
        return None
    if x1_i <= x0_i or y1_i <= y0_i:
        return None
    return (x0_i, y0_i, x1_i, y1_i)


def recommended_aperture_by_color(
    *,
    bp_rp: float | None,
    median_fwhm_blue: float | None,
    median_fwhm_neutral: float | None,
    median_fwhm_red: float | None,
) -> float | None:
    """Return 2.5× median FWHM for the star's coarse color category."""
    if bp_rp is None:
        return None
    try:
        c = float(bp_rp)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(c):
        return None
    if c < 0.5:
        f = median_fwhm_blue
    elif c <= 1.5:
        f = median_fwhm_neutral
    else:
        f = median_fwhm_red
    if f is None:
        return None
    try:
        fv = float(f)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fv) or fv <= 0:
        return None
    return 2.5 * fv



def bad_columns_for_light_frame(
    bpm: dict[str, Any] | None,
    *,
    light_header: Any,
) -> set[int]:
    """Map native ``bad_x`` from BPM JSON to 0-based integer column indices in the light frame."""
    if not bpm or not isinstance(bpm, dict):
        return set()
    raw = bpm.get("bad_x")
    if not raw:
        return set()
    try:
        lb_x, _ = fits_binning_xy_from_header(light_header)
    except Exception:  # noqa: BLE001
        lb_x = 1
    lb_x = max(1, int(lb_x))
    mb = int(bpm.get("native_binning") or 1)
    mb = max(1, mb)
    factor = max(1, lb_x // mb)
    out: set[int] = set()
    for x in raw:
        try:
            xi = int(x)
        except (TypeError, ValueError):
            continue
        out.add(int(xi // factor))
    return out


def _fwhm_moment_at(arr: np.ndarray, xc: float, yc: float, *, half: int = 6) -> float:
    """2D Gaussian moment FWHM estimate (same recipe as pipeline MASTERSTAR block)."""
    if not (math.isfinite(xc) and math.isfinite(yc)):
        return float("nan")
    xi = int(round(float(xc)))
    yi = int(round(float(yc)))
    h, w = int(arr.shape[0]), int(arr.shape[1])
    x0 = max(0, xi - half)
    x1 = min(w - 1, xi + half)
    y0 = max(0, yi - half)
    y1 = min(h - 1, yi + half)
    if x1 <= x0 or y1 <= y0:
        return float("nan")
    patch = arr[y0 : y1 + 1, x0 : x1 + 1].astype(np.float64, copy=False)
    if patch.size < 9:
        return float("nan")
    medp = float(np.nanmedian(patch))
    wgt = patch - medp
    wgt[~np.isfinite(wgt)] = 0.0
    wgt[wgt < 0] = 0.0
    s = float(wgt.sum())
    if not math.isfinite(s) or s <= 0:
        return float("nan")
    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    mx = float((wgt * xx).sum() / s)
    my = float((wgt * yy).sum() / s)
    vx = float((wgt * (xx - mx) ** 2).sum() / s)
    vy = float((wgt * (yy - my) ** 2).sum() / s)
    if not (vx > 0 and vy > 0 and math.isfinite(vx) and math.isfinite(vy)):
        return float("nan")
    sigx = math.sqrt(vx)
    sigy = math.sqrt(vy)
    fwhm = 2.355 * 0.5 * (sigx + sigy)
    return float(fwhm) if math.isfinite(fwhm) else float("nan")


def _moment_to_gaussian_fwhm(moment_fwhm_px: float) -> float:
    """Convert moment-FWHM proxy to Gaussian FWHM (empirical calibration)."""
    MOMENT_TO_GAUSSIAN = 0.47
    if not (math.isfinite(moment_fwhm_px) and moment_fwhm_px > 0):
        return float("nan")
    return float(moment_fwhm_px) * float(MOMENT_TO_GAUSSIAN)



def enhance_catalog_dataframe_aperture_bpm(
    df: pd.DataFrame,
    data: np.ndarray,
    hdr: Any,
    *,
    aperture_enabled: bool,
    aperture_fwhm_factor: float,
    annulus_inner_fwhm: float,
    annulus_outer_fwhm: float,
    nonlinearity_peak_percentile: float,
    nonlinearity_fwhm_ratio: float,
    master_dark_path: Path | str | None,
) -> pd.DataFrame:
    """Replace DAO ``flux`` with aperture photometry when enabled; add linearity/BPM flags."""
    out = df.copy()
    arr = np.asarray(data, dtype=np.float32)

    x = pd.to_numeric(out.get("x"), errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(out.get("y"), errors="coerce").to_numpy(dtype=np.float64)
    n = len(out)
    if n == 0:
        return out

    # Pôvodný DAO flux z detect_stars_and_match_catalog (historicky v stĺpci ``flux``).
    # Pre ďalšie fázy chceme mať konzistentne:
    # - ``flux_raw``: pôvodný flux (čo prišlo z DAO bloku)
    # - ``dao_flux``: sky-subtrahovaný flux (po aperturnej fotometrii, ak je zapnutá)
    flux_raw = pd.to_numeric(out.get("flux"), errors="coerce").to_numpy(dtype=np.float64)
    if "flux_raw" not in out.columns:
        out["flux_raw"] = flux_raw
    if "dao_flux" not in out.columns:
        out["dao_flux"] = flux_raw

    fwhm_per = np.array(
        [_fwhm_moment_at(arr, float(x[i]), float(y[i])) for i in range(n)],
        dtype=np.float64,
    )
    out["fwhm_estimate_px"] = fwhm_per

    fwhm_moment_med = float(np.nanmedian(fwhm_per[np.isfinite(fwhm_per) & (fwhm_per > 0)]))
    if not math.isfinite(fwhm_moment_med) or fwhm_moment_med <= 0:
        fwhm_moment_med = float("nan")

    fwhm_gaussian = _moment_to_gaussian_fwhm(fwhm_moment_med)
    out["fwhm_gaussian_px"] = float(fwhm_gaussian) if math.isfinite(fwhm_gaussian) else float("nan")

    if aperture_enabled and math.isfinite(fwhm_gaussian) and fwhm_gaussian > 0:
        try:
            # Lokálna implementácia: sky-subtracted flux cez CircularAperture + CircularAnnulus.
            from photutils.aperture import CircularAnnulus, CircularAperture
            from photutils.aperture import aperture_photometry as _aphot

            fw = float(fwhm_gaussian)
            r_ap = max(0.5, float(aperture_fwhm_factor) * fw)
            r_in = max(r_ap + 0.5, float(annulus_inner_fwhm) * fw)
            r_out = max(r_in + 0.5, float(annulus_outer_fwhm) * fw)

            pos = np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)])
            ap = CircularAperture(pos, r=r_ap)
            an = CircularAnnulus(pos, r_in=r_in, r_out=r_out)

            d = np.asarray(arr, dtype=np.float64)
            if np.any(~np.isfinite(d)):
                fill = float(np.nanmedian(d)) if np.any(np.isfinite(d)) else 0.0
                d = np.where(np.isfinite(d), d, fill)

            # Len aperture sum (annulus sky cez medián pixelov, nie sum/area).
            phot_ap = _aphot(d, ap)
            sum_ap = np.asarray(phot_ap["aperture_sum"], dtype=np.float64)
            area_ap = float(ap.area)

            sky_pp_arr = np.zeros(n, dtype=np.float64)
            ann_masks = an.to_mask(method="center")
            if not isinstance(ann_masks, (list, tuple)):
                ann_masks = [ann_masks]
            for i, amask in enumerate(ann_masks):
                try:
                    ann_img = amask.to_image(d.shape)
                    sky_pixels = d[ann_img > 0]
                    if sky_pixels.size >= 5:
                        sky_pp_arr[i] = float(np.median(sky_pixels))
                    else:
                        sky_pp_arr[i] = float(np.median(d))
                except Exception:  # noqa: BLE001
                    sky_pp_arr[i] = float(np.median(d))

            flux_arr = sum_ap - sky_pp_arr * area_ap
            out["flux"] = flux_arr.astype(np.float64)
            out["dao_flux"] = out["flux"]
            out["aperture_r_px"] = float(r_ap)
            out["sky_annulus_r_in_px"] = float(r_in)
            out["sky_annulus_r_out_px"] = float(r_out)
            # Uložíme sky_pp per hviezda (nie globálna konštanta)
            out["noise_floor_adu"] = sky_pp_arr.astype(np.float64)
        except Exception:  # noqa: BLE001
            out["dao_flux"] = flux_raw
            out["flux"] = flux_raw
    else:
        out["dao_flux"] = flux_raw
        out["flux"] = flux_raw

    peak = pd.to_numeric(out.get("peak_max_adu"), errors="coerce").to_numpy(dtype=np.float64)
    finite_pk = peak[np.isfinite(peak)]
    thr_pk = float("nan")
    if finite_pk.size > 0:
        pct = min(100.0, max(0.0, 100.0 - float(nonlinearity_peak_percentile)))
        thr_pk = float(np.percentile(finite_pk, pct))

    ratio = float(nonlinearity_fwhm_ratio)
    likely_nl = np.zeros(n, dtype=bool)
    for i in range(n):
        if not (math.isfinite(fwhm_per[i]) and math.isfinite(fwhm_moment_med) and fwhm_moment_med > 0):
            continue
        if not (math.isfinite(peak[i]) and math.isfinite(thr_pk) and peak[i] >= thr_pk):
            continue
        if fwhm_per[i] > ratio * fwhm_moment_med:
            likely_nl[i] = True
    out["likely_nonlinear"] = likely_nl

    bpm_path = None
    bpm: dict[str, Any] | None = None
    if master_dark_path:
        mp = Path(str(master_dark_path))
        bpm_path = mp.parent / f"{mp.stem}_dark_bpm.json"
        if bpm_path.is_file():
            try:
                bpm = json.loads(bpm_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                bpm = None

    bad_x = bad_columns_for_light_frame(bpm, light_header=hdr)
    on_bad = np.zeros(n, dtype=bool)
    if bad_x:
        for i in range(n):
            if not np.isfinite(x[i]):
                continue
            xi = int(round(float(x[i])))
            if xi in bad_x:
                on_bad[i] = True
    out["on_bad_column"] = on_bad

    if "photometry_ok" in out.columns:
        base_ok = out["photometry_ok"].fillna(True).astype(bool).to_numpy()
        out["photometry_ok"] = base_ok & (~likely_nl) & (~on_bad)
    else:
        out["photometry_ok"] = ~(likely_nl | on_bad)

    return out


def select_active_targets(
    variable_targets_csv: Path,
    masterstars_csv: Path,
    *,
    frame_w_px: int = 2082,
    frame_h_px: int = 1397,
    edge_margin_px: int = 50,
    match_radius_arcsec: float = 15.0,
) -> pd.DataFrame:
    """Fáza 0: Filtruj VSX premenné → active_targets.

    Pravidlá:
    - Hviezda musí byť v snímke (x, y v medziach s okrajom)
    - Musí byť nájdená v masterstars_full_match.csv (cross-match < match_radius_arcsec)
    - masterstars záznam musí mať is_usable=True
    - Nesmie byť saturovaná (is_saturated=False)

    Returns:
        DataFrame s active targets — stĺpce z variable_targets + pridané zo masterstars:
        [name, catalog_id, ra_deg, dec_deg, vsx_name, vsx_type, vsx_period,
         x, y, mag, b_v, bp_rp, snr50_ok, is_usable, zone]
    """
    vt = pd.read_csv(variable_targets_csv, low_memory=False)
    ms = pd.read_csv(masterstars_csv, low_memory=False)
    # Normalizuj Gaia ID na string (pandas číta veľké int ako float)
    for _id_col in ("catalog_id", "name"):
        if _id_col in ms.columns:
            ms[_id_col] = _normalize_id_series(ms[_id_col])

    # Normalizuj bool stĺpce v masterstars
    for col in ("is_usable", "is_saturated", "is_noisy", "snr50_ok", "likely_saturated"):
        if col in ms.columns:
            ms[col] = _bool_col(ms[col])

    # Filter: v snímke s okrajom
    vt["x"] = pd.to_numeric(vt["x"], errors="coerce")
    vt["y"] = pd.to_numeric(vt["y"], errors="coerce")
    in_frame = (
        vt["x"].between(edge_margin_px, frame_w_px - edge_margin_px)
        & vt["y"].between(edge_margin_px, frame_h_px - edge_margin_px)
    )
    vt_in = vt[in_frame].copy()

    # Cross-match s masterstars cez ra/dec
    ms["ra_deg"] = pd.to_numeric(ms["ra_deg"], errors="coerce")
    ms["dec_deg"] = pd.to_numeric(ms["dec_deg"], errors="coerce")
    vt_in["ra_deg"] = pd.to_numeric(vt_in["ra_deg"], errors="coerce")
    vt_in["dec_deg"] = pd.to_numeric(vt_in["dec_deg"], errors="coerce")

    ms_arr = ms[["ra_deg", "dec_deg"]].to_numpy(dtype=float)

    matched_rows: list[dict] = []
    for _, vrow in vt_in.iterrows():
        ra_v = float(vrow["ra_deg"])
        dec_v = float(vrow["dec_deg"])
        if not (math.isfinite(ra_v) and math.isfinite(dec_v)):
            continue
        # Nájdi najbližší záznam v masterstars
        dists = [
            _angular_distance_deg(ra_v, dec_v, float(ms_arr[i, 0]), float(ms_arr[i, 1]))
            for i in range(len(ms_arr))
        ]
        best_idx = int(np.argmin(dists))
        best_dist_arcsec = dists[best_idx] * 3600.0
        if best_dist_arcsec > match_radius_arcsec:
            continue
        ms_row = ms.iloc[best_idx]
        zone_val = str(ms_row.get("zone", "")).strip().lower()
        # Zahrnúť: linear (plne použiteľná) + noisy1 (slabší signál, možná premenná)
        # Vylúčiť: saturated, noisy2, noisy3, prázdna zóna
        if zone_val in ("noisy2", "noisy3", "saturated", ""):
            continue
        if zone_val not in ("linear", "noisy1"):
            # Fallback pre staré CSV bez noisy sub-kategórií
            if not bool(ms_row.get("is_usable", False)):
                continue
            if bool(ms_row.get("is_saturated", False)):
                continue
        rec = {
            "name": vrow.get("name", ""),
            "vsx_name": vrow.get("vsx_name", ""),
            "vsx_type": vrow.get("vsx_type", ""),
            "vsx_period": vrow.get("vsx_period", ""),
            "priority": vrow.get("priority", 1),
            "ra_deg": ra_v,
            "dec_deg": dec_v,
            "x": float(vrow["x"]),
            "y": float(vrow["y"]),
            "catalog_id": str(ms_row.get("catalog_id", ms_row.get("name", ""))).strip(),
            "mag": float(ms_row.get("mag", float("nan"))),
            "b_v": float(ms_row.get("b_v", float("nan"))),
            "bp_rp": float(ms_row.get("bp_rp", float("nan"))),
            "snr50_ok": bool(ms_row.get("snr50_ok", False)),
            "zone": str(ms_row.get("zone", "")),
            "is_usable": bool(ms_row.get("is_usable", False)),
            "match_dist_arcsec": best_dist_arcsec,
        }
        matched_rows.append(rec)

    if not matched_rows:
        return pd.DataFrame()

    result = pd.DataFrame(matched_rows)
    # Zabezpeč string formát Gaia ID v output CSV
    if "catalog_id" in result.columns:
        def _gaia_id_str(x: Any) -> str:
            s = str(x).strip()
            if s in ("", "nan"):
                return ""
            try:
                return str(int(float(s)))
            except Exception:  # noqa: BLE001
                return s

        result["catalog_id"] = result["catalog_id"].apply(_gaia_id_str)
    logging.info(
        f"[FÁZA 0] active_targets: {len(result)} / {len(vt)} VSX hviezd "
        f"(in_frame={int(in_frame.sum())}, matched+usable={len(result)})"
    )
    return result.reset_index(drop=True)


def select_comparison_stars_per_target(
    target: pd.Series,
    masterstars_df: pd.DataFrame,
    per_frame_csv_paths: list[Path],
    *,
    fwhm_px: float = 3.7,
    max_dist_deg: float = 1.5,
    max_mag_diff: float = 0.25,  # ±0.25 mag od targetu
    max_bv_diff: float = 0.15,  # ±0.15 B-V od targetu
    n_comp_min: int = 3,
    n_comp_max: int = 5,
    max_comp_rms: float = 0.05,
    min_dist_arcsec: float = 30.0,
    min_frames_frac: float = 0.3,
    rms_outlier_sigma: float = 3.0,
    exclude_gaia_nss: bool = True,
    exclude_gaia_extobj: bool = True,
    max_psf_chi2: float = 3.0,
    max_fwhm_factor: float = 1.5,
    isolation_radius_px: float = 25.0,
    flux_col: str = "dao_flux",
) -> pd.DataFrame:
    """Fáza 1: Pre jeden target vyber najstabilnejšie porovnávacie hviezdy.

    Postup (Možnosť D = B + C):
    1. Priestorový + fotometrický filter kandidátov z masterstars
    2. Načítaj flux zo všetkých per-frame CSV (len _PHASE_USECOLS_PERFRAME)
    3. Normalizuj flux voči ensemble mediánu per snímka
    4. Vypočítaj RMS scatter pre každého kandidáta
    5. Iteratívny ensemble filter — vyraď top outlierov kým RMS neklesá
    6. Vráť top n_comp_max najstabilnejších (min n_comp_min)

    Args:
        exclude_gaia_nss: Vylúč Gaia non-single stars (binárky, vizuálne dvojhviezdy).
            Tieto majú variabilný flux nezávislý od počasia → scatter comp hviezdy.
        exclude_gaia_extobj: Vylúč Gaia QSO a galaxie (gaia_qso, gaia_gal).
            Nie sú bodové zdroje → systematické chyby v aperturnej fotometrii.
        max_psf_chi2: Maximálny mediánový PSF chi² kandidáta cez všetky snímky.
            Vysoké chi² = profil nie je čistý Gaussian = blend alebo rozšírený zdroj.
            Použije sa len ak je stĺpec psf_chi2 dostupný v per-frame CSV.
            Nastavenie na float("inf") filter vypne.
        max_fwhm_factor: Maximálny pomer fwhm_estimate_px kandidáta voči mediánu
            všetkých hviezd na snímke. Hodnota > 1.5 indikuje blend dvoch blízkych
            hviezd. Použije sa len ak je stĺpec fwhm_estimate_px dostupný.
            Nastavenie na float("inf") filter vypne.
        isolation_radius_px: Polomer v pixeloch pre výpočet contamination indexu.
            Súčet flux susedov / flux kandidáta v tomto polomere = contamination.
            Výsledok vstupuje do combined score (soft penalizácia, nie hard exclusion).
            Nastavenie na 0.0 vypne crowding penalizáciu úplne.
        max_comp_rms: Maximálny povolený p2p RMS scatter comp hviezdy (mag).
            Hviezdy s RMS > max_comp_rms sú odmietnuté bez ohľadu na ranking.
            Default 0.05 mag (50 ppt) — štandardná fotometrická stabilita.
        min_dist_arcsec: Minimálna vzdialenosť comp hviezdy od targetu v oblúkových
            sekundách. Zabraňuje PSF overlap pri veľmi blízkych hviezdach.
            Default 30 arcsec (~10px pri typickom scale 3 arcsec/px).

    Returns:
        DataFrame s porovnávacími hviezdami pre tento target, zoradený podľa RMS ASC.
        Prázdny DataFrame ak sa nenájde dostatok stabilných hviezd.
    """
    _ = fwhm_px
    ms = masterstars_df.copy()
    # Normalizuj Gaia ID na string
    for _id_col in ("catalog_id", "name"):
        if _id_col in ms.columns:
            ms[_id_col] = _normalize_id_series(ms[_id_col])

    # Normalizuj bool stĺpce
    for col in (
        "is_usable",
        "is_saturated",
        "is_noisy",
        "snr50_ok",
        "vsx_known_variable",
        "likely_saturated",
    ):
        if col in ms.columns:
            ms[col] = _bool_col(ms[col])

    ra_t = float(target["ra_deg"])
    dec_t = float(target["dec_deg"])
    mag_t = float(target.get("mag", float("nan")))
    target_cid = str(target.get("catalog_id", ""))

    # ── Krok 1: Filter kandidátov ──
    ms["_dist_deg"] = ms.apply(
        lambda r: _angular_distance_deg(
            ra_t,
            dec_t,
            float(r["ra_deg"]) if math.isfinite(float(r["ra_deg"])) else 999.0,
            float(r["dec_deg"]) if math.isfinite(float(r["dec_deg"])) else 999.0,
        ),
        axis=1,
    )
    cand_mask = (
        ms["_dist_deg"].le(max_dist_deg)
        & _bool_col(ms.get("is_usable", pd.Series(True, index=ms.index)))
        & ~_bool_col(ms.get("is_saturated", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("is_noisy", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("vsx_known_variable", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("likely_saturated", pd.Series(False, index=ms.index)))
    )
    # Vylúč samotný target
    if target_cid:
        cand_mask &= ms.get("catalog_id", ms.get("name", pd.Series("", index=ms.index))).astype(str) != target_cid

    # Hard filter: minimálna vzdialenosť od targetu
    if math.isfinite(min_dist_arcsec) and min_dist_arcsec > 0:
        min_dist_deg = float(min_dist_arcsec) / 3600.0
        cand_mask &= ms["_dist_deg"].ge(min_dist_deg)

    # Hard filter: |ΔMag| <= max_mag_diff
    if math.isfinite(mag_t):
        ms["_mag"] = pd.to_numeric(ms.get("mag", ms.get("phot_g_mean_mag")), errors="coerce")
        cand_mask &= ms["_mag"].sub(mag_t).abs().le(max_mag_diff)

    # Hard filter: |ΔB-V| <= max_bv_diff (len ak poznáme B-V targetu aj kandidáta)
    target_bv_pre = float(target.get("b_v", float("nan")))
    if math.isfinite(target_bv_pre) and math.isfinite(max_bv_diff):
        ms["_bv"] = pd.to_numeric(ms.get("b_v", pd.Series(dtype=float)), errors="coerce")
        bv_known_mask = ms["_bv"].notna()
        bv_filter = ~bv_known_mask | ms["_bv"].sub(target_bv_pre).abs().le(max_bv_diff)
        cand_mask &= bv_filter
        n_bv_filtered = int((bv_known_mask & ~bv_filter).sum())
        if n_bv_filtered > 0:
            logging.debug(
                f"[FÁZA 1] Target {target_cid}: |ΔB-V| filter odstránil "
                f"{n_bv_filtered} kandidátov (threshold={max_bv_diff:.2f})"
            )

    # Filter A: Gaia objektové flagy
    _n_before_a = int(cand_mask.sum())

    # gaia_nss=True → non-single star (binárka/dvojhviezda) → variabilný flux
    if exclude_gaia_nss and "gaia_nss" in ms.columns:
        _nss_rej = cand_mask & _bool_col(ms["gaia_nss"])
        cand_mask &= ~_bool_col(ms["gaia_nss"])
        _n_rej = int(_nss_rej.sum())
        if _n_rej > 0:
            logging.info(
                f"[FÁZA 1] Target {target_cid}: Filter A (gaia_nss) vylúčil {_n_rej} kandidátov"
            )

    # gaia_qso, gaia_gal → nie bodový zdroj → systematické chyby
    if exclude_gaia_extobj:
        _rej_ext_total = 0
        for _ext_col in ("gaia_qso", "gaia_gal"):
            if _ext_col in ms.columns:
                _rej_mask = cand_mask & _bool_col(ms[_ext_col])
                cand_mask &= ~_bool_col(ms[_ext_col])
                _rej = int(_rej_mask.sum())
                _rej_ext_total += _rej
                if _rej > 0:
                    logging.info(
                        f"[FÁZA 1] Target {target_cid}: Filter A ({_ext_col}) vylúčil {_rej} kandidátov"
                    )

        if _rej_ext_total == 0:
            _ = _rej_ext_total  # noqa: B018

    _n_after_a = int(cand_mask.sum())
    _rej_a_total = _n_before_a - _n_after_a
    if _rej_a_total > 0:
        logging.debug(
            f"[FÁZA 1] Target {target_cid}: Filter A celkom vylúčil {_rej_a_total} kandidátov "
            f"({_n_before_a} → {_n_after_a})"
        )

    # Zahrň DET hviezdy (bez Gaia ID) ak majú snr50_ok a nie sú saturované.
    # Tieto môžu byť stabilné comp hviezdy aj bez katalógového záznamu.
    det_mask = (
        ms.get("catalog_id", ms.get("name", pd.Series("", index=ms.index)))
        .astype(str)
        .str.startswith("DET")
        & ms["_dist_deg"].le(max_dist_deg)
        & ~_bool_col(ms.get("is_saturated", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("likely_saturated", pd.Series(False, index=ms.index)))
        & _bool_col(ms.get("snr50_ok", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("vsx_known_variable", pd.Series(False, index=ms.index)))
    )
    if target_cid:
        det_mask &= (
            ms.get("catalog_id", ms.get("name", pd.Series("", index=ms.index))).astype(str)
            != target_cid
        )
    if math.isfinite(min_dist_arcsec) and min_dist_arcsec > 0:
        det_mask &= ms["_dist_deg"].ge(float(min_dist_arcsec) / 3600.0)

    cand_mask = cand_mask | det_mask

    candidates = ms[cand_mask].copy()

    # Ak po hard filtri nedostatok kandidátov — uvoľni B-V filter a skús znova
    if len(candidates) < n_comp_min and math.isfinite(target_bv_pre):
        logging.warning(
            f"[FÁZA 1] Target {target_cid}: len {len(candidates)} po B-V filtri "
            f"(ΔB-V<={max_bv_diff:.2f}) — uvoľňujem na 1.5×{max_bv_diff:.2f}"
        )
        relaxed_bv = max_bv_diff * 1.5
        ms["_bv"] = pd.to_numeric(ms.get("b_v", pd.Series(dtype=float)), errors="coerce")
        bv_known_mask = ms["_bv"].notna()
        relaxed_mask = (
            ms["_dist_deg"].le(max_dist_deg)
            & _bool_col(ms.get("is_usable", pd.Series(True, index=ms.index)))
            & ~_bool_col(ms.get("is_saturated", pd.Series(False, index=ms.index)))
            & ~_bool_col(ms.get("is_noisy", pd.Series(False, index=ms.index)))
            & ~_bool_col(ms.get("vsx_known_variable", pd.Series(False, index=ms.index)))
            & ~_bool_col(ms.get("likely_saturated", pd.Series(False, index=ms.index)))
        )
        if target_cid:
            relaxed_mask &= (
                ms.get("catalog_id", ms.get("name", pd.Series("", index=ms.index))).astype(str) != target_cid
            )
        if math.isfinite(min_dist_arcsec) and min_dist_arcsec > 0:
            min_dist_deg = float(min_dist_arcsec) / 3600.0
            relaxed_mask &= ms["_dist_deg"].ge(min_dist_deg)
        if math.isfinite(mag_t):
            if "_mag" not in ms.columns:
                ms["_mag"] = pd.to_numeric(ms.get("mag", ms.get("phot_g_mean_mag")), errors="coerce")
            relaxed_mask &= ms["_mag"].sub(mag_t).abs().le(max_mag_diff)
        bv_relaxed_filter = ~bv_known_mask | ms["_bv"].sub(target_bv_pre).abs().le(relaxed_bv)
        relaxed_mask &= bv_relaxed_filter
        candidates = ms[relaxed_mask].copy()
        logging.warning(
            f"[FÁZA 1] Target {target_cid}: po relaxácii B-V na {relaxed_bv:.2f} "
            f"dostupných {len(candidates)} kandidátov"
        )

    if len(candidates) < n_comp_min:
        logging.warning(
            f"[FÁZA 1] Target {target_cid}: len {len(candidates)} kandidátov "
            f"< n_comp_min={n_comp_min} — preskakujem."
        )
        return pd.DataFrame()

    # Identifikátory kandidátov
    # ``catalog_id`` býva v CSV často uložené ako float (scientific) a môže stratiť presnosť;
    # ``name`` obsahuje presný identifikátor (Gaia ID ako string alebo DET_*). Preferuj preto ``name``.
    id_col = "name" if "name" in candidates.columns else ("catalog_id" if "catalog_id" in candidates.columns else "name")
    cand_ids = set(candidates[id_col].astype(str).str.strip())

    # ── Krok 2: Načítaj flux z per-frame CSV ──
    # Načítaj len potrebné stĺpce — úspora 78 % pamäte
    avail_cols = _PHASE_USECOLS_PERFRAME.copy()
    flux_map: dict[str, list[float]] = {cid: [] for cid in cand_ids}
    n_frames_loaded = 0

    # Filter C → Contamination index (soft penalizácia v scoringu)
    # Namiesto hard-exclusion vypočítaj contamination ratio per kandidát.
    # Hustá oblasť neba: hard filter by vylúčil väčšinu kandidátov.
    # Riešenie: crowding sa prejaví ako penalizácia v combined score (Krok 5).
    contamination_map: dict[str, float] = {}  # cid → sum flux ratio od susedov

    # Pre filter B: zbieraj psf_chi2 a fwhm_estimate_px per kandidát
    psf_chi2_map: dict[str, list[float]] = {cid: [] for cid in cand_ids}
    fwhm_map: dict[str, list[float]] = {cid: [] for cid in cand_ids}
    frame_fwhm_medians: list[float] = []  # mediánový FWHM všetkých hviezd per snímka
    # Saturácia naprieč framami: peak_max_adu > saturate_limit_adu_85pct
    peak_over_map: dict[str, int] = {cid: 0 for cid in cand_ids}
    peak_total_map: dict[str, int] = {cid: 0 for cid in cand_ids}
    # Skutočný 5σ SNR (median cez framy)
    snr_map: dict[str, list[float]] = {cid: [] for cid in cand_ids}

    for csv_path in per_frame_csv_paths:
        try:
            # Zisti dostupné stĺpce v súbore
            header = pd.read_csv(csv_path, nrows=0)
            use_cols = [c for c in avail_cols if c in header.columns]
            # Flux stĺpec — preferuj dao_flux, fallback na flux
            actual_flux_col = flux_col if flux_col in header.columns else "flux"
            if actual_flux_col not in use_cols:
                use_cols.append(actual_flux_col)
            # Per-frame CSV: ``catalog_id`` môže byť NaN (najmä keď sa join MASTERSTAR robí len pre matchnuté riadky),
            # ale ``name`` typicky obsahuje stabilný identifikátor (Gaia ID ako string alebo DET_*).
            # Preferuj preto ``name`` ak existuje.
            name_col = "name" if "name" in header.columns else ("catalog_id" if "catalog_id" in header.columns else "name")
            if "mag" not in use_cols and "mag" in header.columns:
                use_cols.append("mag")
            if "psf_chi2" in header.columns and "psf_chi2" not in use_cols:
                use_cols.append("psf_chi2")
            if "fwhm_estimate_px" in header.columns and "fwhm_estimate_px" not in use_cols:
                use_cols.append("fwhm_estimate_px")
            if "peak_max_adu" in header.columns and "peak_max_adu" not in use_cols:
                use_cols.append("peak_max_adu")
            if (
                "saturate_limit_adu_85pct" in header.columns
                and "saturate_limit_adu_85pct" not in use_cols
            ):
                use_cols.append("saturate_limit_adu_85pct")

            df = pd.read_csv(csv_path, usecols=use_cols, low_memory=False)
            df[name_col] = _normalize_id_series(df[name_col])
            df[actual_flux_col] = pd.to_numeric(df[actual_flux_col], errors="coerce")
            if "peak_max_adu" in df.columns:
                df["peak_max_adu"] = pd.to_numeric(df["peak_max_adu"], errors="coerce")
            if "saturate_limit_adu_85pct" in df.columns:
                df["saturate_limit_adu_85pct"] = pd.to_numeric(
                    df["saturate_limit_adu_85pct"], errors="coerce"
                )

            # Saturácia naprieč framami (nezávisle od flux>0):
            # peak_max_adu > saturate_limit_adu_85pct
            if "peak_max_adu" in df.columns and "saturate_limit_adu_85pct" in df.columns:
                for _, _row in df[df[name_col].isin(cand_ids)].iterrows():
                    _cid = str(_row[name_col])
                    peak = float(_row.get("peak_max_adu", float("nan")))
                    limit = float(_row.get("saturate_limit_adu_85pct", float("nan")))
                    if math.isfinite(peak) and math.isfinite(limit) and limit > 0:
                        peak_total_map[_cid] = int(peak_total_map.get(_cid, 0)) + 1
                        if peak > limit:
                            peak_over_map[_cid] = int(peak_over_map.get(_cid, 0)) + 1

            # Zbieraj psf_chi2 a fwhm_estimate_px pre Filter B
            if "psf_chi2" in df.columns:
                for _, _row in df[df[name_col].isin(cand_ids)].iterrows():
                    _cid = str(_row[name_col])
                    _chi2 = float(_row.get("psf_chi2", float("nan")))
                    if math.isfinite(_chi2) and _chi2 > 0:
                        psf_chi2_map[_cid].append(_chi2)

            if "fwhm_estimate_px" in df.columns:
                _fwhm_col = pd.to_numeric(df["fwhm_estimate_px"], errors="coerce")
                # Mediánový FWHM všetkých hviezd na tejto snímke
                _frame_fwhm_med = float(_fwhm_col.median())
                if math.isfinite(_frame_fwhm_med) and _frame_fwhm_med > 0:
                    frame_fwhm_medians.append(_frame_fwhm_med)
                # FWHM per kandidát
                for _, _row in df[df[name_col].isin(cand_ids)].iterrows():
                    _cid = str(_row[name_col])
                    _fwhm = float(_row.get("fwhm_estimate_px", float("nan")))
                    if math.isfinite(_fwhm) and _fwhm > 0:
                        fwhm_map[_cid].append(_fwhm)

            # Len kandidáti s platným flux
            sub = df[df[name_col].isin(cand_ids) & df[actual_flux_col].gt(0)].copy()
            if sub.empty:
                continue

            # Mag-bin normalizácia: medián zvlášť pre každý mag bin (0.5 mag šírka)
            # Eliminuje sky drift ktorý závisí od jasnosti hviezdy
            mag_col_frame = "mag" if "mag" in df.columns else None
            if mag_col_frame and mag_col_frame in sub.columns:
                sub = sub.copy()
                sub["_mag_num"] = pd.to_numeric(sub[mag_col_frame], errors="coerce")
                sub["_mag_bin"] = (sub["_mag_num"] / 0.5).apply(
                    lambda x: int(x) if math.isfinite(x) else -1
                )
                bin_meds: dict[int, float] = {}
                for b, grp in sub.groupby("_mag_bin"):
                    bmed = float(grp[actual_flux_col].median())
                    if math.isfinite(bmed) and bmed > 0:
                        bin_meds[int(b)] = bmed
                if not bin_meds:
                    continue
            else:
                # Fallback: globálny medián
                frame_med = float(sub[actual_flux_col].median())
                if not math.isfinite(frame_med) or frame_med <= 0:
                    continue
                bin_meds = {}

            n_frames_loaded += 1
            for _, row in sub.iterrows():
                cid = str(row[name_col])
                raw_flux = float(row[actual_flux_col])
                if not math.isfinite(raw_flux) or raw_flux <= 0:
                    continue

                # Skutočný 5σ SNR:
                # SNR = dao_flux / sqrt(dao_flux + noise_floor_adu * aperture_area)
                flux_snr = float(row.get("dao_flux", raw_flux))
                sky = float(row.get("noise_floor_adu", 0.0))
                r_ap = float(row.get("aperture_r_px", 7.0))
                area = math.pi * r_ap * r_ap if math.isfinite(r_ap) and r_ap > 0 else float("nan")
                if math.isfinite(flux_snr) and flux_snr > 0 and math.isfinite(sky) and math.isfinite(area):
                    denom = flux_snr + max(0.0, sky) * area
                    if denom > 0:
                        snr = flux_snr / math.sqrt(denom)
                        if math.isfinite(snr):
                            snr_map[cid].append(float(snr))
                # Normalizácia voči mediánu hviezd rovnakej jasnosti
                if bin_meds:
                    mag_num = (
                        float(row.get("_mag_num", float("nan")))
                        if "_mag_num" in row.index
                        else float("nan")
                    )
                    b = int(mag_num / 0.5) if math.isfinite(mag_num) else -1
                    norm_med = bin_meds.get(b)
                    if norm_med is None:
                        # Fallback na najbližší bin
                        closest = min(bin_meds.keys(), key=lambda k: abs(k - b))
                        norm_med = bin_meds[closest]
                else:
                    norm_med = frame_med  # type: ignore[assignment]
                rel = raw_flux / norm_med
                if math.isfinite(rel) and rel > 0:
                    flux_map[cid].append(rel)

        except Exception:  # noqa: BLE001
            continue

    min_frames = max(3, int(n_frames_loaded * min_frames_frac))

    # Filter SAT: vylúč kandidátov, ktorí sú nad 85% sat limitu vo viac než 10% framov
    _sat_rejected: set[str] = set()
    for cid in list(flux_map.keys()):
        total = int(peak_total_map.get(cid, 0) or 0)
        over = int(peak_over_map.get(cid, 0) or 0)
        if total >= 10 and total > 0 and (float(over) / float(total)) > 0.10:
            flux_map.pop(cid, None)
            _sat_rejected.add(cid)
            logging.info(
                f"[FÁZA 1] Saturácia filter: vylúčený {cid} "
                f"({over}/{total} framov nad 85% limitom)"
            )
    if _sat_rejected:
        logging.info(f"[FÁZA 1] Celkom vylúčených kvôli saturácii: {len(_sat_rejected)}")

    # Filter SNR: vylúč kandidátov s median SNR < 5σ
    _snr_rejected: set[str] = set()
    for cid in list(flux_map.keys()):
        snrs = snr_map.get(cid, [])
        if len(snrs) >= 5:
            snr_median = float(np.median(np.asarray(snrs, dtype=np.float64)))
            if math.isfinite(snr_median) and snr_median < 5.0:
                flux_map.pop(cid, None)
                _snr_rejected.add(cid)
                logging.info(
                    f"[FÁZA 1] SNR filter: vylúčený {cid} "
                    f"(median SNR={snr_median:.1f} < 5)"
                )

    # Filter B: PSF chi² a FWHM blend detekcia
    _global_fwhm_med = float(np.median(frame_fwhm_medians)) if frame_fwhm_medians else float("nan")
    _b_rejected: set[str] = set()

    if math.isfinite(max_psf_chi2):
        for _cid, _chi2_vals in psf_chi2_map.items():
            if len(_chi2_vals) < 3:
                continue
            _med_chi2 = float(np.median(_chi2_vals))
            if _med_chi2 > max_psf_chi2:
                _b_rejected.add(_cid)
                logging.debug(
                    f"[FÁZA 1] Blend filter (PSF chi²): vylúčený {_cid} "
                    f"(median chi²={_med_chi2:.2f} > {max_psf_chi2:.2f})"
                )

    if math.isfinite(max_fwhm_factor) and math.isfinite(_global_fwhm_med) and _global_fwhm_med > 0:
        for _cid, _fwhm_vals in fwhm_map.items():
            if len(_fwhm_vals) < 3:
                continue
            _med_fwhm = float(np.median(_fwhm_vals))
            _fwhm_ratio = _med_fwhm / _global_fwhm_med
            if _fwhm_ratio > max_fwhm_factor:
                _b_rejected.add(_cid)
                logging.debug(
                    f"[FÁZA 1] Blend filter (FWHM): vylúčený {_cid} "
                    f"(median FWHM={_med_fwhm:.2f}px, ratio={_fwhm_ratio:.2f} > {max_fwhm_factor:.2f})"
                )

    if _b_rejected:
        logging.info(
            f"[FÁZA 1] Target {target_cid}: Filter B (PSF/FWHM) vylúčil "
            f"{len(_b_rejected)} kandidátov: {sorted(_b_rejected)}"
        )
        for _cid in _b_rejected:
            flux_map.pop(_cid, None)

    # Filter C → Contamination index (soft penalizácia v scoringu)
    # Namiesto hard-exclusion vypočítaj contamination ratio per kandidát.
    # Hustá oblasť neba: hard filter by vylúčil väčšinu kandidátov.
    # Riešenie: crowding sa prejaví ako penalizácia v combined score (Krok 5).
    if isolation_radius_px > 0 and "x" in ms.columns and "y" in ms.columns:
        ms_reset = ms.reset_index(drop=True)
        _id_col_ms = "catalog_id" if "catalog_id" in ms_reset.columns else "name"

        # Flux proxy: dao_flux > flux > phot_g_mean_mag (mag → relatívny flux)
        _flux_col_ms = next((fc for fc in ("dao_flux", "flux") if fc in ms_reset.columns), None)
        _mag_col_ms = next(
            (mc for mc in ("phot_g_mean_mag", "catalog_mag", "mag") if mc in ms_reset.columns),
            None,
        )

        # Zostavíme vektory pre rýchly výpočet vzdialeností
        _ms_x_all = pd.to_numeric(ms_reset["x"], errors="coerce").to_numpy(dtype=np.float64)
        _ms_y_all = pd.to_numeric(ms_reset["y"], errors="coerce").to_numpy(dtype=np.float64)

        if _flux_col_ms:
            _ms_flux_all = pd.to_numeric(ms_reset[_flux_col_ms], errors="coerce").to_numpy(dtype=np.float64)
        elif _mag_col_ms:
            _mags_all = pd.to_numeric(ms_reset[_mag_col_ms], errors="coerce").to_numpy(dtype=np.float64)
            _ms_flux_all = np.where(np.isfinite(_mags_all), 10 ** (-0.4 * _mags_all), np.nan)
        else:
            _ms_flux_all = np.ones(len(ms_reset))

        _ms_mag_all = (
            pd.to_numeric(ms_reset[_mag_col_ms], errors="coerce").to_numpy(dtype=np.float64)
            if _mag_col_ms
            else np.full(len(ms_reset), np.nan, dtype=np.float64)
        )

        # Lookup: catalog_id → riadok index v ms_reset
        _cid_to_idx: dict[str, int] = {}
        for _ri, _rrow in ms_reset.iterrows():
            _rcid = _normalize_id_value(_rrow.get(_id_col_ms, ""))
            if _rcid:
                _cid_to_idx[_rcid] = int(_ri)

        for _cid in flux_map:
            _ci = _cid_to_idx.get(_cid)
            if _ci is None:
                continue
            _cx = _ms_x_all[_ci]
            _cy = _ms_y_all[_ci]
            _cflux = _ms_flux_all[_ci]
            if not (math.isfinite(_cx) and math.isfinite(_cy)):
                continue
            if not math.isfinite(_cflux) or _cflux <= 0:
                continue

            _dx = _ms_x_all - _cx
            _dy = _ms_y_all - _cy
            _dists = np.sqrt(_dx * _dx + _dy * _dy)
            _neighbor_mask = (
                (_dists > 0.5)
                & (_dists <= isolation_radius_px)
                & np.isfinite(_ms_flux_all)
                & (_ms_flux_all > 0)
            )
            # Zahrnúť len susedov do 3 mag od kandidáta
            mag_cand = float(_ms_mag_all[_ci]) if _ci < len(_ms_mag_all) else float("nan")
            if math.isfinite(mag_cand):
                _neighbor_mask = _neighbor_mask & (
                    ~np.isfinite(_ms_mag_all) | ((_ms_mag_all - mag_cand) <= 3.0)
                )
            if not np.any(_neighbor_mask):
                contamination_map[_cid] = 0.0
                continue

            # Contamination = súčet flux susedov / flux kandidáta
            # (súčet, nie maximum — viac slabých susedov = väčší efekt)
            _neighbor_flux_sum = float(np.sum(_ms_flux_all[_neighbor_mask]))
            contamination_map[_cid] = min(_neighbor_flux_sum / _cflux, 2.0)  # cap na 2.0 (200%)

        if contamination_map:
            _cont_vals = list(contamination_map.values())
            logging.debug(
                f"[FÁZA 1] Target {target_cid}: contamination index "
                f"median={float(np.median(_cont_vals)):.3f}, "
                f"max={max(_cont_vals):.3f} "
                f"(isolation_radius={isolation_radius_px:.0f}px)"
            )

    # ── Krok 2b: Airmass detrending ──
    # Polynomický fit (stupeň 2) na časový rad relatívneho flux odstráni
    # systematický airmass trend. Residuály = skutočná fotometrická variabilita.
    for cid in list(flux_map.keys()):
        vals = flux_map[cid]
        if len(vals) < 6:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        t = np.linspace(0.0, 1.0, len(arr))
        try:
            coeffs = np.polyfit(t, arr, 2)
            trend_fit = np.polyval(coeffs, t)
            safe_trend = np.where(np.abs(trend_fit) > 1e-9, trend_fit, 1.0)
            detrended = arr / safe_trend
            med_dt = float(np.median(detrended))
            if math.isfinite(med_dt) and med_dt > 0:
                flux_map[cid] = (detrended / med_dt).tolist()
        except Exception:  # noqa: BLE001
            pass  # Ponechaj pôvodné hodnoty ak fit zlyhá

    # ── Krok 3: RMS scatter per kandidát ──
    rms_map: dict[str, float] = {}
    for cid, vals in flux_map.items():
        if len(vals) < min_frames:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        rms = float(np.sqrt(np.mean((arr - 1.0) ** 2)))
        if math.isfinite(rms):
            rms_map[cid] = rms

    # Tvrdý RMS limit — odmietni nestabilné hviezdy bez ohľadu na ranking
    if math.isfinite(max_comp_rms) and max_comp_rms > 0:
        n_before = len(rms_map)
        rms_map = {cid: rms for cid, rms in rms_map.items() if rms <= max_comp_rms}
        n_rejected = n_before - len(rms_map)
        if n_rejected > 0:
            logging.info(
                f"[FÁZA 1] Target {target_cid}: tvrdý RMS filter (>{max_comp_rms:.3f}) "
                f"odmietol {n_rejected} kandidátov, zostáva {len(rms_map)}"
            )

    if len(rms_map) < n_comp_min:
        logging.warning(
            f"[FÁZA 1] Target {target_cid}: len {len(rms_map)} kandidátov "
            f"s dostatkom snímok < n_comp_min={n_comp_min}."
        )
        return pd.DataFrame()

    # ── Krok 4: Iteratívny ensemble filter (robustný MAD) ──
    # Prah = median + k × (MAD / 0.6745)
    # MAD / 0.6745 = konzistentný estimátor σ robustný voči outlierom
    # k = rms_outlier_sigma (default 3.0)
    _MAD_CONSISTENCY = 0.6745  # normalizačný faktor MAD → σ ekvivalent
    active = dict(rms_map)
    for _iter in range(10):
        if len(active) <= n_comp_min:
            break
        vals_arr = np.asarray(list(active.values()), dtype=np.float64)
        med = float(np.median(vals_arr))
        mad_raw = float(np.median(np.abs(vals_arr - med)))
        if not math.isfinite(mad_raw) or mad_raw <= 0:
            # MAD = 0 znamená že všetky hodnoty sú rovnaké → konvergencia
            break
        mad_sigma = mad_raw / _MAD_CONSISTENCY  # robustný σ estimátor
        threshold = med + rms_outlier_sigma * mad_sigma
        new_active = {cid: rms for cid, rms in active.items() if rms <= threshold}
        if len(new_active) == len(active):
            break  # Konvergencia — žiadne ďalšie výrazy
        if len(new_active) < n_comp_min:
            break  # Neprekroč minimum
        active = new_active

    # ── Krok 5: Finálny výber ──
    # Score: stabilita (RMS) + vzdialenosť + izolácia (contamination)
    # (nižší = lepší kandidát)
    id_col_cand = "name" if "name" in candidates.columns else ("catalog_id" if "catalog_id" in candidates.columns else "name")
    score_map: dict[str, float] = {}
    for cid, rms in active.items():
        row = candidates[candidates[id_col_cand].astype(str).str.strip() == cid]
        if row.empty:
            continue
        dist_deg = float(row.iloc[0].get("_dist_deg", float("nan")))
        dist_score = (dist_deg * 3600.0 / 600.0) if math.isfinite(dist_deg) else 1.0
        contamination = float(contamination_map.get(cid, 0.0)) if contamination_map else 0.0
        rms_score = float(rms)
        score_map[cid] = rms_score * 0.5 + dist_score * 0.3 + contamination * 0.2

    scored = sorted(score_map.items(), key=lambda x: float(x[1]))
    # Preferuj Gaia-matched (číselné ID) pred DET_* ak máme dostatok možností.
    scored_non_det = [(cid, sc) for cid, sc in scored if not str(cid).startswith("DET")]
    scored_det = [(cid, sc) for cid, sc in scored if str(cid).startswith("DET")]

    selected_ids = [cid for cid, _ in scored_non_det[:n_comp_max]]
    if len(selected_ids) < n_comp_min:
        need = n_comp_min - len(selected_ids)
        selected_ids.extend([cid for cid, _ in scored_det[: max(0, need)]])

    if len(selected_ids) < n_comp_min:
        logging.warning(
            f"[FÁZA 1] Target {target_cid}: po filtrácii len {len(selected_ids)} "
            f"< n_comp_min={n_comp_min}."
        )
        return pd.DataFrame()

    # Zostav výstupný DataFrame
    result_rows = []
    target_bv = float(target.get("b_v", float("nan")))
    for cid in selected_ids:
        row = candidates[candidates[id_col_cand].astype(str).str.strip() == cid]
        if row.empty:
            continue
        r = row.iloc[0].to_dict()
        r["comp_rms"] = active.get(cid, float("nan"))
        r["comp_score"] = score_map.get(cid, float("nan"))
        r["comp_n_frames"] = len(flux_map.get(cid, []))
        r["target_catalog_id"] = target_cid
        r["target_vsx_name"] = str(target.get("vsx_name", ""))
        result_rows.append(r)

    if not result_rows:
        return pd.DataFrame()

    _total_rejected_b = len(_b_rejected) if "_b_rejected" in dir() else 0
    if _total_rejected_b > 0:
        logging.info(
            f"[FÁZA 1] Target {target_cid}: blend filter B celkom vylúčil "
            f"{_total_rejected_b} kandidátov"
        )

    out = pd.DataFrame(result_rows).sort_values("comp_score").reset_index(drop=True)
    if "b_v" in out.columns and math.isfinite(target_bv):
        out_bv = pd.to_numeric(out["b_v"], errors="coerce")
        dbv_out = (out_bv - target_bv).abs()
        bv_info = f"ΔB-V median={float(dbv_out.median()):.3f} max={float(dbv_out.max()):.3f}"
    else:
        bv_info = "ΔB-V N/A"

    logging.info(
        f"[FÁZA 1] Target {target_cid} ({target.get('vsx_name','')}): "
        f"{len(out)} porovnávačiek | RMS min={out['comp_rms'].min():.4f} "
        f"max={out['comp_rms'].max():.4f} | {bv_info}"
    )
    return out


def run_phase0_and_phase1(
    variable_targets_csv: Path,
    masterstars_csv: Path,
    per_frame_csv_dir: Path,
    output_dir: Path,
    *,
    fwhm_px: float = 3.7,
    frame_w_px: int = 2082,
    frame_h_px: int = 1397,
    edge_margin_px: int = 50,
    match_radius_arcsec: float = 15.0,
    max_dist_deg: float = 1.5,
    max_mag_diff: float = 0.25,
    max_bv_diff: float = 0.15,
    n_comp_min: int = 3,
    n_comp_max: int = 5,
    max_comp_rms: float = 0.05,
    min_dist_arcsec: float = 30.0,
    min_frames_frac: float = 0.3,
    rms_outlier_sigma: float = 3.0,
    exclude_gaia_nss: bool = True,
    exclude_gaia_extobj: bool = True,
    max_psf_chi2: float = 3.0,
    max_fwhm_factor: float = 1.5,
    isolation_radius_px: float = 25.0,
    flux_col: str = "dao_flux",
) -> dict[str, Any]:
    """Spusti Fázu 0 + Fázu 1 a uloží výstupy.

    Výstupy (uložené do output_dir):
      active_targets.csv              — filtrované VSX ciele
      comparison_stars_per_target.csv — porovnávacie hviezdy pre každý cieľ
      suspected_variables.csv         — kandidáti na nové premenné (vysoký RMS, nie VSX)

    Returns:
        dict s kľúčmi:
          n_active_targets, n_comparison_pairs,
          active_targets_csv, comparison_stars_csv, suspected_variables_csv,
          targets_without_comps (list catalog_id)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── FÁZA 0 ──
    logging.info("[FÁZA 0] Výber aktívnych cieľov...")
    active = select_active_targets(
        variable_targets_csv,
        masterstars_csv,
        frame_w_px=frame_w_px,
        frame_h_px=frame_h_px,
        edge_margin_px=edge_margin_px,
        match_radius_arcsec=match_radius_arcsec,
    )
    active_csv = output_dir / "active_targets.csv"
    active.to_csv(active_csv, index=False)
    logging.info(f"[FÁZA 0] Uložené: {active_csv} ({len(active)} cieľov)")

    if active.empty:
        return {
            "n_active_targets": 0,
            "n_comparison_pairs": 0,
            "active_targets_csv": str(active_csv),
            "comparison_stars_csv": None,
            "suspected_variables_csv": None,
            "targets_without_comps": [],
        }

    # ── Per-frame CSV súbory ──
    csv_paths = sorted(per_frame_csv_dir.rglob("*.csv"))
    # Vylúč výstupné súbory (active_targets, comparison_stars atď.)
    csv_paths = [p for p in csv_paths if p.stem.startswith("proc_") or p.stem.startswith("ali_")]
    logging.info(f"[FÁZA 1] Načítavam flux z {len(csv_paths)} per-frame CSV súborov...")

    ms_df = pd.read_csv(masterstars_csv, low_memory=False)
    # Normalizuj Gaia ID na string
    for _id_col in ("catalog_id", "name"):
        if _id_col in ms_df.columns:
            ms_df[_id_col] = _normalize_id_series(ms_df[_id_col])

    # ── FÁZA 1 — per target ──
    all_comp_rows: list[pd.DataFrame] = []
    targets_without_comps: list[str] = []

    for _, target_row in active.iterrows():
        comps = select_comparison_stars_per_target(
            target_row,
            ms_df,
            csv_paths,
            fwhm_px=fwhm_px,
            max_dist_deg=max_dist_deg,
            max_mag_diff=max_mag_diff,
            max_bv_diff=max_bv_diff,
            n_comp_min=n_comp_min,
            n_comp_max=n_comp_max,
            max_comp_rms=max_comp_rms,
            min_dist_arcsec=min_dist_arcsec,
            min_frames_frac=min_frames_frac,
            rms_outlier_sigma=rms_outlier_sigma,
            exclude_gaia_nss=exclude_gaia_nss,
            exclude_gaia_extobj=exclude_gaia_extobj,
            max_psf_chi2=max_psf_chi2,
            max_fwhm_factor=max_fwhm_factor,
            isolation_radius_px=isolation_radius_px,
        )
        if comps.empty:
            targets_without_comps.append(str(target_row.get("catalog_id", "")))
        else:
            all_comp_rows.append(comps)

    comp_df = pd.concat(all_comp_rows, ignore_index=True) if all_comp_rows else pd.DataFrame()
    comp_csv = output_dir / "comparison_stars_per_target.csv"
    comp_df.to_csv(comp_csv, index=False)
    logging.info(
        f"[FÁZA 1] Uložené: {comp_csv} "
        f"({len(comp_df)} riadkov, {len(all_comp_rows)} targetov s porovnávačkami)"
    )

    # ── Suspected variables ──
    # Hviezdy s vysokým RMS (>3σ nad mediánom) ktoré nie sú VSX ani active targets
    suspected_csv = output_dir / "suspected_variables.csv"
    _write_suspected_variables(
        ms_df=ms_df,
        csv_paths=csv_paths,
        active_target_ids=set(active["catalog_id"].astype(str)),
        output_path=suspected_csv,
        min_frames_frac=min_frames_frac,
        outlier_sigma=3.0,
    )

    return {
        "n_active_targets": int(len(active)),
        "n_comparison_pairs": int(len(comp_df)),
        "active_targets_csv": str(active_csv),
        "comparison_stars_csv": str(comp_csv),
        "suspected_variables_csv": str(suspected_csv),
        "targets_without_comps": targets_without_comps,
    }


def _write_suspected_variables(
    ms_df: pd.DataFrame,
    csv_paths: list[Path],
    active_target_ids: set[str],
    output_path: Path,
    *,
    flux_col: str = "dao_flux",
    min_frames_frac: float = 0.5,
    outlier_sigma: float = 3.0,
) -> None:
    """Detekuj hviezdy s vysokým RMS scatter ktoré nie sú v VSX — suspected new variables.

    Zapíše suspected_variables.csv s kolumnami:
    catalog_id, ra_deg, dec_deg, mag, comp_rms, n_frames, zone
    """
    # Usable hviezdy ktoré nie sú VSX ani active targets
    ms = ms_df.copy()
    for col in ("is_usable", "is_saturated", "is_noisy", "vsx_known_variable"):
        if col in ms.columns:
            ms[col] = _bool_col(ms[col])

    id_col = "catalog_id" if "catalog_id" in ms.columns else "name"
    base_mask = (
        _bool_col(ms.get("is_usable", pd.Series(True, index=ms.index)))
        & ~_bool_col(ms.get("is_saturated", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("is_noisy", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("vsx_known_variable", pd.Series(False, index=ms.index)))
    )
    pool = ms[base_mask].copy()
    pool_ids = set(pool[id_col].astype(str).str.strip()) - active_target_ids

    if not pool_ids:
        pd.DataFrame().to_csv(output_path, index=False)
        return

    # Načítaj flux pre všetky hviezdy z poolu
    flux_map: dict[str, list[float]] = {cid: [] for cid in pool_ids}
    n_frames = 0

    for csv_path in csv_paths:
        try:
            header = pd.read_csv(csv_path, nrows=0)
            actual_flux = flux_col if flux_col in header.columns else "flux"
            name_c = "catalog_id" if "catalog_id" in header.columns else "name"
            use = [name_c, actual_flux]
            if "mag" in header.columns and "mag" not in use:
                use.append("mag")
            df = pd.read_csv(csv_path, usecols=use, low_memory=False)
            df[name_c] = _normalize_id_series(df[name_c])
            df[actual_flux] = pd.to_numeric(df[actual_flux], errors="coerce")
            sub = df[df[name_c].isin(pool_ids) & df[actual_flux].gt(0)]
            if sub.empty:
                continue

            # Mag-bin normalizácia: medián zvlášť pre každý mag bin (0.5 mag šírka)
            mag_col_frame = "mag" if "mag" in df.columns else None
            if mag_col_frame and mag_col_frame in sub.columns:
                sub = sub.copy()
                sub["_mag_num"] = pd.to_numeric(sub[mag_col_frame], errors="coerce")
                sub["_mag_bin"] = (sub["_mag_num"] / 0.5).apply(
                    lambda x: int(x) if math.isfinite(x) else -1
                )
                bin_meds: dict[int, float] = {}
                for b, grp in sub.groupby("_mag_bin"):
                    bmed = float(grp[actual_flux].median())
                    if math.isfinite(bmed) and bmed > 0:
                        bin_meds[int(b)] = bmed
                if not bin_meds:
                    continue
            else:
                # Fallback: globálny medián
                frame_med = float(sub[actual_flux].median())
                if not math.isfinite(frame_med) or frame_med <= 0:
                    continue
                bin_meds = {}

            n_frames += 1
            for _, row in sub.iterrows():
                cid = str(row[name_c])
                raw_flux = float(row[actual_flux])
                if not math.isfinite(raw_flux) or raw_flux <= 0:
                    continue
                if bin_meds:
                    mag_num = (
                        float(row.get("_mag_num", float("nan")))
                        if "_mag_num" in row.index
                        else float("nan")
                    )
                    b = int(mag_num / 0.5) if math.isfinite(mag_num) else -1
                    norm_med = bin_meds.get(b)
                    if norm_med is None:
                        closest = min(bin_meds.keys(), key=lambda k: abs(k - b))
                        norm_med = bin_meds[closest]
                else:
                    norm_med = frame_med  # type: ignore[assignment]
                rel = raw_flux / norm_med
                if math.isfinite(rel) and rel > 0:
                    flux_map[cid].append(rel)
        except Exception:  # noqa: BLE001
            continue

    # Airmass detrending pre suspected variables
    for cid in list(flux_map.keys()):
        vals = flux_map[cid]
        if len(vals) < 6:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        t = np.linspace(0.0, 1.0, len(arr))
        try:
            coeffs = np.polyfit(t, arr, 2)
            trend_fit = np.polyval(coeffs, t)
            safe_trend = np.where(np.abs(trend_fit) > 1e-9, trend_fit, 1.0)
            detrended = arr / safe_trend
            med_dt = float(np.median(detrended))
            if math.isfinite(med_dt) and med_dt > 0:
                flux_map[cid] = (detrended / med_dt).tolist()
        except Exception:  # noqa: BLE001
            pass

    min_f = max(3, int(n_frames * min_frames_frac))
    rms_map: dict[str, float] = {}
    nframes_map: dict[str, int] = {}
    for cid, vals in flux_map.items():
        if len(vals) < min_f:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        rms = float(np.sqrt(np.mean((arr - 1.0) ** 2)))
        if math.isfinite(rms):
            rms_map[cid] = rms
            nframes_map[cid] = len(vals)

    if not rms_map:
        pd.DataFrame().to_csv(output_path, index=False)
        return

    _MAD_CONSISTENCY = 0.6745
    rms_arr = np.asarray(list(rms_map.values()), dtype=np.float64)
    med = float(np.median(rms_arr))
    mad_raw = float(np.median(np.abs(rms_arr - med)))
    if not math.isfinite(mad_raw) or mad_raw <= 0:
        # Fallback: ak MAD=0, použi normalizovanú std ako estimátor
        mad_sigma = float(np.std(rms_arr)) / _MAD_CONSISTENCY or 1e-9
    else:
        mad_sigma = mad_raw / _MAD_CONSISTENCY
    threshold = med + outlier_sigma * mad_sigma

    suspected = {cid: rms for cid, rms in rms_map.items() if rms > threshold}

    if not suspected:
        pd.DataFrame().to_csv(output_path, index=False)
        return

    rows = []
    pool_idx = pool.set_index(pool[id_col].astype(str).str.strip())
    for cid, rms in sorted(suspected.items(), key=lambda x: -x[1]):
        if cid not in pool_idx.index:
            continue
        r = pool_idx.loc[cid]
        rows.append(
            {
                "catalog_id": cid,
                "ra_deg": r.get("ra_deg", float("nan")),
                "dec_deg": r.get("dec_deg", float("nan")),
                "mag": r.get("mag", float("nan")),
                "comp_rms": rms,
                "n_frames": nframes_map.get(cid, 0),
                "zone": r.get("zone", ""),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    logging.info(
        f"[SUSPECTED] {len(out_df)} kandidátov na nové premenné → {output_path.name} "
        f"(threshold RMS > {threshold:.4f})"
    )


__all__ = [
    # existujúce
    "StressTestResult",
    "stress_test_relative_rms_from_sidecars",
    "recommended_aperture_by_color",
    "enhance_catalog_dataframe_aperture_bpm",
    # nové
    "select_active_targets",
    "select_comparison_stars_per_target",
    "run_phase0_and_phase1",
]

