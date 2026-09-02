"""Moved from photometry_core.py (CONSOLIDATE-01E1). Facade re-exports these names."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence
import logging
import math
from astropy.io import fits as astrofits
import numpy as np
import pandas as pd
from config import AppConfig
from proc_frame_store import ProcFrameStore
from utils import plate_solve_fov_deg_diagonal_from_scale

from photometry_core import (
    BKG_SCALE_R_CLAMP_HI,
    BKG_SCALE_R_CLAMP_LO,
    ERR_BKG_MODE_EMPIRICAL,
    ERR_BKG_MODE_HOWELL,
    ERR_BKG_SOURCE_EMPIRICAL,
    _APERTURE_SIZING_MAG_COLS,
    _GAIA_ID_DTYPE,
    _build_star_exclusion_mask,
    _canonicalize_star_xy,
    _clamp_err_empty_apertures_min,
    _labbe_append_debug_record,
    _normalize_gaia_id,
    _robust_scatter_mad,
    compute_per_frame_cog_correction,
)

if TYPE_CHECKING:
    from photometry_core import _Phase2AState

def _sigma_bkg_r_key(r: float) -> float:
    """Canonical dict key for per-radius ``sigma_bkg_ap`` transport.

    Every ``_sigma_by_r`` store AND lookup must go through this function
    (ERR-518-01: unrounded store key vs rounded lookup key lost 100%
    of empirical measurements in global_fixed mode).
    """
    return round(float(r), 4)

def _assert_inv_err_sigma_acct_01(
    sigma_by_r: dict[float, tuple[float, str]],
    src_col: np.ndarray,
    *,
    n: int,
    r_ap_arr: np.ndarray | None,
    r_ap: float,
) -> None:
    """INV-ERR-SIGMA-ACCT-01: measured empirical radii must project to rows."""
    from invariants_runtime import InvariantViolation  # noqa: PLC0415

    n_empirical_measured = sum(
        1 for _sig, _src in sigma_by_r.values() if _src == ERR_BKG_SOURCE_EMPIRICAL
    )
    n_empirical_assigned = int(np.sum(src_col == ERR_BKG_SOURCE_EMPIRICAL))
    if n_empirical_measured <= 0 or n_empirical_assigned > 0:
        return
    if r_ap_arr is not None:
        radii_requested = sorted(
            {
                _sigma_bkg_r_key(float(v))
                for v in r_ap_arr[np.isfinite(r_ap_arr) & (r_ap_arr > 0)]
            }
        )
    else:
        radii_requested = [_sigma_bkg_r_key(float(r_ap))]
    radii_measured = sorted(
        _sigma_bkg_r_key(r)
        for r, (_sig, src) in sigma_by_r.items()
        if src == ERR_BKG_SOURCE_EMPIRICAL
    )
    raise InvariantViolation(
        "INV-ERR-SIGMA-ACCT-01",
        "Labbe measurement succeeded for "
        f"{n_empirical_measured} radius value(s) but 0 of {n} rows received "
        "empirical sigma_bkg_ap; key projection is broken. "
        f"radii_measured={radii_measured} radii_requested={radii_requested}",
    )

def comp_quality_quality_strings(
    qmap: dict[str, dict[str, str]] | dict[str, str] | None,
) -> dict[str, str]:
    """Flatten parsed comp-quality map to ``catalog_id`` -> quality string (for w_rel / export helpers)."""
    if not qmap:
        return {}
    out: dict[str, str] = {}
    for k, v in qmap.items():
        if isinstance(v, dict):
            out[str(k)] = str(v.get("quality", "") or "").strip().lower()
        else:
            out[str(k)] = str(v).strip().lower()
    return out

def _clamp_err_empty_apertures_n(n: int) -> int:
    """Clamp ``err_empty_apertures_n`` to registry range 16..256."""
    try:
        v = int(n)
    except (TypeError, ValueError):
        v = 64
    return max(16, min(256, v))

def _normalize_err_background_mode(mode: str | None) -> str:
    m = str(mode or ERR_BKG_MODE_EMPIRICAL).strip().lower()
    if m in (ERR_BKG_MODE_HOWELL, "legacy"):
        return ERR_BKG_MODE_HOWELL
    return ERR_BKG_MODE_EMPIRICAL

def _labbe_content_seed_from_header(hdr: Any, *, r_ap: float) -> int:
    """Stable Labbe RNG seed from frame identity + aperture radius (F-431 / LABBE-DET)."""
    import hashlib

    def _hget(key: str) -> str:
        try:
            return str(hdr.get(key) or "")
        except Exception:  # noqa: BLE001
            return ""

    parts = [
        _hget("DATE-OBS"),
        _hget("FILENAME"),
        _hget("FRAME"),
        _hget("NAXIS1"),
        _hget("NAXIS2"),
        f"{float(r_ap):.4f}",
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**63 - 1)

def measure_empty_aperture_sigma_bkg(
    data: np.ndarray,
    star_x: np.ndarray,
    star_y: np.ndarray,
    r_ap: float,
    r_in: float,
    r_out: float,
    *,
    n_apertures: int = 64,
    min_valid: int = 16,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    frame_id: str | None = None,
    star_list_source: str = "in_memory",
) -> tuple[float, int, str]:
    """Empirical aperture background noise via random empty apertures (Labbe et al. 2003).

    Each placement uses the same annulus sky subtraction as production science apertures
    (``_annulus_sky_subtracted_flux``). The robust scatter of net sums is ``sigma_bkg_ap``
    [ADU] and already includes background Poisson, read noise, resampling covariance,
    pedestal offsets, and the Merline & Howell (1995) sky-estimation term - do **not**
    add a separate RN or annulus term (would double-count).

    Determinism (LABBE-DET): star list is canonicalized (sorted by x,y); when ``rng`` is
    None a child Generator is derived via ``SeedSequence`` from ``seed`` (and r_ap) so
    draws are independent of call order / shared parent RNG.

    Args:
        seed: When ``rng`` is None, seed the Generator (F-431). Prefer a content-derived
            seed from the caller so same-draft re-photometry is byte-stable.
        frame_id / star_list_source: optional debug-dump metadata
            (``VYVAR_LABBE_DEBUG_DUMP=1``).

    Returns:
        (sigma_bkg_ap, n_valid, reason) - reason non-empty when measurement failed.
    """
    import hashlib

    xs_c, ys_c, star_list_hash = _canonicalize_star_xy(star_x, star_y)
    seed_value = int(seed) if seed is not None else None
    if rng is None:
        if seed_value is None:
            rng = np.random.default_rng(None)
        else:
            # Independent child RNG keyed on content seed + r_ap (order-independent).
            ss = np.random.SeedSequence(
                [int(seed_value), int(round(float(r_ap) * 10000.0)) & 0xFFFFFFFF]
            )
            rng = np.random.default_rng(ss)
    n_target = _clamp_err_empty_apertures_n(n_apertures)
    n_min = _clamp_err_empty_apertures_min(min_valid)
    if not (
        math.isfinite(r_ap)
        and r_ap > 0
        and math.isfinite(r_in)
        and r_in > 0
        and math.isfinite(r_out)
        and r_out > r_in
    ):
        return float("nan"), 0, "invalid_annulus_geometry"

    d = np.asarray(data, dtype=np.float64)
    if d.ndim != 2 or d.size == 0:
        return float("nan"), 0, "empty_image"

    margin_px = max(2.0, float(r_out) - float(r_in))
    excl_r = float(r_out) + margin_px
    edge_margin = float(r_out) + float(r_ap) + 1.0
    blocked = _build_star_exclusion_mask(d.shape, xs_c, ys_c, excl_r, edge_margin)
    free_y, free_x = np.nonzero(~blocked)
    if free_x.size < n_min:
        # APERTURE-01: FWHM-scaled sky annulus (~9 x FWHM) around every catalog
        # star can fill the chip, so Labbe has nowhere to sit. Keep the aperture
        # star-free (4 x r_ap) and still require the empty aperture+annulus on-chip.
        excl_r = max(4.0 * float(r_ap), 8.0)
        blocked = _build_star_exclusion_mask(d.shape, xs_c, ys_c, excl_r, edge_margin)
        free_y, free_x = np.nonzero(~blocked)
    mask_hash = hashlib.sha256(np.ascontiguousarray(blocked).view(np.uint8)).hexdigest()
    labbe_input_hash = hashlib.sha256(
        f"{star_list_hash}|{mask_hash}|{float(r_ap):.4f}|{seed_value}".encode("utf-8")
    ).hexdigest()

    if free_x.size < n_min:
        _labbe_append_debug_record(
            {
                "frame_id": frame_id,
                "r_ap": float(r_ap),
                "seed_value": seed_value,
                "star_list_source": star_list_source,
                "n_stars": int(xs_c.size),
                "star_list_hash": star_list_hash,
                "mask_hash": mask_hash,
                "labbe_input_hash": labbe_input_hash,
                "n_attempted": 0,
                "n_valid_apertures": 0,
                "first5_aperture_xy": [],
                "sigma_result": None,
                "reason": f"crowding: only {int(free_x.size)} candidate pixels (< {n_min})",
            }
        )
        return float("nan"), 0, f"crowding: only {int(free_x.size)} candidate pixels (< {n_min})"

    n_try = min(int(free_x.size), max(n_target * 8, n_target))
    idx = rng.choice(free_x.size, size=n_try, replace=False)
    net_sums: list[float] = []
    first5: list[list[float]] = []
    for j in idx:
        xc = float(free_x[j]) + 0.5
        yc = float(free_y[j]) + 0.5
        flux_net, _, _ = _annulus_sky_subtracted_flux(d, xc, yc, float(r_ap), float(r_in), float(r_out))
        if math.isfinite(flux_net):
            net_sums.append(float(flux_net))
            if len(first5) < 5:
                first5.append([xc, yc])
        if len(net_sums) >= n_target:
            break

    n_valid = len(net_sums)
    if n_valid < n_min:
        _labbe_append_debug_record(
            {
                "frame_id": frame_id,
                "r_ap": float(r_ap),
                "seed_value": seed_value,
                "star_list_source": star_list_source,
                "n_stars": int(xs_c.size),
                "star_list_hash": star_list_hash,
                "mask_hash": mask_hash,
                "labbe_input_hash": labbe_input_hash,
                "n_attempted": int(n_try),
                "n_valid_apertures": int(n_valid),
                "first5_aperture_xy": first5,
                "sigma_result": None,
                "reason": f"crowding: {n_valid} valid empty apertures (< {n_min})",
            }
        )
        return float("nan"), n_valid, f"crowding: {n_valid} valid empty apertures (< {n_min})"
    sigma = _robust_scatter_mad(np.asarray(net_sums, dtype=np.float64))
    if not math.isfinite(sigma) or sigma < 0:
        return float("nan"), n_valid, "non_finite_scatter"
    _labbe_append_debug_record(
        {
            "frame_id": frame_id,
            "r_ap": float(r_ap),
            "seed_value": seed_value,
            "star_list_source": star_list_source,
            "n_stars": int(xs_c.size),
            "star_list_hash": star_list_hash,
            "mask_hash": mask_hash,
            "labbe_input_hash": labbe_input_hash,
            "n_attempted": int(n_try),
            "n_valid_apertures": int(n_valid),
            "first5_aperture_xy": first5,
            "sigma_result": float(sigma),
            "reason": "",
        }
    )
    return float(sigma), n_valid, ""

def estimate_star_free_per_pixel_variance_adu2(
    data: np.ndarray,
    star_x: np.ndarray | None = None,
    star_y: np.ndarray | None = None,
    exclusion_radius_px: float = 12.0,
) -> float | None:
    """Robust per-pixel background variance [ADU^2/px] from star-free pixels in one frame."""
    d = np.asarray(data, dtype=np.float64)
    if d.ndim != 2 or d.size == 0:
        return None
    xs = np.asarray(star_x if star_x is not None else [], dtype=np.float64)
    ys = np.asarray(star_y if star_y is not None else [], dtype=np.float64)
    blocked = _build_star_exclusion_mask(
        d.shape,
        xs,
        ys,
        float(exclusion_radius_px),
        float(exclusion_radius_px),
    )
    vals = d[~blocked]
    vals = vals[np.isfinite(vals)]
    if vals.size < 64:
        return None
    med = float(np.median(vals))
    resid = vals - med
    sig = _robust_scatter_mad(resid)
    if not math.isfinite(sig) or sig < 0:
        return None
    return float(sig * sig)

def _howell_bkg_variance_adu2(
    sky_pp: float,
    area: float,
    gain: float = 1.0,
    read_noise: float = 10.0,
) -> float:
    """Background + read-noise variance [ADU^2] from Howell (1989) eq. 2 (excludes source Poisson F/g)."""
    if not math.isfinite(sky_pp) or sky_pp < 0:
        sky_pp = 0.0
    if not math.isfinite(area) or area <= 0:
        return float("nan")
    g = float(gain) if math.isfinite(gain) and gain > 0 else 1.0
    rn = float(read_noise) if math.isfinite(read_noise) and read_noise >= 0 else 10.0
    return max(0.0, sky_pp) / g * area + (rn / g) ** 2 * area

def _clamp_bkg_scale_r(r: float) -> float:
    if not math.isfinite(r):
        return float("nan")
    return float(max(BKG_SCALE_R_CLAMP_LO, min(BKG_SCALE_R_CLAMP_HI, float(r))))

def bkg_scale_ratio_empirical_over_howell(
    sigma_bkg_ap: float,
    sky_pp: float,
    area: float,
    *,
    gain: float = 1.0,
    read_noise: float = 10.0,
) -> float:
    """Per-measurement r = sigma_bkg_ap^2 / howell_bkg_variance for hybrid fallback calibration."""
    sig = float(sigma_bkg_ap)
    if not math.isfinite(sig) or sig < 0:
        return float("nan")
    hb = _howell_bkg_variance_adu2(sky_pp, area, gain=gain, read_noise=read_noise)
    if not math.isfinite(hb) or hb <= 0:
        return float("nan")
    return float(sig * sig / hb)

def compute_setup_bkg_scale_r(ratios: list[float]) -> tuple[float, int]:
    """Median empirical/Howell background variance ratio; clamped to [0.05, 2.0]."""
    ok = [float(r) for r in ratios if math.isfinite(float(r)) and float(r) > 0]
    if not ok:
        return float("nan"), 0
    return _clamp_bkg_scale_r(float(np.median(np.asarray(ok, dtype=np.float64)))), len(ok)

def scaled_sigma_bkg_ap_from_howell(
    sky_pp: float,
    area: float,
    *,
    gain: float = 1.0,
    read_noise: float = 10.0,
    r_setup: float,
) -> float:
    """Calibrated fallback: sqrt(r_setup * howell_bkg_variance) [ADU] at aperture scale."""
    r_c = _clamp_bkg_scale_r(float(r_setup))
    hb = _howell_bkg_variance_adu2(sky_pp, area, gain=gain, read_noise=read_noise)
    if not math.isfinite(r_c) or not math.isfinite(hb) or hb < 0:
        return float("nan")
    return float(math.sqrt(r_c * hb))

def measure_growth_curve_ee(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    fwhm_px: float,
    sky_pp: np.ndarray | None = None,
    dao_flux: np.ndarray | None = None,
    peak_max_adu: np.ndarray | None = None,
    sat_limit_adu: np.ndarray | None = None,
    isolation_fwhm: float = 3.0,
    ref_fwhm: float = 4.5,
    ladder_step_px: float = 0.5,
    min_stars: int = 8,
    snr_min: float = 50.0,
    sat_frac: float = 0.85,
    gain: float = 1.0,
    read_noise: float = 10.0,
    max_stars: int = 60,
    aperture_r_px: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measured draft growth curve (encircled energy), reusing COG-A1 selection rules.

    Isolation default is **3 FWHM** (catalogue neighbour exclusion for SNR sizing).
    Normalisation radius is ``ref_fwhm * fwhm_px``; callers should verify flatness there.

    Minimum ``min_stars`` default **8** matches ``cog_min_stars``: a robust median EE
    curve needs a small ensemble; Q4 used 12 when available, but 8 is the production
    COG gate. Below ``min_stars``, ``ok=False`` and the SNR table must fall back to
    the Gaussian model explicitly.
    """
    # Delegate star selection + ladder photometry to the existing COG builder so
    # SNR sizing and aperture-correction share one growth-curve implementation.
    n = int(len(x))
    if n == 0:
        return {
            "ok": False,
            "n_cog": 0,
            "ee_radii": None,
            "ee_curve": None,
            "ref_r_px": float("nan"),
            "flatness_tail_over_norm": float("nan"),
            "min_stars_required": int(min_stars),
            "isolation_fwhm": float(isolation_fwhm),
            "reason": "no_stars",
        }
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if dao_flux is None:
        flux = np.ones(n, dtype=np.float64)
    else:
        flux = np.asarray(dao_flux, dtype=np.float64)
    if sky_pp is None:
        skp = np.zeros(n, dtype=np.float64)
    else:
        skp = np.asarray(sky_pp, dtype=np.float64)
    if aperture_r_px is None:
        rap = np.full(n, max(1.0, 0.8 * float(fwhm_px)), dtype=np.float64)
    else:
        rap = np.asarray(aperture_r_px, dtype=np.float64)

    cog = compute_per_frame_cog_correction(
        np.asarray(data, dtype=np.float64),
        xx,
        yy,
        flux,
        rap,
        skp,
        fwhm_px=float(fwhm_px),
        peak_max_adu=peak_max_adu,
        sat_limit_adu=sat_limit_adu,
        ref_fwhm=float(ref_fwhm),
        ladder_step_px=float(ladder_step_px),
        min_stars=int(min_stars),
        isolation_fwhm=float(isolation_fwhm),
        snr_min=float(snr_min),
        sat_frac=float(sat_frac),
        gain=float(gain),
        read_noise=float(read_noise),
        max_stars=int(max_stars),
        ladder_outer_factor=1.3,
    )
    ee_radii = cog.get("ee_radii")
    ee_curve = cog.get("ee_curve")
    ref_r = float(cog.get("ref_r_px", float("nan")))
    flat = float(cog.get("flatness_outer_over_norm", float("nan")))
    ok = bool(cog.get("cog_ok")) and ee_radii is not None and ee_curve is not None
    r90 = float("nan")
    if ok:
        from aperture_policy import ee_r90_continuous  # noqa: PLC0415

        r90 = ee_r90_continuous(ee_radii, ee_curve)
    return {
        "ok": ok,
        "n_cog": int(cog.get("n_cog", 0) or 0),
        "ee_radii": np.asarray(ee_radii, dtype=np.float64) if ee_radii is not None else None,
        "ee_curve": np.asarray(ee_curve, dtype=np.float64) if ee_curve is not None else None,
        "ref_r_px": ref_r,
        "ladder_outer_r_px": float(cog.get("ladder_outer_r_px", float("nan"))),
        "flatness_tail_over_norm": flat,
        "flatness_outer_over_norm": flat,
        "r90_px": r90,
        "min_stars_required": int(min_stars),
        "isolation_fwhm": float(isolation_fwhm),
        "reason": "" if ok else "too_few_isolated_cog_stars",
    }

def _phase2a_star_mag_lookup(
    at_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    masterstar_fits_path: Path,
) -> dict[str, float]:
    """Best-effort observed-band / catalog mag per ``catalog_id`` for SNR aperture lookup.

    Prefers ``mag`` / ``catalog_mag`` (filter-native) over broad Gaia ``phot_g_mean_mag``.
    """
    out: dict[str, float] = {}
    for df in (at_df, comp_df):
        if df is None or df.empty or "catalog_id" not in df.columns:
            continue
        for mag_col in _APERTURE_SIZING_MAG_COLS:
            if mag_col not in df.columns:
                continue
            for _, r in df.iterrows():
                cid = _normalize_gaia_id(r.get("catalog_id", ""))
                if not cid or cid in out:
                    continue
                v = pd.to_numeric(r.get(mag_col), errors="coerce")
                if math.isfinite(float(v)):
                    out[cid] = float(v)
    try:
        ms_full = Path(masterstar_fits_path).resolve().parent / "masterstars_full_match.csv"
        if ms_full.is_file():
            ms_df0 = pd.read_csv(
                ms_full,
                low_memory=False,
                usecols=lambda c: c in ("catalog_id", *_APERTURE_SIZING_MAG_COLS),
                dtype=_GAIA_ID_DTYPE,
            )
            ms_df0["catalog_id"] = ms_df0["catalog_id"].apply(_normalize_gaia_id)
            for mag_col in _APERTURE_SIZING_MAG_COLS:
                if mag_col not in ms_df0.columns:
                    continue
                for _, r in ms_df0.iterrows():
                    cid = str(r.get("catalog_id") or "").strip()
                    if not cid or cid in out:
                        continue
                    v = pd.to_numeric(r.get(mag_col), errors="coerce")
                    if math.isfinite(float(v)):
                        out[cid] = float(v)
    except Exception as exc:  # noqa: BLE001
        logging.error("[EXC-0126] Per-star mag from masterstars CSV cache load fails - aperture sizing lacks that star's ...: %s", exc)
        pass
    return out

def discover_aligned_science_fits(aligned_root: Path | str, *, max_n: int = 200) -> list[Path]:
    """Science FITS for SNR/CoG (prefer ``*_Light_*.fits``; never require ``proc_*.fits``)."""
    root = Path(aligned_root)
    if not root.is_dir():
        return []
    out: list[Path] = []
    try:
        cands = sorted(p for p in root.rglob("*.fits") if p.is_file())
    except Exception:  # noqa: BLE001
        return []
    for p in cands:
        name_u = p.name.upper()
        if name_u == "MASTERSTAR.FITS" or name_u.startswith("PROC_"):
            continue
        out.append(p)
        if len(out) >= int(max_n):
            break
    if out:
        return out
    # Last resort: calibrated proc frames if that is all that exists.
    for p in cands:
        if p.name.upper().startswith("PROC_"):
            out.append(p)
            if len(out) >= int(max_n):
                break
    return out

def _median_bkg_var_from_aligned_frames(
    *,
    aligned_fits_paths: Sequence[Path | str] | None = None,
    aligned_ram_frames: Sequence[tuple[str, Any, Any]] | None = None,
    max_frames: int = 6,
) -> float | None:
    """Median star-free per-pixel variance across aligned frames (no catalog - edge patches)."""
    vals: list[float] = []
    n_max = max(1, int(max_frames))

    if aligned_ram_frames:
        for _name, _hdr, arr in list(aligned_ram_frames)[:n_max]:
            v = estimate_star_free_per_pixel_variance_adu2(arr)
            if v is not None:
                vals.append(float(v))

    if aligned_fits_paths:
        for raw in list(aligned_fits_paths)[:n_max]:
            p = Path(raw)
            if not p.is_file():
                continue
            try:
                with astrofits.open(p, memmap=True) as hdul:
                    d = hdul[0].data
                if d is None:
                    continue
                v = estimate_star_free_per_pixel_variance_adu2(d)
                if v is not None:
                    vals.append(float(v))
            except Exception:  # noqa: BLE001
                continue

    if not vals:
        return None
    med = float(np.nanmedian(np.asarray(vals, dtype=np.float64)))
    return med if math.isfinite(med) and med >= 0 else None

def _estimate_annulus_sky_pp(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    r_in: float,
    r_out: float,
) -> np.ndarray:
    """Per-star exact annulus sky (ADU/px) for COG when catalog sky is missing."""
    from sky_estimation import sky_exact_mean  # noqa: PLC0415

    n = int(len(x))
    out = np.full(n, float("nan"), dtype=np.float64)
    d = np.asarray(data, dtype=np.float64)
    for i in range(n):
        try:
            out[i] = sky_exact_mean(d, float(x[i]), float(y[i]), r_in=float(r_in), r_out=float(r_out))
        except Exception:  # noqa: BLE001
            continue
    return out

def _annulus_sky_subtracted_flux(
    data: np.ndarray,
    x_c: float,
    y_c: float,
    r_ap: float,
    r_in: float,
    r_out: float,
) -> tuple[float, float, float]:
    """Sky-subtracted aperture sum, annulus sky median, peak in aperture (shared DAO/PSF path)."""
    if not (math.isfinite(x_c) and math.isfinite(y_c) and math.isfinite(r_ap) and r_ap > 0):
        return float("nan"), float("nan"), float("nan")
    try:
        from photutils.aperture import CircularAnnulus, CircularAperture
        from photutils.aperture import aperture_photometry as _aphot
    except ImportError:
        return float("nan"), float("nan"), float("nan")

    d = np.asarray(data, dtype=np.float64)
    if np.any(~np.isfinite(d)):
        fill = float(np.nanmedian(d)) if np.any(np.isfinite(d)) else 0.0
        d = np.where(np.isfinite(d), d, fill)

    pos = np.array([[float(x_c), float(y_c)]], dtype=np.float64)
    ap = CircularAperture(pos, r=float(r_ap))
    an = CircularAnnulus(pos, r_in=float(r_in), r_out=float(r_out))

    phot_ap = _aphot(d, ap, method="exact")
    sum_ap = float(np.asarray(phot_ap["aperture_sum"], dtype=np.float64).ravel()[0])
    area_ap = float(ap.area)

    sky_pp = float("nan")
    sky_ok = False
    ann_masks = an.to_mask(method="center")
    if not isinstance(ann_masks, (list, tuple)):
        ann_masks = [ann_masks]
    for amask in ann_masks:
        try:
            ann_img = amask.to_image(d.shape)
            sky_pp = _sky_pp_from_annulus_image(d, ann_img)
            sky_ok = math.isfinite(sky_pp)
            if sky_ok:
                break
        except (ValueError, TypeError, IndexError, AttributeError) as exc:
            from except_fix_counters import get_except_fix_counters

            get_except_fix_counters().sky_annulus_mask_fail += 1
            logging.error(
                "[PHOT] annulus sky mask failed x=%.2f y=%.2f: %s",
                float(x_c),
                float(y_c),
                exc,
            )
            continue

    if not sky_ok:
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().sky_annulus_invalid += 1
        logging.error(
            "[PHOT] annulus sky invalid (no usable pixels) x=%.2f y=%.2f r_ap=%.2f",
            float(x_c),
            float(y_c),
            float(r_ap),
        )
        peak_local = float("nan")
        try:
            m_ap = ap.to_mask(method="center")
            if isinstance(m_ap, (list, tuple)):
                m0 = m_ap[0]
            else:
                m0 = m_ap
            vals = m0.get_values(d)
            peak_local = (
                float(np.nanmax(np.asarray(vals, dtype=np.float64)))
                if vals is not None
                else float("nan")
            )
        except (ValueError, TypeError, IndexError):
            peak_local = float("nan")
        return float("nan"), float("nan"), peak_local

    flux_net = float(sum_ap - sky_pp * area_ap)
    try:
        m_ap = ap.to_mask(method="center")
        if isinstance(m_ap, (list, tuple)):
            m0 = m_ap[0]
        else:
            m0 = m_ap
        vals = m0.get_values(d)
        peak_local = float(np.nanmax(np.asarray(vals, dtype=np.float64))) if vals is not None else float("nan")
    except Exception:  # noqa: BLE001
        peak_local = float("nan")

    return flux_net, sky_pp, peak_local

def _resolve_star_flux_method(cid: str, all_frames: pd.DataFrame) -> str:
    """One routing decision per star (majority of per-frame lc_flux_method)."""
    sub = all_frames[all_frames["catalog_id"] == cid]
    if sub.empty or "lc_flux_method" not in sub.columns:
        return "aperture"
    counts = sub["lc_flux_method"].astype(str).str.strip().str.lower().value_counts()
    if counts.empty:
        return "aperture"
    if int(counts.get("psf", 0)) > int(counts.get("aperture", 0)):
        return "psf"
    return "aperture"

def _frame_quality_gate_select(
    csv_files: list[Path],
    cfg: AppConfig | None,
    proc_frame_store: ProcFrameStore | None,
) -> tuple[list[Path], list[str]]:
    """Passthrough: MAD/z-score whole-frame rejection removed (zero-clipping 2026-08-12).

    Returns ``(list(csv_files), [])`` always. Call signature kept for call-site compatibility.
    """
    _ = (cfg, proc_frame_store)
    return list(csv_files), []

def _recompute_bjd_hjd_per_target(
    jd_array: np.ndarray,
    ra_deg: float,
    dec_deg: float,
    cfg: AppConfig,
    site: tuple[float | None, float | None, float | None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute per-target BJD(TDB) and HJD from frame JD values.

    Uses target's own RA/Dec instead of field-center coordinates.
    Eliminates Roemer LTT error of up to ~12s for targets at field edge.

    Batch astropy Time() over all frames (scalar compute_hjd_bjd is ~12 ms/call).

    ``site`` (lat, lon, alt) is the per-draft resolved observer location
    (param_resolver: draft ID_LOCATION -> header SITELAT -> flagged config).
    When provided it OVERRIDES ``cfg.observer_*`` so Phase 2A is independent of
    config drift between sessions. ``cfg`` is used only as a legacy fallback.

    References:
        Eastman, Siverd & Gaudi (2010) PASP 122, 935 - BJD standards
        time_utils.compute_hjd_bjd() for scalar equivalence
    """
    from photometry_core import _recompute_bjd_hjd_with_status

    bjd, hjd, _ = _recompute_bjd_hjd_with_status(jd_array, ra_deg, dec_deg, cfg, site=site)
    return bjd, hjd

def photometer_check_star_production_path(
    *,
    state: _Phase2AState,
    parent_target_cid: str,
    check_cid: str,
    masterstar_fits_path: Path,
    lc_dir: Path,
    output_dir: Path,
    annulus_inner_fwhm: float = 4.0,
    annulus_outer_fwhm: float = 6.0,
    outlier_sigma: float = 3.0,
    stability_sigma: float = 3.0,
    _apt_fw: float | None = None,
    _save_png: bool = False,
) -> pd.DataFrame | None:
    """Diagnostic: photometer a check star via the production Phase 2A target path.

    Uses the parent target's comparison ensemble (minus the check star) and production
    ``err`` from ``save_lightcurve_csv``. Intended for check-star chi2 validation.
    """
    from dataclasses import replace

    from photometry_core import _phase2a_process_one_target

    parent_target_cid = _normalize_gaia_id(parent_target_cid)
    check_cid = _normalize_gaia_id(check_cid)
    if not parent_target_cid or not check_cid or parent_target_cid == check_cid:
        return None
    parent_comps = state._comp_index.get(parent_target_cid, pd.DataFrame())
    if parent_comps.empty:
        return None
    comp_subset = parent_comps.loc[parent_comps["catalog_id"] != check_cid].copy()
    if comp_subset.empty:
        return None
    ms = state.masterstars_df
    if ms.empty or "catalog_id" not in ms.columns:
        return None
    check_ms = ms.loc[ms["catalog_id"] == check_cid]
    if check_ms.empty:
        return None
    target_row = check_ms.iloc[0].copy()
    target_row["skip_photometry"] = False
    diag_state = replace(
        state,
        _comp_index={**state._comp_index, check_cid: comp_subset.reset_index(drop=True)},
        _nt=1,
    )
    _apt = float(_apt_fw if _apt_fw is not None else getattr(state._cfg, "aperture_fwhm_factor", 2.5))
    ac_logged: list[bool] = [False]
    summary_rows: list = []
    n_lc = 0
    _phase2a_process_one_target(
        target_row,
        ti=1,
        state=diag_state,
        summary_rows=summary_rows,
        n_lc=n_lc,
        lc_dir=Path(lc_dir),
        output_dir=Path(output_dir),
        progress_cb=None,
        masterstar_fits_path=Path(masterstar_fits_path),
        annulus_inner_fwhm=float(annulus_inner_fwhm),
        annulus_outer_fwhm=float(annulus_outer_fwhm),
        outlier_sigma=float(outlier_sigma),
        stability_sigma=float(stability_sigma),
        _apt_fw=_apt,
        _save_png=bool(_save_png),
        ac_sign_logged=ac_logged,
    )
    lc_path = Path(lc_dir) / f"lightcurve_{check_cid}.csv"
    if not lc_path.is_file():
        return None
    return pd.read_csv(lc_path, low_memory=False)

def _compute_fov_max_dist(
    frame_w_px: int,
    frame_h_px: int,
    plate_scale: float | None,
    fov_fraction: float,
    fallback_deg: float,
) -> float:
    """
    max_dist_deg = (FOV_diagonal / 2) * fov_fraction

    Pouzi plate_solve_fov_deg_diagonal_from_scale z utils.
    Ak plate_scale je None -> vrat fallback_deg.
    """
    logging.info(
        "[FOV] compute: w=%d h=%d scale=%s fraction=%.2f fallback=%.3f",
        int(frame_w_px),
        int(frame_h_px),
        (f"{float(plate_scale):.4f}" if plate_scale else "None"),
        float(fov_fraction),
        float(fallback_deg),
    )
    if not plate_scale or float(plate_scale) <= 0:
        logging.debug(
            "[FAZA 0+1] plate_scale neznamy -> max_dist fallback=%.3f deg",
            float(fallback_deg),
        )
        return float(fallback_deg)
    try:
        diag_deg = plate_solve_fov_deg_diagonal_from_scale(
            int(frame_w_px), int(frame_h_px), float(plate_scale)
        )
        if diag_deg is None or not math.isfinite(float(diag_deg)) or float(diag_deg) <= 0:
            raise ValueError(f"invalid diag_deg={diag_deg!r}")
        result = (float(diag_deg) / 2.0) * float(fov_fraction)
        logging.info(
            "[FAZA 0+1] FOV max_dist: scale=%.3f\"/px, "
            "diag=%.3f deg, fraction=%.2f -> max_dist=%.3f deg",
            float(plate_scale),
            float(diag_deg),
            float(fov_fraction),
            float(result),
        )
        return float(result)
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0182] FOV max_dist degree calculation fails - comp/target cone uses hardcoded fallback radius: %s', exc)
        logging.warning(
            "[FAZA 0+1] FOV max_dist vypocet zlyhal (%s) -> fallback=%.3f deg",
            exc,
            float(fallback_deg),
        )
        return float(fallback_deg)

def _sky_pp_from_annulus_image(d: np.ndarray, ann_img: np.ndarray) -> float:
    """Local sky (ADU/px) from annulus mask image - plain median, no rejection (SKY-CLIP-01)."""
    from sky_estimation import sky_median_mask  # noqa: PLC0415

    return sky_median_mask(d, ann_img)

def _aperture_flux_sky_per_star(
    d: np.ndarray,
    pos: np.ndarray,
    r_ap_arr: np.ndarray,
    r_in_arr: np.ndarray,
    r_out_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-star circular aperture + annulus sky (photutils 2.3 requires scalar ``r`` per aperture).

    Aperture sum uses photutils ``method='exact'`` (fractional pixel overlap). Binary
    ``center`` masking produces radius-parity sawtooth in growth/scatter ladders
    (IMPL-04). Annulus sky still uses ``center`` masks for median sampling only.
    """
    from photutils.aperture import CircularAnnulus, CircularAperture
    from photutils.aperture import aperture_photometry as _aphot

    n = int(len(pos))
    flux_arr = np.full(n, np.nan, dtype=np.float64)
    sky_pp_arr = np.full(n, np.nan, dtype=np.float64)
    n_fail = 0
    for idx in range(n):
        try:
            r_ap = float(r_ap_arr[idx])
            r_in = float(r_in_arr[idx])
            r_out = float(r_out_arr[idx])
            if not (
                math.isfinite(r_ap)
                and r_ap > 0
                and math.isfinite(r_in)
                and r_in > 0
                and math.isfinite(r_out)
                and r_out > r_in
            ):
                n_fail += 1
                continue
            xy = (float(pos[idx, 0]), float(pos[idx, 1]))
            ap_i = CircularAperture([xy], r=r_ap)
            an_i = CircularAnnulus([xy], r_in=r_in, r_out=r_out)
            phot_i = _aphot(d, ap_i, method="exact")
            ann_masks = an_i.to_mask(method="center")
            if not isinstance(ann_masks, (list, tuple)):
                ann_masks = [ann_masks]
            ann_img = ann_masks[0].to_image(d.shape)
            sky_pp = _sky_pp_from_annulus_image(d, ann_img)
            area = float(ap_i.area)
            flux_arr[idx] = float(phot_i["aperture_sum"][0]) - sky_pp * area
            sky_pp_arr[idx] = sky_pp
        except Exception:  # noqa: BLE001
            n_fail += 1
    if n_fail > 0:
        logging.warning(
            "[FAZA 2A] Per-star aperture: %d/%d positions failed or skipped",
            n_fail,
            n,
        )
    return flux_arr, sky_pp_arr
