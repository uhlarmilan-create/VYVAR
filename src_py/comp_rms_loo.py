"""Leave-one-out differential-mag RMS for comparison candidacy (COMP-RMS-DEF-01-B).

Gated statistic is MAD*1.4826 of (mag(star) - its night median) - median over the other pool
stars of (mag - their night median), on loadable proc frames (COMP-RMS-NORM-2026-10). No clipping. The old mag-bin relative-flux MAD is
``comp_relflux_mad`` (diagnostic only).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from comp_pool_rms import _norm_id_val

LOGGER = logging.getLogger(__name__)

LN10_OVER_2P5 = 1.0857362047581294
COMP_RMS_LOO_PHOTON_K_DEFAULT = 5.0
COMP_RMS_FRAMES_BASIS = "all_loadable"
#: COMP-CEIL-FIELDK-2026-10: pool column with the photon (CCD-equation) magnitude error at the comp_rms
#: aperture and frames - the reference of the comp_rms ceiling (``photon_mag_err_map``)
COMP_PHOTON_MAG_COL = "comp_photon_mag_fieldk"
#: COMP-CEIL-NOISEMODEL-2026-10: pool columns of the ceiling reference - each star's scintillation (median over
#: its frames of the per-frame value of the light-curve error) and sigma_ref = sqrt((a e_photon)^2 + e_scint^2 +
#: floor^2) with a / floor fitted on the night's constant pool stars (``comp_ceiling_reference``)
COMP_SCINT_MAG_COL = "comp_scint_mag_fieldk"
COMP_SIGMA_REF_COL = "comp_sigma_ref_fieldk"
#: COMP-CEIL-NOISEMODEL-2026-10: fewest constant pool stars for the two-parameter fit of the ceiling reference -
#: the population an AP-BRIGHT-K brightness group needs (aperture_pertarget.BRIGHT_K_MIN_STARS) for its median
COMP_CEIL_FIT_MIN_STARS: int = 30


#: AP-BRIGHT-K-2026-10: bootstrap resamples of the one-SE aperture-k rule (``k_within_one_se``), used by the
#: brightness groups (``aperture_pertarget.field_k_by_brightness``) and the comparison field k
BRIGHT_K_BOOT: int = 200


def mad_sigma_scale_or_zero(values: np.ndarray) -> float:
    """1.4826*MAD after dropping non-finite; n<3 returns nan; MAD==0 returns 0.0."""
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 3:
        return float("nan")
    med = float(np.median(a))
    return float(1.4826 * np.median(np.abs(a - med)))


def _nan_row_median(m: np.ndarray) -> np.ndarray:
    """``np.median`` of the non-NaN values of every row (NaN for an empty row), bit-identical to it:
    odd n -> the middle value, even n -> (lower + upper) / 2, as ``np.mean`` of the two middle values."""
    out = np.full(m.shape[0], np.nan)
    if m.shape[1] == 0:
        return out
    srt = np.sort(m, axis=1)  # NaN sorts last
    n = np.sum(~np.isnan(m), axis=1)
    r = np.flatnonzero(n > 0)
    nn = n[r]
    lo = srt[r, (nn - 1) // 2]
    hi = srt[r, nn // 2]
    out[r] = np.where(nn % 2 == 1, lo, (lo + hi) / 2)
    return out


def _loo_median(v: np.ndarray) -> np.ndarray:
    """Median of all OTHER elements, for every element (vectorised, exact)."""
    v = np.asarray(v, dtype=np.float64)
    m = int(v.size)
    if m < 2:
        return np.full(m, np.nan)
    order = np.argsort(v, kind="mergesort")
    srt = v[order]
    rank = np.empty(m, dtype=np.int64)
    rank[order] = np.arange(m)
    n2 = m - 1

    def _pick(j: int) -> np.ndarray:
        # j-th element (0-based) of the sorted array with the element of rank r removed
        idx = np.where(j < rank, j, j + 1)
        return srt[idx]

    if n2 % 2 == 1:
        return _pick(n2 // 2)
    return 0.5 * (_pick(n2 // 2 - 1) + _pick(n2 // 2))


def _frame_loo_inputs(
    df: pd.DataFrame | None,
    row_of: dict[str, int],
    flux_cols: list[str],
    strict_flux_col: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """One frame, all flux columns at once: ``{flux_col: (pool rows, mags)}`` (``row_of``: normalised id -> row).

    Per column: rows with a pool id and flux > 0, per-id median flux (groupby, ids sorted), magnitude
    ``-2.5 math.log10`` per value (G07-31: ``np.log10`` differs in the last ulp), finite magnitudes only;
    a column with fewer than two such stars is left out (the frame does not count for that column).
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if df is None or df.empty:
        return out
    id_col = "catalog_id" if "catalog_id" in df.columns else ("name" if "name" in df.columns else None)
    if id_col is None:
        return out
    actual: dict[str, str] = {}
    for fc in flux_cols:
        if strict_flux_col and fc not in df.columns:
            continue  # never mix apertures between frames
        a = fc if fc in df.columns else "dao_flux"
        if a in df.columns:
            actual[fc] = a
    if not actual:
        return out
    nid = df[id_col].map(_norm_id_val)  # once per frame for every column (G07-31)
    keep = nid.isin(row_of.keys()).to_numpy()
    if not keep.any():
        return out
    # integer group codes in sorted-id order, shared by every column (same groups as a groupby on the ids)
    codes, uniq = pd.factorize(nid[keep].to_numpy(), sort=True)
    rows_u = np.fromiter((row_of[u] for u in uniq), dtype=np.int64, count=len(uniq))
    for fc, a in actual.items():
        flux = pd.to_numeric(df[a], errors="coerce").to_numpy()[keep]
        pos = flux > 0
        if not pos.any():
            continue
        flux_by = pd.Series(flux[pos]).groupby(codes[pos], sort=True).median()
        if len(flux_by) < 2:
            continue
        f = flux_by.to_numpy()
        # -2.5 * math.log10 per value (flux > 0 here; +inf gives -inf, dropped below), as _flux_to_mag
        mags = -2.5 * np.fromiter(map(math.log10, f), dtype=np.float64, count=f.size)
        okm = np.isfinite(mags)
        if int(okm.sum()) < 2:
            continue
        out[fc] = (rows_u[flux_by.index.to_numpy()][okm], mags[okm])
    return out


def loo_mag_rms_maps(
    cand_ids: set[str],
    per_frame_csv_paths: list[Path],
    csv_cache,
    flux_cols: list[str],
    *,
    min_frames_frac: float = 0.3,
    strict_flux_col: bool = False,
) -> dict[str, dict[str, float]]:
    """G07-31: ``{flux_col: {normalized_id: MAD*1.4826 of LOO dmag}}`` for several flux columns, one pass.

    The one LOO engine of comparison selection (field aperture k, comp_rms) and of the brightness-based
    aperture k (AP-BRIGHT-K). Pool is ``cand_ids`` excluding the star itself. Frames: every loadable
    proc table of ``csv_cache`` (the ProcFrameStore) among ``per_frame_csv_paths``. Every statistic is a
    median or MAD of a multiset, so the frame order does not enter.
    """
    cols = list(dict.fromkeys(flux_cols))
    if not cand_ids:
        return {c: {} for c in cols}
    norm_ids = sorted({_norm_id_val(x) for x in cand_ids if _norm_id_val(x)})
    if len(norm_ids) < 2:
        return {c: {} for c in cols}
    row_of = {cid: i for i, cid in enumerate(norm_ids)}
    # COMP-RMS-NORM-2026-10 (Milan 2026-10-26, D2): every pool star is first put on its own night
    # level (its median magnitude over the night), then each star is compared with the median of
    # the OTHER stars' offsets in the same frame. Without the normalisation the median of raw
    # magnitudes of stars of different brightness jumps whenever a star is missing in a frame
    # (edge, dropout) and gives spurious scatter (5-62 mmag in simulation). Honeycutt 1992;
    # Burdanov et al. 2014 (Astrokit).
    per_frame: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {c: [] for c in cols}
    for csv_path in per_frame_csv_paths:
        for fc, rows_mags in _frame_loo_inputs(csv_cache.get(str(csv_path)), row_of, cols, strict_flux_col).items():
            per_frame[fc].append(rows_mags)
    out_all: dict[str, dict[str, float]] = {}
    n_ids = len(norm_ids)
    for fc in cols:
        frames = per_frame[fc]
        n_frames = len(frames)
        # stars x frames; NaN = not measured (every stored magnitude is finite)
        mag_m = np.full((n_ids, n_frames), np.nan)
        for j, (rows, mags) in enumerate(frames):
            mag_m[rows, j] = mags
        level = _nan_row_median(mag_m)
        dmag_m = np.full((n_ids, n_frames), np.nan)
        for j, (rows, mags) in enumerate(frames):
            offs = mags - level[rows]
            loo = _loo_median(offs)
            ok = np.isfinite(loo)
            dmag_m[rows[ok], j] = offs[ok] - loo[ok]
        min_frames = max(3, int(n_frames * float(min_frames_frac)))
        n_ok = np.sum(~np.isnan(dmag_m), axis=1)
        # mad_sigma_scale_or_zero per star (n >= min_frames >= 3 here), row-wise
        med = _nan_row_median(dmag_m)
        rms_all = 1.4826 * _nan_row_median(np.abs(dmag_m - med[:, None]))
        out: dict[str, float] = {}
        for i, cid in enumerate(norm_ids):
            if n_ok[i] >= min_frames and math.isfinite(rms_all[i]):
                out[cid] = float(rms_all[i])
        LOGGER.info(
            "[COMP-RMS-LOO] %s frames_basis=%s n_loaded=%d n_stars=%d n_with_rms=%d",
            fc,
            COMP_RMS_FRAMES_BASIS,
            n_frames,
            n_ids,
            len(out),
        )
        out_all[fc] = out
    return out_all


def compute_loo_mag_rms_map(
    cand_ids: set[str],
    per_frame_csv_paths: list[Path],
    csv_cache: dict[str, pd.DataFrame],
    *,
    flux_col: str = "dao_flux",
    min_frames_frac: float = 0.3,
    strict_flux_col: bool = False,
) -> tuple[dict[str, float], str]:
    """Return ``{normalized_id: MAD*1.4826 of LOO dmag}`` and the frames-basis stamp (one column of
    ``loo_mag_rms_maps``)."""
    m = loo_mag_rms_maps(
        cand_ids, per_frame_csv_paths, csv_cache, [flux_col],
        min_frames_frac=min_frames_frac, strict_flux_col=strict_flux_col,
    )
    return m[flux_col], COMP_RMS_FRAMES_BASIS


def k_within_one_se(
    ks: Sequence[float], ratios: np.ndarray, rng: np.random.Generator
) -> tuple[float, np.ndarray, float]:
    """The one aperture-k rule of AP-BRIGHT-K-2026-10 and of the comparison field k (COMP-CEIL-FIELDK-2026-10).

    ``ratios``: stars x ks, each star's LOO scatter at every k divided by its own minimum over k (finite rows
    only). Curve = median over the stars per k; the chosen k is the LARGEST k whose curve value is within one
    bootstrap standard error (BRIGHT_K_BOOT resamples of the stars from ``rng``) of the curve minimum: larger k
    is less sensitive to seeing, centring and pixel phase, and a difference below the SE is not measured.
    Returns ``(k, curve, se)``. The bootstrap draws from ``rng`` in order (callers may share one generator).
    """
    curve = np.median(ratios, axis=0)
    j = int(np.argmin(curve))
    boot = [np.median(ratios[rng.integers(0, len(ratios), len(ratios)), j]) for _ in range(BRIGHT_K_BOOT)]
    se = float(np.std(boot))
    k_best = max(ks_within_one_se(ks, curve, se))
    return float(k_best), curve, se


def ks_within_one_se(ks: Sequence[float], curve: np.ndarray, se: float) -> list[float]:
    """Every k whose curve value is within ``se`` of the curve minimum (``k_within_one_se`` takes the largest).
    AP-BRIGHT-K-2026-10 diagnostic (Milan 2026-10-30, ``k_within_se`` in aperture_per_target.json and
    comp_pool_admission): the k the rule cannot tell apart on this night."""
    lim = curve[int(np.argmin(curve))] + se
    return [k for k, v in zip(ks, curve, strict=False) if v <= lim]


#: COMP-CEIL-FIELDK-2026-10: proc columns of the CCD-equation error of the pool fluxes (besides the flux)
PHOTON_REF_SKY_COLS: tuple[str, ...] = ("sky_adu_per_px_annulus", "sky_n_px")


def photon_mag_err_map(
    cand_ids: set[str],
    per_frame_csv_paths: list[Path],
    csv_cache,
    flux_col: str,
    *,
    k: float | None,
    gain: float,
    read_noise: float,
) -> dict[str, float]:
    """COMP-CEIL-FIELDK-2026-10 (Milan 2026-10-30): ``{normalized_id: photon magnitude error}`` of each pool
    star at the aperture and on the frames of its ``comp_rms`` - the reference of the comparison RMS ceiling.

    Per frame and star: the CCD-equation relative error (``ccd_noise.ccd_rel_error``, Merline & Howell 1995)
    of the proc flux ``flux_col`` with the proc annulus sky (``sky_adu_per_px_annulus``), its real number of
    sky pixels (``sky_n_px``) and the aperture area of that flux: r = ``flux_k_radius_px(k,
    fwhm_px_for_aperture)`` for a grid column (``k`` given), ``aperture_r_px`` for ``dao_flux`` (``k`` None).
    Per star: the median over the frames in which the star has a positive flux (the frames ``loo_mag_rms_maps``
    uses; several rows of one id in a frame -> their median), times 2.5 / ln 10. The MASTERSTAR S/N
    (``snr_ap_pixscaled``: one frame, aperture 1.9 FWHM) did not move with the field k.

    Fail loud: a frame carrying ``flux_col`` but not the sky / aperture columns -> RuntimeError (proc CSVs made
    before CCD-ONE-2026-10); gain / read noise not finite -> ValueError (``ccd_noise``).
    """
    from ccd_noise import ccd_rel_error, check_gain_read_noise  # noqa: PLC0415
    from photometry_defs import flux_k_radius_px  # noqa: PLC0415

    g, rn = check_gain_read_noise(gain, read_noise)
    norm_ids = sorted({_norm_id_val(x) for x in cand_ids if _norm_id_val(x)})
    if not norm_ids:
        return {}
    row_of = {cid: i for i, cid in enumerate(norm_ids)}
    geom_col = "fwhm_px_for_aperture" if k is not None else "aperture_r_px"
    need = (*PHOTON_REF_SKY_COLS, geom_col)
    cols: list[np.ndarray] = []
    for csv_path in per_frame_csv_paths:
        df = csv_cache.get(str(csv_path))
        if df is None or df.empty or flux_col not in df.columns:
            continue
        id_col = "catalog_id" if "catalog_id" in df.columns else ("name" if "name" in df.columns else None)
        if id_col is None:
            continue
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"comparison RMS ceiling: proc CSV {Path(str(csv_path)).name} without {', '.join(missing)} "
                "(made before CCD-ONE-2026-10) - re-run the per-frame photometry of this night"
            )
        nid = df[id_col].map(_norm_id_val)
        keep = nid.isin(row_of.keys()).to_numpy()
        flux = pd.to_numeric(df[flux_col], errors="coerce").to_numpy(dtype=np.float64)
        keep &= flux > 0
        if not keep.any():
            continue
        geom = pd.to_numeric(df[geom_col], errors="coerce").to_numpy(dtype=np.float64)[keep]
        if k is not None:
            with np.errstate(invalid="ignore"):
                r_ap = np.array([flux_k_radius_px(k, v) if math.isfinite(v) else np.nan for v in geom])
        else:
            r_ap = geom
        rel = ccd_rel_error(
            flux[keep],
            pd.to_numeric(df[PHOTON_REF_SKY_COLS[0]], errors="coerce").to_numpy(dtype=np.float64)[keep],
            math.pi * r_ap * r_ap,
            pd.to_numeric(df[PHOTON_REF_SKY_COLS[1]], errors="coerce").to_numpy(dtype=np.float64)[keep],
            gain=g,
            read_noise=rn,
        )
        col = np.full(len(norm_ids), np.nan)
        by_id = pd.Series(rel).groupby(nid[keep].to_numpy()).median()
        col[[row_of[c] for c in by_id.index]] = by_id.to_numpy()
        cols.append(col)
    if not cols:
        return {}
    med = _nan_row_median(np.column_stack(cols))
    return {cid: float(LN10_OVER_2P5 * med[i]) for i, cid in enumerate(norm_ids) if math.isfinite(med[i])}


def scint_mag_err_map(
    cand_ids: set[str],
    per_frame_csv_paths: list[Path],
    csv_cache,
    flux_col: str,
    *,
    rig: Any,
) -> tuple[dict[str, float], dict[str, Any]]:
    """COMP-CEIL-NOISEMODEL-2026-10 (Milan 2026-10-30): ``{normalized_id: scintillation [mag]}`` of each pool
    star - the median over the frames in which it has a positive ``flux_col`` (the frames of its ``comp_rms``
    and of ``photon_mag_err_map``) of the per-frame scintillation the light-curve errors use
    (``sigma_floor_core.frame_scintillation_mag`` at the frame airmass, rig ``sigma_budget``
    ``RigScintillationParams``; 0 for a frame without a valid airmass, as there).
    Returns ``(map, info)``; info counts the frames used and those without an airmass."""
    from sigma_floor_core import frame_scintillation_mag  # noqa: PLC0415

    norm_ids = sorted({_norm_id_val(x) for x in cand_ids if _norm_id_val(x)})
    info: dict[str, Any] = {"n_frames": 0, "n_frames_no_airmass": 0}
    if not norm_ids:
        return {}, info
    row_of = {cid: i for i, cid in enumerate(norm_ids)}
    cols: list[np.ndarray] = []
    per_frame: list[float] = []
    for csv_path in per_frame_csv_paths:
        df = csv_cache.get(str(csv_path))
        if df is None or df.empty or flux_col not in df.columns:
            continue
        id_col = "catalog_id" if "catalog_id" in df.columns else ("name" if "name" in df.columns else None)
        if id_col is None:
            continue
        nid = df[id_col].map(_norm_id_val)
        keep = nid.isin(row_of.keys()).to_numpy()
        keep &= pd.to_numeric(df[flux_col], errors="coerce").to_numpy(dtype=np.float64) > 0
        if not keep.any():
            continue
        am = float("nan")
        if "airmass" in df.columns:
            v = pd.to_numeric(df["airmass"], errors="coerce")
            if v.notna().any():
                am = float(v.median())
        if not (math.isfinite(am) and am >= 1.0):
            info["n_frames_no_airmass"] += 1
        sc = frame_scintillation_mag(rig, am)
        per_frame.append(sc)
        col = np.full(len(norm_ids), np.nan)
        col[[row_of[c] for c in set(nid[keep])]] = sc
        cols.append(col)
    info["n_frames"] = len(cols)
    if per_frame:
        info["scint_frame_mag_median"] = float(np.median(per_frame))
        info["scint_frame_mag_range"] = [float(min(per_frame)), float(max(per_frame))]
    if not cols:
        return {}, info
    med = _nan_row_median(np.column_stack(cols))
    return {cid: float(med[i]) for i, cid in enumerate(norm_ids) if math.isfinite(med[i])}, info


def comp_ceiling_reference(
    rms: np.ndarray, photon: np.ndarray, scint: np.ndarray, constant: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """COMP-CEIL-NOISEMODEL-2026-10 (Milan 2026-10-30): reference of the comparison RMS ceiling per pool star,
    ``sigma_ref = sqrt((a e_photon)^2 + e_scint^2 + floor^2)``.

    ``a`` (scale of the CCD-equation error) and ``floor`` (the night's excess common to all stars at the field
    k: aperture loss with seeing, centring, flat field) are fitted on the constant pool stars (``constant``:
    not a VSX / Gaia DR3 variable, not saturated) with finite positive ``rms`` (comp_rms) and photon error, by
    the one robust log fit of the field variability search (``stats_core.fit_scatter_noise_model``; Collier
    Cameron et al. 2006 / Sokolovsky et al. 2017: an empirical floor measured on the field). Each star is then
    judged against the field population at its own photon and scintillation error.

    Without a fit (fewer than COMP_CEIL_FIT_MIN_STARS constant stars, or the optimiser did not converge) the
    unfitted model a = 1, floor = 0 (photon + scintillation) is used, with a WARNING and ``status`` saying which
    and why. Returns ``(sigma_ref, info)``.
    """
    from stats_core import fit_scatter_noise_model  # noqa: PLC0415

    rms = np.asarray(rms, dtype=np.float64)
    ph = np.asarray(photon, dtype=np.float64)
    sc = np.asarray(scint, dtype=np.float64)
    fit = (np.asarray(constant, dtype=bool) & np.isfinite(rms) & (rms > 0) & np.isfinite(ph) & (ph > 0)
           & np.isfinite(sc))
    n = int(fit.sum())
    info: dict[str, Any] = {
        "model": "sigma_ref = sqrt((a * e_photon)^2 + e_scint^2 + floor^2)",
        "fit": "robust log least squares (soft_l1), stats_core.fit_scatter_noise_model, constant pool stars",
        "n_stars": n,
        "n_min": COMP_CEIL_FIT_MIN_STARS,
    }
    if n < COMP_CEIL_FIT_MIN_STARS:
        a, fl = 1.0, 0.0
        info["status"] = (f"too_few_stars: {n} constant pool stars with comp_rms and photon error "
                          f"< {COMP_CEIL_FIT_MIN_STARS} - unfitted model a = 1, floor = 0 used")
    else:
        a, fl, ok = fit_scatter_noise_model(rms[fit], ph[fit], extra=sc[fit])
        if ok:
            info["status"] = "ok"
        else:
            info["status"] = (f"not_converged: fit on {n} constant pool stars did not converge "
                              f"(a {a:.4g}, floor {fl:.4g}) - unfitted model a = 1, floor = 0 used")
            a, fl = 1.0, 0.0
    if info["status"] != "ok":
        LOGGER.warning("[COMP-CEIL] comparison ceiling reference: %s", info["status"])
    info["a"] = float(a)
    info["floor_mag"] = float(fl)
    sigma_ref = np.sqrt((a * ph) ** 2 + sc**2 + fl**2)
    return sigma_ref, info


def choose_field_aperture_k(
    cand_ids: set[str],
    per_frame_csv_paths: list[Path],
    csv_cache: dict[str, pd.DataFrame],
    *,
    min_frames_frac: float = 0.3,
) -> tuple[float | None, dict[str, float], dict[float, float], dict[str, Any]]:
    """APERTURE-K-2026-10: field aperture k for comparison-star RMS.

    For every k of the light-curve aperture grid (proc columns ``flux_kNNN``) compute the
    LOO magnitude RMS of the pool over the stars measured at every k. COMP-CEIL-FIELDK-2026-10
    (Milan 2026-10-30): the k is chosen by the rule of AP-BRIGHT-K (``k_within_one_se``):
    each star's RMS divided by its own minimum over k, median over the stars, the LARGEST k within one
    bootstrap standard error of the minimum (was: within a fixed 1 % of the minimum median RMS, which a
    0.1 % flux shift flipped on Brno 534 g).
    Returns (k, loo_map at k, {k: median rms}, info) with info = {"se", "ratio_curve", "n_stars", "rule"};
    (None, {}, {}, {}) without grid columns or with fewer than 3 common stars.
    """
    from photometry_defs import DEFAULT_APERTURE_F_GRID  # noqa: PLC0415
    from photometry_shared import flux_k_column  # noqa: PLC0415

    have: set[float] | None = None
    for csv_path in per_frame_csv_paths:
        df = csv_cache.get(str(csv_path))
        if df is None or df.empty:
            continue
        ks = {k for k in DEFAULT_APERTURE_F_GRID if flux_k_column(k) in df.columns}
        have = ks if have is None else (have & ks)
    if not have:
        return None, {}, {}, {}
    ks = sorted(have)
    by_col = loo_mag_rms_maps(
        cand_ids,
        per_frame_csv_paths,
        csv_cache,
        [flux_k_column(k) for k in ks],
        min_frames_frac=min_frames_frac,
        strict_flux_col=True,
    )
    maps: dict[float, dict[str, float]] = {k: by_col[flux_k_column(k)] for k in ks}
    common = set.intersection(*(set(m) for m in maps.values())) if maps else set()
    if len(common) < 3:
        return None, {}, {}, {}
    scores = {k: float(np.median([maps[k][c] for c in common])) for k in maps}
    # the same stars at every k (paired): each star on its own minimum, as AP-BRIGHT-K
    rms_m = np.array([[maps[k][c] for k in ks] for c in sorted(common)], dtype=np.float64)
    mn = rms_m.min(axis=1)
    pos = mn > 0
    ratios = rms_m[pos] / mn[pos, None]
    ratios = ratios[np.all(np.isfinite(ratios), axis=1)]
    if len(ratios) < 3:
        return None, {}, {}, {}
    k_best, curve, se = k_within_one_se(ks, ratios, np.random.default_rng(0))
    info: dict[str, Any] = {
        "rule": "largest k within one bootstrap SE of the minimum median RMS ratio (AP-BRIGHT-K rule)",
        "se": float(se),
        "ratio_curve": {f"{k:.2f}": float(v) for k, v in zip(ks, curve, strict=True)},
        "n_stars": int(len(ratios)),
        "k_within_se": [float(k) for k in ks_within_one_se(ks, curve, se)],
    }
    LOGGER.info(
        "[APERTURE-K] field k for comparison RMS = %.2f (median LOO rms by k: %s; median ratio by k: %s; "
        "SE %.4f; n_stars=%d)",
        k_best,
        ", ".join(f"{k:.2f}:{v:.4f}" for k, v in sorted(scores.items())),
        ", ".join(f"{k:.2f}:{v:.4f}" for k, v in zip(ks, curve, strict=True)),
        se,
        len(ratios),
    )
    return float(k_best), maps[k_best], scores, info
