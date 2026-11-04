# -*- coding: ascii -*-
"""Search for new variable stars in the whole field (VAR-2026-10).

One function for the night runner and the UI (:func:`run_field_variability`); the UI
only reads its products.

Method (Sokolovsky et al. 2017, A&A 598, A108; VaST, Sokolovsky & Lebedev 2018):

* Every star measured on >= 90 % of the frames, on the aperture grid column of the k
  its light curve would use (C1-2026-10: the AP-BRIGHT-K group of its instrumental
  magnitude, aperture_per_target.json; r = k x FWHM_frame), put on one magnitude scale
  by a per-frame aperture correction to the largest group k (median m(k_ref) - m(k)
  of the stars of the k_ref group; Stetson 1990), sky = median of the annulus.
* Common frame zero point (Honeycutt 1992): m_it = M_i + E_t, solved by alternating
  medians over reference stars (expected error < 1 %, all gates passed).
* Expected error of each point from the CCD equation (Merline & Howell 1995) with the
  run's gain / read noise; the field noise model s_exp^2 = (a e)^2 + floor^2 is fitted
  to the robust scatter of all non-variable stars (a, floor from the data).
* Index 1 - robust scatter s = 1.4826 MAD (IQR family): z_s from ln(s / s_exp) against
  the field (median, 1.4826 MAD).
* Index 2 - 1/eta (von Neumann ratio): u = ln(1/eta) standardised against stars of
  similar brightness (+-0.5 mag window, widened to >= 15 stars); the tail of u is a
  skew-normal matched to the 10/50/90 % quantiles of the whole field (Azzalini 1985;
  robust to the field's own variables) and converted to a Gaussian z.
  A Gaussian tail on u gives 20-60x too many false alarms (white noise), a linear
  eta cannot reach the threshold on red-noise nights (eta >= 0); skew-normal: 1-5x
  nominal in Monte Carlo of white, AR(1) and mixed noise, N = 30-87 frames.
* Candidate: z_s or z_eta above the threshold of a field false-alarm probability of
  1 % split over the two indices (z = Phi^-1(1 - 0.005 / N)), and all gates passed:
  not likely_saturated on >= 90 % of frames, median SNR >= snr_min, |neighbour light|
  <= 1 % in its aperture incl. the sky-annulus effect (comp_blend), sky annulus inside every frame footprint, not a
  known variable / target (those are marked, not listed).
* Systematics (G10-08 SYS-FIELD-RESPONSE-2026-10 + SYS-SEEING-DIFF-2026-10): if the signal disappears after
  removing the star's OWN response to frame FWHM, measured from frame-to-frame differences (``fwhm_coef_diff``;
  a slow intrinsic change does not enter it), and the FIELD's airmass response (Theil-Sen line in BP-RP = colour
  extinction + median of similar-magnitude gated non-signal stars), the star is flagged ``systematic`` and not
  listed. The star's own level OLS fit is reported as ``trend_like`` and its own level FWHM coefficient against
  the field as ``seeing_response_z``; neither removes a star.

TRANSIENT-2026-10 (``transient_search_enabled``, off by default): the same run also compares every
star's night magnitude with Gaia - expected = Gaia G through the field relation (robust fit,
colour terms) plus the light of the Gaia neighbours in the aperture (``blend``); ``dmag_vs_gaia``
<= -1 mag and <= -3 sigma of stars of the same brightness -> ``brightened`` (a nova on a faint
catalogued star, a dwarf nova in outburst all night) - and runs the search for new sources
(:mod:`transient_field`).

No sigma clipping and no detrending of the light curves themselves.

Times (G02-18, decision C6 2026-10-29): the frames are ordered and spaced by the field-centre frame
time of the proc catalogues (``bjd_tdb_mid``, else ``jd_mid``; meta ``frame_time_column``); the
``time`` of ``varcand_<id>.csv`` (and of the transients) is the star's own BJD_TDB from ``jd_mid``,
its RA/Dec and the run's site (meta ``time_column`` = ``bjd_tdb``; ``jd_mid`` without a site).
Across a 200 mm field the field-centre value would be off by up to ~30 s at the edge.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stats_core import coerce_bool_series, fit_scatter_noise_model
from ccd_noise import limit_magnitudes  # LC-BELOW-LIMIT-2026-11: the one rule lives there
from frame_footprint import _frame_footprints, _inside_all  # LAYER-2026-10: moved there
from catalog_crossmatch import _catalog_check  # LAYER-2026-10: moved there

LOGGER = logging.getLogger(__name__)

COVERAGE_MIN = 0.90
FAP_FIELD = 0.01
REF_ERR_MAX = 0.01  # reference stars for the frame zero point: expected error < 1 %
#: TRANSIENT-2026-10 "brightened": night magnitude at least this much and this many sigma (stars of the same
#: brightness) above Gaia G with the neighbours' light
BRIGHTENED_DMAG = 1.0
BRIGHTENED_NSIGMA = 3.0
_COLS = [
    "catalog_id", "x", "y", "ra_deg", "dec_deg", "mag", "bp_rp", "likely_saturated",
    "photometry_ok", "vsx_known_variable", "fwhm_px_for_aperture", "sky_adu_per_px_annulus", "sky_n_px",
    "airmass", "bjd_tdb_mid", "jd_mid",
]


def star_times_bjd(
    frame: pd.DataFrame,
    ra_deg: Any,
    dec_deg: Any,
    site: tuple[float, float, float] | None,
) -> tuple[np.ndarray, str]:
    """Per-star time of every frame (G02-18, decision C6): (stars x frames array, time column name),
    from the frames' mid-exposure UTC JD (``frame["jd_utc"]``) - :func:`time_utils.star_times`."""
    if "jd_utc" not in frame:
        raise ValueError("variability: proc catalogues lack jd_mid - no star times")
    from time_utils import star_times  # noqa: PLC0415

    t, col = star_times(frame["jd_utc"].to_numpy(float), ra_deg, dec_deg, site)
    if col == "jd_mid" and site is None:
        LOGGER.warning("[VARIABILITY] observer site unknown - star times are UTC JD (no barycentric correction)")
    return t, col


FIELD_CSV = "variability_field.csv"
_LEGACY_CANDIDATES_CSV = "variability_candidates.csv"  # drafts before OUTPUT-REVIEW-2026-10


def load_variability_candidates(photometry_dir: Path) -> pd.DataFrame:
    """The new candidates (V1, V2, ... by z_s) of a setup, ``label`` first, with ``catalogs``.

    OUTPUT-REVIEW-2026-10: one file, ``variability_field.csv`` (``candidate`` = True rows); drafts made
    before keep a separate ``variability_candidates.csv``, which is read when the field table has no
    ``catalogs`` column. Empty frame (``label``, ``catalog_id``, ``catalogs``) when nothing exists.
    """
    phot = Path(photometry_dir)
    fp = phot / FIELD_CSV
    legacy = phot / _LEGACY_CANDIDATES_CSV
    empty = pd.DataFrame(columns=["label", "catalog_id", "catalogs"])
    if not fp.is_file():
        return pd.read_csv(legacy, dtype={"catalog_id": str}, low_memory=False) if legacy.is_file() else empty
    field = pd.read_csv(fp, dtype={"catalog_id": str}, low_memory=False)
    if "catalogs" not in field.columns and legacy.is_file():
        return pd.read_csv(legacy, dtype={"catalog_id": str}, low_memory=False)
    if "candidate" not in field.columns:
        return empty
    cand = field[coerce_bool_series(field["candidate"])].copy()
    if cand.empty:
        return empty
    cand["catalogs"] = cand.get("catalogs", pd.Series("", index=cand.index)).fillna("").astype(str)
    cand["label"] = cand["label"].fillna("").astype(str)
    cand = cand.sort_values("label", key=lambda s: pd.to_numeric(s.str.lstrip("V"), errors="coerce")).reset_index(drop=True)
    return cand[["label"] + [c for c in cand.columns if c != "label"]]


def write_candidate_catalogs(photometry_dir: Path, catalogs: dict[str, str]) -> None:
    """Store the catalogue check of candidates (catalog_id -> text) in ``variability_field.csv``."""
    fp = Path(photometry_dir) / FIELD_CSV
    field = pd.read_csv(fp, dtype={"catalog_id": str}, low_memory=False)
    if "catalogs" not in field.columns:
        field["catalogs"] = ""
    field["catalogs"] = field["catalogs"].fillna("").astype(str)
    hit = field["catalog_id"].astype(str).isin(catalogs)
    field.loc[hit, "catalogs"] = field.loc[hit, "catalog_id"].astype(str).map(catalogs)
    field.to_csv(fp, index=False)


def _eta(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    if r.size < 10:
        return float("nan")
    den = float(np.sum((r - r.mean()) ** 2))
    return float(np.sum(np.diff(r) ** 2) / den) if den > 0 else float("nan")


class _MagWindows:
    """G10-31: magnitude windows of one (val, mag, use) set over the stars sorted by magnitude. A window is
    the stars with ``abs(mag - mag_i) <= h`` (the exact predicate); the sorted order only narrows the
    candidates (searchsorted bounds with a slack), so membership and statistics equal a scan of all stars."""

    def __init__(self, val: np.ndarray, mag: np.ndarray, use: np.ndarray) -> None:
        self.val = np.asarray(val, dtype=float)
        self.mag = np.asarray(mag, dtype=float)
        pool = np.flatnonzero(np.asarray(use, dtype=bool) & np.isfinite(self.val) & np.isfinite(self.mag))
        order = np.argsort(self.mag[pool], kind="mergesort")
        self.idx = pool[order]
        self.ms = self.mag[self.idx]

    def _members(self, mag_i: float, h: float) -> np.ndarray:
        slack = 1e-9 * (1.0 + abs(mag_i) + h)
        lo = int(np.searchsorted(self.ms, mag_i - h - slack, side="left"))
        hi = int(np.searchsorted(self.ms, mag_i + h + slack, side="right"))
        cand = self.idx[lo:hi]
        return np.sort(cand[np.abs(self.mag[cand] - mag_i) <= h])  # original index order

    def stats(self, mag_i: float, half: float = 0.5, nmin: int = 15) -> tuple[float, float]:
        """(median, 1.4826 MAD) of ``val`` over stars within +-half mag of ``mag_i``,
        window widened x1.5 until it holds ``nmin`` stars (NaN when < 5)."""
        if not math.isfinite(mag_i):
            return float("nan"), float("nan")
        h = half
        while True:
            sel = self._members(mag_i, h)
            if sel.size >= nmin or h > 5:
                break
            h *= 1.5
        if sel.size < 5:
            return float("nan"), float("nan")
        v = self.val[sel]
        md = float(np.median(v))
        return md, 1.4826 * float(np.median(np.abs(v - md)))


def _robust_z_by_mag(val: np.ndarray, mag: np.ndarray, use: np.ndarray, half: float = 0.5, nmin: int = 15) -> np.ndarray:
    """(val - median) / (1.4826 MAD) of stars within +-half mag (widened until nmin)."""
    z = np.full(val.size, np.nan)
    win = _MagWindows(val, mag, use)  # G10-31: one sorted index for all stars
    for i in range(val.size):
        if not np.isfinite(val[i]):
            continue
        md, sc = win.stats(float(mag[i]), half, nmin)
        if math.isfinite(sc) and sc > 0:
            z[i] = (val[i] - md) / sc
    return z


def _skewnorm_quantile_fit(u: np.ndarray) -> tuple[float, float, float]:
    """Skew-normal (shape, loc, scale) whose 10/50/90 % quantiles equal those of ``u``."""
    from scipy import optimize, stats  # noqa: PLC0415

    qs = np.array([0.1, 0.5, 0.9])
    q = np.quantile(u, qs)

    def _r(p: np.ndarray) -> np.ndarray:
        return stats.skewnorm.ppf(qs, p[0], p[1], math.exp(p[2])) - q

    best = None
    for a0 in (-2.0, 0.0, 2.0, 5.0):
        sol = optimize.least_squares(_r, [a0, q[1], math.log(max((q[2] - q[0]) / 2.563, 1e-6))])
        if best is None or sol.cost < best.cost:
            best = sol
    assert best is not None
    return float(best.x[0]), float(best.x[1]), float(math.exp(best.x[2]))


def _var_frame_table(*, cfg, files, kplan, meta, per_frame_dir, phot, ps, rows, t_mid):
    """Night table of all stars (time order), magnitudes / errors and per-star gates."""
    from gaia_catalog_id import normalize_gaia_source_id  # noqa: PLC0415

    order = np.argsort(np.where(np.isfinite(t_mid), t_mid, np.arange(len(t_mid)) * 1e9), kind="stable")
    for t_idx, j in enumerate(order):
        rows[j]["t"] = t_idx
    D = pd.concat([rows[j] for j in order], ignore_index=True)
    D["catalog_id"] = D["catalog_id"].map(lambda v: str(normalize_gaia_source_id(v) or ""))
    D = D[D["catalog_id"] != ""].drop_duplicates(["catalog_id", "t"])
    nT = len(files)
    sat = coerce_bool_series(D["likely_saturated"]) if "likely_saturated" in D else pd.Series(False, index=D.index)
    pok = coerce_bool_series(D["photometry_ok"]) if "photometry_ok" in D else pd.Series(True, index=D.index)
    # C1-2026-10: each star on the aperture k of its brightness group (the k its light curve uses)
    from aperture_pertarget import group_k_fluxes  # noqa: PLC0415

    base = (~sat) & pok
    gk = group_k_fluxes(D, base, kplan)
    k_star, apc, k_ref = gk["k_star"], gk["apcorr"], kplan["k_ref"]
    D["k"] = gk["k_row"]
    F = gk["flux"]
    # COVERAGE-FAINT-2026-10 (Milan 2026-10-29): a measured frame with flux <= 0 is the star below the
    # frame's limit (a variable can be detectable at maximum and not at minimum), not a missing frame
    good = base & np.isfinite(F)
    D["apcorr"] = gk["apcorr_row"]
    D = D[good].copy()
    D["F"] = F[good]
    meta["aperture_k"] = {
        "rule": kplan["rule"], "k_ref": k_ref, "k_for_transients": kplan["k_transient"],
        "n_stars_by_k": {f"{kk:g}": int(sum(1 for v in k_star.values() if v == kk)) for kk in kplan["k_all"]},
        "apcorr_median_mag_by_k": {f"{kk:g}": round(float(np.nanmedian(ser)), 4) for kk, ser in apc.items()},
    }

    # frame quantities
    frame = D.groupby("t").agg(fwhm=("fwhm_px_for_aperture", "median"), stem=("stem", "first"))
    am = D.groupby("t")["airmass"].median() if "airmass" in D else pd.Series(np.nan, index=frame.index)
    if not np.isfinite(am).any():
        am = _airmass_from_lightcurves(phot, frame["stem"])
    frame["airmass"] = am.reindex(frame.index)
    # frame time (G02-18): the proc column of the FIELD CENTRE - used only for the order and spacing of
    # frames (1/eta, moving-object fits), where a star-to-star offset of < 1 min does not matter. Times
    # written for a star (varcand_*.csv, transients) are per star (star_times_bjd).
    tcol = "bjd_tdb_mid" if "bjd_tdb_mid" in D and np.isfinite(pd.to_numeric(D["bjd_tdb_mid"], errors="coerce")).any() else "jd_mid"
    frame["time"] = pd.to_numeric(D[tcol], errors="coerce").groupby(D["t"]).median()
    if "jd_mid" in D:
        frame["jd_utc"] = pd.to_numeric(D["jd_mid"], errors="coerce").groupby(D["t"]).median()
    meta["frame_time_column"] = tcol

    # CCD-equation error of every point (Merline & Howell 1995)
    # GAIN-ONE-2026-10: the gain / read noise of the run's aperture photometry; the annulus of the config
    # (both were config defaults / fixed 2.7-5.2 FWHM here before)
    from aperture_pertarget import load_photometric_gain  # noqa: PLC0415

    g, rn = load_photometric_gain(phot)
    if not (math.isfinite(g) and g > 0 and math.isfinite(rn) and rn >= 0):
        raise RuntimeError("variability: gain / read noise of aperture photometry missing (aperture_per_target.json)")
    a_in, a_out = float(cfg.annulus_inner_fwhm), float(cfg.annulus_outer_fwhm)
    meta.update(gain=g, read_noise=rn)
    fw_row = pd.to_numeric(D["fwhm_px_for_aperture"], errors="coerce")
    r = D["k"] * fw_row
    A = np.pi * r * r
    # CCD-ONE-2026-10 (G06-20): the one CCD equation (ccd_noise), with the real number of sky pixels of the
    # proc sky median (proc column sky_n_px; the former pi FWHM^2 (a_out^2 - a_in^2) ignored the source mask,
    # the footprint cut and the r_in >= r_ap + 0.5 rule) and the pi/2 variance of a median sky
    if "sky_n_px" not in D.columns:
        raise RuntimeError("variability: proc CSVs without sky_n_px (made before CCD-ONE-2026-10) - re-run the "
                           "per-frame photometry of this night")
    from ccd_noise import ccd_variance_adu2  # noqa: PLC0415

    var = pd.Series(
        ccd_variance_adu2(
            D["F"].to_numpy(float), pd.to_numeric(D["sky_adu_per_px_annulus"], errors="coerce").to_numpy(float),
            A.to_numpy(float), pd.to_numeric(D["sky_n_px"], errors="coerce").to_numpy(float), gain=g, read_noise=rn,
        ),
        index=D.index,
    )
    # magnitude on the k_ref scale (aperture correction; its own error - a median over the k_ref
    # group - is small against the point error and not propagated)
    # a point below 1 sigma takes the 1-sigma flux (its upper limit, S/N 1): a magnitude for the indices
    # (MAD scatter, 1/eta) that says "this faint or fainter". Gated stars (median S/N >= snr_min) reach it
    # only by real fading, not by noise.
    m_inst, D["e"], D["below_limit"] = limit_magnitudes(D["F"], var)
    D["m"] = m_inst + D["apcorr"]

    # aperture correction of each group k by frame (0 for k_ref): the transient search measures new
    # sources at k_transient and puts them on the same scale
    for kk in kplan["k_all"]:
        frame[f"apcorr_{kk:g}"] = apc[kk].reindex(frame.index).to_numpy(float) if kk in apc else 0.0
    cnt = D.groupby("catalog_id")["t"].nunique()
    keep_ids = cnt[cnt >= COVERAGE_MIN * nT].index
    D = D[D["catalog_id"].isin(keep_ids)]
    M = D.pivot(index="catalog_id", columns="t", values="m")
    S = D.pivot(index="catalog_id", columns="t", values="e")
    info_bl = D.pivot(index="catalog_id", columns="t", values="below_limit")  # G10-30: per-point fainter-than flag
    info = D.groupby("catalog_id")[["x", "y", "ra_deg", "dec_deg", "mag", "bp_rp", "k"]].first()
    known = coerce_bool_series(D.groupby("catalog_id")["vsx_known_variable"].first()) if "vsx_known_variable" in D else pd.Series(False, index=info.index)
    targets = _target_ids(phot)
    info["known_variable"] = known.reindex(info.index).fillna(False) | info.index.isin(targets)
    info["target"] = info.index.isin(targets)
    info["n_frames"] = cnt.reindex(info.index)
    info["n_below_limit"] = D.groupby("catalog_id")["below_limit"].sum().reindex(info.index).fillna(0).astype(int)
    info["e_med"] = S.median(axis=1)
    info["snr_med"] = 1.0857 / info["e_med"]

    # gates: SNR, footprint, neighbour light
    snr_min = float(cfg.snr_min)
    info["gate_snr"] = info["snr_med"] >= snr_min
    fw_max = float(frame["fwhm"].max())
    quads = _frame_footprints(per_frame_dir, list(frame["stem"]))
    r_out_max = a_out * fw_max
    if quads:
        info["gate_edge"] = _inside_all(info["x"].to_numpy(float), info["y"].to_numpy(float), r_out_max, quads)
    else:
        info["gate_edge"] = True
        meta["edge_note"] = "no alignment matrices - frame footprints not checked"
    try:
        from comp_blend import BLEND_FLUX_FRAC_MAX, blend_contamination_for_pool, blend_fwhm_px  # noqa: PLC0415
        from gaia_bands import filter_band_for_obs_group  # noqa: PLC0415

        ms = None
        msc = ps / "masterstars_full_match.csv"
        if msc.is_file():
            ms = pd.read_csv(msc, low_memory=False, dtype={"catalog_id": str})
        pool = info.reset_index().rename(columns={"index": "catalog_id"})
        fw_med = float(frame["fwhm"].median())
        # BLEND-SEEING-2026-10: neighbour light at the night's bad-seeing FWHM (p90 of the frames)
        fw_b, fw_b_src = blend_fwhm_px(frame["fwhm"].tolist(), fw_med)
        # neighbour light in each star's own aperture (one evaluation per group k, one PSF)
        c = pd.Series(np.nan, index=pool.index)
        _psf = None
        for kk in sorted(pool["k"].dropna().unique()):
            ck, bmeta = blend_contamination_for_pool(
                pool, field_catalog_csv=ps / "field_catalog_cone.csv", masterstar_fits=ps / "MASTERSTAR.fits",
                zp_band=filter_band_for_obs_group(per_frame_dir.name).blend_band, fwhm_px=fw_b,
                profile_fwhm_px=fw_med, r_ap_px=float(kk) * fw_b,
                sat_adu=float(pd.to_numeric(ms["saturate_limit_adu"], errors="coerce").median()) if ms is not None and "saturate_limit_adu" in ms else None,
                annulus_px=(a_in * fw_b, a_out * fw_b), profile_df=ms, psf=_psf,
            )
            if bmeta.get("status") != "ok":
                break
            _psf = bmeta.get("psf")
            sel = pool["k"] == kk
            c[sel] = ck[sel]
        bmeta["fwhm_source"] = fw_b_src
        info["blend"] = c.to_numpy()
        # |c|: extra light in the aperture or a sky pushed up by a neighbour (c < 0)
        # are both errors; unknown neighbour light (NaN) does not pass
        info["gate_blend"] = info["blend"].abs() <= BLEND_FLUX_FRAC_MAX
        meta["blend"] = {k2: v for k2, v in bmeta.items() if k2 != "psf"} | {"psf": bmeta.get("psf")}
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[VARIABILITY] blend gate unavailable: %s", exc)
        info["blend"] = np.nan
        info["gate_blend"] = False
        meta["blend"] = {"status": f"failed: {exc}"}
    return M, S, info_bl, frame, g, info, rn


def _var_frame_zero_point(*, M, S, info, meta):
    """Common frame zero point from reference stars (Honeycutt 1992) and the residual matrix."""
    info["gate_catalog"] = np.isfinite(pd.to_numeric(info["mag"], errors="coerce"))
    info["gates_ok"] = info["gate_snr"] & info["gate_edge"] & info["gate_blend"] & info["gate_catalog"]

    # common frame zero point (Honeycutt 1992) from reference stars
    ref = info.index[info["gates_ok"] & ~info["known_variable"] & (info["e_med"] < REF_ERR_MAX)]
    if len(ref) < 20:
        ref = info[info["gates_ok"] & ~info["known_variable"]].sort_values("e_med").index[:50]
    Mi = M.loc[ref].median(axis=1)
    E = pd.Series(0.0, index=M.columns)
    for _ in range(10):
        E_new = M.loc[ref].sub(Mi, axis=0).median(axis=0)
        Mi = M.loc[ref].sub(E_new, axis=1).median(axis=1)
        if float(np.nanmax(np.abs(E_new - E))) < 1e-5:
            E = E_new
            break
        E = E_new
    R = M.sub(E, axis=1)
    R = R.sub(R.median(axis=1), axis=0)
    # error of the frame zero point E, a MEDIAN over the reference stars: sqrt(pi/2) sigma / sqrt(n) (G10-29)
    sigE = math.sqrt(math.pi / 2.0) * 1.4826 * R.loc[ref].abs().median(axis=0) / math.sqrt(max(len(ref), 1))
    meta["n_ref"] = int(len(ref))

    info["s"] = 1.4826 * R.abs().median(axis=1)
    info["e"] = np.sqrt(S.pow(2).add(sigE.pow(2), axis=1)).median(axis=1)
    info["eta"] = [_eta(R.loc[c].to_numpy(float)) for c in info.index]
    q = R.quantile([0.05, 0.95], axis=1).T
    info["amplitude_mag"] = (q[0.95] - q[0.05]).reindex(info.index)

    fitset = info[info["gates_ok"] & ~info["known_variable"] & np.isfinite(info["s"]) & (info["s"] > 0) & np.isfinite(info["e"])]

    # NOISE-FIT-ONE-2026-10: the one robust log fit, shared with the comparison-star ceiling
    a, fl, _ok = fit_scatter_noise_model(fitset["s"].to_numpy(dtype=float), fitset["e"].to_numpy(dtype=float))
    info["s_exp"] = np.sqrt((a * info["e"]) ** 2 + fl**2)
    meta["noise_model"] = {"a": a, "floor_mag": fl, "n_fit": int(len(fitset))}

    use = (info["gates_ok"] & ~info["known_variable"]).to_numpy()
    return E, R, a, fl, ref, use


def scatter_index_center_scale(lq: np.ndarray) -> tuple[float, float]:
    """Centre and robust width of the log scatter index ln(s / s_exp) of the reference stars:
    (median, 1.4826 x MAD), NaN ignored. G10-48: shared by the z_s index and the plotted candidate limit."""
    lq = np.asarray(lq, dtype=np.float64)
    md = float(np.nanmedian(lq))
    return md, 1.4826 * float(np.nanmedian(np.abs(lq - md)))


def _var_indices(*, E, M, R, cfg, frame, info, meta, ref, use):
    """Variability indices (scatter z, von Neumann eta) by magnitude, transients, zero-point check."""
    from scipy import stats  # noqa: PLC0415

    lq = np.log(info["s"] / info["s_exp"]).to_numpy(float)
    md, sc = scatter_index_center_scale(lq[use])
    info["z_s"] = (lq - md) / sc if sc > 0 else np.nan
    # small eta = smooth = variable. u = ln(1/eta) standardised against stars of similar
    # brightness (median, 1.4826 MAD). Its field distribution is skewed (white noise:
    # ~20-60x too many Gaussian 4-sigma events) and on real nights red noise sets its
    # shape, so the tail is taken from a skew-normal fitted to the field itself
    # (Azzalini 1985) and converted to an equivalent Gaussian z. The skew-normal is
    # matched to the 10/50/90 % quantiles of the field (not maximum likelihood), so the
    # few real variables of a field cannot stretch its tail.
    li = np.log(1.0 / info["eta"].clip(lower=1e-9)).to_numpy(float)
    mags = info["mag"].to_numpy(float)
    u = _robust_z_by_mag(li, mags, use)
    uf = u[use & np.isfinite(u)]
    sk = _skewnorm_quantile_fit(uf) if uf.size >= 50 else (0.0, 0.0, 1.0)
    meta["eta_null_skewnorm"] = {"shape": float(sk[0]), "loc": float(sk[1]), "scale": float(sk[2]), "n": int(uf.size)}

    def _z_from_u(x: np.ndarray) -> np.ndarray:
        lsf = stats.skewnorm.logsf(x, *sk)
        return -stats.norm.ppf(np.clip(np.exp(lsf), 1e-300, 1.0 - 1e-16))

    info["z_eta"] = _z_from_u(u)
    n_test = int(use.sum())
    zthr = float(stats.norm.isf(0.5 * FAP_FIELD / max(n_test, 1)))
    meta.update(n_stars=int(len(info)), n_gates_ok=n_test, z_threshold=zthr, fap_field=FAP_FIELD)
    info["signal"] = (info["z_s"] > zthr) | (info["z_eta"] > zthr)

    # systematics (G10-08, SYS-FIELD-RESPONSE-2026-10, after Kepler PDC-MAP, Smith et al. 2012): the FIELD's typical
    # response to frame FWHM and airmass is removed, not the star's own fit (a per-star OLS on [1, FWHM, airmass]
    # absorbed 80-88 % of a slow sinusoid on Brno 534 g, where both run monotonically through the night)
    fw_f = frame["fwhm"].to_numpy(float)
    am_f = np.nan_to_num(frame["airmass"].to_numpy(float), nan=float(np.nanmedian(frame["airmass"])) if np.isfinite(frame["airmass"]).any() else 1.0)
    X = np.c_[np.ones(len(frame)), fw_f, am_f]

    def _zs_ze(cid: str, rr: np.ndarray) -> tuple[float, float]:
        ok = np.isfinite(rr)
        s2 = 1.4826 * float(np.nanmedian(np.abs(rr[ok] - np.nanmedian(rr[ok]))))
        zs2 = (math.log(max(s2, 1e-9) / float(info.at[cid, "s_exp"])) - md) / sc if sc > 0 else np.nan
        e2 = _eta(rr)
        m2, sc2 = _li_windows.stats(float(info.at[cid, "mag"]))
        ze2 = (float(_z_from_u(np.array([(math.log(1.0 / max(e2, 1e-9)) - m2) / sc2]))[0])
               if (math.isfinite(e2) and math.isfinite(sc2) and sc2 > 0) else np.nan)
        return zs2, ze2

    _li_windows = _MagWindows(li, mags, use)
    resp = field_response(R.reindex(columns=frame.index), X, info, use & ~info["signal"].to_numpy(bool), meta)
    info["seeing_response_z"] = resp["seeing_response_z"]
    info["systematic"] = False
    info["trend_like"] = False
    info["fwhm_coef_diff"] = np.nan
    n_fw_fallback = 0
    for cid in info.index[info["signal"] & info["gates_ok"]]:
        y = R.loc[cid].reindex(frame.index).to_numpy(float)
        ok = np.isfinite(y)
        if ok.sum() < 10:
            continue
        # decision (SYS-SEEING-DIFF-2026-10): the star's signal vanishes once its OWN seeing response, measured from
        # frame-to-frame changes (first differences: a slow intrinsic change does not enter it), and the field's
        # colour-extinction response (airmass is smooth, so it has no frame-to-frame information) are removed
        b_fw = seeing_coef_from_differences(y, fw_f)
        info.at[cid, "fwhm_coef_diff"] = b_fw
        if not math.isfinite(b_fw):
            b_fw = float(resp["fwhm_coef"][cid])
            n_fw_fallback += 1
        pred = b_fw * (fw_f - fw_f.mean()) + resp["airmass_coef"][cid] * (am_f - am_f.mean())
        zs2, ze2 = _zs_ze(cid, y - pred)
        info.at[cid, "systematic"] = bool(not (zs2 > zthr or ze2 > zthr))
        # flag only (never removes a star): the star's own OLS on [1, FWHM, airmass] removes its signal
        beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        zs3, ze3 = _zs_ze(cid, y - X @ beta)
        info.at[cid, "trend_like"] = bool(not (zs3 > zthr or ze3 > zthr))

    meta["systematics"]["seeing_coef"] = ("own, from first differences (SYS-SEEING-DIFF-2026-10); field median when "
                                          "the frame FWHM has no frame-to-frame variation")
    meta["systematics"]["n_seeing_coef_field_fallback"] = int(n_fw_fallback)
    if n_fw_fallback:
        LOGGER.warning("[VARIABILITY] %d signal stars: no frame-to-frame FWHM variation - field seeing response used",
                       n_fw_fallback)
    info["candidate"] = info["signal"] & info["gates_ok"] & ~info["known_variable"] & ~info["systematic"]
    transients = bool(cfg.transient_search_enabled)
    zp_g = float("nan")
    if transients:
        zp_g = _brightened_vs_gaia(info, M.sub(E, axis=1).median(axis=1), use, ref)
        meta["brightened"] = {"n": int(info["brightened"].sum()), "zp_g_solar": zp_g}
    info["known_detected"] = info["signal"] & info["gates_ok"] & info["known_variable"]
    info["index"] = np.where(
        (info["z_s"] > zthr) & (info["z_eta"] > zthr), "scatter+1/eta",
        np.where(info["z_s"] > zthr, "scatter", np.where(info["z_eta"] > zthr, "1/eta", "")),
    )
    info.index.name = "catalog_id"
    out = info.reset_index()
    order_c = out[out["candidate"]].sort_values("z_s", ascending=False).index
    out["label"] = ""
    out.loc[order_c, "label"] = [f"V{i + 1}" for i in range(len(order_c))]
    out["catalogs"] = ""
    return n_test, order_c, out, transients, zp_g, zthr


#: SYS-SEEING-DIFF-2026-10: fewest consecutive frame pairs for a star's own seeing coefficient
SEEING_DIFF_MIN_PAIRS = 10


def seeing_coef_from_differences(y: np.ndarray, fwhm: np.ndarray) -> float:
    """A star's own response to the frame FWHM [mag / px] from first differences of consecutive frames (time order):
    sum(dy dF) / sum(dF^2) over pairs where both frames are finite. Regressing differences instead of levels removes
    any slow common trend - a slow intrinsic variation or a monotonic FWHM run through the night cannot alias into the
    coefficient (spurious regression of trending series, Granger & Newbold 1974, J. Econometrics 2, 111). NaN when
    fewer than SEEING_DIFF_MIN_PAIRS pairs or no FWHM change between frames."""
    y = np.asarray(y, dtype=float)
    f = np.asarray(fwhm, dtype=float)
    ok = np.isfinite(y[1:]) & np.isfinite(y[:-1]) & np.isfinite(f[1:]) & np.isfinite(f[:-1])
    if int(ok.sum()) < SEEING_DIFF_MIN_PAIRS:
        return float("nan")
    dy = (y[1:] - y[:-1])[ok]
    df = (f[1:] - f[:-1])[ok]
    den = float(np.sum(df * df))
    return float(np.sum(dy * df) / den) if den > 0 else float("nan")


#: G10-08: bootstrap resamples of the field-response coefficients (their errors in variability_meta.json; the
#: relative error of a bootstrap standard deviation is ~1/sqrt(2 (B - 1)) = 7 %)
FIELD_RESPONSE_BOOTSTRAP = 100


def field_response(R: pd.DataFrame, X: np.ndarray, info: pd.DataFrame, field: np.ndarray, meta: dict[str, Any]) -> dict[str, Any]:
    """G10-08 (SYS-FIELD-RESPONSE-2026-10): the field's typical linear response to frame FWHM and airmass.

    Every star with >= 10 frames gets its own OLS coefficients on ``X`` = [1, FWHM, airmass]. The field set
    (``field``: gated, not known variables, no signal - variables would bias the prior) gives, for each star,
    the FWHM coefficient = median over field stars within +-0.5 mag (widened to >= 15 stars), and the airmass
    coefficient = a Theil-Sen line in BP-RP (colour extinction) plus the median residual from that line in the same
    magnitude window. A star without BP-RP takes the field's median colour (counted in meta). Returns per-star
    ``fwhm_coef`` / ``airmass_coef`` (the field prediction) and ``seeing_response_z`` = (own FWHM coefficient - field) /
    field scale (1.4826 MAD); records the field coefficients with bootstrap errors in ``meta["systematics"]``.
    """
    from scipy import stats  # noqa: PLC0415

    ids = list(info.index)
    B = np.full((len(ids), 3), np.nan)
    for i, cid in enumerate(ids):
        y = R.loc[cid].to_numpy(float) if cid in R.index else np.full(len(X), np.nan)
        ok = np.isfinite(y)
        if ok.sum() >= 10:
            B[i] = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
    mags = pd.to_numeric(info["mag"], errors="coerce").to_numpy(float)
    col = pd.to_numeric(info["bp_rp"], errors="coerce").to_numpy(float)
    fset = np.asarray(field, dtype=bool) & np.isfinite(B).all(axis=1)
    col_med = float(np.nanmedian(col[fset])) if np.isfinite(col[fset]).any() else 0.0
    col_use = np.where(np.isfinite(col), col, col_med)
    fit_c = fset & np.isfinite(col)
    if int(fit_c.sum()) < 15:
        raise RuntimeError(f"variability: only {int(fit_c.sum())} gated non-signal stars with BP-RP - no field response")
    ts = stats.theilslopes(B[fit_c, 2], col[fit_c])
    ba_res = B[:, 2] - (ts.intercept + ts.slope * col_use)
    win_f = _MagWindows(B[:, 1], mags, fset)
    win_a = _MagWindows(ba_res, mags, fset)
    bf_pred = np.full(len(ids), np.nan)
    ba_pred = np.full(len(ids), np.nan)
    srz = np.full(len(ids), np.nan)
    for i in range(len(ids)):
        mf, sf = win_f.stats(float(mags[i]))
        ma, _ = win_a.stats(float(mags[i]))
        bf_pred[i] = mf
        ba_pred[i] = ts.intercept + ts.slope * col_use[i] + ma
        if math.isfinite(sf) and sf > 0 and math.isfinite(B[i, 1]):
            srz[i] = (B[i, 1] - mf) / sf
    # bootstrap over the field stars (fixed seed: reproducible meta)
    rng = np.random.default_rng(0)
    idx_c, idx_f = np.flatnonzero(fit_c), np.flatnonzero(fset)
    bs_slope, bs_int, bs_bf = [], [], []
    for _ in range(FIELD_RESPONSE_BOOTSTRAP):
        k = rng.choice(idx_c, idx_c.size, replace=True)
        t = stats.theilslopes(B[k, 2], col[k])
        bs_slope.append(t.slope)
        bs_int.append(t.intercept)
        bs_bf.append(float(np.median(B[rng.choice(idx_f, idx_f.size, replace=True), 1])))
    meta["systematics"] = {
        "method": "field response (G10-08 R2+F, after Kepler PDC-MAP): systematic = signal vanishes after removing the "
                  "field's typical FWHM / airmass response; trend_like and seeing_response_z are flags only",
        "n_field": int(fset.sum()),
        "n_field_with_colour": int(fit_c.sum()),
        "n_stars_colour_missing": int(np.sum(~np.isfinite(col))),
        "colour_for_missing_bp_rp": round(col_med, 4),
        "fwhm_coef_field_median_mag_per_px": round(float(np.median(B[fset, 1])), 6),
        "fwhm_coef_field_median_err_bootstrap": round(float(np.std(bs_bf)), 6),
        "airmass_coef_vs_bp_rp_theil_sen": {
            "slope_mag_per_airmass_per_mag": round(float(ts.slope), 6),
            "slope_err_bootstrap": round(float(np.std(bs_slope)), 6),
            "intercept_mag_per_airmass": round(float(ts.intercept), 6),
            "intercept_err_bootstrap": round(float(np.std(bs_int)), 6),
        },
        "n_bootstrap": FIELD_RESPONSE_BOOTSTRAP,
        "magnitude_window": "+-0.5 mag, widened x1.5 to >= 15 field stars",
    }
    return {
        "fwhm_coef": pd.Series(bf_pred, index=ids),
        "airmass_coef": pd.Series(ba_pred, index=ids),
        "seeing_response_z": srz,
    }


def run_field_variability(
    per_frame_dir: Path,
    photometry_dir: Path,
    cfg: Any,
    *,
    platesolve_dir: Path | None = None,
    check_catalogs: bool = True,
) -> dict[str, Any]:
    """Run the search; writes variability_field.csv (candidates: ``candidate``, ``label``, ``catalogs``),
    variability_meta.json and lightcurves/varcand_<id>.csv. Returns the meta dict."""

    from photometry_defs import DEFAULT_APERTURE_F_GRID  # noqa: PLC0415
    from photometry_shared import flux_k_column  # noqa: PLC0415

    per_frame_dir = Path(per_frame_dir)
    phot = Path(photometry_dir)
    ps = Path(platesolve_dir) if platesolve_dir else phot.parent
    # G10-06: the products of a previous search go first - a rerun that stops early, or runs with the
    # transient search off, leaves no stale candidates behind (TRANSIENT-2026-10: "off: no file")
    from run_outputs import remove_files  # noqa: PLC0415
    from transient_field import BODIES_CSV, META_JSON, MOVING_CSV, TRANSIENT_CSV  # noqa: PLC0415

    remove_files(phot, (FIELD_CSV, TRANSIENT_CSV, MOVING_CSV, BODIES_CSV, META_JSON))
    remove_files(phot / "lightcurves", ("varcand_*.csv", "transient_*.csv"))
    files = sorted(per_frame_dir.glob("proc_*.csv"))
    meta: dict[str, Any] = {"method": "VAR-2026-10", "n_frames": len(files)}
    if len(files) < 10:
        meta["status"] = "too few frames"
        _write_meta(phot, meta)
        return meta
    from aperture_pertarget import field_k_plan  # noqa: PLC0415

    kplan = field_k_plan(phot, DEFAULT_APERTURE_F_GRID)
    if kplan is None:
        meta["status"] = "aperture k unknown - aperture_per_target.json missing (aperture photometry did not run)"
        _write_meta(phot, meta)
        LOGGER.warning("[VARIABILITY] search not run: %s", meta["status"])
        return meta
    kcols = {flux_k_column(kk) for kk in kplan["k_all"]}
    rows, t_mid = [], []
    for i, f in enumerate(files):
        from proc_frame_store import read_proc_csv  # noqa: PLC0415

        d = read_proc_csv(f, usecols=lambda c: c in set(_COLS) | kcols, low_memory=False, dtype={"catalog_id": str})
        if not kcols <= set(d.columns):
            meta["status"] = f"proc catalogues lack {sorted(kcols - set(d.columns))}"
            _write_meta(phot, meta)
            return meta
        d["stem"] = f.stem[len("proc_"):]
        t_mid.append(float(pd.to_numeric(d["jd_mid"], errors="coerce").median()) if "jd_mid" in d else float(i))
        rows.append(d)
    # 1/eta needs the frames in time order (file names are not guaranteed to be)
    M, S, BL, frame, g, info, rn = _var_frame_table(
        cfg=cfg,
        files=files,
        kplan=kplan,
        meta=meta,
        per_frame_dir=per_frame_dir,
        phot=phot,
        ps=ps,
        rows=rows,
        t_mid=t_mid,
    )
    if str((meta.get("blend") or {}).get("status", "")) != "ok":
        meta["status"] = "neighbour light unknown - " + str((meta.get("blend") or {}).get("status"))
        _write_meta(phot, meta)
        LOGGER.warning("[VARIABILITY] search not run: %s", meta["status"])
        return meta
    E, R, a, fl, ref, use = _var_frame_zero_point(M=M, S=S, info=info, meta=meta)
    if int(use.sum()) < 50:
        meta.update(status=f"only {int(use.sum())} stars pass the gates (< 50) - no noise model", n_stars=int(len(info)),
                    n_gates_ok=int(use.sum()))
        _write_meta(phot, meta)
        return meta
    n_test, order_c, out, transients, zp_g, zthr = _var_indices(
        E=E,
        M=M,
        R=R,
        cfg=cfg,
        frame=frame,
        info=info,
        meta=meta,
        ref=ref,
        use=use,
    )

    cand = out.loc[order_c].reset_index(drop=True)
    lc_dir = phot / "lightcurves"
    lc_dir.mkdir(parents=True, exist_ok=True)
    lc_ids = list(cand["catalog_id"]) + list(out.loc[out["known_detected"], "catalog_id"])
    from time_utils import run_site  # noqa: PLC0415

    site = run_site(phot)
    # G02-18 / C6: the published time of a star is that star's own BJD_TDB
    t_star, meta["time_column"] = star_times_bjd(frame, info.loc[lc_ids, "ra_deg"], info.loc[lc_ids, "dec_deg"], site)
    write_candidate_lightcurves(lc_dir, lc_ids, R=R, S=S, BL=BL, frame=frame, t_star=t_star, a=a, fl=fl)
    if check_catalogs and not cand.empty:
        out.loc[order_c, "catalogs"] = [_catalog_check(row, cfg) for _, row in cand.iterrows()]
    out.to_csv(phot / FIELD_CSV, index=False)
    if transients:
        try:
            from transient_field import run_field_transients  # noqa: PLC0415

            tm = run_field_transients(per_frame_dir=per_frame_dir, photometry_dir=phot, platesolve_dir=ps, cfg=cfg,
                                      frame=frame, E=E.sub(frame[f"apcorr_{kplan['k_transient']:g}"]), zp_g=zp_g,
                                      k=kplan["k_transient"], gain=g, read_noise=rn,
                                      check_catalogs=check_catalogs, site=site)
            meta["transients"] = {kk: tm.get(kk) for kk in ("status", "n_sources", "n_candidates", "n_minor_bodies")}
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("[TRANSIENT] search failed: %s", exc)
            meta["transients"] = {"status": f"failed: {exc}"}
    meta.update(n_candidates=int(len(cand)), n_known_detected=int(out["known_detected"].sum()),
                n_systematic=int((out["systematic"] & out["gates_ok"]).sum()),
                n_rejected_gates=int((~out["gates_ok"]).sum()), status="ok")
    _write_meta(phot, meta)
    LOGGER.info("[VARIABILITY] %d stars, %d pass gates, noise a=%.2f floor=%.4f, z>%.2f: %d candidates, %d known detected, %d systematic",
                len(out), n_test, a, fl, zthr, len(cand), meta["n_known_detected"], meta["n_systematic"])
    return meta


def write_candidate_lightcurves(
    lc_dir: Path, lc_ids: list[str], *, R: pd.DataFrame, S: pd.DataFrame, BL: pd.DataFrame, frame: pd.DataFrame,
    t_star: np.ndarray, a: float, fl: float,
) -> None:
    """``varcand_<id>.csv`` of each star: its own time (``t_star`` row), residual magnitude, the field noise model's
    error ((a e)^2 + floor^2), ``below_limit`` (G10-30: the point is the star's 1-sigma flux limit,
    COVERAGE-FAINT-2026-10 - a fainter-than value, not a measurement), frame FWHM and airmass."""
    for j, cid in enumerate(lc_ids):
        rr = R.loc[cid].reindex(frame.index)
        ee = S.loc[cid].reindex(frame.index)
        em = np.sqrt((a * ee) ** 2 + fl**2)
        bl = BL.loc[cid].reindex(frame.index).astype("boolean").fillna(False).astype(bool)
        pd.DataFrame({"time": t_star[j], "dmag": rr.to_numpy(), "err": em.to_numpy(), "below_limit": bl.to_numpy(),
                      "fwhm": frame["fwhm"].to_numpy(), "airmass": frame["airmass"].to_numpy()}).to_csv(
            Path(lc_dir) / f"varcand_{cid}.csv", index=False)


def _brightened_vs_gaia(info: pd.DataFrame, Mi: pd.Series, use: np.ndarray, ref: Any) -> float:
    """TRANSIENT-2026-10: ``dmag_vs_gaia`` (night magnitude - expected from Gaia incl. the neighbours'
    light in the aperture; negative = brighter) and ``brightened``; returns the field zero point for a
    star of solar colour (SOLAR_BP_RP): G = m_inst - E - zp."""
    from mag_constants import SOLAR_BP_RP  # noqa: PLC0415
    from scipy import optimize  # noqa: PLC0415

    mi = Mi.reindex(info.index).to_numpy(float)
    G = pd.to_numeric(info["mag"], errors="coerce").to_numpy(float)
    c = pd.to_numeric(info["bp_rp"], errors="coerce").to_numpy(float)
    bl = pd.to_numeric(info["blend"], errors="coerce").to_numpy(float)
    fit = info.index.isin(ref) & np.isfinite(mi) & np.isfinite(G) & np.isfinite(c)
    if fit.sum() < 10:
        info["dmag_vs_gaia"] = np.nan
        info["brightened"] = False
        return float("nan")

    def _r(p: np.ndarray) -> np.ndarray:
        return mi[fit] - G[fit] - (p[0] + p[1] * c[fit] + p[2] * c[fit] ** 2)

    p = optimize.least_squares(_r, [float(np.median(mi[fit] - G[fit])), 0.0, 0.0], loss="soft_l1", f_scale=0.05).x
    cc = np.where(np.isfinite(c), c, SOLAR_BP_RP)
    expect = G + p[0] + p[1] * cc + p[2] * cc**2 - 2.5 * np.log10(np.clip(1.0 + np.nan_to_num(bl, nan=0.0), 1e-3, None))
    dm = mi - expect
    info["dmag_vs_gaia"] = dm
    ok = use & np.isfinite(dm)
    _dm_windows = _MagWindows(dm, G, ok)
    sig = np.array([_dm_windows.stats(float(G[i]))[1] for i in range(len(dm))])
    gate = (info["gate_snr"] & info["gate_edge"]).to_numpy(bool)
    info["brightened"] = gate & np.isfinite(dm) & (dm <= -BRIGHTENED_DMAG) & (dm <= -BRIGHTENED_NSIGMA * sig)
    return float(p[0] + p[1] * SOLAR_BP_RP + p[2] * SOLAR_BP_RP**2)


def _target_ids(phot: Path) -> set[str]:
    from gaia_catalog_id import normalize_gaia_source_id  # noqa: PLC0415

    p = phot / "active_targets.csv"
    if not p.is_file():
        return set()
    try:
        at = pd.read_csv(p, usecols=["catalog_id"], dtype={"catalog_id": str})
    except Exception:  # noqa: BLE001
        return set()
    return {str(normalize_gaia_source_id(v) or "") for v in at["catalog_id"]} - {""}


def _airmass_from_lightcurves(phot: Path, stems: pd.Series) -> pd.Series:
    """Airmass per frame from any written light curve (source_file -> airmass)."""
    lcd = phot / "lightcurves"
    mp: dict[str, float] = {}
    for lc in sorted(lcd.glob("lightcurve_*.csv")):
        try:
            d = pd.read_csv(lc, usecols=["source_file", "airmass"])
        except Exception:  # noqa: BLE001
            continue
        keys = d["source_file"].astype(str).str.replace(".csv", "", regex=False).str.replace("proc_", "", regex=False)
        for kk, vv in zip(keys, pd.to_numeric(d["airmass"], errors="coerce"), strict=False):
            if kk not in mp and math.isfinite(float(vv)):
                mp[kk] = float(vv)
    return stems.map(lambda s: mp.get(str(s), np.nan)).astype(float)


def _write_meta(phot: Path, meta: dict[str, Any]) -> None:
    from utils import write_json_finite  # noqa: PLC0415

    write_json_finite(Path(phot) / "variability_meta.json", meta)
