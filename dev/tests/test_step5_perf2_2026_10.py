"""Step 5 batch B part 2 (perf2, 2026-10): equality tests old vs new implementation (byte-identical outputs).

Each section holds a reference copy of the code on HEAD 08e985a (``_old_*``) and compares it with the
refactored code on realistic synthetic input, bit for bit.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_py"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import photometry_core  # noqa: E402,F401  (import order)
from comp_pool_rms import _norm_id_val, comp_frame_stats  # noqa: E402
from comp_rms_loo import (  # noqa: E402
    _loo_median,
    choose_field_aperture_k,
    compute_loo_mag_rms_map,
    loo_mag_rms_maps,
    mad_sigma_scale_or_zero,
)
from photometry_defs import DEFAULT_APERTURE_F_GRID, flux_k_column  # noqa: E402
from pipeline_constants import LINEARITY_FRAC  # noqa: E402
from proc_frame_store import ProcFrameStore  # noqa: E402
from stats_core import _flux_to_mag  # noqa: E402


# ----------------------------------------------------------------------------------------------------------
# G07-31: reference copies of the multi-pass code (HEAD 08e985a)
# ----------------------------------------------------------------------------------------------------------
def _old_sort(paths, cache):
    def _key(p):
        bjd = float("inf")
        df = cache.get(str(p))
        if df is not None and not getattr(df, "empty", True) and "bjd_tdb_mid" in df.columns:
            med = float(pd.to_numeric(df["bjd_tdb_mid"], errors="coerce").median())
            if math.isfinite(med):
                bjd = med
        return (bjd, str(p).casefold())

    return sorted(paths, key=_key)


def _old_loo(cand_ids, paths, cache, *, flux_col="dao_flux", min_frames_frac=0.3, strict_flux_col=False):
    if not cand_ids:
        return {}
    norm_ids = sorted({_norm_id_val(x) for x in cand_ids if _norm_id_val(x)})
    if len(norm_ids) < 2:
        return {}
    id_set = set(norm_ids)
    per_frame = []
    for csv_path in _old_sort(paths, cache):
        df = cache.get(str(csv_path))
        if df is None or df.empty:
            continue
        if strict_flux_col and flux_col not in df.columns:
            continue
        actual_flux = flux_col if flux_col in df.columns else "dao_flux"
        if actual_flux not in df.columns:
            continue
        id_col = "catalog_id" if "catalog_id" in df.columns else ("name" if "name" in df.columns else None)
        if id_col is None:
            continue
        work = df[[id_col, actual_flux]].copy()
        work["_nid"] = work[id_col].map(_norm_id_val)
        work["_flux"] = pd.to_numeric(work[actual_flux], errors="coerce")
        work = work[work["_nid"].isin(id_set) & work["_flux"].gt(0)]
        if work.empty:
            continue
        flux_by = work.groupby("_nid", sort=True)["_flux"].median()
        if len(flux_by) < 2:
            continue
        mags = np.asarray([_flux_to_mag(float(f)) for f in flux_by.to_numpy()], dtype=np.float64)
        okm = np.isfinite(mags)
        if int(okm.sum()) < 2:
            continue
        per_frame.append(([c for c, o in zip(flux_by.index, okm, strict=False) if o], mags[okm]))
    n_frames = len(per_frame)
    series = {cid: [] for cid in norm_ids}
    for cids_f, mags in per_frame:
        for cid, m in zip(cids_f, mags, strict=False):
            series[cid].append(float(m))
    night_level = {cid: float(np.median(v)) for cid, v in series.items() if v}
    dmag_map = {cid: [] for cid in norm_ids}
    for cids_f, mags in per_frame:
        offs = mags - np.asarray([night_level[c] for c in cids_f], dtype=np.float64)
        loo = _loo_median(offs)
        for cid, off_s, off_c in zip(cids_f, offs, loo, strict=False):
            if math.isfinite(off_c):
                dmag_map[cid].append(float(off_s - off_c))
    min_frames = max(3, int(n_frames * float(min_frames_frac)))
    out = {}
    for cid, vals in dmag_map.items():
        if len(vals) < min_frames:
            continue
        rms = mad_sigma_scale_or_zero(np.asarray(vals, dtype=np.float64))
        if math.isfinite(rms):
            out[cid] = rms
    return out


def _old_choose_k(cand_ids, paths, cache, *, min_frames_frac=0.3):
    have = None
    for csv_path in paths:
        df = cache.get(str(csv_path))
        if df is None or df.empty:
            continue
        ks = {k for k in DEFAULT_APERTURE_F_GRID if flux_k_column(k) in df.columns}
        have = ks if have is None else (have & ks)
    if not have:
        return None, {}, {}
    maps = {}
    for k in sorted(have):
        maps[k] = _old_loo(cand_ids, paths, cache, flux_col=flux_k_column(k), min_frames_frac=min_frames_frac,
                           strict_flux_col=True)
    common = set.intersection(*(set(m) for m in maps.values())) if maps else set()
    if len(common) < 3:
        return None, {}, {}
    scores = {k: float(np.median([maps[k][c] for c in common])) for k in maps}
    best = min(scores.values())
    k_best = max(k for k, v in scores.items() if v <= best * 1.01)
    return float(k_best), maps[k_best], scores


def _old_frame_stats(cand_ids, paths, cache, *, comp_peak_sat_frac, flux_col="dao_flux"):
    out = {str(c): {"n_frames": 0, "n_peak": 0, "n_over": 0} for c in cand_ids}
    if not cand_ids:
        return out
    for csv_path in _old_sort(paths, cache):
        df = cache.get(str(csv_path))
        if df is None or df.empty:
            continue
        name_col = "catalog_id" if "catalog_id" in df.columns else ("name" if "name" in df.columns else None)
        if name_col is None:
            continue
        sub = df[df[name_col].astype(str).isin(cand_ids)]
        if sub.empty:
            continue
        ids = sub[name_col].astype(str)
        fcol = flux_col if flux_col in sub.columns else ("dao_flux" if "dao_flux" in sub.columns else None)
        if fcol is not None:
            ok_f = pd.to_numeric(sub[fcol], errors="coerce").gt(0)
            for cid, n in ids[ok_f].value_counts().items():
                out[str(cid)]["n_frames"] += int(n)
        flagged = (
            sub["likely_saturated"].map(lambda v: str(v).strip().lower() in ("true", "1", "1.0", "yes"))
            if "likely_saturated" in sub.columns else pd.Series(False, index=sub.index)
        )
        over = flagged.copy()
        if "peak_max_adu" in sub.columns and "saturate_limit_adu_85pct" in sub.columns:
            pk = pd.to_numeric(sub["peak_max_adu"], errors="coerce")
            lim = pd.to_numeric(sub["saturate_limit_adu_85pct"], errors="coerce") * (
                float(comp_peak_sat_frac) / float(LINEARITY_FRAC))
            ok_p = pk.notna() & lim.gt(0)
            for cid, n in ids[ok_p].value_counts().items():
                out[str(cid)]["n_peak"] += int(n)
            over = over | (ok_p & (pk >= lim))
        for cid, n in ids[over].value_counts().items():
            out[str(cid)]["n_over"] += int(n)
    return out


def _old_field_k_by_brightness(proc_csv_paths, f_grid):
    import aperture_pertarget as ap
    from stats_core import coerce_bool_series
    from proc_frame_store import read_proc_csv

    ks = sorted(float(k) for k in f_grid)
    cols = [flux_k_column(k) for k in ks]
    cache = {}
    bad = set()
    seen = set()
    for p in proc_csv_paths:
        df = read_proc_csv(p, low_memory=False, dtype={"catalog_id": str})
        if "catalog_id" not in df.columns or not all(c in df.columns for c in cols):
            continue
        cid = df["catalog_id"].astype(str).str.strip()
        flag = np.zeros(len(df), dtype=bool)
        for f in ("vsx_known_variable", "gaia_dr3_variable_catalog", "likely_saturated", "is_saturated"):
            if f in df.columns:
                flag |= coerce_bool_series(df[f]).to_numpy()  # s5_bug: _truthy -> one bool vocabulary
        bad.update(cid[flag].tolist())
        seen.update(cid.tolist())
        cache[str(p)] = pd.DataFrame({"catalog_id": cid, **{c: pd.to_numeric(df[c], errors="coerce") for c in cols}})
    paths = [Path(p) for p in cache]
    ids = {c for c in seen - bad if c}
    if len(ids) < ap.BRIGHT_K_MIN_STARS or not paths:
        return None
    rms = {k: _old_loo(ids, paths, cache, flux_col=flux_k_column(k), strict_flux_col=True) for k in ks}
    common = sorted(set.intersection(*(set(m) for m in rms.values())))
    if len(common) < ap.BRIGHT_K_MIN_STARS:
        return None
    allf = pd.concat([d[["catalog_id", cols[0]]] for d in cache.values()], ignore_index=True)
    med_f = allf[allf[cols[0]] > 0].groupby("catalog_id")[cols[0]].median()
    mag = {c: float(-2.5 * np.log10(med_f[c])) for c in common if c in med_f.index}
    stars = sorted((m, c) for c, m in mag.items() if math.isfinite(m))
    ratios = []
    for _m, c in stars:
        v = np.array([rms[k][c] for k in ks], dtype=float)
        mn = float(np.min(v))
        ratios.append(v / mn if mn > 0 else np.full(len(ks), np.nan))
    R = np.array(ratios)
    M = np.array([m for m, _c in stars])
    b = np.floor((M - M.min()) / ap.BRIGHT_K_BIN_MAG).astype(int)
    groups = []
    cur = []
    for v in np.unique(b):
        cur.append(np.flatnonzero(b == v))
        if sum(len(x) for x in cur) >= ap.BRIGHT_K_MIN_STARS:
            groups.append(np.concatenate(cur))
            cur = []
    if cur:
        if not groups:
            return None
        groups[-1] = np.concatenate([groups[-1]] + cur)
    rng = np.random.default_rng(0)
    out = ap.BrightnessK([], [], [], [], [], [])
    for g in groups:
        r = R[g]
        r = r[np.all(np.isfinite(r), axis=1)]
        if len(r) < ap.BRIGHT_K_MIN_STARS:
            continue
        curve = np.median(r, axis=0)
        j = int(np.argmin(curve))
        boot = [np.median(r[rng.integers(0, len(r), len(r)), j]) for _ in range(ap.BRIGHT_K_BOOT)]
        se = float(np.std(boot))
        k_g = max(k for k, v in zip(ks, curve, strict=False) if v <= curve[j] + se)
        out.mag_lo.append(float(M[g].min()))
        out.mag_hi.append(float(M[g].max()))
        out.k.append(float(k_g))
        out.n_stars.append(int(len(r)))
        out.curves.append({f"{k:.4g}": float(v) for k, v in zip(ks, curve, strict=False)})
        out.se.append(se)
    if not out.k:
        return None
    for i in range(len(out.k) - 1):
        out.mag_hi[i] = out.mag_lo[i + 1]
    out.mag_hi[-1] = float("inf")
    out.mag_lo[0] = float("-inf")
    return out


def _synthetic_proc_dir(root: Path, *, n_frames=40, n_stars=260, seed=7) -> list[Path]:
    """Realistic proc CSVs: Gaia ids as text, missing stars, duplicate rows, NaN / zero / negative fluxes,
    one frame without the grid columns, flags, saturation columns, transparency and seeing per frame."""
    rng = np.random.default_rng(seed)
    ids = [str(4_000_000_000_000_000_000 + 7919 * i) for i in range(n_stars)]
    base = 10 ** rng.uniform(2.5, 6.0, n_stars)
    var = rng.random(n_stars) < 0.05
    paths = []
    for j in range(n_frames):
        keep = rng.random(n_stars) > 0.07
        idx = np.flatnonzero(keep)
        dup = rng.choice(idx, size=4, replace=False)
        idx = np.concatenate([idx, dup])
        trans = 10 ** (-0.4 * rng.normal(0.0, 0.05))
        df = pd.DataFrame({
            "catalog_id": [ids[i] for i in idx],
            "bjd_tdb_mid": 2460000.5 + 0.003 * ((j * 17) % n_frames) + rng.normal(0, 1e-6, len(idx)),
        })
        for k in DEFAULT_APERTURE_F_GRID:
            f = base[idx] * trans * (1 - math.exp(-k * k)) * (1 + rng.normal(0, 0.01 + 0.003 / k, len(idx)))
            f[rng.random(len(idx)) < 0.01] = np.nan
            f[rng.random(len(idx)) < 0.005] = 0.0
            f[rng.random(len(idx)) < 0.005] = -5.0
            df[flux_k_column(k)] = f
        df["dao_flux"] = df[flux_k_column(1.0)] * 1.01
        df["peak_max_adu"] = base[idx] * 0.02 * rng.uniform(0.8, 1.2, len(idx))
        df["saturate_limit_adu_85pct"] = 0.9 * 60000.0
        df["likely_saturated"] = df["peak_max_adu"] > 0.9 * 60000.0
        df["is_saturated"] = False
        df["vsx_known_variable"] = var[idx]
        df["gaia_dr3_variable_catalog"] = False
        if j == 5:
            df.loc[3, "catalog_id"] = np.nan
        if j == n_frames - 1:  # an old frame without the grid columns
            df = df.drop(columns=[flux_k_column(k) for k in DEFAULT_APERTURE_F_GRID])
        p = root / f"proc_frame{j:03d}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    return paths


@pytest.fixture(scope="module")
def proc_store(tmp_path_factory):
    root = tmp_path_factory.mktemp("g0731")
    paths = _synthetic_proc_dir(root)
    store = ProcFrameStore.build(root)
    return root, sorted(paths), store


def _bits(d):
    return {k: np.float64(v).tobytes() for k, v in d.items()}


def test_g07_31_loo_maps_all_k_equal_old_per_k(proc_store):
    _root, paths, store = proc_store
    cand = {str(x) for x in pd.concat([store.get(str(p)) for p in paths])["catalog_id"] if str(x)}
    cand = set(sorted(cand)[:150]) | {"not_an_id"}
    cols = [flux_k_column(k) for k in DEFAULT_APERTURE_F_GRID]
    new = loo_mag_rms_maps(cand, paths, store, cols, strict_flux_col=True)
    for c in cols:
        old = _old_loo(cand, paths, store, flux_col=c, strict_flux_col=True)
        assert list(new[c]) == list(old) and _bits(new[c]) == _bits(old)
    # non-strict single column (the fallback of the pool admission) and a missing column -> dao_flux
    for c in ("dao_flux", "flux_k999"):
        old = _old_loo(cand, paths, store, flux_col=c)
        new1, _ = compute_loo_mag_rms_map(cand, paths, store, flux_col=c)
        assert _bits(new1) == _bits(old) and len(old) > 50


def test_g07_31_choose_k_equal(proc_store):
    _root, paths, store = proc_store
    cand = {str(x) for x in store.get(str(paths[0]))["catalog_id"] if str(x)}
    old = _old_choose_k(cand, paths[:-1], store)
    new = choose_field_aperture_k(cand, paths[:-1], store)
    # COMP-CEIL-FIELDK-2026-10 (Milan 2026-10-30): the k rule changed (one bootstrap SE of the median RMS
    # ratio instead of 1 % of the minimum median RMS); the scores and the map at the chosen k are still the
    # old engine's, bit for bit
    assert old[0] is not None and new[0] is not None and _bits(old[2]) == _bits(new[2])
    ref = _old_loo(cand, paths[:-1], store, flux_col=flux_k_column(new[0]), min_frames_frac=0.3, strict_flux_col=True)
    assert _bits(ref) == _bits(new[1])


def test_g07_31_frame_stats_equal(proc_store):
    _root, paths, store = proc_store
    cand = {str(x) for x in store.get(str(paths[0]))["catalog_id"] if str(x)}
    for frac in (0.0005, 0.8):
        assert comp_frame_stats(cand, paths, store, comp_peak_sat_frac=frac) == _old_frame_stats(
            cand, paths, store, comp_peak_sat_frac=frac)


def test_g07_31_bright_k_from_store_equal_disk(proc_store):
    from aperture_pertarget import field_k_by_brightness

    root, _paths, store = proc_store
    glob = sorted(root.glob("proc_*.csv"))
    old = _old_field_k_by_brightness(glob, DEFAULT_APERTURE_F_GRID)
    new = field_k_by_brightness(glob, DEFAULT_APERTURE_F_GRID, frames=store)
    disk = field_k_by_brightness(glob, DEFAULT_APERTURE_F_GRID)
    assert old is not None and len(old.k) >= 2
    # AP-BRIGHT-K diagnostic k_within_se (2026-10-30) is new: the old reference has no such key
    no_kse = {"groups": [{k: v for k, v in g.items() if k != "k_within_se"} for g in new.to_dict()["groups"]]}
    assert repr(old.to_dict()) == repr(no_kse)
    assert repr(new.to_dict()) == repr(disk.to_dict())
    assert old.mag_lo == new.mag_lo and old.se == new.se


def test_g07_31_nan_row_median_bit_identical_to_np_median():
    from comp_rms_loo import _nan_row_median

    rng = np.random.default_rng(3)
    m = rng.normal(0, 0.01, (400, 37))
    m[rng.random(m.shape) < 0.3] = np.nan
    m[5] = np.nan
    got = _nan_row_median(m)
    for i in range(m.shape[0]):
        v = m[i][~np.isnan(m[i])]
        ref = float(np.median(v)) if v.size else float("nan")
        assert np.float64(ref).tobytes() == np.float64(got[i]).tobytes()


# ----------------------------------------------------------------------------------------------------------
# G07-09: the primary comparison set is a prefix of the wide (primary + reserve) ranking
# ----------------------------------------------------------------------------------------------------------
def _comp_case(seed: int, nan_target_colour: bool):
    from test_comp_determinism_synthetic import _TARGET_ID, _build_synthetic_dataset

    from config import AppConfig
    from photometry_comp import build_global_comp_pool

    target, ms, paths, cache = _build_synthetic_dataset()
    rng = np.random.default_rng(seed)
    ms = ms.copy()
    bp = rng.uniform(0.3, 2.6, len(ms))
    bp[rng.random(len(ms)) < 0.1] = np.nan
    ms["bp_rp"] = bp
    ms.loc[ms["catalog_id"] == _TARGET_ID, "bp_rp"] = np.nan if nan_target_colour else 1.0
    target = ms.loc[ms["catalog_id"] == _TARGET_ID].iloc[0]
    cfg = AppConfig()
    cfg.gs11_dilution_enabled = False
    pool = build_global_comp_pool(ms, paths, cache, frozenset({_TARGET_ID}), None, 800, 600, 0, cfg,
                                  min_frames_frac=0.3, fwhm_px=3.7, gain=1.0, read_noise=10.0, scint_params=_NO_SCINT)
    kw = dict(cfg=cfg, chip_fw=800, chip_fh=600, chip_interior_margin_px=0, min_dist_arcsec=30.0,
              global_comp_pool_df=pool, variable_target_catalog_ids=frozenset({_TARGET_ID}), fwhm_px=3.7,
              max_comp_rms=0.1)
    return target, ms, kw


@pytest.mark.parametrize("seed,nan_tc", [(1, False), (2, False), (3, True), (4, False)])
def test_g07_09_primary_is_prefix_of_wide_ranking(seed, nan_tc):
    from photometry_comp import select_comparison_stars_per_target

    target, ms, kw = _comp_case(seed, nan_tc)
    for n_min, n_max, extra in ((2, 3, 4), (3, 5, 8), (5, 4, 10), (3, 8, 1)):
        a = select_comparison_stars_per_target(target, ms, n_comp_min=n_min, n_comp_max=n_max, **kw)
        b = select_comparison_stars_per_target(target, ms, n_comp_min=n_min, n_comp_max=n_max + extra, **kw)
        ia, ib = list(a.get("catalog_id", [])), list(b.get("catalog_id", []))
        assert ia == ib[: len(ia)], (seed, n_min, n_max)


@pytest.mark.parametrize("seed,nan_tc", [(2, False), (3, True)])
def test_g07_09_one_ranking_primary_and_reserve_equal_separate_selections(seed, nan_tc):
    """Primary rows == the n_comp_max selection (all columns, notes and tier counts of the head);
    reserve ids == the n_comp_max + n_reserve selection beyond the primary, in rank order."""
    from photometry_comp import select_comparison_stars_per_target
    from target_comp_selection import _split_comp_reserve

    target, ms, kw = _comp_case(seed, nan_tc)
    for n_min, n_max, n_res in ((2, 3, 4), (3, 5, 8), (5, 4, 10)):
        base = select_comparison_stars_per_target(target, ms, n_comp_min=n_min, n_comp_max=n_max, **kw)
        wide = select_comparison_stars_per_target(target, ms, n_comp_min=n_min, n_comp_max=n_max + n_res, **kw)
        both = select_comparison_stars_per_target(target, ms, n_comp_min=n_min, n_comp_max=n_max, n_reserve=n_res,
                                                  **kw)
        prim, resv = _split_comp_reserve(both)
        pd.testing.assert_frame_equal(prim, base.reset_index(drop=True), check_exact=True)
        w = list(wide["catalog_id"])
        assert list(resv.get("catalog_id", [])) == w[len(prim):]
        if len(resv):
            assert list(resv["reserve_rank"]) == list(range(len(resv)))


# ----------------------------------------------------------------------------------------------------------
# G07-12: blend gate loads catalogue / WCS / neighbour fluxes once for the k grid
# ----------------------------------------------------------------------------------------------------------
import comp_blend as cb  # noqa: E402


def _old_blend_contamination_for_pool(
    pool: pd.DataFrame,
    *,
    field_catalog_csv: Path | None,
    masterstar_fits: Path | None,
    zp_band: str,
    fwhm_px: float,
    profile_fwhm_px: float,
    r_ap_px: float,
    sat_adu: float | None,
    annulus_px: tuple[float, float] | None = None,
    profile_df: pd.DataFrame | None = None,
    psf: dict[str, Any] | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Reference copy of comp_blend.blend_contamination_for_pool on HEAD 08e985a."""
    from astropy.io import fits  # noqa: PLC0415
    from astropy.wcs import WCS  # noqa: PLC0415

    from gaia_bands import band_mag_from_gaia  # noqa: PLC0415

    meta: dict[str, Any] = {
        "rule_id": "COMP_BLEND_NEIGHBOUR_LIGHT",
        "threshold_flux_frac": cb.BLEND_FLUX_FRAC_MAX,
        "r_ap_px": float(r_ap_px),
        "fwhm_px": float(fwhm_px),
        "annulus_px": list(annulus_px) if annulus_px is not None else None,
        "zp_band": str(zp_band),
    }
    nan = pd.Series(np.nan, index=pool.index, dtype=float)
    if field_catalog_csv is None or not Path(field_catalog_csv).is_file():
        meta["status"] = "skipped: field_catalog_cone.csv missing"
        return nan, meta
    if masterstar_fits is None or not Path(masterstar_fits).is_file():
        meta["status"] = "skipped: MASTERSTAR missing"
        return nan, meta
    if not (math.isfinite(fwhm_px) and fwhm_px > 0 and math.isfinite(r_ap_px) and r_ap_px > 0):
        meta["status"] = "skipped: FWHM / aperture unknown"
        return nan, meta
    cat = cb.read_vyvar_csv(field_catalog_csv, low_memory=False)  # ids as text (G07-13)
    with fits.open(masterstar_fits, memmap=False) as hdul:
        hdr = hdul[0].header
        img = np.asarray(hdul[0].data, dtype=np.float32)
    wcs = WCS(hdr)
    cx, cy = wcs.all_world2pix(
        pd.to_numeric(cat["ra_deg"], errors="coerce").to_numpy(),
        pd.to_numeric(cat["dec_deg"], errors="coerce").to_numpy(),
        0,
    )
    cg = pd.to_numeric(cat["mag"], errors="coerce").to_numpy(dtype=float)
    cbp = pd.to_numeric(cat.get("bp_rp"), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(cg)
    cx, cy, cg, cbp = cx[ok], cy[ok], cg[ok], cbp[ok]

    def _band_flux(g: np.ndarray, bp: np.ndarray) -> np.ndarray:
        """Band flux, NaN when the Gaia -> band relation is not valid for the star."""
        m = np.array([band_mag_from_gaia(a, b, zp_band)[0] for a, b in zip(g, bp, strict=False)])
        return 10.0 ** (-0.4 * m)

    from gaia_catalog_id import normalize_gaia_source_id  # noqa: PLC0415

    nb_ids = [str(normalize_gaia_source_id(v) or "") for v in cat["catalog_id"].to_numpy()[ok]]
    nb_xy = np.c_[cx, cy]
    # B4-2026-10: a known variable neighbour shines at its maximum (Gaia mean + VSX amplitude is an
    # upper bound: the mean lies between maximum and minimum)
    amp, amp_src = cb.variable_amplitudes(Path(field_catalog_csv).parent / cb.VARIABLE_TARGETS_CSV)
    nb_boost = np.array([10.0 ** (0.4 * amp[i]) if i in amp else 1.0 for i in nb_ids], dtype=float)
    nb_flux = _band_flux(cg, cbp) * nb_boost
    meta["variable_neighbours"] = {"source": amp_src, "n_at_maximum": int(np.count_nonzero(nb_boost > 1.0))}
    meta["neighbour_catalog_mag_limit"] = float(np.nanmax(cg)) if cg.size else float("nan")
    meta["n_neighbour_catalog"] = int(cg.size)

    px = pd.to_numeric(pool.get("x"), errors="coerce").to_numpy(dtype=float)
    py = pd.to_numeric(pool.get("y"), errors="coerce").to_numpy(dtype=float)
    pg = pd.to_numeric(pool.get("phot_g_mean_mag", pool.get("mag")), errors="coerce").to_numpy(dtype=float)
    if "mag" in pool.columns:
        pg = np.where(np.isfinite(pg), pg, pd.to_numeric(pool["mag"], errors="coerce").to_numpy(dtype=float))
    pbp = pd.to_numeric(pool.get("bp_rp"), errors="coerce").to_numpy(dtype=float)

    # PSF wing (beta) from all detected stars of the MASTERSTAR (``profile_df``), not only the pool.
    _pf = profile_df if profile_df is not None and not profile_df.empty else pool
    _ppx = pd.to_numeric(_pf.get("x"), errors="coerce").to_numpy(dtype=float)
    _ppy = pd.to_numeric(_pf.get("y"), errors="coerce").to_numpy(dtype=float)
    _ppg = pd.to_numeric(_pf.get("mag"), errors="coerce").to_numpy(dtype=float)
    _pk = _pf.get("peak_max_adu")
    peaks = pd.to_numeric(_pk, errors="coerce").to_numpy(dtype=float) if _pk is not None else np.full(len(_pf), np.nan)
    sat = float(sat_adu) if sat_adu is not None and math.isfinite(float(sat_adu)) else float("nan")
    if psf is not None and math.isfinite(float(psf.get("beta", float("nan")))):
        prof = dict(psf)
    elif math.isfinite(sat) and sat > 0 and np.isfinite(peaks).any():
        prof = cb.measure_moffat_beta(img, _ppx, _ppy, peaks, sat, float(profile_fwhm_px), nb_xy, cg, _ppg)
    else:
        prof = {"beta": cb.MOFFAT_BETA_FALLBACK, "fwhm_fit_px": float("nan"), "n_stars": 0, "source": "fallback"}
    meta["psf"] = prof
    # Geometry from catalogue positions (star and neighbours on the same WCS projection).
    pra = pd.to_numeric(pool.get("ra_deg"), errors="coerce").to_numpy(dtype=float)
    pdec = pd.to_numeric(pool.get("dec_deg"), errors="coerce").to_numpy(dtype=float)
    sx, sy = wcs.all_world2pix(pra, pdec, 0)
    sx = np.where(np.isfinite(sx), sx, px)
    sy = np.where(np.isfinite(sy), sy, py)
    _idc = "catalog_id" if "catalog_id" in pool.columns else "name"
    s_ids = [str(normalize_gaia_source_id(v) or "") for v in pool[_idc].to_numpy()]
    c = cb.contamination_fractions(
        np.c_[sx, sy], _band_flux(pg, pbp), nb_xy, nb_flux,
        r_ap_px=float(r_ap_px), fwhm_px=float(fwhm_px), beta=float(prof["beta"]),
        annulus_px=annulus_px,
        star_ids=s_ids, neighbour_ids=nb_ids,
        star_flux_g=10.0 ** (-0.4 * pg), neighbour_flux_g=10.0 ** (-0.4 * cg) * nb_boost,
    )
    meta["status"] = "ok"
    return pd.Series(c, index=pool.index, dtype=float), meta


def _blend_field(root: Path):
    from astropy.io import fits
    from astropy.wcs import WCS

    rng = np.random.default_rng(11)
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [200.0, 40.0]
    w.wcs.crpix = [200.0, 200.0]
    w.wcs.cdelt = [-1.2 / 3600.0, 1.2 / 3600.0]
    n = 220
    x = rng.uniform(10, 390, n)
    y = rng.uniform(10, 390, n)
    g = rng.uniform(11.0, 19.5, n)
    yy, xx = np.mgrid[0:400, 0:400]
    img = np.full((400, 400), 500.0)
    alpha = cb.moffat_alpha(3.4, 2.7)
    peak = 20000.0 * 10 ** (-0.4 * (g - 12.0))
    for xi, yi, pk in zip(x, y, peak, strict=True):
        d2 = (xx - xi) ** 2 + (yy - yi) ** 2
        m = d2 < 400
        img[m] += pk * (1 + d2[m] / alpha**2) ** -2.7
    fits.PrimaryHDU(img.astype(np.float32), header=w.to_header()).writeto(root / "MASTERSTAR.fits")
    ra, de = w.all_pix2world(x, y, 0)
    ids = [str(5_000_000_000_000_000_000 + 13 * i) for i in range(n)]
    bp = rng.uniform(0.2, 2.8, n)
    bp[::9] = np.nan
    pd.DataFrame({"catalog_id": ids, "ra_deg": ra, "dec_deg": de, "mag": g, "bp_rp": bp}).to_csv(
        root / "field_catalog_cone.csv", index=False)
    pd.DataFrame({"name": ["V1", "V2"], "catalog_id": [ids[3], ids[40]], "vsx_mag_max": [12.5, 14.0],
                  "vsx_amplitude_mag": [0.8, 1.6]}).to_csv(root / "variable_targets.csv", index=False)
    ms = pd.DataFrame({"catalog_id": ids, "x": x, "y": y, "ra_deg": ra, "dec_deg": de, "mag": g, "bp_rp": bp,
                       "peak_max_adu": peak + 500.0})
    pool = ms.iloc[::4].copy()
    return pool, ms


def test_g07_12_blend_grid_one_load_equal_old(tmp_path):
    import json

    pool, ms = _blend_field(tmp_path)
    fc, mf = tmp_path / "field_catalog_cone.csv", tmp_path / "MASTERSTAR.fits"
    field = cb.BlendField(fc, mf, "g")
    for band in ("g", "G"):
        fld = field if band == "g" else None
        psf_o = psf_n = None
        for k in DEFAULT_APERTURE_F_GRID:
            kw = dict(field_catalog_csv=fc, masterstar_fits=mf, zp_band=band, fwhm_px=4.1, profile_fwhm_px=3.4,
                      r_ap_px=k * 4.1, sat_adu=60000.0, annulus_px=(2.7 * 4.1, 5.2 * 4.1), profile_df=ms)
            co, mo = _old_blend_contamination_for_pool(pool, psf=psf_o, **kw)
            cn, mn = cb.blend_contamination_for_pool(pool, psf=psf_n, field=fld, **kw)
            assert mo["status"] == "ok" and np.isfinite(co.to_numpy()).sum() > 20
            assert co.to_numpy().tobytes() == cn.to_numpy().tobytes() and list(co.index) == list(cn.index)
            assert json.dumps(mo, sort_keys=True, default=str) == json.dumps(mn, sort_keys=True, default=str)
            psf_o, psf_n = mo["psf"], mn["psf"]
    with pytest.raises(ValueError):
        cb.blend_contamination_for_pool(pool, field=field, **{**kw, "zp_band": "r"})


# ----------------------------------------------------------------------------------------------------------
# G07-16: night FWHM from the in-memory proc store == from the CSVs
# ----------------------------------------------------------------------------------------------------------
def test_g07_16_night_fwhm_from_store_equal_disk(tmp_path):
    from photometry_comp import night_frame_fwhms_px, night_median_fwhm_px
    from proc_frame_store import write_frame_table

    rng = np.random.default_rng(5)
    inline, table = tmp_path / "inline", tmp_path / "table"
    inline.mkdir()
    table.mkdir()
    rows = {}
    for j in range(30):
        n = 50
        fw = rng.uniform(2.5, 9.0) + rng.normal(0, 1e-3, n)
        if j == 4:
            fw[:] = np.nan
        df = pd.DataFrame({"catalog_id": [str(1000 + i) for i in range(n)], "dao_flux": rng.uniform(1, 9, n),
                           "fwhm_px_for_aperture": fw})
        df.to_csv(inline / f"proc_f{j:02d}.csv", index=False)
        df.drop(columns=["fwhm_px_for_aperture"]).to_csv(table / f"proc_f{j:02d}.csv", index=False)
        rows[f"proc_f{j:02d}.csv"] = {"fwhm_px_for_aperture": float(rng.uniform(2.5, 9.0))}
    write_frame_table(table, rows)
    for d in (inline, table):
        paths = sorted(d.glob("proc_*.csv"))
        store = ProcFrameStore.build(d)
        old = night_frame_fwhms_px(paths)
        new = night_frame_fwhms_px(paths, store)
        assert len(old) >= 29 and np.asarray(old).tobytes() == np.asarray(new).tobytes()
        assert night_median_fwhm_px(paths, 3.0) == night_median_fwhm_px(paths, 3.0, store=store) == \
            night_median_fwhm_px(None, 3.0, frame_fwhms=new)
    # a frame missing from the store is read from disk
    part = {str(p): store.get(str(p)) for p in paths[:10]}
    assert night_frame_fwhms_px(paths, part) == night_frame_fwhms_px(paths)


# ----------------------------------------------------------------------------------------------------------
# G07-18: the linearity check runs outside comparison selection, one read per frame, same result
# ----------------------------------------------------------------------------------------------------------
def _lin_frames(root: Path, n_frames=4):
    rng = np.random.default_rng(0)
    paths = []
    for k in range(n_frames):
        n = 400
        g = rng.uniform(9, 15, n)
        c = rng.uniform(0.3, 1.5, n)
        flux = 10 ** (-0.4 * (g - 25))
        peak = flux * 0.1
        frac = peak / 65535.0
        df = pd.DataFrame({"catalog_id": [str(10**15 + i) for i in range(n)], "x": rng.uniform(0, 99, n),
                           "peak_max_adu": peak, "dao_flux": flux * np.where(frac > 0.7, 1 - 0.3 * (frac - 0.7), 1.0),
                           "phot_g_mean_mag": g, "bp_rp": c, "airmass": 1.2})
        if k == 1:
            df["peak_max_adu_aligned"] = peak * 1.01
        p = root / f"proc_{k}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    return paths


def test_g07_18_linearity_one_read_same_frame(tmp_path):
    from gaia_catalog_id import read_vyvar_csv

    want = ("peak_max_adu_aligned", "peak_max_adu", "dao_flux", "flux", "phot_g_mean_mag", "bp_rp")
    for p in _lin_frames(tmp_path):
        hdr = pd.read_csv(p, nrows=0)
        old = read_vyvar_csv(p, usecols=[c for c in want if c in hdr.columns])
        new = read_vyvar_csv(p, usecols=lambda c: c in want)
        pd.testing.assert_frame_equal(old, new, check_exact=True)


def test_g07_18_record_linearity_check_writes_block(tmp_path):
    import json

    import target_comp_selection
    from sat_diag import linearity_check, record_linearity_check

    paths = _lin_frames(tmp_path)
    sd = tmp_path / "sat_diag.json"
    sd.write_text(json.dumps({"sat_adu": 65535.0, "sat_source": "HEADER"}), encoding="ascii")
    lin = record_linearity_check(sd, paths)
    assert lin == linearity_check(paths, 65535.0) and lin["ok"] and lin["warning"]
    assert json.loads(sd.read_text())["linearity_check"] == json.loads(json.dumps(lin))
    assert record_linearity_check(tmp_path / "none.json", paths) is None
    # comparison selection no longer runs it
    assert "linearity_check" not in Path(target_comp_selection.__file__).read_text(encoding="utf-8")


# ----------------------------------------------------------------------------------------------------------
# G07-19 (option b): field density from the already loaded MASTERSTAR header and masterstars table
# ----------------------------------------------------------------------------------------------------------
def _old_read_field_density_inputs(ms_fits, masterstars_csv, frame_w_px, frame_h_px):
    from astropy.io import fits as astrofits
    from gaia_catalog_id import read_vyvar_csv

    cw, ch = int(frame_w_px), int(frame_h_px)
    n_stars = 0
    src = "defaults"
    vy_ndao_raw = None
    if ms_fits.is_file():
        try:
            with astrofits.open(ms_fits, memmap=False) as hdul:
                v = hdul[0].header.get("VY_NDAO")
                if v is not None and str(v).strip() != "":
                    try:
                        vy_ndao_raw = int(float(v))
                    except (TypeError, ValueError):
                        vy_ndao_raw = None
        except Exception:  # noqa: BLE001
            pass
    msc_path = Path(masterstars_csv)
    if msc_path.is_file():
        try:
            _msc_df = read_vyvar_csv(msc_path, usecols=["catalog_id"], low_memory=False, dtype={"catalog_id": str})
            _cid = _msc_df["catalog_id"].astype(str).str.strip()
            _n_gaia = int((~_cid.isin(["", "nan", "None"])).sum())
            if _n_gaia > 0:
                n_stars = _n_gaia
                src = "masterstars_gaia_matched"
        except Exception:  # noqa: BLE001
            pass
    if n_stars <= 0 and vy_ndao_raw is not None and vy_ndao_raw > 0:
        n_stars = int(vy_ndao_raw)
        src = "VY_NDAO_fallback"
    return int(max(0, n_stars)), cw, ch, src, vy_ndao_raw


@pytest.mark.parametrize("ndao,ids", [
    (1234, ["1", "", "None", "nan", " 77 ", "3"]),
    ("812.0", [None, ""]),
    ("garbage", ["5"]),
    (None, [None]),
])
def test_g07_19_field_density_inputs_equal(tmp_path, ndao, ids):
    from astropy.io import fits

    from gaia_catalog_id import VYVAR_CSV_DTYPE
    from photometry_comp import field_density_inputs
    from target_comp_selection import _read_masterstar_header

    hdr = fits.Header()
    if ndao is not None:
        hdr["VY_NDAO"] = ndao
    fits.PrimaryHDU(np.zeros((4, 4), np.float32), header=hdr).writeto(tmp_path / "MASTERSTAR.fits")
    pd.DataFrame({"catalog_id": ids, "x": range(len(ids))}).to_csv(tmp_path / "ms.csv", index=False)
    old = _old_read_field_density_inputs(tmp_path / "MASTERSTAR.fits", tmp_path / "ms.csv", 4144, 2822)
    ms_raw = pd.read_csv(tmp_path / "ms.csv", low_memory=False, dtype=VYVAR_CSV_DTYPE)
    new = field_density_inputs(_read_masterstar_header(tmp_path / "MASTERSTAR.fits"), ms_raw["catalog_id"], 4144, 2822)
    assert old == new


# ----------------------------------------------------------------------------------------------------------
# G08-19: spike-evidence frames from a night-level LRU (one FITS read for all targets), same evidence
# ----------------------------------------------------------------------------------------------------------
def test_g08_19_evidence_frames_read_once_and_equal(tmp_path, monkeypatch):
    from astropy.io import fits

    import lc_outlier
    from lc_outlier import EvidenceCache, evaluate_image_evidence, similar_flux_peers

    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:120, 0:140]
    ids = [str(9000 + i) for i in range(12)]
    xs, ys = rng.uniform(15, 125, 12), rng.uniform(15, 105, 12)
    names = []
    for j, kind in enumerate(("f4", "i2", "f8")):
        img = 300.0 + rng.normal(0, 5, xx.shape)
        for x0, y0 in zip(xs, ys, strict=True):
            img += 4000.0 * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / 4.5)
        hdu = fits.PrimaryHDU(np.round(img).astype(np.int16) if kind == "i2" else img.astype(kind))
        hdu.writeto(tmp_path / f"fr{j}.fits")
        pd.DataFrame({"catalog_id": ids, "x": xs, "y": ys, "dao_flux": rng.uniform(1e4, 2e4, 12),
                      "sigma_bkg_ap": 5.0}).to_csv(tmp_path / f"proc_fr{j}.csv", index=False)
        names.append(f"proc_fr{j}.csv")
    opens = []
    real_open = fits.open
    monkeypatch.setattr(fits, "open", lambda *a, **k: (opens.append(a[0]), real_open(*a, **k))[1])
    lc_outlier._FRAME_LRU.clear()
    for cid in ids[:6]:
        ev = EvidenceCache(frames_dir=tmp_path, catalog_id=cid)
        for i in range(3):
            got = ev.evidence_at(i, names)
            with real_open(tmp_path / f"fr{i}.fits") as h:
                full = np.asarray(h[0].data, dtype=np.float64)  # the old cache: whole frame as float64
            proc = pd.read_csv(tmp_path / names[i], dtype={"catalog_id": str})
            row = proc[proc["catalog_id"] == cid].iloc[0]
            ref = evaluate_image_evidence(full, float(row["x"]), float(row["y"]),
                                          peer_xy=similar_flux_peers(proc, cid), n_sigma=ev.n_sigma,
                                          psf_chi2_locus_resid_sigma=None,
                                          star_annulus_sky_sigma_night=ev.night_annulus_sky_sigma(names),
                                          star_annulus_sky_sigma_now=5.0)
            assert repr(got) == repr(ref), (repr(got), repr(ref))
    assert len(opens) == 3  # three frames, six targets: each frame read once
    assert len(lc_outlier._FRAME_LRU) <= lc_outlier.EVIDENCE_FRAME_LRU_SIZE


# ----------------------------------------------------------------------------------------------------------
# G06-40: one catalog_id index of masterstars_df instead of a column scan per target
# ----------------------------------------------------------------------------------------------------------
def test_g06_40_masterstar_row_equals_scan():
    from types import SimpleNamespace

    from aperture_photometry import _AperturePhotometryState
    from photometry_shared import _normalize_gaia_id

    rng = np.random.default_rng(4)
    raw = [str(10**17 + int(i)) for i in rng.integers(0, 300, 900)] + ["", "nan", "4.0", " 77"]
    ms = pd.DataFrame({"catalog_id": raw, "mag": rng.uniform(9, 17, len(raw)), "bp_rp": rng.uniform(0, 2, len(raw))})
    ms["catalog_id"] = ms["catalog_id"].map(_normalize_gaia_id)  # as aperture photometry loads it
    st = SimpleNamespace(masterstars_df=ms, _ms_first_pos=None)
    for cid in list(dict.fromkeys(ms["catalog_id"]))[:200] + ["", "77", "123", "nope"]:
        old_strip = ms.loc[ms["catalog_id"].astype(str).str.strip() == cid]
        old_norm = ms.loc[ms["catalog_id"].astype(str).map(_normalize_gaia_id) == cid]
        new = _AperturePhotometryState.masterstar_row(st, cid)
        for old in (old_strip, old_norm):
            if old.empty:
                assert new is None
            else:
                pd.testing.assert_series_equal(old.iloc[0], new)


# ----------------------------------------------------------------------------------------------------------
# G08-50: target airmass and BJD/HJD on one targets x frames grid == per-target calls (bit-identical)
# ----------------------------------------------------------------------------------------------------------
def _old_target_airmass(jd_array, ra_deg, dec_deg, site, fallback=None):
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    from time_utils import _airmass_from_altitude_deg

    jd_arr = np.asarray(jd_array, dtype=float)
    out = (np.asarray(fallback, dtype=float).copy() if fallback is not None and len(fallback) == len(jd_arr)
           else np.full(len(jd_arr), np.nan))
    lat, lon, alt = site
    finite = np.isfinite(jd_arr)
    location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt * u.m)
    t = Time(jd_arr[finite], format="jd", scale="utc", location=location)
    alt_deg = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs").transform_to(
        AltAz(obstime=t, location=location)).alt.deg
    am = np.array([_airmass_from_altitude_deg(a) if 5.0 <= a <= 90.0 else np.nan
                   for a in np.atleast_1d(alt_deg)], dtype=float)
    idx = np.flatnonzero(finite)
    ok = np.isfinite(am)
    out[idx[ok]] = np.round(am[ok], 6)
    return out


def test_g08_50_time_grid_equals_per_target():
    from config import AppConfig
    from photometry_lightcurve import TargetTimeGrid, _recompute_bjd_hjd_with_status, target_airmass
    from time_utils import light_time_corrected_jd

    cfg = AppConfig()
    site = (50.0815, 14.6833, 290.0)  # Jirny
    rng = np.random.default_rng(8)
    jd_night = 2461000.25 + np.sort(rng.uniform(0, 0.42, 160))
    coords = {str(10**18 + i): (float(rng.uniform(150, 260)), float(rng.uniform(-5, 75))) for i in range(9)}
    grid = TargetTimeGrid(jd_night, coords, cfg, site=site)
    for n, (cid, (ra, de)) in enumerate(coords.items()):
        jd = jd_night[rng.random(jd_night.size) > 0.1].copy()
        if n == 2:
            jd[[3, 7]] = np.nan
        if n == 3:
            jd = np.concatenate([jd, jd[:5]])  # repeated epochs
        fb = rng.uniform(1.0, 2.0, jd.size)
        old_am = _old_target_airmass(jd, ra, de, site, fallback=fb)
        new_am = target_airmass(jd, ra, de, cfg, site=site, fallback=fb, grid=grid, cid=cid)
        alone = target_airmass(jd, ra, de, cfg, site=site, fallback=fb)
        assert old_am.tobytes() == new_am.tobytes() == alone.tobytes()
        ob, oh = light_time_corrected_jd(jd, ra, de, site)
        nb, nh, tb = _recompute_bjd_hjd_with_status(jd, ra, de, cfg, site=site, grid=grid, cid=cid)
        ab, ah, _ = _recompute_bjd_hjd_with_status(jd, ra, de, cfg, site=site)
        assert tb == "BJD_TDB" or tb
        assert ob.tobytes() == nb.tobytes() == ab.tobytes() and oh.tobytes() == nh.tobytes() == ah.tobytes()
    # a JD outside the grid or other coordinates -> computed for the target alone (same values)
    cid0, (ra0, de0) = next(iter(coords.items()))
    jd_x = np.array([jd_night[0], jd_night[0] + 0.0123])
    assert target_airmass(jd_x, ra0, de0, cfg, site=site, grid=grid, cid=cid0).tobytes() == \
        _old_target_airmass(jd_x, ra0, de0, site).tobytes()
    assert target_airmass(jd_night, ra0 + 1, de0, cfg, site=site, grid=grid, cid=cid0).tobytes() == \
        _old_target_airmass(jd_night, ra0 + 1, de0, site).tobytes()


# ----------------------------------------------------------------------------------------------------------
# G08-44: trust flags read each file of a target once, same values
# ----------------------------------------------------------------------------------------------------------
def _trust_dir(root: Path) -> list[str]:
    import json

    rng = np.random.default_rng(6)
    lc = root / "lightcurves"
    lc.mkdir(parents=True)
    tids = [str(2 * 10**18 + i) for i in range(4)]
    for n, t in enumerate(tids):
        m = 60
        src = [f"proc_f{j:03d}.csv" for j in range(m)]
        mag = 12 + rng.normal(0, 0.01, m)
        mag[3] = np.nan
        flag = ["normal"] * m
        flag[5] = "saturated"
        pd.DataFrame({"source_file": src, "jd": 2461000 + np.arange(m) * 0.003, "bjd": 2461000 + np.arange(m) * 0.003,
                      "mag_calib": mag, "err": 0.01, "flag": flag}).to_csv(lc / f"lightcurve_{t}.csv", index=False)
        if n == 3:
            continue  # a target without check star / comp_quality
        pd.DataFrame({"source_file": src[::-1], "kmag": 13 + rng.normal(0, 0.02, m), "bjd": 0.0,
                      "kmag_err": rng.uniform(0.01, 0.03, m)}).to_csv(lc / f"check_kmag_{t}.csv", index=False)
        kc = str(3 * 10**18 + n)
        cq = {"band_system": "SDSS" if n != 1 else "NONE", "zero_point": {"sigma_zp_data_mag": 0.012},
              "comp_band_mag": {kc: 13.01}, "check_star": {"catalog_id": kc, "check_sparse": n == 2, "trust_R": 1.2}}
        (lc / f"comp_quality_{t}.json").write_text(json.dumps(cq), encoding="utf-8")
    return tids


def test_g08_44_trust_files_read_once_equal(tmp_path, monkeypatch):
    import json

    import trust_flag_core as tfc
    from check_star_kmag import exported_check_stats, load_check_star
    from gaia_catalog_id import read_vyvar_csv
    from photometry_lightcurve import export_row_mask, lc_export_mag_column

    tids = _trust_dir(tmp_path)
    lc = tmp_path / "lightcurves"

    def old_chi2(t):
        p = lc / f"check_kmag_{t}.csv"
        if not p.is_file():
            return float("nan"), float("nan")
        sdf = read_vyvar_csv(p, low_memory=False)
        k = pd.to_numeric(sdf["kmag"], errors="coerce").to_numpy(dtype=np.float64)
        e = pd.to_numeric(sdf["kmag_err"], errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(k) & np.isfinite(e) & (e > 0)
        kk, ee = k[ok], e[ok]
        z = (kk - float(np.median(kk))) / ee
        ratio = float(1.4826 * np.median(np.abs(z - np.median(z))))
        w = 1.0 / (ee**2)
        mu = float(np.sum(w * kk) / np.sum(w))
        from scipy.stats import chi2 as _chi2

        return ratio, float(_chi2.sf(float(np.sum(w * (kk - mu) ** 2)), int(ok.sum()) - 1))

    def old_delta(t):
        try:
            cq = json.loads((lc / f"comp_quality_{t}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return float("nan"), float("nan")
        if str(cq.get("band_system", "")).upper() in ("", "NONE"):
            return float("nan"), float("nan")
        zs = cq["zero_point"]["sigma_zp_data_mag"]
        kc = str(load_check_star(lc, t).get("catalog_id", "") or "")
        kref = cq["comp_band_mag"].get(kc)
        return float(exported_check_stats(lc, t).median) - float(kref), float(zs)

    def old_counts(t):
        df = read_vyvar_csv(lc / f"lightcurve_{t}.csv", low_memory=False)
        return int(export_row_mask(df, mag_column=lc_export_mag_column(df)).sum()), int(len(df))

    reads = []
    real = tfc.read_vyvar_csv
    monkeypatch.setattr(tfc, "read_vyvar_csv", lambda p, **k: (reads.append(Path(p).name), real(p, **k))[1])
    for t in tids:
        reads.clear()
        f = tfc.TargetTrustFiles(tmp_path, t)
        st = exported_check_stats(lc, t)
        assert tfc.check_star_scatter(tmp_path, t, f) == (st.scatter, st.n) or (
            np.isnan(st.scatter) and tfc.check_star_scatter(tmp_path, t, f)[1] == st.n)
        assert repr(tfc.check_star_chi2(tmp_path, t, f)) == repr(old_chi2(t))
        assert repr(tfc.check_delta_and_zp_sigma(tmp_path, t, f)) == repr(old_delta(t))
        assert tfc.lc_epoch_counts(tmp_path, t, f) == old_counts(t)
        side = tfc.read_sparse_trust_sidecar(tmp_path, t, f)
        assert repr(side) == repr(tfc.read_sparse_trust_sidecar(tmp_path, t))
        assert sorted(reads) == sorted(set(reads)) and len(reads) <= 2  # each CSV of the target once


# ----------------------------------------------------------------------------------------------------------
# G09-27 / G09-28: internal PSF LC stack read with usecols, split by star once
# ----------------------------------------------------------------------------------------------------------
def _old_read_proc_stack(root):
    import psf_internal_lc as pil
    from gaia_catalog_id import normalize_gaia_source_id
    from proc_frame_store import read_proc_csv

    chunks = []
    for p in sorted(Path(root).glob("proc_*.csv")):
        df = read_proc_csv(p, low_memory=False, dtype={"catalog_id": str})
        keep = [c for c in pil._PROC_USECOLS if c in df.columns]
        if "catalog_id" not in keep:
            continue
        df = df[keep].copy()
        df["catalog_id"] = df["catalog_id"].map(normalize_gaia_source_id)
        df["proc_csv"] = p.name
        df["epoch_key"] = p.name
        chunks.append(df)
    return pd.concat(chunks, ignore_index=True)


def _old_star_frame_table(stack, cid, epoch_keys):
    from psf_internal_lc import coerce_bool as _coerce_bool  # s5_bug: the one bool vocabulary

    sub = stack.loc[stack["catalog_id"] == cid]
    by_key = {}
    for _, row in sub.iterrows():
        by_key[str(row.get("epoch_key") or "")] = row
    rows = []
    for key in epoch_keys:
        rec = by_key.get(str(key))
        if rec is None:
            rows.append({"epoch_key": key, "psf_flux": np.nan, "psf_flux_err": np.nan, "psf_chi2": np.nan,
                         "psf_fit_ok": False, "sat_masked": False, "edge_masked": False, "aperture_flux": np.nan})
            continue
        flux = float(pd.to_numeric(rec.get("dao_flux"), errors="coerce"))
        sat = _coerce_bool(rec.get("psf_sat_masked"))
        edge = _coerce_bool(rec.get("psf_edge_masked"))
        rows.append({"epoch_key": key,
                     "psf_flux": np.nan if (sat or edge) else float(pd.to_numeric(rec.get("psf_flux"), errors="coerce")),
                     "psf_flux_err": float(pd.to_numeric(rec.get("psf_flux_err"), errors="coerce")),
                     "psf_chi2": float(pd.to_numeric(rec.get("psf_chi2"), errors="coerce")),
                     "psf_fit_ok": _coerce_bool(rec.get("psf_fit_ok")) and not (sat or edge),
                     "sat_masked": sat, "edge_masked": edge, "aperture_flux": flux})
    return pd.DataFrame(rows)


def test_g09_27_28_psf_stack_and_star_tables_equal(tmp_path):
    from proc_frame_store import write_frame_table
    from psf_internal_lc import _read_proc_stack, _stack_by_star, _star_frame_table

    rng = np.random.default_rng(12)
    ids = [str(6 * 10**18 + i) for i in range(40)]
    rows = {}
    for j in range(25):
        idx = rng.choice(40, 36, replace=False)
        n = idx.size
        df = pd.DataFrame({"catalog_id": [ids[i] for i in idx], "x": rng.uniform(0, 9, n),
                           "dao_flux": rng.uniform(1e3, 1e5, n), "psf_flux": rng.uniform(1e3, 1e5, n),
                           "psf_flux_err": rng.uniform(1, 9, n), "psf_chi2": rng.uniform(0.5, 3, n),
                           "psf_fit_ok": rng.random(n) > 0.1, "psf_sat_masked": rng.random(n) > 0.95,
                           "psf_edge_masked": rng.random(n) > 0.97, "sky": 1.0})
        if j == 3:
            df = pd.concat([df, df.iloc[:2].assign(psf_flux=7.0)], ignore_index=True)  # duplicate rows: last wins
        if j == 4:
            df.loc[0, "catalog_id"] = "VSX_NAME"
        df.to_csv(tmp_path / f"proc_f{j:02d}.csv", index=False)
        rows[f"proc_f{j:02d}.csv"] = {"bjd_tdb_mid": 2461000.5 + j * 0.01, "jd_mid": 2461000.49 + j * 0.01}
    write_frame_table(tmp_path, rows)
    old = _old_read_proc_stack(tmp_path)
    new = _read_proc_stack(tmp_path)
    pd.testing.assert_frame_equal(old, new, check_exact=True)
    keys = sorted(rows) + ["proc_missing.csv"]
    by = _stack_by_star(new)
    for cid in ids + ["", "nope"]:
        pd.testing.assert_frame_equal(_old_star_frame_table(old, cid, keys), _star_frame_table(by, cid, keys),
                                      check_exact=True)


# ----------------------------------------------------------------------------------------------------------
# G09-29 / G09-27: ePSF candidate rows without iterrows; vectorised locus fit_ok; one locus read
# ----------------------------------------------------------------------------------------------------------
def _old_epsf_star_rows(csv_ok):
    ra_by, de_by, mg_by = {}, {}, {}
    mag_col = next((c for c in ("mag", "catalog_mag", "phot_g_mean_mag") if c in csv_ok.columns), None)
    if "ra_deg" in csv_ok.columns and "dec_deg" in csv_ok.columns:
        for _, r in csv_ok.iterrows():
            cid = str(r.get("_cid", "") or "").strip()
            if not cid:
                continue
            ra_v = float(pd.to_numeric(r.get("ra_deg"), errors="coerce"))
            de_v = float(pd.to_numeric(r.get("dec_deg"), errors="coerce"))
            if math.isfinite(ra_v) and math.isfinite(de_v):
                ra_by[cid] = ra_v
                de_by[cid] = de_v
            if mag_col is not None:
                mg = float(pd.to_numeric(r.get(mag_col), errors="coerce"))
                if math.isfinite(mg):
                    mg_by[cid] = mg
    rows = []
    for _, r in csv_ok.iterrows():
        cid = str(r.get("_cid", "") or "").strip()
        if not cid:
            continue
        x_f = float(pd.to_numeric(r.get("x"), errors="coerce"))
        y_f = float(pd.to_numeric(r.get("y"), errors="coerce"))
        if not (math.isfinite(x_f) and math.isfinite(y_f)):
            continue
        rows.append({"catalog_id": cid, "x": x_f, "y": y_f, "ra_deg": ra_by.get(cid, float("nan")),
                     "dec_deg": de_by.get(cid, float("nan")), "mag": mg_by.get(cid, float("nan"))})
    return rows


def test_g09_29_epsf_star_rows_equal():
    from psf_photometry import _epsf_star_rows

    rng = np.random.default_rng(13)
    n = 300
    df = pd.DataFrame({"_cid": [str(7 * 10**18 + int(i)) for i in rng.integers(0, 200, n)],
                       "x": rng.uniform(0, 2000, n), "y": rng.uniform(0, 1500, n),
                       "ra_deg": rng.uniform(0, 360, n), "dec_deg": rng.uniform(-80, 80, n),
                       "mag": rng.uniform(9, 16, n), "other": "a"})
    df.loc[3, "_cid"] = ""
    df.loc[5, "_cid"] = np.nan
    df.loc[7:12, "x"] = np.nan
    df.loc[20:30, "ra_deg"] = np.nan
    df.loc[31:35, "mag"] = np.nan
    df.loc[40, "y"] = "bad"
    assert repr(_old_epsf_star_rows(df)) == repr(_epsf_star_rows(df))
    d2 = df.drop(columns=["mag", "dec_deg"]).assign(catalog_mag=df["mag"])
    assert repr(_old_epsf_star_rows(d2)) == repr(_epsf_star_rows(d2))


def test_g09_29_locus_fit_ok_vectorised_equal():
    from psf_chi2_locus import chi2_ok_from_locus

    rng = np.random.default_rng(14)
    n = 5000
    conv = rng.random(n) > 0.1
    chi2 = rng.uniform(0, 5, n)
    chi2[rng.random(n) < 0.05] = np.nan
    chi2[rng.random(n) < 0.01] = np.inf
    resid = rng.normal(0, 3, n)
    resid[rng.random(n) < 0.05] = np.nan
    resid[:5] = 3.0  # == n_sigma
    ns = 3.0
    old = np.array([chi2_ok_from_locus(converged=bool(conv[i]),
                                       chi2=float(chi2[i]) if math.isfinite(chi2[i]) else float("nan"),
                                       resid_sigma=float(resid[i]) if math.isfinite(resid[i]) else float("nan"),
                                       n_sigma=ns) for i in range(n)])
    new = np.asarray(conv, dtype=bool) & np.isfinite(chi2) & np.isfinite(resid) & (resid <= float(ns))
    assert np.array_equal(old, new)


def test_g09_27_locus_single_read_same_frame(tmp_path):
    from gaia_catalog_id import read_vyvar_csv

    usecols = ["psf_flux", "psf_chi2", "psf_converged", "psf_fit_ok", "likely_saturated", "is_saturated"]
    rng = np.random.default_rng(15)
    p = tmp_path / "proc_a.csv"
    pd.DataFrame({"catalog_id": ["1", "2", "3"], "psf_chi2": rng.uniform(0, 2, 3), "x": 1.0,
                  "psf_flux": [1.0, np.nan, 3.0], "is_saturated": [True, False, False]}).to_csv(p, index=False)
    hdr = pd.read_csv(p, nrows=0).columns.tolist()
    old = read_vyvar_csv(p, usecols=[c for c in usecols if c in hdr], low_memory=False)
    new = read_vyvar_csv(p, usecols=lambda c: c in usecols, low_memory=False)
    pd.testing.assert_frame_equal(old, new, check_exact=True)


# ----------------------------------------------------------------------------------------------------------
# G09-48: TESS keeps only the cube of the sector that is localised (the first one with the most points)
# ----------------------------------------------------------------------------------------------------------
def test_g09_48_tess_keeps_only_localised_cube(tmp_path, monkeypatch):
    import tess_analysis as ta
    from config import AppConfig

    n_pts = [5, 9, 9, 3]
    auxs_seen = []

    def fake_sector(tpf, cid, sources):
        i = int(tpf)
        sec = ta.TessSector(i, 120.0, n_pts[i], 0.0, 1.0, 0.01, 4, 1.0)
        aux = {"t": np.arange(4.0) + 10 * i, "m": np.zeros(4), "e": np.ones(4), "cube": np.full((4, 2, 2), float(i)),
               "mask": np.ones((2, 2), bool), "xy": {}, "sigma": 1.0, "median": 0.0}
        auxs_seen.append(aux)
        return sec, aux

    best = ta.PeriodSolution("ls", 1.5, 0.01, 1e-5, 0.5)
    got = {}
    monkeypatch.setattr(ta, "sector_light_curve", fake_sector)
    monkeypatch.setattr(ta, "analyse_periods", lambda *a, **k: ({"best": best}, False, {"mult": 1}))
    monkeypatch.setattr(ta, "combine_sectors", lambda t, m, e, auxs, b: (best, 1))
    monkeypatch.setattr(ta, "localise", lambda cube, *a: (got.setdefault("cube", cube), ({}, None))[1])
    monkeypatch.setattr(ta, "_plots", lambda *a, **k: None)
    ta.run_tess("1", 10.0, 20.0, tmp_path, cfg=AppConfig(), sources=[], tpfs=[0, 1, 2, 3])
    assert float(got["cube"][0, 0, 0]) == 1.0  # the first sector with the most points, as max() picks it
    assert [("cube" in a) for a in auxs_seen] == [False, True, False, False]


# ----------------------------------------------------------------------------------------------------------
# G10-31: sorted magnitude windows == the O(N^2) scan (exact window predicate)
# ----------------------------------------------------------------------------------------------------------
def _old_window_stats(val, mag, use, mag_i, half=0.5, nmin=15):
    if not math.isfinite(mag_i):
        return float("nan"), float("nan")
    h = half
    while True:
        sel = use & np.isfinite(val) & (np.abs(mag - mag_i) <= h)
        if sel.sum() >= nmin or h > 5:
            break
        h *= 1.5
    if sel.sum() < 5:
        return float("nan"), float("nan")
    md = float(np.median(val[sel]))
    return md, 1.4826 * float(np.median(np.abs(val[sel] - md)))


def test_g10_31_sorted_windows_equal_scan():
    from variability_field import _MagWindows, _robust_z_by_mag

    rng = np.random.default_rng(16)
    n = 4000
    mag = np.round(rng.uniform(8, 19, n), 3)  # ties and window edges at exact 0.5 / 0.75 steps
    mag[:200] = 12.0 + 0.25 * rng.integers(0, 8, 200)
    mag[rng.random(n) < 0.01] = np.nan
    mag[5] = np.inf
    val = rng.normal(0, 1, n) * (1 + (mag - 8) / 5)
    val[rng.random(n) < 0.02] = np.nan
    use = rng.random(n) > 0.2
    use[mag > 18.5] = rng.random(int((mag > 18.5).sum())) > 0.9  # sparse faint end: widening
    z_old = np.full(n, np.nan)
    for i in range(n):
        if np.isfinite(val[i]):
            md, sc = _old_window_stats(val, mag, use, float(mag[i]))
            if math.isfinite(sc) and sc > 0:
                z_old[i] = (val[i] - md) / sc
    assert z_old.tobytes() == _robust_z_by_mag(val, mag, use).tobytes()
    win = _MagWindows(val, mag, use)
    for m_i in list(mag[:300]) + [12.5, 12.25, 30.0, float("nan")]:
        assert repr(win.stats(float(m_i))) == repr(_old_window_stats(val, mag, use, float(m_i)))


# ----------------------------------------------------------------------------------------------------------
# G11-28: report aggregates from light curves read once; G11-44: transient LC plotted from the table
# ----------------------------------------------------------------------------------------------------------
def _report_stub(tmp_path):
    from types import SimpleNamespace

    from photometry_report import _PhotometryReportBuilder
    from target_page import TargetPages

    rng = np.random.default_rng(17)
    lc_dir = tmp_path / "lightcurves"
    lc_dir.mkdir()
    for k in range(4):
        m = 30
        df = pd.DataFrame({"source_file": [f"proc_Frame_{j:03d}.csv" for j in range(k, k + m)],
                           "jd": 2461000.3 + np.arange(m) * 0.004 + k, "mag_calib": rng.normal(12, 0.01, m),
                           "flag": rng.choice(["normal", "artifact", "frame_qc", "high_err", "saturated"], m)})
        if k == 1:
            df["bjd"] = df["jd"] + 0.0007
            df["airmass"] = rng.uniform(1.0, 1.9, m)
            df.loc[2, "jd"] = np.nan
        if k == 2:
            df = df.drop(columns=["flag", "jd"])
        df.to_csv(lc_dir / f"lightcurve_{4 * 10**18 + k}.csv", index=False)
    (lc_dir / "lightcurve_1_psf.csv").write_text("x\n1\n")
    (lc_dir / "lightcurve_5.csv").write_text('a,"b\n1,2\n')  # broken file
    pages = SimpleNamespace(lc_dir=lc_dir, _lc={})
    pages.lc_df = lambda cid: TargetPages.lc_df(pages, cid)
    b = object.__new__(_PhotometryReportBuilder)
    b.lc_dir = lc_dir
    b._pages = pages
    summ = pd.DataFrame({"lc_quality_flag": ["good", "noisy", "good"], "n_frames": [3, 3, 3]})
    b.summary_df = summ
    b.measured_summary_df = summ
    b.n_lc = 3
    return b, lc_dir


def test_g11_28_report_lc_aggregates_equal_old(tmp_path):
    from collections import Counter

    b, lc_dir = _report_stub(tmp_path)
    files = b._lc_files()
    # old readers (HEAD 08e985a), one read per aggregate
    lo, hi = float("inf"), float("-inf")
    for p in files:
        try:
            jd = pd.to_numeric(pd.read_csv(p, usecols=["jd"])["jd"], errors="coerce").dropna()
        except (OSError, ValueError):
            continue
        if len(jd):
            lo, hi = min(lo, float(jd.min())), max(hi, float(jd.max()))
    assert b._measured_jd_range() == (lo, hi)
    flags = Counter()
    for p in files:
        try:
            fl = pd.read_csv(p, usecols=["flag"])["flag"]
        except (OSError, ValueError, KeyError):
            continue
        flags.update(fl.astype(str).str.strip().str.lower().tolist())
    disp = b._format_lc_count_display()
    for key in ("artifact", "frame_qc", "high_err", "saturated"):
        assert f"{flags[key]}" in disp
    stems = set()
    for p in files:
        try:
            sf = pd.read_csv(p, usecols=["source_file"])["source_file"].dropna().astype(str)
        except (OSError, ValueError, KeyError):
            continue
        stems.update(b._frame_stem(x) for x in sf.unique())
    assert b._measured_frame_stems() == stems
    vals = []
    for p in files:
        try:
            d = pd.read_csv(p, usecols=lambda c: c in {"bjd", "bjd_tdb", "hjd", "jd"}, low_memory=False)
        except Exception:  # noqa: BLE001
            continue
        xc = next((x for x in ("bjd_tdb", "bjd", "hjd", "jd") if x in d.columns), None)
        if xc:
            vals.extend(float(x) for x in pd.to_numeric(d[xc], errors="coerce").dropna() if np.isfinite(x))
    out = b._add_lc_bjd_span({})
    assert (out["bjd_min"], out["bjd_max"]) == (min(vals), max(vals))
    assert len(b._pages._lc) == len(files)  # every light curve read once, cached for the pages


def test_g11_44_transient_lc_png_from_table_equal_csv(tmp_path):
    from variability_plots import candidate_lc_png

    rng = np.random.default_rng(18)
    n = 80
    d = pd.DataFrame({"time": 2461000.3 + np.sort(rng.uniform(0, 0.3, n)), "mag": rng.normal(15.2, 0.05, n),
                      "snr": rng.uniform(2, 40, n), "err": rng.uniform(0.01, 0.1, n), "fwhm": rng.uniform(3, 6, n),
                      "airmass": rng.uniform(1.1, 1.6, n)})
    d = d.assign(dmag=d["mag"].where(d["snr"] >= 5))
    tmp = tmp_path / "t_lc.csv"
    d.to_csv(tmp, index=False)
    a = candidate_lc_png(tmp, tmp_path / "a.png", "T1 (new), RA 1.0 Dec +2.0", time_label="JD", ylabel="G (est.)")
    b = candidate_lc_png(d, tmp_path / "b.png", "T1 (new), RA 1.0 Dec +2.0", time_label="JD", ylabel="G (est.)")
    assert a.read_bytes() == b.read_bytes()


# ----------------------------------------------------------------------------------------------------------
# G14-26 / G14-27 / G14-29: UI caches keyed on hashed file stamps; cut-out before float; deferred downloads
# ----------------------------------------------------------------------------------------------------------
_UI_FILES = ("ui_hrd.py", "ui_variability.py", "ui_masterstar_qa.py", "ui_quality_dashboard.py", "ui_epsf_dashboard.py",
             "ui_aperture_photometry.py")


def test_g14_ui_cache_stamps_are_hashed_arguments():
    import ast

    src = Path(__file__).resolve().parents[2] / "src_py"
    n = 0
    for name in _UI_FILES:
        for node in ast.walk(ast.parse((src / name).read_text(encoding="utf-8"))):
            if not isinstance(node, ast.FunctionDef):
                continue
            deco = [ast.unparse(d) for d in node.decorator_list]
            if not any("cache_data" in d or "cache_resource" in d for d in deco):
                continue
            n += 1
            for a in node.args.args:
                assert not (a.arg.startswith("_") and "stamp" in a.arg), (name, node.name, a.arg)
    assert n >= 7


def test_g14_26_cached_preview_follows_file_state(tmp_path, monkeypatch):
    import os

    import ui_quality_dashboard as uq
    from ui_cache import dir_stamp, file_stamp

    p = tmp_path / "f.fits"
    p.write_bytes(b"1")
    calls = []
    monkeypatch.setattr(uq, "quicklook_preview_png_bytes", lambda path: (calls.append(path), b"png")[1])
    uq._preview_png_cached.clear()
    assert uq._preview_png_cached(str(p), file_stamp(p)) == b"png"
    assert uq._preview_png_cached(str(p), file_stamp(p)) == b"png"
    assert len(calls) == 1
    st0 = p.stat()
    os.utime(p, ns=(st0.st_atime_ns, st0.st_mtime_ns + 10**9))
    uq._preview_png_cached(str(p), file_stamp(p))
    assert len(calls) == 2  # a rewritten file is read again
    assert file_stamp(tmp_path / "none")[0][1:] == (-1, -1) and len(dir_stamp(tmp_path)) == 1


def test_g14_27_cutout_slice_then_float_equal(tmp_path):
    from astropy.io import fits

    rng = np.random.default_rng(19)
    for dt in (np.float32, np.int16):
        p = tmp_path / f"c_{np.dtype(dt).name}.fits"
        fits.PrimaryHDU((rng.normal(500, 50, (300, 400))).astype(dt)).writeto(p)
        for x, y in ((5, 7), (200, 150), (399, 299)):
            h = 15
            sl = (slice(max(0, y - h), y + h + 1), slice(max(0, x - h), x + h + 1))
            old = fits.getdata(p).astype(float)[sl]
            new = fits.getdata(p)[sl].astype(float)
            assert old.dtype == new.dtype and old.tobytes() == new.tobytes()


def test_g14_27_hrd_hover_texts_equal_per_row():
    from ui_hrd import _hover, _side_hover, _texts

    df = pd.DataFrame({"catalog_id": ["1", "2", "3"], "bp_rp": [0.5, 1.5, np.nan], "abs_mag_g": [4.0, 0.0, np.nan],
                       "phot_g_mean_mag": [10.0, 9.0, 14.0], "teff_gspphot": [5800.0, np.nan, np.nan],
                       "parallax": [5.0, 1.0, np.nan], "parallax_over_error": [50.0, 10.0, np.nan],
                       "vsx_name": [np.nan, "XX Cvn", np.nan], "vsx_type": [np.nan, "EW", np.nan],
                       "measured": [True, False, False]})
    assert _texts(df, _hover).tolist() == [_hover(r) for _, r in df.iterrows()]
    assert _texts(df, _side_hover).tolist() == [_side_hover(r) for _, r in df.iterrows()]


def test_g14_29_download_data_is_deferred():
    import ast

    src = Path(__file__).resolve().parents[2] / "src_py"
    n = 0
    for name in ("ui_aperture_photometry.py", "ui_epsf_dashboard.py", "ui_variability.py"):
        for node in ast.walk(ast.parse((src / name).read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "download_button":
                data = next((k.value for k in node.keywords if k.arg == "data"), node.args[1] if len(node.args) > 1 else None)
                n += 1
                # a bound method / partial (called on click), not the bytes / text built on every rerun
                assert not (isinstance(data, ast.Call) and getattr(data.func, "attr", "") != "partial"), ast.unparse(data)
    assert n == 4


# COMP-CEIL-NOISEMODEL-2026-10: build_global_comp_pool requires the rig scintillation parameters; none here
# (no telescope diameter / site altitude -> scintillation 0, as sigma_budget does for such a rig)
from sigma_budget import RigScintillationParams  # noqa: E402

_NO_SCINT = RigScintillationParams(None, "test", float("nan"), float("nan"), 60.0, 1.5, [])
