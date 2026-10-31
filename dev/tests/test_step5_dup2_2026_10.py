"""Step 5 follow-up s5_dup2 (2026-10-30): G01-11, G01-17, G01-23, G07-07, G03-43, G04-20."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_py"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # shared rigs of earlier step tests

import calib_library as cl  # noqa: E402
import cal_diag  # noqa: E402
from database import VyvarDatabase  # noqa: E402

T0 = datetime(2026, 3, 15, 22, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------------------------- G01-11
def _master_dark(path: Path, level: float, exp: float, *, seed: int, imagetyp: str = "Dark Frame") -> Path:
    rng = np.random.default_rng(seed)
    a = level + rng.normal(0.0, 1.7, (40, 50))
    h = fits.Header()
    h["IMAGETYP"] = imagetyp
    h["EXPTIME"] = exp
    h["DATE-OBS"] = T0.strftime("%Y-%m-%dT%H:%M:%S")
    h["XBINNING"] = 1
    h["YBINNING"] = 1
    h["CCD-TEMP"] = -10.0
    h["SET-TEMP"] = -10.0
    h["GAIN"] = 0
    h["OFFSET"] = 10
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(path, a.astype(np.float32), header=h, overwrite=True)
    return path


@pytest.fixture()
def lib_db(tmp_path: Path):
    db = VyvarDatabase(tmp_path / "t.sqlite3")
    cam = db.insert_equipment("C5A-150M", "cam", "CMOS", "x", 3.76)
    cl.ensure_schema(db)
    return db, cam, tmp_path / "lib"


def _register(db, cam, path: Path, kind: str) -> None:
    cl.register_master(db, path, kind=kind, camera_id=cam, telescope_id=None, origin=cl.ORIGIN_VYVAR, combine_method="median")


def _old_measure_offset_polyfit(paths_exps: list[tuple[str, float]]) -> float:
    """Reference: the pre-G01-11 calib_library intercept (np.polyfit over nanmedian levels)."""
    t = np.array([e for _p, e in paths_exps], dtype=np.float64)
    y = np.array([float(np.nanmedian(fits.getdata(p).astype(np.float64))) for p, _e in paths_exps])
    _slope, icpt = np.polyfit(t, y, 1)
    return float(icpt)


def test_g0111_darkflat_is_not_a_pedestal_point(lib_db):
    """Old code: measure_offset_adu took DARK + DARKFLAT masters; now DARK only (Milan: no dark-flats)."""
    db, cam, lib = lib_db
    d60 = _master_dark(lib / "d60.fits", 500.0 + 0.5 * 60, 60.0, seed=1)
    d120 = _master_dark(lib / "d120.fits", 500.0 + 0.5 * 120, 120.0, seed=2)
    df = _master_dark(lib / "df1.fits", 520.0, 1.0, seed=3, imagetyp="Dark Flat")  # off the line
    _register(db, cam, d60, cl.KIND_DARK)
    _register(db, cam, d120, cl.KIND_DARK)
    _register(db, cam, df, cl.KIND_DARKFLAT)
    est = cl.measure_offset_adu(db, camera_id=cam, gain=0, offset=10, binning=1)
    assert est.n_darks == 2 and str(df) not in est.darks
    assert est.adu == pytest.approx(500.0, abs=0.3)
    assert est.sigma_adu is not None and est.sigma_adu > 0


def test_g0111_one_estimator_for_library_and_cal_diag(lib_db):
    """calib_library.measure_offset_adu and cal_diag.derive_pedestal_from_masters give the same P / sigma_P."""
    db, cam, lib = lib_db
    paths = [
        _master_dark(lib / "d30.fits", 500.0 + 0.5 * 30, 30.0, seed=4),
        _master_dark(lib / "d90.fits", 500.0 + 0.5 * 90, 90.0, seed=5),
        _master_dark(lib / "d300.fits", 500.0 + 0.5 * 300, 300.0, seed=6),
    ]
    for p in paths:
        _register(db, cam, p, cl.KIND_DARK)
    est = cl.measure_offset_adu(db, camera_id=cam, gain=0, offset=10, binning=1)
    ped = cal_diag.derive_pedestal_from_masters(paths)
    assert ped.method == "INTERCEPT"
    assert est.adu == ped.p_adu and est.sigma_adu == ped.sigma_p
    # Milan: the last bits of the flat offset may change (np.polyfit -> the OLS formula). Measured here:
    old = _old_measure_offset_polyfit([(str(p), e) for p, e in zip(paths, (30.0, 90.0, 300.0))])
    print(f"G01-11 flat offset: OLS {est.adu!r} vs old polyfit {old!r}, diff {est.adu - old:.3g} ADU")
    assert abs(est.adu - old) < 1e-9 * abs(old)


def test_g0111_single_exposure_is_the_shortest_dark_with_warning(tmp_path, monkeypatch):
    """Old cal_diag took the first dark found; now the shortest dark, logged as WARNING, recorded as method."""
    events: list[str] = []
    monkeypatch.setattr(cl, "log_event", lambda m: events.append(str(m)))
    long_ = _master_dark(tmp_path / "a_long.fits", 600.0, 120.0, seed=7)
    no_exp = _master_dark(tmp_path / "b_noexp.fits", 700.0, 0.0, seed=8)
    ped = cal_diag.derive_pedestal_from_masters([no_exp, long_])
    assert ped.method == "SINGLE_MASTER_MEDIAN"
    assert ped.p_adu == pytest.approx(600.0, abs=0.3)
    assert any(e.startswith("WARNING pedestal") and "a_long.fits" in e for e in events)
    fit = cl.fit_pedestal([(120.0, 600.0, 0.1, "x120"), (60.0, 560.0, 0.1, "x60b"), (60.0, 559.0, 0.1, "x60a")])
    assert fit is not None and fit.method == cl.PEDESTAL_INTERCEPT
    one = cl.fit_pedestal([(60.0, 560.0, 0.1, "b"), (60.0, 559.0, 0.1, "a")])
    assert one is not None and one.method == cl.PEDESTAL_SHORTEST_DARK and one.p_adu == 559.0


def test_g0111_gate_result_records_the_pedestal_method():
    assert "pedestal_method" in cal_diag.CalDiagGateResult.__dataclass_fields__


# ---------------------------------------------------------------------------------------------- G01-17
def test_g0117_ram_qc_calibrates_with_the_flat_offsets_and_substitutes_of_the_run(tmp_path, monkeypatch):
    """Old code: the RAM-QC path called the in-RAM calibration without flat_offsets / flat_substitutes."""
    import pipeline_calibrate as pc
    import test_step4_t01_2026_10 as t01

    db, cam, tel, loc, data_root = t01._rig(tmp_path)
    lib = data_root / "CalibrationLibrary"
    t01._register_dark(db, cam, lib)
    rng = np.random.default_rng(6)
    flat = t01._register_flat(db, cam, tel, lib, "V",
                              (20000.0 * (1 + rng.normal(0, 0.004, (32, 48)))).astype(np.float32))
    src = tmp_path / "night" / "Lights"
    for i in range(2):
        t01._write(src / f"V_{i}.fits", np.full((32, 48), 3000.0), t01._light_hdr("V"))
    plan, res = t01._scan_import(db, cam, tel, loc, data_root, src)

    seen: list[dict] = []
    real = pc._calibrate_one_light_apply_masters_in_ram

    def _spy(**kw):
        seen.append({"flat_offsets": kw.get("flat_offsets"), "flat_substitutes": kw.get("flat_substitutes")})
        return real(**kw)

    monkeypatch.setattr(pc, "_calibrate_one_light_apply_masters_in_ram", _spy)
    og = next(iter(plan.masterflat_by_obs_key))
    out = pc.run_draft_ram_calibration_qc_to_obs_files(
        db=db, draft_id=int(res.draft_id), archive_path=res.archive_path, master_dark_path=None,
        masterflat_by_filter={}, masterflat_by_obs_key=dict(plan.masterflat_by_obs_key),
        master_dark_by_obs_key=dict(plan.dark_master_by_obs_key), equipment_id=cam, pedestal_temp_tol_c=0.5,
        flat_substitute_by_obs_key={og: "2026-01-02"})
    assert out["errors"] == []
    want = pc._flat_offsets_for_masters([flat], db)
    assert want and len(seen) == 2
    assert all(s["flat_offsets"] == want and s["flat_substitutes"] == {og: "2026-01-02"} for s in seen)


# ---------------------------------------------------------------------------------------------- G01-23
def _old_prepare_flat_global(flat: np.ndarray) -> tuple[np.ndarray, float]:
    """Reference: prepare_flat_for_division's own normalisation for flats without VYFLNRD at s5_merge."""
    flat = np.asarray(flat, dtype=np.float32)
    valid = np.isfinite(flat) & (flat > 0)
    fm = float(np.median(flat[valid]))
    return (flat / fm).astype(np.float32), fm


def _old_normalise_flat(a: np.ndarray, pattern: str | None) -> tuple[np.ndarray, list[float]]:
    """Reference: calib_masters.normalise_flat at s5_merge (nanmedian over all pixels, <= 0 included)."""
    import math

    out = np.asarray(a, dtype=np.float32).copy()
    if pattern:
        meds = []
        for dy in (0, 1):
            for dx in (0, 1):
                v = out[dy::2, dx::2]
                m = float(np.nanmedian(v))
                meds.append(m)
                if math.isfinite(m) and m > 0:
                    out[dy::2, dx::2] = v / m
        return out, meds
    m = float(np.nanmedian(out))
    if math.isfinite(m) and m > 0:
        out /= m
    return out, [m]


def _flat_with_zero_rows(seed: int = 123, osc: bool = False, dead_frac: float | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = 400, 600
    yy, xx = np.mgrid[0:h, 0:w]
    vign = 1.0 - 0.25 * (((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2)
    a = (20000.0 * vign * rng.normal(1, 0.01, (h, w))).astype(np.float32)
    if osc:
        a[0::2, 1::2] *= 1.6
        a[1::2, 0::2] *= 1.55
        a[1::2, 1::2] *= 0.7
    if dead_frac is None:
        a[::20, :] = 0.0  # 5 % zero rows
    else:
        a[rng.random(a.shape) < dead_frac] = 0.0  # scattered dead pixels
    a[5, 7] = np.nan
    return a


def test_g0123_flat_without_vyflnrd_uses_the_one_normaliser_bit_identical():
    from calibration import ProcessedMasterResult, prepare_flat_for_division

    for osc in (False, True):
        a = _flat_with_zero_rows(7, osc)
        ref, fm_ref = _old_prepare_flat_global(a)
        new, fm_new, _nbad = prepare_flat_for_division(ProcessedMasterResult(a, 1, False, 1), cfa=False, name="x")
        ref_nan = ref.copy()
        from calibration import flat_bad_pixel_mask

        ref_nan[flat_bad_pixel_mask(ref, cfa=False)] = np.nan
        assert fm_new == fm_ref
        assert new.dtype == ref_nan.dtype and np.array_equal(new, ref_nan, equal_nan=True)


def test_g0123_one_median_rule_changes_adu_by_0p1_percent_not_magnitudes():
    """Build / VYFLNRD normaliser: median of the valid pixels (old: zeros included). The calibrated ADU move by
    one global factor per channel (+0.2 / +0.5 % here with 5 % zero rows; real flats
    have far fewer pixels <= 0); differential magnitudes stay equal within float noise."""
    import calib_masters as cm

    # mono: 5 % zero rows (extreme). OSC: 0.5 % scattered dead pixels - per-channel factors that differ between
    # channels are not a global scale on the CFA mosaic; with 5 % zero rows on the even rows only the
    # differential magnitudes of this test move by up to 1.1 mmag (reported to Milan).
    for pat, dead in ((None, None), ("RGGB", 0.005)):
        a = _flat_with_zero_rows(11, osc=pat is not None, dead_frac=dead)
        old, old_m = _old_normalise_flat(a, pat)
        new, new_m = cm.normalise_flat(a, pat)
        # each channel: one factor (old median / new median), the same on every pixel
        ok = np.isfinite(a) & (a > 0)
        for i, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1)) if pat else ((0, 0),)):
            st = 2 if pat else 1
            sub = (slice(dy, None, st), slice(dx, None, st))
            r = new[sub][ok[sub]].astype(np.float64) / old[sub][ok[sub]].astype(np.float64)
            assert np.allclose(r, old_m[i] / new_m[i], rtol=2e-7, atol=0)
            # the zeros pulled the old median down (zero rows: even rows only, the odd-row channels keep 1)
            if dead is None:
                assert (1.0005 < new_m[i] / old_m[i] < 1.01) if dy == 0 else (new_m[i] == old_m[i])
            else:
                assert 1.0 < new_m[i] / old_m[i] < 1.001
        # differential magnitudes on a synthetic light: star fluxes x flat, then divided by each normalised flat
        rng = np.random.default_rng(5)
        light = (300.0 + rng.normal(0, 3.0, a.shape)).astype(np.float32)
        yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
        stars = [(101.3, 203.7, 5.0e4), (251.8, 402.2, 2.0e4), (155.5, 520.5, 8.0e3)]
        for y0, x0, f in stars:
            light += (f / (2 * np.pi * 2.0**2) * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 2.0**2))).astype(np.float32)
        light *= np.where(ok, a / np.nanmedian(a[ok]), 1.0).astype(np.float32)

        def _mags(flat):
            cal = light / flat
            fl = []
            for y0, x0, _f in stars:
                r2 = (xx - x0) ** 2 + (yy - y0) ** 2
                ap = (r2 <= 8.0**2) & np.isfinite(cal)
                ann = (r2 >= 12.0**2) & (r2 <= 18.0**2) & np.isfinite(cal)
                fl.append(float(np.sum(cal[ap].astype(np.float64)) - ap.sum() * np.mean(cal[ann].astype(np.float64))))
            fl = np.array(fl)
            return -2.5 * np.log10(fl[1:] / fl[0])

        with np.errstate(divide="ignore", invalid="ignore"):
            f_old = np.where(ok, old, np.nan).astype(np.float32)
            f_new = np.where(ok, new, np.nan).astype(np.float32)
            d = _mags(f_new) - _mags(f_old)
        print(f"G01-23 pattern={pat}: flat factors {[round(n / o, 5) for n, o in zip(new_m, old_m)]}, max |dmag| {np.max(np.abs(d)):.2g}")
        assert np.all(np.abs(d) < 1e-5), d  # float32 noise only


# ---------------------------------------------------------------------------------------------- G07-07
def test_g0707_db_colour_wins_everywhere_csv_fills_only_what_the_db_lacks(tmp_path):
    """Old code: comps / active targets kept a CSV bp_rp even when the Gaia DB had one (one target: DB won)."""
    import pandas as pd

    import photometry_comp as pc
    from test_step5_dup_2026_10 import _bprp_gaia_db

    db, sids, truth = _bprp_gaia_db(tmp_path, n=60)
    with_c = [s for s in sids if truth[s] is not None][:3]
    no_c = [s for s in sids if truth[s] is None][:1]
    ids = with_c + no_c + ["999999999999", ""]
    csv = [9.0, np.nan, 9.0, 8.0, 7.0, 6.0]
    df = pd.DataFrame({"catalog_id": ids, "bp_rp": csv})
    want = [truth[with_c[0]], truth[with_c[1]], truth[with_c[2]], 8.0, 7.0, 6.0]
    for out in (pc._enrich_comp_bp_rp(df, db),
                pc._enrich_comp_bp_rp(df, db, gaia_prefetch=pc.gaia_bp_rp(ids[:2], db)),
                pc._enrich_active_targets_bp_rp(df, gaia_db_path=db)):
        assert out["bp_rp"].tolist() == want
    for i, w in enumerate(want):
        row = df.iloc[i].copy()
        assert pc._enrich_target_bp_rp_from_gaia_db(row, gaia_db_path=db)["bp_rp"] == w
    # no DB: every source keeps its own value
    assert pc._enrich_comp_bp_rp(df, None)["bp_rp"].equals(df["bp_rp"])


# ---------------------------------------------------------------------------------------------- G03-43
def test_g0343_nan_pass1_centroid_no_longer_disables_the_dedup():
    """Old code: one NaN pass-1 centroid made NO pass-2 detection a duplicate; now it is dropped."""
    from pipeline_catalog import _pass2_within_dedup_of_pass1

    rng = np.random.default_rng(343)
    p1x, p1y = rng.uniform(0, 4000, 5000), rng.uniform(0, 2800, 5000)
    near = rng.integers(0, 5000, 300)
    qx = np.concatenate([p1x[near] + 1.0, rng.uniform(0, 4000, 700)])
    qy = np.concatenate([p1y[near] - 1.0, rng.uniform(0, 2800, 700)])
    clean = _pass2_within_dedup_of_pass1(p1x, p1y, qx, qy, 3.0)
    p1x[17] = np.nan
    p1y[4000] = np.inf
    got = _pass2_within_dedup_of_pass1(p1x, p1y, qx, qy, 3.0)
    keep = np.ones(p1x.size, dtype=bool)
    keep[[17, 4000]] = False
    ref = np.array([bool(np.min(np.hypot(p1x[keep] - x, p1y[keep] - y)) < 3.0) for x, y in zip(qx, qy)])
    assert np.array_equal(got, ref)
    # synthetic count: 300 pass-2 detections 1.4 px from a pass-1 star (+ chance hits among 700 random ones)
    n_dup = int(got.sum())
    assert n_dup >= 298 and n_dup == int(clean.sum()) - int((clean & ~got).sum())
    print(f"G03-43 synthetic: {n_dup} of {qx.size} pass-2 detections are duplicates (old code with a NaN: 0)")


# ---------------------------------------------------------------------------------------------- G04-20
def test_g0420_every_label_path_fails_loud_on_a_non_unique_index(monkeypatch):
    """The G04-20 paths that address one row per label / position check it the same way (ValueError)."""
    import pandas as pd

    import masterstar_gaia_accounting as mga
    from test_step5_perf1_2026_10 import _census_case

    # masterstars: checked before the first write (identity locks), not only in the census step
    gdf, owner, det_pass, out = _census_case(3)
    dup = pd.concat([out, out])
    with pytest.raises(ValueError, match="masterstars table index is not unique"):
        mga.enrich_masterstar_gaia_complete(dup, gaia_on_chip=gdf, wpx=1000, h=800, fwhm_px=3.5, target_depth_g=17.0)

    # Gaia census: old code wrote census.loc[label] on every row sharing the label
    real = mga.build_gaia_census_rows

    def _dup_census(*a, **k):
        c = real(*a, **k)
        return c.set_axis([0] * len(c))

    monkeypatch.setattr(mga, "build_gaia_census_rows", _dup_census)
    kw = dict(det_pass=det_pass, edge_m=40.0, fwhm_px=3.5, gaia_owner=owner, gdf=gdf, h=800,
              n_g=len(gdf), target_depth_g=17.0, wpx=1000)
    with pytest.raises(ValueError, match="Gaia census table index is not unique"):
        mga._acc_census(meta={}, out=out.copy(), **kw)

    # check_ambiguous_owner: the caller's Gaia x/y must be the positions of gaia_df
    gx = pd.to_numeric(gdf["x_gaia"]).to_numpy(dtype=np.float64)
    gy = pd.to_numeric(gdf["y_gaia"]).to_numpy(dtype=np.float64)
    assert mga.check_ambiguous_owner(100.0, 100.0, 5, gdf, gxy=(gx, gy)) == mga.check_ambiguous_owner(100.0, 100.0, 5, gdf)
    with pytest.raises(ValueError, match="gxy has"):
        mga.check_ambiguous_owner(100.0, 100.0, 5, gdf, gxy=(gx[:-1], gy[:-1]))
