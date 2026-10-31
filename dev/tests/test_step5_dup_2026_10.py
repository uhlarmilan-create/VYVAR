"""Step 5 batch C (tag dup): merged duplicate implementations.

Each test compares the surviving implementation with a verbatim copy of the removed one
(reference functions prefixed ``_old_``) on realistic synthetic input - bit-identical results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src_py"
sys.path.insert(0, str(SRC))


# --- G02-26: one skimage estimator, one minimum-pairs rule -----------------------------------------


def _old_similarity_estimate(src: np.ndarray, dst: np.ndarray):
    from skimage.transform import SimilarityTransform

    if hasattr(SimilarityTransform, "from_estimate"):
        tt = SimilarityTransform.from_estimate(src, dst)
        ok = bool(tt)
    else:
        tt = SimilarityTransform()
        ok = bool(tt.estimate(src, dst))
    if not ok or not np.all(np.isfinite(np.asarray(tt.params, dtype=np.float64))):
        return None
    return tt


def test_g02_26_similarity_estimate_equal() -> None:
    from skimage.transform import SimilarityTransform

    import vyvar_alignment_frame as vaf

    assert not hasattr(vaf, "_similarity_estimate")
    rng = np.random.default_rng(226)
    src = rng.uniform(0, 2000, size=(150, 2))
    th = np.deg2rad(0.7)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    dst = 1.0003 * src @ rot.T + np.array([12.3, -7.9]) + rng.normal(0, 0.2, size=src.shape)
    old = _old_similarity_estimate(src, dst)
    new = vaf._estimate(SimilarityTransform, src, dst)
    assert np.array_equal(np.asarray(old.params), np.asarray(new.params))


def test_g02_26_poly_min_pairs_matches_both_old_formulas() -> None:
    from vyvar_alignment_frame import poly_min_pairs

    for order in range(1, 8):
        k_poly = (order + 1) * (order + 2)
        assert poly_min_pairs(order) == 3 * (order + 1) * (order + 2) // 2 == 3 * k_poly // 2
    src = (SRC / "vyvar_alignment_frame.py").read_text(encoding="utf-8")
    assert "3 * k_poly // 2" not in src and "3 * (order + 1) * (order + 2) // 2" not in src


# --- G02-33: one VY_AM reader -----------------------------------------------------------------------


def _vy_am_header(m: np.ndarray):
    from astropy.io import fits

    h = fits.Header()
    for k, v in enumerate(np.asarray(m, dtype=np.float64).ravel()):
        h[f"VY_AM{k}"] = float(v)
    return h


def test_g02_33_alignment_matrix_equals_old_parsers() -> None:
    from badpix_sidecar import repaired_xy_on_reference
    from frame_footprint import alignment_matrix
    from sat_diag import alignment_inverse_from_header

    m = np.array([[0.99998, -0.00612, 13.25], [0.00611, 0.99997, -4.125], [1.2e-9, -3.0e-9, 1.0]])
    h = _vy_am_header(m)
    old_tf = np.array([float(h[f"VY_AM{j}"]) for j in range(9)]).reshape(3, 3)  # transient_field copy
    assert np.array_equal(alignment_matrix(h), old_tf)
    old_inv = np.linalg.inv(np.asarray([float(h[f"VY_AM{k}"]) for k in range(9)], dtype=np.float64).reshape(3, 3))
    assert np.array_equal(alignment_inverse_from_header(h), old_inv)
    bp = np.array([[10.0, 20.0, 0.0], [1500.5, 900.25, 1.0], [3000.0, 2000.0, 0.0]])
    p = old_tf @ np.c_[bp[:, 0], bp[:, 1], np.ones(len(bp))].T
    old_rep = [[round(float(u), 2), round(float(v), 2), int(k)] for u, v, k in zip(p[0] / p[2], p[1] / p[2], bp[:, 2], strict=True)]
    assert repaired_xy_on_reference(bp, h) == old_rep


def test_g02_33_bad_headers_fail_loud_or_none() -> None:
    from astropy.io import fits

    from frame_footprint import alignment_matrix
    from sat_diag import alignment_inverse_from_header

    h = _vy_am_header(np.eye(3))
    del h["VY_AM4"]
    with pytest.raises(ValueError):
        alignment_matrix(h)
    assert alignment_inverse_from_header(h) is None
    h2 = {f"VY_AM{k}": float(v) for k, v in enumerate(np.eye(3).ravel())}  # astropy refuses NaN cards
    h2["VY_AM2"] = float("nan")
    with pytest.raises(ValueError):
        alignment_matrix(h2)
    assert alignment_inverse_from_header(h2) is None
    assert alignment_inverse_from_header(fits.Header()) is None
    for name in ("sat_diag.py", "transient_field.py", "badpix_sidecar.py"):
        assert 'f"VY_AM{' not in (SRC / name).read_text(encoding="utf-8"), name


def test_g02_33_frame_footprints_warns_on_missing_transform(tmp_path: Path, caplog) -> None:
    from astropy.io import fits

    from frame_footprint import _frame_footprints

    good = _vy_am_header(np.eye(3))
    fits.PrimaryHDU(np.zeros((20, 30), np.float32), header=good).writeto(tmp_path / "a.fits")
    fits.PrimaryHDU(np.zeros((20, 30), np.float32)).writeto(tmp_path / "b.fits")
    with caplog.at_level("WARNING"):
        quads = _frame_footprints(tmp_path, ["a", "b"])
    assert len(quads) == 1
    assert np.array_equal(quads[0], np.array([[0.0, 0.0], [29.0, 0.0], [29.0, 19.0], [0.0, 19.0]]))
    assert "without a usable VY_AM" in caplog.text


# --- G02-44: one Gaia (ra, dec) lookup by source_id; vectorised table map --------------------------


def _old_gaia_radec_map_from_table(df, *, id_col="catalog_id", ra_col="ra_deg", dec_col="dec_deg"):
    import math

    import pandas as pd

    from gaia_catalog_id import normalize_gaia_source_id

    out: dict = {}
    if df is None or getattr(df, "empty", True):
        return out
    cid = df[id_col] if id_col in df.columns else df.get("source_id")
    if cid is None:
        return out
    ra = pd.to_numeric(df[ra_col] if ra_col in df.columns else df.get("ra"), errors="coerce")
    de = pd.to_numeric(df[dec_col] if dec_col in df.columns else df.get("dec"), errors="coerce")
    for i in range(len(df)):
        k = normalize_gaia_source_id(str(cid.iloc[i] if hasattr(cid, "iloc") else ""))
        if not k:
            continue
        try:
            rv = float(ra.iloc[i])
            dv = float(de.iloc[i])
        except Exception:  # noqa: BLE001
            continue
        if math.isfinite(rv) and math.isfinite(dv):
            out[k] = (rv, dv)
    return out


def _gaia_db(tmp_path: Path, n: int = 900) -> tuple[str, list[str], np.ndarray]:
    import sqlite3

    rng = np.random.default_rng(244)
    sids = [str(4000000000000000000 + 7919 * i) for i in range(n)]
    radec = np.c_[rng.uniform(200, 210, n), rng.uniform(40, 45, n)]
    p = tmp_path / "gaia.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE gaia_dr3 (source_id INTEGER, ra REAL, dec REAL)")
    con.executemany("INSERT INTO gaia_dr3 VALUES (?,?,?)", [(int(s), float(a), float(b)) for s, (a, b) in zip(sids, radec, strict=True)])
    con.commit()
    con.close()
    return str(p), sids, radec


def test_g02_44_table_map_equal() -> None:
    import pandas as pd

    from wcs_invertibility import gaia_radec_map_from_table

    rng = np.random.default_rng(44)
    n = 500
    cid = [str(3000000000000000000 + i % 450) for i in range(n)]  # repeated ids: the later row wins
    cid[3] = ""
    cid[7] = "nan"
    cid[11] = "Gaia DR3 3000000000000000011"
    ra = rng.uniform(0, 360, n)
    ra[5] = np.nan
    de = rng.uniform(-30, 80, n)
    de[9] = np.inf
    df = pd.DataFrame({"catalog_id": cid, "ra_deg": ra, "dec_deg": de})
    old = _old_gaia_radec_map_from_table(df)
    new = gaia_radec_map_from_table(df)
    assert list(old.items()) == list(new.items())
    df2 = pd.DataFrame({"source_id": cid, "ra": ra.astype(object), "dec": de})
    assert list(_old_gaia_radec_map_from_table(df2).items()) == list(gaia_radec_map_from_table(df2).items())
    assert gaia_radec_map_from_table(pd.DataFrame()) == {}


def test_g02_44_one_lookup_reused(tmp_path: Path, monkeypatch) -> None:
    import pandas as pd
    from astropy.wcs import WCS

    import wcs_invertibility as wi

    db, sids, radec = _gaia_db(tmp_path)
    w = WCS(naxis=2)
    w.wcs.crpix = [1000.0, 1000.0]
    w.wcs.crval = [205.0, 42.5]
    w.wcs.cd = np.array([[-2.6 / 3600.0, 0.0], [0.0, 2.6 / 3600.0]])
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    rng = np.random.default_rng(1)
    n = 850
    df = pd.DataFrame({"x": rng.uniform(0, 2000, n), "y": rng.uniform(0, 2000, n), "catalog_id": sids[:n]})
    df.loc[::17, "catalog_id"] = ""
    gm = wi.gaia_radec_by_ids(db, sorted(set(sids[:n])))
    assert gm == {s: (float(a), float(b)) for s, (a, b) in zip(sids[:n], radec[:n], strict=True)}

    calls: list[int] = []
    real = wi.gaia_radec_by_ids
    monkeypatch.setattr(wi, "gaia_radec_by_ids", lambda p, ids: calls.append(len(ids)) or real(p, ids))
    got: dict = {}
    out = wi.finalize_masterstar_sky_coords(df, w, gaia_db_path=db, gaia_coords_out=got)
    qa_reuse = wi.evaluate_matched_world2pix_identity_px(out, w, gaia_ra_dec_by_cid=got, gaia_db_path="")
    assert len(calls) == 1
    qa_query = wi.evaluate_matched_world2pix_identity_px(out, w, gaia_db_path=db)
    assert len(calls) == 2
    assert qa_reuse == qa_query
    assert (out["coord_source"] == "gaia_catalog").sum() == (df["catalog_id"] != "").sum()


# --- G03-34: one "setup directory" rule; one key per path ------------------------------------------


def _old_list_platesolve_setup_dirs(ps_root: Path) -> list[Path]:
    from platesolve_ui_paths import masterstars_csv_in_dir

    if not ps_root.is_dir():
        return []
    out: list[Path] = []
    for sub in sorted(ps_root.iterdir()):
        if not sub.is_dir():
            continue
        if (sub / "per_frame_catalog_index.csv").is_file():
            out.append(sub)
            continue
        if (sub / "MASTERSTAR.fits").is_file() and masterstars_csv_in_dir(sub):
            out.append(sub)
    return out


def test_g03_34_setup_dirs(tmp_path: Path) -> None:
    from types import SimpleNamespace

    import platesolve_ui_paths as pup

    d = tmp_path / "MN_Boo_20260422_001"
    ps = d / "platesolve"
    for name, idx, ms in (("R_60_1", True, True), ("V_60_1", True, False), ("B_60_1", False, True), ("junk", False, False)):
        sub = ps / name
        sub.mkdir(parents=True)
        if idx:
            (sub / "per_frame_catalog_index.csv").write_text("a\n")
        if ms:
            (sub / "MASTERSTAR.fits").write_bytes(b"")
            (sub / "masterstars.csv").write_text("a\n")
    (d / "detrended_aligned" / "lights" / "R_60_1").mkdir(parents=True)
    assert pup.setup_dirs(ps, require_index=False) == _old_list_platesolve_setup_dirs(ps)
    assert [p.name for p in pup.setup_dirs(ps, require_index=True)] == ["R_60_1", "V_60_1"]
    assert not hasattr(pup, "list_platesolve_setup_dirs")
    got = pup.find_photometry_setups(SimpleNamespace(archive_root=tmp_path), None, draft_dir_override=d)
    assert list(got) == ["R_60_1", "V_60_1"]
    r = got["R_60_1"]
    assert r["setup_name"] == "R_60_1" and isinstance(r["setup_name"], str)
    assert r["per_frame_csv_dir"] == (d / "detrended_aligned" / "lights" / "R_60_1").resolve()
    assert r["photometry_dir"] == (ps / "R_60_1" / "photometry").resolve()
    assert "detrended_aligned_dir" not in r and "output_dir" not in r
    assert got["V_60_1"]["per_frame_csv_dir"] is None


# --- G05-19: one job_state ----------------------------------------------------------------------------


def _old_catalog_job_state(job_dir) -> str:
    from run_job import pid_alive, read_status

    st = read_status(job_dir) or {}
    s = str(st.get("state") or "")
    if s in ("done", "failed"):
        return s
    return "running" if pid_alive(st.get("pid")) else "lost"


def test_g05_19_catalog_job_state_equal(tmp_path: Path) -> None:
    import json
    import os

    import catalog_jobs
    from run_job import STATUS_FILE

    cases = {
        "missing": None,
        "empty": {},
        "done": {"state": "done", "pid": os.getpid()},
        "failed": {"state": "failed"},
        "running": {"state": "running", "pid": os.getpid()},
        "dead": {"state": "running", "pid": 2**22 + 12345},
        "nopid": {"state": "running"},
    }
    for name, st in cases.items():
        d = tmp_path / name
        d.mkdir()
        if st is not None:
            (d / STATUS_FILE).write_text(json.dumps(st), encoding="utf-8")
        assert catalog_jobs.job_state(d) == _old_catalog_job_state(d), name
    src = (SRC / "catalog_jobs.py").read_text(encoding="utf-8")
    assert "pid_alive(st.get" not in src


# --- G05-26: one great-circle cone cut; G05-27: one 2-D mean binner -------------------------------


def _old_cone_cut_vsx(df, center, radius_deg):
    import astropy.units as u
    import pandas as pd
    from astropy.coordinates import SkyCoord

    _raq = pd.to_numeric(df["ra_deg"], errors="coerce")
    _deq = pd.to_numeric(df["dec_deg"], errors="coerce")
    _okq = _raq.notna() & _deq.notna()
    if not bool(_okq.any()):
        return pd.DataFrame()
    sub = df.loc[_okq].copy()
    _coo_q = SkyCoord(
        ra=pd.to_numeric(sub["ra_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
        dec=pd.to_numeric(sub["dec_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
        frame="icrs",
    )
    _inner = center.separation(_coo_q).deg <= float(radius_deg) + 1e-9
    return sub.loc[_inner].reset_index(drop=True)


def test_g05_26_cone_cut_equal() -> None:
    import astropy.units as u
    import pandas as pd
    from astropy.coordinates import SkyCoord

    from pipeline_catalog import _cone_cut

    rng = np.random.default_rng(526)
    n = 4000
    for ra0, de0 in ((209.5, 41.0), (0.2, 63.0), (359.9, -12.0)):
        center = SkyCoord(ra0 * u.deg, de0 * u.deg)
        r = 0.6
        ra = (ra0 + rng.uniform(-1.5, 1.5, n)) % 360.0
        de = de0 + rng.uniform(-0.8, 0.8, n)
        df = pd.DataFrame({"catalog_id": np.arange(n).astype(str), "ra_deg": ra, "dec_deg": de.astype(object), "mag": rng.uniform(8, 18, n)})
        df.loc[::97, "ra_deg"] = np.nan
        df.loc[5::131, "dec_deg"] = "x"
        old = _old_cone_cut_vsx(df, center, r)
        new = _cone_cut(df, center, r)
        pd.testing.assert_frame_equal(old, new, check_exact=True)
        assert 0 < len(new) < n
    empty = pd.DataFrame({"ra_deg": [np.nan], "dec_deg": [1.0]})
    assert _cone_cut(empty, SkyCoord(1 * u.deg, 1 * u.deg), 1.0).empty
    src = (SRC / "pipeline_catalog.py").read_text(encoding="utf-8")
    assert src.count("center.separation(_coo_q)") == 1


def _old_bin2d_mean(arr, factor):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or int(factor) < 2:
        return arr
    f = int(factor)
    h, w = arr.shape
    h2, w2 = (h // f) * f, (w // f) * f
    if h2 < f or w2 < f:
        return arr
    a = arr[:h2, :w2].reshape(h2 // f, f, w2 // f, f)
    return np.mean(a, axis=(1, 3)).astype(np.float32)


def _old_mean_bin2d_for_dao(data0, factor):
    f = int(factor)
    a = np.asarray(data0, dtype=np.float32)
    if f <= 1:
        return a, 1
    h, w = a.shape
    h2, w2 = (h // f) * f, (w // f) * f
    if h2 < f or w2 < f:
        return a, 1
    b = a[:h2, :w2].reshape(h2 // f, f, w2 // f, f).mean(axis=(1, 3)).astype(np.float32)
    return b, f


def test_g05_27_bin2d_mean_equal() -> None:
    import pipeline_catalog
    from pipeline_catalog import _bin2d_mean

    assert not hasattr(pipeline_catalog, "_mean_bin2d_for_dao")
    rng = np.random.default_rng(527)
    img = (1000.0 + rng.normal(0, 30, size=(1037, 1551))).astype(np.float32)
    img[100:103, 200:203] += 40000.0
    img64 = img.astype(np.float64)
    for f in (0, 1, 2, 3, 4, 5):
        for a in (img, img64):
            b_new, f_new = _bin2d_mean(a, f)
            b_old, f_old = _old_mean_bin2d_for_dao(a, f)
            assert f_new == f_old and b_new.dtype == b_old.dtype and np.array_equal(b_new, b_old)
            assert np.array_equal(b_new, _old_bin2d_mean(a, f))
    small = img[:3, :3]
    assert _bin2d_mean(small, 4)[1] == 1 and np.array_equal(_bin2d_mean(small, 4)[0], _old_bin2d_mean(small, 4))


# --- G06-22: one progress reporter -----------------------------------------------------------------


def test_g06_22_safe_progress_equal() -> None:
    from infolog import safe_progress

    def _old(progress_cb):
        def _p(msg: str) -> None:
            if progress_cb is None:
                return
            try:
                progress_cb(str(msg))
            except UnicodeEncodeError:
                progress_cb(str(msg).encode("ascii", "backslashreplace").decode("ascii"))

        return _p

    def _cp1252_console(out: list):
        def cb(m: str) -> None:
            m.encode("cp1252")
            out.append(m)

        return cb

    msgs = ["Aperture photometry: preparing shared state...", "BO CVn \u0161\u0165\u017e 3/40", "\u2192 done", 17]
    a: list = []
    b: list = []
    for m in msgs:
        _old(_cp1252_console(a))(m)
        safe_progress(_cp1252_console(b))(m)
    assert a == b and len(a) == len(msgs)
    safe_progress(None)("x")  # no-op
    for name in ("aperture_photometry_state.py", "aperture_photometry.py", "aperture_photometry_target.py",
                 "photometry_core.py", "target_comp_selection.py"):
        assert "backslashreplace" not in (SRC / name).read_text(encoding="utf-8"), name


# --- G06-23: chip size from the MASTERSTAR only ------------------------------------------------------


def test_g06_23_chip_dims_from_masterstar_only() -> None:
    from astropy.io import fits

    from aperture_photometry_state import _chip_dims

    h = fits.Header()
    h["NAXIS"] = 2
    h["NAXIS1"] = 3096
    h["NAXIS2"] = 2078
    assert _chip_dims(h) == (3096, 2078)
    for bad in (None, fits.Header(), {"NAXIS1": 3096, "NAXIS2": 0}, {"NAXIS1": "x", "NAXIS2": 5}):
        with pytest.raises(ValueError):
            _chip_dims(bad, "MASTERSTAR.fits")
    for name in ("aperture_photometry_state.py", "aperture_photometry_target.py"):
        src = (SRC / name).read_text(encoding="utf-8")
        assert "math.ceil(_xm)) + 2" not in src and "math.ceil(xm)) + 2" not in src, name
        assert "state.chip_fw = " not in src, name


# --- G08-36: one leave-one-out sigma (Broeg iteration and ensemble inverse variance) --------------


def _old_broeg_weights(y, tol=1e-5, max_iter=200):
    import math

    y = np.asarray(y, dtype=np.float64)
    n = int(y.shape[0])
    if n == 0:
        return np.zeros(0), 0
    if n == 1:
        return np.ones(1), 0
    fin = np.isfinite(y)
    y0 = np.where(fin, y, 0.0)

    def _sig(r, m):
        v = r[m]
        return float(np.std(v, ddof=1)) if v.size >= 3 else float("nan")

    w = np.array([_sig(y[j], fin[j]) for j in range(n)], dtype=np.float64)
    w = np.where(np.isfinite(w) & (w > 0), 1.0 / (w * w), np.nan)
    if not np.isfinite(w).any():
        return np.full(n, 1.0 / n), 0
    w = np.where(np.isfinite(w), w, np.nanmin(w))
    it = 0
    for it in range(1, int(max_iter) + 1):
        wf = w[:, None] * fin
        tot_w = wf.sum(axis=0)
        tot_wy = (wf * y0).sum(axis=0)
        new = np.empty(n, dtype=np.float64)
        for j in range(n):
            ow = tot_w - wf[j]
            ens = np.where(ow > 0, (tot_wy - wf[j] * y0[j]) / np.where(ow > 0, ow, 1.0), np.nan)
            r = y[j] - ens
            sj = _sig(r, np.isfinite(r))
            new[j] = 1.0 / (sj * sj) if math.isfinite(sj) and sj > 0 else np.nan
        if not np.isfinite(new).any():
            break
        new = np.where(np.isfinite(new), new, np.nanmin(new))
        rel = np.max(np.abs(new - w) / np.maximum(w, 1e-300))
        w = new
        if rel < float(tol):
            break
    return w / float(np.sum(w)), int(it)


def _old_ensemble_loo_sigma(y, w):
    n = len(w)
    sig = np.full(n, np.nan)
    for j in range(n):
        if n >= 2:
            oth = np.delete(np.arange(n), j)
            fin_o = np.isfinite(y[oth])
            wo = w[oth][:, None] * fin_o
            den = wo.sum(axis=0)
            ens = np.where(den > 0, (wo * np.where(fin_o, y[oth], 0.0)).sum(axis=0) / np.where(den > 0, den, 1.0), np.nan)
            r = y[j] - ens
        else:
            r = y[j]
        r = r[np.isfinite(r)]
        sig[j] = float(np.std(r, ddof=1)) if r.size >= 3 else np.nan
    return sig


def _comp_matrix(seed: int, n_comp: int = 8, n_fr: int = 300) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.uniform(0.002, 0.02, n_comp)[:, None]
    y = rng.normal(0, 1, (n_comp, n_fr)) * noise + 0.05 * np.sin(np.linspace(0, 6, n_fr))
    y[rng.random(y.shape) < 0.1] = np.nan
    if n_comp > 2:
        y[2, ::3] += 0.08  # a variable comp
    return y - np.nanmedian(y, axis=1)[:, None]


def test_g08_36_broeg_weights_bit_identical() -> None:
    from photometry_lightcurve import broeg_weights

    for seed in range(6):
        y = _comp_matrix(seed, n_comp=3 + seed)
        w_old, it_old = _old_broeg_weights(y)
        w_new, it_new = broeg_weights(y)
        assert it_old == it_new and np.array_equal(w_old, w_new)
    for y in (np.zeros((0, 5)), np.ones((1, 5))):
        assert all(np.array_equal(a, b) for a, b in zip(_old_broeg_weights(y), broeg_weights(y), strict=True))


def test_g08_36_loo_sigma_matches_old_recompute() -> None:
    """The ensemble inverse variances now use the Broeg LOO sigma (running totals instead of
    np.delete): same quantity, rounding-level difference only (allowed by Milan up to ~1e-7)."""
    from photometry_lightcurve import _loo_sigma, broeg_weights

    worst = 0.0
    for seed in range(6):
        y = _comp_matrix(seed, n_comp=3 + seed)
        w, _ = broeg_weights(y)
        old = _old_ensemble_loo_sigma(y, w)
        new = _loo_sigma(y, w)
        assert np.array_equal(np.isfinite(old), np.isfinite(new))
        worst = max(worst, float(np.nanmax(np.abs(new - old) / old)))
    assert worst < 1e-12, worst
    y1 = _comp_matrix(9, n_comp=1)
    assert np.array_equal(_loo_sigma(y1, np.ones(1)), _old_ensemble_loo_sigma(y1, np.ones(1)))


# --- G14-18: one ADC container scale ----------------------------------------------------------------


def test_g14_18_container_scale_equal() -> None:
    from gain_ptc import container_scale, file_gain, native_gain

    for bits in (None, 0, 1, 8, 12, 14, 16, 17, 20):
        old_rig = float(2 ** (16 - int(bits))) if bits is not None and 1 <= int(bits) <= 16 else 1.0
        old_ui = 2 ** (16 - int(bits)) if bits and 1 <= int(bits) <= 16 else 1
        assert container_scale(bits) == old_rig == old_ui
        for g in (0.0, 0.123, 1.0, 3.6, 25.0):
            assert file_gain(g, bits) == g / old_rig  # rig_photometry (scale > 0 always)
            old_native = g if bits is None or not (1 <= int(bits) <= 16) else g * float(2 ** (16 - int(bits)))
            assert native_gain(g, bits) == old_native
    for name in ("rig_photometry.py", "ui_database_explorer.py"):
        assert "2 ** (" not in (SRC / name).read_text(encoding="utf-8"), name


# --- G10-21: one forced-photometry driver (fixed positions and tracks) ------------------------------


def test_g10_21_forced_photometry_equal(tmp_path: Path) -> None:
    from astropy.io import fits

    import transient_field as tf
    from parallel_frames import map_frames

    def _old_fixed(frame_paths, fwhm, xy, *, k, a_in, a_out, gain, rn, workers):
        jobs = [(str(p), xy, float(f), k, a_in, a_out, gain, rn) for p, f in zip(frame_paths, fwhm, strict=True)]
        return np.stack(list(map_frames(tf._measure_one, jobs, workers=workers, stage="t")))

    def _old_tracks(frame_paths, fwhm, xy_t, *, k, a_in, a_out, gain, rn, workers):
        jobs = [(str(p), xy_t[t], float(f), k, a_in, a_out, gain, rn) for t, (p, f) in enumerate(zip(frame_paths, fwhm, strict=True))]
        return np.stack(list(map_frames(tf._measure_one, jobs, workers=workers, stage="t")))

    assert not hasattr(tf, "forced_photometry_tracks")
    rng = np.random.default_rng(1021)
    yy, xx = np.mgrid[0:120, 0:160]
    stars = np.array([[40.3, 50.7], [100.1, 30.2], [80.0, 90.5]])
    paths = []
    for t in range(4):
        img = 500.0 + rng.normal(0, 8, size=yy.shape)
        for x, y in stars + 0.4 * t:
            img += 3000.0 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.5**2))
        h = fits.Header()
        for kk, v in enumerate(np.eye(3).ravel()):
            h[f"VY_AM{kk}"] = float(v)
        p = tmp_path / f"f{t}.fits"
        fits.PrimaryHDU(img.astype(np.float32), header=h).writeto(p)
        paths.append(p)
    fwhm = np.array([3.4, 3.5, 3.6, 3.5])
    kw = dict(k=1.5, a_in=3.0, a_out=5.0, gain=1.2, rn=3.5, workers=1)
    assert np.array_equal(tf.forced_photometry(paths, fwhm, stars, **kw), _old_fixed(paths, fwhm, stars, **kw), equal_nan=True)
    xy_t = np.stack([stars + 0.4 * t for t in range(4)])
    assert np.array_equal(tf.forced_photometry(paths, fwhm, xy_t, **kw), _old_tracks(paths, fwhm, xy_t, **kw), equal_nan=True)


# --- G10-35: one finite-JSON meta writer ------------------------------------------------------------


def test_g10_35_write_json_finite_equal(tmp_path: Path) -> None:
    import json
    import math

    import transient_field
    import variability_field
    from utils import write_json_finite

    def _old_c(o):
        if isinstance(o, dict):
            return {k: _old_c(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_old_c(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return float(o) if math.isfinite(float(o)) else None
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    meta = {"n": np.int64(5), "z": np.float32(2.5), "bad": float("nan"), "inf": np.float64(np.inf), "ok": np.bool_(True),
            "s": "BO CVn", "l": [1, np.float64(0.1), (np.int32(3), None)], "d": {"x": [np.nan, 2.0]}, "none": None}
    write_json_finite(tmp_path / "a.json", meta)
    old = json.dumps(_old_c(meta), indent=2)
    assert (tmp_path / "a.json").read_bytes() == old.encode("utf-8")
    variability_field._write_meta(tmp_path, meta)
    assert (tmp_path / "variability_meta.json").read_bytes() == old.encode("utf-8")
    transient_field._write_meta(tmp_path, meta)
    assert (tmp_path / transient_field.META_JSON).read_bytes() == old.encode("utf-8")


# --- G10-48: one centre / width of the scatter index (z_s and the plotted limit) -------------------


def test_g10_48_scatter_index_center_scale_equal() -> None:
    import pandas as pd

    from variability_field import scatter_index_center_scale

    rng = np.random.default_rng(1048)
    for n in (7, 8, 101, 598, 599, 600, 601, 2500, 2501):
        s = pd.Series(np.exp(rng.normal(0, 0.3, n)) * rng.uniform(0.005, 0.2, n))
        sexp = pd.Series(rng.uniform(0.005, 0.2, n))
        s.iloc[::13] = np.nan
        lq = np.log(s / sexp)
        # old _var_indices (nanmedian over the masked values)
        a = lq.to_numpy(float)
        md0 = float(np.nanmedian(a))
        sc0 = 1.4826 * float(np.nanmedian(np.abs(a - md0)))
        assert scatter_index_center_scale(a) == (md0, sc0)
        # old noise_curves (finite values of a Series, np.median)
        lf = lq[np.isfinite(lq)]
        md1 = float(np.median(lf))
        sc1 = 1.4826 * float(np.median(np.abs(lf - md1)))
        assert scatter_index_center_scale(lf.to_numpy(float)) == (md1, sc1)
    src = (SRC / "variability_plots.py").read_text(encoding="utf-8")
    assert "1.4826 * float" not in src


def test_g10_48_noise_curves_unchanged() -> None:
    import pandas as pd

    from variability_plots import noise_curves

    rng = np.random.default_rng(48)
    n = 1500
    mag = rng.uniform(9, 17, n)
    e = 0.002 * 10 ** (0.2 * (mag - 9))
    s = np.sqrt((1.1 * e) ** 2 + 0.004**2) * np.exp(rng.normal(0, 0.2, n))
    f = pd.DataFrame({"mag": mag, "e": e, "s": s, "gates_ok": rng.random(n) > 0.1, "known_variable": rng.random(n) < 0.02})
    meta = {"noise_model": {"a": 1.1, "floor_mag": 0.004}, "z_threshold": 4.2}
    mids, sexp, lim = noise_curves(f, meta)
    g = f["gates_ok"].astype(bool)
    ref = g & ~f["known_variable"].astype(bool)
    lq = np.log(f["s"][ref] / np.sqrt((1.1 * f["e"][ref]) ** 2 + 0.004**2))
    lq = lq[np.isfinite(lq)]
    md = float(np.median(lq))
    sc = 1.4826 * float(np.median(np.abs(lq - md)))
    assert np.array_equal(lim, sexp * np.exp(md + 4.2 * sc))


# --- G11-30: one JPEG encoder for the PDF images -----------------------------------------------------

# verbatim copies of the two removed methods (HEAD 08e985a), self unused
from io import BytesIO  # noqa: E402
from typing import Any  # noqa: E402


def _old_compress_image_for_pdf(self, 
    src_path: str | Path,
    max_width_px: int = 1200,
    jpeg_quality: int = 72,
) -> tuple[BytesIO, str]:
    """
    Open image, resize if wider than max_width_px (preserve aspect ratio),
    convert to JPEG, return (BytesIO, "JPEG"). RGBA is flattened onto white.
    ``force_jpeg`` reserved for API compatibility with the Phase 4 spec.
    """
    src = Path(src_path)
    if not src.is_file():
        raise FileNotFoundError(str(src))
    try:
        from PIL import Image as PILImage
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Pillow required for PDF image compression") from exc

    # OUTPUT-REVIEW-2026-10 step 7: in memory (a disk cache keyed by the path never hit for the
    # images the report draws again every run)

    def _to_rgb_white_bg(im: Any) -> Any:
        if im.mode == "RGBA":
            bg = PILImage.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[3])
            return bg
        if im.mode == "LA":
            bg = PILImage.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[1])
            return bg
        if im.mode == "P":
            if "transparency" in im.info:
                im = im.convert("RGBA")
                return _to_rgb_white_bg(im)
            return im.convert("RGB")
        return im.convert("RGB")

    with PILImage.open(src) as im0:
        if im0.format == "JPEG" and im0.size[0] <= int(max_width_px):
            # already a JPEG at or below the embed width: no second lossy encode
            return BytesIO(src.read_bytes()), "JPEG"
        im = _to_rgb_white_bg(im0)
        w, h = im.size
        mw = int(max_width_px)
        if w > mw and w > 0:
            scale = float(mw) / float(w)
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            im = im.resize((nw, nh), resample=PILImage.Resampling.LANCZOS)

        buf = BytesIO()
        im.save(
            buf,
            format="JPEG",
            quality=int(jpeg_quality),
            optimize=True,
            progressive=True,
        )
        buf.seek(0)
        return buf, "JPEG"
def _old_compress_png_bytes_for_pdf(self, 
    png_bytes: bytes,
    max_width_px: int = 1200,
    jpeg_quality: int = 72,
) -> tuple[BytesIO, str]:
    """Same as _compress_image_for_pdf but from an in-memory PNG (e.g. matplotlib buffer)."""
    try:
        from PIL import Image as PILImage
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Pillow required for PDF image compression") from exc

    def _to_rgb_white_bg(im: Any) -> Any:
        if im.mode == "RGBA":
            bg = PILImage.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[3])
            return bg
        if im.mode == "LA":
            bg = PILImage.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[1])
            return bg
        if im.mode == "P":
            if "transparency" in im.info:
                im = im.convert("RGBA")
                return _to_rgb_white_bg(im)
            return im.convert("RGB")
        return im.convert("RGB")

    with PILImage.open(BytesIO(png_bytes)) as im0:
        im = _to_rgb_white_bg(im0)
        w, h = im.size
        mw = int(max_width_px)
        if w > mw and w > 0:
            scale = float(mw) / float(w)
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            im = im.resize((nw, nh), resample=PILImage.Resampling.LANCZOS)
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=int(jpeg_quality), optimize=True, progressive=True)
        buf.seek(0)
        return buf, "JPEG"


def _pdf_test_images(tmp_path: Path) -> list[Path]:
    from PIL import Image

    rng = np.random.default_rng(1130)
    out = []
    arr = (rng.random((300, 1700, 4)) * 255).astype(np.uint8)
    arr[..., 3] = np.where(arr[..., 3] > 60, 255, 0)
    for name, im in (
        ("rgba_wide.png", Image.fromarray(arr, "RGBA")),
        ("rgb_small.png", Image.fromarray(arr[:200, :500, :3], "RGB")),
        ("la.png", Image.fromarray(arr[:, :900, :2], "LA")),
        ("gray_wide.png", Image.fromarray(arr[:, :, 0], "L")),
    ):
        p = tmp_path / name
        im.save(p)
        out.append(p)
    pal = Image.fromarray(arr[:, :1300, :3], "RGB").convert("P", palette=Image.Palette.ADAPTIVE, colors=64)
    pal.info["transparency"] = 3
    p = tmp_path / "pal_t.png"
    pal.save(p, transparency=3)
    out.append(p)
    p = tmp_path / "pal.png"
    Image.fromarray(arr[:, :700, :3], "RGB").convert("P", palette=Image.Palette.ADAPTIVE, colors=64).save(p)
    out.append(p)
    for name, w in (("small.jpg", 800), ("big.jpg", 1700)):
        p = tmp_path / name
        Image.fromarray(arr[:, :w, :3], "RGB").save(p, quality=90)
        out.append(p)
    return out


def test_g11_30_pdf_image_compression_equal(tmp_path: Path) -> None:
    from photometry_report import _PhotometryReportBuilder as B

    for p in _pdf_test_images(tmp_path):
        for mw, q in ((1200, 72), (600, 60), (2000, 85)):
            old = _old_compress_image_for_pdf(None, p, mw, q)
            new = B._compress_image_for_pdf(None, p, mw, q)
            assert old[1] == new[1] and old[0].getvalue() == new[0].getvalue(), (p.name, mw)
            if p.suffix == ".png":
                b = p.read_bytes()
                old = _old_compress_png_bytes_for_pdf(None, b, mw, q)
                new = B._compress_png_bytes_for_pdf(None, b, mw, q)
                assert old[0].getvalue() == new[0].getvalue(), (p.name, mw)
    src = (SRC / "photometry_report.py").read_text(encoding="utf-8")
    assert "force_jpeg" not in src and src.count("def _to_rgb_white_bg") == 1


# --- G13-23: band size from astro_index; G13-28: one masterstars CSV resolver ----------------------


def test_g13_23_band_size_equal() -> None:
    from astro_index import N_BANDS, band_size

    for k in range(N_BANDS + 3):
        assert band_size(k) == 0.15 * 2**k
    assert "0.15 * 2 **" not in (SRC / "settings_model.py").read_text(encoding="utf-8")


def test_g13_28_masterstars_csv_resolver(tmp_path: Path) -> None:
    import pipeline_ui_helpers
    from platesolve_ui_paths import masterstars_csv_in_dir

    assert not hasattr(pipeline_ui_helpers, "resolve_masterstars_metadata_csv")
    assert masterstars_csv_in_dir(str(tmp_path)) is None
    (tmp_path / "masterstars.csv").write_text("a\n")
    assert masterstars_csv_in_dir(str(tmp_path)) == tmp_path / "masterstars.csv"
    (tmp_path / "masterstars_full_match.csv").write_text("a\n")
    assert masterstars_csv_in_dir(tmp_path) == tmp_path / "masterstars_full_match.csv"


# --- G10-19: one PDM Theta (callers keep their bins and phase origin) -------------------------------


def _old_tess_pdm_theta(t, m, period, nbins=10):
    ph = (t / period) % 1.0
    b = np.minimum((ph * nbins).astype(int), nbins - 1)
    s_tot = float(np.var(m, ddof=1))
    num = den = 0.0
    for k in range(nbins):
        x = m[b == k]
        if x.size > 1:
            num += float(np.var(x, ddof=1)) * (x.size - 1)
            den += x.size - 1
    return float((num / den) / s_tot) if den > 0 and s_tot > 0 else float("nan")


def _old_gui_pdm_theta(phase, mag, n_bins=20):
    ph = np.asarray(phase, float)
    m = np.asarray(mag, float)
    ok = np.isfinite(ph) & np.isfinite(m)
    ph, m = ph[ok], m[ok]
    if m.size < 3:
        return float("nan")
    b = np.minimum((ph * n_bins).astype(int), n_bins - 1)
    num = 0.0
    dof = 0
    for k in range(n_bins):
        x = m[b == k]
        if x.size > 1:
            num += float(np.sum((x - x.mean()) ** 2))
            dof += x.size - 1
    vt = float(np.var(m, ddof=1))
    return (num / dof) / vt if dof > 0 and vt > 0 else float("nan")


def test_g10_19_pdm_theta_tess_bit_identical_gui_rounding_only() -> None:
    import tess_analysis
    from variability_store import pdm_theta, phase_of

    assert not hasattr(tess_analysis, "pdm_theta")
    rng = np.random.default_rng(1019)
    # TESS sector: 27 d at 2 min cadence, BJD - 2457000
    t = np.sort(rng.uniform(3000.0, 3027.0, 12000))
    m = 10.0 + 0.05 * np.sin(2 * np.pi * t / 0.37) + rng.normal(0, 0.004, t.size)
    worst = 0.0
    for p in np.r_[0.37, 0.74, 0.185, rng.uniform(0.1, 5.0, 60)]:
        assert pdm_theta(phase_of(t, p, 0.0), m, n_bins=10) == _old_tess_pdm_theta(t, m, p)
        # one night, GUI: 20 bins, phase from t0 (display only; 1-ulp level differences allowed)
        tn, mn = t[:400] + 2457000.0, m[:400].copy()
        mn[::37] = np.nan
        a = pdm_theta(phase_of(tn, p, tn[0]), mn)
        b = _old_gui_pdm_theta(phase_of(tn, p, tn[0]), mn)
        worst = max(worst, abs(a - b) / b)
    assert worst < 1e-12, worst


# --- G07-07: one batched Gaia BP-RP lookup -----------------------------------------------------------
# verbatim copies of the four lookups at HEAD 08e985a (photometry_comp)
import logging  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402

import pandas as pd  # noqa: E402

from catalog_db import connect_ro, query_local_gaia_by_source_ids  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from infolog import log_event  # noqa: E402
from photometry_comp import _sid_int  # noqa: E402
from photometry_defs import LOGGER  # noqa: E402


def _old_enrich_comp_bp_rp(
    candidates: pd.DataFrame,
    gaia_db_path: str | None,
    *,
    gaia_prefetch: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Fill ``bp_rp`` for comp stars where it is missing (Gaia DR3 by ``source_id``)."""
    if candidates is None or getattr(candidates, "empty", True):
        return pd.DataFrame()

    df = candidates.copy()
    if "bp_rp" not in df.columns:
        df["bp_rp"] = float("nan")

    df["bp_rp"] = pd.to_numeric(df.get("bp_rp"), errors="coerce")
    if "ra_deg" in df.columns:
        df["ra_deg"] = pd.to_numeric(df.get("ra_deg"), errors="coerce")
    if "dec_deg" in df.columns:
        df["dec_deg"] = pd.to_numeric(df.get("dec_deg"), errors="coerce")

    con = None
    gaia_cols: set[str] = set()
    gaia_path = str(gaia_db_path or "").strip()
    if gaia_path and os.path.exists(gaia_path):
        try:
            import sqlite3  # noqa: PLC0415

            con = connect_ro(gaia_path)
            con.row_factory = sqlite3.Row
            gaia_cols = {
                str(r[1]).strip().lower() for r in con.execute("PRAGMA table_info('gaia_dr3')").fetchall()
            }
        except Exception:  # noqa: BLE001
            con = None
            gaia_cols = set()

    sel_bp = "bp_rp" in gaia_cols
    gaia_bp_cache: dict[int, float] = {}

    def _gaia_bp_rp(sid_i: int) -> float:
        if sid_i in gaia_bp_cache:
            return gaia_bp_cache[sid_i]
        bp_r = float("nan")
        gid_pf = normalize_gaia_source_id(sid_i)
        if gaia_prefetch and gid_pf and gid_pf in gaia_prefetch:
            try:
                vbp = gaia_prefetch[gid_pf].get("bp_rp")
                if vbp is not None and math.isfinite(float(vbp)):
                    bp_r = float(vbp)
            except (TypeError, ValueError):
                pass
            gaia_bp_cache[int(sid_i)] = bp_r
            return bp_r
        if con is not None and sel_bp:
            try:
                rw = con.execute(
                    "SELECT bp_rp FROM gaia_dr3 WHERE source_id=? LIMIT 1;",
                    (int(sid_i),),
                ).fetchone()
                if rw is not None and rw["bp_rp"] is not None:
                    bp_r = float(rw["bp_rp"])
            except Exception as exc:  # noqa: BLE001
                logging.error('[EXC-0122] Gaia DB bp_rp row fetch fails - comp star keeps NaN bp_rp and wrong colour tier: %s', exc)
                pass
        gaia_bp_cache[int(sid_i)] = bp_r if math.isfinite(bp_r) else float("nan")
        return gaia_bp_cache[int(sid_i)]

    try:
        # SPEED-2026-10: visit only the rows whose bp_rp is missing (the column is numeric here);
        # each row is re-read before use, as iterrows did.
        _missing = np.flatnonzero(~np.isfinite(df["bp_rp"].to_numpy(dtype=float)))
        for _pos in _missing.tolist():
            idx = df.index[_pos]
            row = df.iloc[_pos]
            try:
                bp_now = float(pd.to_numeric(row.get("bp_rp"), errors="coerce"))
            except Exception:  # noqa: BLE001
                bp_now = float("nan")
            if math.isfinite(bp_now):
                continue
            sid_i = _sid_int(row.get("source_id") or row.get("catalog_id") or row.get("name"))
            if sid_i is None:
                continue
            gaia_bp = _gaia_bp_rp(sid_i)
            if math.isfinite(gaia_bp):
                df.at[idx, "bp_rp"] = float(gaia_bp)
    finally:
        try:
            if con is not None:
                con.close()
        except Exception:  # noqa: BLE001
            # EXC-0123: T2 -- sqlite con.close() failure during comp bp_rp enrichment ignored (EXCEPT-BULK-2 2026-07-08)
            pass

    return df


def _old_enrich_active_targets_bp_rp(
    targets_df: pd.DataFrame,
    *,
    gaia_db_path: str | Path | None,
) -> pd.DataFrame:
    """Fill ``bp_rp`` for active targets from Gaia DR3 by ``catalog_id``."""
    if targets_df is None or getattr(targets_df, "empty", True):
        return targets_df

    df = targets_df.copy()
    if "bp_rp" not in df.columns:
        df["bp_rp"] = float("nan")
    df["bp_rp"] = pd.to_numeric(df["bp_rp"], errors="coerce")
    if "ra_deg" in df.columns:
        df["ra_deg"] = pd.to_numeric(df["ra_deg"], errors="coerce")
    if "dec_deg" in df.columns:
        df["dec_deg"] = pd.to_numeric(df["dec_deg"], errors="coerce")

    gaia_path = str(gaia_db_path or "").strip()
    con = None
    gaia_cols: set[str] = set()
    if gaia_path and os.path.exists(gaia_path):
        try:
            import sqlite3  # noqa: PLC0415

            con = connect_ro(gaia_path)
            con.row_factory = sqlite3.Row
            gaia_cols = {
                str(r[1]).strip().lower() for r in con.execute("PRAGMA table_info('gaia_dr3')").fetchall()
            }
        except Exception:  # noqa: BLE001
            # EXC-0193: T2 -- sqlite con.close() after active-target bp_rp enrichment ignored (EXCEPT-BULK-2 2026-07-08)
            con = None
            gaia_cols = set()
    sel_bp = "bp_rp" in gaia_cols
    gaia_cache: dict[int, float] = {}

    def _gaia_bp(sid_i: int) -> float:
        if sid_i in gaia_cache:
            return gaia_cache[sid_i]
        bp_r = float("nan")
        if con is not None and sel_bp:
            try:
                rw = con.execute(
                    "SELECT bp_rp FROM gaia_dr3 WHERE source_id=? LIMIT 1;",
                    (int(sid_i),),
                ).fetchone()
                if rw is not None and rw["bp_rp"] is not None:
                    bp_r = float(rw["bp_rp"])
            except Exception as exc:  # noqa: BLE001
                logging.error('[EXC-0192] Gaia DB bp_rp fetch for active targets fails - target row keeps NaN bp_rp: %s', exc)
                pass
        gaia_cache[int(sid_i)] = bp_r if math.isfinite(bp_r) else float("nan")
        return gaia_cache[int(sid_i)]

    try:
        for idx, row in df.iterrows():
            try:
                bp_now = float(pd.to_numeric(row.get("bp_rp"), errors="coerce"))
            except Exception:  # noqa: BLE001
                bp_now = float("nan")
            if math.isfinite(bp_now):
                continue
            sid_i = _sid_int(row.get("catalog_id"))
            if sid_i is None:
                continue
            gaia_bp = _gaia_bp(sid_i)
            if math.isfinite(gaia_bp):
                df.at[idx, "bp_rp"] = float(gaia_bp)
    finally:
        try:
            if con is not None:
                con.close()
        except Exception:  # noqa: BLE001
            # EXC-0195: T4 -- DB manifest files[] NAXIS query fails - returns caller-supplied default frame width/height (EXCEPT-BULK-2 2026-07-08)
            pass

    return df


def _old_batch_enrich_targets_bp_rp_from_gaia_db(
    target_cids: list[str],
    gaia_db_path: str,
) -> dict[str, dict[str, Any]]:
    """Prefetch Gaia ``bp_rp`` for comparison-star selection targets (batched SQL)."""
    gdb = str(gaia_db_path or "").strip()
    if not target_cids or not gdb:
        return {}
    try:
        gp = Path(gdb).expanduser().resolve()
        if not gp.is_file():
            return {}
    except OSError:
        return {}

    ids_norm: list[str] = []
    seen: set[str] = set()
    for raw in target_cids:
        g = normalize_gaia_source_id(raw)
        if not g or not g.isdigit() or g in seen:
            continue
        seen.add(g)
        ids_norm.append(g)
    if not ids_norm:
        return {}

    try:
        base = query_local_gaia_by_source_ids(gp, ids_norm)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[COMP-SELECT] Batch Gaia bp_rp lookup failed: %s", exc)
        return {}
    # G07-20: only bp_rp is used by comp selection (the teff query had no reader)
    out: dict[str, dict[str, Any]] = {k: {"bp_rp": v.get("bp_rp")} for k, v in base.items()}

    return out


def _old_enrich_target_bp_rp_from_gaia_db(
    target: pd.Series,
    *,
    gaia_db_path: str,
    gaia_prefetch: dict[str, dict[str, Any]] | None = None,
) -> pd.Series:
    """Fill ``bp_rp`` for one active target (comparison-star selection) from Gaia by ``source_id``."""
    out = target.copy()
    vsx = str(out.get("vsx_name", "") or "").strip() or str(out.get("name", "") or "").strip() or "?"

    def _fscalar(key: str) -> float:
        try:
            v = float(pd.to_numeric(out.get(key), errors="coerce"))
        except Exception:  # noqa: BLE001
            return float("nan")
        return v if math.isfinite(v) else float("nan")

    bpr_ms = _fscalar("bp_rp")

    gid = normalize_gaia_source_id(out.get("catalog_id"))
    gdb = str(gaia_db_path or "").strip()
    try:
        gp = Path(gdb).expanduser().resolve()
        gdb_ok = bool(gdb) and gp.is_file()
    except OSError:
        gdb_ok = False

    bpr_nf = float("nan")
    _prefetched = bool(gaia_prefetch and gid and gid in gaia_prefetch)
    if _prefetched:
        pf = gaia_prefetch[gid]  # type: ignore[index]
        try:
            vbp = pf.get("bp_rp")
            if vbp is not None and math.isfinite(float(vbp)):
                bpr_nf = float(vbp)
        except (TypeError, ValueError):
            pass
    elif gid and gdb_ok and gid.isdigit():
        try:
            import sqlite3  # noqa: PLC0415

            con = connect_ro(gp)
            con.row_factory = sqlite3.Row
            try:
                cols = {str(r[1]).strip().lower() for r in con.execute("PRAGMA table_info('gaia_dr3')").fetchall()}
                parts = [c for c in ("bp_rp",) if c in cols]
                if parts:
                    rw = con.execute(
                        f"SELECT {', '.join(parts)} FROM gaia_dr3 WHERE source_id=? LIMIT 1;",
                        (int(gid),),
                    ).fetchone()
                    if rw is not None:
                        if "bp_rp" in parts:
                            try:
                                vbp = rw["bp_rp"]
                                if vbp is not None:
                                    bpr_nf = float(vbp)
                            except (TypeError, ValueError, KeyError, IndexError):
                                pass
            finally:
                con.close()
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0202] Outer Gaia SQL for target bp_rp logs failure - target keeps CSV bp_rp or NaN: %s', exc)
            log_event(f"TARGET Gaia SQL: {vsx} - {exc!s}")

    if math.isfinite(bpr_nf):
        out["bp_rp"] = float(bpr_nf)
    elif math.isfinite(bpr_ms):
        out["bp_rp"] = float(bpr_ms)
    elif not gid:
        log_event(f"TARGET bp_rp: {vsx} - no valid Gaia catalog_id")
    return out


def _bprp_gaia_db(tmp_path: Path, n: int = 700) -> tuple[str, list[str], dict[str, float | None]]:
    import sqlite3

    rng = np.random.default_rng(707)
    sids = [str(1500000000000000000 + 104729 * i) for i in range(n)]
    con = sqlite3.connect(tmp_path / "gaia_bprp.db")
    con.execute("CREATE TABLE gaia_dr3 (source_id INTEGER, ra REAL, dec REAL, g_mag REAL, bp_mag REAL, rp_mag REAL, bp_rp REAL)")
    truth: dict[str, float | None] = {}
    rows = []
    for i, s in enumerate(sids):
        g = float(rng.uniform(8, 19))
        if i % 23 == 0:  # no colour (Gaia: bp_rp NULL with BP or RP missing)
            rows.append((int(s), 10.0, 20.0, g, None, float(g - 0.4), None))
            truth[s] = None
        else:
            c = float(rng.uniform(-0.2, 3.5))
            rows.append((int(s), 10.0, 20.0, g, g + 0.3, g + 0.3 - c, c))
            truth[s] = c
    con.executemany("INSERT INTO gaia_dr3 VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return str(tmp_path / "gaia_bprp.db"), sids, truth


def test_g07_07_gaia_bp_rp_lookups_equal(tmp_path: Path) -> None:
    import photometry_comp as pc

    db, sids, _truth = _bprp_gaia_db(tmp_path)
    assert not hasattr(pc, "_batch_enrich_targets_bp_rp_from_gaia_db")
    rng = np.random.default_rng(77)
    n = 400
    ids = list(rng.choice(sids, n, replace=True))
    ids[5] = "Gaia DR3 " + ids[5]
    ids[9] = ""
    ids[11] = "not-a-gaia-id"
    ids[13] = "999999999999"  # not in the DB
    bp = np.where(rng.random(n) < 0.3, np.round(rng.uniform(0, 2, n), 3), np.nan)
    df = pd.DataFrame({"catalog_id": ids, "bp_rp": bp, "ra_deg": rng.uniform(0, 1, n), "dec_deg": rng.uniform(0, 1, n)})
    dfs = df.rename(columns={"catalog_id": "source_id"})

    # lookup 3 (batch prefetch) -> dict; survivors: same ids, value only
    old_pf = _old_batch_enrich_targets_bp_rp_from_gaia_db(ids, db)
    new_pf = pc.gaia_bp_rp(ids, db)
    assert {k: v["bp_rp"] for k, v in old_pf.items()} == new_pf
    part = {k: v for i, (k, v) in enumerate(new_pf.items()) if i % 2 == 0}
    part_old = {k: {"bp_rp": v} for k, v in part.items()}

    # lookup 1 (comps), with and without prefetch; lookup 2 (active targets)
    # s5_dup2 (Milan 2026-10-30, G07-07 "DB wins"): the old copies kept a CSV value even when the DB had a
    # colour; the reference is now the old copy on the frame whose CSV value is cleared where the DB has one.
    def _db_wins(frame, col):
        out = frame.copy()
        g = out[col].map(lambda v: pc.normalize_gaia_source_id(v) if pc._sid_int(v) is not None else None)
        has_db = g.map(lambda k: k is not None and new_pf.get(k) is not None and np.isfinite(new_pf[k]))
        out.loc[has_db.to_numpy(dtype=bool), "bp_rp"] = np.nan
        return out

    for frame, col in ((df, "catalog_id"), (dfs, "source_id")):
        ref_in = _db_wins(frame, col)
        pd.testing.assert_frame_equal(_old_enrich_comp_bp_rp(ref_in, db), pc._enrich_comp_bp_rp(frame, db), check_exact=True)
        pd.testing.assert_frame_equal(
            _old_enrich_comp_bp_rp(ref_in, db, gaia_prefetch=part_old),
            pc._enrich_comp_bp_rp(frame, db, gaia_prefetch=part),
            check_exact=True,
        )
    pd.testing.assert_frame_equal(_old_enrich_active_targets_bp_rp(_db_wins(df, "catalog_id"), gaia_db_path=db),
                                  pc._enrich_active_targets_bp_rp(df, gaia_db_path=db), check_exact=True)
    pd.testing.assert_frame_equal(_old_enrich_comp_bp_rp(df, None), pc._enrich_comp_bp_rp(df, None), check_exact=True)

    # lookup 4 (one target): prefetched, DB, no DB
    for i in range(0, n, 7):
        row = df.iloc[i].copy()
        row["vsx_name"] = f"V{i}"
        for pf_old, pf_new, path in ((part_old, part, db), (None, None, db), (None, None, "")):
            a = _old_enrich_target_bp_rp_from_gaia_db(row, gaia_db_path=path, gaia_prefetch=pf_old)
            b = pc._enrich_target_bp_rp_from_gaia_db(row, gaia_db_path=path, gaia_prefetch=pf_new)
            pd.testing.assert_series_equal(a, b, check_exact=True)


# --- G01-17: one frame-QC store and one QC tail (calibration-time and RAM QC) -----------------------


class _QcDb:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def update_obs_file_quality_by_id(self, draft_id, rid, **kw):
        self.calls.append((draft_id, rid, tuple(sorted(kw.items()))))


def test_g01_17_qc_tail_and_store_equal(monkeypatch) -> None:
    import math

    import pipeline_calibrate as pcal

    drift: list = []
    monkeypatch.setattr(pcal, "draft_median_pointing_icrs_deg", lambda db, d: (209.5, 41.25))
    monkeypatch.setattr(pcal, "sync_obs_files_drift_arcmin_for_draft", lambda db, d, **kw: drift.append((d, kw)))

    def _old_tail(db, draft_id, rows, fwhm_by_id):
        vals = [v for v in fwhm_by_id.values() if math.isfinite(v) and v > 0]
        if vals:
            med = float(np.median(np.asarray(vals, dtype=np.float64)))
            if not math.isfinite(med) or med <= 0:
                med = None
        else:
            med = None
        thr = med * 1.5 if med is not None else None
        auto_n = 0
        for row in rows:
            rid = int(row["ID"])
            rej = 0
            if med is not None and thr is not None:
                fv = fwhm_by_id.get(rid)
                if fv is not None and math.isfinite(float(fv)) and float(fv) > thr:
                    rej = 1
            if rej:
                auto_n += 1
            db.update_obs_file_quality_by_id(int(draft_id), rid, rejected_auto=rej)
        med_ra, med_de = pcal.draft_median_pointing_icrs_deg(db, int(draft_id))
        pcal.sync_obs_files_drift_arcmin_for_draft(db, int(draft_id), ref_ra_deg=med_ra, ref_de_deg=med_de)
        return med, auto_n, med_ra, med_de

    def _old_store(db, draft_id, rid, m):
        _el = m.get("elongation_mean")
        _el_db = float(_el) if _el is not None and math.isfinite(float(_el)) and float(_el) > 0.0 else None
        db.update_obs_file_quality_by_id(int(draft_id), rid, fwhm=m.get("fwhm_mean"), sky_level=m.get("sky_background"),
                                         star_count=int(m.get("star_count") or 0), rejected_auto=0,
                                         inspection_jd=m.get("inspection_jd"), ra_deg=m.get("ra_deg"), de_deg=m.get("de_deg"),
                                         exptime_sec=m.get("exposure_sec"), elongation_mean=_el_db, noise_sigma=m.get("noise_sigma"))

    rng = np.random.default_rng(117)
    rows = [{"ID": i} for i in range(1, 61)]
    for fwhm_by_id in ({}, {r["ID"]: float(rng.uniform(2.5, 4.0)) for r in rows},
                       {**{r["ID"]: float(rng.uniform(2.5, 4.0)) for r in rows[:40]}, 41: 9.0, 42: float("nan"), 43: -1.0}):
        a, b = _QcDb(), _QcDb()
        assert _old_tail(a, 7, rows, fwhm_by_id) == pcal._qc_reject_and_drift(b, 7, rows, fwhm_by_id)
        assert a.calls == b.calls
    metrics = [{"fwhm_mean": 3.1, "sky_background": 812.5, "star_count": 412, "inspection_jd": 2461000.5, "ra_deg": 209.5,
                "de_deg": 41.2, "exposure_sec": 60.0, "elongation_mean": e, "noise_sigma": 11.2} for e in (1.08, 0.0, None, float("nan"))]
    metrics.append({})
    a, b = _QcDb(), _QcDb()
    for i, m in enumerate(metrics):
        _old_store(a, 7, i, m)
        pcal._store_frame_qc(b, 7, i, m)
    assert repr(a.calls) == repr(b.calls)
    src = (SRC / "pipeline_calibrate.py").read_text(encoding="utf-8")
    assert "suggested_drift_limit_arcmin" not in src and src.count("thr = med * 1.5") == 1
    assert src.count("_alt = master_dark_by_obs_key.get(") == 1


# --- G01-23: one median flat normaliser at build and at calibrate time ------------------------------
# verbatim copy of calibration.normalize_flat_master at HEAD 08e985a
from astropy.io import fits  # noqa: E402

from calibration import _db_equipment_suggests_osc, _valid_bayer_pattern_4  # noqa: E402,F401
from database import VyvarDatabase  # noqa: E402,F401


def _old_normalize_flat_master(
    master: np.ndarray,
    header: fits.Header,
    *,
    db: VyvarDatabase | None = None,
    id_equipments: int | None = None,
    bin_factor: int = 1,
) -> np.ndarray:
    """Scale master flat to median ~1 (global or per-Bayer-tile). Called after resample to light binning."""
    m = np.asarray(master, dtype=np.float32).copy()
    if m.ndim != 2:  # G01-31: masters are 2-D images; the old global branch had no producer
        raise ValueError(f"master flat must be a 2-D image, got shape {m.shape}")

    h, w = int(m.shape[0]), int(m.shape[1])
    hdr_pat = _valid_bayer_pattern_4(str(header.get("BAYERPAT") or ""))
    assumed_pat = False
    pat = hdr_pat
    if pat is None and bin_factor == 1 and db is not None and id_equipments is not None:
        try:
            db_pat = db.get_equipment_bayermask(int(id_equipments))
        except Exception:  # noqa: BLE001
            db_pat = None
        if db_pat and h % 2 == 0 and w % 2 == 0 and h >= 2 and w >= 2:
            pat = db_pat
            assumed_pat = True
    if pat is None and bin_factor == 1 and _db_equipment_suggests_osc(db, id_equipments):
        if h % 2 == 0 and w % 2 == 0 and h >= 2 and w >= 2:
            pat = "RGGB"
            assumed_pat = True

    use_bayer = (
        pat is not None
        and bin_factor == 1
        and h >= 2
        and w >= 2
        and h % 2 == 0
        and w % 2 == 0
    )
    if hdr_pat and bin_factor > 1:
        log_event(
            "Master flat: global normalization is used after binning (binning mixes the Bayer planes)."
        )

    if use_bayer and pat is not None:
        m = m[: h - (h % 2), : w - (w % 2)]
        slices = (
            m[0::2, 0::2],
            m[0::2, 1::2],
            m[1::2, 0::2],
            m[1::2, 1::2],
        )
        labels = (f"[0,0]={pat[0]}", f"[0,1]={pat[1]}", f"[1,0]={pat[2]}", f"[1,1]={pat[3]}")
        medians: list[float] = []
        for sl, lab in zip(slices, labels, strict=True):
            mv = float(np.nanmedian(sl))
            medians.append(mv)
            if np.isfinite(mv) and mv > 0:
                sl /= mv
            else:
                log_event(f"Master flat: warning - invalid median for tile {lab}, division skipped.")
        note = f"Bayer {pat}" + (" (from EQUIPMENTS.BAYERMASK)" if assumed_pat else "")
        log_event(
            "Master flat: normalization per Bayer tile "
            f"({note}); ADU medians before norm: "
            f"{labels[0]}={medians[0]:.6g}, {labels[1]}={medians[1]:.6g}, "
            f"{labels[2]}={medians[2]:.6g}, {labels[3]}={medians[3]:.6g}."
        )
        header["VYFLTNRM"] = ("BAYER4", "Per-tile median normalization (OSC/Bayer)")
        header["VYFLTPAT"] = (pat, "Bayer pattern used for flat normalization")
        gpost = float(np.nanmedian(m))
        if np.isfinite(gpost):
            header["VYFLTMD"] = (gpost, "Global median ADU after per-tile norm")
    else:
        gmed = float(np.nanmedian(m))
        if np.isfinite(gmed) and abs(gmed) > 0:
            m = (m / gmed).astype(np.float32)
        else:
            log_event("Master flat: warning - global median invalid, normalization skipped.")
        log_event(f"Master flat: global median (normalization factor): {gmed:.6g} ADU.")
        header["VYFLTNRM"] = ("GLOBAL", "Global median flat normalization")
        if np.isfinite(gmed):
            header["VYFLTMD"] = (float(gmed), "Global median ADU before normalization")

    # G01-02/G01-03/G01-29: a pixel without a valid flat value is bad (NaN), never replaced by 1.0
    bad = (m <= 0) | ~np.isfinite(m)
    if np.any(bad):
        log_event(f"Master flat: {int(np.sum(bad))} pixels <= 0 or invalid -> bad (NaN in the calibrated frame).")
    m = np.where(bad, np.float32(np.nan), m).astype(np.float32)

    gfin = float(np.nanmedian(m))
    if np.isfinite(gfin) and abs(gfin - 1.0) > 0.02 and abs(gfin) > 0:
        m = (m / gfin).astype(np.float32)
        log_event(f"Master flat: extra scale 1/{gfin:.4f} so that the global median ~ 1.0.")
    mean_a = float(np.nanmean(m))
    log_event(f"Master flat: after normalization median={float(np.nanmedian(m)):.4f}, mean ~ {mean_a:.4f}.")
    return m


def test_g01_23_normalize_flat_master_equal() -> None:
    from calibration import normalize_flat_master

    rng = np.random.default_rng(123)
    h, w = 400, 600
    yy, xx = np.mgrid[0:h, 0:w]
    vign = 1.0 - 0.25 * (((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2)
    base = (20000.0 * vign * rng.normal(1, 0.01, (h, w))).astype(np.float32)
    osc = base.copy()
    osc[0::2, 1::2] *= 1.6
    osc[1::2, 0::2] *= 1.55
    osc[1::2, 1::2] *= 0.7
    for arr in (base, osc):
        a = arr.copy()
        # G01-23 follow-up (s5_dup2): the zero rows were removed - the one normaliser now takes the median of
        # the valid (> 0) pixels, so it differs from the HEAD copy exactly there (tested in test_step5_dup2).
        a[5, 7] = np.nan
        for pat, bf in ((None, 1), ("RGGB", 1), ("GBRG", 1), ("RGGB", 2)):
            h_old, h_new = fits.Header(), fits.Header()
            if pat:
                h_old["BAYERPAT"] = pat
                h_new["BAYERPAT"] = pat
            old = _old_normalize_flat_master(a, h_old, bin_factor=bf)
            new = normalize_flat_master(a, h_new, bin_factor=bf)
            assert old.dtype == new.dtype and np.array_equal(old, new, equal_nan=True), (pat, bf)
            assert list(h_old.items()) == list(h_new.items())


# --- G05-36: one finisher for the disk and the RAM-handoff per-frame export -------------------------


def _proc_frames(root: Path) -> tuple[list, dict, list]:
    import pandas as pd

    from proc_frame_store import round_proc_columns

    rng = np.random.default_rng(536)
    deferred, frame_rows, rows_out = [], {}, []
    for i in range(3):
        n = 50
        df = pd.DataFrame({
            "catalog_id": [str(4000000000000000000 + 31 * k) if k % 9 else "" for k in range(n)],
            "name": [f"S{k}" for k in range(n)],
            "x": rng.uniform(0, 3000, n), "y": rng.uniform(0, 2000, n),
            "dao_flux": rng.uniform(100, 1e6, n), "dao_flux_err": rng.uniform(1, 100, n),
            "bkg_fallback_used": rng.random(n) < 0.1, "sky_pp": rng.uniform(800, 900, n),
        })
        p = root / f"proc_{i:04d}.csv"
        deferred.append((p, round_proc_columns(df)))
        frame_rows[str(p)] = {"jd_mid": 2461000.5 + i / 1440.0, "airmass": 1.2 + 0.01 * i, "aperture_r_px": 5.25,
                              "catalog_match_mode": "wcs"}
        rows_out.append({"file": f"f{i}.fits", "status": "ok", "csv": p.name, "n_stars": n})
    return deferred, frame_rows, rows_out


def _dir_bytes(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(d.iterdir()) if p.is_file()}


def test_g05_36_ram_and_disk_finisher_equal(tmp_path: Path) -> None:
    import pandas as pd

    from frame_export import write_per_frame_index, write_per_frame_outputs
    from gaia_catalog_id import normalize_gaia_source_id_series
    from pipeline_catalog import _finalize_hybrid_bkg_fallback_sidecar
    from proc_frame_store import _vyvar_df_to_csv, write_frame_tables, write_proc_csv

    g, rn = 1.25, 3.6
    # old RAM tail (astrometry_align at HEAD 08e985a)
    a = tmp_path / "old_ram" / "R_60_1"
    a.mkdir(parents=True)
    deferred, frame_rows, rows_out = _proc_frames(a)
    for pcsv, df in deferred:
        if isinstance(df, pd.DataFrame) and "catalog_id" in df.columns:
            df = df.copy()
            df["catalog_id"] = normalize_gaia_source_id_series(df["catalog_id"])
        write_proc_csv(df, pcsv)
    write_frame_tables(frame_rows)
    write_per_frame_index(rows_out, a / "index.csv")
    h_old = _finalize_hybrid_bkg_fallback_sidecar(a, write_sidecar=True, gain=g, read_noise=rn, setup_label=a.name)
    # new RAM tail
    b = tmp_path / "new_ram" / "R_60_1"
    b.mkdir(parents=True)
    deferred, frame_rows, rows_out = _proc_frames(b)
    h_new = write_per_frame_outputs(root=b, index_path=b / "index.csv", rows_out=rows_out, frame_rows=frame_rows,
                                    deferred_csv_writes=deferred, write_sidecar=True, gain=g, read_noise=rn)
    # disk export: the workers write the proc CSVs (_vyvar_df_to_csv), the finisher the rest
    c = tmp_path / "disk" / "R_60_1"
    c.mkdir(parents=True)
    deferred, frame_rows, rows_out = _proc_frames(c)
    for pcsv, df in deferred:
        _vyvar_df_to_csv(df, pcsv)
    h_disk = write_per_frame_outputs(root=c, index_path=c / "index.csv", rows_out=rows_out, frame_rows=frame_rows,
                                     deferred_csv_writes=(), write_sidecar=True, gain=g, read_noise=rn)
    old_b, new_b, disk_b = _dir_bytes(a), _dir_bytes(b), _dir_bytes(c)
    assert set(old_b) >= {"proc_0000.csv", "frames.csv", "index.csv"}
    assert old_b == new_b == disk_b
    assert h_old == h_new == h_disk


# --- G06-21: one vectorised flux -> magnitude ---------------------------------------------------------


def test_g06_21_flux_to_mag_equal() -> None:
    import math

    from stats_core import _flux_to_mag, flux_to_mag

    rng = np.random.default_rng(621)
    f = np.r_[10 ** rng.uniform(-3, 8, 200_000), 0.0, -5.0, np.nan, np.inf, -np.inf, 5e-324, 1e-300]
    rng.shuffle(f)
    # aperture_photometry comp LC copy (full-array log10 + where)
    with np.errstate(divide="ignore", invalid="ignore"):
        old_ap = np.where(np.isfinite(f) & (f > 0), -2.5 * np.log10(f), np.nan)
    new = flux_to_mag(f)
    assert np.array_equal(old_ap, new, equal_nan=True)
    # psf_internal_lc._flux_to_inst_mag copy
    old_psf = np.full(f.shape, np.nan)
    ok = np.isfinite(f) & (f > 0)
    old_psf[ok] = -2.5 * np.log10(f[ok])
    assert np.array_equal(old_psf, new, equal_nan=True)
    # tess / sat_diag: positive finite input, plain -2.5 log10
    pos = f[ok]
    assert np.array_equal(-2.5 * np.log10(pos), flux_to_mag(pos))
    # the scalar survivor agrees with the vector one on the same values (as both are used)
    sample = f[:2000]
    assert np.allclose(np.array([_flux_to_mag(float(v)) for v in sample]), flux_to_mag(sample), rtol=0, atol=1e-12, equal_nan=True)
    assert math.isnan(_flux_to_mag(0.0))
    assert "_flux_to_inst_mag" not in (SRC / "psf_internal_lc.py").read_text(encoding="utf-8")


# --- G11-29: LC / check paths from report_methods; one JD axis offset ------------------------------


def test_g11_29_lc_paths_and_jd_offset() -> None:
    import math
    import re

    import pandas as pd

    from check_star_kmag import check_kmag_sidecar_path as ck_reexport
    from jd_axis_format import jd_offset, jd_series_relative
    from report_methods import check_kmag_sidecar_path, lc_csv_path

    d = Path("/x/photometry/lightcurves")
    for cid in ("1400549806859236864", "V0842_Her", "manual_3"):
        assert lc_csv_path(d, cid) == d / f"lightcurve_{cid}.csv"
        assert lc_csv_path(d, cid, "psf") == d / f"lightcurve_{cid}_psf.csv"
        assert check_kmag_sidecar_path(d, cid) == d / f"check_kmag_{cid}.csv"
    assert ck_reexport is check_kmag_sidecar_path
    pat = re.compile(r"f\"(lightcurve|check_kmag)_\{[^}]+\}(_psf)?\.csv\"")
    for p in SRC.glob("*.py"):
        if p.name != "report_methods.py":
            assert not pat.search(p.read_text(encoding="utf-8")), p.name

    rng = np.random.default_rng(1129)
    for _ in range(50):
        x = 2461000.0 + rng.uniform(0, 3, 300)
        x[rng.random(300) < 0.05] = np.nan
        ok = np.isfinite(x)
        assert float(jd_offset(x[ok]) or 0) == float(np.floor(np.nanmin(x[ok])))  # star_page_m3
        s = pd.Series(x)
        assert float(jd_offset(s) or 0) == float(np.floor(np.nanmin(s)))
        t0_old = math.floor(float(s.min())) if s.notna().any() else 0.0  # variability_plots
        t0_new = jd_offset(s) or 0
        assert t0_new == t0_old and type(t0_new) is type(t0_old)
        rel, off = jd_series_relative(x)
        assert off == int(np.floor(np.nanmin(x))) and np.array_equal(rel, x - float(off), equal_nan=True)
    assert jd_offset(pd.Series([np.nan])) is None and jd_offset(np.array([0.2, 0.9])) is None
    assert jd_offset(np.array([3000.1, 3027.9])) is None  # TESS BTJD is not full JD: no offset
