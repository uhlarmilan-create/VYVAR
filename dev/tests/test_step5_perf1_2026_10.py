"""Step 5 batch B part 1 (perf1): frame / I/O / database performance items.

Every change here must keep the pipeline outputs byte-identical. Each test compares the new
implementation with a copy of the old one (the reference, kept in this file) on realistic synthetic
input, bit for bit, or counts calls / file writes where the item removes repeated work.
"""

from __future__ import annotations

import numpy as np
import pytest

import pipeline  # noqa: F401  app import order


def _bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# --- G01-12 sky surface evaluated in row blocks --------------------------------------------------

def _poly_surface_fit_ref(work, fit_mask, order, step, bg):
    """Copy of pipeline_calibrate._poly_surface_fit at 08e985a (full-frame design evaluation)."""
    h, w = work.shape
    yy_s, xx_s = np.mgrid[0:h:step, 0:w:step]
    z_s = (work[::step, ::step] - float(bg)).astype(np.float64)
    use = fit_mask[::step, ::step] & np.isfinite(z_s)
    min_coef = (order + 1) * (order + 2) // 2
    if int(np.count_nonzero(use)) < min_coef + 10:
        return None
    x_fit, y_fit, z_fit = xx_s[use].astype(np.float64), yy_s[use].astype(np.float64), z_s[use]
    terms = [(i, j) for i in range(order + 1) for j in range(order + 1 - i)]
    coef, *_ = np.linalg.lstsq(np.column_stack([(x_fit**i) * (y_fit**j) for i, j in terms]), z_fit, rcond=None)
    yy_f, xx_f = np.mgrid[0:h, 0:w]
    x_flat, y_flat = xx_f.ravel().astype(np.float64), yy_f.ravel().astype(np.float64)
    surf = sum(c * (x_flat**i) * (y_flat**j) for c, (i, j) in zip(coef, terms, strict=True))
    return np.asarray(surf).reshape(h, w).astype(np.float32), int(z_fit.size)


@pytest.mark.parametrize("shape", [(700, 1037), (513, 256), (1, 300), (257, 3)])
@pytest.mark.parametrize("order", [1, 2])
def test_g01_12_surface_bit_identical(shape, order):
    from pipeline_calibrate import _poly_surface_fit

    rng = np.random.default_rng(1200 + order + shape[0])
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    work = (1000.0 + 0.031 * xx - 0.017 * yy + 2.3e-5 * xx * yy - 1.1e-5 * xx**2
            + rng.normal(0, 12.0, (h, w))).astype(np.float32)
    mask = rng.random((h, w)) > 0.2
    step = 1 if min(h, w) < 10 else 4
    ref = _poly_surface_fit_ref(work, mask, order, step, 997.25)
    new = _poly_surface_fit(work, mask, order, step, 997.25)
    assert ref is not None and new is not None
    assert ref[1] == new[1]
    assert _bits_equal(ref[0], new[0])


# --- G05-07 / G05-21 / G04-22: no full-frame diagnostics when DEBUG is off ---------------------

class _CountingFrame:
    """Stands in for the full frame ``arr``: counts every conversion to an ndarray."""

    def __init__(self, a):
        self._a = a
        self.n_array = 0

    def __array__(self, dtype=None, copy=None):
        self.n_array += 1
        return self._a if dtype is None else self._a.astype(dtype)


def _mr_dao_detect_inputs():
    import pandas as pd
    from astropy.io import fits
    from astropy.wcs import WCS

    from config import AppConfig

    rng = np.random.default_rng(507)
    data0 = rng.normal(0.0, 5.0, (96, 96)).astype(np.float32)
    yy, xx = np.mgrid[0:96, 0:96]
    data0 += (800.0 * np.exp(-((xx - 40.3) ** 2 + (yy - 51.7) ** 2) / (2 * 1.5**2))).astype(np.float32)
    hdr = fits.Header()
    hdr["FILENAME"] = "f001.fits"
    return dict(
        _base_fw_m=3.5, bfac=1, cfg=AppConfig(), data0=data0, data_dao=data0,
        equipment_saturate_adu=None, fallback_saturate_adu=60000.0, frame_name="f001.fits",
        fwhm_eff=3.5, hdr=hdr, m_valid=pd.DataFrame(columns=["ra_deg", "dec_deg"]),
        match_sep_arcsec=8.0, rms_conv=5.0, sat_diag_ctx=None, std_dao=5.0, wcs_obj=WCS(naxis=2),
    )


def test_g05_07_g05_21_no_full_frame_stats_when_debug_off(caplog):
    import logging

    import pipeline_catalog

    kw = _mr_dao_detect_inputs()
    frame = _CountingFrame(kw["data0"])
    caplog.set_level(logging.INFO, logger="pipeline_catalog")
    pipeline_catalog._mr_dao_detect(arr=frame, **kw)
    assert frame.n_array == 0
    caplog.set_level(logging.DEBUG, logger="pipeline_catalog")
    pipeline_catalog._mr_dao_detect(arr=frame, **kw)
    assert frame.n_array > 0
    assert any("DAO input: f001.fits" in r.getMessage() for r in caplog.records)
    # the per-frame sigma_pp diagnostic is retired (its only consumer was this DEBUG line)
    assert not hasattr(pipeline_catalog, "_dao_noise_sigma_adu")
    assert not hasattr(pipeline_catalog, "_pixel_noise_sigma_pp_adu")


# --- G05-20 star peaks from boxes at the centres; clip level without a float64 frame copy -------

def _box_peaks_at_centroids_ref(arr, x, y, *, half=3):
    """Copy of pipeline_catalog._box_peaks_at_centroids at 08e985a (full-frame maximum_filter)."""
    from scipy.ndimage import maximum_filter

    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    a = np.asarray(np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0))
    mf = maximum_filter(a, size=int(2 * int(half) + 1), mode="nearest")
    xi = np.rint(xa).astype(np.intp, copy=False)
    yi = np.rint(ya).astype(np.intp, copy=False)
    h, w = mf.shape
    return np.asarray(mf[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)], dtype=np.float64)


def _detect_empirical_clip_level_adu_ref(data):
    """Copy of pipeline_catalog._detect_empirical_clip_level_adu at 08e985a."""
    import math

    arr = np.asarray(data, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    vmax = float(np.max(finite))
    if not math.isfinite(vmax) or vmax <= 0:
        return None
    if int(np.count_nonzero(np.abs(finite - vmax) <= 0.5)) >= 8:
        return vmax
    return None


def _star_frame(seed, h=410, w=530, dtype=np.float32):
    rng = np.random.default_rng(seed)
    a = rng.normal(1200.0, 30.0, (h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(60):
        cx, cy = rng.uniform(-3, w + 3), rng.uniform(-3, h + 3)
        a += rng.uniform(500, 60000) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.6**2))
    a = a.astype(dtype)
    a[rng.integers(0, h, 300), rng.integers(0, w, 300)] = np.nan
    a[0, 5] = np.inf
    a[h - 1, w - 2] = -np.inf
    a[2, 0:3] = np.nan
    return a


@pytest.mark.parametrize("half", [1, 3, 5])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_g05_20_box_peaks_bit_identical(half, dtype):
    from pipeline_catalog import _box_peaks_at_centroids

    a = _star_frame(2000 + half, dtype=dtype)
    h, w = a.shape
    rng = np.random.default_rng(7 + half)
    x = np.concatenate([rng.uniform(-10, w + 10, 3000), [0, w - 1, -0.5, 0.5, w - 0.5, 2.5, 3.5, -40, w + 40]])
    y = np.concatenate([rng.uniform(-10, h + 10, 3000), [0, h - 1, 0.49, -0.51, h - 0.5, 1.5, 2.5, h + 40, -40]])
    ref = _box_peaks_at_centroids_ref(a, x, y, half=half)
    new = _box_peaks_at_centroids(a, x, y, half=half)
    assert _bits_equal(ref, new)


def test_g05_20_box_peaks_tiny_frame_and_empty():
    from pipeline_catalog import _box_peaks_at_centroids

    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    x, y = np.array([0.0, 2.0, 1.0, 9.0]), np.array([0.0, 1.0, -3.0, 1.0])
    assert _bits_equal(_box_peaks_at_centroids_ref(a, x, y), _box_peaks_at_centroids(a, x, y))
    empty = _box_peaks_at_centroids(a, np.array([]), np.array([]))
    assert _bits_equal(np.zeros([], dtype=np.float64), empty)


def test_g05_20_clip_level_identical():
    from pipeline_catalog import _detect_empirical_clip_level_adu

    cases = []
    for dtype in (np.float32, np.float64):
        a = _star_frame(31, dtype=dtype)
        cases.append(a)                                   # no clip
        b = a.copy()
        b[100:103, 200:203] = 65535.0                     # 9 px at the clip -> clipped
        cases.append(b)
        c = a.copy()
        c[10, 10:17] = 65535.0                            # 7 px only -> not clipped
        c[11, 10] = 65534.4                               # 0.6 below: does not count
        cases.append(c)
        d = a.copy()
        d[10, 10:17] = 65535.0
        d[11, 10] = 65534.5                               # exactly 0.5 below: counts
        cases.append(d)
        e = np.full((40, 40), 0.75, dtype=dtype)          # small vmax (< 1): float32 rounding edge
        e[:3, :3] = 0.25
        cases.append(e)
        cases.append(np.full((8, 8), np.nan, dtype=dtype))
        cases.append(np.full((8, 8), -3.0, dtype=dtype))
    big = np.full((20, 20), 3.0e9, dtype=np.float32)      # float32 spacing 256 near 3e9
    big[0, 0] = np.float32(3.0e9) + np.float32(256.0)
    cases.append(big)
    cases.append(np.array([[65535, 65535, 65535, 65535], [65535, 65535, 65535, 65535]], dtype=np.uint16))
    for arr in cases:
        assert _detect_empirical_clip_level_adu_ref(arr) == _detect_empirical_clip_level_adu(arr)


# --- G05-30 the star-count reference pick is gone: the MASTERSTAR always replaces the reference ---

def _write_ms(path, *, wcs=True):
    from astropy.io import fits

    hdr = fits.Header()
    if wcs:
        hdr.update({"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRVAL1": 150.0, "CRVAL2": 30.0,
                    "CRPIX1": 16.0, "CRPIX2": 16.0, "CDELT1": -0.0005, "CDELT2": 0.0005})
    fits.PrimaryHDU(np.full((32, 32), 7.0, dtype=np.float32), header=hdr).writeto(path)
    return path


def _al_ms_kwargs(tmp_path, **over):
    from config import AppConfig

    kw = dict(
        _cat_info_root={}, _cfg_align=AppConfig(), _masterstar_built=False, _prog=lambda m: None,
        _ps_root=tmp_path, ap=tmp_path, build_masterstar_and_catalogs=False, detrended_root=tmp_path,
        draft_id=None, equip_sat_adu=None, has_wcs=False, id_equipment=None, job={},
        n_comparison_stars=0, n_files=3, obs_group_key="", platesolve_dir=tmp_path,
        ref_data="SENTINEL-DATA", ref_fp=tmp_path / "processed_light_007.fits",
        ref_hdr="SENTINEL-HDR", require_non_variable_comparisons=True,
        sibling_recovery_use_masterstar=False,
    )
    kw.update(over)
    return kw


def test_g05_30_reference_is_always_the_masterstar(tmp_path, monkeypatch):
    import astrometry_align as aa
    import masterstar_build
    import pipeline_catalog

    ms = _write_ms(tmp_path / "MASTERSTAR.fits")
    # build path: the reference passed in is replaced by the built MASTERSTAR
    monkeypatch.setattr(masterstar_build, "generate_masterstar_and_catalog",
                        lambda **k: {"masterstar_fits": str(ms)})
    out = aa._al_masterstar_for_reference(**_al_ms_kwargs(tmp_path, build_masterstar_and_catalogs=True))
    _, built, has_wcs, ref_data, ref_fp, ref_hdr = out
    assert built and has_wcs and ref_fp == ms.resolve()
    assert ref_data.shape == (32, 32) and ref_hdr["CTYPE1"] == "RA---TAN"
    # sibling recovery with an existing MASTERSTAR: replaced as well
    out = aa._al_masterstar_for_reference(**_al_ms_kwargs(tmp_path, sibling_recovery_use_masterstar=True))
    assert out[4] == ms.resolve() and out[2] is True and out[5]["CTYPE1"] == "RA---TAN"
    # every other path stops the run (MASTER-LOCK): the input reference is never used
    with pytest.raises(RuntimeError, match="MASTER-LOCK"):
        aa._al_masterstar_for_reference(**_al_ms_kwargs(tmp_path))
    ms.unlink()
    _write_ms(tmp_path / "MASTERSTAR.fits", wcs=False)
    with pytest.raises(RuntimeError, match="MASTER-LOCK"):
        aa._al_masterstar_for_reference(**_al_ms_kwargs(tmp_path, sibling_recovery_use_masterstar=True))
    assert not hasattr(pipeline_catalog, "_pick_reference_frame_by_star_count")


# --- G12-20 / G01-13 / G12-42 manifest: id -> folder cache, one atomic write per stage ----------

class _ArchiveDb:
    def __init__(self, root):
        self._root = root

    def resolve_archive_root(self):
        return self._root


def _make_field(archive, name, draft_id, n_files=40):
    import json

    d = archive / "Drafts" / name
    (d / "Raw" / "lights").mkdir(parents=True)
    files = []
    for i in range(n_files):
        fp = d / "Raw" / "lights" / f"L_{i:03d}.fits"
        files.append({"obs_file_id": i + 1, "file_path": str(fp), "imagetyp": "light", "filter": "V",
                      "inspection": {"fwhm": 3.0 + 0.01 * i}})
    files.append("not-a-dict")
    files.append({"obs_file_id": 7, "file_path": str(d / "dup.fits"), "imagetyp": "light"})  # duplicate id
    manifest = {"schema_version": 3, "draft_id": int(draft_id), "calibration_mode": "vyvar_calibrated",
                "status": "INGESTED", "files": files}
    (d / "draft_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return d


def _stage_updates(db, did, field):
    """A calibration-like stage: per-frame updates by path and id, interleaved reads and patches."""
    import draft_provenance as dp

    out = []
    for i in range(0, 40, 3):
        fp = field / "Raw" / "lights" / f"L_{i:03d}.fits"
        out.append(dp.update_manifest_file_entry(db, did, file_path=str(fp), qc={"hfr": 1.5 + i, "stars": i}))
        out.append(dp.update_manifest_file_entry(
            db, did, file_path=str(field / "Raw" / "lights" / ".." / "lights" / fp.name),
            cal_stage={"cal_stage": "PURE", "cal_datasum": str(1000 + i), "cal_stage_ut": "2026-10-30T00:00:00",
                       "cal_pstbg": 12.5}))
        out.append(dp.update_manifest_file_entry(db, did, obs_file_id=i + 1,
                                                 inspection={"rejected_auto": i % 2, "drift": 0.1 * i}))
        if i == 9:
            out.append(len(dp.light_rows_from_manifest(db, did, imagetyp="light") or []))
            dp.patch_draft_manifest(field, did, status="CALIBRATING")
            dp.set_manifest_light_qc_decisions(db, did, {str(field / "Raw" / "lights" / "L_004.fits"): "fwhm"})
    out.append(dp.update_manifest_file_entry(db, did, obs_file_id=7, inspection={"star_count": 77}))
    out.append(dp.update_manifest_file_entry(db, did, obs_file_id=999, inspection={"star_count": 1}))
    out.append(dp.update_manifest_file_entry(db, did, file_path=str(field / "missing.fits"), qc={"hfr": 1.0}))
    rows = dp.light_rows_from_manifest(db, did, imagetyp="light") or []
    out.append([(r.get("ID"), r.get("REJECTED_AUTO"), r.get("IS_REJECTED")) for r in rows])
    return out


def _manifest_without_stamp(field):
    import json

    m = json.loads((field / "draft_manifest.json").read_text(encoding="utf-8"))
    m.pop("updated_utc", None)
    return m


def test_g01_13_batched_stage_equals_per_call_writes(tmp_path, monkeypatch):
    import draft_provenance as dp

    arch_a, arch_b = tmp_path / "A", tmp_path / "B"
    fa, fb = _make_field(arch_a, "BO_CVn_20261020_001", 534), _make_field(arch_b, "BO_CVn_20261020_001", 534)
    db_a, db_b = _ArchiveDb(arch_a), _ArchiveDb(arch_b)
    writes = []
    real_write = dp._write_manifest_file
    monkeypatch.setattr(dp, "_write_manifest_file", lambda p, payload: (writes.append(p), real_write(p, payload)))

    res_a = _stage_updates(db_a, 534, fa)                    # per-call writes (old behaviour)
    n_per_call = len(writes)
    writes.clear()
    with dp.manifest_batch(db_b, 534):
        res_b = _stage_updates(db_b, 534, fb)
        assert writes == []                                  # nothing written during the stage
    assert len(writes) == 1                                  # one write at the end of the stage
    assert n_per_call > 30
    assert res_a == res_b
    ma, mb = _manifest_without_stamp(fa), _manifest_without_stamp(fb)
    assert res_a.count(True) > 30 and res_a.count(False) == 2
    import json

    assert json.dumps(ma).replace(str(arch_a), "X") == json.dumps(mb).replace(str(arch_b), "X")
    assert not list(fb.glob("*.tmp"))                         # atomic write leaves no temp file
    assert dp._MANIFEST_BATCHES == {}


def test_g01_13_batch_flushes_on_error_and_nests(tmp_path):
    import draft_provenance as dp

    arch = tmp_path / "A"
    field = _make_field(arch, "X_20261020_001", 12)
    db = _ArchiveDb(arch)
    with pytest.raises(RuntimeError):
        with dp.manifest_batch(db, 12):
            with dp.manifest_batch(db, 12):                  # nested: joins the outer batch
                dp.update_manifest_file_entry(db, 12, obs_file_id=3, inspection={"fwhm": 9.5})
            assert dp.load_draft_manifest(field)["files"][2]["inspection"]["fwhm"] == 9.5
            raise RuntimeError("stage failed")
    assert _manifest_without_stamp(field)["files"][2]["inspection"]["fwhm"] == 9.5
    assert dp._MANIFEST_BATCHES == {}
    with dp.manifest_batch(None, 12), dp.manifest_batch(db, None):   # no db / id: no batch
        assert dp._MANIFEST_BATCHES == {}


def test_g01_13_calibration_stage_functions_are_batched():
    import pipeline_calibrate as pc

    for fn in (pc.sync_obs_files_drift_arcmin_for_draft, pc.apply_perf10_dao_qc_to_obs_files,
               pc.run_draft_ram_calibration_qc_to_obs_files, pc._qc_enrich_calibrated_in_place):
        assert getattr(fn, "__wrapped__", None) is not None, fn.__name__


def test_g12_20_draft_dir_lookup_cached_and_checked(tmp_path, monkeypatch):
    import shutil

    import draft_provenance as dp

    arch = tmp_path / "A"
    for k in range(6):
        _make_field(arch, f"T{k}_20261020_001", 100 + k, n_files=2)
    calls = []
    real_iter = dp.iter_draft_archive_dirs
    monkeypatch.setattr(dp, "iter_draft_archive_dirs", lambda r: (calls.append(1), real_iter(r))[1])
    first = dp.draft_dir_for_id(arch, 103)
    assert first.name == "T3_20261020_001"
    for _ in range(20):
        assert dp.draft_dir_for_id(arch, 103) == first
        assert dp.resolve_draft_dir_for_id(_ArchiveDb(arch), 103) == first.resolve()
    assert len(calls) == 1                                   # one archive scan, then the cache
    # the folder is deleted and its name reused by another field: never returned for the old id
    shutil.rmtree(first)
    _make_field(arch, "T3_20261020_001", 200, n_files=2)
    assert dp.draft_dir_for_id(arch, 103) == (arch / "Drafts" / "draft_000103").resolve()
    assert dp.draft_dir_for_id(arch, 200).name == "T3_20261020_001"


def test_g12_42_no_archive_wide_manifest_search(tmp_path):
    import draft_provenance as dp

    arch = tmp_path / "A"
    _make_field(arch, "T_20261020_001", 5, n_files=1)
    stray = arch / "Some" / "Deep" / "Tree"
    stray.mkdir(parents=True)
    (stray / "draft_manifest.json").write_text('{"draft_id": 77, "files": []}', encoding="utf-8")
    assert dp.resolve_draft_dir_for_id(_ArchiveDb(arch), 77) is None
    assert dp.resolve_draft_dir_for_id(_ArchiveDb(arch), 5).name == "T_20261020_001"


# --- G05-31 / G13-15: the per-frame site / centre lookups use facts resolved once ---------------

def _site_db(tmp_path, *, with_location=True, centre=(201.1234, 41.4321)):
    from types import SimpleNamespace

    import draft_provenance as dp
    from database import VyvarDatabase

    db_path = tmp_path / "vyvar.sqlite3"
    archive = tmp_path / "Archive"
    db = VyvarDatabase(str(db_path))
    loc = db.insert_location(place_name="Jirny", latitude=50.112, longitude=14.698, altitude=275.0)
    cfg = SimpleNamespace(database_path=str(db_path), archive_root=str(archive), observer_lat=0.0,
                          observer_lon=0.0, observer_alt_m=0.0)
    did = dp.create_draft_manifest(archive, {"id_location": loc if with_location else None,
                                             "object_name": "BO CVn", "observation_start_jd": 2460970.4})
    db = VyvarDatabase.for_config(cfg)
    if centre is not None:
        # S6-FIX CENTRE-PER-SETUP: the centre of the headers' setup (NoFilter, 60 s, bin 1)
        dp.patch_draft_manifest(dp.resolve_draft_dir_for_id(db, did), did, center_by_setup={
            "NoFilter_60_1": {"ra_deg": centre[0], "de_deg": centre[1], "source": "MASTERSTAR_WCS"}})
    return db, did, cfg


def _site_headers():
    from astropy.io import fits

    base = {"DATE-OBS": "2026-10-20T21:14:05.5", "EXPTIME": 60.0}
    h_plain = fits.Header(base)                                         # centre from the draft
    h_site = fits.Header(dict(base, SITELAT=49.0, SITELONG=17.0, SITEELEV=300.0,
                              OBJCTRA="13 24 30", OBJCTDEC="+41 25 00"))  # header site disagrees
    h_wcs = fits.Header(dict(base, CTYPE1="RA---TAN", CTYPE2="DEC--TAN", CRVAL1=201.0, CRVAL2=41.0,
                             CRPIX1=500.0, CRPIX2=400.0, CDELT1=-0.0004, CDELT2=0.0004,
                             NAXIS=2, NAXIS1=1000, NAXIS2=800))
    return [h_plain, h_site, h_wcs]


@pytest.mark.parametrize("with_location", [True, False])
@pytest.mark.parametrize("centre", [(201.1234, 41.4321), None])
def test_g05_31_draft_site_facts_give_identical_time_and_airmass(tmp_path, with_location, centre):
    import pickle

    from param_resolver import DraftSiteFacts
    from pipeline_catalog import _extract_airmass_from_header
    from time_utils import compute_time_columns

    db, did, cfg = _site_db(tmp_path, with_location=with_location, centre=centre)
    facts = pickle.loads(pickle.dumps(DraftSiteFacts.from_db(db, did)))   # crosses the process pool
    if with_location and centre is not None:          # the draft paths are really exercised
        assert compute_time_columns(_site_headers()[0], db=facts, draft_id=did, cfg=cfg)["bjd_tdb_mid"]
    for hdr in _site_headers():
        ref_t = compute_time_columns(hdr, db=db, draft_id=did, cfg=cfg)
        new_t = compute_time_columns(hdr, db=facts, draft_id=did, cfg=cfg)
        assert ref_t == new_t
        ref_a = _extract_airmass_from_header(hdr, cfg=cfg, db=db, draft_id=did)
        new_a = _extract_airmass_from_header(hdr, cfg=cfg, db=facts, draft_id=did)
        assert ref_a == new_a or (ref_a != ref_a and new_a != new_a)
    if with_location:
        assert facts.location == (50.112, 14.698, 275.0)
    with pytest.raises(ValueError):
        facts.get_draft_location_id(did + 1)


def test_g05_31_worker_does_not_open_the_database(monkeypatch):
    import inspect

    import pipeline_catalog

    src = inspect.getsource(pipeline_catalog._pf_annotate_and_measure)
    assert "VyvarDatabase(" not in src and "VyvarDatabase.for_config" not in src
    assert 'st.get("draft_site")' in src


# --- G08-15 / G06-33: rig scintillation / floor resolved once per run ---------------------------

def test_g08_15_g06_33_rig_params_resolved_once(tmp_path, monkeypatch):
    import inspect
    from types import SimpleNamespace

    from astropy.io import fits

    import aperture_photometry as ap
    import aperture_photometry_target as apt
    import sigma_budget
    import sigma_floor_core

    calls = {"scint": 0, "ssm": 0}
    real_scint = sigma_budget.resolve_rig_scintillation_params
    real_ssm = sigma_floor_core.resolve_sigma_sys_mag

    def _scint(**kw):
        calls["scint"] += 1
        return real_scint(**kw)

    def _ssm(*a, **kw):
        calls["ssm"] += 1
        return real_ssm(*a, **kw)

    monkeypatch.setattr(sigma_budget, "resolve_rig_scintillation_params", _scint)
    monkeypatch.setattr(sigma_floor_core, "resolve_sigma_sys_mag", _ssm)
    cfg = SimpleNamespace(database_path="", archive_root=str(tmp_path))
    state = SimpleNamespace(rig_scint_by_dir={}, rig_sigma_sys_mag=None, equipment_id=None, _cfg=cfg,
                            obs_group="V|60|1", _ms_header=fits.Header({"EXPTIME": 60.0}),
                            site_ok=True, site_alt=275.0)
    out_dir = tmp_path / "photometry"
    ref = real_scint(draft_id=None, setup="V|60|1", exposure_s=60.0, cfg=cfg,
                     pipeline_meta={"observer_location": {"alt_m": 275.0}})
    got = [ap._rig_scint_params(state, out_dir) for _ in range(25)]   # night level + 24 targets
    ssm = [ap._rig_sigma_sys_mag(state) for _ in range(25)]
    assert calls == {"scint": 1, "ssm": 1}
    assert all(g is got[0] for g in got)
    assert (got[0].telescope_diameter_m != got[0].telescope_diameter_m) and got[0].altitude_m == ref.altitude_m
    assert got[0].exposure_s == ref.exposure_s and got[0].source_notes == ref.source_notes
    assert ssm == [real_ssm(None, cfg)] * 25
    src = inspect.getsource(apt._tgt_time_and_errors)
    assert "resolve_rig_scintillation_params" not in src and "resolve_sigma_sys_mag" not in src
    fields = ap._AperturePhotometryState.__dataclass_fields__
    assert "rig_scint_by_dir" in fields and "rig_sigma_sys_mag" in fields


# --- G02-13 raw-peak placement + peaks vectorised; G02-12 Check A once per sample ---------------

def _placement_loop_ref(arr, pl_x, pl_y):
    """The per-star loop of sat_diag.measure_raw_peaks_frame at 08e985a (finite placements)."""
    import sat_diag as sd

    n = pl_x.size
    px, py, peaks = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        placed_x, placed_y = float(pl_x[i]), float(pl_y[i])
        seed_peak = sd.box_peak_max(arr, placed_x, placed_y)
        com = sd._centroid_com_cutout(arr, placed_x, placed_y, sd.DRIFT_CENTROID_BOX_HALF)
        if com is not None:
            trial_peak = sd.box_peak_max(arr, com[0], com[1])
            if trial_peak >= max(seed_peak * 0.75, 2500.0):
                placed_x, placed_y = float(com[0]), float(com[1])
        px[i], py[i] = placed_x, placed_y
        peaks[i] = sd.box_peak_max(arr, placed_x, placed_y)
    return px, py, peaks


def _raw_frame(seed, h=700, w=1040, n_stars=600):
    rng = np.random.default_rng(seed)
    a = rng.normal(900.0, 25.0, (h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    xs, ys = rng.uniform(-6, w + 6, n_stars), rng.uniform(-6, h + 6, n_stars)
    for cx, cy, f in zip(xs, ys, rng.lognormal(9.0, 1.3, n_stars), strict=True):
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        m = r2 < 150
        a[m] += f * np.exp(-r2[m] / (2 * 1.7**2))
    a = np.minimum(np.round(a), 65535.0)                      # integer ADU, clipped cores
    a[rng.integers(0, h, 400), rng.integers(0, w, 400)] = np.nan
    a[100:103, 0:12] = np.nan                                 # NaN block at an edge
    return a, xs, ys


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_g02_13_placement_and_peaks_bit_identical(seed):
    import sat_diag as sd

    arr, xs, ys = _raw_frame(seed)
    h, w = arr.shape
    rng = np.random.default_rng(100 + seed)
    pl_x = np.concatenate([xs + rng.normal(0, 1.2, xs.size), rng.uniform(-30, w + 30, 400),
                           [0.5, 1.5, w - 0.5, 4.5, 5.5, -3.6, w + 3.4, 2.5e12, -7e15]])
    pl_y = np.concatenate([ys + rng.normal(0, 1.2, ys.size), rng.uniform(-30, h + 30, 400),
                           [0.5, h - 1.5, 5.5, h - 5.5, 6.5, 40.0, 41.0, 3.0, 9.0]])
    ref = _placement_loop_ref(arr, pl_x, pl_y)
    fx, fy = sd._refine_placement_com(arr, pl_x, pl_y)
    new = (fx, fy, sd._box_peaks_nanmax(arr, fx, fy, sd.PEAK_BOX_HALF))
    for r, n_ in zip(ref, new, strict=True):
        assert _bits_equal(r, n_)
    moved = np.count_nonzero(ref[0] != pl_x)
    assert moved > 100                                         # the COM rule is really exercised


def test_g02_13_measure_raw_peaks_frame_matrix_and_wcs_paths():
    """End to end: the vectorised frame function equals the reference loop on its placements."""
    from astropy.io import fits

    import sat_diag as sd

    arr, xs, ys = _raw_frame(9)
    h, w = arr.shape
    raw_hdr = fits.Header({"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRVAL1": 200.0, "CRVAL2": 40.0,
                           "CRPIX1": w / 2, "CRPIX2": h / 2, "CD1_1": -2.0e-4, "CD1_2": 0.0,
                           "CD2_1": 0.0, "CD2_2": 2.0e-4, "NAXIS": 2, "NAXIS1": w, "NAXIS2": h})
    from astropy.wcs import WCS

    ra, dec = WCS(raw_hdr).all_pix2world(xs, ys, 0)
    ax = xs + 0.3
    ax[::7] = np.nan                                          # some stars fall back to WCS + drift
    peaks, px, py, drift = sd.measure_raw_peaks_frame(
        arr, raw_hdr, ra_deg=ra, dec_deg=dec, aligned_x=ax, aligned_y=ys - 0.2,
        aligned_peak=np.full(xs.size, 9000.0))
    wx, wy = WCS(raw_hdr).all_world2pix(ra, dec, 0)
    pl_x = np.where(np.isfinite(ax), ax, wx + drift.dx)
    pl_y = np.where(np.isfinite(ax), ys - 0.2, wy + drift.dy)
    ref = _placement_loop_ref(np.asarray(arr, dtype=np.float64), pl_x, pl_y)
    assert _bits_equal(ref[0], px) and _bits_equal(ref[1], py) and _bits_equal(ref[2], peaks)


def _derive_ceiling_ref(paths):
    """Copy of the histogram branch of sat_diag.derive_ceiling_from_paths at 08e985a."""
    import math
    from collections import Counter

    from astropy.io import fits

    import sat_diag as sd

    counts = Counter()
    max_px, bitpix_ceil = -math.inf, None
    for fp in paths:
        with fits.open(fp, memmap=False) as hdul:
            if bitpix_ceil is None:
                bitpix_ceil = sd.bitpix_container_ceiling(hdul[0].header)
            arr = sd.image_adu_array(hdul[0])
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                continue
            max_px = max(max_px, float(np.max(finite)))
            vals, cnts = np.unique(finite.astype(np.int64), return_counts=True)
            for v, c in zip(vals, cnts, strict=False):
                counts[int(v)] += int(c)
    if not counts:
        return sd.PileupResult(False, None, 0, 0, max_px, len(paths), bitpix_ceil)
    v_max = max(counts)
    n_max = counts[v_max]
    lower = [v for v in counts if v < v_max]
    n_sh = counts[max(lower)] if lower else 0
    pile = n_max >= sd.N_PILEUP_MIN and bitpix_ceil is not None and v_max >= int(bitpix_ceil) - 1 and (
        n_sh <= 0 or n_max >= sd.PILEUP_RATIO * n_sh)
    if pile:
        return sd.PileupResult(True, float(v_max), n_max, n_sh, max_px, len(paths), bitpix_ceil)
    return sd.PileupResult(False, None, n_max, n_sh, max_px, len(paths), bitpix_ceil)


def _write_int_frames(tmp_path, kind):
    from astropy.io import fits

    rng = np.random.default_rng(len(kind))
    paths = []
    for k in range(4):
        if kind == "pileup":
            a = rng.integers(800, 4000, (120, 160)).astype(np.uint16)
            a[:15, :15] = 65535
            hdu = fits.PrimaryHDU(a)
        elif kind == "int32_negative":
            a = rng.integers(-500, 3000, (120, 160)).astype(np.int32)
            if k == 2:
                a = np.abs(a)
            hdu = fits.PrimaryHDU(a)
        elif kind == "uint32_big":
            a = rng.integers(0, 3_000_000, (120, 160)).astype(np.uint32)
            hdu = fits.PrimaryHDU(a)
        else:  # "empty-ish": a constant frame
            hdu = fits.PrimaryHDU(np.full((50, 50), 1000, dtype=np.int16))
        fp = tmp_path / f"{kind}_{k}.fits"
        hdu.writeto(fp)
        paths.append(fp)
    return paths


@pytest.mark.parametrize("kind", ["pileup", "int32_negative", "uint32_big", "flat"])
def test_g02_12_histogram_bincount_identical(tmp_path, kind):
    import sat_diag as sd

    paths = _write_int_frames(tmp_path, kind)
    assert sd.derive_ceiling_from_paths(paths) == _derive_ceiling_ref(paths)


def test_g02_12_check_a_runs_once_for_both_sat_diag_calls(tmp_path, monkeypatch):
    from astropy.io import fits

    import sat_diag as sd

    lights = tmp_path / "Raw" / "lights" / "V"
    lights.mkdir(parents=True)
    for k, fp in enumerate(_write_int_frames(tmp_path, "pileup")):
        fp.rename(lights / f"L_{k}.fits")
    calls = []
    real = sd.derive_ceiling_from_paths
    monkeypatch.setattr(sd, "derive_ceiling_from_paths", lambda p: (calls.append(1), real(p))[1])
    sd._PILEUP_CACHE.clear()
    ctx_ms = sd.run_sat_diag(tmp_path, equipment_adu=60000.0)                    # MASTERSTAR build
    ctx_fe = sd.run_sat_diag(tmp_path, equipment_adu=None,                        # per-frame export
                             hdr=fits.Header({"SATURATE": 65000.0, "BITPIX": 16}))
    assert len(calls) == 1
    ctx_ms_ref = sd.resolve_sat_limit(hdr=fits.getheader(sorted(lights.glob("*.fits"))[0]),
                                      pileup=real(sd.sample_light_paths(tmp_path)), equipment_adu=60000.0)
    ctx_fe_ref = sd.resolve_sat_limit(hdr=fits.Header({"SATURATE": 65000.0, "BITPIX": 16}),
                                      pileup=real(sd.sample_light_paths(tmp_path)), equipment_adu=None)
    assert ctx_ms == ctx_ms_ref and ctx_fe == ctx_fe_ref
    assert ctx_ms.pileup is not ctx_fe.pileup                 # each caller gets its own copy
    # a changed frame is a new sample
    fits.PrimaryHDU(np.full((120, 160), 7, dtype=np.uint16)).writeto(lights / "L_0.fits", overwrite=True)
    sd.run_sat_diag(tmp_path, equipment_adu=60000.0)
    assert len(calls) == 2


# --- G03-43 pass-2 dedup: KD-tree candidates + the same hypot test ------------------------------

def _dedup_ref(p1x, p1y, qx, qy, dedup):
    # s5_dup2 (Milan 2026-10-30): non-finite pass-1 centroids are ignored (the old full scan let one NaN
    # centroid make nothing a duplicate - that case is tested in test_step5_dup2)
    fin = np.isfinite(p1x) & np.isfinite(p1y)
    p1x, p1y = p1x[fin], p1y[fin]
    out = []
    for xf, yf in zip(qx, qy, strict=True):
        if p1x.size:
            d = np.hypot(p1x - float(xf), p1y - float(yf))
            out.append(bool(float(np.min(d)) < float(dedup)))
        else:
            out.append(False)
    return np.asarray(out, dtype=bool)


@pytest.mark.parametrize("case", ["random", "boundary", "nan_p1", "inf_p1", "nan_q", "empty"])
def test_g03_43_dedup_identical(case):
    from pipeline_catalog import _pass2_within_dedup_of_pass1

    rng = np.random.default_rng(43)
    p1x, p1y = rng.uniform(0, 4000, 30000), rng.uniform(0, 2800, 30000)
    qx, qy = rng.uniform(0, 4000, 4000), rng.uniform(0, 2800, 4000)
    if case == "boundary":                       # exactly at / one ulp around the dedup radius
        ang = rng.uniform(0, 2 * np.pi, 3000)
        r = np.concatenate([np.full(1000, 3.0), np.nextafter(np.full(1000, 3.0), 0), np.nextafter(np.full(1000, 3.0), 9)])
        idx = rng.integers(0, p1x.size, 3000)
        qx = np.concatenate([qx, p1x[idx] + r * np.cos(ang)])
        qy = np.concatenate([qy, p1y[idx] + r * np.sin(ang)])
        qx = np.concatenate([qx, p1x[:500] + 3.0, p1x[500:1000]])
        qy = np.concatenate([qy, p1y[:500], p1y[500:1000] - 3.0])
    elif case == "nan_p1":
        p1x[17] = np.nan
    elif case == "inf_p1":
        p1x[17], p1y[99] = np.inf, -np.inf
    elif case == "nan_q":
        qx[::50] = np.nan
        qy[::77] = np.inf
    elif case == "empty":
        p1x, p1y = p1x[:0], p1y[:0]
    ref = _dedup_ref(p1x, p1y, qx, qy, 3.0)
    new = _pass2_within_dedup_of_pass1(p1x, p1y, qx, qy, 3.0)
    assert np.array_equal(ref, new)
    if case in ("random", "boundary", "inf_p1"):
        assert ref.any() and not ref.all()


# --- G05-43 field cone radius: one SkyCoord for the whole border --------------------------------

def _field_radius_ref(w, h, wpx):
    import math

    import astropy.units as u
    from astropy.coordinates import SkyCoord

    center = SkyCoord.from_pixel((wpx - 1) / 2.0, (h - 1) / 2.0, wcs=w, origin=0)
    max_sep_deg = 0.0
    step = max(1, int(round(float(max(wpx, h)) / 120.0)))
    pixels = {(0, 0), (wpx - 1, 0), (0, h - 1), (wpx - 1, h - 1)}
    for x in range(0, wpx, step):
        pixels |= {(x, 0), (x, h - 1)}
    for y in range(0, h, step):
        pixels |= {(0, y), (wpx - 1, y)}
    for cx, cy in pixels:
        cc = SkyCoord.from_pixel(float(cx), float(cy), wcs=w, origin=0)
        max_sep_deg = max(max_sep_deg, float(center.separation(cc).deg))
    radius_deg = max(max_sep_deg * 1.38, max_sep_deg + 45.0 / 3600.0)
    scales = w.proj_plane_pixel_scales()
    r_cd = float(math.hypot(0.5 * (wpx - 1) * abs(float(scales[0].to(u.deg).value)),
                            0.5 * (h - 1) * abs(float(scales[1].to(u.deg).value))))
    return center, float(max(radius_deg, r_cd * 1.22))


def test_g05_43_field_radius_identical():
    from astropy.wcs import WCS

    from pipeline_catalog import _field_center_and_radius_from_wcs

    rng = np.random.default_rng(543)
    for k in range(25):
        h, wpx = int(rng.integers(500, 4200)), int(rng.integers(500, 4200))
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = [rng.uniform(0, 360), rng.uniform(-89, 89) if k else 89.9]
        w.wcs.crpix = [wpx / 2 + rng.normal(0, 40), h / 2 + rng.normal(0, 40)]
        sc, rot = rng.uniform(2e-5, 2e-3), rng.uniform(0, 2 * np.pi)
        w.wcs.cd = sc * np.array([[-np.cos(rot), np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        c_ref, r_ref = _field_radius_ref(w, h, wpx)
        c_new, r_new = _field_center_and_radius_from_wcs(w, h, wpx)
        assert r_ref == r_new
        assert c_ref.ra.deg == c_new.ra.deg and c_ref.dec.deg == c_new.dec.deg


# --- G05-51 VSX table counted once ---------------------------------------------------------------

def test_g05_51_vsx_rows_counted_once(tmp_path, monkeypatch):
    import sqlite3

    from astropy.coordinates import SkyCoord

    import catalog_db
    import pipeline_catalog

    db = tmp_path / "objects.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE vsx_data (name TEXT, ra_deg REAL, dec_deg REAL, var_type TEXT, "
                "mag_max REAL, mag_min REAL, period REAL, oid INTEGER)")
    con.executemany("INSERT INTO vsx_data VALUES (?,?,?,?,?,?,?,?)",
                    [(f"V{i}", 200.0 + 0.01 * i, 40.0, "EW", 12.0, 12.5, 0.3, i) for i in range(5)])
    con.commit()
    con.close()
    ok, _ = catalog_db.validate_vsx_schema(db)
    if not ok:
        pytest.skip("VSX schema of this test table is not the current one")
    calls = []
    real = catalog_db.count_vsx_rows
    monkeypatch.setattr(catalog_db, "count_vsx_rows", lambda p: (calls.append(1), real(p))[1])
    assert catalog_db.require_vsx_db_with_count(db) == (db.resolve(), 5)
    calls.clear()
    pipeline_catalog._query_vsx_local(center=SkyCoord(200.02, 40.0, unit="deg"), radius_deg=0.5,
                                      vsx_db_path=db, require_db=True)
    assert len(calls) == 1


def test_g10_45_alignment_matrix_one_header_read(tmp_path, monkeypatch):
    import inspect

    from astropy.io import fits

    import transient_field as tf
    from frame_footprint import alignment_matrix  # G02-33 (s5_dup): the one VY_AM reader

    hdr = fits.Header({f"VY_AM{j}": float(v) for j, v in enumerate([1.0, 0.01, 3.5, -0.01, 1.0, -2.25, 0, 0, 1])})
    ref = np.array([float(hdr[f"VY_AM{j}"]) for j in range(9)]).reshape(3, 3)
    assert _bits_equal(ref, alignment_matrix(hdr))
    src = inspect.getsource(tf)
    line = next(ln for ln in src.splitlines() if ln.strip().startswith("mats = "))
    assert line.count("getheader") == 1 and "alignment_matrix(fits.getheader(p))" in line


# --- G05-56 proper motion for all Gaia rows at once (math.cos kept) -----------------------------

def _apply_pm_scalar_ref(ra, dec, pmra, pmdec, obs_year):
    """Copy of gaia_pm.apply_proper_motion at 08e985a."""
    import math

    try:
        ra_f, dec_f = float(ra), float(dec)
    except (TypeError, ValueError):
        return ra, dec
    if not (math.isfinite(ra_f) and math.isfinite(dec_f)):
        return ra, dec
    try:
        pmra_f = float(pmra) if pmra is not None else float("nan")
        pmdec_f = float(pmdec) if pmdec is not None else float("nan")
    except (TypeError, ValueError):
        return ra_f, dec_f
    if not (math.isfinite(pmra_f) and math.isfinite(pmdec_f)):
        return ra_f, dec_f
    dt = float(obs_year) - 2016.0
    if not math.isfinite(dt) or abs(dt) < 1e-9:
        return ra_f, dec_f
    cos_dec = math.cos(math.radians(dec_f))
    if not math.isfinite(cos_dec) or abs(cos_dec) < 1e-6:
        return ra_f, dec_f
    delta_ra = (pmra_f / 3_600_000.0) * dt / cos_dec
    delta_dec = (pmdec_f / 3_600_000.0) * dt
    return (ra_f + delta_ra) % 360.0, max(-90.0, min(90.0, dec_f + delta_dec))


@pytest.mark.parametrize("epoch", [2000.0, 2026.8, 2016.0])
def test_g05_56_pm_vectorised_bit_identical(epoch):
    import vsx_gaia_crossmatch as vgc
    from gaia_pm import apply_proper_motion

    rng = np.random.default_rng(56)
    n = 50000
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    pmra, pmdec = rng.normal(0, 40, n), rng.normal(0, 40, n)
    ra[:200] = rng.uniform(-1e-9, 1e-9, 200)                  # wrap across 0 / 360
    pmra[:100] = -5000.0
    pmra[200:220] = 1e7                                       # dec clipped at the poles
    dec[220:240], dec[240:250] = 90.0, -89.9999999999
    ra[250], dec[251], pmra[252], pmdec[253] = np.nan, np.inf, np.nan, -np.inf
    ref_ra = np.array([_apply_pm_scalar_ref(a, d, p, q, epoch)[0] for a, d, p, q in zip(ra, dec, pmra, pmdec)])
    ref_de = np.array([_apply_pm_scalar_ref(a, d, p, q, epoch)[1] for a, d, p, q in zip(ra, dec, pmra, pmdec)])
    new_ra, new_de, mode, _sig, n_fin = vgc._apply_pm_to_gaia_catalog(ra, dec, pmra, pmdec, epoch)
    assert _bits_equal(ref_ra, new_ra) and _bits_equal(ref_de, new_de)
    assert mode == "propagated" and n_fin == n - 2
    for i in (0, 5, 210, 230, 245, 250, 252, 999):
        assert apply_proper_motion(ra[i], dec[i], pmra[i], pmdec[i], epoch) == \
            _apply_pm_scalar_ref(ra[i], dec[i], pmra[i], pmdec[i], epoch) or i in (250,)
    assert apply_proper_motion("x", 1.0, 1.0, 1.0, epoch) == ("x", 1.0)


# --- G02-43 quantize_dist_deg: Python round kept (np.round is not identical) ---------------------

def _quantize_dist_deg_ref(values):
    scalar = isinstance(values, (float, int, np.floating, np.integer)) or (
        isinstance(values, np.ndarray) and values.ndim == 0)
    arr = np.atleast_1d(np.asarray(values, dtype=np.float64))
    out = np.empty(arr.shape, dtype=np.float64)
    flat = out.ravel()
    for i, x in enumerate(arr.ravel()):
        if not np.isfinite(x):
            flat[i] = np.nan
        elif float(x) >= 900.0:
            flat[i] = 999.0
        else:
            flat[i] = round(float(x), 9)
    return float(out.ravel()[0]) if scalar else out


def test_g02_43_quantize_identical_incl_ties():
    from sky_separation import DIST_DEG_NDIGITS, quantize_dist_deg

    rng = np.random.default_rng(243)
    ties = (rng.integers(0, 3_000_000_000, 200_000) + 0.5) / 1e9           # half-quantum ties
    vals = np.concatenate([rng.uniform(0, 3, 200_000), ties, np.nextafter(ties, 0), np.nextafter(ties, 9),
                           [np.nan, np.inf, -np.inf, 900.0, 899.9999999999, 1e6, -0.0, -1.5e-10, 2.5127264415]])
    assert DIST_DEG_NDIGITS == 9
    assert _bits_equal(_quantize_dist_deg_ref(vals), quantize_dist_deg(vals))
    m = vals[:12].reshape(3, 4)
    assert _bits_equal(_quantize_dist_deg_ref(m), quantize_dist_deg(m))
    for v in (2.5127264415, np.float64(0.1234567895), np.array(1.0000000005), 7, float("nan")):
        r, n_ = _quantize_dist_deg_ref(v), quantize_dist_deg(v)
        assert type(r) is type(n_) and (r == n_ or (r != r and n_ != n_))
    assert quantize_dist_deg(np.zeros(0)).shape == (0,)


# --- G12-38 pointing scan only when the RAM-QC medians are missing ------------------------------

def _ui_pointing_ref(qsum, scan):
    import math

    pointing = scan()
    r_pref = next((r for r in pointing["rows"]
                   if r.get("display_ra_deg") is not None and r.get("display_dec_deg") is not None), None)
    ra_ui = de_ui = None
    if r_pref:
        ra_ui, de_ui = float(r_pref["display_ra_deg"]), float(r_pref["display_dec_deg"])
    try:
        mra_q, mde_q = qsum.get("median_ra_deg"), qsum.get("median_de_deg")
        if mra_q is not None and math.isfinite(float(mra_q)):
            ra_ui = float(mra_q)
        if mde_q is not None and math.isfinite(float(mde_q)):
            de_ui = float(mde_q)
    except (TypeError, ValueError):
        pass
    return ra_ui, de_ui


@pytest.mark.parametrize("qsum", [
    {"median_ra_deg": 201.5, "median_de_deg": 41.2},
    {"median_ra_deg": None, "median_de_deg": 41.2},
    {"median_ra_deg": 201.5, "median_de_deg": float("nan")},
    {"median_ra_deg": "bad", "median_de_deg": 41.2},
    {"median_ra_deg": 201.5, "median_de_deg": "bad"},
    {},
])
@pytest.mark.parametrize("rows", [
    [{"display_ra_deg": None, "display_dec_deg": None}, {"display_ra_deg": 10.0, "display_dec_deg": -5.0}],
    [],
])
def test_g12_38_ui_pointing_identical(qsum, rows, monkeypatch, tmp_path):
    import night_run
    import pipeline_calibrate

    calls = []

    def _scan(root, max_files=None):
        calls.append(root)
        return {"rows": rows}

    monkeypatch.setattr(pipeline_calibrate, "scan_calibrated_lights_pointing", _scan)
    pointing, ra, de = night_run._ui_pointing(qsum, tmp_path)
    assert (ra, de) == _ui_pointing_ref(qsum, lambda: {"rows": rows})
    both = qsum.get("median_ra_deg") == 201.5 and qsum.get("median_de_deg") == 41.2
    assert (len(calls) == 0) == both and (pointing == {}) == both


# --- G12-25 the job child starts as the light run_job module (spawned workers re-import it) -----

def test_g12_25_job_entry_is_light_and_dispatches(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path

    import run_job

    src = Path(run_job.__file__).resolve().parent
    code = ("import sys; import run_job; from astropy.utils import iers; "
            "heavy = [m for m in ('night_run', 'pipeline', 'pipeline_catalog', 'importer') if m in sys.modules]; "
            "print(heavy, iers.conf.auto_download)")
    out = subprocess.run([sys.executable, "-c", code], cwd=str(src), capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[] False"                    # IERS-OFF kept, night_run graph not loaded
    seen = {}
    monkeypatch.setattr(run_job, "run_job_file", lambda p: seen.setdefault("p", p) and 0)
    run_job.main(["--job", str(tmp_path / "job.json")])
    assert Path(seen["p"]) == tmp_path / "job.json"
    started = {}

    class _P:
        pid = 4242

    monkeypatch.setattr(run_job.subprocess, "Popen", lambda cmd, **kw: (started.setdefault("cmd", cmd), _P())[1])
    monkeypatch.setattr(run_job, "params_to_job", lambda p: {})
    run_job.launch_job(object(), tmp_path)
    assert started["cmd"][1:3] == ["-m", "run_job"]


# --- G01-35 gain PTC: headers first, the central region only (same PtcResult) -------------------

def _measure_from_files_ref(flat_files, dark_files, *, saturation_adu=65535.0):
    """Copy of gain_ptc.measure_from_files at 08e985a (full float64 frames)."""
    import math
    from pathlib import Path

    from astropy.io import fits

    import gain_ptc as g

    def _load(path):
        with fits.open(path, memmap=False) as hd:
            return np.asarray(hd[0].data, dtype=np.float64), hd[0].header.copy()

    res = g.PtcResult(ok=False)
    frames = [(p, *_load(p)) for p in sorted({Path(f) for f in flat_files})]
    darks = [(p, *_load(p)) for p in sorted({Path(f) for f in dark_files})]
    if len(frames) < 2:
        res.messages.append("At least two raw flats of the same exposure are needed (better 5 levels x 2).")
        return res
    keys = {(g._hdr_num(h, "XBINNING") or 1, g._hdr_num(h, "GAIN")) for _, _, h in frames + darks}
    if len(keys) > 1:
        res.messages.append(f"Frames differ in binning / gain setting: {sorted(keys, key=str)} - use one setting.")
        return res
    res.binning = int((next(iter(keys))[0]) or 1)
    res.gain_setting = next(iter(keys))[1]

    def _exp(h):
        return float(g._hdr_num(h, "EXPTIME", "EXPOSURE") or 0.0)

    dark_by_exp = {}
    for _, a, h in darks:
        dark_by_exp.setdefault(round(_exp(h), 4), []).append(a)
    rn_adu = None
    if dark_by_exp:
        e_min = min(dark_by_exp)
        if len(dark_by_exp[e_min]) >= 2:
            rn_adu = g.read_noise_adu(dark_by_exp[e_min][0], dark_by_exp[e_min][1])
            res.read_noise_adu = rn_adu
        else:
            res.messages.append("Only one dark at the shortest exposure - read noise needs a dark pair.")
    else:
        res.messages.append("No darks: the dark level is not removed and read noise is not measured.")
    by_exp = {}
    for p, a, h in frames:
        by_exp.setdefault(round(_exp(h), 4), []).append((p, a))
    for e, fl in sorted(by_exp.items()):
        dks = dark_by_exp.get(e)
        if dks:
            dark = np.mean(np.stack(dks), axis=0) if len(dks) > 1 else dks[0]
        elif dark_by_exp:
            near = min(dark_by_exp, key=lambda k: abs(k - e))
            dark = np.mean(np.stack(dark_by_exp[near]), axis=0)
            res.messages.append(f"Flats {e:g} s: no dark of that exposure, dark of {near:g} s used.")
        else:
            dark = 0.0
        for (p1, a1), (p2, a2) in zip(fl[0::2], fl[1::2]):
            lev, var_h = g.pair_variance(a1, a2, dark)
            m1, m2 = float(np.median(g._roi(a1))), float(np.median(g._roi(a2)))
            pt = g.PairPoint((p1.name, p2.name), e, lev, var_h)
            if abs(m1 - m2) > g.PAIR_LEVEL_TOL * max(abs(m1), abs(m2), 1.0):
                pt.used, pt.note = False, "levels of the two flats differ (light changed)"
            elif max(m1, m2) > g.MAX_LEVEL_FRACTION * float(saturation_adu):
                pt.used, pt.note = False, f"above {g.MAX_LEVEL_FRACTION:.0%} of saturation"
            elif lev <= 0:
                pt.used, pt.note = False, "level not above the dark"
            res.points.append(pt)
        if len(fl) % 2:
            res.messages.append(f"Flats {e:g} s: odd number of frames, the last one is not paired.")
    gg, g_err, _, r2 = g.fit_ptc(res.points, rn_adu)
    res.n_levels = sum(p.used for p in res.points)
    if not (math.isfinite(gg) and gg > 0):
        res.messages.append("No usable gain: check that the flats are raw, unsaturated and of steady light.")
        return res
    res.ok, res.gain, res.r2 = True, float(gg), (float(r2) if math.isfinite(r2) else None)
    res.gain_err = float(g_err) if math.isfinite(g_err) else None
    if rn_adu is not None:
        res.read_noise_e = float(gg * rn_adu)
    if res.n_levels < 3:
        res.messages.append(f"Only {res.n_levels} usable level(s): take flats at 3 or more levels for a reliable line.")
    return res


def _ptc_files(tmp_path, *, darks=True, mixed_gain=False, odd=False):
    from astropy.io import fits

    rng = np.random.default_rng(35)
    flats, dk = [], []
    gain = 1.6
    for e, lvl in ((1.0, 3000), (2.0, 9000), (4.0, 20000), (8.0, 61000)):
        for k in range(3 if (odd and e == 2.0) else 2):
            sig = rng.poisson((lvl * (1 + 0.002 * k)) * gain, (101, 139)) / gain + rng.normal(0, 3, (101, 139))
            a = np.clip(np.round(sig + 500), 0, 65535).astype(np.uint16)
            fp = tmp_path / f"flat_{e:g}_{k}.fits"
            fits.PrimaryHDU(a, header=fits.Header({"EXPTIME": e, "XBINNING": 1, "GAIN": 120})).writeto(fp)
            flats.append(fp)
    if darks:
        for e in (1.0, 2.0, 8.0):
            for k in range(2):
                a = np.clip(np.round(rng.normal(500, 3, (101, 139))), 0, 65535).astype(np.uint16)
                fp = tmp_path / f"dark_{e:g}_{k}.fits"
                fits.PrimaryHDU(a, header=fits.Header({"EXPTIME": e, "XBINNING": 1,
                                                       "GAIN": 121 if (mixed_gain and k) else 120})).writeto(fp)
                dk.append(fp)
    return flats, dk


@pytest.mark.parametrize("variant", ["full", "no_darks", "mixed_gain", "odd"])
def test_g01_35_ptc_result_identical(tmp_path, variant):
    from gain_ptc import measure_from_files

    flats, darks = _ptc_files(tmp_path, darks=variant != "no_darks", mixed_gain=variant == "mixed_gain",
                              odd=variant == "odd")
    ref = _measure_from_files_ref(flats, darks, saturation_adu=65535.0)
    new = measure_from_files(flats, darks, saturation_adu=65535.0)
    assert repr(ref) == repr(new)
    if variant == "full":
        assert new.ok and new.n_levels >= 3


# --- G01-38 calibration library: schema once per connection, CAPTURE_DATE back-filled once ------

def test_g01_38_schema_once_and_capture_date_backfill(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    from astropy.io import fits

    import calib_library as cl
    from database import VyvarDatabase

    db = VyvarDatabase(tmp_path / "t.sqlite3")
    cam = db.insert_equipment("C5A-150M", "cam", "CMOS", "x", 3.76)
    tel = db.insert_telescope("AZ800", "az", 800.0, 5480.0)
    paths = []
    for k, (kind, when) in enumerate([("dark", "2026-10-01T20:00:00"), ("flat", "2026-10-02T05:10:11"),
                                      ("dark", None)]):
        h = fits.Header({"IMAGETYP": "Master Dark" if kind == "dark" else "Master Flat", "EXPTIME": 10.0,
                         "XBINNING": 1, "YBINNING": 1, "CCD-TEMP": -10.0, "GAIN": 100, "OFFSET": 20,
                         "FILTER": "V"})
        if when:
            h["DATE-OBS"] = when
        p = tmp_path / f"m{k}.fits"
        fits.writeto(p, np.ones((8, 8), np.float32), header=h)
        cl.register_master(db, p, kind=kind, camera_id=cam, telescope_id=tel)
        paths.append(p)
    # legacy rows: no CAPTURE_DATE (and one master file gone)
    db.conn.execute("UPDATE CALIBRATION_LIBRARY SET CAPTURE_DATE = NULL;")
    db.conn.commit()
    gone = tmp_path / "gone.fits"
    db.conn.execute("INSERT INTO CALIBRATION_LIBRARY (KIND, FILE_PATH, XBINNING, EXPTIME, ID_EQUIPMENTS, GAIN_EXACT, "
                    "REGISTERED_AT) VALUES ('dark', ?, 1, 5.0, ?, 1, '2026-01-01T00:00:00');", (str(gone), cam))
    db.conn.commit()

    def _expected(db_):
        out = []
        for r in db_.conn.execute("SELECT * FROM CALIBRATION_LIBRARY").fetchall():
            m = cl._row_to_master(r)
            out.append((m.path, m.capture_utc))
        return sorted(out, key=str)

    db_ref = VyvarDatabase(tmp_path / "t.sqlite3")      # old listing behaviour: FITS read per listing
    ref = _expected(db_ref)
    assert dict(ref)[str(paths[0])] == datetime(2026, 10, 1, 20, 0, tzinfo=timezone.utc)

    db2 = VyvarDatabase(tmp_path / "t.sqlite3")
    n_ddl = []
    real = db2._ensure_calibration_library_table
    monkeypatch.setattr(db2, "_ensure_calibration_library_table", lambda: (n_ddl.append(1), real())[1])
    first = sorted(((m.path, m.capture_utc) for m in cl.list_masters(db2)), key=str)
    opened = []
    real_open = cl.fits.open
    monkeypatch.setattr(cl.fits, "open", lambda *a, **k: (opened.append(a[0]), real_open(*a, **k))[1])
    for _ in range(5):
        again = sorted(((m.path, m.capture_utc) for m in cl.list_masters(db2)), key=str)
        assert again == first
        cl.get_master_by_path(db2, paths[1])
    assert first == ref
    assert len(n_ddl) == 0  # schema / migrations once per connection, when it opens (S6-FIX CLIB-READONLY)
    # only the master without a capture time in its header is still opened per listing
    assert {str(p) for p in opened} <= {str(paths[2])}


# --- G01-44 CAL-DIAG gate: the SUM and MEAN darks are read once each ----------------------------

@pytest.mark.parametrize("case", [("bin2_sum", 120.0, 2), ("bin2_mean", 25.0, 2), ("bin1", 120.0, 1)])
def test_g01_44_gate_reads_each_dark_once(tmp_path, monkeypatch, case):
    import dataclasses

    from astropy.io import fits

    import cal_diag
    from pipeline import _match_and_crop_pair

    _name, level, lb = case
    rng = np.random.default_rng(44)
    n_l = 64 // lb
    light = (level + rng.normal(0, 2.0, (n_l, n_l))).astype(np.float32)
    dark = (10.0 + rng.normal(0, 0.5, (64, 64))).astype(np.float32)
    lp, dp = tmp_path / "light.fits", tmp_path / "dark.fits"
    fits.writeto(lp, light, header=fits.Header({"XBINNING": lb, "YBINNING": lb, "EXPTIME": 30.0}))
    fits.writeto(dp, dark, header=fits.Header({"XBINNING": 1, "YBINNING": 1, "EXPTIME": 30.0}))
    kw = dict(repr_light_path=lp, dark_path=dp, obs_group_key=f"V|30|{lb}", light_binning=lb,
              master_binning=1, pedestal_dark_paths=[dp], match_and_crop_pair=_match_and_crop_pair,
              saturation_adu=None)
    real = cal_diag._load_resampled_dark
    modes = []

    def _spy(**k):
        modes.append(k["dark_resample_mode"])
        a, bf = real(**k)
        return a.copy(), bf                                    # independent arrays, as separate reads

    monkeypatch.setattr(cal_diag, "_load_resampled_dark", _spy)
    res = cal_diag.cal_diag_gate_for_obs_group(**kw)
    assert sorted(modes) == ["mean", "sum"]                   # was 3 reads (a mode read twice)
    monkeypatch.setattr(cal_diag, "_load_resampled_dark", real)
    res2 = cal_diag.cal_diag_gate_for_obs_group(**kw)
    assert repr(dataclasses.asdict(res)) == repr(dataclasses.asdict(res2))


# --- G03-11 VSX match lookups indexed once ------------------------------------------------------

def test_g03_11_vsx_lookups_identical():
    import pandas as pd

    from pipeline_astrometry import _vsx_match_lookups

    rng = np.random.default_rng(311)
    n = 3000
    cids = [str(4000000 + int(i)) for i in rng.integers(0, 2500, n)]          # duplicates
    gmag = rng.uniform(8, 19, n).astype(object)
    gmag[::97] = np.nan
    gdf = pd.DataFrame({"catalog_id": cids, "ra_deg": rng.uniform(0, 1, n), "g_mag": list(gmag),
                        "note": ["x"] * n})
    gdf["g_mag"] = gdf["g_mag"].astype(object)
    gdf.loc[5, "g_mag"] = ""
    ga = pd.DataFrame({"catalog_id": cids[:1500], "mag": rng.uniform(8, 17, 1500),
                       "zone": rng.choice(["linear", " Saturated ", None], 1500)})
    ref_gaia = {str(r["catalog_id"]): r for _, r in gdf.iterrows()}
    gmag_by, first = _vsx_match_lookups(gdf, ga)

    def _mag(gm):
        try:
            return float(gm) if gm is not None and str(gm).strip() != "" else float("nan")
        except (TypeError, ValueError):
            return float("nan")

    for cid in set(cids) | {"missing"}:
        assert (cid in gmag_by) == (ref_gaia.get(cid) is not None)
        if cid in gmag_by:
            a, b = _mag(ref_gaia[cid].get("g_mag")), _mag(gmag_by[cid])
            assert a == b or (a != a and b != b)
        gro = ga.loc[ga["catalog_id"].astype(str) == cid]
        assert gro.empty == (first.get(cid) is None)
        if not gro.empty:
            r0, r1 = gro.iloc[0], ga.iloc[first[cid]]
            assert repr(r0.get("mag")) == repr(r1.get("mag")) and repr(r0.get("zone")) == repr(r1.get("zone"))
    no_g = gdf.drop(columns=["g_mag"])
    gm2, _ = _vsx_match_lookups(no_g, ga.drop(columns=["catalog_id"]))
    assert all(v is None for v in gm2.values())


# --- G04-21 identity candidate pairs vectorised (stable sort = list.sort order) ------------------

def _pairs_ref(sep_k, idx, oix_rows, nc, thr_f):
    n_oix = int(len(oix_rows))
    pairs = []
    for i in range(sep_k.shape[0]):
        for kk in range(int(idx.shape[1])):
            j_comp = int(idx[i, kk])
            if j_comp < 0 or j_comp >= n_oix:
                continue
            s = float(sep_k[i, kk])
            if not np.isfinite(s) or s > thr_f:
                continue
            cr = int(oix_rows[j_comp])
            if 0 <= cr < nc:
                pairs.append((s, i, cr))
    pairs.sort(key=lambda t: t[0])
    return pairs


@pytest.mark.parametrize("k", [1, 48])
def test_g04_21_candidate_pairs_identical(k):
    from catalog_match import _identity_candidate_pairs

    rng = np.random.default_rng(421 + k)
    n, n_oix, nc = 6000, 9000, 8800
    sep = np.round(rng.uniform(0, 12, (n, k)), 1)              # many exact ties
    sep[rng.random((n, k)) < 0.02] = np.inf
    sep[rng.random((n, k)) < 0.01] = np.nan
    idx = rng.integers(-2, n_oix + 3, (n, k))
    oix = rng.integers(-5, nc + 50, n_oix)
    for thr in (2.0, 5.55, 0.0):
        assert _identity_candidate_pairs(sep, idx, oix, nc, thr) == _pairs_ref(sep, idx, oix, nc, thr)


# --- G04-20 MASTERSTAR Gaia accounting: lock loop, ambiguity check, state labels -----------------

# Copy of masterstar_gaia_accounting.lock_existing_and_leftover_assign at 08e985a (the reference).
import math  # noqa: E402

import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from masterstar_gaia_accounting import _norm_cid_strip_dotzero  # noqa: E402


def _lock_ref(
    det_x: np.ndarray,
    det_y: np.ndarray,
    gaia_df: pd.DataFrame,
    *,
    locked_pairs: dict[str, tuple[float, float]] | None = None,
    leftover_radius_px: float = 3.0,
    lock_tol_px: float = 3.0,
    identity_fail_px: float | None = None,
    det_catalog_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Return (det_to_gaia_row, gaia_owner_det, vy_match_mode per det, geometry_reject_det_indices).

    det_to_gaia_row[i] = row index in gaia_df or -1
    gaia_owner_det[j] = det index or -1
    vy_match_mode[i] = 'locked' | 'leftover_promotion' | ''
    geometry_reject_det_indices: born-owned detections farther than
    identity_fail_px from Gaia xy (D4: reject threshold is the identity-gate
    fail radius 3 x FWHM_dao, not the derived lock preference). When
    identity_fail_px is omitted, reject uses lock_tol_px (STAGE-01 callers).
    """
    n_d = int(len(det_x))
    n_g = int(len(gaia_df))
    det_to_g = np.full(n_d, -1, dtype=np.int64)
    gaia_owner = np.full(n_g, -1, dtype=np.int64)
    match_mode = np.array([""] * n_d, dtype=object)

    gx = pd.to_numeric(gaia_df.get("x_gaia", gaia_df.get("x")), errors="coerce").to_numpy(
        dtype=np.float64
    )
    gy = pd.to_numeric(gaia_df.get("y_gaia", gaia_df.get("y")), errors="coerce").to_numpy(
        dtype=np.float64
    )
    cids = gaia_df["catalog_id"].map(_norm_cid_strip_dotzero).to_numpy(dtype=object) if "catalog_id" in gaia_df.columns else np.array([""] * n_g, dtype=object)
    cid_to_row = {_norm_cid_strip_dotzero(cids[i]): i for i in range(n_g) if _norm_cid_strip_dotzero(cids[i])}

    used_det: set[int] = set()
    used_g: set[int] = set()
    geometry_reject_dets: list[int] = []
    fail_px = float(identity_fail_px) if identity_fail_px is not None else float(lock_tol_px)
    if not math.isfinite(fail_px) or fail_px <= 0:
        fail_px = float(lock_tol_px)

    if locked_pairs:
        det_cids = (
            np.asarray([_norm_cid_strip_dotzero(c) for c in det_catalog_ids], dtype=object)
            if det_catalog_ids is not None
            else np.array([""] * n_d, dtype=object)
        )
        for cid, (_lx, _ly) in locked_pairs.items():
            gr = cid_to_row.get(_norm_cid_strip_dotzero(cid))
            if gr is None or gr in used_g:
                continue
            nc = _norm_cid_strip_dotzero(cid)
            gx_g = float(gx[gr]) if gr < len(gx) else float("nan")
            gy_g = float(gy[gr]) if gr < len(gy) else float("nan")
            best_i = -1
            best_d = float("inf")
            # Preference: born-owned within lock_tol_px. Keep (still lock) when
            # lock_tol < d <= identity_fail. Reject only past identity_fail_px (D4).
            for i in range(n_d):
                if i in used_det:
                    continue
                dc = _norm_cid_strip_dotzero(det_cids[i]) if i < len(det_cids) else ""
                if dc and dc == nc:
                    d_gaia = float(math.hypot(det_x[i] - gx_g, det_y[i] - gy_g))
                    if not math.isfinite(d_gaia):
                        continue
                    if d_gaia <= float(lock_tol_px):
                        best_i = i
                        best_d = d_gaia
                        break
                    if d_gaia > float(fail_px):
                        geometry_reject_dets.append(i)
                    elif d_gaia < best_d:
                        best_i = i
                        best_d = d_gaia
            if best_i < 0:
                for i in range(n_d):
                    if i in used_det:
                        continue
                    d = float(math.hypot(det_x[i] - gx_g, det_y[i] - gy_g))
                    if d <= float(lock_tol_px) and d < best_d:
                        best_d = d
                        best_i = i
            if best_i >= 0:
                used_det.add(best_i)
                used_g.add(gr)
                det_to_g[best_i] = gr
                gaia_owner[gr] = best_i
                match_mode[best_i] = "locked"

    pairs: list[tuple[float, int, int]] = []
    if n_d and n_g:
        free_d = [i for i in range(n_d) if i not in used_det]
        free_g = [j for j in range(n_g) if j not in used_g]
        if free_d and free_g:
            tree = cKDTree(np.column_stack([gx[free_g], gy[free_g]]))
            dist, idx = tree.query(
                np.column_stack([det_x[free_d], det_y[free_d]]),
                k=min(32, len(free_g)),
                distance_upper_bound=float(leftover_radius_px),
            )
            dist = np.atleast_2d(np.asarray(dist, dtype=np.float64))
            idx = np.atleast_2d(np.asarray(idx, dtype=np.int64))
            for k, i in enumerate(free_d):
                for kk in range(idx.shape[1]):
                    jloc = int(idx[k, kk])
                    d = float(dist[k, kk])
                    if jloc < 0 or jloc >= len(free_g) or not math.isfinite(d):
                        continue
                    j = free_g[jloc]
                    if d <= float(leftover_radius_px):
                        pairs.append((d, i, j))
    pairs.sort(key=lambda t: t[0])
    for _d, i, j in pairs:
        if i in used_det or j in used_g:
            continue
        used_det.add(i)
        used_g.add(j)
        det_to_g[i] = j
        gaia_owner[j] = i
        if not match_mode[i]:
            match_mode[i] = "leftover_promotion"

    return det_to_g, gaia_owner, match_mode, geometry_reject_dets


def _lock_case(seed, n_g=2500, n_d=2400, n_lock=1500):
    rng = np.random.default_rng(seed)
    gx, gy = rng.uniform(0, 900, n_g), rng.uniform(0, 700, n_g)
    # (Gaia positions on the chip are finite: the leftover KD-tree requires it)
    cids = [f"{100000 + k}" for k in range(n_g)]
    cids[7] = "100003.0"                                       # duplicate after the .0 strip
    gdf = pd.DataFrame({"x_gaia": gx, "y_gaia": gy, "catalog_id": cids})
    src = rng.integers(0, n_g, n_d)
    det_x = gx[src] + rng.normal(0, 1.8, n_d)
    det_y = gy[src] + rng.normal(0, 1.8, n_d)
    det_x[rng.random(n_d) < 0.05] += rng.uniform(4, 15)       # some beyond the lock / fail radius
    # (detections are finite: the leftover KD-tree query requires it)
    det_x[5] = gx[0] + 3.0                                     # exactly at lock_tol from Gaia row 0
    det_y[5] = gy[0]
    det_x[6], det_y[6] = gx[0] - 3.0, gy[0]                   # tie with det 5
    det_cids = np.array([cids[s] if rng.random() < 0.8 else "" for s in src], dtype=object)
    det_cids[9] = cids[src[9]] + ".0"
    lock_ids = rng.choice(n_g, n_lock, replace=False).tolist() + [0, 0]
    locked = {cids[j] if k % 5 else f" {cids[j]} ": (0.0, 0.0) for k, j in enumerate(lock_ids)}
    return det_x, det_y, gdf, locked, det_cids


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("fail", [None, 7.0])
def test_g04_20_lock_assign_identical(seed, fail):
    from masterstar_gaia_accounting import lock_existing_and_leftover_assign

    det_x, det_y, gdf, locked, det_cids = _lock_case(seed)
    kw = dict(locked_pairs=locked, leftover_radius_px=3.0, lock_tol_px=3.0, identity_fail_px=fail,
              det_catalog_ids=det_cids)
    ref = _lock_ref(det_x, det_y, gdf, **kw)
    new = lock_existing_and_leftover_assign(det_x, det_y, gdf, **kw)
    assert np.array_equal(ref[0], new[0]) and np.array_equal(ref[1], new[1])
    assert ref[2].tolist() == new[2].tolist() and ref[3] == new[3]
    assert (ref[2] == "locked").sum() > 500 and len(ref[3]) > 0


def test_g04_20_ambiguous_owner_precomputed_xy():
    from masterstar_gaia_accounting import check_ambiguous_owner

    rng = np.random.default_rng(420)
    gdf = pd.DataFrame({"x_gaia": rng.uniform(0, 100, 400).astype(object), "y_gaia": rng.uniform(0, 100, 400)})
    gdf.loc[3, "x_gaia"] = "bad"
    gx = pd.to_numeric(gdf["x_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    gy = pd.to_numeric(gdf["y_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    for j in range(0, 400, 7):
        xd, yd = float(gy[j]) + 0.7, float(gy[j]) - 0.4
        assert check_ambiguous_owner(xd, yd, j, gdf) == check_ambiguous_owner(xd, yd, j, gdf, gxy=(gx, gy))


# Copy of masterstar_gaia_accounting._acc_census at 08e985a (the reference of the state labels).
from masterstar_gaia_accounting import (  # noqa: E402
    SOURCE_BLENDED,
    SOURCE_CATALOG_MEMBERSHIP,
    SOURCE_CATALOG_ONLY,
    SOURCE_DETECTED_P1,
    SOURCE_DETECTED_P2,
    _NONDET_CENSUS_STATES,
    annotate_blended_groups,
    build_gaia_census_rows,
)


def _acc_census_ref(
    *,
    det_pass,
    edge_m,
    fwhm_px,
    gaia_owner,
    gdf,
    h,
    meta,
    n_g,
    out,
    target_depth_g,
    wpx,
):
    """Blended groups and the MASTERSTAR Gaia census rows (detected / blended / catalogue only / below limit)."""
    gdf = annotate_blended_groups(gdf, gaia_owner, fwhm_px=fwhm_px)
    census = build_gaia_census_rows(
        gdf,
        gaia_owner,
        det_pass,
        target_depth_g=float(target_depth_g),
        edge_margin_px=edge_m,
        wpx=int(wpx),
        h=int(h),
    )
    _bg_col = gdf["blend_group_id"] if "blend_group_id" in gdf.columns else pd.Series([""] * n_g, index=gdf.index)
    _bg_arr = _bg_col.fillna("").astype(str).str.strip().to_numpy(dtype=object)
    if len(census) and "catalog_id" in census.columns and bool((_bg_arr != "").any()):
        _census_rows: dict[str, list[Any]] = {}
        for _ci, _cc in zip(census.index, census["catalog_id"]):
            _census_rows.setdefault(_cc, []).append(_ci)
        _bfr = gdf["blend_flux_ratio"].to_numpy() if "blend_flux_ratio" in gdf.columns else None
        _gcid = gdf["catalog_id"] if "catalog_id" in gdf.columns else pd.Series([""] * n_g, index=gdf.index)
        for j in np.nonzero(_bg_arr != "")[0]:
            cid = _norm_cid_strip_dotzero(_gcid.iloc[j])
            if cid and int(gaia_owner[j]) < 0:
                _rows = _census_rows.get(cid)
                if _rows:
                    census.loc[_rows, "source_state"] = SOURCE_BLENDED
                    census.loc[_rows, "blend_group_id"] = _bg_arr[j]
                    census.loc[_rows, "blend_flux_ratio"] = _bfr[j] if _bfr is not None else None

    if "source_state" not in out.columns:
        out["source_state"] = ""
    cid_out = out.get("catalog_id", pd.Series([""] * len(out))).map(_norm_cid_strip_dotzero)
    census_st: dict[str, str] = {}
    if len(census):
        for ci in range(len(census)):
            cc = _norm_cid_strip_dotzero(census.iloc[ci].get("catalog_id"))
            if cc:
                census_st[cc] = str(census.iloc[ci].get("source_state", "")).strip()
    for i in range(len(out)):
        cid = _norm_cid_strip_dotzero(cid_out.iloc[i])
        if not cid:
            out.loc[out.index[i], "source_state"] = "DAO_ONLY"
            continue
        st = out.loc[out.index[i], "source_state"] if "source_state" in out.columns else ""
        if str(st).strip() in (SOURCE_DETECTED_P1, SOURCE_DETECTED_P2):
            # Pre-set DETECTED_* is only kept when this row itself is a DAO detection.
            pass
        elif str(st).strip():
            continue
        vp = int(det_pass[i])
        if vp in (1, 2):
            out.loc[out.index[i], "source_state"] = (
                SOURCE_DETECTED_P2 if vp == 2 else SOURCE_DETECTED_P1
            )
            continue
        mode = str(out.loc[out.index[i], "vy_match_mode"] or "").strip() if "vy_match_mode" in out.columns else ""
        if mode == "catalog_membership":
            out.loc[out.index[i], "source_state"] = SOURCE_CATALOG_MEMBERSHIP
            continue
        cst = str(census_st.get(cid, "") or "").strip()
        if cst in _NONDET_CENSUS_STATES:
            out.loc[out.index[i], "source_state"] = cst
        else:
            out.loc[out.index[i], "source_state"] = SOURCE_CATALOG_ONLY

    meta["n_on_chip_gaia"] = n_g
    meta["census_by_state"] = (
        census["source_state"].value_counts().to_dict() if len(census) else {}
    )
    if "ambiguous_owner" in out.columns and len(census):
        amb_map: dict[str, bool] = {}
        for i in range(len(out)):
            cid = _norm_cid_strip_dotzero(out.iloc[i].get("catalog_id"))
            if not cid:
                continue
            raw_amb = out.iloc[i].get("ambiguous_owner")
            amb_map[cid] = (
                bool(raw_amb)
                if raw_amb is not None and not (isinstance(raw_amb, float) and not math.isfinite(raw_amb))
                else False
            )
        for ci in range(len(census)):
            cid = _norm_cid_strip_dotzero(census.iloc[ci].get("catalog_id"))
            if amb_map.get(cid):
                census.loc[census.index[ci], "ambiguous_owner"] = True
    return census


def _census_case(seed, n_out=1800, n_g=1500):
    rng = np.random.default_rng(seed)
    cids = [f"{200000 + k}" for k in range(n_g)]
    gdf = pd.DataFrame({"catalog_id": cids, "x_gaia": rng.uniform(40, 960, n_g), "y_gaia": rng.uniform(40, 760, n_g),
                        "g_mag": rng.uniform(9, 18, n_g), "flux": rng.uniform(1e3, 1e5, n_g)})
    gdf.loc[::9, "x_gaia"] = gdf.loc[::9, "x_gaia"].to_numpy()[:1].repeat(len(gdf.loc[::9])) + 0.5  # blends
    det_pass = rng.choice([0, 1, 2], n_out)
    dets = np.flatnonzero(det_pass > 0)
    owner = np.full(n_g, -1, dtype=np.int64)
    own = rng.random(n_g) < 0.6
    owner[own] = rng.choice(dets, int(own.sum()), replace=False)
    out_cid = [cids[k] if rng.random() < 0.85 else ("" if rng.random() < 0.5 else "nan")
               for k in rng.integers(0, n_g, n_out)]
    pre = rng.choice(["", "", "DETECTED_P1", "DETECTED_P2", "BLENDED", None], n_out)
    out = pd.DataFrame({"catalog_id": out_cid, "source_state": pre,
                        "vy_match_mode": rng.choice(["", "locked", "catalog_membership", None], n_out),
                        "ambiguous_owner": rng.choice([True, False, np.nan], n_out)})
    return gdf, owner, det_pass, out


@pytest.mark.parametrize("seed", [3, 4])
def test_g04_20_census_labels_identical(seed):
    import masterstar_gaia_accounting as mga

    gdf, owner, det_pass, out = _census_case(seed)
    kw = dict(det_pass=det_pass, edge_m=40.0, fwhm_px=3.5, gaia_owner=owner, gdf=gdf, h=800,
              n_g=len(gdf), target_depth_g=17.0, wpx=1000)
    out_ref, out_new = out.copy(), out.copy()
    meta_ref, meta_new = {}, {}
    c_ref = _acc_census_ref(meta=meta_ref, out=out_ref, **kw)
    c_new = mga._acc_census(meta=meta_new, out=out_new, **kw)
    assert out_ref.to_csv() == out_new.to_csv()
    assert c_ref.to_csv() == c_new.to_csv() and meta_ref == meta_new
    assert out_ref["source_state"].nunique() >= 4
    with pytest.raises(ValueError, match="not unique"):
        mga._acc_census(meta={}, out=pd.concat([out, out]), **dict(kw, det_pass=np.r_[det_pass, det_pass]))


# --- G04-22 (I/O part) field_catalog_cone.csv parsed once per MASTERSTAR build -------------------

def test_g04_22_cone_csv_parsed_once(tmp_path, monkeypatch):
    import os

    import masterstar_build as mb
    from gaia_catalog_id import read_vyvar_csv

    cone = tmp_path / "field_catalog_cone.csv"
    pd.DataFrame({"catalog_id": ["4001", "4002"], "ra_deg": [1.123456789012, 2.0],
                  "dec_deg": [3.0, 4.5], "mag": [12.25, 17.0]}).to_csv(cone, index=False)
    calls = []
    real = mb.read_vyvar_csv
    monkeypatch.setattr(mb, "read_vyvar_csv", lambda p, **k: (calls.append(p), real(p, **k))[1])
    mb._CONE_CSV_CACHE.clear()
    a = mb._read_field_cone_csv(cone)
    a.loc[0, "mag"] = -1.0                                     # a caller's change stays its own
    b = mb._read_field_cone_csv(tmp_path / "." / "field_catalog_cone.csv")
    c = mb._read_field_cone_csv(cone)
    assert len(calls) == 1
    ref = read_vyvar_csv(cone, low_memory=False)
    assert b.to_csv() == ref.to_csv() and c.to_csv() == ref.to_csv()
    pd.DataFrame({"catalog_id": ["9"], "ra_deg": [5.0], "dec_deg": [6.0], "mag": [10.0]}).to_csv(cone, index=False)
    os.utime(cone, ns=(1, 1))                                 # a rewritten cone is read again
    assert mb._read_field_cone_csv(cone)["catalog_id"].astype(str).tolist() == ["9"] and len(calls) == 2


# --- bad-pixel repair (step 4): identical-output memory cleanups ---------------------------------

# Copies of badpix_repair._star_models / repair_missing_pixels at 08e985a (the reference).
from typing import Any  # noqa: E402

from badpix_repair import (  # noqa: E402
    _FIT_FAILED_FLAGS,
    BP_EPSF_BUILD_STARS_MAX,
    RepairResult,
    _build_epsf,
)


def _bp_star_models_ref(d, missing, xy, peak, stellar, near, sky, bg, mx, my, *, fw, half, fsz, cutout, sat_cal_adu):
    """ePSF model image of the stars near missing pixels; (model, star_ok, n_build, n_fitted, reason).

    ``star_ok`` is False (scalar) when no model can be made, or per missing pixel False where a star
    whose fit failed reaches it."""
    from astropy.table import Table  # noqa: PLC0415
    from photutils.psf import PSFPhotometry, SourceGrouper  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415

    from pipeline_constants import EPSF_MIN_BUILD_STARS, LINEARITY_FRAC  # noqa: PLC0415
    from sky_estimation import annulus_sky_medians  # noqa: PLC0415

    none = np.zeros_like(d)
    if sat_cal_adu is None:
        return none, False, 0, 0, "no linearity limit for the frame (VY_CSAT)"
    lin = LINEARITY_FRAC * float(sat_cal_adu)
    h, w = d.shape
    st = cKDTree(xy)
    n_close = np.array([len(st.query_ball_point(p, half)) - 1 for p in xy])
    inside = (xy[:, 0] >= half) & (xy[:, 0] < w - half) & (xy[:, 1] >= half) & (xy[:, 1] < h - half)
    d_miss, _ = cKDTree(np.c_[mx, my].astype(float)).query(xy)
    ok_b = stellar & inside & (n_close == 0) & (peak < lin) & (d_miss > half)
    cand = xy[ok_b][np.argsort(-peak[ok_b])[:BP_EPSF_BUILD_STARS_MAX]]
    if len(cand) < int(EPSF_MIN_BUILD_STARS):
        return none, False, int(len(cand)), 0, f"{len(cand)} ePSF build stars < {int(EPSF_MIN_BUILD_STARS)}"
    try:
        psf, n_build = _build_epsf(np.where(missing, np.nan, d), cand, fw, cutout)
        phot = PSFPhotometry(psf, (fsz, fsz), aperture_radius=1.5 * fw, grouper=SourceGrouper(min_separation=2.0 * fsz),
                             xy_bounds=(fw, fw), fitter_maxiters=1000)
        # xy_bounds: a fitted centre stays within one FWHM of its DAO centroid; 1000 iterations: the photutils
        # default (100) left joint fits of a bright star and a faint neighbour unconverged (Brno 534 g_0080)
        fxy = xy[near]
        lbkg = annulus_sky_medians(sky, fxy, 4.0 * fw, 6.0 * fw)
        init = Table({"id": np.arange(1, len(fxy) + 1), "x": fxy[:, 0], "y": fxy[:, 1],
                      "local_bkg": np.nan_to_num(lbkg, nan=bg)})
        r = phot(np.where(missing, bg, d), mask=missing | (d >= lin), init_params=init)
        model = np.asarray(phot.make_model_image(d.shape, include_local_bkg=False), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001  - the star pixels stay missing, with the reason
        return none, False, 0, 0, f"ePSF build / fit error: {type(exc).__name__}: {exc}"
    flags = np.asarray(r["flags"], int)
    flux = np.asarray(r["flux_fit"], float)
    # a source fitted with no positive flux is not a star: no light of its own, its model is removed
    for j in np.flatnonzero(~(flux > 0) & np.isfinite(flux)):
        x0, y0 = float(r["x_fit"][j]), float(r["y_fit"][j])
        if not (math.isfinite(x0) and math.isfinite(y0)):
            continue
        y_a, y_b = max(0, int(y0 - half)), min(d.shape[0], int(y0 + half) + 2)
        x_a, x_b = max(0, int(x0 - half)), min(d.shape[1], int(x0 + half) + 2)
        if y_b <= y_a or x_b <= x_a:
            continue  # its model lies off the frame
        yy, xx = np.mgrid[y_a:y_b, x_a:x_b]
        model[y_a:y_b, x_a:x_b] -= psf.evaluate(xx, yy, flux=float(flux[j]), x_0=x0, y_0=y0)
    # failed: the fit did not converge (8) or left the image (2)
    bad = ((flags & _FIT_FAILED_FLAGS) != 0) & (flux > 0)
    if not bad.any():
        return model, True, n_build, int(len(r)), ""
    # a missing pixel within the model reach of a star whose fit failed stays missing
    bxy = np.c_[np.asarray(r["x_fit"], float), np.asarray(r["y_fit"], float)][bad]
    bxy = np.where(np.isfinite(bxy), bxy, fxy[bad])
    dpx, _ = cKDTree(bxy).query(np.c_[mx, my].astype(float))
    return model, dpx > half, n_build, int(len(r)), f"PSF fit failed for {int(bad.sum())} star(s)"


def _bp_repair_ref(
    data: np.ndarray,
    *,
    fwhm_px: float,
    detection_sigma: float,
    sat_cal_adu: float | None,
) -> RepairResult:
    """Fill the missing (non-finite) pixels of a calibrated frame from the frame's PSF model (module doc)."""
    from photutils.detection import DAOStarFinder  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415

    from epsf_frame import fit_box_px  # noqa: PLC0415
    from psf_photometry import _EPSF_CUTOUT_FWHM, _to_odd_cutout  # noqa: PLC0415
    from sky_estimation import annulus_sky_medians, sky_image  # noqa: PLC0415

    d = np.asarray(data, dtype=np.float64)
    missing = ~np.isfinite(d)
    res = RepairResult(data=np.asarray(data))
    res.n_missing = int(missing.sum())
    if res.n_missing == 0:
        return res
    fw = float(fwhm_px)
    if not math.isfinite(fw) or fw <= 0:
        res.n_left, res.reason_left = res.n_missing, "no FWHM"
        return res
    my, mx = np.nonzero(missing)
    out = d.copy()
    sky = sky_image(np.where(missing, np.nan, d), fw)
    sky_px = annulus_sky_medians(sky, np.c_[mx, my].astype(float), 4.0 * fw, 6.0 * fw)

    cutout = _to_odd_cutout(int(round(fw * _EPSF_CUTOUT_FWHM)))
    fsz = fit_box_px(fw)
    half = cutout / 2.0
    reach = half + fsz / 2.0 + 1.0
    finite = d[~missing]
    bg = float(np.median(finite))
    sig = 1.4826 * float(np.median(np.abs(finite - bg)))
    # Detection sees each missing pixel as the median of its finite 3x3 neighbours: masked, a missing
    # pixel at a star's peak hides the star from DAOStarFinder and the pixel would get the sky. The
    # filled peak is a dip, so a source within 2 px of a missing pixel is kept whatever its DAOFIND
    # sharpness / roundness; elsewhere those limits (Stetson 1987; photutils defaults) reject noise peaks,
    # whose free-position fits do not converge and would fail the fit of a real star in their group.
    det = np.where(missing, bg, d)
    for y, x in zip(my, mx, strict=True):
        nb = d[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        nb = nb[np.isfinite(nb)]
        if nb.size:
            det[y, x] = float(np.median(nb))
    src = DAOStarFinder(fwhm=fw, threshold=float(detection_sigma) * sig, sharpness_range=None,
                        roundness_range=None)(det - bg)
    if src is None or len(src) == 0:
        xy = np.zeros((0, 2))
        peak = np.zeros(0)
        stellar = np.zeros(0, dtype=bool)
    else:
        xcol = "x_centroid" if "x_centroid" in src.colnames else "xcentroid"
        ycol = "y_centroid" if "y_centroid" in src.colnames else "ycentroid"
        xy = np.c_[np.asarray(src[xcol], float), np.asarray(src[ycol], float)]
        peak = np.asarray(src["peak"], float) + bg
        stellar = ((np.asarray(src["sharpness"], float) >= 0.2) & (np.asarray(src["sharpness"], float) <= 1.0)
                   & (np.abs(np.asarray(src["roundness1"], float)) <= 1.0)
                   & (np.abs(np.asarray(src["roundness2"], float)) <= 1.0))
    miss_tree = cKDTree(np.c_[mx, my].astype(float))
    if len(xy):
        d_m, _ = miss_tree.query(xy)
        keep = stellar | (d_m <= 2.0)
        xy, peak, stellar = xy[keep], peak[keep], stellar[keep]
    near = np.zeros(len(xy), dtype=bool)
    if len(xy):
        dist, _ = miss_tree.query(xy)
        near = dist <= reach  # fitted (their model, or their neighbours', can reach a missing pixel)
    model = np.zeros_like(d)
    star_ok: Any = True
    if near.any():
        model, star_ok, res.n_build_stars, res.n_stars_fitted, res.reason_left = _bp_star_models_ref(
            d, missing, xy, peak, stellar, near, sky, bg, mx, my, fw=fw, half=half, fsz=fsz, cutout=cutout,
            sat_cal_adu=sat_cal_adu,
        )
    # which missing pixels a fitted star's model reaches
    if len(xy) and near.any():
        dstar, _ = cKDTree(xy[near]).query(np.c_[mx, my].astype(float))
        star_px = dstar <= half
    else:
        star_px = np.zeros(len(mx), dtype=bool)
    ok = np.isfinite(sky_px) & (~star_px | star_ok)
    for i in np.flatnonzero(ok):
        out[my[i], mx[i]] = sky_px[i] + (model[my[i], mx[i]] if star_px[i] else 0.0)
        res.xy_kind.append((int(mx[i]), int(my[i]), "P" if star_px[i] else "S"))
    res.n_psf = int(np.count_nonzero(ok & star_px))
    res.n_sky = int(np.count_nonzero(ok & ~star_px))
    res.n_left = int(np.count_nonzero(~ok))
    if res.n_left and not res.reason_left:
        res.reason_left = "no local sky"
    res.data = out.astype(np.asarray(data).dtype, copy=False)
    return res


def _bp_field(seed, kind):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "_bp_fixture", Path(__file__).with_name("test_badpix_repair_2026_10.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    img, stars = mod._field(seed=seed)
    rng = np.random.default_rng(seed)
    for k, (x0, y0, _f) in enumerate(stars[:25]):                # star peaks and wings missing
        img[int(round(y0)) + (k % 3) - 1, int(round(x0))] = np.nan
    img[rng.integers(0, 640, 60), rng.integers(0, 640, 60)] = np.nan   # sky pixels
    img[0:2, 100:104] = np.nan                                   # at the frame edge
    if kind == "inf":
        img[300, 300] = np.inf
        img[310, 41] = -np.inf
    return img


@pytest.mark.parametrize("seed,kind,sat", [(1, "nan", 60000.0), (2, "inf", 60000.0), (3, "nan", None),
                                           (4, "nan", 900.0)])
def test_badpix_repair_bit_identical(seed, kind, sat):
    from badpix_repair import repair_missing_pixels

    img = _bp_field(seed, kind)
    kw = dict(fwhm_px=3.2, detection_sigma=5.0, sat_cal_adu=sat)
    ref = _bp_repair_ref(img.copy(), **kw)
    new = repair_missing_pixels(img.copy(), **kw)
    assert _bits_equal(np.asarray(ref.data), np.asarray(new.data))
    for f in ("n_missing", "n_psf", "n_sky", "n_left", "reason_left", "n_build_stars", "n_stars_fitted", "xy_kind"):
        assert getattr(ref, f) == getattr(new, f), f
    if sat == 60000.0:
        assert new.n_psf > 0 and new.n_sky > 0
