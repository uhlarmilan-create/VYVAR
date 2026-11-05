"""v11 item 26c / 26d (2026-11-04, Milan approved): pure refactors, no output byte changes.

26c (G12-19 rest): the ``non_calibrated`` archive-path rule has one implementation
(``draft_provenance.draft_root_of_import_path``); every former inline copy gives the identical path.
26d (G09-27 rest): fewer reads of each proc CSV in the ePSF night stage; the written proc CSVs, the
job summary and the invariant records are byte / value identical to the code before the change.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_py"))

from draft_provenance import draft_archive_root, draft_root_of_import_path  # noqa: E402

_SRC_PY = Path(__file__).resolve().parents[2] / "src_py"


# --- 26c: the non_calibrated rule ----------------------------------------------------------------------


def _old_inline_rule(ap):
    """The rule as it was written inline at night_run.py (x2), pipeline.py and masterstar_build.py."""
    return ap.parent if ap.name.casefold() == "non_calibrated" else ap


_PATHS = [
    "/data/Archive/draft_000527",
    "/data/Archive/draft_000527/",
    "/data/Archive/draft_000527/non_calibrated",
    "/data/Archive/draft_000527/non_calibrated/",
    "/data/Archive/draft_000527/non_calibrated//",
    "/data/Archive/draft_000527/NON_CALIBRATED",
    "/data/Archive/draft_000527/Non_Calibrated",
    "/data/Archive/draft_000527/non_calibrated/lights",
    "/data/Archive/draft_000527/calibrated",
    "/data/Archive/draft_000527/lights",
    "/data/Archive/draft_000527/non_calibrated_old",
    "relative/draft_000527/non_calibrated",
    "non_calibrated",
    "~/Archive/draft_000527/non_calibrated",
    "",
    ".",
    "/",
    "C:\\ASTRO\\Archive\\draft_000527\\non_calibrated",
]


@pytest.mark.parametrize("raw", _PATHS)
@pytest.mark.parametrize("kind", [Path, PurePosixPath, PureWindowsPath])
def test_26c_rule_equals_old_inline_rule(raw: str, kind) -> None:
    ap = kind(raw)
    got = draft_root_of_import_path(ap)
    assert got == _old_inline_rule(ap)
    assert type(got) is type(ap)
    assert str(got) == str(_old_inline_rule(ap))


@pytest.mark.parametrize("raw", _PATHS)
def test_26c_call_site_forms_equal_old(raw: str) -> None:
    """The exact expressions of the four former call sites (str input as night_run / pipeline get it)."""
    # night_run._nr_calibrate: Path(str(import_result.archive_path))
    ap = Path(str(raw))
    assert draft_root_of_import_path(ap) == _old_inline_rule(ap)
    # masterstar_build.generate_masterstar_and_catalog: expanduser first
    ap2 = Path(raw).expanduser()
    old2 = ap2.parent if ap2.name.casefold() == "non_calibrated" else ap2
    assert draft_root_of_import_path(Path(raw).expanduser()) == old2


def test_26c_none_raises_like_old() -> None:
    with pytest.raises(TypeError):
        Path(None)  # type: ignore[arg-type]  # every call site builds Path(...) first: None never reaches the rule
    with pytest.raises(AttributeError):
        draft_root_of_import_path(None)  # type: ignore[arg-type]
    with pytest.raises(AttributeError):
        _old_inline_rule(None)


@pytest.mark.parametrize("raw", [
    "/data/Archive/draft_000527", "/data/Archive/draft_000527/non_calibrated",
    "/data/Archive/draft_000527/NON_CALIBRATED/", "/data/Archive/draft_000527/calibrated",
    "/data/Archive/draft_000527/calibrated/lights", "/data/Archive/draft_000527/Raw/lights",
])
def test_26c_draft_archive_root_unchanged(raw: str) -> None:
    """draft_archive_root now delegates its non_calibrated branch: same result as its old body."""
    ap = Path(raw).expanduser().resolve()
    if ap.name.casefold() == "non_calibrated":
        old = ap.parent
    elif ap.name.casefold() == "lights":
        old = ap.parent.parent
    elif ap.name.casefold() == "calibrated":
        old = ap.parent
    else:
        old = ap
    assert draft_archive_root(raw) == old


def test_26c_no_module_reimplements_the_rule() -> None:
    """Only draft_provenance holds a ``name ... == "non_calibrated"`` comparison."""
    pat = re.compile(r"""\.name\s*\.\s*(?:casefold|lower)\(\)\s*==\s*["']non_calibrated["']""")
    hits = {}
    for fp in sorted(_SRC_PY.rglob("*.py")):
        n = len(pat.findall(fp.read_text(encoding="utf-8", errors="replace")))
        if n:
            hits[fp.name] = n
    assert hits == {"draft_provenance.py": 1}


# --- 26d: ePSF night stage proc-CSV I/O ----------------------------------------------------------------

import collections  # noqa: E402
import shutil  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_N_FRAMES = 6
_N_ROWS = 400


def _write_proc_set(root: Path) -> list[Path]:
    """Synthetic proc set: per frame a proc CSV (ids, flags, saturation limit, an earlier run's PSF columns),
    frames.csv with the frame FWHM, and a small FITS; one extra proc file without a FITS."""
    from astropy.io import fits

    from proc_frame_store import _vyvar_df_to_csv, write_frame_table

    rng = np.random.default_rng(11)
    ids = [str(4_000_000_000_000_000_000 + 7919 * i) for i in range(_N_ROWS)]
    flux = 10 ** rng.uniform(3.0, 6.0, _N_ROWS)
    fits_paths = []
    frame_rows = {}
    for i in range(_N_FRAMES + 1):
        name = f"proc_F_{i:03d}.csv"
        df = pd.DataFrame({
            "ra_deg": rng.uniform(200, 201, _N_ROWS).round(7), "dec_deg": rng.uniform(40, 41, _N_ROWS).round(7),
            "mag": rng.uniform(8, 16, _N_ROWS).round(6), "catalog_id": ids,
            "x": rng.uniform(0, 2000, _N_ROWS).round(4), "y": rng.uniform(0, 1300, _N_ROWS).round(4),
            "likely_saturated": rng.random(_N_ROWS) < 0.03, "is_saturated": rng.random(_N_ROWS) < 0.02,
            "saturate_limit_adu_85pct": 58981.5, "dao_flux": (flux * (1 + 0.01 * rng.normal(size=_N_ROWS))).round(2),
            "err_bkg_source": "empirical",
            "psf_flux": 1.0, "psf_chi2": 2.0, "psf_converged": True, "psf_fit_ok": True,
            "psf_chi2_locus_resid_sigma": 0.5,
        })
        df.loc[5, "dao_flux"] = np.nan
        _vyvar_df_to_csv(df, root / name)
        frame_rows[name] = {"jd_mid": 2461154.3 + i * 0.001, "fwhm_px_for_aperture": 2.0 + 0.1 * i}
        if i < _N_FRAMES:
            fp = root / f"F_{i:03d}.fits"
            d = np.zeros((8, 8), dtype=np.float32)
            d[0, 0] = i
            fits.PrimaryHDU(d).writeto(fp)
            fits_paths.append(fp)
    write_frame_table(root, frame_rows)
    return fits_paths


@pytest.fixture
def epsf_stubs(monkeypatch: pytest.MonkeyPatch):
    """The heavy parts of the ePSF job (model build, fit, spatial delta, fit sources) as deterministic stubs;
    frame 2's fit fails (the clear path)."""
    import epsf_frame
    import frame_footprint
    import parallel_frames
    import psf_photometry

    ids = [str(4_000_000_000_000_000_000 + 7919 * i) for i in range(_N_ROWS)]
    stars = pd.DataFrame({"catalog_id": ids, "x": 1.0, "y": 1.0})

    def _fit(data, model, st, nb, *, fwhm_px, gain, read_noise_e, sat_limit_adu, covered):
        seed = int(data[0, 0])
        if seed == 2:
            raise RuntimeError("synthetic fit failure")
        rng = np.random.default_rng(seed)
        f = 10 ** np.random.default_rng(0).uniform(3.0, 6.0, _N_ROWS) * (1 + 0.02 * rng.normal(size=_N_ROWS))
        chi2 = 10 ** (-3.9 + 1.06 * np.log10(f) + 0.08 * rng.normal(size=_N_ROWS))
        f[rng.random(_N_ROWS) < 0.02] = np.nan
        return pd.DataFrame({
            "catalog_id": st["catalog_id"].to_numpy(), "psf_flux": f, "psf_flux_err": np.sqrt(np.abs(f)),
            "psf_chi2": chi2, "psf_converged": rng.random(_N_ROWS) > 0.03,
            "psf_sat_masked": rng.random(_N_ROWS) < 0.05, "psf_edge_masked": rng.random(_N_ROWS) < 0.02,
            "psf_sky": 2400 + rng.normal(size=_N_ROWS), "psf_sky_pass1": 2400 + rng.normal(size=_N_ROWS),
        })

    monkeypatch.setattr(epsf_frame, "night_spatial_delta", lambda files, bs, **k: {"file": None, "grid": [1, 1]})
    monkeypatch.setattr(epsf_frame, "frame_fit_sources", lambda ps, ids_, **k: (stars, None, {}))
    monkeypatch.setattr(epsf_frame, "build_frame_model",
                        lambda fp, bs, **k: (object(), {"n_stars_used": 200, "cutout_size": k["cutout_size"],
                                                        "fwhm_px": k["fwhm_px"]}))
    monkeypatch.setattr(epsf_frame, "fit_frame_psf", _fit)
    monkeypatch.setattr(epsf_frame, "frame_fit_stats", lambda df, name: {"group_size_max": 1})
    monkeypatch.setattr(frame_footprint, "aligned_footprint", lambda hdr, shape: None)
    monkeypatch.setattr(psf_photometry, "_psf_resolve_gain_read_noise", lambda *a, **k: (1.0, 5.0))
    monkeypatch.setattr(parallel_frames, "stage_workers", lambda *a, **k: 1)

    class _FitSet:
        catalog_ids = set(ids)
        empty_reason = ""

        def to_meta_dict(self):
            return {"n": len(ids)}

    return _FitSet()


def _setup(tmp: Path) -> tuple[Path, Path, list[Path]]:
    import json

    frames, ps = tmp / "frames", tmp / "ps"
    frames.mkdir(parents=True)
    ps.mkdir()
    files = _write_proc_set(frames)
    (ps / "epsf_build_stars_meta.json").write_text(json.dumps({"cutout_size": 25, "fwhm_px": 2.0}))
    pd.DataFrame({"catalog_id": ["1"], "x": [1.0], "y": [1.0]}).to_csv(ps / "masterstar_epsf_build_stars.csv",
                                                                       index=False)
    return frames, ps, files


def _old_sequence(frames: Path, ps: Path, files: list[Path], fit_set) -> tuple[list[dict], dict]:
    """The per-CSV I/O sequence before v11 26d, from the unchanged primitives: the worker reads the frame FWHM
    and saturation limit itself, the merge keeps nothing, the locus reads every file and n_ok is counted from
    disk (reference copy of the old finalize loop)."""
    import epsf_psf_merge as em
    from proc_frame_store import proc_csv_path_for_aligned_fits
    from psf_chi2_locus import _count_psf_fit_ok_in_sidecar, reapply_night_locus_to_proc_dir

    build_stars = pd.read_csv(ps / "masterstar_epsf_build_stars.csv", dtype={"catalog_id": str})
    stars, _, _ = __import__("epsf_frame").frame_fit_sources(ps, fit_set.catalog_ids)
    em._fit_worker_init({"build_stars": build_stars, "delta": None, "grid": (1, 1), "platesolve_dir": str(ps),
                         "app_config": None, "fit_stars": stars, "neighbours": None})
    meta: dict = {}
    records = []
    for i, fp in enumerate(files):
        res = em._fit_frame_task(fp)
        rec = {"frame_name": fp.name, "frame_index": i, "n_fit": 0, "n_ok": 0, "exception_class": None}
        if res["exception_class"] is None:
            rec["n_fit"] = em.merge_psf_frame_result(proc_csv_path_for_aligned_fits(fp), res["fit_df"],
                                                     frame_name=fp.name, pipeline_meta=meta)
        else:
            rec["exception_class"] = res["exception_class"]
            em.clear_psf_frame_result(proc_csv_path_for_aligned_fits(fp), frame_name=fp.name, pipeline_meta=meta)
        records.append(rec)
    em._fit_worker_init({})
    summary = reapply_night_locus_to_proc_dir(frames)
    for rec in records:
        sc = proc_csv_path_for_aligned_fits(frames / rec["frame_name"])
        if sc.is_file():
            rec["n_ok"] = _count_psf_fit_ok_in_sidecar(sc)
    return records, summary


def test_26d_epsf_job_bytes_equal_old_sequence(tmp_path: Path, epsf_stubs) -> None:
    """Proc CSV bytes, n_ok per frame and the night locus summary equal the old read-per-pass sequence."""
    import epsf_psf_merge as em

    frames_a, ps_a, _ = _setup(tmp_path / "a")
    shutil.copytree(tmp_path / "a", tmp_path / "b")
    frames_b, ps_b = tmp_path / "b" / "frames", tmp_path / "b" / "ps"
    files_b = sorted(frames_b.glob("F_*.fits"))

    meta: dict = {}
    out = em.run_epsf_psf_merge_job(frames_root=frames_a, platesolve_dir=ps_a, app_config=None, fit_set=epsf_stubs,
                                    pipeline_meta=meta)
    old_recs, old_summary = _old_sequence(frames_b, ps_b, files_b, epsf_stubs)

    names = sorted(p.name for p in frames_a.glob("proc_*.csv"))
    assert names == sorted(p.name for p in frames_b.glob("proc_*.csv")) and len(names) == _N_FRAMES + 1
    for n in names:
        assert (frames_a / n).read_bytes() == (frames_b / n).read_bytes(), n
    assert [(r["frame_name"], r["n_fit"], r["n_ok"], r["exception_class"]) for r in out["frames"]] == [
        (r["frame_name"], r["n_fit"], r["n_ok"], r["exception_class"]) for r in old_recs]
    assert out["frames"][2]["exception_class"] == "RuntimeError"
    assert sum(r["n_ok"] for r in out["frames"]) > 0
    new_locus = meta["psf_chi2_locus_night"]
    assert new_locus["night_locus"] is not None
    for k in ("n_sigma", "night_locus", "frames", "n_rewritten", "n_kept_frame", "offsets", "offset_spread"):
        assert new_locus[k] == old_summary[k], k


def test_26d_epsf_job_reads_each_proc_csv_once_before_the_merge(tmp_path: Path, epsf_stubs,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """Per successfully fitted frame: one column read (FWHM + saturation limit) instead of five (night FWHM,
    frame FWHM, saturation limit, locus columns, psf_fit_ok recount); the full reads / writes are unchanged
    (merge + guarded read-back, locus reapply + guarded read-back)."""
    import epsf_psf_merge as em

    frames, ps, _ = _setup(tmp_path)
    calls: collections.Counter = collections.Counter()
    orig = pd.read_csv

    def _counting(path, *a, **k):
        name = Path(str(path)).name
        if name.startswith("proc_"):
            calls[(name, "full" if k.get("usecols") is None else "cols")] += 1
        return orig(path, *a, **k)

    monkeypatch.setattr(pd, "read_csv", _counting)
    em.run_epsf_psf_merge_job(frames_root=frames, platesolve_dir=ps, app_config=None, fit_set=epsf_stubs,
                              pipeline_meta={})
    for i in range(_N_FRAMES):
        name = f"proc_F_{i:03d}.csv"
        if i == 2:  # the failed fit: cleared (merge + clear rewrite), its locus columns read from disk
            assert calls[(name, "cols")] == 2 and calls[(name, "full")] == 6
        else:
            assert calls[(name, "cols")] == 1, (name, dict(calls))
            assert calls[(name, "full")] == 4, (name, dict(calls))
    # the extra proc file (no FITS, not fitted) is seen by the night locus only
    assert calls[("proc_F_006.csv", "cols")] == 1 and calls[("proc_F_006.csv", "full")] == 2


def test_26d_locus_columns_of_full_read_equal_column_read(tmp_path: Path, epsf_stubs) -> None:
    """The kept locus table (full read-back, column subset) equals a read of those columns only."""
    import epsf_psf_merge as em
    from gaia_catalog_id import read_vyvar_csv
    from psf_chi2_locus import LOCUS_PROC_COLUMNS, locus_columns_of

    frames, ps, _ = _setup(tmp_path)
    em.run_epsf_psf_merge_job(frames_root=frames, platesolve_dir=ps, app_config=None, fit_set=epsf_stubs,
                              pipeline_meta={})
    for fp in sorted(frames.glob("proc_*.csv")):
        a = locus_columns_of(read_vyvar_csv(fp, low_memory=False))
        b = read_vyvar_csv(fp, usecols=lambda c: c in LOCUS_PROC_COLUMNS, low_memory=False)
        pd.testing.assert_frame_equal(a, b, check_exact=True)
