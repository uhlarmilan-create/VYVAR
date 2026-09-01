# -*- coding: ascii -*-
"""INVARIANTS P1 golden mini: headless SHA, UI-order identity, census, physics.

Opt-in: set VYVAR_INVARIANTS_P1=1 (same gate as test_invariants_p1_seed.py).
--fast stays unaffected (these tests skip without the env flag).

Runbook:
  python dev/tools/build_p1_golden_mini.py
  set VYVAR_INVARIANTS_P1=1
  pytest dev/tests/test_invariants_p1_seed.py dev/tests/test_invariants_p1_golden.py -q
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import AppConfig
from database import VyvarDatabase
from except_fix_counters import reset_except_fix_counters
from photometry_core import run_full_photometry_pipeline
from tests.photometry_sha import (
    PHOTOMETRY_PROVENANCE_COLS,
    PHOTOMETRY_QC_COLS_LC,
    compare_photometry_science_meaningful,
    compute_photometry_sha,
)
from tools.reference_seed import seed_reference_observatory
from ui_aperture_photometry import _find_phase2a_paths

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "dev" / "validation" / "VYVAR_VALIDATION_LEDGER.json"
MINI_NAME = "draft_000516_p1mini"
SETUP = "NoFilter_60_2"
DRAFT_ID = 516


def _enabled() -> bool:
    return str(os.environ.get("VYVAR_INVARIANTS_P1", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in ("1", "true", "yes", "on")


def _p1_force_execution() -> bool:
    """When True, headless_mini must run the pipeline (lock/verify path).

    Default under VYVAR_INVARIANTS_P1=1: always execute. Opt-in reuse only when
    VYVAR_P1_REUSE_FROZEN=1 (visible pytest.skip, never a silent PASS).
    """
    if _truthy_env("VYVAR_P1_FORCE"):
        return True
    if _truthy_env("VYVAR_P1_REUSE_FROZEN"):
        return False
    # Full P1 gate runs and session --full both set FORCE explicitly.
    return True


def _mini_matches_locked_core(mini: Path, gold: dict) -> bool:
    try:
        core, nc = compute_photometry_sha(mini, include_comp_qa=False)
        return core == gold["core_sha"] and nc == gold["core_n"]
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _enabled(), reason="set VYVAR_INVARIANTS_P1=1 to run P1 golden"
)


def _ledger_p1() -> dict:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    for it in ledger["items"]:
        if it["id"] == "VL-P1-GOLD":
            return it
    raise AssertionError("VL-P1-GOLD missing from validation ledger")


def _mini_root(cfg: AppConfig | None = None) -> Path:
    cfg = cfg or AppConfig()
    return Path(cfg.archive_root) / "Drafts" / MINI_NAME


def _cfg_for_p1() -> AppConfig:
    cfg = AppConfig()
    cfg.k2_mode = "literature"
    cfg.save_lightcurve_png = False
    cfg.per_frame_saturation_enabled = True
    return cfg


def _wipe_photometry(mini: Path) -> Path:
    out = mini / "platesolve" / SETUP / "photometry"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    return out


def _p1_headless_chain(mini: Path, *, output_dir: Path | None = None) -> Path:
    """Headless composition matching session_baseline_check.run_full_baseline.

    Call map:
      session_baseline_check.py:370-380  run_full_photometry_pipeline(...)
    """
    cfg = _cfg_for_p1()
    ps = mini / "platesolve" / SETUP
    lights = mini / "detrended_aligned" / "lights" / SETUP
    out = output_dir or _wipe_photometry(mini)
    out.mkdir(parents=True, exist_ok=True)
    reset_except_fix_counters()
    db = VyvarDatabase(cfg.database_path)
    seed_reference_observatory(db)
    try:
        run_full_photometry_pipeline(
            masterstar_fits_path=ps / "MASTERSTAR.fits",
            variable_targets_csv=ps / "variable_targets.csv",
            masterstars_csv=ps / "masterstars_full_match.csv",
            per_frame_csv_dir=lights,
            detrended_aligned_dir=lights,
            output_dir=out,
            cfg=cfg,
            db=db,
            draft_id=DRAFT_ID,
        )
    finally:
        try:
            db.conn.close()
        except Exception:  # noqa: BLE001
            pass
    return out


def _p1_ui_chain(mini: Path, *, output_dir: Path | None = None) -> Path:
    """UI-order photometry composition (photometry-ready mini scope).

    Stage map (src_py/app.py _run_vyvar_full_pipeline):
      calibrate  app.py:311  quick_calibrate_last_import
                 SKIPPED -- mini starts photometry-ready (see DECISIONS
                 INVARIANTS-P1-GOLDEN-MINI scope note; no in-draft masters).
      QC         app.py:344  run_draft_ram_calibration_qc_to_obs_files
                 SKIPPED -- same scope.
      preprocess app.py:532  _vyvar_execute_preprocess_pending
                 SKIPPED -- proc products frozen from parent stride.
      align/MS   app.py:538  _vyvar_execute_platesolve_pending
                 SKIPPED -- MASTERSTAR + per-frame catalogs frozen.
      phase0/1/2a app.py:549  _find_phase2a_paths
                 app.py:594  run_full_photometry_pipeline
                 EXECUTED below (UI path discovery + photometry call).

    night_run.py twin for the executed block: L970 _find_phase2a_paths,
    L1028 run_full_photometry_pipeline.
    """
    cfg = _cfg_for_p1()
    all_setups = _find_phase2a_paths(cfg, None, draft_dir_override=mini)
    assert all_setups, "UI chain: _find_phase2a_paths returned no setups"
    assert SETUP in all_setups, f"UI chain: missing setup {SETUP} in {list(all_setups)}"
    p = all_setups[SETUP]
    ms_fits = Path(p["masterstar_fits"])
    og_dir = Path(p["obs_group_dir"])
    ms_csv = og_dir / "masterstars_full_match.csv"
    vt_csv = og_dir / "variable_targets.csv"
    pf_dir = Path(p["per_frame_csv_dir"])
    dt_dir = Path(p["detrended_aligned_dir"])
    out = Path(output_dir) if output_dir is not None else Path(p["output_dir"])
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    reset_except_fix_counters()
    db = VyvarDatabase(cfg.database_path)
    seed_reference_observatory(db)
    try:
        run_full_photometry_pipeline(
            masterstar_fits_path=ms_fits,
            variable_targets_csv=vt_csv,
            masterstars_csv=ms_csv,
            per_frame_csv_dir=pf_dir,
            detrended_aligned_dir=dt_dir,
            output_dir=out,
            cfg=cfg,
            db=db,
            draft_id=DRAFT_ID,
        )
    finally:
        try:
            db.conn.close()
        except Exception:  # noqa: BLE001
            pass
    return out


def _diff_science_csvs(root_a: Path, root_b: Path, setup: str = SETUP) -> list[str]:
    """Per-file / per-column science diff summary (F-431 style report)."""
    lines: list[str] = []
    lc_a = root_a / "platesolve" / setup / "photometry" / "lightcurves"
    lc_b = root_b / "platesolve" / setup / "photometry" / "lightcurves"
    if not lc_a.is_dir() or not lc_b.is_dir():
        return [f"missing lightcurves dir a={lc_a.is_dir()} b={lc_b.is_dir()}"]
    names_a = {p.name for p in lc_a.glob("lightcurve_*.csv")}
    names_b = {p.name for p in lc_b.glob("lightcurve_*.csv")}
    if names_a != names_b:
        lines.append(
            f"LC set mismatch only_a={sorted(names_a - names_b)[:5]} "
            f"only_b={sorted(names_b - names_a)[:5]}"
        )
    for name in sorted(names_a & names_b):
        da = pd.read_csv(lc_a / name, low_memory=False)
        db = pd.read_csv(lc_b / name, low_memory=False)
        if len(da) != len(db):
            lines.append(f"{name}: row_count {len(da)} vs {len(db)}")
            continue
        for col in sorted(set(da.columns) & set(db.columns)):
            if col in PHOTOMETRY_PROVENANCE_COLS or col in PHOTOMETRY_QC_COLS_LC:
                continue
            if da[col].dtype == bool or db[col].dtype == bool:
                if not da[col].equals(db[col]):
                    lines.append(f"{name}.{col}: bool diff")
                continue
            na = pd.to_numeric(da[col], errors="coerce")
            nb = pd.to_numeric(db[col], errors="coerce")
            if not (na.notna().any() and nb.notna().any()):
                if not da[col].astype(str).equals(db[col].astype(str)):
                    lines.append(f"{name}.{col}: string/non-numeric diff")
                continue
            delta = float(np.nanmax(np.abs(na.to_numpy() - nb.to_numpy())))
            if math.isfinite(delta) and delta > 1e-6:
                lines.append(f"{name}.{col}: max|d|={delta:.3e}")
    return lines


@pytest.fixture(scope="module")
def mini_and_gold() -> tuple[Path, dict]:
    cfg = AppConfig()
    mini = _mini_root(cfg)
    gold = _ledger_p1()
    assert mini.is_dir(), (
        f"missing mini {mini}; run: python dev/tools/build_p1_golden_mini.py"
    )
    return mini, gold


@pytest.fixture(scope="module")
def headless_mini(mini_and_gold: tuple[Path, dict]) -> Path:
    """Module-scoped headless run; leaves science outputs on the mini.

    When VYVAR_P1_REUSE_FROZEN=1 and the mini already matches VL-P1-GOLD, dependent
    tests skip visibly (never report PASS without executing the pipeline).
    """
    mini, gold = mini_and_gold
    if not _p1_force_execution() and _mini_matches_locked_core(mini, gold):
        pytest.skip(
            "SKIPPED (reused frozen outputs at VL-P1-GOLD core SHA); "
            "unset VYVAR_P1_REUSE_FROZEN to force execution"
        )
    t0 = time.time()
    _p1_headless_chain(mini)
    print(f"\n[P1] headless chain {time.time() - t0:.1f}s")
    return mini


def test_mini_present_or_buildable(mini_and_gold: tuple[Path, dict]) -> None:
    mini, gold = mini_and_gold
    man_path = mini / "p1_manifest.json"
    assert man_path.is_file(), f"missing {man_path}"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    assert man["n_frames"] == 16
    assert man["inputs_manifest_sha256"] == gold["inputs_manifest_sha256"]
    # Verify each input file SHA matches the build manifest.
    # Skip files the photometry pipeline is allowed to rewrite in-place.
    _rewriteable = {"platesolve/NoFilter_60_2/alignment_report.csv"}
    for item in man["inputs"]:
        if item["rel"].replace("\\", "/") in _rewriteable:
            continue
        path = mini / item["rel"]
        assert path.is_file(), f"missing input {item['rel']}"
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        assert h == item["sha256"], f"SHA drift on {item['rel']}"


def test_headless_chain_sha(headless_mini: Path, mini_and_gold: tuple[Path, dict]) -> None:
    _mini, gold = mini_and_gold
    core, nc = compute_photometry_sha(headless_mini, include_comp_qa=False)
    ext, ne = compute_photometry_sha(headless_mini, include_comp_qa=True)
    assert core == gold["core_sha"], f"core SHA mismatch: {core} != {gold['core_sha']}"
    assert ext == gold["extended_sha"], f"ext SHA mismatch: {ext} != {gold['extended_sha']}"
    assert nc == gold["core_n"]
    assert ne == gold["extended_n"]


def test_ui_chain_byte_identity(headless_mini: Path, mini_and_gold: tuple[Path, dict]) -> None:
    """UI-order vs headless science identity on P1 mini (both chains must execute)."""
    # F3 discriminator PROMOTED from forensic_disc_ui_match2.py (P3 pilot):
    # app.py RUN VYVAR pins cat_match_arc=2.0 (call site ~app.py:2169); NightRun
    # default must match that UI parity value.
    _app_src = (REPO_ROOT / "src_py" / "app.py").read_text(encoding="ascii")
    assert "cat_match_arc=2.0" in _app_src, (
        "UI RUN VYVAR must pin cat_match_arc=2.0 (F3/F-431 parity; app.py call site)"
    )
    from dataclasses import fields  # noqa: PLC0415
    from night_run import NightRunParams  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    _nr_default = next(
        f.default
        for f in fields(NightRunParams)
        if f.name == "catalog_match_max_sep_arcsec"
    )
    assert float(_nr_default) == 2.0, (
        "NightRunParams.catalog_match_max_sep_arcsec default must be 2.0 "
        "(UI RUN VYVAR parity)"
    )
    # Also pin via a minimal constructed instance (required ctor args are dummies).
    _nr = NightRunParams(
        source_dir=_Path("."), equipment_id=1, telescope_id=1
    )
    assert float(_nr.catalog_match_max_sep_arcsec) == 2.0

    mini, _gold = mini_and_gold
    # Snapshot mini skeleton without photometry outputs for a second chain
    work = REPO_ROOT / "tmp" / "p1_ui_chain_work"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(mini, work, ignore=shutil.ignore_patterns("photometry"))
    phot_work = work / "platesolve" / SETUP / "photometry"
    if phot_work.exists():
        shutil.rmtree(phot_work)

    t0 = time.time()
    _p1_ui_chain(work)
    print(f"\n[P1] UI chain {time.time() - t0:.1f}s")

    cmp = compare_photometry_science_meaningful(
        headless_mini, work, setups=(SETUP,)
    )
    summary = cmp.get("summary") or {}
    diffs = _diff_science_csvs(headless_mini, work)
    if not summary.get("benign") or diffs:
        report = (
            "F-431-class divergence: UI-order chain != headless on P1 mini.\n"
            f"comparator summary={summary}\n"
            + "\n".join(diffs[:40])
        )
        pytest.fail(report)

    # Byte-compare normalized science CSV payloads (mag*/flux* + times)
    core_a, _ = compute_photometry_sha(headless_mini, include_comp_qa=False)
    core_b, _ = compute_photometry_sha(work, include_comp_qa=False)
    assert core_a == core_b, (
        f"F-431-class SHA divergence after UI chain: {core_a} vs {core_b}\n"
        + "\n".join(diffs[:40])
    )


def test_census_bands(headless_mini: Path, mini_and_gold: tuple[Path, dict]) -> None:
    _mini, gold = mini_and_gold
    bands = gold["census"]
    from astropy.io import fits

    with fits.open(headless_mini / "platesolve" / SETUP / "MASTERSTAR.fits") as hdul:
        dao = int(hdul[0].header["VY_NDAO"])
    idx = pd.read_csv(
        headless_mini / "platesolve" / SETUP / "per_frame_catalog_index.csv"
    )
    n_det = float(idx["n_detected"].mean())
    n_mat = float(idx["n_matched"].mean())

    def _within(actual: float, expected: float, label: str) -> None:
        lo = expected * 0.95
        hi = expected * 1.05
        assert lo <= actual <= hi, (
            f"census band fail {label}: {actual} not in [{lo}, {hi}] (+-5% of {expected})"
        )

    _within(float(dao), float(bands["dao_pass1_vy_ndao"]), "dao_pass1")
    _within(n_det, float(bands["n_detected_mean"]), "n_detected_mean")
    _within(n_mat, float(bands["n_matched_mean"]), "n_matched_mean")


def test_physics_asserts(headless_mini: Path, mini_and_gold: tuple[Path, dict]) -> None:
    _mini, gold = mini_and_gold
    lc_dir = headless_mini / "platesolve" / SETUP / "photometry" / "lightcurves"
    lcs = sorted(lc_dir.glob("lightcurve_*.csv"))
    assert lcs, "no lightcurves for physics asserts"

    # WCS identity p95 band (inherited MASTERSTAR / parent baseline)
    p95 = float(gold["census"]["identity_p95_parent"])
    assert p95 <= 2.0, f"identity p95 {p95} > 2.0 px"

    # Spot-check saturated rows not used as ensemble comps
    comp_path = (
        headless_mini
        / "platesolve"
        / SETUP
        / "photometry"
        / "comparison_stars_per_target.csv"
    )
    if comp_path.is_file():
        comps = pd.read_csv(comp_path, low_memory=False)
        role_col = next(
            (c for c in comps.columns if c.lower() in ("role", "comp_role", "status")),
            None,
        )
        sat_col = next(
            (c for c in comps.columns if "saturat" in c.lower()),
            None,
        )
        if role_col and sat_col:
            ens = comps[
                comps[role_col].astype(str).str.lower().isin(
                    ("comp", "comparison", "ensemble", "good")
                )
            ]
            if len(ens) and ens[sat_col].dtype != object:
                assert not bool((ens[sat_col].fillna(0).astype(float) > 0).any()), (
                    "saturated star present in ensemble roles"
                )

    checked = 0
    for lc in lcs:
        df = pd.read_csv(lc, low_memory=False)
        if df.empty:
            continue
        checked += 1
        # jd strictly increasing
        if "jd" in df.columns:
            jd = pd.to_numeric(df["jd"], errors="coerce").to_numpy()
            finite = np.isfinite(jd)
            if finite.sum() >= 2:
                assert np.all(np.diff(jd[finite]) > 0), f"{lc.name}: jd not strictly increasing"
        # airmass >= 1
        if "airmass" in df.columns:
            am = pd.to_numeric(df["airmass"], errors="coerce").to_numpy()
            ok = np.isfinite(am)
            assert np.all(am[ok] >= 1.0 - 1e-9), f"{lc.name}: airmass < 1"

        # err finite and > 0 on scientifically usable rows (finite mag + normal flag)
        if "err" in df.columns and "mag_calib" in df.columns:
            flag = (
                df["flag"].astype(str).str.lower()
                if "flag" in df.columns
                else pd.Series(["normal"] * len(df))
            )
            mag = pd.to_numeric(df["mag_calib"], errors="coerce")
            good = flag.eq("normal") & mag.notna() & np.isfinite(mag.to_numpy())
            if good.any():
                err = pd.to_numeric(df.loc[good, "err"], errors="coerce")
                assert err.notna().all() and (err > 0).all(), (
                    f"{lc.name}: bad err on normal finite-mag rows"
                )

        # mag_calib_final == mag_calib + ct + ac (where ok)
        if {"mag_calib", "mag_calib_final"}.issubset(df.columns):
            base = pd.to_numeric(df["mag_calib"], errors="coerce").to_numpy()
            final = pd.to_numeric(df["mag_calib_final"], errors="coerce").to_numpy()
            ct_ok = (
                df["ct_ok"].astype(bool).to_numpy()
                if "ct_ok" in df.columns
                else np.zeros(len(df), dtype=bool)
            )
            ac_ok = (
                df["ac_ok"].astype(bool).to_numpy()
                if "ac_ok" in df.columns
                else np.zeros(len(df), dtype=bool)
            )
            ct = (
                pd.to_numeric(df["ct_correction"], errors="coerce").fillna(0.0).to_numpy()
                if "ct_correction" in df.columns
                else np.zeros(len(df))
            )
            ac = (
                pd.to_numeric(df["ac_correction"], errors="coerce").fillna(0.0).to_numpy()
                if "ac_correction" in df.columns
                else np.zeros(len(df))
            )
            expected = base + np.where(ct_ok, ct, 0.0) + np.where(ac_ok, ac, 0.0)
            mask = np.isfinite(base) & np.isfinite(final)
            if mask.any():
                # LC CSV rounds mag columns to 6 decimals; allow 1.5e-6 slack.
                assert np.nanmax(np.abs(final[mask] - expected[mask])) <= 1.5e-6, (
                    f"{lc.name}: mag_calib_final composition fail"
                )

        # err^2 >= each present component^2 (slack 1e-9); only columns that exist
        if "err" in df.columns:
            err = pd.to_numeric(df["err"], errors="coerce").to_numpy()
            for col in ("err_photon", "err_bkg", "err_sem", "sem_ens", "sigma_sys_mag"):
                if col not in df.columns:
                    continue
                comp = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
                m = np.isfinite(err) & (err > 0)
                if not m.any():
                    continue
                assert np.all(err[m] ** 2 + 1e-9 >= np.clip(comp[m], 0.0, None) ** 2), (
                    f"{lc.name}: err composition vs {col}"
                )

    assert checked >= 10, f"physics asserts covered only {checked} LCs"
