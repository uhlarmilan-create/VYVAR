#!/usr/bin/env python3
"""Legacy vs scoped solver re-cut on anchor 387 (same harness control). Read-only on archive."""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from database import query_local_gaia  # noqa: E402
from tests.photometry_sha import (  # noqa: E402
    PHOTOMETRY_SCIENCE_COLS_LC,
    compare_photometry_science_meaningful,
    _lc_map,
)
from ui_aperture_photometry import _find_phase2a_paths  # noqa: E402
from vyvar_platesolver import solve_wcs_with_local_gaia  # noqa: E402

ARCHIVE = _ROOT / "Archive" / "Drafts" / "draft_000387"
TMP_LEGACY = _ROOT / "tmp" / "anchor387_legacy_recut"
TMP_SCOPED = _ROOT / "tmp" / "anchor387_scoped_recut"
REPORT_JSON = _ROOT / "tmp" / "anchor387_legacy_vs_scoped_report.json"
SETUPS = ("B_20_2", "L_20_2", "R_20_2", "V_20_2")
HOME_SCALE = 1.292594
HOME_PIX_UM = 4.63
HOME_FOCAL_MM = 200.0
FOV_DEG = 1.0

LEGACY_FLAGS = dict(
    solver_use_cone_for_sip=False,
    solver_apply_roworder_yflip=False,
    solver_legacy_masterstar_mirror_sweep=True,
    solver_fits_header_hint_sep_escape=False,
)
SCOPED_FLAGS = dict(
    solver_use_cone_for_sip=True,
    solver_apply_roworder_yflip=False,
    solver_legacy_masterstar_mirror_sweep=True,
    solver_fits_header_hint_sep_escape=True,
)


def _gaia_delta(w_arch: WCS, w_new: WCS, *, naxis1: int, naxis2: int, gaia_db: str) -> dict:
    ra0, de0 = float(w_arch.wcs.crval[0]), float(w_arch.wcs.crval[1])
    rows = query_local_gaia(
        gaia_db,
        ra_min=ra0 - 0.12,
        ra_max=ra0 + 0.12,
        dec_min=de0 - 0.12,
        dec_max=de0 + 0.12,
        mag_limit=14.5,
        max_rows=8000,
    )
    ra = np.asarray([float(r["ra"]) for r in rows], dtype=np.float64)
    de = np.asarray([float(r["dec"]) for r in rows], dtype=np.float64)
    xa, ya = w_arch.all_world2pix(ra, de, 0)
    xb, yb = w_new.all_world2pix(ra, de, 0)
    ok = (
        np.isfinite(xa)
        & np.isfinite(ya)
        & np.isfinite(xb)
        & np.isfinite(yb)
        & (xa >= 0)
        & (xa < float(naxis1))
        & (ya >= 0)
        & (ya < float(naxis2))
    )
    d = np.hypot(xa[ok] - xb[ok], ya[ok] - yb[ok])
    return {
        "delta_median_px": float(np.median(d)) if d.size else float("nan"),
        "delta_p95_px": float(np.percentile(d, 95)) if d.size else float("nan"),
        "n_on_chip": int(d.size),
    }


def _max_dmag_per_setup(root_a: Path, root_b: Path, setup: str) -> float:
    lca, lcb = _lc_map(root_a, setup), _lc_map(root_b, setup)
    max_d = 0.0
    for tid in set(lca) & set(lcb):
        da = pd.read_csv(lca[tid], low_memory=False)
        db = pd.read_csv(lcb[tid], low_memory=False)
        for col in da.columns:
            if col.lower() in PHOTOMETRY_SCIENCE_COLS_LC or col.lower().startswith(("mag", "flux")):
                if col not in db.columns:
                    continue
                na = pd.to_numeric(da[col], errors="coerce")
                nb = pd.to_numeric(db[col], errors="coerce")
                if na.notna().any() and nb.notna().any():
                    delta = float(np.nanmax(np.abs(na - nb)))
                    if np.isfinite(delta):
                        max_d = max(max_d, delta)
    return max_d


def _science_failures_by_setup(report: dict) -> dict[str, int]:
    out = {s: 0 for s in SETUPS}
    for row in report.get("_all_science_failures", []):
        out[row["setup"]] = out.get(row["setup"], 0) + 1
    return out


def _compare_with_details(root_a: Path, root_b: Path) -> dict:
    rep = compare_photometry_science_meaningful(root_a, root_b)
    # Re-walk failures for per-setup counts and max dmag
    failures: list[dict] = []
    for setup in SETUPS:
        lca, lcb = _lc_map(root_a, setup), _lc_map(root_b, setup)
        for tid in sorted(set(lca) & set(lcb)):
            from tests.photometry_sha import _compare_lc_science  # noqa: PLC0415

            cmp = _compare_lc_science(lca[tid], lcb[tid])
            if not cmp.get("science_ok", True):
                failures.append({"setup": setup, "tid": tid, **cmp})
    rep["_all_science_failures"] = failures
    per_setup: dict[str, dict] = {}
    for setup in SETUPS:
        sf = [f for f in failures if f["setup"] == setup]
        per_setup[setup] = {
            "science_failures": len(sf),
            "max_abs_dmag": _max_dmag_per_setup(root_a, root_b, setup),
        }
    rep["per_setup_metrics"] = per_setup
    return rep


def _solve_all(tmp_root: Path, cfg: AppConfig, *, flags: dict, label: str) -> dict:
    gaia_db = str(cfg.gaia_db_path)
    out: dict[str, dict] = {}
    for setup in SETUPS:
        src = ARCHIVE / "platesolve" / setup / "MASTERSTAR.fits"
        dst = tmp_root / "platesolve" / setup / "MASTERSTAR.fits"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        with fits.open(src, memmap=False) as hd:
            w_arch = WCS(hd[0].header)
            n1 = int(hd[0].header.get("NAXIS1", 0))
            n2 = int(hd[0].header.get("NAXIS2", 0))
        res = solve_wcs_with_local_gaia(
            dst,
            hint_ra_deg=None,
            hint_dec_deg=None,
            fov_diameter_deg=FOV_DEG,
            gaia_db_path=gaia_db,
            effective_pixel_um=HOME_PIX_UM,
            focal_length_mm=HOME_FOCAL_MM,
            expected_plate_scale_arcsec_per_px=HOME_SCALE,
            enable_sip=True,
            app_config=cfg,
            **flags,
        )
        with fits.open(dst, memmap=False) as hd:
            w_new = WCS(hd[0].header)
        stats = _gaia_delta(w_arch, w_new, naxis1=n1, naxis2=n2, gaia_db=gaia_db)
        stats["solve_ok"] = bool(res.get("solved"))
        stats["match_rate"] = res.get("match_rate")
        out[setup] = stats
        print(f"[{label} solve] {setup}: median dpx={stats['delta_median_px']:.6f}")
    return out


def _photometry_all(tmp_root: Path, cfg: AppConfig, *, label: str) -> None:
    from photometry_core import run_full_photometry_pipeline  # noqa: PLC0415

    paths = _find_phase2a_paths(cfg, 387, draft_dir_override=ARCHIVE)
    for setup in SETUPS:
        p = paths.get(setup)
        if not p:
            raise RuntimeError(f"missing phase2a paths for {setup}")
        out_dir = tmp_root / "platesolve" / setup / "photometry"
        out_dir.mkdir(parents=True, exist_ok=True)
        ms = tmp_root / "platesolve" / setup / "MASTERSTAR.fits"
        print(f"[{label} phot] {setup} ...", flush=True)
        run_full_photometry_pipeline(
            masterstar_fits_path=ms,
            variable_targets_csv=Path(p["obs_group_dir"]) / "variable_targets.csv",
            masterstars_csv=Path(p["obs_group_dir"]) / "masterstars_full_match.csv",
            per_frame_csv_dir=Path(p["per_frame_csv_dir"]),
            detrended_aligned_dir=Path(p["detrended_aligned_dir"]),
            output_dir=out_dir,
            cfg=cfg,
            draft_id=387,
        )


def _brno_check(cfg: AppConfig) -> dict:
    candidates = (
        _ROOT / "Archive" / "Drafts" / "draft_000400" / "platesolve" / "g_60_4" / "MASTERSTAR.fits",
        _ROOT / "Archive" / "Drafts" / "draft_000399" / "platesolve" / "g_60_4" / "MASTERSTAR.fits",
    )
    brno_ms = next((p for p in candidates if p.is_file()), None)
    if brno_ms is None:
        print("brno check: skipped (no draft_400/399 g_60_4 MASTERSTAR.fits on disk)")
        return {"skipped": True, "reason": "MASTERSTAR.fits not found"}
    dst = _ROOT / "tmp" / "brno_scoped_check" / "MASTERSTAR.fits"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(brno_ms, dst)
    print(f"brno check: source={brno_ms.name} draft={brno_ms.parent.parent.parent.name}")
    res = solve_wcs_with_local_gaia(
        dst,
        hint_ra_deg=None,
        hint_dec_deg=None,
        fov_diameter_deg=0.6,
        gaia_db_path=str(cfg.gaia_db_path),
        effective_pixel_um=15.04,
        focal_length_mm=5480.0,
        expected_plate_scale_arcsec_per_px=0.566,
        enable_sip=True,
        app_config=cfg,
        **SCOPED_FLAGS,
    )
    with fits.open(dst) as hd:
        has_wcs = WCS(hd[0].header).has_celestial
    return {
        "solved": bool(res.get("solved")),
        "match_rate": res.get("match_rate"),
        "rms_px": res.get("rms_px"),
        "wcs_persist": has_wcs,
    }


def main() -> int:
    import sys

    compare_only = "--compare-only" in sys.argv
    cfg = AppConfig()
    report: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(ARCHIVE),
        "legacy_flags": LEGACY_FLAGS,
        "scoped_flags": SCOPED_FLAGS,
        "compare_only": bool(compare_only),
    }

    if compare_only:
        if not TMP_LEGACY.is_dir() or not TMP_SCOPED.is_dir():
            raise SystemExit(
                f"compare-only: missing {TMP_LEGACY} or {TMP_SCOPED} - run full gate first."
            )
        print("=== compare-only (skip re-cut; reuse tmp/) ===")
    else:
        print("=== (A) Legacy re-cut ===")
        report["legacy_wcs"] = _solve_all(TMP_LEGACY, cfg, flags=LEGACY_FLAGS, label="legacy")
        _photometry_all(TMP_LEGACY, cfg, label="legacy")

        print("=== (B) Scoped re-cut ===")
        report["scoped_wcs"] = _solve_all(TMP_SCOPED, cfg, flags=SCOPED_FLAGS, label="scoped")
        _photometry_all(TMP_SCOPED, cfg, label="scoped")

    print("=== Compare (B) vs (A) ===")
    report["scoped_vs_legacy"] = _compare_with_details(TMP_LEGACY, TMP_SCOPED)
    sf_ba = int(report["scoped_vs_legacy"]["summary"]["science_failures"])
    print(f"scoped_vs_legacy science_failures={sf_ba}")

    print("=== Compare (A) vs archive ===")
    report["legacy_vs_archive"] = _compare_with_details(ARCHIVE, TMP_LEGACY)
    sf_aa = int(report["legacy_vs_archive"]["summary"]["science_failures"])
    print(f"legacy_vs_archive science_failures={sf_aa}")

    report["brno"] = _brno_check(cfg)
    print(f"brno match={report['brno'].get('match_rate')} wcs={report['brno'].get('wcs_persist')}")

    report["decision"] = (
        "lock_scoped_solver"
        if sf_ba == 0
        else "STOP_investigate_scoped_photometry_drift"
    )
    report.pop("_all_science_failures", None)
    for key in ("scoped_vs_legacy", "legacy_vs_archive"):
        if "_all_science_failures" in report[key]:
            del report[key]["_all_science_failures"]

    REPORT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "scoped_vs_legacy_sf": sf_ba}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
