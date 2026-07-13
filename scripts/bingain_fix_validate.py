#!/usr/bin/env python3
"""F-BINGAIN-1 FIX validation: check-star chi2 before/after empirical err."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from check_star_kmag import resolve_proc_csv_dir  # noqa: E402
from config import AppConfig  # noqa: E402
from photometry_core import ERR_BKG_MODE_EMPIRICAL, ERR_BKG_MODE_HOWELL  # noqa: E402
from scripts.chi2_sigma_gate import reduced_chi2_constant  # noqa: E402
from scripts.provenance_guard import (  # noqa: E402
    add_allow_unstamped_arg,
    assert_stamped,
    stamp_output_meta,
)
from scripts.sparse_comp_diag import SS_CAM_CID, V0611_CID, _check_star_chi2_rows  # noqa: E402
from sigma_budget import (  # noqa: E402
    SIGMA_VARIANT_HOWELL_SCINT_FRESID_FLOOR_ENSEMBLE,
    resolve_rig_scintillation_params,
)

PRODUCTION_CHI2_VARIANT = SIGMA_VARIANT_HOWELL_SCINT_FRESID_FLOOR_ENSEMBLE
_MAG_ERR_SCALE = 2.5 / math.log(10)


def resolve_archive_root(explicit: str | None = None, *, cfg: AppConfig | None = None) -> Path:
    """Resolve Archive root; fail loudly when Drafts/ is missing or empty."""
    if explicit is not None and str(explicit).strip():
        root = Path(str(explicit)).expanduser().resolve()
    else:
        _cfg = cfg or AppConfig()
        root = Path(_cfg.archive_root).expanduser().resolve()
    drafts = root / "Drafts"
    if not drafts.is_dir():
        raise SystemExit(
            f"ERROR: Drafts directory not found under archive root: {drafts}\n"
            f"  archive_root={root}\n"
            f"  Pass --archive-root explicitly (expected e.g. C:\\ASTRO\\python\\VYVAR\\Archive)."
        )
    draft_dirs = sorted(drafts.glob("draft_*"))
    if not draft_dirs:
        raise SystemExit(
            f"ERROR: archive root has no draft_* folders: {drafts}\n"
            f"  archive_root={root}\n"
            f"  Previous empty finding was likely a wrong hardcoded path (e.g. C:\\ASTRO\\Archive "
            f"instead of config archive_root under the repo)."
        )
    return root


def _chi2_lc_err(
    *,
    lc_path: Path,
    side_path: Path,
) -> tuple[float | None, dict[str, Any]]:
    """Production acceptance chi2: check_kmag vs LC ``err`` (includes empirical bkg when present)."""
    meta: dict[str, Any] = {
        "lc_path": str(lc_path),
        "side_path": str(side_path),
        "lc_exists": lc_path.is_file(),
        "side_exists": side_path.is_file(),
    }
    if not lc_path.is_file() or not side_path.is_file():
        return None, meta
    lc_df = pd.read_csv(lc_path, low_memory=False)
    side_df = pd.read_csv(side_path, low_memory=False)
    mags = pd.to_numeric(side_df.get("kmag"), errors="coerce").to_numpy(dtype=np.float64)
    n = min(int(mags.size), len(lc_df))
    if n < 3:
        return None, meta
    err_rel = pd.to_numeric(lc_df.get("err"), errors="coerce").iloc[:n].to_numpy(dtype=np.float64)
    sig_mag = _MAG_ERR_SCALE * err_rel
    ok = np.isfinite(mags) & np.isfinite(sig_mag) & (sig_mag > 0)
    if int(np.sum(ok)) < 3:
        return None, meta
    _, _, chi2_dof, mag_ref = reduced_chi2_constant(mags[ok], sig_mag[ok])
    meta["n_frames"] = int(np.sum(ok))
    meta["mag_ref"] = float(mag_ref) if math.isfinite(float(mag_ref)) else None
    if math.isfinite(float(chi2_dof)):
        return float(chi2_dof), meta
    return None, meta


def _chi2_production(
    *,
    archive_root: Path,
    draft_id: int,
    setup: str,
    target_cid: str,
    lc_dir: Path | None = None,
    cfg: AppConfig,
) -> tuple[float | None, dict[str, Any]]:
    phot_dir = archive_root / "Drafts" / f"draft_{draft_id:06d}" / "platesolve" / setup / "photometry"
    _lc_dir = lc_dir if lc_dir is not None else (phot_dir / "lightcurves")
    lc_path = _lc_dir / f"lightcurve_{target_cid}.csv"
    side = _lc_dir / f"check_kmag_{target_cid}.csv"
    proc_dir = resolve_proc_csv_dir(phot_dir, setup)
    meta: dict[str, Any] = {
        "phot_dir": str(phot_dir),
        "lc_dir": str(_lc_dir),
        "proc_dir": str(proc_dir) if proc_dir else None,
        "lc_exists": lc_path.is_file(),
        "side_exists": side.is_file(),
    }
    if not side.is_file() or proc_dir is None or not lc_path.is_file():
        return None, meta
    lc_df = pd.read_csv(lc_path, low_memory=False)
    side_df = pd.read_csv(side, low_memory=False)
    meta_path = phot_dir / "pipeline_meta.json"
    meta_json: dict[str, Any] = {}
    if meta_path.is_file():
        meta_json = json.loads(meta_path.read_text(encoding="utf-8"))
    rig = resolve_rig_scintillation_params(draft_id=draft_id, setup=setup, cfg=cfg, pipeline_meta=meta_json)
    rows, _ = _check_star_chi2_rows(
        phot_dir=phot_dir,
        setup=setup,
        target_cid=target_cid,
        lc_df=lc_df,
        side_df=side_df,
        proc_dir=proc_dir,
        rig=rig,
        cfg=cfg,
    )
    for r in rows:
        if str(r.get("variant", "")) == PRODUCTION_CHI2_VARIANT:
            v = r.get("chi2_dof", r.get("chi2_reduced"))
            if v is not None and math.isfinite(float(v)):
                return float(v), meta
    return None, meta


def _pooled_check_star_chi2(
    *,
    archive_root: Path,
    draft_id: int,
    setup: str,
    lc_dir: Path | None,
    cfg: AppConfig,
    max_checks: int = 40,
    use_lc_err: bool = True,
) -> dict[str, Any]:
    phot_dir = archive_root / "Drafts" / f"draft_{draft_id:06d}" / "platesolve" / setup / "photometry"
    _lc_dir = lc_dir if lc_dir is not None else (phot_dir / "lightcurves")
    if not _lc_dir.is_dir():
        return {"n": 0, "median": None, "p25": None, "p75": None, "values": []}
    checks = sorted(_lc_dir.glob("check_kmag_*.csv"))
    values: list[float] = []
    for side_path in checks[: max(1, int(max_checks))]:
        cid = side_path.stem.replace("check_kmag_", "", 1)
        lc_path = _lc_dir / f"lightcurve_{cid}.csv"
        if not lc_path.is_file():
            continue
        if use_lc_err:
            v, _ = _chi2_lc_err(lc_path=lc_path, side_path=side_path)
            if v is not None and math.isfinite(float(v)):
                values.append(float(v))
            continue
        lc_df = pd.read_csv(lc_path, low_memory=False)
        side_df = pd.read_csv(side_path, low_memory=False)
        proc_dir = resolve_proc_csv_dir(phot_dir, setup)
        if proc_dir is None:
            continue
        meta_path = phot_dir / "pipeline_meta.json"
        meta_json: dict[str, Any] = {}
        if meta_path.is_file():
            meta_json = json.loads(meta_path.read_text(encoding="utf-8"))
        rig = resolve_rig_scintillation_params(draft_id=draft_id, setup=setup, cfg=cfg, pipeline_meta=meta_json)
        rows, _ = _check_star_chi2_rows(
            phot_dir=phot_dir,
            setup=setup,
            target_cid=cid,
            lc_df=lc_df,
            side_df=side_df,
            proc_dir=proc_dir,
            rig=rig,
            cfg=cfg,
        )
        for r in rows:
            if str(r.get("variant", "")) == PRODUCTION_CHI2_VARIANT:
                v = r.get("chi2_dof", r.get("chi2_reduced"))
                if v is not None and math.isfinite(float(v)):
                    values.append(float(v))
                break
    if not values:
        return {"n": 0, "median": None, "p25": None, "p75": None, "values": []}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
        "values": values,
    }


def _err_ratio_stats(
    *,
    archive_root: Path,
    draft_id: int,
    setup: str,
    lc_howell: Path,
    lc_emp: Path,
    target_cids: list[str],
) -> dict[str, Any]:
    ratios: list[float] = []
    for cid in target_cids:
        p0 = lc_howell / f"lightcurve_{cid}.csv"
        p1 = lc_emp / f"lightcurve_{cid}.csv"
        if not p0.is_file() or not p1.is_file():
            continue
        df0 = pd.read_csv(p0, usecols=["source_file", "err"], low_memory=False)
        df1 = pd.read_csv(p1, usecols=["source_file", "err"], low_memory=False)
        m = df0.merge(df1, on="source_file", suffixes=("_howell", "_emp"))
        e0 = pd.to_numeric(m["err_howell"], errors="coerce")
        e1 = pd.to_numeric(m["err_emp"], errors="coerce")
        ok = e0.notna() & e1.notna() & (e0 > 0) & (e1 > 0)
        if ok.any():
            ratios.extend((e1[ok] / e0[ok]).tolist())
    if not ratios:
        return {"n": 0, "median": None, "p25": None, "p75": None}
    arr = np.asarray(ratios, dtype=np.float64)
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--archive-root",
        type=str,
        default=None,
        help="Archive root (must contain Drafts/draft_*). Default: AppConfig.archive_root.",
    )
    ap.add_argument("--drafts", type=int, nargs="+", default=[424, 425, 426])
    ap.add_argument(
        "--setups",
        nargs="+",
        default=None,
        help="Setup names; default auto per draft.",
    )
    ap.add_argument(
        "--after-lc-root",
        type=Path,
        default=None,
        help="Root with tmp/bingain_acceptance/<draft>/<setup>/photometry/lightcurves outputs.",
    )
    ap.add_argument("--out", type=Path, default=Path("tmp/bingain_fix/validation_report.json"))
    add_allow_unstamped_arg(ap)
    args = ap.parse_args()

    cfg = AppConfig()
    archive_root = resolve_archive_root(args.archive_root, cfg=cfg)
    cfg.archive_root = archive_root

    default_setups: dict[int, list[str]] = {
        424: ["NoFilter_60_2"],
        425: ["B_20_2", "V_20_2", "R_20_2"],
        426: ["g_60_4", "i_70_4", "r_60_4", "z_90_4"],
    }

    report: dict[str, Any] = {
        "archive_root": str(archive_root),
        "config_archive_root": str(Path(cfg.archive_root).resolve()),
        "drafts": {},
    }
    guard_flags: list[dict[str, Any]] = []

    for did in args.drafts:
        setups = args.setups or default_setups.get(did, [])
        report["drafts"][str(did)] = {}
        for setup in setups:
            phot_dir = archive_root / "Drafts" / f"draft_{did:06d}" / "platesolve" / setup / "photometry"
            guard = assert_stamped(
                phot_dir, draft_id=did, setup=setup, allow_unstamped=args.allow_unstamped,
            )
            guard_flags.append({"draft_id": did, "setup": setup, **guard})
            after_lc = None
            if args.after_lc_root is not None:
                after_lc = (
                    Path(args.after_lc_root) / f"draft_{did:06d}" / setup / "photometry" / "lightcurves"
                )
            entry: dict[str, Any] = {"setup": setup}
            for label, cid in [("V0611", V0611_CID), ("SS_CAM", SS_CAM_CID)]:
                phot_dir = archive_root / "Drafts" / f"draft_{did:06d}" / "platesolve" / setup / "photometry"
                lc_before = phot_dir / "lightcurves" / f"lightcurve_{cid}.csv"
                side_before = phot_dir / "lightcurves" / f"check_kmag_{cid}.csv"
                chi2_lc_before, meta_lc_before = _chi2_lc_err(lc_path=lc_before, side_path=side_before)
                chi2_sb_before, meta_sb_before = _chi2_production(
                    archive_root=archive_root,
                    draft_id=did,
                    setup=setup,
                    target_cid=cid,
                    lc_dir=None,
                    cfg=cfg,
                )
                chi2_lc_after = chi2_sb_after = None
                meta_lc_after: dict[str, Any] = {}
                if after_lc is not None and after_lc.is_dir():
                    chi2_lc_after, meta_lc_after = _chi2_lc_err(
                        lc_path=after_lc / f"lightcurve_{cid}.csv",
                        side_path=after_lc / f"check_kmag_{cid}.csv",
                    )
                    chi2_sb_after, _ = _chi2_production(
                        archive_root=archive_root,
                        draft_id=did,
                        setup=setup,
                        target_cid=cid,
                        lc_dir=after_lc,
                        cfg=cfg,
                    )
                entry[label] = {
                    "target_cid": cid,
                    "chi2_before": chi2_lc_before,
                    "chi2_after": chi2_lc_after,
                    "chi2_sigma_budget_before": chi2_sb_before,
                    "chi2_sigma_budget_after": chi2_sb_after,
                    "meta_before": meta_lc_before,
                    "meta_sigma_budget_before": meta_sb_before,
                    "meta_after": meta_lc_after,
                }
            entry["pooled_before"] = _pooled_check_star_chi2(
                archive_root=archive_root, draft_id=did, setup=setup, lc_dir=None, cfg=cfg
            )
            if after_lc is not None and after_lc.is_dir():
                entry["pooled_after"] = _pooled_check_star_chi2(
                    archive_root=archive_root, draft_id=did, setup=setup, lc_dir=after_lc, cfg=cfg
                )
            report["drafts"][str(did)][setup] = entry

    stamped_all = all(g.get("stamped") for g in guard_flags) if guard_flags else True
    report = stamp_output_meta(
        report,
        {
            "stamped": stamped_all,
            "provenance_unstamped": any(g.get("provenance_unstamped") for g in guard_flags),
            "git_hash": None,
        },
    )
    report["provenance_guard_details"] = guard_flags
    if any(g.get("provenance_unstamped") for g in guard_flags):
        print("WARNING: PROVENANCE-GUARD --allow-unstamped: one or more setups lack provenance block.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
