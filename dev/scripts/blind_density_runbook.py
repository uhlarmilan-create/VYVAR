#!/usr/bin/env python3
"""Closed-loop runbook: density-matched blind index for draft_000380 (Chi_and_H)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
_GAIA = _ROOT / "GAIA_DR3"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_build_mod():
    mod_path = _GAIA / "build_blind_index.py"
    spec = importlib.util.spec_from_file_location("build_blind_index", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _angular_sep(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    dra = (ra1 - ra2) * math.cos(math.radians((dec1 + dec2) / 2.0))
    ddec = dec1 - dec2
    return math.sqrt(dra * dra + ddec * ddec)


def phase0_metrics(
    *,
    draft: Path,
    index_path: Path,
    plate_scale: float,
    truth_ra: float,
    truth_dec: float,
    out_json: Path,
) -> dict[str, Any]:
    """Run diagnose and derive density targets from report + fresh metrics."""
    diag_script = _ROOT / "scripts" / "diagnose_blind_solver_380.py"
    out_dir = draft / "diag" / "blind_solver"
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(diag_script),
            "--draft",
            str(draft),
            "--index",
            str(index_path),
            "--plate-scale",
            str(plate_scale),
            "--truth-ra",
            str(truth_ra),
            "--truth-dec",
            str(truth_dec),
            "--out",
            str(out_dir),
        ],
        check=True,
        cwd=str(_ROOT),
    )

    diag_path = _ROOT / "scripts" / "diagnose_blind_solver_380.py"
    spec_d = importlib.util.spec_from_file_location("diag380", diag_path)
    if spec_d is None or spec_d.loader is None:
        raise RuntimeError(diag_path)
    diag = importlib.util.module_from_spec(spec_d)
    spec_d.loader.exec_module(diag)
    _analyze_image_triangles = diag._analyze_image_triangles
    _deciding_pass = diag._deciding_pass
    _detect_dao_stars = diag._detect_dao_stars
    _discover_masterstars = diag._discover_masterstars
    _fov_deg_from_fits = diag._fov_deg_from_fits
    _load_index_meta = diag._load_index_meta
    from config import AppConfig
    from vyvar_blind_solver import find_blind_candidates, _index_k_neighbors
    from vyvar_platesolver import _verify_blind_candidates

    cfg = AppConfig()
    cfg.debug_platesolver = True
    index_meta = _load_index_meta(index_path)
    tri_k = _index_k_neighbors({"k_neighbors": index_meta.get("k_neighbors")})
    log_L3_min = float(index_meta["log_L3_min"])
    log_L3_max = float(index_meta["log_L3_max"])

    groups: dict[str, Any] = {}
    for group, fits_path in _discover_masterstars(draft):
        from astropy.io import fits

        _, dao_df, n_dao, nx = _detect_dao_stars(fits_path)
        with fits.open(fits_path, memmap=True) as hdul:
            ny = int(hdul[0].data.shape[0])

        fov_deg = _fov_deg_from_fits(nx, ny, plate_scale)
        tri = _analyze_image_triangles(
            dao_df,
            log_L3_min=log_L3_min,
            log_L3_max=log_L3_max,
            plate_scale=plate_scale,
            fov_deg=fov_deg,
            img_budget=int(cfg.blind_img_star_budget),
            tri_k=tri_k,
        )
        n_c = int(tri.get("n_central", 0))
        r_px = float(tri.get("R_px", 1.0))
        rho_px = n_c / (math.pi * r_px * r_px) if r_px > 0 else 0.0
        arcsec_per_deg = 3600.0 / plate_scale
        rho_deg2 = rho_px * arcsec_per_deg * arcsec_per_deg
        rho_budget_deg2 = (
            float(cfg.blind_img_star_budget) / max(fov_deg * fov_deg, 1e-6)
        )

        sink: dict[str, Any] = {}
        best_near_truth: dict[str, Any] | None = None
        hint = None
        if len(dao_df) >= 3:
            cands = find_blind_candidates(
                dao_df,
                index_path,
                plate_scale_arcsec_per_px=plate_scale,
                fov_deg=fov_deg,
                app_config=cfg,
                debug_sink=sink,
                debug_truth_radec=(truth_ra, truth_dec),
            )
            gaia_db = Path(cfg.gaia_db_path).expanduser()
            if gaia_db.is_file() and cands:
                hint = _verify_blind_candidates(
                    cands,
                    dao_df=dao_df,
                    gaia_db_path=gaia_db,
                    fov_deg=fov_deg,
                    naxis1=nx,
                    naxis2=ny,
                    pixel_pitch_um=None,
                    focal_length_mm=None,
                    max_cat_mag=16.0,
                    app_config=cfg,
                    debug_sink=sink,
                )
            verified = sink.get("verified_candidates") or []
            for row in verified:
                cra = float(row.get("field_center_ra", row.get("center_ra", 0)))
                cde = float(row.get("field_center_dec", row.get("center_dec", 0)))
                sep = _angular_sep(cra, cde, truth_ra, truth_dec)
                row["sep_from_truth_deg"] = sep
                if best_near_truth is None or sep < float(
                    best_near_truth.get("sep_from_truth_deg", 999)
                ):
                    best_near_truth = dict(row)

        deciding = _deciding_pass(sink)
        groups[group] = {
            "log_L3_img": {
                "min": tri.get("log_L3_min"),
                "p10": tri.get("log_L3_p10"),
                "med": tri.get("log_L3_med"),
                "max": tri.get("log_L3_max"),
            },
            "n_central": n_c,
            "R_px": r_px,
            "rho_img_px2": rho_px,
            "rho_img_deg2": rho_deg2,
            "rho_budget_deg2": rho_budget_deg2,
            "fov_deg": fov_deg,
            "votes_near_truth_2deg": deciding.get("votes_near_truth_2deg") if deciding else None,
            "votes_near_truth_5deg": deciding.get("votes_near_truth_5deg") if deciding else None,
            "best_near_truth": best_near_truth,
            "verify_hint": {"ra": hint[0], "dec": hint[1]} if hint else None,
            "verify_winner": sink.get("verify_winner"),
        }

    cell_deg = 1.0
    b = groups.get("B_20_2") or next(iter(groups.values()))
    rho_nc = float(b["rho_img_deg2"])
    rho_budget = float(b["rho_budget_deg2"])
    stars_per_cell_spec = int(round(rho_nc * cell_deg * cell_deg))
    stars_per_cell_budget = int(round(rho_budget * cell_deg * cell_deg))
    stars_per_cell_initial = max(80, min(500, stars_per_cell_budget))

    payload = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "draft": str(draft),
        "index": str(index_path),
        "plate_scale_arcsec_px": plate_scale,
        "truth_radec": [truth_ra, truth_dec],
        "index_meta": index_meta,
        "groups": groups,
        "density_target": {
            "cell_deg": cell_deg,
            "stars_per_cell_from_Nc": stars_per_cell_spec,
            "stars_per_cell_from_budget": stars_per_cell_budget,
            "stars_per_cell_initial": stars_per_cell_initial,
        },
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")
    return payload


def phase2_build_loop(
    *,
    db_path: Path,
    out_pkl: Path,
    med_logl3_img: float,
    stars_per_cell_initial: int,
    cell_deg: float = 1.0,
    mag_limit: float = 16.0,
    max_iters: int = 2,
) -> list[dict[str, Any]]:
    mod = _load_build_mod()
    spc = int(stars_per_cell_initial)
    history: list[dict[str, Any]] = []
    for it in range(max_iters + 1):
        print(f"\n=== Build iter {it} STARS_PER_CELL={spc} ===")
        summary = mod.build_and_save(
            db_path=str(db_path),
            output_pkl=str(out_pkl),
            mag_limit=mag_limit,
            cell_deg=cell_deg,
            stars_per_cell=spc,
        )
        l3 = summary["log_L3_index"]
        med_idx = float(l3["med"])
        delta = float(med_logl3_img) - med_idx
        summary["delta_med_logL3"] = delta
        history.append(summary)
        if abs(delta) <= 0.10:
            print(f"Converged |Δ|={abs(delta):.3f} dex <= 0.10")
            break
        if it >= max_iters:
            print(f"Max iterations; |Δ|={abs(delta):.3f}")
            break
        spc = max(20, int(round(spc * (10.0 ** (-2.0 * delta)))))
        print(f"Next STARS_PER_CELL={spc} (10^(-2Δ) with Δ={delta:.3f})")
    return history


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["0", "2", "3", "all"], default="all")
    ap.add_argument("--draft", type=Path, default=_ROOT / "Archive" / "Drafts" / "draft_000380")
    ap.add_argument("--index", type=Path, default=_GAIA / "gaia_triangles.pkl")
    ap.add_argument("--test-index", type=Path, default=_GAIA / "gaia_triangles_test.pkl")
    ap.add_argument("--plate-scale", type=float, default=1.3)
    ap.add_argument("--truth-ra", type=float, default=35.03)
    ap.add_argument("--truth-dec", type=float, default=57.14)
    args = ap.parse_args()

    out_dir = args.draft / "diag" / "blind_solver"
    metrics_path = out_dir / "phase0_metrics.json"

    if args.phase in ("0", "all"):
        m0 = phase0_metrics(
            draft=args.draft,
            index_path=args.index,
            plate_scale=args.plate_scale,
            truth_ra=args.truth_ra,
            truth_dec=args.truth_dec,
            out_json=metrics_path,
        )
    else:
        m0 = json.loads(metrics_path.read_text(encoding="utf-8"))

    if args.phase in ("2", "all"):
        b = m0["groups"].get("B_20_2") or {}
        med_img = float((b.get("log_L3_img") or {}).get("med") or 2.57)
        spc0 = int(m0["density_target"]["stars_per_cell_initial"])
        db = _GAIA / "vyvar_gaia_dr3.db"
        hist = phase2_build_loop(
            db_path=db,
            out_pkl=args.test_index,
            med_logl3_img=med_img,
            stars_per_cell_initial=spc0,
        )
        (out_dir / "phase2_build_history.json").write_text(
            json.dumps(hist, indent=2), encoding="utf-8"
        )

    if args.phase in ("3", "all"):
        idx = args.test_index if args.test_index.is_file() else args.index
        phase0_metrics(
            draft=args.draft,
            index_path=idx,
            plate_scale=args.plate_scale,
            truth_ra=args.truth_ra,
            truth_dec=args.truth_dec,
            out_json=out_dir / "phase3_metrics.json",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
