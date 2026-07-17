#!/usr/bin/env python3
"""Wide-rig blind diagnostic (draft_365 + wide tier)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from scripts.diagnose_blind_solver_380 import (  # noqa: E402
    _analyze_image_triangles,
    _detect_dao_stars,
    _fov_deg_from_fits,
    _load_index_meta,
)
from vyvar_blind_solver import (  # noqa: E402
    _index_k_neighbors,
    _side_arcsec_flat,
    _side_arcsec_gnomonic,
    find_blind_candidates,
    find_blind_hint,
)
from vyvar_platesolver import _verify_blind_candidates  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--masterstar",
        type=Path,
        default=_ROOT
        / "Archive/Drafts/draft_000365/platesolve/NoFilter_60_2/MASTERSTAR.fits",
    )
    ap.add_argument(
        "--index",
        type=Path,
        default=_ROOT / "GAIA_DR3/gaia_triangles_wide.pkl",
    )
    ap.add_argument("--truth-ra", type=float, default=241.53869)
    ap.add_argument("--truth-dec", type=float, default=50.29571)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = AppConfig()
    cfg.debug_platesolver = True
    index_meta = _load_index_meta(args.index)
    tri_k = _index_k_neighbors({"k_neighbors": index_meta.get("k_neighbors")})

    _, dao_df, n_dao, nx = _detect_dao_stars(args.masterstar)
    with __import__("astropy.io", fromlist=["fits"]).fits.open(args.masterstar, memmap=True) as hdul:
        hdr = hdul[0].header
        ny = int(hdul[0].data.shape[0])
    ps = float(hdr.get("VY_PLTS", 9.55))
    fov = _fov_deg_from_fits(nx, ny, ps)

    tri = _analyze_image_triangles(
        dao_df,
        log_L3_min=float(index_meta["log_L3_min"]),
        log_L3_max=float(index_meta["log_L3_max"]),
        plate_scale=ps,
        fov_deg=fov,
        img_budget=int(cfg.blind_img_star_budget),
        tri_k=tri_k,
    )

    sink: dict = {}
    cands = find_blind_candidates(
        dao_df,
        args.index,
        plate_scale_arcsec_per_px=ps,
        fov_deg=fov,
        app_config=cfg,
        debug_truth_radec=(args.truth_ra, args.truth_dec),
        debug_sink=sink,
    )
    verify = _verify_blind_candidates(
        cands or [],
        dao_df=dao_df,
        gaia_db_path=cfg.gaia_db_path,
        fov_deg=fov,
        naxis1=nx,
        naxis2=ny,
        pixel_pitch_um=None,
        focal_length_mm=None,
        max_cat_mag=16.0,
        known_plate_scale_arcsec_per_px=ps,
        app_config=cfg,
        debug_sink=sink,
    )
    vote = find_blind_hint(
        dao_df,
        args.index,
        plate_scale_arcsec_per_px=ps,
        fov_deg=fov,
        app_config=cfg,
        debug_truth_radec=(args.truth_ra, args.truth_dec),
    )

    # Flat vs gnomonic edge sample
    x_c, y_c = float(dao_df["x"].max()) / 2, float(dao_df["y"].max()) / 2
    edge = dao_df.nlargest(1, "flux").iloc[0]
    p0 = np.array([x_c, y_c])
    p1 = np.array([float(edge["x"]), float(edge["y"])])
    flat_s = _side_arcsec_flat(p0, p1, plate_scale_arcsec_per_px=ps)
    gno_s = _side_arcsec_gnomonic(p0, p1, x_cen=x_c, y_cen=y_c, plate_scale_arcsec_per_px=ps)
    rel_edge = abs(flat_s - gno_s) / max(flat_s, 1e-6)

    def _sep(ra: float, dec: float) -> float:
        dra = (ra - args.truth_ra) * math.cos(math.radians((dec + args.truth_dec) / 2))
        ddec = dec - args.truth_dec
        return math.sqrt(dra * dra + ddec * ddec)

    nearest_sep = None
    if cands:
        nearest_sep = min(_sep(c.center_ra, c.center_dec) for c in cands)

    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "masterstar": str(args.masterstar),
        "index": str(args.index),
        "truth": [args.truth_ra, args.truth_dec],
        "plate_scale_arcsec_px": ps,
        "fov_deg": fov,
        "n_dao": n_dao,
        "log_L3_img": tri.get("log_L3_med"),
        "log_L3_index_med": (index_meta["log_L3_min"] + index_meta["log_L3_max"]) / 2,
        "votes_near_truth_2deg": sink.get("votes_near_truth_2deg"),
        "votes_near_truth_5deg": sink.get("votes_near_truth_5deg"),
        "nearest_vote_sep_deg": nearest_sep,
        "verify_hint": list(verify) if verify else None,
        "verify_sep_deg": _sep(verify[0], verify[1]) if verify else None,
        "vote_hint_sep_deg": _sep(vote[0], vote[1]) if vote else None,
        "edge_flat_vs_gnomonic_rel": rel_edge,
        "n_candidates": len(cands or []),
    }

    out = args.out or (
        _ROOT / "Archive/Drafts/draft_000365/diag/blind_solver/wide_diagnostic_report.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    hit = verify and report["verify_sep_deg"] is not None and report["verify_sep_deg"] <= 2.0
    return 0 if hit else 1


if __name__ == "__main__":
    raise SystemExit(main())
