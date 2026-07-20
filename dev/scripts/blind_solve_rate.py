#!/usr/bin/env python3
"""Blind solve-rate harness: archive battery + series orchestrator metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from photutils.detection import DAOStarFinder

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from vyvar_blind_series import solve_blind_with_series  # noqa: E402


def _sep(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    dra = (ra1 - ra2) * math.cos(math.radians((dec1 + dec2) / 2.0))
    ddec = dec1 - dec2
    return math.sqrt(dra * dra + ddec * ddec)


def _plate_scale(hdr) -> float | None:
    v = hdr.get("VY_PLTS")
    if v is not None:
        try:
            ps = float(v)
            if math.isfinite(ps) and ps > 0:
                return ps
        except (TypeError, ValueError):
            pass
    try:
        w = WCS(hdr)
        return float(abs(w.pixel_scale_matrix).mean() * 3600.0)
    except Exception:  # noqa: BLE001
        return None


def _dao_df(fits_path: Path):
    with fits.open(fits_path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        hdr = hdul[0].header
        ny, nx = data.shape
    _, med, std = sigma_clipped_stats(data, sigma=3.0)
    finder = DAOStarFinder(fwhm=3.0, threshold=5.0 * std, min_separation=0)
    srcs = finder(data - med)
    if srcs is None or len(srcs) < 3:
        return None, hdr, int(nx), int(ny)
    import pandas as pd

    df = srcs.to_pandas().rename(columns={"x_centroid": "x", "y_centroid": "y"})
    if "peak" in df.columns and "flux" not in df.columns:
        df["flux"] = df["peak"]
    elif "flux" not in df.columns:
        df["flux"] = 1.0
    return df.sort_values("flux", ascending=False), hdr, int(nx), int(ny)


def _truth_near_row(rows: list[dict], truth_ra: float, truth_dec: float) -> dict | None:
    best: dict | None = None
    best_sep = float("inf")
    for r in rows:
        try:
            fra = float(r.get("field_center_ra"))
            fde = float(r.get("field_center_dec"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(fra) and math.isfinite(fde)):
            continue
        s = _sep(fra, fde, truth_ra, truth_dec)
        if s < best_sep:
            best_sep = s
            best = r
    return best


def _run_field(
    *,
    archive: Path,
    field: dict,
    cfg: AppConfig,
    hit_threshold_deg: float,
) -> dict:
    draft = archive / "Drafts" / str(field["draft"])
    ms = draft / str(field["masterstar"])
    if not ms.is_file():
        return {
            "id": field.get("id", ms.name),
            "status": "SKIP",
            "reason": f"missing {ms}",
        }
    dao_df, hdr, nx, ny = _dao_df(ms)
    if dao_df is None:
        return {"id": field.get("id"), "status": "SKIP", "reason": "too few DAO stars"}

    ps = field.get("plate_scale")
    if ps is None:
        ps = _plate_scale(hdr)
    else:
        ps = float(ps)
    fov = max(nx, ny) * float(ps) / 3600.0
    truth_ra = float(field["truth_ra"])
    truth_dec = float(field["truth_dec"])

    t0 = time.time()
    sink: dict = {}
    out = solve_blind_with_series(
        dao_df,
        app_config=cfg,
        plate_scale_arcsec_per_px=ps,
        fov_deg=fov,
        gaia_db_path=cfg.gaia_db_path,
        naxis1=nx,
        naxis2=ny,
        max_cat_mag=16.0,
        debug_sink=sink,
    )
    elapsed = time.time() - t0
    if out is None:
        return {
            "id": field.get("id"),
            "draft": field.get("draft"),
            "rig": field.get("rig"),
            "status": "MISS",
            "sep_deg": "",
            "tier": "",
            "elapsed_s": round(elapsed, 2),
        }
    ra, dec, tier = out
    sep = _sep(ra, dec, truth_ra, truth_dec)
    hit = sep <= hit_threshold_deg
    vm = sink.get("verify_metrics") or {}
    vw = sink.get("verify_winner") or {}
    vc = sink.get("verified_candidates") or []
    truth_row = _truth_near_row(vc, truth_ra, truth_dec)
    return {
        "id": field.get("id"),
        "draft": field.get("draft"),
        "rig": field.get("rig"),
        "status": "HIT" if hit else "MISS",
        "sep_deg": round(sep, 4),
        "tier": tier,
        "ra": round(ra, 4),
        "dec": round(dec, 4),
        "truth_ra": truth_ra,
        "truth_dec": truth_dec,
        "plate_scale": ps,
        "total_s": round(elapsed, 2),
        "elapsed_s": round(elapsed, 2),
        "verify_mag_limit": vm.get("verify_mag_limit"),
        "catalog_load_s": vm.get("catalog_load_s"),
        "verify_s": vm.get("verify_s"),
        "cone_n_cat": vw.get("cone_n_cat"),
        "n_matched": vw.get("n_matched"),
        "winner_fraction": vw.get("fraction"),
        "truth_near_sep_deg": round(
            _sep(
                float(truth_row["field_center_ra"]),
                float(truth_row["field_center_dec"]),
                truth_ra,
                truth_dec,
            ),
            4,
        )
        if truth_row
        else "",
        "truth_near_n_matched": truth_row.get("n_matched") if truth_row else "",
        "max_false_n_matched": vm.get("max_false_n_matched"),
        "early_exit_fired": vm.get("early_exit_fired"),
        "early_exit_cand_idx": vm.get("early_exit_cand_idx"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--battery",
        type=Path,
        default=_ROOT / "validation" / "blind_solve_battery.json",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=_ROOT / "validation" / "blind_solve_rate.csv",
    )
    ap.add_argument("--mode", choices=["auto", "series_all", "single"], default=None)
    ap.add_argument("--min-hit-rate", type=float, default=0.9)
    ap.add_argument(
        "--verify-mag-limit",
        type=float,
        default=None,
        help="Override config verify_mag_limit for in-memory catalog depth (A/B).",
    )
    ap.add_argument(
        "--fields",
        type=str,
        default=None,
        help="Comma-separated field ids to run (default: all in battery).",
    )
    args = ap.parse_args()

    battery = json.loads(args.battery.read_text(encoding="utf-8"))
    fields = battery.get("fields", [])
    hit_thr = float(battery.get("hit_threshold_deg", 2.0))
    archive = Path(AppConfig().archive_root).expanduser()

    cfg = AppConfig()
    if args.mode:
        cfg.blind_index_select_mode = args.mode
    if args.verify_mag_limit is not None:
        cfg.verify_mag_limit = float(args.verify_mag_limit)

    if args.fields:
        wanted = {x.strip() for x in args.fields.split(",") if x.strip()}
        fields = [f for f in fields if str(f.get("id")) in wanted]

    rows = []
    for field in fields:
        row = _run_field(archive=archive, field=field, cfg=cfg, hit_threshold_deg=hit_thr)
        rows.append(row)
        print(
            f"{row.get('id')}: {row.get('status')} "
            f"sep={row.get('sep_deg', 'n/a')} tier={row.get('tier', '')}"
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    tested = [r for r in rows if r.get("status") in ("HIT", "MISS")]
    n_hit = sum(1 for r in tested if r.get("status") == "HIT")
    rate = n_hit / len(tested) if tested else 0.0
    seps = [float(r["sep_deg"]) for r in tested if r.get("sep_deg") not in ("", None)]
    summary = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "n_fields": len(fields),
        "n_tested": len(tested),
        "n_hit": n_hit,
        "hit_rate": round(rate, 4),
        "median_sep_deg": float(np.median(seps)) if seps else None,
        "p90_sep_deg": float(np.percentile(seps, 90)) if seps else None,
        "mode": cfg.blind_index_select_mode,
        "verify_mag_limit": float(cfg.verify_mag_limit),
        "csv": str(args.out_csv),
    }
    print(json.dumps(summary, indent=2))
    (args.out_csv.parent / "blind_solve_rate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0 if rate >= args.min_hit_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
