# -*- coding: ascii -*-
"""A2 MEASURE FIRST: r_out from qc_metrics.fwhm_px vs header VY_FWHM on era04."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src_py"))

from aperture_policy import resolve_aperture_geometry  # noqa: E402
from config import AppConfig  # noqa: E402

SNAP = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era04_20260826"
QC = SNAP / "calibrated" / "lights" / "qc_metrics.csv"
MS = SNAP / "platesolve" / "NoFilter_60_2" / "MASTERSTAR.fits"
OUT = Path(__file__).resolve().parent


def _r_out(fw: float, cfg: AppConfig) -> float:
    _, _, r_out = resolve_aperture_geometry(
        f=float(cfg.aperture_fwhm_factor),
        fwhm_px=float(fw),
        annulus_inner_fwhm=float(cfg.annulus_inner_fwhm),
        annulus_outer_fwhm=float(cfg.annulus_outer_fwhm),
    )
    return float(r_out)


def main() -> int:
    cfg = AppConfig()
    rows_in = list(csv.DictReader(QC.open(encoding="utf-8")))
    science = []
    for rec in rows_in:
        src = Path(str(rec.get("src") or rec.get("dst") or ""))
        status = str(rec.get("status") or "").strip()
        try:
            qc_fw = float(rec.get("fwhm_px"))
        except (TypeError, ValueError):
            qc_fw = float("nan")
        science.append((src, status, qc_fw))

    ok_rows = [r for r in science if r[1] == "ok"]
    # era04 science product is status==ok (expect 134).
    target = ok_rows if len(ok_rows) == 134 else science

    table = []
    n_diff = 0
    n_missing_hdr = 0
    n_missing_file = 0
    max_abs = 0.0
    for src, status, qc_fw in target:
        if not src.is_file():
            n_missing_file += 1
            table.append(
                {
                    "frame": src.name,
                    "status": status,
                    "qc_fwhm_px": qc_fw,
                    "vy_fwhm": None,
                    "r_out_qc": None,
                    "r_out_hdr": None,
                    "r_out_equal_ulp": False,
                    "note": "missing_fits",
                }
            )
            n_diff += 1
            continue
        with fits.open(src, memmap=True) as hdul:
            raw = hdul[0].header.get("VY_FWHM")
        if isinstance(raw, tuple):
            raw = raw[0]
        try:
            hdr_fw = float(raw)
        except (TypeError, ValueError):
            hdr_fw = float("nan")
        if not math.isfinite(hdr_fw):
            n_missing_hdr += 1
            n_diff += 1
            table.append(
                {
                    "frame": src.name,
                    "status": status,
                    "qc_fwhm_px": qc_fw,
                    "vy_fwhm": None,
                    "r_out_qc": _r_out(qc_fw, cfg) if math.isfinite(qc_fw) else None,
                    "r_out_hdr": None,
                    "r_out_equal_ulp": False,
                    "note": "missing_VY_FWHM",
                }
            )
            continue
        r_qc = _r_out(qc_fw, cfg)
        r_hdr = _r_out(hdr_fw, cfg)
        equal = bool(r_qc == r_hdr and qc_fw == hdr_fw)
        if not equal:
            n_diff += 1
            max_abs = max(max_abs, abs(r_qc - r_hdr), abs(qc_fw - hdr_fw))
        table.append(
            {
                "frame": src.name,
                "status": status,
                "qc_fwhm_px": qc_fw,
                "vy_fwhm": hdr_fw,
                "r_out_qc": r_qc,
                "r_out_hdr": r_hdr,
                "delta_fwhm": float(qc_fw - hdr_fw),
                "delta_r_out": float(r_qc - r_hdr),
                "r_out_equal_ulp": equal,
                "note": "" if equal else "DIFF",
            }
        )

    ms_vy = None
    if MS.is_file():
        with fits.open(MS, memmap=True) as hdul:
            raw = hdul[0].header.get("VY_FWHM")
        if isinstance(raw, tuple):
            raw = raw[0]
        try:
            ms_vy = float(raw)
        except (TypeError, ValueError):
            ms_vy = None
    ok_fwhm = [r[2] for r in ok_rows if math.isfinite(r[2])]
    night_med = float(np.median(np.asarray(ok_fwhm, dtype=np.float64))) if ok_fwhm else None
    bbox_old = None
    bbox_resolver_ms = None
    bbox_resolver_night = None
    if ms_vy is not None and math.isfinite(ms_vy):
        bbox_old = float(cfg.annulus_outer_fwhm) * float(ms_vy)
        bbox_resolver_ms = _r_out(ms_vy, cfg)
    if night_med is not None:
        bbox_resolver_night = _r_out(night_med, cfg)

    summary = {
        "n_qc_rows": len(science),
        "n_ok": len(ok_rows),
        "n_measured": len(target),
        "n_diff": n_diff,
        "n_missing_fits": n_missing_file,
        "n_missing_vy_fwhm": n_missing_hdr,
        "max_abs_delta": max_abs,
        "swap_free": n_diff == 0,
        "cfg_f": float(cfg.aperture_fwhm_factor),
        "cfg_inner": float(cfg.annulus_inner_fwhm),
        "cfg_outer": float(cfg.annulus_outer_fwhm),
        "masterstar_vy_fwhm": ms_vy,
        "night_median_qc_fwhm_ok": night_med,
        "bbox_old_outer_times_ms": bbox_old,
        "bbox_resolver_ms": bbox_resolver_ms,
        "bbox_resolver_night_med": bbox_resolver_night,
        "bbox_ms_vs_night_r_out_equal": (
            bbox_resolver_ms == bbox_resolver_night
            if bbox_resolver_ms is not None and bbox_resolver_night is not None
            else None
        ),
    }
    (OUT / "a2_summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
    with (OUT / "a2_r_out_table.csv").open("w", encoding="ascii", newline="") as f:
        if table:
            w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)
    print(json.dumps(summary, indent=2))
    diffs = [t for t in table if not t.get("r_out_equal_ulp")]
    print(f"n_diff={len(diffs)}")
    if diffs[:8]:
        print("first diffs:", diffs[:8])
    return 0 if n_diff == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
