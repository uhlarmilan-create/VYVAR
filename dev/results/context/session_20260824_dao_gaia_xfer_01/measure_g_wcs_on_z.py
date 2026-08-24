#!/usr/bin/env python3
"""XFER-01 W6 cheap measure: project 520 g VERIFIED WCS onto z MASTERSTAR.

Read-only. Does not write WCS, does not lower recovery_min, does not wire.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUT = Path(__file__).resolve().parent
DRAFT = REPO / "Archive" / "Drafts" / "draft_000520"
G_MS = DRAFT / "platesolve" / "g_60_4" / "MASTERSTAR.fits"
Z_MS = DRAFT / "platesolve" / "z_90_4" / "MASTERSTAR.fits"
I_MS = DRAFT / "platesolve" / "i_70_4" / "MASTERSTAR.fits"


def _adu_contrast(data: np.ndarray) -> dict:
    a = np.asarray(data, dtype=np.float64)
    a = a[np.isfinite(a)]
    med = float(np.median(a))
    p99 = float(np.percentile(a, 99))
    return {
        "median_adu": med,
        "p99_adu": p99,
        "p99_minus_median_adu": p99 - med,
        "n_pix": int(a.size),
    }


def _hdr_facts(hdr) -> dict:
    return {
        "date_obs": str(hdr.get("DATE-OBS") or hdr.get("DATE") or ""),
        "filter": str(hdr.get("FILTER") or ""),
        "naxis1": int(hdr.get("NAXIS1") or 0),
        "naxis2": int(hdr.get("NAXIS2") or 0),
        "vy_fwhm": hdr.get("VY_FWHM"),
    }


def _project(donor_wcs: WCS, recipient_path: Path, *, donor_name: str) -> dict:
    from config import AppConfig
    from pipeline import _plate_solve_input_bundle
    from vyvar_platesolver import (
        _compute_masterstar_catalog_recovery,
        _sibling_adopt_and_confirm,
        _sibling_cfg_thresholds,
        _sibling_detect_dao_on_image,
        _sibling_load_gaia_catalog,
        plate_scale_arcsec_per_px_from_wcs,
    )

    cfg = AppConfig()
    bundle = _plate_solve_input_bundle(
        recipient_path,
        app_config=cfg,
        equipment_id=4,
        draft_id=520,
    )
    with fits.open(recipient_path, memmap=False) as hd:
        hdr = hd[0].header.copy()
        data = np.asarray(hd[0].data, dtype=np.float32)
        naxis1 = int(hdr.get("NAXIS1", data.shape[1]))
        naxis2 = int(hdr.get("NAXIS2", data.shape[0]))
    dao_sigma = float(getattr(cfg, "masterstar_dao_threshold_sigma", 2.1) or 2.1)
    xs, ys = _sibling_detect_dao_on_image(data, hdr, dao_sigma=dao_sigma)
    scale = bundle.get("expected_arcsec_per_px")
    if scale is None:
        scale = plate_scale_arcsec_per_px_from_wcs(donor_wcs)
    ra_cat, de_cat = _sibling_load_gaia_catalog(
        donor_wcs,
        hdr,
        naxis1,
        naxis2,
        gaia_db_path=Path(str(cfg.gaia_db_path)),
        fov_diameter_deg=float(cfg.plate_solve_fov_deg),
        expected_plate_scale_arcsec_per_px=scale,
        effective_pixel_um=bundle.get("eff_um"),
        focal_length_mm=bundle.get("focal_mm"),
        app_config=cfg,
    )
    rec_raw = _compute_masterstar_catalog_recovery(
        donor_wcs,
        ra_cat,
        de_cat,
        xs,
        ys,
        naxis1=naxis1,
        naxis2=naxis2,
        qa_px=18.3,
        tight_px=2.5,
    )
    recovery_min = float(getattr(cfg, "masterstar_catalog_recovery_min", 0.65))
    gate = float(rec_raw.get("catalog_recovery_tight_gate") or 0.0)
    thresholds = _sibling_cfg_thresholds(cfg, arcsec_per_px=scale)
    adopt = _sibling_adopt_and_confirm(
        donor_wcs,
        ra_cat,
        de_cat,
        xs,
        ys,
        naxis1,
        naxis2,
        thresholds=thresholds,
        cat_pred_flip=None,
    )
    after = adopt.get("after") or {}
    return {
        "donor": donor_name,
        "n_det": int(len(xs)),
        "raw_g_wcs_on_z": {
            "n_cat_in_frame": rec_raw.get("n_cat_in_frame"),
            "n_matched_tight": rec_raw.get("n_matched_tight"),
            "n_detections_used": rec_raw.get("n_detections_used"),
            "catalog_recovery_tight": rec_raw.get("catalog_recovery_tight"),
            "catalog_recovery_tight_gate": rec_raw.get("catalog_recovery_tight_gate"),
            "would_pass_recovery_min": gate >= recovery_min,
            "recovery_min": recovery_min,
        },
        "in_memory_bulk_shift_no_write": {
            "confirmed": bool(adopt.get("confirmed")),
            "n_matched_tight": after.get("n_matched_tight"),
            "catalog_recovery_tight_gate": after.get("catalog_recovery_tight_gate"),
            "rms_px": after.get("rms_px"),
            "median_dpx": after.get("median_dpx"),
        },
    }


def main() -> int:
    out: dict = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "note": "read-only; no FITS write; recovery_min not changed",
        "paths": {"g": str(G_MS), "z": str(Z_MS), "i": str(I_MS)},
    }
    with fits.open(G_MS, memmap=False) as hd:
        out["g_header"] = _hdr_facts(hd[0].header)
        out["g_contrast"] = _adu_contrast(hd[0].data)
        g_wcs = WCS(hd[0].header)
    with fits.open(Z_MS, memmap=False) as hd:
        out["z_header"] = _hdr_facts(hd[0].header)
        out["z_contrast"] = _adu_contrast(hd[0].data)
    with fits.open(I_MS, memmap=False) as hd:
        out["i_header"] = _hdr_facts(hd[0].header)
        i_wcs = WCS(hd[0].header)

    out["g_on_z"] = _project(g_wcs, Z_MS, donor_name="g_60_4")
    out["i_on_z"] = _project(i_wcs, Z_MS, donor_name="i_70_4")
    gate = float(out["g_on_z"]["raw_g_wcs_on_z"]["catalog_recovery_tight_gate"] or 0)
    if gate >= 0.65:
        reading = (
            "decent recovery: blind solve was the failure; sibling seed can rescue z"
        )
    elif gate <= 0.20:
        reading = (
            "low recovery (~shallow): z is physically shallow; reject stands "
            "with a measured reason"
        )
    else:
        reading = "intermediate: neither a clear rescue nor the ~9% shallow floor"
    out["reading"] = reading
    dest = OUT / "w6_g_wcs_on_z.json"
    dest.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="ascii")
    print(json.dumps({"ok": True, "out": str(dest), "g_on_z": out["g_on_z"], "reading": reading}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
