#!/usr/bin/env python3
"""EPSF-NEWTON-518-01 N0 inventory (read-only on 518; no 516 writes)."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

D518 = REPO / "Archive" / "Drafts" / "draft_000518"
D516 = REPO / "Archive" / "Drafts" / "draft_000516"
PS518 = D518 / "platesolve" / "V_60_2"
PHOT518 = PS518 / "photometry"
LC518 = PHOT518 / "lightcurves"
AL518 = D518 / "detrended_aligned" / "lights" / "V_60_2"
RAW518 = D518 / "non_calibrated" / "lights" / "V_60_2"
PS516 = D516 / "platesolve" / "NoFilter_60_2"
OUT = REPO / "dev" / "results" / "session_20260824_epsf_newton_518_01"
PROD_EPSF_SHA = "172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO)).replace("\\", "/")


def header_facts(fits_path: Path) -> dict:
    with fits.open(fits_path, memmap=False) as hdul:
        hdr = hdul[0].header
        naxis = int(hdr.get("NAXIS", 0) or 0)
        shape = None
        if naxis >= 2:
            shape = [int(hdr.get("NAXIS2", 0) or 0), int(hdr.get("NAXIS1", 0) or 0)]
        keys = [
            "TELESCOP",
            "INSTRUME",
            "CAMERA",
            "DETECTOR",
            "OBSERVAT",
            "FILTER",
            "EXPTIME",
            "XBINNING",
            "YBINNING",
            "XPIXSZ",
            "YPIXSZ",
            "PIXSCALE",
            "SECPIX",
            "CDELT1",
            "CDELT2",
            "EGAIN",
            "GAIN",
            "SATURATE",
            "PEDESTAL",
            "NAXIS1",
            "NAXIS2",
            "BITPIX",
            "BUNIT",
            "OBJECT",
            "IMAGETYP",
        ]
        out = {"path": _rel(fits_path), "shape": shape}
        for k in keys:
            if k in hdr:
                v = hdr[k]
                if isinstance(v, (bytes,)):
                    v = v.decode("ascii", "replace")
                out[k] = v
        # plate scale from CD/CDELT if present
        try:
            cdelt1 = abs(float(hdr.get("CDELT1") or 0.0))
            if cdelt1 > 0:
                out["cdelt1_arcsec"] = cdelt1 * 3600.0
        except (TypeError, ValueError):
            pass
        try:
            cd1_1 = hdr.get("CD1_1")
            cd1_2 = hdr.get("CD1_2")
            if cd1_1 is not None:
                a = abs(float(cd1_1))
                b = abs(float(cd1_2 or 0.0))
                out["cd_arcsec_per_px"] = math.hypot(a, b) * 3600.0
        except (TypeError, ValueError):
            pass
        return out


def snapshot_516() -> dict:
    files = []
    epsf = PS516 / "masterstar_epsf.fits"
    meta = PS516 / "masterstar_epsf_meta.json"
    files.extend([p for p in (epsf, meta) if p.is_file()])
    lc_dir = PS516 / "photometry" / "lightcurves"
    if lc_dir.is_dir():
        files.extend(sorted(lc_dir.glob("lightcurve_*.csv")))
        files.extend(sorted(lc_dir.glob("lightcurve_*_psf.csv")))
    aavso = PS516 / "photometry" / "lightcurves_reports" / "aavso"
    varastro = PS516 / "photometry" / "lightcurves_reports" / "varastro"
    if aavso.is_dir():
        files.extend(sorted(aavso.glob("*.txt")))
    if varastro.is_dir():
        files.extend(sorted(varastro.glob("*.txt")))
    out = {_rel(p): _sha(p) for p in files if p.is_file()}
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    inv: dict = {"draft": 518, "paths": {"archive": str(D518)}}

    man = json.loads((D518 / "draft_manifest.json").read_text(encoding="utf-8"))
    fwhm = []
    expt = []
    for rec in man.get("files") or []:
        insp = rec.get("inspection") or {}
        v = insp.get("fwhm")
        if v is not None and math.isfinite(float(v)) and float(v) > 0:
            fwhm.append(float(v))
        e = insp.get("exptime")
        if e is not None:
            expt.append(float(e))
    inv["manifest"] = {
        "calibration_mode": man.get("calibration_mode"),
        "status": man.get("status"),
        "is_calibrated": man.get("is_calibrated"),
        "rig": man.get("rig"),
        "center": man.get("center"),
        "n_manifest_files": len(man.get("files") or []),
        "fwhm_px_median": float(np.median(fwhm)) if fwhm else None,
        "fwhm_px_p16": float(np.percentile(fwhm, 16)) if fwhm else None,
        "fwhm_px_p84": float(np.percentile(fwhm, 84)) if fwhm else None,
        "n_fwhm": len(fwhm),
        "exptime_s": sorted(set(expt)),
    }

    raw_fits = sorted(RAW518.glob("*.fits")) + sorted(RAW518.glob("*.fit"))
    al_fits = sorted(AL518.glob("*.fits")) + sorted(AL518.glob("*.fit"))
    proc = sorted(AL518.glob("proc_*.csv"))
    inv["counts"] = {
        "raw_fits": len(raw_fits),
        "aligned_fits": len(al_fits),
        "proc_csv": len(proc),
        "aperture_lc": len(
            [
                p
                for p in LC518.glob("lightcurve_*.csv")
                if "_psf" not in p.name and "_adaptive" not in p.name
            ]
        )
        if LC518.is_dir()
        else 0,
        "psf_lc": len(list(LC518.glob("*_psf.csv"))) if LC518.is_dir() else 0,
        "epsf_fits": (PS518 / "masterstar_epsf.fits").is_file(),
        "epsf_meta": (PS518 / "masterstar_epsf_meta.json").is_file(),
        "snr_table": (D518 / "aperture_snr_table.json").is_file(),
        "snr_table_rejected": (D518 / "aperture_snr_table_REJECTED.json").is_file(),
    }

    sample_al = al_fits[0] if al_fits else None
    sample_raw = raw_fits[0] if raw_fits else None
    inv["header_aligned"] = header_facts(sample_al) if sample_al else None
    inv["header_raw"] = header_facts(sample_raw) if sample_raw else None
    if (PS518 / "MASTERSTAR.fits").is_file():
        inv["header_masterstar"] = header_facts(PS518 / "MASTERSTAR.fits")

    if LC518.is_dir():
        ap = [
            p
            for p in sorted(LC518.glob("lightcurve_*.csv"))
            if "_psf" not in p.name and "_adaptive" not in p.name
        ]
        inv["aperture_lc_stems"] = [p.stem.replace("lightcurve_", "", 1) for p in ap]

    cpt = PHOT518 / "comparison_stars_per_target.csv"
    if cpt.is_file():
        df = pd.read_csv(cpt, low_memory=False, dtype=str)
        inv["comparison_stars_per_target_cols"] = list(df.columns)
        inv["comparison_stars_per_target_n"] = int(len(df))
        if "status" in df.columns:
            inv["comp_status_counts"] = df["status"].value_counts().to_dict()
        if "selected" in df.columns:
            inv["comp_selected_counts"] = df["selected"].value_counts().to_dict()

    pin = PHOT518 / "photometry_plan.json"
    if pin.is_file():
        inv["photometry_plan"] = json.loads(pin.read_text(encoding="utf-8"))

    at = PHOT518 / "active_targets.csv"
    if at.is_file():
        adf = pd.read_csv(at, low_memory=False)
        inv["n_active_targets"] = int(len(adf))
        inv["n_skip_photometry"] = (
            int(adf["skip_photometry"].map(lambda x: str(x).lower() in ("1", "true")).sum())
            if "skip_photometry" in adf.columns
            else None
        )
        inv["target_origins"] = (
            adf["target_origin"].value_counts().to_dict() if "target_origin" in adf.columns else {}
        )

    sat = D518 / "sat_diag.json"
    if sat.is_file():
        inv["sat_diag"] = json.loads(sat.read_text(encoding="utf-8"))

    gpt = PHOT518 / "gain_photon_transfer.json"
    if gpt.is_file():
        inv["gain_pt"] = json.loads(gpt.read_text(encoding="utf-8"))

    # science set
    from epsf_science_set import build_epsf_science_set

    sci = build_epsf_science_set(PS518)
    inv["science_set"] = sci.to_meta_dict() if hasattr(sci, "to_meta_dict") else {
        "n_ids": len(sci.catalog_ids),
        "empty_reason": getattr(sci, "empty_reason", None),
    }

    hashes_516 = snapshot_516()
    (OUT / "hashes_516_before.json").write_text(
        json.dumps(hashes_516, indent=2) + "\n", encoding="ascii"
    )
    inv["g2_516"] = {
        "n_hashed": len(hashes_516),
        "epsf_sha": hashes_516.get(_rel(PS516 / "masterstar_epsf.fits")),
        "epsf_sha_expected": PROD_EPSF_SHA,
        "epsf_match": hashes_516.get(_rel(PS516 / "masterstar_epsf.fits")) == PROD_EPSF_SHA,
    }

    (OUT / "n0_inventory.json").write_text(json.dumps(inv, indent=2, default=str) + "\n", encoding="ascii")
    print(json.dumps({"ok": True, "out": str(OUT), "counts": inv["counts"], "g2": inv["g2_516"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
