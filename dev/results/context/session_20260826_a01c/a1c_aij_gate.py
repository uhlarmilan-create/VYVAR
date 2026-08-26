# -*- coding: ascii -*-
"""APERTURE-01c independent gate: BO at f=1.35, comps pinned from XVAL_AIJ_01 tbl.

Compare CSV has no comp list. The AIJ table does: RA_C2..RA_C6 / DEC_C2..DEC_C6
(hours/deg). Match to masterstars; STOP if match fails.
VYVAR r = 1.35 x night QC FWHM (mode a); AIJ Source_Radius=7.
rel_flux = T1 / sum(C2..C6). RMS of median-normalized mag difference.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(r"C:\ASTRO\python\VYVAR")
sys.path.insert(0, str(ROOT / "src_py"))

from aperture_policy import load_qc_fwhm_map, resolve_aperture_geometry  # noqa: E402
from aperture_scatter_select import flux_to_inst_mag  # noqa: E402
from masterstar_gaia_accounting import _norm_cid  # noqa: E402
from photometry_core import _aperture_flux_sky_batch  # noqa: E402

ERA04 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era04_20260826"
SETUP = "NoFilter_60_2"
OUT = ROOT / "dev" / "results" / "context" / "session_20260826_a01c"
AIJ_CSV = ROOT / "dev" / "results" / "XVAL_AIJ_01_bo_compare.csv"
AIJ_TBL = ROOT / "dev" / "results" / "XVAL_AIJ_01_Table.tbl"
F = 1.35
ANN_IN = 4.75
ANN_OUT = 9.0
MATCH_MAX_ARCSEC = 8.0


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def frame_key(s: str) -> str:
    name = Path(str(s)).name
    stem = Path(name).stem
    if stem.startswith("proc_"):
        stem = stem[5:]
    return stem.lower()


def sep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    dra = (ra1 - ra2) * math.cos(math.radians(0.5 * (dec1 + dec2)))
    return math.hypot(dra, dec1 - dec2) * 3600.0


def read_aij_ensemble(tbl: Path) -> dict:
    if not tbl.is_file():
        return {"ok": False, "reason": "tbl_missing"}
    with tbl.open(encoding="utf-8", errors="replace", newline="") as f:
        header = f.readline().rstrip("\n").split("\t")
        row = f.readline().rstrip("\n").split("\t")
    idx = {h: i for i, h in enumerate(header)}
    csv_cols = []
    if AIJ_CSV.is_file():
        with AIJ_CSV.open(encoding="utf-8", newline="") as cf:
            csv_cols = next(csv.reader(cf))
    if not any(c.startswith("RA_C") for c in header):
        return {
            "ok": False,
            "reason": "no_comp_list",
            "tbl_cols_sample": header[:20],
            "csv_cols": csv_cols,
        }
    stars = {}
    for role in ("T1", "C2", "C3", "C4", "C5", "C6"):
        rk, dk = f"RA_{role}", f"DEC_{role}"
        if rk not in idx or dk not in idx:
            return {"ok": False, "reason": f"missing_{role}", "header": header}
        ra_h = float(row[idx[rk]])
        dec = float(row[idx[dk]])
        stars[role] = {"ra_hours": ra_h, "ra_deg": ra_h * 15.0, "dec_deg": dec}
    return {
        "ok": True,
        "source_radius_px": float(row[idx["Source_Radius"]]) if "Source_Radius" in idx else None,
        "sky_in": float(row[idx["Sky_Rad(min)"]]) if "Sky_Rad(min)" in idx else None,
        "sky_out": float(row[idx["Sky_Rad(max)"]]) if "Sky_Rad(max)" in idx else None,
        "stars": stars,
        "csv_has_comp_list": any("C2" in c or "comp" in c.lower() for c in csv_cols),
        "csv_cols": csv_cols,
    }


def match_ids(ens: dict, ms: pd.DataFrame) -> dict:
    out = {}
    for role, rec in ens["stars"].items():
        best = None
        for _, row in ms.iterrows():
            try:
                s = sep_arcsec(
                    rec["ra_deg"],
                    rec["dec_deg"],
                    float(row["ra_deg"]),
                    float(row["dec_deg"]),
                )
            except (TypeError, ValueError, KeyError):
                continue
            cid = _norm_cid(row["catalog_id"]) or str(row["catalog_id"]).strip()
            if best is None or s < best[0]:
                best = (s, cid, float(row["x"]), float(row["y"]))
        out[role] = {
            "catalog_id": None if best is None else best[1],
            "sep_arcsec": None if best is None else round(best[0], 3),
            "x": None if best is None else best[2],
            "y": None if best is None else best[3],
            "ok": bool(best is not None and best[0] <= MATCH_MAX_ARCSEC),
        }
    return out


def main() -> int:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    rec: dict = {
        "task": "APERTURE-01c independent AIJ gate",
        "f": F,
        "mode": "f_fixed_night",
        "aij_csv": str(AIJ_CSV),
        "aij_tbl": str(AIJ_TBL),
        "aij_csv_sha256": sha256_file(AIJ_CSV) if AIJ_CSV.is_file() else None,
        "aij_tbl_sha256": sha256_file(AIJ_TBL) if AIJ_TBL.is_file() else None,
    }
    ens = read_aij_ensemble(AIJ_TBL)
    rec["ensemble_from_file"] = {k: ens[k] for k in ens if k != "stars"}
    rec["ensemble_stars_raw"] = ens.get("stars")
    if not ens.get("ok"):
        rec["stop"] = True
        rec["stop_reason"] = ens.get("reason", "no_comp_list")
        (OUT / "a1c_aij_gate.json").write_text(json.dumps(rec, indent=2), encoding="ascii")
        print("STOP: " + rec["stop_reason"])
        return 2

    ms = pd.read_csv(
        ERA04 / "platesolve" / SETUP / "masterstars_full_match.csv",
        dtype={"catalog_id": str},
        low_memory=False,
    )
    matched = match_ids(ens, ms)
    rec["matched"] = matched
    if not all(v["ok"] for v in matched.values()):
        rec["stop"] = True
        rec["stop_reason"] = "comp_match_failed"
        (OUT / "a1c_aij_gate.json").write_text(json.dumps(rec, indent=2), encoding="ascii")
        print("STOP: comp_match_failed")
        return 2

    t1 = matched["T1"]["catalog_id"]
    comps = [matched[f"C{i}"]["catalog_id"] for i in range(2, 7)]
    rec["t1"] = t1
    rec["comp_ids"] = comps
    ids = [t1] + comps
    xy = {role: (matched[role]["x"], matched[role]["y"]) for role in matched}
    pos = np.array([xy[r] for r in ("T1", "C2", "C3", "C4", "C5", "C6")], dtype=np.float64)

    qc_map, night = load_qc_fwhm_map(ERA04 / "calibrated" / "lights" / "qc_metrics.csv")
    night_f = float(night) if night is not None else 5.191733
    r_ap, r_in, r_out = resolve_aperture_geometry(
        f=F, fwhm_px=night_f, annulus_inner_fwhm=ANN_IN, annulus_outer_fwhm=ANN_OUT
    )
    rec["night_fwhm_px"] = round(night_f, 6)
    rec["r_ap_px"] = round(float(r_ap), 4)
    rec["r_in_px"] = round(float(r_in), 4)
    rec["r_out_px"] = round(float(r_out), 4)
    rec["aij_r_px"] = ens.get("source_radius_px")
    rec["qc_n"] = len(qc_map)

    lights = sorted(
        p
        for p in (ERA04 / "detrended_aligned" / "lights" / SETUP).glob("*.fits")
        if p.stem.upper() != "MASTERSTAR"
    )
    flux = {cid: [] for cid in ids}
    keys = []
    for fp in lights:
        with fits.open(fp, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
        keys.append(frame_key(fp.name))
        fl, _sky = _aperture_flux_sky_batch(data, pos, r_ap, r_in, r_out)
        for j, cid in enumerate(ids):
            flux[cid].append(float(fl[j]))

    tflux = np.asarray(flux[t1], dtype=np.float64)
    csum = np.zeros(len(lights), dtype=np.float64)
    for c in comps:
        csum += np.asarray(flux[c], dtype=np.float64)
    vy_rel = tflux / csum
    rec["n_frames"] = len(lights)

    aij = pd.read_csv(AIJ_CSV)
    aij["Label"] = aij["Label"].map(frame_key)
    tmp = pd.DataFrame({"Label": keys, "vy_rel_pin": vy_rel})
    j = aij.merge(tmp, on="Label", how="inner")
    aij_rel = pd.to_numeric(j["rel_flux_T1"], errors="coerce").to_numpy()
    vy = pd.to_numeric(j["vy_rel_pin"], errors="coerce").to_numpy()
    ok = np.isfinite(aij_rel) & np.isfinite(vy) & (aij_rel > 0) & (vy > 0)
    rec["n_join"] = int(ok.sum())
    rms = float("nan")
    if int(ok.sum()) >= 8:
        a = aij_rel[ok] / float(np.median(aij_rel[ok]))
        b = vy[ok] / float(np.median(vy[ok]))
        diff = -2.5 * np.log10(a / b) * 1000.0
        rms = float(np.sqrt(np.mean(diff * diff)))
        rec["rms_diff_mmag"] = round(rms, 4)
        rec["median_diff_mmag"] = round(float(np.median(diff)), 4)
        rec["max_abs_diff_mmag"] = round(float(np.max(np.abs(diff))), 4)
    rec["gate_mmag"] = 4.0
    rec["pass"] = bool(math.isfinite(rms) and rms <= 4.0)
    rec["elapsed_s"] = round(time.perf_counter() - t0, 2)
    (OUT / "a1c_aij_gate.json").write_text(json.dumps(rec, indent=2, default=str), encoding="ascii")
    print(
        "RMS=%.4f mmag n=%d r_ap=%.3f AIJ_r=%s pass=%s elapsed=%.1fs"
        % (rms, rec["n_join"], r_ap, rec["aij_r_px"], rec["pass"], rec["elapsed_s"])
    )
    return 0 if rec["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
