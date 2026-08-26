# -*- coding: ascii -*-
"""APERTURE-01 P-A1..P-A4 on era04 (mode a) vs era03; AIJ gate on BO."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(r"C:\ASTRO\python\VYVAR")
sys.path.insert(0, str(ROOT / "src_py"))
from masterstar_gaia_accounting import _norm_cid  # noqa: E402

ERA03 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era03_20260820"
ERA04 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era04_20260826"
SETUP = "NoFilter_60_2"
OUT = ROOT / "dev" / "results" / "context" / "session_20260826_c6"
BO = "1498613634033133184"
FW = "1497343732462852864"
GH = "1498804639818507904"
NAMES = {"BO": BO, "FW": FW, "GH": GH}
AIJ_BO = ROOT / "dev" / "results" / "XVAL_AIJ_01_bo_compare.csv"
AIJ_FW = None  # not on disk


def phot(root: Path) -> Path:
    return root / "platesolve" / SETUP / "photometry"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def demeaned_rms_mmag(mag: np.ndarray) -> float:
    x = np.asarray(mag, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return float("nan")
    d = x - float(np.median(x))
    return float(np.sqrt(np.mean(d * d)) * 1000.0)


def load_lc(root: Path, cid: str) -> pd.DataFrame:
    p = phot(root) / "lightcurves" / f"lightcurve_{cid}.csv"
    df = pd.read_csv(p, low_memory=False)
    df["source_file"] = df["source_file"].astype(str)
    return df


def frame_key(s: str) -> str:
    name = Path(str(s)).name
    stem = Path(name).stem
    if stem.startswith("proc_"):
        stem = stem[5:]
    return stem.lower()


def median_dmag_mmag(a: pd.Series, b: pd.Series) -> float:
    d = pd.to_numeric(a, errors="coerce") - pd.to_numeric(b, errors="coerce")
    d = d[np.isfinite(d)]
    if d.empty:
        return float("nan")
    return float(np.median(d) * 1000.0)


def main() -> int:
    qc = pd.read_csv(ERA04 / "calibrated" / "lights" / "qc_metrics.csv")
    qc["frame"] = qc["src"].map(frame_key)
    qc["fwhm_px"] = pd.to_numeric(qc["fwhm_px"], errors="coerce")

    stars = {}
    for name, cid in NAMES.items():
        e3 = load_lc(ERA03, cid)
        e4 = load_lc(ERA04, cid)
        e3["key"] = e3["source_file"].map(frame_key)
        e4["key"] = e4["source_file"].map(frame_key)
        m = e3.merge(e4, on="key", suffixes=("_e3", "_e4"))
        d_cal = median_dmag_mmag(m["mag_calib_e4"], m["mag_calib_e3"])
        rms3 = demeaned_rms_mmag(pd.to_numeric(e3["mag_calib"], errors="coerce").to_numpy())
        rms4 = demeaned_rms_mmag(pd.to_numeric(e4["mag_calib"], errors="coerce").to_numpy())
        j = e4.merge(qc[["frame", "fwhm_px"]], left_on="key", right_on="frame", how="inner")
        mag = pd.to_numeric(j["mag_calib"], errors="coerce")
        dmag = mag - float(np.nanmedian(mag))
        fw = pd.to_numeric(j["fwhm_px"], errors="coerce")
        ok = np.isfinite(dmag) & np.isfinite(fw)
        rho = float("nan")
        if int(ok.sum()) >= 8:
            rho = float(spearmanr(dmag[ok], fw[ok]).statistic)
        pol = str(e4["aperture_policy"].iloc[0]) if "aperture_policy" in e4.columns else ""
        af = float(e4["aperture_f"].iloc[0]) if "aperture_f" in e4.columns else float("nan")
        r_ap = float(pd.to_numeric(e4["aperture_r_px"], errors="coerce").median())
        stars[name] = {
            "cid": cid,
            "n_join": int(len(m)),
            "dmag_calib_mmag": None if not math.isfinite(d_cal) else round(d_cal, 4),
            "rms_e3_mmag": None if not math.isfinite(rms3) else round(rms3, 4),
            "rms_e4_mmag": None if not math.isfinite(rms4) else round(rms4, 4),
            "rms_ratio_e4_over_e3": (
                None
                if not (math.isfinite(rms3) and math.isfinite(rms4) and rms3 > 0)
                else round(rms4 / rms3, 4)
            ),
            "spearman_dmag_vs_fwhm": None if not math.isfinite(rho) else round(rho, 4),
            "aperture_policy": pol,
            "aperture_f": None if not math.isfinite(af) else af,
            "median_r_ap_px": None if not math.isfinite(r_ap) else round(r_ap, 4),
        }

    aij = {"path": str(AIJ_BO), "present": AIJ_BO.is_file(), "sha256": None, "rms_diff_mmag": None}
    if AIJ_BO.is_file():
        aij["sha256"] = sha256_file(AIJ_BO)
        tbl = pd.read_csv(AIJ_BO)
        e4 = load_lc(ERA04, BO)
        e4["Label"] = e4["source_file"].map(frame_key)
        tbl["Label"] = tbl["Label"].map(frame_key)
        j = tbl.merge(e4, on="Label", how="inner")
        aij_rel = pd.to_numeric(j["rel_flux_T1"], errors="coerce")
        vy_mag = pd.to_numeric(j["mag_calib"], errors="coerce")
        vy_rel = np.power(10.0, -0.4 * vy_mag.to_numpy())
        ok = np.isfinite(aij_rel) & np.isfinite(vy_rel) & (aij_rel > 0) & (vy_rel > 0)
        aij_n = aij_rel.to_numpy()[ok]
        vy_n = vy_rel[ok]
        aij_n = aij_n / float(np.median(aij_n))
        vy_n = vy_n / float(np.median(vy_n))
        diff = -2.5 * np.log10(aij_n / vy_n) * 1000.0
        aij["n"] = int(diff.size)
        aij["rms_diff_mmag"] = round(float(np.sqrt(np.mean(diff * diff))), 4)
        aij["median_diff_mmag"] = round(float(np.median(diff)), 4)
        aij["known_gate_mmag"] = 3.3
        aij["pass"] = bool(aij["rms_diff_mmag"] <= 3.3)

    pa1 = all(
        stars[n]["dmag_calib_mmag"] is not None and abs(stars[n]["dmag_calib_mmag"]) <= 3.0
        for n in NAMES
    )
    pa2 = {}
    for n in NAMES:
        r = stars[n]["rms_ratio_e4_over_e3"]
        pa2[n] = r is not None and abs(r - 1.0) <= 0.05

    out = {
        "mode": "f_fixed_night",
        "P-A1": {"pass": pa1, "stars": {k: v["dmag_calib_mmag"] for k, v in stars.items()}},
        "P-A2": {
            "pass": all(pa2.values()),
            "within_5pct": pa2,
            "rms_e3": {k: v["rms_e3_mmag"] for k, v in stars.items()},
            "rms_e4": {k: v["rms_e4_mmag"] for k, v in stars.items()},
            "ratio": {k: v["rms_ratio_e4_over_e3"] for k, v in stars.items()},
        },
        "P-A3": {
            "BO": aij,
            "FW": {
                "present": False,
                "note": "FW CVn AIJ table is not on disk; gate NOT MEASURED",
            },
        },
        "P-A4": {
            "mode_a_rho": {k: v["spearman_dmag_vs_fwhm"] for k, v in stars.items()},
            "note": "mode b rho filled by a1_mode_b_harness.py",
        },
        "stars": stars,
    }
    (OUT / "a1_pa_mode_a.json").write_text(json.dumps(out, indent=2), encoding="ascii")
    print(json.dumps({k: out[k] for k in ("P-A1", "P-A2", "P-A3")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
