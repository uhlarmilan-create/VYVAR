#!/usr/bin/env python3
"""WIDE-ERR E4: per-comp excess noise measurement (read-only, E3 data paths)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src_py"))
sys.path.insert(0, str(REPO / "dev" / "scripts"))
import _bootstrap  # noqa: E402,F401

from config import AppConfig  # noqa: E402

sys.path.insert(0, str(REPO / "dev" / "tools"))
from wide_err_e2 import _field_comp_lc, _iqr  # noqa: E402
from wide_err_e3 import _photon_rel_median  # noqa: E402

SETUP = "NoFilter_60_2"
DRAFT = REPO / "Archive" / "Drafts" / "draft_000435_snapshot_skysurface_20260716"
CHECK_CID = "1499906247391001088"
OUT = REPO / "tmp" / "wide_err_e4"
DIAG_LC_ROOT = REPO / "tmp" / "wide_err_w1w2" / "diag_check_lc"
MAD_SCALE = 1.4826
MAG_ERR_SCALE = 1000.0
REL_TO_MAG = 2.5 / math.log(10.0)


def _load_proc_index(proc_dir: Path) -> dict[str, pd.DataFrame]:
    from gaia_catalog_id import normalize_gaia_source_id  # noqa: PLC0415

    out: dict[str, pd.DataFrame] = {}
    for p in sorted(proc_dir.glob("proc_*.csv")):
        df = pd.read_csv(p, low_memory=False, dtype={"catalog_id": str})
        id_col = "catalog_id" if "catalog_id" in df.columns else "name"
        df["_nid"] = df[id_col].map(lambda x: str(normalize_gaia_source_id(x) or "").strip())
        out[p.name] = df
    return out


def _proc_peak_median_for_comp(
    proc_index: dict[str, pd.DataFrame],
    cid: str,
    csv_files: list[Path],
    comp_mag: np.ndarray,
) -> float:
    peak_vals: list[float] = []
    for i, cp in enumerate(csv_files):
        if i >= len(comp_mag) or not math.isfinite(float(comp_mag[i])):
            continue
        df = proc_index.get(cp.name)
        if df is None:
            continue
        sub = df.loc[df["_nid"] == cid]
        if sub.empty:
            continue
        peak = float(pd.to_numeric(sub.iloc[0].get("peak_max_adu"), errors="coerce"))
        if math.isfinite(peak):
            peak_vals.append(peak)
    return float(np.median(peak_vals)) if peak_vals else float("nan")


def _mad_sigma_mag(x: np.ndarray) -> float:
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return float("nan")
    med = float(np.median(v))
    return float(MAD_SCALE * np.median(np.abs(v - med)))


def _excess_mmag(sigma_meas_mag: float, err_photon_rel_median: float) -> float:
    if not (math.isfinite(sigma_meas_mag) and math.isfinite(err_photon_rel_median)):
        return float("nan")
    sig_m = sigma_meas_mag * MAG_ERR_SCALE
    phot_m = err_photon_rel_median * MAG_ERR_SCALE
    return float(math.sqrt(max(0.0, sig_m * sig_m - phot_m * phot_m)))


def _bin_stats(rows: list[dict[str, Any]], key: str, bins: list[tuple[str, float, float]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label, lo, hi in bins:
        sub = [r for r in rows if math.isfinite(r[key]) and lo < r[key] <= hi]
        if not sub:
            out.append({"bin": label, "n": 0, "excess_median_mmag": float("nan")})
            continue
        ex = [r["excess_mmag"] for r in sub if math.isfinite(r["excess_mmag"])]
        out.append(
            {
                "bin": label,
                "n": len(sub),
                "excess_median_mmag": float(np.median(ex)) if ex else float("nan"),
            }
        )
    return out


def main() -> int:
    cfg = AppConfig()
    OUT.mkdir(parents=True, exist_ok=True)
    ps = DRAFT / "platesolve" / SETUP
    phot = ps / "photometry"
    lights = DRAFT / "detrended_aligned" / "lights" / SETUP
    proc_index = _load_proc_index(lights)
    comp_all = pd.read_csv(
        phot / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    ms = pd.read_csv(ps / "masterstars_full_match.csv", dtype={"catalog_id": str})
    gmap = {
        str(r["catalog_id"]).strip(): float(pd.to_numeric(r.get("phot_g_mean_mag"), errors="coerce"))
        for _, r in ms.iterrows()
    }
    csv_files = sorted(lights.glob("proc_*.csv"))
    all_frames_stub = pd.DataFrame({"catalog_id": [], "bjd": []})

    target_ids = sorted(
        t
        for t in {str(x).strip() for x in comp_all["target_catalog_id"].astype(str)}
        if t and (phot / "lightcurves" / f"check_kmag_{t}.csv").is_file()
    )

    comp_rows: list[dict[str, Any]] = []
    for t in target_ids:
        field = _field_comp_lc(t, comp_all, csv_files, cfg, all_frames_stub)
        if field is None:
            continue
        for cid in field["good_ids"]:
            arr = field["comp_lc"].get(cid)
            ref = field["comp_ref"].get(cid, float("nan"))
            if arr is None or not math.isfinite(ref):
                continue
            m = np.asarray(arr, dtype=np.float64)
            ok = np.isfinite(m)
            if int(np.count_nonzero(ok)) < 5:
                continue
            resid = m[ok] - ref
            sigma_meas = _mad_sigma_mag(resid)
            err_med = _photon_rel_median(proc_index, cid, csv_files, m)
            if not math.isfinite(err_med):
                continue
            peak_med = _proc_peak_median_for_comp(proc_index, cid, csv_files, m)
            ex = _excess_mmag(sigma_meas, err_med)
            comp_rows.append(
                {
                    "field": t,
                    "catalog_id": cid,
                    "g": gmap.get(cid, float("nan")),
                    "peak_max_adu_median": peak_med,
                    "sigma_meas_mag": sigma_meas,
                    "err_photon_rel_median": err_med,
                    "excess_mmag": ex,
                }
            )

    ex_all = np.asarray([r["excess_mmag"] for r in comp_rows if math.isfinite(r["excess_mmag"])], dtype=np.float64)
    n_gt_10 = int(np.count_nonzero(ex_all > 10.0))
    n_gt_20 = int(np.count_nonzero(ex_all > 20.0))

    g_bins = _bin_stats(
        comp_rows,
        "g",
        [
            ("G_8_10", 8.0, 10.0),
            ("G_10_11", 10.0, 11.0),
            ("G_11_12", 11.0, 12.0),
            ("G_12_13", 12.0, 13.0),
            ("G_13_14", 13.0, 14.0),
            ("G_14_16", 14.0, 16.0),
        ],
    )
    peak_vals = np.asarray([r["peak_max_adu_median"] for r in comp_rows if math.isfinite(r["peak_max_adu_median"])])
    if peak_vals.size >= 4:
        qs = np.quantile(peak_vals, [0.2, 0.4, 0.6, 0.8])
        adu_bins = [
            ("ADU_q0_20", float("-inf"), float(qs[0])),
            ("ADU_q20_40", float(qs[0]), float(qs[1])),
            ("ADU_q40_60", float(qs[1]), float(qs[2])),
            ("ADU_q60_80", float(qs[2]), float(qs[3])),
            ("ADU_q80_100", float(qs[3]), float("inf")),
        ]
    else:
        adu_bins = []
    adu_bin_stats = _bin_stats(comp_rows, "peak_max_adu_median", adu_bins) if adu_bins else []

    check_rows: list[dict[str, Any]] = []
    err_photon_check: list[float] = []
    for ck_path in sorted((phot / "lightcurves").glob("check_kmag_*.csv")):
        target_cid = ck_path.stem.replace("check_kmag_", "")
        id_col = "check_catalog_id" if "check_catalog_id" in pd.read_csv(ck_path, nrows=0).columns else "check_cid"
        ckdf = pd.read_csv(ck_path, nrows=1, low_memory=False)
        if str(ckdf[id_col].iloc[0]).strip() != CHECK_CID:
            continue
        lc_path = DIAG_LC_ROOT / target_cid / f"lightcurve_{CHECK_CID}.csv"
        if not lc_path.is_file():
            continue
        lc = pd.read_csv(lc_path, low_memory=False)
        m = pd.to_numeric(lc.get("mag_calib_final"), errors="coerce").to_numpy(dtype=np.float64)
        ep = pd.to_numeric(lc.get("err_photon"), errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(m)
        if int(np.count_nonzero(ok)) < 5:
            continue
        sigma_meas = _mad_sigma_mag(m[ok])
        ep_ok = ep[np.isfinite(ep) & (ep > 0)]
        if ep_ok.size == 0:
            continue
        err_med = float(np.median(ep_ok))
        err_photon_check.append(err_med)
        ex = _excess_mmag(sigma_meas, err_med)
        check_rows.append(
            {
                "field": target_cid,
                "sigma_meas_mag": sigma_meas,
                "err_photon_rel_median": err_med,
                "excess_mmag": ex,
            }
        )

    ex_check = np.asarray([r["excess_mmag"] for r in check_rows if math.isfinite(r["excess_mmag"])], dtype=np.float64)
    bright_ex = [
        r["excess_mmag"]
        for r in comp_rows
        if math.isfinite(r.get("g", float("nan"))) and 8.0 < r["g"] <= 10.0 and math.isfinite(r["excess_mmag"])
    ]

    med_all = float(np.median(ex_all)) if ex_all.size else float("nan")
    med_bright = float(np.median(bright_ex)) if bright_ex else float("nan")
    med_check = float(np.median(ex_check)) if ex_check.size else float("nan")
    ratio = med_check / med_bright if math.isfinite(med_check) and math.isfinite(med_bright) and med_bright > 0 else float("nan")

    # Verdict thresholds (mechanical)
    if math.isfinite(ratio):
        if ratio < 0.5:
            verdict = "WIDE-ERR-E4-UNDECIDED"
        elif ratio <= 2.0:
            verdict = f"WIDE-ERR-RIG-EXCESS: {med_check:.1f} mmag"
        else:
            verdict = "WIDE-ERR-CHECKSTAR-SPECIAL"
    else:
        verdict = "WIDE-ERR-E4-UNDECIDED"

    out = {
        "E4_1": {
            "n_fields": len(target_ids),
            "n_comp_instances": len(comp_rows),
            "err_photon_note": "proc CSV has no err_photon column; median photon err from proc rows (E3 _photon_rel_median path)",
            "excess_mmag_median": med_all,
            "excess_mmag_iqr": _iqr(ex_all),
            "excess_mmag_min": float(np.min(ex_all)) if ex_all.size else float("nan"),
            "excess_mmag_max": float(np.max(ex_all)) if ex_all.size else float("nan"),
            "n_excess_gt_10_mmag": n_gt_10,
            "n_excess_gt_20_mmag": n_gt_20,
            "by_g_bin": g_bins,
            "by_peak_adu_quintile": adu_bin_stats,
        },
        "E4_2": {
            "n_fields": len(check_rows),
            "excess_mmag_median": med_check,
            "excess_mmag_iqr": _iqr(ex_check),
            "err_photon_rel_median_across_fields": float(np.median(err_photon_check)) if err_photon_check else float("nan"),
        },
        "E4_3": {
            "median_excess_all_comps_mmag": med_all,
            "median_excess_bright_comps_G_8_10_mmag": med_bright,
            "n_bright_comps": len(bright_ex),
            "median_excess_check_star_mmag": med_check,
            "ratio_check_to_bright_comp": ratio,
            "verdict_line": verdict,
        },
    }
    (OUT / "wide_err_e4.json").write_text(json.dumps(out, indent=2) + "\n", encoding="ascii")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
