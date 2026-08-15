"""IMPL-02 Part E: re-run Q3 level regression on colour-corrected magnitudes."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))
sys.path.insert(0, str(ROOT / "dev" / "tools"))

from pre_impl_01_measure import (  # noqa: E402
    PHOT,
    _sha,
    load_mags,
    measure_q3,
)


def main() -> None:
    at = pd.read_csv(PHOT / "active_targets.csv")
    comps = pd.read_csv(PHOT / "comparison_stars_per_target.csv")
    suspected: set[str] = set()
    sp = PHOT / "suspected_variables.csv"
    if sp.is_file():
        suspected = set(pd.read_csv(sp)["catalog_id"].astype(str))

    fids, mag, sat, _xy, _gmag = load_mags()
    q3_raw = measure_q3(fids, mag, sat, comps, at, suspected)

    try:
        from config import AppConfig

        k = float(AppConfig().color_level_k_mag_per_bprp)
    except Exception:  # noqa: BLE001
        k = -0.373

    hit = at[at["name"] == "BO CVn"]
    tid = str(hit.iloc[0]["catalog_id"])
    sub = comps[comps["target_catalog_id"].astype(str) == tid]
    g_by = {
        str(r["catalog_id"]): float(pd.to_numeric(r.get("phot_g_mean_mag"), errors="coerce"))
        for _, r in sub.iterrows()
    }

    rows = []
    for star in q3_raw.get("stars") or []:
        cid = str(star["catalog_id"])
        dcol = float(star["delta_colour_bprp"])  # ens - target (Q3 convention)
        level_raw = float(star["mean_level_offset_mag"])
        # Production: corr = k * (target - ens) = -k * dcol
        corr = (-float(k)) * dcol if math.isfinite(k) and math.isfinite(dcol) else 0.0
        rows.append(
            {
                "catalog_id": cid,
                "delta_colour_bprp": dcol,
                "G": g_by.get(cid, float("nan")),
                "level_raw_mag": level_raw,
                "level_corr_mag": level_raw + corr,
                "corr_mag": corr,
            }
        )

    dc = np.asarray([r["delta_colour_bprp"] for r in rows], float)
    lv = np.asarray([r["level_corr_mag"] for r in rows], float)
    G = np.asarray([r["G"] for r in rows], float)
    ok = np.isfinite(dc) & np.isfinite(lv)
    okg = ok & np.isfinite(G)

    def _fit1(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        A = np.vstack([x, np.ones(len(x))]).T
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        dof = max(1, len(x) - 2)
        s2 = float(np.sum(resid**2) / dof)
        se0 = math.sqrt(s2 * float(np.linalg.inv(A.T @ A)[0, 0]))
        return float(coef[0]), se0

    lev, se = _fit1(dc[ok], lv[ok]) if int(ok.sum()) >= 10 else (float("nan"), float("nan"))
    lev_g, se_g = float("nan"), float("nan")
    if int(okg.sum()) >= 15:
        A = np.vstack([dc[okg], G[okg], np.ones(int(okg.sum()))]).T
        y = lv[okg]
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        dof = max(1, len(y) - 3)
        s2 = float(np.sum(resid**2) / dof)
        lev_g = float(coef[0])
        se_g = math.sqrt(s2 * float(np.linalg.inv(A.T @ A)[0, 0]))

    out = {
        "commit_sha": _sha(),
        "k_config_mag_per_bprp": k,
        "n_stars": len(rows),
        "formula": "corr = k*(target_bp_rp - ens_bp_rp) = -k*delta_colour_Q3",
        "post_correction_level_mmag_per_bprp": lev * 1000.0 if math.isfinite(lev) else None,
        "post_correction_level_se_mmag": se * 1000.0 if math.isfinite(se) else None,
        "post_correction_level_G_controlled_mmag_per_bprp": lev_g * 1000.0
        if math.isfinite(lev_g)
        else None,
        "post_correction_level_G_controlled_se_mmag": se_g * 1000.0 if math.isfinite(se_g) else None,
        "pre_impl_level_G_controlled_mmag_per_bprp": -373.09,
        "sign_inverted_would_be_near_mmag_per_bprp": -746.0,
        "consistent_with_zero_2se": bool(
            math.isfinite(lev_g) and math.isfinite(se_g) and abs(lev_g) < 2.0 * se_g
        ),
        "q3_raw_level_mmag_per_bprp": q3_raw.get("level_term_mmag_per_bprp"),
        "q3_raw_level_G_controlled_mmag_per_bprp": q3_raw.get(
            "level_term_mmag_per_bprp_G_controlled"
        ),
    }
    dest = ROOT / "dev" / "results" / "IMPL_02_part_e_colour.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
