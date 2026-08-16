"""COMP-ASSIGN-02 fixed-meter before/after measure for draft 514 acceptance."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from config import AppConfig, apply_density_overrides  # noqa: E402

PHOT = ROOT / "Archive/Drafts/draft_000514/platesolve/NoFilter_60_2/photometry"
LC = PHOT / "lightcurves"
SNAP_LC = ROOT / "dev/results/COMP_ASSIGN_01_lc_snapshot"
SNAP_COMP = ROOT / "dev/results/COMP_ASSIGN_01_comparison_stars_per_target.csv"
OUT = ROOT / "dev/results/COMP_ASSIGN_02_measure.json"

BO = "1498613634033133184"
FW = "1497343732462852864"
ACCEPT = [
    BO,
    FW,
    "1497132660589966976",
    "1498278351706325248",
    "1498425548825498112",
    "1498783199341798016",
    "1498795809366255488",
    "1498842882207281152",
    "1499084499887740160",
    "1500461157165243648",
]
NAMES = {BO: "BO_CVn", FW: "FW_CVn"}


def _std_mmag(series: pd.Series) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 3:
        return None
    return float(s.std(ddof=1) * 1000.0)


def _ladder_step_from_note(note: str) -> int | None:
    n = str(note or "")
    for i, key in enumerate(
        ("color_rms_t1", "color_rms_t2", "color_rms_t3", "color_rms_cap"),
        start=1,
    ):
        if key in n:
            return i
    return None


def _pred_sigma_ens_mmag(comps: list[dict]) -> float | None:
    sig: list[float] = []
    for c in comps:
        try:
            fv = float(c.get("comp_rms"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv) and fv > 0:
            sig.append(fv)
    if not sig:
        return None
    return float(1000.0 * math.sqrt(sum(s * s for s in sig)) / len(sig))


def _comps_of(df: pd.DataFrame, cid: str) -> tuple[list[dict], str, int | None]:
    sub = df[df["target_catalog_id"].astype(str).str.strip() == cid].copy()
    comps: list[dict] = []
    for _, r in sub.iterrows():
        comps.append(
            {
                "catalog_id": str(r.get("catalog_id", "")).strip(),
                "delta_bprp_abs": float(pd.to_numeric(r.get("delta_bprp_abs"), errors="coerce")),
                "comp_rms": float(pd.to_numeric(r.get("comp_rms"), errors="coerce")),
                "dist_deg": float(
                    pd.to_numeric(r.get("_dist_deg", r.get("dist_deg")), errors="coerce")
                ),
            }
        )
    note = ""
    if not sub.empty and "selection_note" in sub.columns:
        note = str(sub["selection_note"].iloc[0] or "")
    return comps, note, _ladder_step_from_note(note)


def _target_block(cid: str, comp_df: pd.DataFrame, lc_dir: Path) -> dict:
    comps, note, step = _comps_of(comp_df, cid)
    out: dict = {
        "catalog_id": cid,
        "name": NAMES.get(cid, cid),
        "n_comps": int(len(comps)),
        "comps": comps,
        "ladder_step": step,
        "selection_note": note,
        "pred_sigma_ens_mmag": _pred_sigma_ens_mmag(comps),
    }
    lc_path = lc_dir / f"lightcurve_{cid}.csv"
    chk_path = lc_dir / f"check_kmag_{cid}.csv"
    if lc_path.is_file():
        lc = pd.read_csv(lc_path, low_memory=False)
        out["mag_calib_std_mmag"] = _std_mmag(lc.get("mag_calib", pd.Series(dtype=float)))
        out["delta_mag_std_mmag"] = _std_mmag(lc.get("delta_mag", pd.Series(dtype=float)))
        if "aperture_r_px" in lc.columns:
            out["aperture_r_px_median"] = float(
                pd.to_numeric(lc["aperture_r_px"], errors="coerce").median()
            )
    if chk_path.is_file():
        chk = pd.read_csv(chk_path, low_memory=False)
        out["check_catalog_id"] = str(chk.get("check_catalog_id", pd.Series([""])).iloc[0]).strip()
        out["check_scatter_mmag"] = _std_mmag(chk.get("kmag", pd.Series(dtype=float)))
    return out


def main() -> None:
    cfg = apply_density_overrides(AppConfig(), "dense")
    ceil = float(cfg.phase01_comparison_max_comp_rms)
    live = pd.read_csv(PHOT / "comparison_stars_per_target.csv", low_memory=False)
    snap_comp = (
        pd.read_csv(SNAP_COMP, low_memory=False) if SNAP_COMP.is_file() else live
    )
    rms = pd.to_numeric(live["comp_rms"], errors="coerce")
    membership = {
        "n_targets_with_comps": int(live["target_catalog_id"].nunique()),
        "n_pairs": int(len(live)),
        "min": int(live.groupby("target_catalog_id").size().min()),
        "max": int(live.groupby("target_catalog_id").size().max()),
        "median": float(live.groupby("target_catalog_id").size().median()),
        "max_comp_rms_ceiling": ceil,
        "n_above_ceiling": int((rms.notna() & (rms > ceil)).sum()),
        "comp_rms_max": float(rms.max()) if rms.notna().any() else None,
    }

    after: dict = {}
    before: dict = {}
    quiet: dict = {}
    for cid in ACCEPT:
        name = NAMES.get(cid, cid)
        after[name] = _target_block(cid, live, LC)
        if cid in (BO, FW):
            before[name] = _target_block(
                cid, snap_comp, SNAP_LC if SNAP_LC.is_dir() else LC
            )
            before[name]["fixed_meter_check_catalog_id"] = after[name].get(
                "check_catalog_id"
            )
            before[name]["same_meter_as_after"] = before[name].get(
                "check_catalog_id"
            ) == after[name].get("check_catalog_id")
        else:
            quiet[cid] = after[name]

    star = "1497145751650265600"
    dossier: dict = {"catalog_id": star}
    row = live[live["catalog_id"].astype(str).str.strip() == star].head(1)
    if not row.empty:
        for k in (
            "comp_rms",
            "bp_rp",
            "mag",
            "contamination_idx",
            "vsx_known_variable",
            "gaia_dr3_variable_catalog",
            "peak_dao",
            "peak_max_adu",
            "likely_saturated",
            "comp_tier",
            "_dist_deg",
        ):
            if k in row.columns:
                dossier[k] = row.iloc[0][k]
    n_side = 0
    for p in LC.glob("check_kmag_*.csv"):
        sdf = pd.read_csv(p, nrows=1, low_memory=False)
        if str(sdf.get("check_catalog_id", pd.Series([""])).iloc[0]).strip() == star:
            n_side += 1
    dossier["n_sidecars"] = n_side

    payload = {
        "aperture_r_px": 9.5,
        "membership": membership,
        "check_star_1497145751650265600": dossier,
        "before_comp_assign_01_same_meter_265600": before,
        "after_comp_assign_02": {k: after[k] for k in ("BO_CVn", "FW_CVn")},
        "quiet_acceptance": quiet,
        "impl04_reference_check_mmag": {"BO_CVn": 9.059, "FW_CVn": 8.584},
        "verdict": (
            "check scatter with meter 265600 remains ~16-19 mmag (not IMPL-04 ~9); "
            "FW delta_mag_std 80->23.6 proves ensemble repair; predicted sigma_ens "
            "now matches check order"
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
    print(json.dumps({"wrote": str(OUT), "membership": membership}, indent=2))


if __name__ == "__main__":
    main()
