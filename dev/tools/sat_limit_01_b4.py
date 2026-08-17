#!/usr/bin/env python3
"""SAT-LIMIT-01 B4: reclassify draft 515 catalog; BO ensemble without C2; check MAD."""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))
sys.path.insert(0, str(ROOT / "dev" / "tools"))

from pipeline import (  # noqa: E402
    SAT_LIMIT_CONTAINER_CLIP_ADU,
    SAT_LIMIT_NO_KNEE_FRAC,
    _annotate_masterstars_flux_zones,
)

DRAFT = ROOT / "Archive" / "Drafts" / "draft_000515"
SETUP = "NoFilter_60_2"
PS = DRAFT / "platesolve" / SETUP
PHOT = PS / "photometry"
MS_CSV = PS / "masterstars_full_match.csv"
MS_BAK = PS / "masterstars_full_match_before_sat_limit_01.csv"
COMP_CSV = PHOT / "comparison_stars_per_target.csv"
COMP_BAK = PHOT / "comparison_stars_per_target_before_sat_limit_01.csv"
PROC = DRAFT / "detrended_aligned" / "lights" / SETUP
BO = "1498613634033133184"
C2 = "1500748301498613248"
CHK_BO = "1498020894186918144"
MAD_SCALE = 1.4826
PEAK_TEST = SAT_LIMIT_CONTAINER_CLIP_ADU * SAT_LIMIT_NO_KNEE_FRAC  # 52428


def _cid(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    t = s.astype(str).str.strip().str.lower()
    return t.isin(["1", "true", "yes"])


def mad_mmag(arr: np.ndarray) -> float | None:
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return None
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)) * MAD_SCALE * 1000.0)


def reclassify(ms: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    before_sat = _bool(ms["is_saturated"]) if "is_saturated" in ms.columns else pd.Series(False, index=ms.index)
    before_zone = ms["zone"].astype(str) if "zone" in ms.columns else pd.Series("", index=ms.index)
    peak = pd.to_numeric(ms.get("peak_max_adu"), errors="coerce")
    out = ms.copy()
    out["saturate_limit_adu"] = float(SAT_LIMIT_CONTAINER_CLIP_ADU)
    out["saturate_limit_adu_85pct"] = float(PEAK_TEST)
    newly = peak > float(PEAK_TEST)
    newly = newly.fillna(False)
    out["is_saturated"] = newly
    if "likely_saturated" in out.columns:
        out["likely_saturated"] = _bool(out["likely_saturated"]) | newly
    else:
        out["likely_saturated"] = newly
    z = before_zone.copy()
    z = z.mask(newly, "saturated")
    # Stars previously tagged saturated only via empty-limit hole cannot happen (none were).
    out["zone"] = z
    if "is_usable" in out.columns:
        out.loc[newly, "is_usable"] = False
    n_zone_to_sat = int((newly & ~before_zone.astype(str).str.strip().str.lower().eq("saturated")).sum())
    affected_ids = out.loc[newly, "catalog_id"].astype(str).str.strip().tolist() if newly.any() else []
    stats = {
        "n_rows": int(len(out)),
        "peak_test_adu": float(PEAK_TEST),
        "clip_adu": float(SAT_LIMIT_CONTAINER_CLIP_ADU),
        "frac": float(SAT_LIMIT_NO_KNEE_FRAC),
        "n_is_saturated_before": int(before_sat.sum()),
        "n_is_saturated_after": int(newly.sum()),
        "n_zone_changed_to_saturated": n_zone_to_sat,
        "saturated_catalog_ids": affected_ids,
    }
    return out, stats


def comps_of(comp: pd.DataFrame, tid: str) -> list[str]:
    sub = comp[_cid(comp["target_catalog_id"]) == str(tid)]
    return [str(x).strip() for x in sub["catalog_id"].tolist()]


def load_proc_mags(ids: list[str]) -> tuple[list[str], dict[str, np.ndarray]]:
    files = sorted(PROC.glob("*.csv"))
    if not files:
        files = sorted(PROC.glob("proc_*.csv"))
    want = {str(i).strip() for i in ids}
    mag_by: dict[str, list[float]] = {i: [] for i in want}
    names: list[str] = []
    for fp in files:
        df = pd.read_csv(fp, dtype={"catalog_id": str}, usecols=lambda c: c in ("catalog_id", "dao_flux", "name"))
        if "catalog_id" not in df.columns:
            continue
        df["catalog_id"] = df["catalog_id"].astype(str).str.strip()
        flux = pd.to_numeric(df["dao_flux"], errors="coerce") if "dao_flux" in df.columns else None
        if flux is None:
            continue
        names.append(fp.name)
        got = dict(zip(df["catalog_id"], flux, strict=False))
        for cid in want:
            v = got.get(cid, float("nan"))
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = float("nan")
            mag_by[cid].append(-2.5 * math.log10(fv) if math.isfinite(fv) and fv > 0 else float("nan"))
    return names, {k: np.asarray(v, dtype=float) for k, v in mag_by.items()}


def check_mad(ens: list[str], check_id: str, cat_g: dict[str, float], rms: dict[str, float]) -> dict:
    from wide_err_03c import mag_calib_series

    ids = list(dict.fromkeys([*ens, check_id, BO]))
    _names, mag_by = load_proc_mags(ids)
    kcal, case, weights = mag_calib_series(
        m_star=mag_by[check_id],
        ens_ids=list(ens),
        mag_by=mag_by,
        cat_g=cat_g,
        rms_phase1=rms,
        self_exclude=False,
        focus_id=check_id,
    )
    return {
        "meter": "mag_calib_pytics_zp (XVAL-BO-01 / WIDE-ERR-03C product frame)",
        "case": case,
        "n_epochs": int(np.isfinite(kcal).sum()),
        "n_frames_csv": len(_names),
        "check_scatter_mad_mmag": mad_mmag(kcal),
        "ensemble": list(ens),
        "n_comp": len(ens),
        "weights": {k: float(v) for k, v in weights.items()},
    }


def gaia_g(ms: pd.DataFrame) -> dict[str, float]:
    gcol = "phot_g_mean_mag" if "phot_g_mean_mag" in ms.columns else "mag"
    out = {}
    for _, r in ms.iterrows():
        cid = str(r.get("catalog_id") or "").strip()
        if not cid:
            continue
        v = pd.to_numeric(r.get(gcol), errors="coerce")
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            out[cid] = fv
    return out


def reselect_bo(ms: pd.DataFrame, old_ens: list[str]) -> dict:
    """Layer-2 on the fixed pool: drop members that fail the saturation gate.

    Remaining members stay in their previous rank order. n_old=5, drop C2 -> 4,
    which is still >= n_comp_min=3, so no fill. A full RMS re-rank of the night
    is a later Phase 1 rebuild; this is the SAT-LIMIT membership correction.
    """
    ms = ms.copy()
    ms["catalog_id"] = _cid(ms["catalog_id"])
    sat = _bool(ms["is_saturated"]) if "is_saturated" in ms.columns else pd.Series(False, index=ms.index)
    likely = _bool(ms["likely_saturated"]) if "likely_saturated" in ms.columns else pd.Series(False, index=ms.index)
    z = ms["zone"].astype(str).str.strip().str.lower() if "zone" in ms.columns else pd.Series("", index=ms.index)
    blocked = set(ms.loc[sat | likely | z.isin(["saturated", "nonlinear"]), "catalog_id"].astype(str))
    kept = [c for c in old_ens if c not in blocked]
    dropped = [c for c in old_ens if c not in kept]
    return {
        "old_ensemble": old_ens,
        "new_ensemble": kept,
        "dropped": dropped,
        "n_old": len(old_ens),
        "n_new": len(kept),
        "c2_was_member": C2 in old_ens,
        "c2_in_new": C2 in kept,
        "note": "drop_saturated_keep_rank",
        "method": "layer2_membership_gate_on_fixed_pool (no RMS re-rank, no G-fill)",
    }


def affected_targets(comp: pd.DataFrame, sat_ids: set[str]) -> list[dict]:
    rows = []
    if "target_catalog_id" not in comp.columns:
        return rows
    for tid, sub in comp.groupby(_cid(comp["target_catalog_id"])):
        members = [str(x).strip() for x in sub["catalog_id"].tolist()]
        hit = [m for m in members if m in sat_ids]
        if hit:
            rows.append({"target_catalog_id": str(tid), "n_comp": len(members), "saturated_members": hit})
    return rows


def main() -> None:
    ms0 = pd.read_csv(MS_CSV, dtype={"catalog_id": str}, low_memory=False)
    ms0["catalog_id"] = _cid(ms0["catalog_id"])
    # Prove annotate fire-proof on the real C2 row with unresolved limit.
    c2 = ms0[ms0["catalog_id"] == C2].copy()
    fire = _annotate_masterstars_flux_zones(
        c2,
        noise_floor_adu=c2["noise_floor_adu"].iloc[0] if "noise_floor_adu" in c2.columns else 2100.0,
        equipment_saturate_adu=None,
        saturate_limit_adu_fallback=None,
        sigma_px=10.0,
        sky_median_adu=1401.0,
        prematch_peak_sigma_floor=1.8,
        frame_max_adu=68429.0,
        dao_detection_n_equiv=3.78,
    )
    ms1, stats = reclassify(ms0)
    if not MS_BAK.is_file():
        shutil.copy2(MS_CSV, MS_BAK)
    ms1.to_csv(MS_CSV, index=False, lineterminator="\n")

    comp = pd.read_csv(COMP_CSV, dtype={"catalog_id": str, "target_catalog_id": str})
    old_ens = comps_of(comp, BO)
    sat_ids = set(stats["saturated_catalog_ids"])
    affected = affected_targets(comp, sat_ids)
    sel = reselect_bo(ms1, old_ens)
    bo_tmpl = comp[_cid(comp["target_catalog_id"]) == BO]
    sidecar = PHOT / "comparison_stars_bo_sat_limit_01.csv"
    bo_kept = bo_tmpl[_cid(bo_tmpl["catalog_id"]).isin(sel["new_ensemble"])]
    bo_kept.to_csv(sidecar, index=False, lineterminator="\n")

    gmap = gaia_g(ms1)
    rms: dict[str, float] = {}
    if "comp_rms" in bo_tmpl.columns:
        for _, r in bo_tmpl.iterrows():
            rms[str(r["catalog_id"]).strip()] = float(pd.to_numeric(r.get("comp_rms"), errors="coerce") or 0.01)
    mad_old = check_mad(old_ens, CHK_BO, gmap, rms)
    mad_new = check_mad(sel["new_ensemble"], CHK_BO, gmap, rms)

    payload = {
        "draft": 515,
        "photometry_sha": "da9cce4",
        "chosen_limit": {
            "peak_test_adu": float(PEAK_TEST),
            "clip_adu": float(SAT_LIMIT_CONTAINER_CLIP_ADU),
            "frac": float(SAT_LIMIT_NO_KNEE_FRAC),
            "source": "conservative_default_0.80x_container_clip_65535",
            "knee_note": (
                "Cheap residual-vs-peak auto-detector fired at 25000 ADU on n=8 "
                "(-76 mmag) then reversed; that is not a resolved linearity knee. "
                "Bright-end +213 mmag at peak>60k is saturation, not a sub-clip knee. "
                "D1-2 remains OPEN (dome-flat ramp)."
            ),
        },
        "fire_proof_c2_unresolved": {
            "is_saturated": bool(fire.loc[fire.index[0], "is_saturated"]),
            "zone": str(fire.loc[fire.index[0], "zone"]),
            "saturate_limit_adu": float(fire.loc[fire.index[0], "saturate_limit_adu"]),
            "saturate_limit_adu_85pct": float(fire.loc[fire.index[0], "saturate_limit_adu_85pct"]),
        },
        "reclassify": stats,
        "c2_after": {
            "is_saturated": bool(ms1.loc[ms1["catalog_id"] == C2, "is_saturated"].iloc[0]),
            "zone": str(ms1.loc[ms1["catalog_id"] == C2, "zone"].iloc[0]),
            "peak_max_adu": float(pd.to_numeric(ms1.loc[ms1["catalog_id"] == C2, "peak_max_adu"], errors="coerce").iloc[0]),
        },
        "affected_ensembles": affected,
        "bo_reselect": sel,
        "check_mad_old": mad_old,
        "check_mad_new": mad_new,
        "accept_01b_note": (
            "D515-ACCEPT-01B 515 BO cell (7.0498 mmag, check 1498020894186918144, "
            "ensemble including C2) is superseded for ensemble-membership questions; "
            "pointer: CURSOR_RESULT_SAT_LIMIT_01.md. 01B 2x2 vs 514 same-meter table "
            "is not silently overwritten."
        ),
        "files": {
            "masterstars_backup": str(MS_BAK),
            "bo_ensemble_sidecar": str(PHOT / "comparison_stars_bo_sat_limit_01.csv"),
            "production_comp_csv_unchanged": str(COMP_CSV),
        },
    }
    outp = ROOT / "dev" / "results" / "SAT_LIMIT_01_summary.json"
    prev = {}
    if outp.is_file():
        try:
            prev = json.loads(outp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev = {}
    prev["b4"] = payload
    # Keep B1/B3 keys if the measure script wrote them.
    outp.write_text(json.dumps(prev, indent=2, default=str) + "\n", encoding="ascii")
    print(json.dumps(payload, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
