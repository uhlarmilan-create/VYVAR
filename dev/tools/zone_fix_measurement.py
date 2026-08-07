# -*- coding: ascii -*-
"""Measure peak-significance zone thresholds across draft masterstars CSVs."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
_SRC_PY = ROOT / "src_py"
if str(_SRC_PY) not in sys.path:
    sys.path.insert(0, str(_SRC_PY))

from pipeline import _annotate_masterstars_flux_zones  # noqa: E402

T1_GRID = (3.0, 3.5, 4.0, 4.5, 5.0)
N_COMP_MIN = 2
MAG_TOL = 3.0
MIN_DIST_ARCSEC = 60.0


def _load_meta(meta_path: Path) -> dict:
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _resolve_sigma_and_sky(meta: dict, nf: float, k_default: float = 1.8) -> tuple[float | None, float | None, float]:
    dyn = meta.get("dynamic_params") or {}
    prov = meta.get("provenance") or {}
    cfg = prov.get("config_snapshot") or {}
    k = float(cfg.get("masterstar_prematch_peak_sigma_floor", k_default))
    sky = dyn.get("sky_adu_per_px")
    if sky is not None and math.isfinite(float(sky)):
        sky_f = float(sky)
    else:
        sky_f = None
    if sky_f is not None and math.isfinite(nf) and k > 0:
        sigma = (nf - sky_f) / k
        if math.isfinite(sigma) and sigma > 0:
            return sigma, sky_f, k
    return None, sky_f, k


def _zone_counts(df: pd.DataFrame) -> dict[str, int]:
    vc = df["zone"].value_counts()
    return {str(k): int(vc.get(k, 0)) for k in ("linear", "noisy1", "noisy2", "noisy3", "saturated", "unknown", "")}


def _source_split(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if "source_type" not in df.columns:
        return out
    for st in ("GAIA_MATCHED", "DAO_ONLY"):
        sub = df[df["source_type"].astype(str).str.upper().eq(st)]
        if sub.empty:
            continue
        n_lin = int((sub["zone"] == "linear").sum())
        out[st] = {"n": int(len(sub)), "linear": n_lin, "linear_frac": n_lin / len(sub)}
    return out


def _monotonic_subclass_medians(df: pd.DataFrame, sigma: float) -> dict[str, float]:
    peak_sig = pd.to_numeric(df["peak_dao"], errors="coerce") / float(sigma)
    med: dict[str, float] = {}
    for z in ("linear", "noisy1", "noisy2", "noisy3"):
        m = peak_sig[df["zone"] == z].median()
        if pd.notna(m):
            med[z] = float(m)
    ordered = [med.get(k) for k in ("noisy3", "noisy2", "noisy1", "linear") if k in med]
    mono = ordered == sorted(ordered) if len(ordered) >= 2 else True
    return {"medians": med, "monotonic": mono}


def _reclassify(
    ms: pd.DataFrame,
    *,
    nf: float,
    sat_lim: float,
    sigma: float,
    sky: float | None,
    k: float,
    t1: float,
) -> pd.DataFrame:
    t2 = t1 - 1.0
    t3 = t1 - 2.0
    base = ms.drop(columns=["zone", "is_usable", "is_noisy", "is_saturated"], errors="ignore")
    return _annotate_masterstars_flux_zones(
        base,
        noise_floor_adu=nf,
        equipment_saturate_adu=sat_lim / 0.85 if sat_lim > 0 else None,
        saturate_limit_adu_fallback=sat_lim,
        zone_mode="peak_significance",
        zone_sigma_linear=t1,
        zone_sigma_noisy1=t2,
        zone_sigma_noisy2=t3,
        sigma_px=sigma,
        sky_median_adu=sky,
        prematch_peak_sigma_floor=k,
    )


def _draft502_comp_clearance(ms: pd.DataFrame, active_path: Path, plate_scale: float) -> dict[str, int]:
    if not active_path.is_file():
        return {"targets": 0, "targets_ge_n_comp_min": 0}
    at = pd.read_csv(active_path, low_memory=False)
    usable = ms[ms["is_usable"].astype(bool)].copy()
    if usable.empty:
        return {"targets": int(len(at)), "targets_ge_n_comp_min": 0}
    cleared = 0
    for _, tgt in at.iterrows():
        if bool(tgt.get("skip_photometry", False)):
            continue
        try:
            tmag = float(tgt["mag"])
        except (TypeError, ValueError):
            continue
        tx = float(tgt["x"])
        ty = float(tgt["y"])
        tcid = str(int(tgt["catalog_id"])) if pd.notna(tgt.get("catalog_id")) else ""
        n_cand = 0
        for _, c in usable.iterrows():
            ccid = str(int(c["catalog_id"])) if pd.notna(c.get("catalog_id")) else ""
            if tcid and ccid == tcid:
                continue
            if bool(c.get("vsx_known_variable", False)):
                continue
            try:
                cmag = float(c["mag"])
            except (TypeError, ValueError):
                continue
            if abs(cmag - tmag) > MAG_TOL:
                continue
            dx = float(c["x"]) - tx
            dy = float(c["y"]) - ty
            dist_arcsec = math.hypot(dx, dy) * plate_scale
            if dist_arcsec < MIN_DIST_ARCSEC:
                continue
            n_cand += 1
        if n_cand >= N_COMP_MIN:
            cleared += 1
    n_eligible = int((~at["skip_photometry"].astype(bool)).sum()) if "skip_photometry" in at.columns else len(at)
    return {"targets": n_eligible, "targets_ge_n_comp_min": cleared}


def _premise_fixture() -> dict:
    p = ROOT / "dev/results/context/session_20260727/draft_452_masterstars_full_match.csv"
    ms = pd.read_csv(p, low_memory=False)
    nf = float(ms["noise_floor_adu"].iloc[0])
    sky = 1955.0
    k = 1.8
    sigma = (nf - sky) / k
    peak_dao = pd.to_numeric(ms["peak_dao"], errors="coerce")
    peak_max = pd.to_numeric(ms["peak_max_adu"], errors="coerce")
    peak_sig = peak_dao / sigma
    matched = ms[ms["source_type"].astype(str).str.upper().eq("GAIA_MATCHED")]
    dao = ms[ms["source_type"].astype(str).str.upper().eq("DAO_ONLY")]
    return {
        "n": len(ms),
        "zone_current": ms["zone"].value_counts().to_dict(),
        "sigma_px": sigma,
        "peak_max_matched_median": float(peak_max[matched.index].median()),
        "peak_max_minus_sky": float(peak_max[matched.index].median()) - sky,
        "peak_dao_matched_median": float(peak_dao[matched.index].median()),
        "peak_sig_dao_only_median": float(peak_sig[dao.index].median()),
        "peak_sig_matched_median": float(peak_sig[matched.index].median()),
        "zone_peak_sig_medians": {
            z: float(peak_sig[ms["zone"] == z].median())
            for z in ("linear", "noisy1", "noisy2", "noisy3")
            if (ms["zone"] == z).any()
        },
        "linear_p05_peak_sig": float(peak_sig[ms["zone"] == "linear"].quantile(0.05)),
        "peak_sig_ge_3_5": int((peak_sig >= 3.5).sum()),
        "peak_sig_ge_4_0": int((peak_sig >= 4.0).sum()),
        "linear_count": int((ms["zone"] == "linear").sum()),
    }


def _draft_specs() -> list[dict]:
    return [
        {
            "label": "draft_452_fixture",
            "csv": ROOT / "dev/results/context/session_20260727/draft_452_masterstars_full_match.csv",
            "meta": None,
            "sky_override": 1955.0,
            "k_override": 1.8,
            "active_targets": None,
            "plate_scale": 1.301,
        },
        {
            "label": "draft_435",
            "csv": ROOT / "Archive/Drafts/draft_000435/platesolve/NoFilter_60_2/masterstars_full_match.csv",
            "meta": ROOT / "Archive/Drafts/draft_000435/platesolve/NoFilter_60_2/photometry/pipeline_meta.json",
            "sky_override": None,
            "k_override": None,
            "active_targets": None,
            "plate_scale": 1.301,
        },
        {
            "label": "draft_500",
            "csv": ROOT / "Archive/Drafts/draft_000500/platesolve/NoFilter_60_2/masterstars_full_match.csv",
            "meta": ROOT / "Archive/Drafts/draft_000500/platesolve/NoFilter_60_2/photometry/pipeline_meta.json",
            "sky_override": None,
            "k_override": None,
            "active_targets": None,
            "plate_scale": 1.301,
        },
        {
            "label": "draft_502",
            "csv": ROOT / "Archive/Drafts/draft_000502/platesolve/V_60_2/masterstars_full_match.csv",
            "meta": ROOT / "Archive/Drafts/draft_000502/platesolve/V_60_2/photometry/pipeline_meta.json",
            "sky_override": None,
            "k_override": None,
            "active_targets": ROOT
            / "Archive/Drafts/draft_000502/platesolve/V_60_2/photometry/active_targets.csv",
            "plate_scale": 1.3010910511796954,
        },
    ]


def run_measurement() -> dict:
    premise = _premise_fixture()
    rows: list[dict] = []
    for spec in _draft_specs():
        csv_path = Path(spec["csv"])
        if not csv_path.is_file():
            continue
        ms = pd.read_csv(csv_path, low_memory=False)
        nf = float(ms["noise_floor_adu"].iloc[0])
        sat = float(ms["saturate_limit_adu_85pct"].iloc[0])
        meta = _load_meta(Path(spec["meta"])) if spec["meta"] else {}
        sigma, sky_meta, k_meta = _resolve_sigma_and_sky(meta, nf)
        sky = spec["sky_override"] if spec["sky_override"] is not None else sky_meta
        k = spec["k_override"] if spec["k_override"] is not None else k_meta
        if sigma is None and sky is not None:
            sigma = (nf - float(sky)) / float(k)
        if sigma is None:
            continue
        current_usable = int(ms["is_usable"].sum()) if "is_usable" in ms.columns else int((ms["zone"] == "linear").sum())
        current_zones = _zone_counts(ms)
        for t1 in T1_GRID:
            out = _reclassify(ms, nf=nf, sat_lim=sat, sigma=float(sigma), sky=sky, k=float(k), t1=t1)
            mono = _monotonic_subclass_medians(out, float(sigma))
            row = {
                "draft": spec["label"],
                "T1": t1,
                "T2": t1 - 1.0,
                "T3": t1 - 2.0,
                "sigma_px": float(sigma),
                "zones_current": current_zones,
                "zones_new": _zone_counts(out),
                "is_usable_current": current_usable,
                "is_usable_new": int(out["is_usable"].sum()),
                "source_split_new": _source_split(out),
                "subclass_monotonic": mono["monotonic"],
                "subclass_peak_sig_medians": mono["medians"],
            }
            if spec["label"] == "draft_502" and spec["active_targets"]:
                row["draft502_comp_clearance"] = _draft502_comp_clearance(
                    out, Path(spec["active_targets"]), float(spec["plate_scale"])
                )
            rows.append(row)
    return {"premise": premise, "rows": rows}


def main() -> None:
    result = run_measurement()
    out_path = ROOT / "dev/results/zone_fix_measurement.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
