#!/usr/bin/env python3
"""ZONE-SIMPLIFY: offline zone distributions and draft_502 saturation measurements."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

REPO = Path(__file__).resolve().parents[2]
_SRC = REPO / "src_py"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline import (  # noqa: E402
    _annotate_masterstars_flux_zones,
    _detect_empirical_clip_level_adu,
    _resolve_peak_saturation_limit_adu,
)

DRAFTS = {
    "draft_452_fixture": REPO / "dev/results/context/session_20260727/draft_452_masterstars_full_match.csv",
    "draft_435": REPO / "Archive/Drafts/draft_000435/platesolve/NoFilter_60_2/masterstars_full_match.csv",
    "draft_500": REPO / "Archive/Drafts/draft_000500/platesolve/NoFilter_60_2/masterstars_full_match.csv",
    "draft_502": REPO / "Archive/Drafts/draft_000502/platesolve/V_60_2/masterstars_full_match.csv",
}

REF_FRAMES = {
    "draft_502": REPO / "Archive/Drafts/draft_000502/non_calibrated/lights/V_60_2/TOI-1131.01.b_2025-04-22_23-05-09_V.fits",
}


def zone_counts(ms: pd.DataFrame) -> dict[str, int]:
    return dict(Counter(ms["zone"].fillna("").astype(str)))


def _load_meta(meta_path: Path | None) -> dict:
    if meta_path is None or not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _resolve_sigma_and_sky(meta: dict, nf: float, *, sky_override: float | None = None, k_default: float = 1.8) -> tuple[float | None, float | None, float]:
    dyn = meta.get("dynamic_params") or {}
    prov = meta.get("provenance") or {}
    cfg = prov.get("config_snapshot") or {}
    k = float(cfg.get("masterstar_prematch_peak_sigma_floor", k_default))
    if sky_override is not None:
        sky_f = float(sky_override)
    elif dyn.get("sky_adu_per_px") is not None:
        sky_f = float(dyn["sky_adu_per_px"])
    else:
        sky_f = None
    if sky_f is not None and math.isfinite(nf) and k > 0:
        sigma = (nf - sky_f) / k
        if math.isfinite(sigma) and sigma > 0:
            return sigma, sky_f, k
    return None, sky_f, k


DRAFT_SPECS = {
    "draft_452_fixture": {
        "csv": REPO / "dev/results/context/session_20260727/draft_452_masterstars_full_match.csv",
        "meta": None,
        "sky_override": 1955.0,
        "frame_max": None,
    },
    "draft_435": {
        "csv": REPO / "Archive/Drafts/draft_000435/platesolve/NoFilter_60_2/masterstars_full_match.csv",
        "meta": REPO / "Archive/Drafts/draft_000435/platesolve/NoFilter_60_2/photometry/pipeline_meta.json",
        "sky_override": None,
        "frame_max": None,
    },
    "draft_500": {
        "csv": REPO / "Archive/Drafts/draft_000500/platesolve/NoFilter_60_2/masterstars_full_match.csv",
        "meta": REPO / "Archive/Drafts/draft_000500/platesolve/NoFilter_60_2/photometry/pipeline_meta.json",
        "sky_override": None,
        "frame_max": None,
    },
    "draft_502": {
        "csv": REPO / "Archive/Drafts/draft_000502/platesolve/V_60_2/masterstars_full_match.csv",
        "meta": REPO / "Archive/Drafts/draft_000502/platesolve/V_60_2/photometry/pipeline_meta.json",
        "sky_override": None,
        "frame_max": 98232.375,
    },
}


def reclassify(ms: pd.DataFrame, spec: dict) -> pd.DataFrame:
    nf = float(ms["noise_floor_adu"].iloc[0])
    sat85 = float(ms["saturate_limit_adu_85pct"].iloc[0]) if "saturate_limit_adu_85pct" in ms.columns else float("nan")
    equip = sat85 / 0.85 if math.isfinite(sat85) and sat85 > 0 else 65535.0
    sample = ms.drop(columns=["zone", "is_usable", "is_noisy", "is_saturated"], errors="ignore")
    meta = _load_meta(spec.get("meta"))
    sigma, sky, k = _resolve_sigma_and_sky(meta, nf, sky_override=spec.get("sky_override"))
    if sigma is None:
        raise ValueError(f"cannot resolve sigma for {spec}")
    clip = None
    frame_max = spec.get("frame_max")
    return _annotate_masterstars_flux_zones(
        sample,
        noise_floor_adu=nf,
        equipment_saturate_adu=equip,
        saturate_limit_adu_fallback=sat85 if math.isfinite(sat85) else None,
        sigma_px=sigma,
        sky_median_adu=sky,
        prematch_peak_sigma_floor=k,
        frame_max_adu=frame_max,
        empirical_clip_adu=clip,
        dao_detection_n_equiv=spec.get("dao_detection_n_equiv", 3.78),
    )


def q_statistic(data: np.ndarray, x: float, y: float) -> float:
    xi, yi = int(round(x)), int(round(y))
    h, w = data.shape
    if not (1 <= xi < w - 1 and 1 <= yi < h - 1):
        return float("nan")
    center = float(data[yi, xi])
    neigh = [
        data[yi - 1, xi - 1],
        data[yi - 1, xi],
        data[yi - 1, xi + 1],
        data[yi, xi - 1],
        data[yi, xi + 1],
        data[yi + 1, xi - 1],
        data[yi + 1, xi],
        data[yi + 1, xi + 1],
    ]
    m = float(np.mean(neigh))
    if m <= 0 or not math.isfinite(m):
        return float("nan")
    return center / m


def measure_draft_502_saturation(out: dict) -> None:
    ms_path = DRAFT_SPECS["draft_502"]["csv"]
    if not ms_path.is_file():
        out["draft_502_saturation"] = {"error": f"missing {ms_path}"}
        return
    ms = pd.read_csv(ms_path, low_memory=False, dtype={"catalog_id": str})
    at_path = REPO / "Archive/Drafts/draft_000502/platesolve/V_60_2/photometry/active_targets.csv"
    at = pd.read_csv(at_path, low_memory=False) if at_path.is_file() else pd.DataFrame()

    tic_row = ms[ms["name"].astype(str).str.contains("198213332", na=False)]
    if tic_row.empty and at_path.is_file():
        tic_ids = at[at["name"].astype(str).str.contains("198213332", na=False)]
        if not tic_ids.empty and "catalog_id" in tic_ids.columns:
            cid = str(tic_ids["catalog_id"].iloc[0])
            tic_row = ms[ms["catalog_id"].astype(str) == cid]

    sat_ms = ms[ms["zone"] == "saturated"] if "zone" in ms.columns else ms.iloc[0:0]
    sat_at = at[at["zone_flag"] == "saturated"] if "zone_flag" in at.columns else at.iloc[0:0]

    ref = REF_FRAMES["draft_502"]
    hist: dict = {}
    if ref.is_file():
        with fits.open(ref, memmap=False) as hdul:
            arr = np.asarray(hdul[0].data, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        vmax = float(np.max(finite))
        at_max = int(np.count_nonzero(np.abs(finite - vmax) <= 0.5))
        clip = _detect_empirical_clip_level_adu(arr)
        hist = {
            "frame_max_adu": vmax,
            "count_at_max": at_max,
            "truncation_detected": clip is not None,
            "empirical_clip_adu": clip,
            "peak_sat_limit_resolved": _resolve_peak_saturation_limit_adu(
                camera_sat_limit_adu=65535.0,
                saturate_fraction=0.85,
                sky_median_adu=float(np.median(finite)),
                frame_max_adu=vmax,
                empirical_clip_adu=clip,
            ),
        }
        qs_flagged = []
        for _, r in sat_ms.head(20).iterrows():
            qs_flagged.append(
                {
                    "name": r.get("name"),
                    "peak_max_adu": r.get("peak_max_adu"),
                    "Q": q_statistic(arr, float(r["x"]), float(r["y"])),
                }
            )
        unsat = ms[ms["zone"] != "saturated"].copy()
        unsat["flux"] = pd.to_numeric(unsat["flux"], errors="coerce")
        bright = unsat.nlargest(10, "flux")
        qs_bright = []
        for _, r in bright.iterrows():
            qs_bright.append(
                {
                    "name": r.get("name"),
                    "peak_max_adu": r.get("peak_max_adu"),
                    "Q": q_statistic(arr, float(r["x"]), float(r["y"])),
                }
            )
        hist["Q_flagged_sample"] = qs_flagged
        hist["Q_bright_unflagged_sample"] = qs_bright

    tic_before = None
    cid = "1625373404725030528"
    tic_row = ms[ms["catalog_id"].astype(str) == cid]
    if not tic_row.empty:
        r = tic_row.iloc[0]
        tic_before = {
            "name": "TIC 198213332",
            "catalog_id": cid,
            "peak_max_adu": r.get("peak_max_adu"),
            "saturate_limit_adu_85pct": r.get("saturate_limit_adu_85pct"),
            "zone": r.get("zone"),
            "likely_saturated": r.get("likely_saturated"),
            "flux": r.get("flux"),
        }

    ms_new = reclassify(ms, DRAFT_SPECS["draft_502"])
    tic_after_row = ms_new[ms_new["catalog_id"].astype(str) == cid]
    tic_after = None
    if not tic_after_row.empty:
        r = tic_after_row.iloc[0]
        tic_after = {"zone": r.get("zone"), "is_saturated": bool(r.get("is_saturated"))}

    out["draft_502_saturation"] = {
        "n_masterstars_saturated_before": int(len(sat_ms)),
        "n_targets_saturated_before": int(len(sat_at)),
        "histogram": hist,
        "TIC_198213332_before": tic_before,
        "TIC_198213332_after_reclassify": tic_after,
        "n_masterstars_saturated_after": int((ms_new["zone"] == "saturated").sum()),
        "is_usable_after": int(ms_new["is_usable"].sum()),
    }


def main() -> None:
    out: dict = {"zone_distributions": {}}
    for label, spec in DRAFT_SPECS.items():
        path = spec["csv"]
        if not path.is_file():
            out["zone_distributions"][label] = {"error": f"missing {path}"}
            continue
        ms = pd.read_csv(path, low_memory=False, dtype={"catalog_id": str})
        before = zone_counts(ms)
        ms_new = reclassify(ms, spec)
        after = zone_counts(ms_new)
        out["zone_distributions"][label] = {
            "n": len(ms),
            "before": before,
            "after": after,
            "is_usable_before": int(ms["is_usable"].sum()) if "is_usable" in ms.columns else None,
            "is_usable_after": int(ms_new["is_usable"].sum()),
        }
    measure_draft_502_saturation(out)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
