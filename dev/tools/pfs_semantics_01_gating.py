"""PFS-SEMANTICS-01 post-rebuild gating + SHA + B2. Run after Phase 2A."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))
sys.path.insert(0, str(ROOT / "dev" / "tests"))

from photometry_sha import compute_photometry_sha  # noqa: E402

DRAFT = ROOT / "Archive" / "Drafts" / "draft_000515"
SETUP = "NoFilter_60_2"
PHOT = DRAFT / "platesolve" / SETUP / "photometry"
LC = PHOT / "lightcurves"
SAT_JSON = ROOT / "dev" / "results" / "SAT_LIMIT_01_summary.json"
GATE_PREV = ROOT / "dev" / "results" / "SAT_RERANK_01_gating_97.csv"
OUT = ROOT / "dev" / "results" / "PFS_SEMANTICS_01_gating.json"

CV = "1497007144465726080"
DEPTH_IDS = {
    "1497622939696499840",
    "1498089270065726464",
    "1497037209236836736",
}
DA9_N_LC = 49


def _ids(s) -> str:
    return str(s or "").strip()


def main() -> int:
    at = pd.read_csv(PHOT / "active_targets.csv", dtype={"catalog_id": str}, low_memory=False)
    at["catalog_id"] = at["catalog_id"].map(_ids)
    prev = pd.read_csv(GATE_PREV, dtype={"catalog_id": str})
    prev["catalog_id"] = prev["catalog_id"].map(_ids)
    noise_ids = set(prev.loc[prev["was_zone_noise"].astype(str).str.lower().isin(["true", "1"]), "catalog_id"])
    sat_ids = json.loads(SAT_JSON.read_text(encoding="utf-8"))["b4"]["reclassify"]["saturated_catalog_ids"]
    sat_ids = {_ids(x) for x in sat_ids}

    lc_files = sorted(LC.glob("lightcurve_*.csv"))
    lc_ids = {_ids(p.stem.replace("lightcurve_", "").split("_")[0]) for p in lc_files}
    # aperture-only names: lightcurve_<id>.csv (no method suffix) plus maybe _psf
    lc_ap = {
        _ids(p.stem.replace("lightcurve_", ""))
        for p in lc_files
        if "_" not in p.stem.replace("lightcurve_", "", 1)
        or p.stem.count("_") == 1
    }
    # Prefer unsuffixed aperture files
    lc_aperture = set()
    for p in lc_files:
        rest = p.stem[len("lightcurve_") :]
        if rest.endswith("_psf") or rest.endswith("_adaptive"):
            continue
        lc_aperture.add(_ids(rest))

    n_phase1 = int(len(at))
    skip = at["skip_photometry"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    reason = at["skip_reason"].astype(str).str.strip() if "skip_reason" in at.columns else pd.Series([""] * len(at))
    n_noise = int(((reason == "zone_noise") & skip).sum())
    n_depth = int(((reason == "below_target_depth") & skip).sum())
    n_pfs = int(((reason == "per_frame_saturation") & skip).sum())
    skipped_ids = set(at.loc[skip, "catalog_id"])
    measured_ids = set(at.loc[~skip, "catalog_id"])

    cv = at[at["catalog_id"] == CV]
    cv_row = {}
    if not cv.empty:
        r = cv.iloc[0]
        scf = pd.to_numeric(r.get("sat_clean_frac"), errors="coerce")
        cv_row = {
            "catalog_id": CV,
            "zone_flag": str(r.get("zone_flag", "") or ""),
            "skip_photometry": bool(skip.loc[r.name]),
            "skip_reason": str(r.get("skip_reason", "") or ""),
            "sat_clean_frac": None if pd.isna(scf) else float(scf),
            "lc_emitted": CV in lc_aperture,
            "mag": None if pd.isna(pd.to_numeric(r.get("mag"), errors="coerce")) else float(pd.to_numeric(r.get("mag"), errors="coerce")),
        }

    expected_skip = set(noise_ids) | set(DEPTH_IDS)
    vsx_ids = set(at.loc[reason == "vsx_type_out_of_scope", "catalog_id"])
    expected_skip |= vsx_ids
    extra_skip = skipped_ids - expected_skip
    missing_skip = expected_skip - skipped_ids
    # CV CVn may move from measured to skip via per_frame_saturation
    allowed_extra_skip = {CV} if cv_row.get("skip_reason") == "per_frame_saturation" else set()
    unexpected_skip = extra_skip - allowed_extra_skip
    unexpected_measured = (missing_skip - vsx_ids) - ({CV} if CV in measured_ids else set())

    core_sha, core_n = compute_photometry_sha(DRAFT, include_comp_qa=False)
    full_sha, full_n = compute_photometry_sha(DRAFT, include_comp_qa=True)

    comp = pd.read_csv(
        PHOT / "comparison_stars_per_target.csv",
        dtype={"catalog_id": str, "target_catalog_id": str},
        low_memory=False,
    )
    ens_ids = set(comp["catalog_id"].astype(str).str.strip())
    sat_in_ens = sorted(sat_ids & ens_ids)

    meta = {}
    mp = PHOT / "pipeline_meta.json"
    if mp.is_file():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    prov = meta.get("provenance") if isinstance(meta.get("provenance"), dict) else {}
    snap = prov.get("config_snapshot") if isinstance(prov.get("config_snapshot"), dict) else {}
    pfs_snap = snap.get("per_frame_saturation_enabled")
    pfs_meta = {k: meta.get(k) for k in meta if str(k).startswith("per_frame_sat")}

    stop = bool(unexpected_skip or unexpected_measured)
    out = {
        "n_phase1_active": n_phase1,
        "n_lc_aperture": len(lc_aperture),
        "n_skip_zone_noise": n_noise,
        "n_skip_below_target_depth": n_depth,
        "n_skip_vsx_type_out_of_scope": int((reason == "vsx_type_out_of_scope").sum()),
        "n_photometry_set_97": int((reason != "vsx_type_out_of_scope").sum()),
        "da9cce4_n_lc": DA9_N_LC,
        "cv_cvn": cv_row,
        "gating_delta_vs_da9_49": int(len(lc_aperture) - DA9_N_LC),
        "expected_skip_n": len(expected_skip),
        "unexpected_skip_ids": sorted(unexpected_skip),
        "unexpected_measured_ids": sorted(unexpected_measured),
        "missing_skip_ids": sorted(missing_skip),
        "extra_skip_ids": sorted(extra_skip),
        "stop_and_report": stop,
        "photometry_sha_core": core_sha,
        "photometry_sha_core_n": core_n,
        "photometry_sha_full": full_sha,
        "photometry_sha_full_n": full_n,
        "b2_sat_ids_in_any_ensemble": sat_in_ens,
        "b2_n": len(sat_in_ens),
        "pfs_config_snapshot": pfs_snap,
        "pfs_pipeline_meta": pfs_meta,
        "git_hash": prov.get("git_hash") or meta.get("git_hash"),
        "quarantined_sha": "8f107cf",
        "lc_ids": sorted(lc_aperture),
        "skipped_ids": sorted(skipped_ids),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="ascii")
    print("N_LC", len(lc_aperture), flush=True)
    print("N_NOISE", n_noise, "N_DEPTH", n_depth, "N_PFS", n_pfs, flush=True)
    print("CV", json.dumps(cv_row), flush=True)
    print("STOP", stop, "unexpected_skip", sorted(unexpected_skip), "unexpected_measured", sorted(unexpected_measured), flush=True)
    print("SHA_CORE", core_sha[:12], "n", core_n, flush=True)
    print("B2_SAT_IN_ENS", len(sat_in_ens), flush=True)
    print("PFS_SNAP", pfs_snap, flush=True)
    print("WROTE", OUT, flush=True)
    return 1 if stop else 0


if __name__ == "__main__":
    raise SystemExit(main())
