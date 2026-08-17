"""EXPORT-HDR-01: re-export BO CVn AAVSO + VarAstro from the Part 3 product."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from citations import build_run_citation_context, load_pipeline_meta  # noqa: E402
from config import AppConfig  # noqa: E402
from export_reports import (  # noqa: E402
    _select_export_lc_rows,
    export_all_method_lightcurve_reports,
    find_truncated_gaia_ids,
)
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402

DRAFT = ROOT / "Archive" / "Drafts" / "draft_000515"
SETUP = "NoFilter_60_2"
PHOT = DRAFT / "platesolve" / SETUP / "photometry"
LC = PHOT / "lightcurves"
REPORTS = PHOT / "lightcurves_reports"
BO = "1498613634033133184"
OUT = ROOT / "dev" / "results" / "EXPORT_HDR_01_summary.json"


def _parse_aavso(path: Path) -> tuple[str, list[dict]]:
    text = path.read_text(encoding="utf-8")
    header = []
    rows = []
    for ln in text.splitlines():
        if ln.startswith("#"):
            header.append(ln)
            continue
        if not ln.strip() or ln.startswith("STARID"):
            continue
        parts = ln.split(",")
        if len(parts) < 15:
            continue
        rows.append(
            {
                "bjd": float(parts[1]),
                "mag": float(parts[2]),
                "kname": parts[9],
                "kmag": parts[10],
                "notes": parts[14],
            }
        )
    return "\n".join(header), rows


def main() -> int:
    cfg = AppConfig()
    meta = load_pipeline_meta(PHOT)
    if not isinstance(meta, dict):
        meta = {}
    prov = meta.get("provenance") if isinstance(meta.get("provenance"), dict) else {}
    snap = prov.get("config_snapshot") if isinstance(prov.get("config_snapshot"), dict) else {}
    if not snap and isinstance(meta.get("config_snapshot"), dict):
        snap = meta["config_snapshot"]
    pfs_run = bool(snap.get("per_frame_saturation_enabled", False))
    cfg.per_frame_saturation_enabled = pfs_run
    print(f"PFS_EXPORT_OVERRIDE {cfg.per_frame_saturation_enabled} from snapshot", flush=True)

    at = pd.read_csv(PHOT / "active_targets.csv", dtype={"catalog_id": str}, low_memory=False)
    at["catalog_id"] = at["catalog_id"].astype(str).str.strip()
    trow = at[at["catalog_id"] == BO]
    if trow.empty:
        raise SystemExit("BO CVn not in active_targets")
    trow = trow.iloc[0]
    comp = pd.read_csv(
        PHOT / "comparison_stars_per_target.csv",
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    comp_t = comp[comp["target_catalog_id"].astype(str).str.strip() == BO].copy()
    summary = pd.read_csv(PHOT / "photometry_summary.csv", dtype={"catalog_id": str})
    srow = summary[summary["catalog_id"].astype(str).str.strip() == BO]
    srow = srow.iloc[0] if not srow.empty else pd.Series(dtype=object)

    qmap = None
    cq = LC / f"comp_quality_{BO}.json"
    if cq.is_file():
        raw = json.loads(cq.read_text(encoding="utf-8"))
        qmap = {}
        if isinstance(raw, dict):
            items = raw.get("stars") or raw.get("comps") or raw
            if isinstance(items, dict):
                for k, v in items.items():
                    nk = str(normalize_gaia_source_id(k) or k).strip()
                    q = v.get("quality") if isinstance(v, dict) else v
                    qmap[nk] = str(q or "").strip().lower()

    run_cite = build_run_citation_context(cfg, pipeline_meta=meta, targets_df=at)
    reports_dir = REPORTS
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "aavso").mkdir(exist_ok=True)
    (reports_dir / "varastro").mkdir(exist_ok=True)

    paths = export_all_method_lightcurve_reports(
        reports_dir,
        trow,
        lc_dir=LC,
        target_cid=BO,
        comp_df=comp_t,
        summary_row=srow,
        observer_code=str(cfg.observer_code or ""),
        observer_name=str(cfg.observer_name or "Unknown Observer"),
        comp_quality_map=qmap,
        cfg=cfg,
        obs_group=SETUP,
        targets_df=at,
        run_citation_ctx=run_cite,
    )
    print("PATHS", {k: {kk: str(vv) for kk, vv in v.items()} for k, v in paths.items()}, flush=True)

    aavso = None
    varastro = None
    if "aperture" in paths:
        aavso = paths["aperture"].get("aavso")
        varastro = paths["aperture"].get("varastro")
    if aavso is None:
        cands = list((reports_dir / "aavso").glob("*BO*CVn*")) + list((reports_dir / "aavso").glob("*BO_CVn*"))
        aavso = cands[0] if cands else None
    if varastro is None:
        cands = list((reports_dir / "varastro").glob("*BO*CVn*")) + list((reports_dir / "varastro").glob("*BO_CVn*"))
        varastro = cands[0] if cands else None
    if aavso is None or not Path(aavso).is_file():
        raise SystemExit("AAVSO file missing after export")

    header, rows = _parse_aavso(Path(aavso))
    notes0 = rows[0]["notes"] if rows else ""
    full_ids = [str(x).strip() for x in comp_t["catalog_id"].tolist()]
    trunc = find_truncated_gaia_ids(header + "\n" + notes0, full_ids)
    pfs_line = next((ln for ln in header.splitlines() if "per-frame saturation" in ln.lower()), "")
    pfs_on = pfs_line.rstrip().endswith("ON")
    err_model = next((ln for ln in header.splitlines() if ln.startswith("#ERR_MODEL")), "")
    w_pre = False
    w_post = False
    vtext = Path(varastro).read_text(encoding="utf-8") if varastro and Path(varastro).is_file() else ""
    if "w_pre" in vtext and "w_post" in vtext:
        w_pre = True
        w_post = True

    kname_na = sum(1 for r in rows if str(r["kname"]).strip().lower() in ("na", ""))
    kmag_na = sum(1 for r in rows if str(r["kmag"]).strip().lower() in ("na", ""))

    lc = pd.read_csv(LC / f"lightcurve_{BO}.csv", low_memory=False)
    exp = _select_export_lc_rows(lc)
    exp_mag = pd.to_numeric(exp["mag_calib"], errors="coerce").to_numpy(dtype=float)
    exp_bjd = pd.to_numeric(exp["bjd"], errors="coerce").to_numpy(dtype=float)
    deltas = []
    for r in rows:
        j = int(np.argmin(np.abs(exp_bjd - r["bjd"])))
        if abs(exp_bjd[j] - r["bjd"]) > 1e-6:
            continue
        deltas.append(abs(r["mag"] - exp_mag[j]) * 1000.0)
    max_delta_mmag = float(max(deltas)) if deltas else None

    # Formula chain: mag_calib_ac vs mag_calib_raw + ac_correction
    chain_max = None
    if "mag_calib_raw" in lc.columns and "mag_calib_ac" in lc.columns and "ac_correction" in lc.columns:
        raw = pd.to_numeric(lc["mag_calib_raw"], errors="coerce")
        ac = pd.to_numeric(lc["mag_calib_ac"], errors="coerce")
        corr = pd.to_numeric(lc["ac_correction"], errors="coerce")
        recon = raw + corr
        m = ac.notna() & recon.notna() & np.isfinite(ac) & np.isfinite(recon)
        if bool(m.any()):
            chain_max = float(np.nanmax(np.abs((ac[m] - recon[m]).to_numpy(dtype=float))) * 1e6)
            # convert mag -> 0.0001 mmag units: 1 mag = 1e6 * 0.0001 mmag
            # report in mmag:
            chain_max = float(np.nanmax(np.abs((ac[m] - recon[m]).to_numpy(dtype=float))) * 1000.0)

    out = {
        "aavso": str(aavso),
        "varastro": str(varastro) if varastro else None,
        "n_comp_notes": notes0,
        "truncated_gaia": trunc,
        "pfs_matrix_line": pfs_line,
        "pfs_on": "ON" in pfs_line.upper(),
        "err_model": err_model,
        "w_pre_w_post": bool(w_pre and w_post),
        "kname_na_n": kname_na,
        "kmag_na_n": kmag_na,
        "n_aavso_rows": len(rows),
        "c5_max_abs_delta_export_vs_lc_mmag": max_delta_mmag,
        "c5_n_matched": len(deltas),
        "c5_chain_ac_vs_raw_plus_corr_mmag": chain_max,
        "pfs_from_snapshot": pfs_run,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="ascii")
    print("NOTES", notes0, flush=True)
    print("TRUNC", trunc, flush=True)
    print("PFS", pfs_line, flush=True)
    print("ERR_MODEL", err_model, flush=True)
    print("WPRE_WPOST", w_pre and w_post, flush=True)
    print("KNAME_NA", kname_na, "KMAG_NA", kmag_na, flush=True)
    print("C5_MAX_MMAG", max_delta_mmag, "chain_mmag", chain_max, flush=True)
    print("WROTE", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
