#!/usr/bin/env python3
"""D515-ACCEPT-01B: same-meter check MAD re-read on draft 515 (measurement only)."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))
os.environ.setdefault("VYVAR_PARALLEL_WORKERS", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from photometry_core import run_phase2a  # noqa: E402

RUN_SHA = "da9cce4a5edd1392b8ba842d3c8488589b9d0ac9"
DRAFT515 = ROOT / "Archive" / "Drafts" / "draft_000515"
DRAFT514 = ROOT / "Archive" / "Drafts" / "draft_000514"
SETUP = "NoFilter_60_2"

BO = "1498613634033133184"
FW = "1497343732462852864"
CHK_BO = "1498020894186918144"  # IMPL-05 C / 514 subset meter for BO
CHK_FW = "1497368849430107904"  # IMPL-05 C / 514 subset meter for FW
NEW = "1497613731286514432"  # 515-selected check for both

SUBSET_BO_MAD = 8.594632200000406
SUBSET_FW_MAD = 9.819259800000395

OUT_JSON = ROOT / "dev" / "results" / "D515_ACCEPT_01B_numbers.json"
OUT_MD = ROOT / "dev" / "results" / "CURSOR_RESULT_D515_ACCEPT_01B.md"

MAD_SCALE = 1.4826


def mad_mmag(arr: np.ndarray) -> float | None:
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return None
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)) * MAD_SCALE * 1000.0)


def read_sidecar_mad(path: Path) -> dict:
    if not path.is_file():
        return {"missing": True, "check_scatter_mad_mmag": None, "n_epochs": 0}
    df = pd.read_csv(path)
    cid = str(df["check_catalog_id"].iloc[0]).strip() if "check_catalog_id" in df.columns else None
    k = pd.to_numeric(df["kmag"], errors="coerce").to_numpy()
    return {
        "check_catalog_id": cid,
        "check_scatter_mad_mmag": mad_mmag(k),
        "n_epochs": int(np.isfinite(k).sum()),
        "path": str(path),
    }


def comps_of(comp: pd.DataFrame, tid: str) -> list[str]:
    sub = comp[comp["target_catalog_id"].astype(str).str.strip() == str(tid)]
    return [str(x).strip() for x in sub["catalog_id"].tolist()]


def run_forced_check(
    *,
    draft: Path,
    draft_id: int,
    target_id: str,
    check_id: str,
    label: str,
) -> dict:
    """Phase 2A one target into a temp dir with is_check_star forced."""
    og = draft / "platesolve" / SETUP
    phot = og / "photometry"
    proc = draft / "detrended_aligned" / "lights" / SETUP
    comp_src = phot / "comparison_stars_per_target.csv"
    at_src = phot / "active_targets.csv"
    if not at_src.is_file():
        # restore from full backup names used historically
        for cand in (
            phot / "active_targets_full_before_impl05.csv",
            phot / "active_targets_full_before_impl04.csv",
        ):
            if cand.is_file():
                at_src = cand
                break

    comp = pd.read_csv(comp_src, dtype={"catalog_id": str, "target_catalog_id": str})
    ens = set(comps_of(comp, target_id))
    in_ensemble = check_id in ens
    ens_for_calc = sorted(ens - {check_id}) if in_ensemble else sorted(ens)

    at = pd.read_csv(at_src, dtype={"catalog_id": str})
    at["catalog_id"] = at["catalog_id"].astype(str).str.strip()
    ms = pd.read_csv(og / "masterstars_full_match.csv", dtype={"catalog_id": str}, low_memory=False)
    ms["catalog_id"] = ms["catalog_id"].astype(str).str.strip()

    def _at_row(cid: str) -> pd.DataFrame:
        hit = at[at["catalog_id"] == cid]
        if not hit.empty:
            return hit.iloc[:1].copy()
        hit = ms[ms["catalog_id"] == cid]
        if hit.empty:
            raise SystemExit(f"{label}: {cid} not in active_targets or masterstars")
        return hit.iloc[:1].copy()

    sub_at = _at_row(target_id)
    # PERF-8 flux matrix = active targets + their comps only. A forced check that
    # is not this target's ensemble member is absent unless injected. Prefer a
    # carrier active target (not the check itself): add check as a fake comp of
    # the carrier so it enters the matrix without becoming an active target
    # (activating the check as a target broke resolve_comp_weight_coeffs on 514).
    carrier_id = None
    # Prefer a co-target whose ensemble already contains the check (flux matrix).
    for cid in at["catalog_id"].astype(str).str.strip().tolist():
        if cid and cid not in (target_id, check_id) and check_id in set(comps_of(comp, cid)):
            skip_row = at[at["catalog_id"] == cid]
            if not skip_row.empty:
                sp = skip_row.iloc[0].get("skip_photometry", False)
                if str(sp).strip().lower() in ("1", "true", "yes"):
                    continue
            carrier_id = cid
            break
    if carrier_id is None:
        for cid in at["catalog_id"].astype(str).str.strip().tolist():
            if cid and cid not in (target_id, check_id):
                carrier_id = cid
                break
    if carrier_id is None:
        # fallback: last resort co-target (worked on 515 self-check)
        chk_at = _at_row(check_id)
        sub_at = pd.concat([sub_at, chk_at], ignore_index=True)
    else:
        sub_at = pd.concat([sub_at, _at_row(carrier_id)], ignore_index=True)

    # Force check via is_check_star on the field pool (one row).
    # Do NOT attach the check row to the science target's ensemble (that would
    # put it in target_comps and break weight regression when comp_rms is missing).
    comp2 = comp.copy()
    comp2["is_check_star"] = False
    mask = comp2["catalog_id"].astype(str).str.strip() == check_id
    attach_tid = carrier_id or target_id
    if not bool(mask.any()):
        add = ms[ms["catalog_id"] == check_id].iloc[:1].copy()
        add["target_catalog_id"] = attach_tid
        add["is_check_star"] = True
        if "comp_rms" in add.columns:
            add["comp_rms"] = 0.01
        else:
            add["comp_rms"] = 0.01
        comp2 = pd.concat([comp2, add], ignore_index=True)
    else:
        idx = comp2.index[mask][0]
        comp2.loc[:, "is_check_star"] = False
        comp2.loc[idx, "is_check_star"] = True
        # Keep that row's original target_catalog_id (must not be science target
        # if it would enter the science ensemble; exclude if it is).
        if str(comp2.loc[idx, "target_catalog_id"]).strip() == str(target_id):
            # Move the flag row off the science ensemble onto the carrier.
            if carrier_id is not None:
                moved = comp2.loc[[idx]].copy()
                moved["target_catalog_id"] = carrier_id
                moved["is_check_star"] = True
                comp2 = comp2.drop(index=idx)
                comp2 = pd.concat([comp2, moved], ignore_index=True)
            else:
                comp2 = comp2.drop(index=idx)

    cfg = AppConfig()
    # Measurement only: skip Comp QA / trust (minutes on full pool; not needed for check MAD).
    cfg.comp_qa_enabled = False
    cfg.trust_flag_enabled = False
    db = VyvarDatabase(Path(cfg.database_path))

    with tempfile.TemporaryDirectory(prefix=f"d515_01b_{label}_") as td:
        tdir = Path(td)
        out_phot = tdir / "photometry"
        out_phot.mkdir(parents=True)
        lc_dir = out_phot / "lightcurves"
        lc_dir.mkdir()
        at_path = tdir / "active_targets.csv"
        comp_path = tdir / "comparison_stars_per_target.csv"
        # Comp QA looks under output_dir for comparison_stars_per_target.csv
        sub_at.to_csv(at_path, index=False)
        comp2.to_csv(comp_path, index=False)
        shutil.copy2(comp_path, out_phot / "comparison_stars_per_target.csv")

        # Copy aperture table if present (515 may lack it; fall back to 514).
        for src in (
            phot / "aperture_scatter_table.json",
            draft / "aperture_scatter_table.json",
            DRAFT514 / "aperture_scatter_table.json",
        ):
            if src.is_file():
                shutil.copy2(src, out_phot / "aperture_scatter_table.json")
                break

        t0 = time.perf_counter()

        def _prog(msg: str) -> None:
            print(f"[{time.perf_counter()-t0:7.1f}s] {label}: {msg}", flush=True)

        result = run_phase2a(
            masterstar_fits_path=og / "MASTERSTAR.fits",
            active_targets_csv=at_path,
            comparison_stars_csv=comp_path,
            per_frame_csv_dir=proc,
            detrended_aligned_dir=proc,
            output_dir=out_phot,
            fwhm_px=None,
            cfg=cfg,
            force_aperture_px=None,
            progress_cb=_prog,
            db=db,
            draft_id=draft_id,
        )
        side = read_sidecar_mad(lc_dir / f"check_kmag_{target_id}.csv")
        # copy sidecar out for audit
        audit = ROOT / "tmp" / f"d515_01b_check_kmag_{label}.csv"
        audit.parent.mkdir(parents=True, exist_ok=True)
        src_side = lc_dir / f"check_kmag_{target_id}.csv"
        if src_side.is_file():
            shutil.copy2(src_side, audit)
            side["audit_copy"] = str(audit)
        side.update(
            {
                "label": label,
                "draft_id": draft_id,
                "target_catalog_id": target_id,
                "forced_check_id": check_id,
                "check_was_in_ensemble": in_ensemble,
                "ensemble_ids_used": ens_for_calc,
                "n_comp": len(ens_for_calc),
                "phase2a_result_keys": list(result.keys())[:12] if isinstance(result, dict) else None,
                "elapsed_s": float(time.perf_counter() - t0),
                "quantity": "check_scatter_mad_mmag = 1.4826 * MAD(kmag) * 1000",
                "run_sha_of_archive_photometry": RUN_SHA,
            }
        )
        return side


def reverse_new_on_514(comp515: pd.DataFrame) -> dict:
    """515's NEW check on draft 514 subset data (production path, forced)."""
    phot514 = DRAFT514 / "platesolve" / SETUP / "photometry"
    if not phot514.is_dir():
        return {"skipped": True, "reason": "draft 514 photometry dir missing"}
    # Prefer reading existing sidecars if NEW already selected - it is not.
    # Force via Phase 2A like above.
    out = {}
    for tid, name, chk in ((BO, "BO_CVn", NEW), (FW, "FW_CVn", NEW)):
        try:
            out[name] = run_forced_check(
                draft=DRAFT514,
                draft_id=514,
                target_id=tid,
                check_id=chk,
                label=f"514_{name}_NEW",
            )
        except Exception as exc:  # noqa: BLE001
            out[name] = {"error": str(exc), "skipped": True}
    return out


def main() -> None:
    phot515 = DRAFT515 / "platesolve" / SETUP / "photometry"
    comp515 = pd.read_csv(
        phot515 / "comparison_stars_per_target.csv",
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    bo_ens = comps_of(comp515, BO)
    fw_ens = comps_of(comp515, FW)

    print("Self-check: force NEW check on BO (must ~6.71 mmag)...", flush=True)
    bo_self = run_forced_check(
        draft=DRAFT515, draft_id=515, target_id=BO, check_id=NEW, label="515_BO_NEW_self"
    )
    print(
        "BO_NEW_self",
        bo_self.get("check_scatter_mad_mmag"),
        "prod",
        6.713212799999471,
        flush=True,
    )

    print("Running forced Phase 2A BO with IMPL-05 C check...", flush=True)
    bo_515 = run_forced_check(
        draft=DRAFT515, draft_id=515, target_id=BO, check_id=CHK_BO, label="515_BO"
    )
    print("BO515", bo_515.get("check_scatter_mad_mmag"), bo_515.get("check_catalog_id"), flush=True)

    print("Running forced Phase 2A FW with IMPL-05 C check...", flush=True)
    fw_515 = run_forced_check(
        draft=DRAFT515, draft_id=515, target_id=FW, check_id=CHK_FW, label="515_FW"
    )
    print("FW515", fw_515.get("check_scatter_mad_mmag"), fw_515.get("check_catalog_id"), flush=True)

    # Subset cells from production 514 sidecars (identical meter)
    lc514 = DRAFT514 / "platesolve" / SETUP / "photometry" / "lightcurves"
    bo_514 = read_sidecar_mad(lc514 / f"check_kmag_{BO}.csv")
    fw_514 = read_sidecar_mad(lc514 / f"check_kmag_{FW}.csv")

    # Sanity: subset numbers must match IMPL-05 C
    table = {
        "BO_CVn": {
            "check_id": CHK_BO,
            "subset_IMPL05C": {
                "check_scatter_mad_mmag": bo_514.get("check_scatter_mad_mmag"),
                "n_epochs": bo_514.get("n_epochs"),
                "check_catalog_id_in_sidecar": bo_514.get("check_catalog_id"),
                "reference_IMPL05C_json_mmag": SUBSET_BO_MAD,
            },
            "draft_515": {
                "check_scatter_mad_mmag": bo_515.get("check_scatter_mad_mmag"),
                "n_epochs": bo_515.get("n_epochs"),
                "check_catalog_id_in_sidecar": bo_515.get("check_catalog_id"),
                "ensemble_ids": bo_515.get("ensemble_ids_used"),
                "n_comp": bo_515.get("n_comp"),
                "check_was_in_ensemble": bo_515.get("check_was_in_ensemble"),
                "elapsed_s": bo_515.get("elapsed_s"),
            },
        },
        "FW_CVn": {
            "check_id": CHK_FW,
            "subset_IMPL05C": {
                "check_scatter_mad_mmag": fw_514.get("check_scatter_mad_mmag"),
                "n_epochs": fw_514.get("n_epochs"),
                "check_catalog_id_in_sidecar": fw_514.get("check_catalog_id"),
                "reference_IMPL05C_json_mmag": SUBSET_FW_MAD,
            },
            "draft_515": {
                "check_scatter_mad_mmag": fw_515.get("check_scatter_mad_mmag"),
                "n_epochs": fw_515.get("n_epochs"),
                "check_catalog_id_in_sidecar": fw_515.get("check_catalog_id"),
                "ensemble_ids": fw_515.get("ensemble_ids_used"),
                "n_comp": fw_515.get("n_comp"),
                "check_was_in_ensemble": fw_515.get("check_was_in_ensemble"),
                "elapsed_s": fw_515.get("elapsed_s"),
            },
        },
    }

    def delta(row: dict) -> float | None:
        a = row["draft_515"].get("check_scatter_mad_mmag")
        b = row["subset_IMPL05C"].get("check_scatter_mad_mmag")
        if a is None or b is None:
            return None
        return float(a) - float(b)

    d_bo = delta(table["BO_CVn"])
    d_fw = delta(table["FW_CVn"])

    # Pre-registered interpretation
    def row_verdict(d: float | None) -> str:
        if d is None:
            return "missing"
        if d <= 1.0:
            return "ok_le_subset_plus_1"
        if d > 2.0:
            return "degraded_gt_2"
        return "inconclusive_between_1_and_2"

    v_bo = row_verdict(d_bo)
    v_fw = row_verdict(d_fw)
    if v_bo == "degraded_gt_2" or v_fw == "degraded_gt_2":
        overall = "RETRACT_D4"
    elif v_bo == "ok_le_subset_plus_1" and v_fw == "ok_le_subset_plus_1":
        overall = "CONFIRMED_IDENTICAL_METER"
    elif "missing" in (v_bo, v_fw):
        overall = "INCOMPLETE"
    else:
        overall = "INCONCLUSIVE"

    print("Reverse NEW check on 514...", flush=True)
    reverse = reverse_new_on_514(comp515)

    # Also note: CHK_BO is an FW ensemble member on 515 (not used as FW check here)
    notes = {
        "CHK_BO_in_FW_515_ensemble": CHK_BO in set(fw_ens),
        "CHK_FW_in_BO_515_ensemble": CHK_FW in set(bo_ens),
        "CHK_FW_in_FW_515_ensemble": CHK_FW in set(fw_ens),
        "CHK_BO_in_BO_515_ensemble": CHK_BO in set(bo_ens),
        "NEW_in_BO_515_ensemble": NEW in set(bo_ens),
        "NEW_in_FW_515_ensemble": NEW in set(fw_ens),
    }

    payload = {
        "task": "D515-ACCEPT-01B",
        "run_sha_archive_515": RUN_SHA,
        "quantity": "check_scatter_mad_mmag [mmag] = 1.4826 * MAD(kmag) * 1000; "
        "kmag = check star differenced against the TARGET ensemble (production check_kmag)",
        "table_2x2": table,
        "deltas_515_minus_subset_mmag": {"BO_CVn": d_bo, "FW_CVn": d_fw},
        "row_verdicts": {"BO_CVn": v_bo, "FW_CVn": v_fw},
        "overall_interpretation": overall,
        "ensemble_notes": notes,
        "reverse_NEW_check_on_514": reverse,
        "production_515_NEW_meter_for_reference": {
            "check_id": NEW,
            "BO_CVn": read_sidecar_mad(
                phot515 / "lightcurves" / f"check_kmag_{BO}.csv"
            ),
            "FW_CVn": read_sidecar_mad(
                phot515 / "lightcurves" / f"check_kmag_{FW}.csv"
            ),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("WROTE", OUT_JSON, flush=True)
    print("OVERALL", overall, "d_bo", d_bo, "d_fw", d_fw, flush=True)


if __name__ == "__main__":
    main()
