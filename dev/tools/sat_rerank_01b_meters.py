"""SAT-RERANK-01B meters: ensembles, COMP log, n_eff, B2, BIN-8-9. Does not overwrite 01/01B."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dev" / "tools"))

from d515_accept_01_measure import (  # noqa: E402
    load_inst_mags,
    weighted_loo_mad_mmag,
)

DRAFT = ROOT / "Archive" / "Drafts" / "draft_000515"
SETUP = "NoFilter_60_2"
PHOT = DRAFT / "platesolve" / SETUP / "photometry"
LOG = ROOT / "tmp" / "draft_515_pfs_semantics_01.log"
SAT_JSON = ROOT / "dev" / "results" / "SAT_LIMIT_01_summary.json"
OUT = ROOT / "dev" / "results" / "SAT_RERANK_01B_meters.json"

BO = "1498613634033133184"
FW = "1497343732462852864"
CHK_BO = "1498020894186918144"
CHK_FW = "1497368849430107904"
GATE_BIN89_MMAG = 11.988543441702657
GATE_BIN89_N = 15
DA9_BO_MAD = 7.0498
DA9_FW_MAD = 10.6836
MAD_SCALE = 1.4826


def n_eff(weights: list[float]) -> float | None:
    w = np.asarray([float(x) for x in weights if math.isfinite(float(x)) and float(x) > 0], dtype=float)
    if w.size == 0:
        return None
    s = float(w.sum())
    s2 = float(np.square(w).sum())
    if s2 <= 0:
        return None
    return (s * s) / s2


def mad_mmag(arr: np.ndarray) -> float | None:
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return None
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)) * MAD_SCALE * 1000.0)


def ensemble(comp: pd.DataFrame, tid: str) -> pd.DataFrame:
    sub = comp[comp["target_catalog_id"].astype(str).str.strip() == tid].copy()
    return sub


def parse_comp_block(text: str, tid: str) -> dict:
    begin = f"[COMP] target={tid} rms-then-color begin"
    end = f"[COMP] target={tid} rms-then-color end"
    lines = text.splitlines()
    i0 = next((i for i, ln in enumerate(lines) if begin in ln), None)
    i1 = next((i for i, ln in enumerate(lines) if end in ln), None)
    block = lines[i0 : (i1 + 1 if i1 is not None else None)] if i0 is not None else []
    out: dict = {"begin_line": i0, "end_line": i1, "n_block_lines": len(block), "lines": block}
    for ln in block:
        if "clean pool n=" in ln:
            out["clean_pool_line"] = ln.strip()
        if "best not-admitted" in ln:
            out["best_not_admitted_line"] = ln.strip()
        if "nothing better in the pool" in ln:
            out["nothing_better_line"] = ln.strip()
        if "best ceiling-rejected" in ln:
            out["best_ceiling_line"] = ln.strip()
        if "best isolation-rejected" in ln:
            out["best_isolation_line"] = ln.strip()
        if "best colour-rejected" in ln:
            out["best_colour_line"] = ln.strip()
        if "max_comp_rms ceiling=" in ln:
            out["ceiling_line"] = ln.strip()
        if "single-source isolation" in ln:
            out["isolation_line"] = ln.strip()
        if "color filter" in ln:
            out["colour_line"] = ln.strip()
    return out


def bin_loo() -> dict:
    ms = pd.read_csv(
        DRAFT / "platesolve" / SETUP / "masterstars_full_match.csv",
        dtype={"catalog_id": str},
        low_memory=False,
    )
    gcol = "phot_g_mean_mag" if "phot_g_mean_mag" in ms.columns else "mag"
    ms["catalog_id"] = ms["catalog_id"].astype(str).str.strip()
    ms["_g"] = pd.to_numeric(ms[gcol], errors="coerce")
    bins = [(8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15)]
    by_bin: dict[str, list[str]] = {f"{a}-{b}": [] for a, b in bins}
    for _, r in ms.iterrows():
        g = float(r["_g"])
        cid = str(r["catalog_id"])
        if not math.isfinite(g):
            continue
        for a, b in bins:
            if a <= g < b:
                by_bin[f"{a}-{b}"].append(cid)
                break
    sample_ids: list[str] = []
    for ids in by_bin.values():
        sample_ids.extend(ids[:15])
    sample_ids = sorted(set(sample_ids))
    mag = load_inst_mags(sample_ids)
    keep = [c for c in sample_ids if c in mag and float(np.isfinite(mag[c]).mean()) >= 0.5]
    mag = {c: mag[c] for c in keep}
    e_rows = {}
    for a, b in bins:
        key = f"{a}-{b}"
        vals = []
        cands = [c for c in by_bin[key] if c in mag]
        for cid in cands[:15]:
            mad = weighted_loo_mad_mmag(cid, keep, mag, weights=None)
            if mad is not None and mad > 0:
                vals.append(mad)
        e_rows[key] = {
            "n": len(vals),
            "n_candidates_in_bin": len(by_bin[key]),
            "median_loo_scatter_mmag": float(np.median(vals)) if vals else None,
            "min_mmag": float(min(vals)) if vals else None,
            "max_mmag": float(max(vals)) if vals else None,
        }
    return e_rows


def main() -> int:
    comp = pd.read_csv(
        PHOT / "comparison_stars_per_target.csv",
        dtype={"catalog_id": str, "target_catalog_id": str},
        low_memory=False,
    )
    sat_ids = set(
        str(x).strip()
        for x in json.loads(SAT_JSON.read_text(encoding="utf-8"))["b4"]["reclassify"]["saturated_catalog_ids"]
    )
    text = LOG.read_text(encoding="utf-8") if LOG.is_file() else ""
    bo = ensemble(comp, BO)
    fw = ensemble(comp, FW)
    bo_ids = [str(x).strip() for x in bo["catalog_id"].tolist()]
    fw_ids = [str(x).strip() for x in fw["catalog_id"].tolist()]
    wcol = "comp_weight" if "comp_weight" in bo.columns else None
    bo_w = [float(pd.to_numeric(x, errors="coerce")) for x in bo[wcol].tolist()] if wcol else []
    fw_w = [float(pd.to_numeric(x, errors="coerce")) for x in fw[wcol].tolist()] if wcol else []

    chk_fw_in_fw = CHK_FW in set(fw_ids)
    chk_fw_in_bo = CHK_FW in set(bo_ids)
    chk_bo_in_bo = CHK_BO in set(bo_ids)
    chk_bo_in_fw = CHK_BO in set(fw_ids)

    sat_in = sorted(sat_ids & set(comp["catalog_id"].astype(str).str.strip()))

    prod_bo = None
    prod_fw = None
    side_bo = PHOT / "lightcurves" / f"check_kmag_{BO}.csv"
    side_fw = PHOT / "lightcurves" / f"check_kmag_{FW}.csv"
    if side_bo.is_file():
        k = pd.to_numeric(pd.read_csv(side_bo)["kmag"], errors="coerce").to_numpy()
        prod_bo = {"mad_mmag": mad_mmag(k), "n_epochs": int(np.isfinite(k).sum()), "check_id": str(pd.read_csv(side_bo)["check_catalog_id"].iloc[0]) if "check_catalog_id" in pd.read_csv(side_bo).columns else None}
    if side_fw.is_file():
        dfw = pd.read_csv(side_fw)
        k = pd.to_numeric(dfw["kmag"], errors="coerce").to_numpy()
        prod_fw = {
            "mad_mmag": mad_mmag(k),
            "n_epochs": int(np.isfinite(k).sum()),
            "check_id": str(dfw["check_catalog_id"].iloc[0]) if "check_catalog_id" in dfw.columns else None,
        }

    print("Computing BIN LOO (proc CSVs)...", flush=True)
    e_rows = bin_loo()
    b89 = e_rows.get("8-9", {})
    b89_mad = b89.get("median_loo_scatter_mmag")
    if b89_mad is None:
        verdict = "OPEN"
    elif abs(float(b89_mad) - GATE_BIN89_MMAG) < 0.5 and int(b89.get("n") or 0) == GATE_BIN89_N:
        verdict = "OPEN"
    elif float(b89_mad) < GATE_BIN89_MMAG - 1.0:
        verdict = "CLOSED" if float(b89_mad) <= 8.5 else "DOWNGRADED"
    elif float(b89_mad) > GATE_BIN89_MMAG + 0.5:
        verdict = "OPEN"
    else:
        verdict = "OPEN"

    out = {
        "bo": {
            "ids": bo_ids,
            "comp_weight": bo_w,
            "n_eff": n_eff(bo_w),
            "n_comp": len(bo_ids),
        },
        "fw": {
            "ids": fw_ids,
            "comp_weight": fw_w,
            "n_eff": n_eff(fw_w),
            "n_comp": len(fw_ids),
        },
        "chk_fw_in_fw_ensemble": chk_fw_in_fw,
        "chk_fw_in_bo_ensemble": chk_fw_in_bo,
        "chk_bo_in_bo_ensemble": chk_bo_in_bo,
        "chk_bo_in_fw_ensemble": chk_bo_in_fw,
        "stop_fw_meter_consumed": bool(chk_fw_in_fw),
        "b2_sat_in_ensemble": sat_in,
        "comp_log_bo": parse_comp_block(text, BO),
        "production_check_sidecar": {"BO": prod_bo, "FW": prod_fw},
        "da9cce4_fixed_meter": {"BO": DA9_BO_MAD, "FW": DA9_FW_MAD, "n_epochs": 134},
        "bin_loo": e_rows,
        "bin_8_9": {
            "median_loo_mmag": b89_mad,
            "n": b89.get("n"),
            "gate_mmag": GATE_BIN89_MMAG,
            "gate_n": GATE_BIN89_N,
            "verdict": verdict,
        },
    }
    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="ascii")
    print("BO ids", bo_ids, "n_eff", n_eff(bo_w), flush=True)
    print("FW ids", fw_ids, "n_eff", n_eff(fw_w), flush=True)
    print("CHK_FW in FW ensemble", chk_fw_in_fw, "STOP" if chk_fw_in_fw else "ok", flush=True)
    print("B2 sat in ens", len(sat_in), flush=True)
    print("BIN-8-9", b89_mad, "n", b89.get("n"), verdict, flush=True)
    print("COMP clean", out["comp_log_bo"].get("clean_pool_line"), flush=True)
    print("WROTE", OUT, flush=True)
    return 1 if chk_fw_in_fw else 0


if __name__ == "__main__":
    raise SystemExit(main())
