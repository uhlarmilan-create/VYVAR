#!/usr/bin/env python3
"""D515-ACCEPT-01: measure draft 515 acceptance (read-only Archive)."""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from comp_qa_core import loo_diff_series  # noqa: E402

DRAFT = ROOT / "Archive" / "Drafts" / "draft_000515"
SETUP = "NoFilter_60_2"
PHOT = DRAFT / "platesolve" / SETUP / "photometry"
LC = PHOT / "lightcurves"
PROC = DRAFT / "detrended_aligned" / "lights" / SETUP
LOG = ROOT / "tmp" / "draft_515_headless_phase012a.log"
OUT = ROOT / "dev" / "results"
RUN_SHA = "da9cce4a5edd1392b8ba842d3c8488589b9d0ac9"

BO = "1498613634033133184"
FW = "1497343732462852864"
CHECK_METER_NAMED = "1497145751650265600"
IMPL05C_BO_CHECK = "1498020894186918144"
IMPL05C_FW_CHECK = "1497368849430107904"
IMPL05C_BO_COMPS = {
    "1497368849430107904",
    "1497974027502858240",
    "1500748301498613248",
    "1497771992240531712",
    "1499200223486564608",
}
IMPL05C_FW_COMPS = {
    "1498020894186918144",
    "1497196054307837696",
    "1497442379271632384",
    "1497997117247042816",
    "1497313255374892800",
    "1497631048594737408",
    "1499066907701715456",
    "1498626793812916480",
}

MAD_SCALE = 1.4826


def _mad_mmag(arr: np.ndarray) -> float | None:
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return None
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)) * MAD_SCALE * 1000.0)


def _pct(a: list[float], p: float) -> float | None:
    if not a:
        return None
    return float(np.percentile(np.asarray(a, dtype=float), p))


def _read_log_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] == b"\xff\xfe" or raw[:2] == b"\xfe\xff":
        return raw.decode("utf-16")
    return raw.decode("utf-8", errors="replace")


def parse_log(path: Path) -> dict:
    text = _read_log_text(path)
    # strip powershell noise lines that wrap warnings
    lines = text.splitlines()
    t_re = re.compile(r"^\[\s*([0-9.]+)s\]\s+(.*)$")
    events: list[tuple[float, str]] = []
    for ln in lines:
        m = t_re.match(ln.strip())
        if m:
            events.append((float(m.group(1)), m.group(2)))
    p1 = [(t, m) for t, m in events if m.startswith("Phase 1: target")]
    # Only true per-ciel lines (exclude ProcFrameStore / FWHM / aperture headers).
    p2 = [(t, m) for t, m in events if "Faza 2A: ciel" in m]
    t0 = next((t for t, m in events if "Faza 0:" in m or "Faza 0 " in m), None)
    t0_done = next((t for t, m in events if "Faza 0 hotova" in m), None)
    t1_start = next((t for t, m in events if "Faza 1:" in m and "ProcFrameStore" in m), None)
    t1_end = p1[-1][0] if p1 else None
    # Phase 2A wall starts at aperture-photometry header (before ciel loop).
    t2_start = None
    for t, m in events:
        if m.startswith("Faza 2A:") and "aperture" in m.lower():
            t2_start = t
            break
    if t2_start is None:
        for t, m in events:
            if "Faza 2A: ciel" in m:
                t2_start = t
                break
    # find after phase1 completion markers
    t_phase1_done = None
    for t, m in events:
        if "comparison_stars" in m.lower() or "Faza 1 hotov" in m or "suspected" in m.lower():
            pass
    # Use last Phase 1 status as end of selection loop; Comp QA is after 2A photometry
    elapsed = None
    for ln in lines:
        if ln.startswith("ELAPSED_S"):
            try:
                elapsed = float(ln.split()[1])
            except (IndexError, ValueError):
                pass

    def per_target_deltas(pts: list[tuple[float, str]]) -> list[dict]:
        out = []
        for i, (t, msg) in enumerate(pts):
            name = msg.split(":", 2)[-1].strip() if ":" in msg else msg
            # Phase 1: "Phase 1: target N/97: NAME"
            m = re.search(r"target\s+(\d+)/(\d+):\s*(.+)$", msg)
            if m:
                idx, tot, name = int(m.group(1)), int(m.group(2)), m.group(3).strip()
            else:
                m2 = re.search(r"ciel\s+(\d+)/(\d+):\s*(.+)$", msg)
                if m2:
                    idx, tot, name = int(m2.group(1)), int(m2.group(2)), m2.group(3).strip()
                else:
                    idx, tot = i + 1, len(pts)
            t_next = pts[i + 1][0] if i + 1 < len(pts) else None
            dt = (t_next - t) if t_next is not None else None
            out.append(
                {
                    "index": idx,
                    "total": tot,
                    "name": name,
                    "t_s": t,
                    "dt_s": dt,
                }
            )
        return out

    p1_rows = per_target_deltas(p1)
    p2_rows = per_target_deltas(p2)
    # last p1 dt: to t2_start if available
    if p1_rows and t2_start is not None and p1_rows[-1]["dt_s"] is None:
        p1_rows[-1]["dt_s"] = float(t2_start) - float(p1_rows[-1]["t_s"])

    def summarize(rows: list[dict]) -> dict:
        dts = [float(r["dt_s"]) for r in rows if r.get("dt_s") is not None and r["dt_s"] >= 0]
        slow = sorted(
            [r for r in rows if r.get("dt_s") is not None],
            key=lambda r: -float(r["dt_s"]),
        )[:5]
        return {
            "n_intervals": len(dts),
            "min_s": float(min(dts)) if dts else None,
            "median_s": float(np.median(dts)) if dts else None,
            "p90_s": _pct(dts, 90),
            "max_s": float(max(dts)) if dts else None,
            "five_slowest": [
                {"name": s["name"], "index": s["index"], "dt_s": s["dt_s"]} for s in slow
            ],
        }

    phase0_s = None
    if t0 is not None and t0_done is not None:
        phase0_s = float(t0_done) - float(t0)
    elif t0_done is not None and t1_start is not None:
        phase0_s = float(t0_done)  # from run start approx
    phase1_s = None
    if t1_start is not None and t2_start is not None:
        phase1_s = float(t2_start) - float(t1_start)
    phase2a_s = None
    if t2_start is not None and elapsed is not None:
        phase2a_s = float(elapsed) - float(t2_start)

    return {
        "elapsed_s": elapsed,
        "phase0_wall_s": phase0_s,
        "phase1_wall_s": phase1_s,
        "phase2a_wall_s": phase2a_s,
        "t0_s": t0,
        "t0_done_s": t0_done,
        "t1_start_s": t1_start,
        "t2_start_s": t2_start,
        "phase1_per_target": summarize(p1_rows),
        "phase2a_per_target": summarize(p2_rows),
        "n_phase1_status_lines": len(p1),
        "n_phase2a_status_lines": len(p2),
        "events_sample": events[:8] + events[-8:],
        "log_caveat_phase1": (
            "Phase 1 status was logged sparsely (every ~8 targets after RUN-HARDEN-01); "
            "B2 Phase 1 stats are inter-checkpoint deltas, not true per-target walls."
        ),
        "log_caveat_phase2a": (
            "Phase 2A ciel status is also sparse (~every 18 of 218); photometry of the "
            "49 LC targets is short; Comp QA is timed separately."
        ),
    }


def empty_dao_rate() -> dict:
    procs = sorted(PROC.glob("proc_*.csv"))
    empty = []
    for p in procs:
        df = pd.read_csv(p, nrows=5)
        cols = {c.lower() for c in df.columns}
        has_peak = "peak_max_adu" in cols or "dao_flux" in cols
        # forced-only signature: catalog rows present but no peak/dao_flux columns at all
        if not has_peak:
            # confirm full file
            df2 = pd.read_csv(p)
            cols2 = {c.lower() for c in df2.columns}
            if "peak_max_adu" not in cols2 and "dao_flux" not in cols2:
                empty.append(p.name)
    return {
        "n_proc_csv": len(procs),
        "n_empty_dao_forced_only": len(empty),
        "rate": (len(empty) / len(procs)) if procs else None,
        "frames": empty,
        "setup": SETUP,
        "definition": "proc CSV lacks peak_max_adu and dao_flux columns (EMPTY-DAO-01 forced-only signature)",
    }


def check_mad_for_target(tid: str) -> dict:
    chk = LC / f"check_kmag_{tid}.csv"
    out: dict = {"target_catalog_id": tid, "check_kmag_path": str(chk.name)}
    if not chk.is_file():
        out["check_scatter_mad_mmag"] = None
        out["missing"] = True
        return out
    df = pd.read_csv(chk)
    cid = None
    for col in ("check_catalog_id", "catalog_id"):
        if col in df.columns:
            cid = str(df[col].iloc[0]).strip()
            break
    out["check_catalog_id"] = cid
    if "kmag" in df.columns:
        out["check_scatter_mad_mmag"] = _mad_mmag(pd.to_numeric(df["kmag"], errors="coerce").to_numpy())
        out["check_scatter_std_mmag"] = float(
            np.nanstd(pd.to_numeric(df["kmag"], errors="coerce").to_numpy(), ddof=1) * 1000.0
        )
        out["n_epochs"] = int(pd.to_numeric(df["kmag"], errors="coerce").notna().sum())
    return out


def comps_of(comp: pd.DataFrame, tid: str) -> list[str]:
    sub = comp[comp["target_catalog_id"].astype(str).str.strip() == tid]
    return [str(x).strip() for x in sub["catalog_id"].tolist()]


def load_inst_mags(cids: list[str]) -> dict[str, np.ndarray]:
    """Build per-star instrumental mag series from proc CSVs (frame-aligned)."""
    procs = sorted(PROC.glob("proc_*.csv"))
    want = set(cids)
    series: dict[str, list[float]] = {c: [] for c in cids}
    for p in procs:
        df = pd.read_csv(p, dtype={"catalog_id": str}, low_memory=False)
        df["catalog_id"] = df["catalog_id"].astype(str).str.strip()
        # prefer mag_inst / mag
        mag_col = None
        # Prefer measured flux -> instrumental mag. Column 'mag' on proc CSVs is
        # often catalog G (constant) and must NOT be used for LOO scatter.
        if "flux" in df.columns:
            flux = pd.to_numeric(df["flux"], errors="coerce")
            df = df.assign(_m_inst=-2.5 * np.log10(np.where(flux > 0, flux, np.nan)))
            mag_col = "_m_inst"
        elif "dao_flux" in df.columns:
            flux = pd.to_numeric(df["dao_flux"], errors="coerce")
            df = df.assign(_m_inst=-2.5 * np.log10(np.where(flux > 0, flux, np.nan)))
            mag_col = "_m_inst"
        else:
            for c in ("mag_inst", "mag_dao"):
                if c in df.columns:
                    mag_col = c
                    break
        if mag_col is None:
            for c in cids:
                series[c].append(float("nan"))
            continue
        by = {
            str(r["catalog_id"]): float(pd.to_numeric(r[mag_col], errors="coerce"))
            for _, r in df.iterrows()
            if str(r["catalog_id"]) in want
        }
        for c in cids:
            series[c].append(by.get(c, float("nan")))
    return {c: np.asarray(v, dtype=float) for c, v in series.items()}


def weighted_loo_mad_mmag(
    focus: str,
    peer_ids: list[str],
    mag: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> float | None:
    peers = [p for p in peer_ids if p != focus and p in mag]
    if len(peers) < 2 or focus not in mag:
        return None
    n = len(mag[focus])
    out = np.full(n, np.nan)
    mf = mag[focus]
    for i in range(n):
        if not math.isfinite(mf[i]):
            continue
        num = den = 0.0
        for p in peers:
            mv = mag[p][i]
            w = 1.0 if weights is None else float(weights.get(p, 0.0))
            if math.isfinite(mv) and math.isfinite(w) and w > 0:
                num += w * mv
                den += w
        if den > 0:
            out[i] = mf[i] - num / den
    fin = np.isfinite(out)
    if not fin.any():
        return None
    out[fin] = out[fin] - float(np.median(out[fin]))
    return _mad_mmag(out)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    at = pd.read_csv(PHOT / "active_targets.csv", dtype={"catalog_id": str})
    comp = pd.read_csv(PHOT / "comparison_stars_per_target.csv", dtype={"catalog_id": str, "target_catalog_id": str})
    summary = pd.read_csv(PHOT / "photometry_summary.csv", dtype={"catalog_id": str})

    phase1 = at[at["skip_reason"].astype(str) != "vsx_type_out_of_scope"].copy()
    skip_hist = (
        phase1["skip_reason"].fillna("(empty)").astype(str).value_counts().to_dict()
    )
    # also zone for empty skip
    zone_hist = phase1["zone_flag"].fillna("(empty)").astype(str).value_counts().to_dict()
    lc_files = sorted(LC.glob("lightcurve_*.csv")) if LC.is_dir() else []
    lc_ids = {p.stem.replace("lightcurve_", "") for p in lc_files}

    phase1_ids = set(phase1["catalog_id"].astype(str).str.strip())
    missing_lc = []
    for _, row in phase1.iterrows():
        cid = str(row["catalog_id"]).strip()
        if cid in lc_ids:
            continue
        reason = str(row.get("skip_reason") or "").strip()
        if not reason or reason == "nan":
            # fall back to skip_photometry / zone
            if bool(row.get("skip_photometry")):
                z = str(row.get("zone_flag") or "")
                reason = f"skip_photometry:zone={z}"
            else:
                reason = "NO_RECORDED_REASON"
        missing_lc.append({"catalog_id": cid, "vsx_name": str(row.get("vsx_name") or ""), "reason": reason})

    reason_hist = Counter(m["reason"] for m in missing_lc)
    silent = [m for m in missing_lc if m["reason"] == "NO_RECORDED_REASON"]

    # qa_degraded
    qa_deg = []
    meta_path = PHOT / "pipeline_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    if "qa_degraded" in summary.columns:
        qa_deg = summary.loc[summary["qa_degraded"].astype(str).str.lower().isin(["true", "1"]), "catalog_id"].astype(str).tolist()
    # search meta
    for k, v in meta.items():
        if "qa_degraded" in str(k).lower():
            qa_deg.append({"meta_key": k, "value": v})

    part_a = {
        "run_sha": RUN_SHA,
        "exit_status": 0,
        "ended_in_phase": "Phase 2A complete (post Comp QA + trust)",
        "frames_aligned_fits": len(list(PROC.glob("*.fits"))),
        "frames_photometered_proc_csv": len(list(PROC.glob("proc_*.csv"))),
        "phase1_targets_total": int(len(phase1)),
        "lc_files_written": int(len(lc_files)),
        "photometry_targets_completed": int(len(lc_ids)),
        "skip_reason_histogram_phase1": skip_hist,
        "zone_flag_histogram_phase1": zone_hist,
        "gap_97_to_49": {
            "phase1_n": int(len(phase1)),
            "lc_n": int(len(lc_ids)),
            "missing_lc_n": int(len(missing_lc)),
            "missing_reason_histogram": dict(reason_hist),
            "silent_missing_n": int(len(silent)),
            "silent_missing": silent,
        },
        "qa_degraded": qa_deg,
    }

    part_b = parse_log(LOG)
    # dominance sentences from structure
    part_b["dominance"] = {
        "phase0": "Log shows brief VSX/active-target selection (~seconds); not resolvable further.",
        "phase1": (
            "Dominated by per-target select_comparison_stars_per_target / "
            "_accumulate_per_frame_comp_metrics (long gaps between Phase 1 status lines)."
            if part_b.get("phase1_per_target", {}).get("median_s")
            else "Log does not resolve sub-step; py-spy earlier showed per-frame comp metrics."
        ),
        "phase2a": (
            "Per-target photometry is fast in the log; Comp QA block is a large wall chunk "
            f"(from last Faza 2A status to Trust flag ~{None})."
        ),
    }
    # refine Comp QA timing from log events
    text = _read_log_text(LOG)
    t_comp_qa = None
    t_trust = None
    for ln in text.splitlines():
        m = re.match(r"^\[\s*([0-9.]+)s\]\s+(.*)$", ln.strip())
        if not m:
            continue
        t, msg = float(m.group(1)), m.group(2)
        if "Comp QA" in msg:
            t_comp_qa = t
        if "Trust flag" in msg:
            t_trust = t
    if t_comp_qa is not None and t_trust is not None:
        part_b["comp_qa_wall_s"] = float(t_trust) - float(t_comp_qa)
        part_b["dominance"]["phase2a"] = (
            f"Aperture photometry over targets is short in the log; Comp QA dominates "
            f"late Phase 2A wall ({part_b['comp_qa_wall_s']:.1f} s Comp QA -> Trust)."
        )

    part_c = empty_dao_rate()

    # Part D
    bo_meter = check_mad_for_target(BO)
    fw_meter = check_mad_for_target(FW)
    bo_comps = comps_of(comp, BO)
    fw_comps = comps_of(comp, FW)
    # named check star presence
    named_as_bo_check = bo_meter.get("check_catalog_id") == CHECK_METER_NAMED
    # distribution over LC targets: prefer check MAD else lc_rms from summary
    dist_rows = []
    for p in lc_files:
        tid = p.stem.replace("lightcurve_", "")
        mad = check_mad_for_target(tid).get("check_scatter_mad_mmag")
        name = ""
        hit = at[at["catalog_id"].astype(str) == tid]
        if not hit.empty:
            name = str(hit.iloc[0].get("vsx_name") or hit.iloc[0].get("name") or "")
        if mad is None:
            # fallback lc_rms from summary (mag)
            srow = summary[summary["catalog_id"].astype(str) == tid]
            if not srow.empty and "lc_rms" in srow.columns:
                v = float(pd.to_numeric(srow.iloc[0]["lc_rms"], errors="coerce"))
                mad = v * 1000.0 if math.isfinite(v) else None
                metric = "lc_rms_mmag_from_summary"
            else:
                metric = "missing"
        else:
            metric = "check_kmag_MAD_mmag"
        if mad is not None and math.isfinite(mad):
            dist_rows.append({"catalog_id": tid, "name": name, "scatter_mmag": mad, "metric": metric})

    scat = [r["scatter_mmag"] for r in dist_rows]
    med = float(np.median(scat)) if scat else None
    tail = [r for r in dist_rows if med and r["scatter_mmag"] > 3.0 * med]

    part_d = {
        "run_sha": RUN_SHA,
        "science_identity_4fe84b4_to_da9cce4_src_py": "empty (verified separately)",
        "BO_CVn": {
            **bo_meter,
            "n_comp": len(bo_comps),
            "comp_ids": bo_comps,
            "set_equal_impl05c": set(bo_comps) == IMPL05C_BO_COMPS,
            "impl05c_subset_check_mad_mmag": 8.594632200000406,
            "impl05c_check_id": IMPL05C_BO_CHECK,
            "named_meter_check_id_149714_is_selected": named_as_bo_check,
        },
        "FW_CVn": {
            **fw_meter,
            "n_comp": len(fw_comps),
            "comp_ids": fw_comps,
            "set_equal_impl05c": set(fw_comps) == IMPL05C_FW_COMPS,
            "impl05c_subset_check_mad_mmag": 9.819259800000395,
            "impl05c_check_id": IMPL05C_FW_CHECK,
        },
        "distribution_all_lc_targets": {
            "n": len(scat),
            "min_mmag": float(min(scat)) if scat else None,
            "median_mmag": med,
            "p90_mmag": _pct(scat, 90),
            "max_mmag": float(max(scat)) if scat else None,
            "gt_3x_median": tail,
            "metric_note": "Prefer check_kmag MAD (1.4826*MAD*1000); else summary lc_rms*1000",
        },
        "spec_note_check_star": (
            "Task names check 1497145751650265600; IMPL-05 C fixed meter used "
            f"BO check {IMPL05C_BO_CHECK} / FW check {IMPL05C_FW_CHECK}. "
            "D1 reports production check_kmag MAD for BO/FW on 515."
        ),
    }

    # Part E: LOO by mag bin on field stars present in proc (not only selected comps).
    # Selected comps under RMS-first are bright-biased; G14-15 needs masterstars.
    print("Loading proc mags for LOO (may take a minute)...", flush=True)
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
    for k, ids in by_bin.items():
        # prefer isolated-ish: take first 15 ids (catalog order); LOO needs peers in sample
        sample_ids.extend(ids[:15])
    sample_ids = sorted(set(sample_ids))
    peer_pool = sample_ids[:]
    mag = load_inst_mags(peer_pool)
    # drop stars with <50% finite frames
    keep = []
    for cid in peer_pool:
        fin = np.isfinite(mag[cid]).mean() if cid in mag else 0.0
        if fin >= 0.5:
            keep.append(cid)
    peer_pool = keep
    mag = {c: mag[c] for c in peer_pool}
    e_rows = {}
    for a, b in bins:
        key = f"{a}-{b}"
        vals = []
        cands = [c for c in by_bin[key] if c in mag]
        for cid in cands[:15]:
            peers = [p for p in peer_pool if p != cid]
            mad = weighted_loo_mad_mmag(cid, peers, mag, weights=None)
            if mad is not None and mad > 0:
                vals.append(mad)
        e_rows[key] = {
            "n": len(vals),
            "n_candidates_in_bin": len(by_bin[key]),
            "n_with_proc": len(cands),
            "median_loo_scatter_mmag": float(np.median(vals)) if vals else None,
            "values_mmag": vals,
        }
    part_e = {
        "estimator": "1.4826*MAD of focus - equal-weight mean peer inst mags (PRE-IMPL Q2-style; not flux-sum loo_diff_series)",
        "run_sha": RUN_SHA,
        "bins": e_rows,
        "BIN_8_9": {
            "subset_before_r95_mmag": 7.786723789851446,
            "subset_after_permag_mmag": 12.348001171995847,
            "full_515_median_loo_mmag": e_rows.get("8-9", {}).get("median_loo_scatter_mmag"),
            "n": e_rows.get("8-9", {}).get("n"),
        },
        "FAINT_14_15": {
            "subset_after_permag_mmag": 172.7804604591969,
            "full_515_median_loo_mmag": e_rows.get("14-15", {}).get("median_loo_scatter_mmag"),
            "n": e_rows.get("14-15", {}).get("n"),
        },
    }

    # Part F
    print("Building COMP-RMS-DEF dataset...", flush=True)
    f_rows = []
    # reuse mag cache; extend for all comps appearing (cap for time: all unique comps)
    all_ids = sorted(set(comp["catalog_id"].astype(str).str.strip()))
    # If too many, still try - 698 rows but fewer unique
    print(f"unique comps={len(all_ids)}", flush=True)
    mag_all = load_inst_mags(all_ids) if len(all_ids) <= 400 else mag
    if len(all_ids) > 400:
        # load remaining in chunks
        missing = [c for c in all_ids if c not in mag_all]
        chunk = 150
        for i in range(0, len(missing), chunk):
            part = load_inst_mags(missing[i : i + chunk])
            mag_all.update(part)
            print(f"  loaded comps {i+len(part)}/{len(missing)}", flush=True)

    for tid, grp in comp.groupby(comp["target_catalog_id"].astype(str)):
        peer_ids = [str(x).strip() for x in grp["catalog_id"].tolist()]
        wmap = {}
        for _, r in grp.iterrows():
            cid = str(r["catalog_id"]).strip()
            w = float(pd.to_numeric(r.get("comp_weight"), errors="coerce"))
            wmap[cid] = w if math.isfinite(w) and w > 0 else 1.0
        for _, r in grp.iterrows():
            cid = str(r["catalog_id"]).strip()
            crm = float(pd.to_numeric(r.get("comp_rms"), errors="coerce"))
            g = float(pd.to_numeric(r.get("phot_g_mean_mag", r.get("mag")), errors="coerce"))
            nf = float(pd.to_numeric(r.get("comp_n_frames"), errors="coerce"))
            loo = weighted_loo_mad_mmag(cid, peer_ids, mag_all, weights=wmap)
            # LOO returns mmag; comp_rms is typically mag
            f_rows.append(
                {
                    "source_id": cid,
                    "target_catalog_id": str(tid),
                    "comp_rms_mag": crm,
                    "comp_rms_mmag": crm * 1000.0 if math.isfinite(crm) else None,
                    "G_mag": g,
                    "n_frames": nf,
                    "loo_scatter_mmag": loo,
                    "loo_estimator": "weighted_mean_peers_MAD_mmag",
                }
            )

    ratios = []
    for r in f_rows:
        if r["loo_scatter_mmag"] and r["comp_rms_mmag"] and r["loo_scatter_mmag"] > 0:
            ratios.append(r["comp_rms_mmag"] / r["loo_scatter_mmag"])
    # correlation
    xs = np.asarray([r["comp_rms_mmag"] for r in f_rows if r["comp_rms_mmag"] and r["loo_scatter_mmag"]], dtype=float)
    ys = np.asarray([r["loo_scatter_mmag"] for r in f_rows if r["comp_rms_mmag"] and r["loo_scatter_mmag"]], dtype=float)
    corr = float(np.corrcoef(xs, ys)[0, 1]) if xs.size >= 3 else None

    ratio_by_bin = {}
    for a, b in bins:
        rr = []
        for r in f_rows:
            g = r["G_mag"]
            if not (math.isfinite(g) and a <= g < b):
                continue
            if r["loo_scatter_mmag"] and r["comp_rms_mmag"] and r["loo_scatter_mmag"] > 0:
                rr.append(r["comp_rms_mmag"] / r["loo_scatter_mmag"])
        ratio_by_bin[f"{a}-{b}"] = {
            "n": len(rr),
            "median_ratio_comp_rms_mmag_over_loo_mmag": float(np.median(rr)) if rr else None,
        }

    part_f_summary = {
        "n_rows": len(f_rows),
        "correlation_comp_rms_mmag_vs_loo_mmag": corr,
        "median_ratio_comp_rms_mmag_over_loo_mmag": float(np.median(ratios)) if ratios else None,
        "ratio_by_G_bin": ratio_by_bin,
        "ordering_note": (
            "If median ratio is roughly constant across G bins, RMS-first ordering is likely "
            "preserved under a monotone remapping; absolute thresholds still need a unified definition."
        ),
    }

    dataset = {
        "header": {
            "run_sha": RUN_SHA,
            "draft_id": 515,
            "columns": {
                "source_id": "Gaia source_id of comparison star",
                "target_catalog_id": "target Gaia source_id",
                "comp_rms_mag": "CSV comp_rms as selection sorted on [mag]",
                "comp_rms_mmag": "comp_rms * 1000 [mmag]",
                "G_mag": "phot_g_mean_mag or mag [mag]",
                "n_frames": "comp_n_frames",
                "loo_scatter_mmag": "1.4826*MAD of (focus - weighted-mean peers) [mmag]",
                "loo_estimator": "PRE-IMPL Q2-style weighted mean peers (not flux-sum loo_diff_series)",
            },
        },
        "summary": part_f_summary,
        "rows": f_rows,
    }

    # write outputs
    out_all = {
        "run_sha": RUN_SHA,
        "part_a": part_a,
        "part_b": part_b,
        "part_c": part_c,
        "part_d": part_d,
        "part_e": part_e,
        "part_f_summary": part_f_summary,
    }
    (OUT / "D515_ACCEPT_01_numbers.json").write_text(
        json.dumps(_jsonable(out_all), indent=2), encoding="ascii", errors="replace"
    )
    (OUT / "COMP_RMS_DEF_01_dataset.json").write_text(
        json.dumps(_jsonable(dataset), indent=2), encoding="ascii", errors="replace"
    )
    print("BO MAD", part_d["BO_CVn"].get("check_scatter_mad_mmag"), flush=True)
    print("FW MAD", part_d["FW_CVn"].get("check_scatter_mad_mmag"), flush=True)
    print("E 8-9", part_e["BIN_8_9"], flush=True)
    print("E 14-15", part_e["FAINT_14_15"], flush=True)
    print("F median ratio", part_f_summary.get("median_ratio_comp_rms_mmag_over_loo_mmag"), flush=True)
    print("WROTE", OUT / "D515_ACCEPT_01_numbers.json", flush=True)
    print("WROTE", OUT / "COMP_RMS_DEF_01_dataset.json", "rows", len(f_rows), flush=True)

if __name__ == "__main__":
    main()
