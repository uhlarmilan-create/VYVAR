#!/usr/bin/env python3
"""EPSF-PIN-CENSUS-01: why pinned comps fail psf_fit_ok (measure only).

Read-only on draft 516 catalogs. Does not rewrite proc CSVs or LCs.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from photometry_core import ensemble_normalize  # noqa: E402
from psf_internal_lc import (  # noqa: E402
    _flux_to_inst_mag,
    _proc_csv_key,
    resolve_ensemble_ids,
)

DRAFT = REPO / "Archive" / "Drafts" / "draft_000516"
PS = DRAFT / "platesolve" / "NoFilter_60_2"
FRAMES = DRAFT / "detrended_aligned" / "lights" / "NoFilter_60_2"
PHOT = PS / "photometry"
LC_DIR = PHOT / "lightcurves"
OUT = REPO / "dev" / "results" / "session_20260824_epsf_pin_census_01"
PROD_EPSF = PS / "masterstar_epsf.fits"
PROD_META = PS / "masterstar_epsf_meta.json"
BO_CVN = "1498613634033133184"
FW_CVN = "1497343732462852864"
BO_COMP = "1499200223486564608"
FW_COMP = "1497442379271632384"
PROD_EPSF_SHA = "172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20"
AC02_BO_N_FULL = 23
AC02_BO_RMS_MMAG = 38.76979400310864
AC02_BO_OFFSET_MMAG = 40.464
USECOLS = [
    "catalog_id",
    "source_file",
    "psf_flux",
    "psf_chi2",
    "psf_fit_ok",
    "psf_quality",
    "psf_quality_fallback",
    "phot_g_mean_mag",
]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO)).replace("\\", "/")


def _norm(raw: Any) -> str:
    try:
        return str(normalize_gaia_source_id(raw)).strip()
    except Exception:  # noqa: BLE001
        s = str(raw or "").strip()
        return "" if s.lower() in ("", "nan", "none") else s


def _is_true(val: Any) -> bool:
    if isinstance(val, bool):
        return bool(val)
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return False
    return str(val).strip().lower() in ("1", "true", "t", "yes", "y")


def must_not_change_files() -> list[Path]:
    files = [PROD_EPSF, PROD_META]
    files.extend(
        sorted(
            p
            for p in LC_DIR.glob("lightcurve_*.csv")
            if "_psf" not in p.name and "_adaptive" not in p.name
        )
    )
    aavso = PHOT / "lightcurves_reports" / "aavso"
    varastro = PHOT / "lightcurves_reports" / "varastro"
    if aavso.is_dir():
        files.extend(sorted(aavso.glob("*.txt")))
    if varastro.is_dir():
        files.extend(sorted(varastro.glob("*.txt")))
    return [p for p in files if p.is_file()]


def snapshot_hashes(label: str) -> dict[str, str]:
    out = {_rel(p): _sha(p) for p in must_not_change_files()}
    (OUT / f"hashes_{label}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="ascii")
    return out


def aperture_targets() -> list[str]:
    wanted = []
    for p in sorted(LC_DIR.glob("lightcurve_*.csv")):
        stem = p.stem
        if stem.endswith("_psf") or stem.endswith("_adaptive"):
            continue
        cid = _norm(stem.replace("lightcurve_", "", 1))
        if cid:
            wanted.append(cid)
    return wanted


def load_stack() -> pd.DataFrame:
    files = sorted(FRAMES.glob("proc_*.csv"))
    chunks = []
    for p in files:
        df = pd.read_csv(p, low_memory=False, dtype={"catalog_id": str}, usecols=lambda c: c in USECOLS)
        keep = [c for c in USECOLS if c in df.columns]
        df = df[keep].copy()
        df["catalog_id"] = df["catalog_id"].map(_norm)
        df["proc_csv"] = p.name
        if "source_file" in df.columns:
            df["epoch_key"] = [_proc_csv_key(str(s), p) for s in df["source_file"].tolist()]
        else:
            df["epoch_key"] = p.name
        chunks.append(df)
    return pd.concat(chunks, ignore_index=True)


def classify_row(rec: pd.Series | None) -> str:
    """Exclusive stored-column class. Last class is inferred (flags not persisted)."""
    if rec is None:
        return "missing_row"
    flux = float(pd.to_numeric(rec.get("psf_flux"), errors="coerce"))
    chi2 = float(pd.to_numeric(rec.get("psf_chi2"), errors="coerce"))
    ok = _is_true(rec.get("psf_fit_ok"))
    if ok:
        return "fit_ok"
    if not (math.isfinite(flux) and flux > 0):
        return "nonfinite_psf_flux"
    if math.isfinite(chi2) and chi2 >= 50.0:
        return "chi2_ge_50"
    if not math.isfinite(chi2):
        return "nonfinite_chi2"
    return "inferred_non_converged_or_quality_fallback"


Index = dict[tuple[str, str], pd.Series]


def build_index(stack: pd.DataFrame) -> Index:
    idx: Index = {}
    for _, rec in stack.iterrows():
        idx[(str(rec["catalog_id"]), str(rec["epoch_key"]))] = rec
    return idx


def lookup(index: Index, cid: str, epoch_key: str) -> pd.Series | None:
    return index.get((cid, epoch_key))


def g_mag(stack: pd.DataFrame, cid: str) -> float:
    sub = stack.loc[stack["catalog_id"] == cid, "phot_g_mean_mag"]
    if sub.empty:
        return float("nan")
    v = pd.to_numeric(sub, errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def zp_ok_current(rec: pd.Series | None) -> bool:
    if rec is None:
        return False
    flux = float(pd.to_numeric(rec.get("psf_flux"), errors="coerce"))
    return _is_true(rec.get("psf_fit_ok")) and math.isfinite(flux) and flux > 0


def zp_ok_chi2_lt(rec: pd.Series | None, limit: float) -> bool:
    """Raise chi2 gate to limit. Does not admit inferred class (chi2 < 50, fit_ok false)."""
    if zp_ok_current(rec):
        return True
    if rec is None:
        return False
    flux = float(pd.to_numeric(rec.get("psf_flux"), errors="coerce"))
    chi2 = float(pd.to_numeric(rec.get("psf_chi2"), errors="coerce"))
    if not (math.isfinite(flux) and flux > 0 and math.isfinite(chi2)):
        return False
    return 50.0 <= chi2 < float(limit)


def zp_ok_admit_chi2_ge50(rec: pd.Series | None) -> bool:
    if zp_ok_current(rec):
        return True
    if rec is None:
        return False
    flux = float(pd.to_numeric(rec.get("psf_flux"), errors="coerce"))
    chi2 = float(pd.to_numeric(rec.get("psf_chi2"), errors="coerce"))
    return math.isfinite(flux) and flux > 0 and math.isfinite(chi2) and chi2 >= 50.0


def zp_ok_admit_inferred(rec: pd.Series | None) -> bool:
    if zp_ok_current(rec):
        return True
    if rec is None:
        return False
    flux = float(pd.to_numeric(rec.get("psf_flux"), errors="coerce"))
    chi2 = float(pd.to_numeric(rec.get("psf_chi2"), errors="coerce"))
    ok = _is_true(rec.get("psf_fit_ok"))
    return (
        (not ok)
        and math.isfinite(flux)
        and flux > 0
        and math.isfinite(chi2)
        and chi2 < 50.0
    )


def zp_ok_conv_finite(rec: pd.Series | None) -> bool:
    """fit_ok OR (finite flux AND finite chi2)."""
    if zp_ok_current(rec):
        return True
    if rec is None:
        return False
    flux = float(pd.to_numeric(rec.get("psf_flux"), errors="coerce"))
    chi2 = float(pd.to_numeric(rec.get("psf_chi2"), errors="coerce"))
    return math.isfinite(flux) and flux > 0 and math.isfinite(chi2)


def meters_for_target(
    *,
    tid: str,
    index: Index,
    epoch_keys: list[str],
    ap_delta: np.ndarray,
    comp_ids: list[str],
    weight_map: dict[str, float],
    pred: Callable[[pd.Series | None], bool],
) -> dict[str, Any]:
    n = len(epoch_keys)
    tgt_flux = np.full(n, np.nan)
    tgt_ok = np.zeros(n, dtype=bool)
    for i, key in enumerate(epoch_keys):
        rec = lookup(index, tid, key)
        tgt_ok[i] = pred(rec)
        if rec is not None:
            tgt_flux[i] = float(pd.to_numeric(rec.get("psf_flux"), errors="coerce"))
    target_mag = np.full(n, np.nan)
    usable = tgt_ok & np.isfinite(tgt_flux) & (tgt_flux > 0)
    target_mag[usable] = _flux_to_inst_mag(tgt_flux)[usable]

    comp_mag: dict[str, np.ndarray] = {}
    pin_ok = np.ones(n, dtype=bool)
    for cid in comp_ids:
        mag = np.full(n, np.nan)
        for i, key in enumerate(epoch_keys):
            rec = lookup(index, cid, key)
            if pred(rec):
                fl = float(pd.to_numeric(rec.get("psf_flux"), errors="coerce"))
                if math.isfinite(fl) and fl > 0:
                    mag[i] = float(_flux_to_inst_mag(np.array([fl]))[0])
            else:
                pin_ok[i] = False
        comp_mag[cid] = mag
    dummy = {cid: 0.0 for cid in comp_ids}
    quality = {cid: {"quality": "good"} for cid in comp_ids}
    _cal, psf_delta, _sc = ensemble_normalize(
        target_mag,
        comp_mag,
        dummy,
        quality,
        comp_weight_map=weight_map or None,
    )
    _ = _cal
    psf_delta = np.asarray(psf_delta, dtype=float)
    psf_delta[~pin_ok] = np.nan
    n_full = int(pin_ok.sum())
    both = pin_ok & np.isfinite(psf_delta) & np.isfinite(ap_delta)
    res = psf_delta[both] - ap_delta[both]
    n_both = int(both.sum())
    med = float(np.median(res)) if n_both else float("nan")
    rms = float(np.sqrt(np.mean(res**2))) if n_both else float("nan")
    dem = float(np.sqrt(np.mean((res - med) ** 2))) if n_both else float("nan")
    rms_vs = bool(
        n_both > 0 and abs(rms - abs(med)) <= 0.25 * max(abs(med), abs(rms), 1e-12)
    )
    return {
        "n_epochs": n,
        "n_full_membership": n_full,
        "n_dropped_pin": n - n_full,
        "coverage": (n_full / n) if n else float("nan"),
        "n_finite_pairs": n_both,
        "level_offset_mag": med,
        "level_offset_mmag": med * 1000.0 if math.isfinite(med) else float("nan"),
        "rms_mag": rms,
        "rms_mmag": rms * 1000.0 if math.isfinite(rms) else float("nan"),
        "demeaned_rms_mmag": dem * 1000.0 if math.isfinite(dem) else float("nan"),
        "rms_vs_abs_median": rms_vs,
    }


def copy_context() -> None:
    ctx = REPO / "dev" / "results" / "context" / "session_20260824_epsf_pin_census_01"
    ctx.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    blobs: dict[str, str] = {}
    for p in sorted(OUT.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() in {".csv", ".json", ".txt", ".md"}:
            shutil.copy2(p, ctx / p.name)
            copied.append(p.name)
        else:
            blobs[p.name] = _sha(p)
    (ctx / "BLOB_SHA_MANIFEST.json").write_text(
        json.dumps({"copied_text": copied, "blobs_not_copied": blobs}, indent=2) + "\n",
        encoding="ascii",
    )


def main() -> int:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    epsf_sha = _sha(PROD_EPSF)
    if epsf_sha != PROD_EPSF_SHA:
        raise SystemExit(f"G2 FAIL production ePSF SHA {epsf_sha}")
    h0 = snapshot_hashes("before")

    targets = aperture_targets()
    stack = load_stack()
    index = build_index(stack)
    membership: list[dict[str, Any]] = []
    per_comp_fail_rows: list[dict[str, Any]] = []
    star_epoch_causes: list[dict[str, Any]] = []
    pin_drop_causes: list[str] = []
    unique_comps: dict[str, dict[str, Any]] = {}

    target_epochs: dict[str, tuple[list[str], np.ndarray, list[str], dict[str, float]]] = {}

    for tid in targets:
        ap_path = LC_DIR / f"lightcurve_{tid}.csv"
        ap = pd.read_csv(ap_path, low_memory=False)
        epoch_keys = [str(s).strip() for s in ap["source_file"].tolist()]
        ap_delta = pd.to_numeric(ap.get("delta_mag"), errors="coerce").to_numpy(dtype=float)
        comp_ids, weight_map, ens_source = resolve_ensemble_ids(tid, PHOT)
        target_epochs[tid] = (epoch_keys, ap_delta, comp_ids, weight_map or {})
        membership.append(
            {
                "target_id": tid,
                "n_comp": len(comp_ids),
                "ensemble_source": ens_source,
                "comp_ids": ",".join(comp_ids),
            }
        )
        for cid in comp_ids:
            unique_comps.setdefault(cid, {"g": g_mag(stack, cid), "n_targets": 0})
            unique_comps[cid]["n_targets"] += 1
        for i, key in enumerate(epoch_keys):
            first_fail = ""
            for cid in comp_ids:
                rec = lookup(index, cid, key)
                cause = classify_row(rec)
                qfb = _is_true(rec.get("psf_quality_fallback")) if rec is not None else False
                qlab = str(rec.get("psf_quality") or "") if rec is not None else ""
                chi2 = (
                    float(pd.to_numeric(rec.get("psf_chi2"), errors="coerce"))
                    if rec is not None
                    else float("nan")
                )
                flux = (
                    float(pd.to_numeric(rec.get("psf_flux"), errors="coerce"))
                    if rec is not None
                    else float("nan")
                )
                row = {
                    "target_id": tid,
                    "comp_id": cid,
                    "epoch_key": key,
                    "cause": cause,
                    "psf_chi2": chi2,
                    "psf_flux": flux,
                    "psf_quality": qlab,
                    "psf_quality_fallback": qfb,
                }
                star_epoch_causes.append(row)
                if cause != "fit_ok":
                    per_comp_fail_rows.append(row)
                    if not first_fail:
                        first_fail = cause
            if first_fail:
                pin_drop_causes.append(first_fail)

    # Unique-comp cause table (134 epochs each; first target's epoch backbone = 134 lights)
    # Use BO epochs if present else first target.
    ref_epochs = target_epochs.get(BO_CVN, next(iter(target_epochs.values())))[0]
    comp_table_rows = []
    for cid, meta in sorted(unique_comps.items()):
        counts = {
            "missing_row": 0,
            "nonfinite_psf_flux": 0,
            "chi2_ge_50": 0,
            "nonfinite_chi2": 0,
            "inferred_non_converged_or_quality_fallback": 0,
            "fit_ok": 0,
        }
        inferred_qfb = 0
        inferred_no_qfb = 0
        chi2_vals = []
        for key in ref_epochs:
            rec = lookup(index, cid, key)
            cause = classify_row(rec)
            counts[cause] = counts.get(cause, 0) + 1
            if cause == "inferred_non_converged_or_quality_fallback":
                if rec is not None and _is_true(rec.get("psf_quality_fallback")):
                    inferred_qfb += 1
                else:
                    inferred_no_qfb += 1
            if cause == "chi2_ge_50" and rec is not None:
                chi2_vals.append(float(pd.to_numeric(rec.get("psf_chi2"), errors="coerce")))
        n_fail = 134 - counts["fit_ok"]
        comp_table_rows.append(
            {
                "comp_id": cid,
                "phot_g_mean_mag": meta["g"],
                "n_targets_using": meta["n_targets"],
                "n_fit_ok": counts["fit_ok"],
                "n_fail": n_fail,
                "n_missing_row": counts["missing_row"],
                "n_nonfinite_psf_flux": counts["nonfinite_psf_flux"],
                "n_chi2_ge_50": counts["chi2_ge_50"],
                "n_nonfinite_chi2": counts["nonfinite_chi2"],
                "n_inferred_non_converged_or_quality_fallback": counts[
                    "inferred_non_converged_or_quality_fallback"
                ],
                "n_inferred_quality_fallback_stored": inferred_qfb,
                "n_inferred_no_quality_fallback_flag": inferred_no_qfb,
                "chi2_fail_median": float(np.median(chi2_vals)) if chi2_vals else float("nan"),
            }
        )
    pd.DataFrame(comp_table_rows).to_csv(OUT / "c1_per_comp_cause.csv", index=False)
    pd.DataFrame(membership).to_csv(OUT / "c1_target_membership.csv", index=False)

    chi2_fail = [
        float(r["psf_chi2"])
        for r in star_epoch_causes
        if r["cause"] == "chi2_ge_50" and math.isfinite(float(r["psf_chi2"]))
    ]
    chi2_arr = np.asarray(chi2_fail, dtype=float)
    hist = {
        "n_chi2_ge_50_star_epochs": int(chi2_arr.size),
        "bin_50_100": int(((chi2_arr >= 50) & (chi2_arr < 100)).sum()) if chi2_arr.size else 0,
        "bin_100_200": int(((chi2_arr >= 100) & (chi2_arr < 200)).sum()) if chi2_arr.size else 0,
        "bin_ge_200": int((chi2_arr >= 200).sum()) if chi2_arr.size else 0,
        "median_chi2": float(np.median(chi2_arr)) if chi2_arr.size else float("nan"),
        "p16_chi2": float(np.percentile(chi2_arr, 16)) if chi2_arr.size else float("nan"),
        "p84_chi2": float(np.percentile(chi2_arr, 84)) if chi2_arr.size else float("nan"),
    }
    (OUT / "c1_chi2_histogram.json").write_text(json.dumps(hist, indent=2) + "\n", encoding="ascii")

    from collections import Counter

    se_counts = Counter(r["cause"] for r in star_epoch_causes)
    drop_counts = Counter(pin_drop_causes)
    n_se = len(star_epoch_causes)
    n_se_fail = n_se - se_counts.get("fit_ok", 0)
    n_pin = len(pin_drop_causes)
    c1_summary = {
        "n_targets": len(targets),
        "n_unique_comps": len(unique_comps),
        "n_star_epochs": n_se,
        "star_epoch_cause_counts": dict(se_counts),
        "star_epoch_fail_n": n_se_fail,
        "frac_star_epoch_fail_chi2_ge_50": (se_counts.get("chi2_ge_50", 0) / n_se_fail) if n_se_fail else float("nan"),
        "frac_star_epoch_fail_inferred": (
            se_counts.get("inferred_non_converged_or_quality_fallback", 0) / n_se_fail
        )
        if n_se_fail
        else float("nan"),
        "frac_star_epoch_fail_nonfinite_or_missing": (
            (
                se_counts.get("nonfinite_psf_flux", 0)
                + se_counts.get("missing_row", 0)
                + se_counts.get("nonfinite_chi2", 0)
            )
            / n_se_fail
        )
        if n_se_fail
        else float("nan"),
        "n_target_epoch_pin_drops": n_pin,
        "pin_drop_first_comp_cause_counts": dict(drop_counts),
        "frac_pin_drops_chi2_ge_50": (drop_counts.get("chi2_ge_50", 0) / n_pin) if n_pin else float("nan"),
        "frac_pin_drops_inferred": (
            drop_counts.get("inferred_non_converged_or_quality_fallback", 0) / n_pin
        )
        if n_pin
        else float("nan"),
        "frac_pin_drops_nonfinite_or_missing": (
            (
                drop_counts.get("nonfinite_psf_flux", 0)
                + drop_counts.get("missing_row", 0)
                + drop_counts.get("nonfinite_chi2", 0)
            )
            / n_pin
        )
        if n_pin
        else float("nan"),
        "chi2_histogram": hist,
        "note_finite_chi2": "A row with finite stored psf_chi2 implies the PSF fit completed.",
        "inferred_class_name": "inferred_non_converged_or_quality_fallback",
        "inferred_class_definition": "fit_ok false AND finite psf_flux>0 AND finite chi2 < 50. Convergence flags are not persisted; this class is inferred.",
    }
    (OUT / "c1_summary.json").write_text(json.dumps(c1_summary, indent=2) + "\n", encoding="ascii")

    def _series(cid: str, tag: str) -> pd.DataFrame:
        rows = []
        for key in ref_epochs:
            rec = lookup(index, cid, key)
            cause = classify_row(rec)
            chi2 = (
                float(pd.to_numeric(rec.get("psf_chi2"), errors="coerce"))
                if rec is not None
                else float("nan")
            )
            flux = (
                float(pd.to_numeric(rec.get("psf_flux"), errors="coerce"))
                if rec is not None
                else float("nan")
            )
            rows.append(
                {
                    "epoch_key": key,
                    "phot_g_mean_mag": g_mag(stack, cid),
                    "psf_fit_ok": _is_true(rec.get("psf_fit_ok")) if rec is not None else False,
                    "psf_chi2": chi2,
                    "psf_flux": flux,
                    "psf_quality": str(rec.get("psf_quality") or "") if rec is not None else "",
                    "psf_quality_fallback": _is_true(rec.get("psf_quality_fallback"))
                    if rec is not None
                    else False,
                    "cause": cause,
                }
            )
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"c1_{tag}_chi2_series.csv", index=False)
        return df

    bo_s = _series(BO_COMP, "bo_comp_1499200223486564608")
    fw_s = _series(FW_COMP, "fw_comp_1497442379271632384")
    callout = {
        "bo_comp": {
            "catalog_id": BO_COMP,
            "phot_g_mean_mag": g_mag(stack, BO_COMP),
            "cause_counts": dict(Counter(bo_s["cause"].tolist())),
            "n_chi2_ge_50": int((bo_s["cause"] == "chi2_ge_50").sum()),
            "median_chi2_when_ge50": float(
                np.median(pd.to_numeric(bo_s.loc[bo_s["cause"] == "chi2_ge_50", "psf_chi2"], errors="coerce"))
            )
            if int((bo_s["cause"] == "chi2_ge_50").sum())
            else float("nan"),
        },
        "fw_comp": {
            "catalog_id": FW_COMP,
            "phot_g_mean_mag": g_mag(stack, FW_COMP),
            "cause_counts": dict(Counter(fw_s["cause"].tolist())),
            "n_chi2_ge_50": int((fw_s["cause"] == "chi2_ge_50").sum()),
            "median_chi2_when_ge50": float(
                np.median(pd.to_numeric(fw_s.loc[fw_s["cause"] == "chi2_ge_50", "psf_chi2"], errors="coerce"))
            )
            if int((fw_s["cause"] == "chi2_ge_50").sum())
            else float("nan"),
        },
    }
    (OUT / "c1_callout_bo_fw.json").write_text(json.dumps(callout, indent=2) + "\n", encoding="ascii")

    variants: list[tuple[str, Callable[[pd.Series | None], bool]]] = [
        ("chi2_lt50_current", zp_ok_current),
        ("chi2_lt100", lambda r: zp_ok_chi2_lt(r, 100.0)),
        ("chi2_lt200", lambda r: zp_ok_chi2_lt(r, 200.0)),
        ("admit_chi2_ge50", zp_ok_admit_chi2_ge50),
        ("admit_inferred_fallback", zp_ok_admit_inferred),
        ("conv_finite_both", zp_ok_conv_finite),
    ]
    c2_rows = []
    t_c2 = time.perf_counter()
    for tid, label in ((BO_CVN, "BO_CVn"), (FW_CVN, "FW_CVn")):
        epoch_keys, ap_delta, comp_ids, wmap = target_epochs[tid]
        for vname, pred in variants:
            m = meters_for_target(
                tid=tid,
                index=index,
                epoch_keys=epoch_keys,
                ap_delta=ap_delta,
                comp_ids=comp_ids,
                weight_map=wmap,
                pred=pred,
            )
            m["target"] = label
            m["catalog_id"] = tid
            m["variant"] = vname
            c2_rows.append(m)
    pd.DataFrame(c2_rows).to_csv(OUT / "c2_threshold_meters.csv", index=False)
    c2_elapsed = time.perf_counter() - t_c2

    bo_cur = next(r for r in c2_rows if r["target"] == "BO_CVn" and r["variant"] == "chi2_lt50_current")
    c2_check = {
        "ac02_n_full": AC02_BO_N_FULL,
        "measured_n_full": bo_cur["n_full_membership"],
        "n_full_match": int(bo_cur["n_full_membership"]) == AC02_BO_N_FULL,
        "ac02_rms_mmag": AC02_BO_RMS_MMAG,
        "measured_rms_mmag": bo_cur["rms_mmag"],
        "rms_delta_mmag": float(bo_cur["rms_mmag"]) - AC02_BO_RMS_MMAG
        if math.isfinite(float(bo_cur["rms_mmag"]))
        else float("nan"),
    }
    (OUT / "c2_ac02_positive_control.json").write_text(json.dumps(c2_check, indent=2) + "\n", encoding="ascii")
    (OUT / "c2_meters.json").write_text(json.dumps(c2_rows, indent=2) + "\n", encoding="ascii")

    h1 = snapshot_hashes("after")
    summary = {
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "c2_elapsed_s": round(c2_elapsed, 3),
        "g2_epsf_sha": epsf_sha,
        "hash_guard_ok": h0 == h1,
        "n_targets": len(targets),
        "c1": c1_summary,
        "c1_callout": callout,
        "c2_positive_control": c2_check,
        "c2": c2_rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    copy_context()
    if h0 != h1:
        raise SystemExit("hash guard FAIL")
    print(
        json.dumps(
            {
                "ok": True,
                "elapsed_s": summary["elapsed_s"],
                "n_full_match": c2_check["n_full_match"],
                "bo_current_n_full": bo_cur["n_full_membership"],
                "hash_guard_ok": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
