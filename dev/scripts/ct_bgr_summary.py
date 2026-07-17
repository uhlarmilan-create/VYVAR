#!/usr/bin/env python3
"""Field-agnostic B/G/R (+ Luminance) colour-term validation summary."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from photometry_core import _check_color_term_extrapolation, should_apply_color_term  # noqa: E402

_BGR_FILTERS = ("Blue", "Green", "Red", "B", "V", "R", "Rc")
_LUM_FILTER = "Luminance"
_L_FILTERS = ("Luminance", "L", "Lum")
_CT_FILTER_SET = frozenset(_BGR_FILTERS)


def _is_ct_filter(flt: str) -> bool:
    return str(flt or "").split("_")[0] in _CT_FILTER_SET


def _is_lum_filter(flt: str) -> bool:
    return str(flt or "").split("_")[0] in _L_FILTERS


def _draft_dir(draft: int) -> Path:
    return _ROOT / "Archive" / "Drafts" / f"draft_{draft:06d}"


def _file_md5(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _lc_ct_stats(phot_dir: Path) -> pd.DataFrame:
    lc_dir = phot_dir / "lightcurves"
    if not lc_dir.is_dir():
        return pd.DataFrame(columns=["catalog_id", "ct_ok", "ct_correction", "ct_c1"])
    rows: list[dict] = []
    for lc in sorted(lc_dir.glob("lightcurve_*.csv")):
        try:
            df = pd.read_csv(lc, nrows=1, low_memory=False)
        except Exception:  # noqa: BLE001
            continue
        if df.empty:
            continue
        r0 = df.iloc[0]
        cid = lc.stem.replace("lightcurve_", "", 1)
        ct_ok_raw = r0.get("ct_ok", False)
        rows.append(
            {
                "catalog_id": cid,
                "ct_ok": str(ct_ok_raw).strip().lower() in ("true", "1", "yes"),
                "ct_correction": pd.to_numeric(r0.get("ct_correction"), errors="coerce"),
                "ct_c1": pd.to_numeric(r0.get("ct_c1"), errors="coerce"),
            }
        )
    return pd.DataFrame(rows)


def _filter_group(proto: pd.DataFrame, obs_group: str) -> pd.DataFrame:
    flt = obs_group.split("_")[0]
    og = proto["obs_group"].astype(str)
    return proto[og.eq(flt) | og.eq(obs_group)].copy()


def _comp_bp_rp_values(comp_csv: Path, target_cid: str) -> list[float]:
    per_target = comp_csv.parent / "comparison_stars_per_target.csv"
    if per_target.is_file():
        comp = pd.read_csv(per_target, low_memory=False, dtype={"catalog_id": str, "target_catalog_id": str})
        sub = comp[comp["target_catalog_id"].astype(str) == str(target_cid)]
        if not sub.empty and "bp_rp" in sub.columns:
            bps = pd.to_numeric(sub["bp_rp"], errors="coerce").dropna()
            vals = [float(v) for v in bps if np.isfinite(v)]
            if len(vals) >= 2:
                return vals
    if not comp_csv.is_file():
        return []
    comp = pd.read_csv(comp_csv, low_memory=False, dtype={"catalog_id": str})
    if "bp_rp" not in comp.columns:
        return []
    bps = pd.to_numeric(comp["bp_rp"], errors="coerce").dropna()
    return [float(v) for v in bps if np.isfinite(v)]


def _discover_setups(ps: Path) -> list[str]:
    out: list[str] = []
    for d in sorted(ps.iterdir()):
        if not d.is_dir():
            continue
        flt = d.name.split("_")[0]
        if _is_ct_filter(flt) or _is_lum_filter(flt):
            if (d / "MASTERSTAR.fits").is_file():
                out.append(d.name)
    return out


def summarize_filter(
    proto: pd.DataFrame,
    lc: pd.DataFrame,
    comp_csv: Path,
    obs_group: str,
    *,
    min_comp_ct: int = 7,
) -> dict:
    flt = obs_group.split("_")[0]
    sub = _filter_group(proto, obs_group)
    if sub.empty and _is_lum_filter(flt):
        return {
            "obs_group": obs_group,
            "filter": flt,
            "apply_ct_expected": False,
            "n_proto": 0,
        }
    if sub.empty:
        return {"obs_group": obs_group, "filter": flt, "n_proto": 0}
    sub = sub.copy()
    sub["catalog_id"] = sub["catalog_id"].astype(str)
    if not lc.empty:
        lc = lc.copy()
        lc["catalog_id"] = lc["catalog_id"].astype(str)

    merged = sub.merge(lc, on="catalog_id", how="left") if not lc.empty else sub.copy()
    if not lc.empty:
        merged["ct_ok"] = merged["ct_ok"].fillna(False).astype(bool)
    else:
        merged["ct_ok"] = False
    c1 = pd.to_numeric(merged["c1"], errors="coerce")
    gate = merged["gate_would_pass"].astype(str).str.lower().isin(("true", "1", "yes"))

    apply_flags: list[bool] = []
    in_range_flags: list[bool] = []
    for _, row in merged.iterrows():
        apply_ct, _ = should_apply_color_term(
            obs_group=flt,
            c1=float(row.get("c1") or 0.0),
            c1_stderr=float(row.get("c1_stderr") or float("nan")),
            n_comp=int(row.get("n_comp_used") or 0),
            min_comp_for_ct=min_comp_ct,
        )
        apply_flags.append(apply_ct)
        tgt = float(pd.to_numeric(row.get("target_bp_rp"), errors="coerce"))
        bps = _comp_bp_rp_values(comp_csv, str(row.get("catalog_id", "")))
        in_range_flags.append(
            _check_color_term_extrapolation(tgt, bps, extrapolation_tol=0.0) if len(bps) >= 2 else True
        )

    merged["apply_ct"] = apply_flags
    merged["in_range"] = in_range_flags
    merged["extrapolated"] = ~merged["in_range"]
    ct_ok = merged["ct_ok"].astype(bool)

    gate_pass = gate & merged["apply_ct"]
    blocked = gate_pass & merged["extrapolated"] & ~ct_ok
    corrected = ct_ok & (pd.to_numeric(merged["ct_correction"], errors="coerce").abs() > 1e-9)

    c1_gp = c1[gate_pass & ct_ok]
    ct_abs = pd.to_numeric(merged.loc[corrected, "ct_correction"], errors="coerce").abs()
    sc = pd.to_numeric(merged["cat_inst_scatter"], errors="coerce")
    scr = pd.to_numeric(merged["cat_inst_scatter_resid"], errors="coerce")

    example = None
    in_range_ok = merged[ct_ok & merged["in_range"] & gate_pass]
    if not in_range_ok.empty:
        ex = in_range_ok.iloc[0]
        example = {
            "catalog_id": str(ex.get("catalog_id")),
            "target_bp_rp": float(ex.get("target_bp_rp")),
            "comp_med_bp_rp": float(ex.get("comp_med_bp_rp")),
            "ct_corr": float(ex.get("ct_corr")),
            "cat_inst_scatter": float(ex.get("cat_inst_scatter")),
            "cat_inst_scatter_resid": float(ex.get("cat_inst_scatter_resid")),
        }

    return {
        "obs_group": obs_group,
        "filter": flt,
        "apply_ct_expected": _is_ct_filter(flt),
        "n_proto": int(len(sub)),
        "n_in_summary": int(len(merged)),
        "apply_ct_true": int(merged["apply_ct"].sum()),
        "ct_ok_true": int(ct_ok.sum()),
        "in_range_ct_ok_true": int((ct_ok & merged["in_range"]).sum()),
        "gate_would_pass": int(gate.sum()),
        "extrapolated_count": int(merged["extrapolated"].sum()),
        "extrap_blocked_present": int(blocked.sum()),
        "c1_median_gate_pass": float(c1[gate_pass].median()) if gate_pass.any() else float("nan"),
        "c1_median_gate_pass_ct_ok": float(c1_gp.median()) if len(c1_gp) else float("nan"),
        "c1_iqr_gate_pass_ct_ok": (
            [float(c1_gp.quantile(0.25)), float(c1_gp.quantile(0.75))] if len(c1_gp) else [float("nan"), float("nan")]
        ),
        "ct_corr_median": float(ct_abs.median()) if len(ct_abs) else float("nan"),
        "ct_corr_p90": float(ct_abs.quantile(0.9)) if len(ct_abs) else float("nan"),
        "cat_inst_scatter_median": float(sc.median()),
        "cat_inst_scatter_resid_median": float(scr.median()),
        "scatter_improved_fraction": float((scr < sc).mean()) if len(sc) else float("nan"),
        "example_in_range_target": example,
    }


def summarize_draft(draft_id: int) -> dict:
    draft = _draft_dir(draft_id)
    proto_path = draft / "ct_prototype.csv"
    ps = draft / "platesolve"
    report: dict = {"draft_id": draft_id, "filters": [], "masterstar_hashes": {}}
    if ps.is_dir():
        for setup in _discover_setups(ps):
            ms = ps / setup / "MASTERSTAR.fits"
            report["masterstar_hashes"][setup] = _file_md5(ms)
    if not proto_path.is_file():
        report["error"] = f"Missing {proto_path}"
        return report
    proto = pd.read_csv(proto_path)
    report["ct_prototype_rows"] = int(len(proto))
    for setup in _discover_setups(ps):
        phot = ps / setup / "photometry"
        comp = phot / "comparison_stars_per_target.csv"
        if not comp.is_file():
            comp = ps / setup / "comparison_stars.csv"
        lc = _lc_ct_stats(phot)
        stats = summarize_filter(proto, lc, comp, setup)
        report["filters"].append(stats)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft", type=int, required=True)
    args = ap.parse_args()
    report = summarize_draft(args.draft)
    print(f"draft_{args.draft:06d}  ct_prototype rows={report.get('ct_prototype_rows', 0)}")
    if report.get("masterstar_hashes"):
        print("\n=== MASTERSTAR hashes ===")
        for k, v in sorted(report["masterstar_hashes"].items()):
            print(f"  {k}: {v}")
    for stats in report.get("filters", []):
        print(f"\n=== {stats.get('obs_group')} ===")
        for k, v in stats.items():
            if k != "example_in_range_target":
                print(f"  {k}: {v}")
        ex = stats.get("example_in_range_target")
        if ex:
            print(f"  example_in_range_target: {ex}")
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
