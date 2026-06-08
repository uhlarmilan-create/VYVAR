#!/usr/bin/env python3
"""Per-target trust flag (GREEN / YELLOW / RED) from comp QA, VYVAR LC, and sep xval."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def norm_id(x) -> str:
    s = str(x or "").strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


def check_star_scatter(phot_dir: Path, target_id: str) -> float:
    p = phot_dir / "lightcurves" / f"check_kmag_{target_id}.csv"
    if not p.is_file():
        return float("nan")
    try:
        sdf = pd.read_csv(p, low_memory=False)
        if sdf.empty or "kmag" not in sdf.columns:
            return float("nan")
        km = pd.to_numeric(sdf["kmag"], errors="coerce")
        if int(km.notna().sum()) < 2:
            return float("nan")
        return float(np.nanstd(km))
    except Exception:  # noqa: BLE001
        return float("nan")


def load_sep_confidence(xval_csv: Path, catalog_id: str) -> str:
    if not xval_csv.is_file():
        return "no_independent"
    df = pd.read_csv(xval_csv, dtype=str, low_memory=False)
    id_col = next((c for c in ("catalog_id", "target_catalog_id") if c in df.columns), None)
    if id_col is None:
        return "no_independent"
    df[id_col] = df[id_col].map(norm_id)
    row = df[df[id_col] == norm_id(catalog_id)]
    if row.empty:
        return "no_independent"
    if "confidence" in row.columns:
        return str(row["confidence"].iloc[0] or "no_independent").strip()
    return "no_independent"


def trust_level(n_clean: int, warnings: int) -> str:
    if n_clean < 3 or warnings >= 2:
        return "RED"
    if warnings == 1:
        return "YELLOW"
    return "GREEN"


def build_reason(
    trust: str,
    n_clean: int,
    sep_conf: str,
    lc_quality: str,
    check_scatter: float,
    n_comps: int,
) -> str:
    parts: list[str] = []
    if sep_conf == "confirmed":
        parts.append("independent check confirmed")
    elif sep_conf == "vyvar_ok_indep_failed":
        parts.append("independent re-measure failed (faint target)")
    elif sep_conf == "review":
        parts.append("independent check needs review")
    else:
        parts.append("no independent sep confirmation")

    parts.append(f"{n_clean} clean comp{'s' if n_clean != 1 else ''}")

    if lc_quality == "noisy":
        parts.append("noisy LC")
    if np.isfinite(check_scatter) and check_scatter >= 0.02:
        parts.append(f"check-star scatter {check_scatter:.3f} mag")
    if n_clean < 3:
        parts.append(f"only {n_clean} clean comps (<3)")

    text = ", ".join(parts)
    if trust == "GREEN":
        return text
    if trust == "YELLOW":
        return text + " — review before submitting"
    return text + " — inspect before submitting"


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-target trust flag from QA + xval + summary")
    ap.add_argument(
        "--comp-qa-targets",
        type=Path,
        default=ROOT / "xval_out" / "comp_qa_targets.csv",
    )
    ap.add_argument(
        "--photometry-dir",
        type=Path,
        default=ROOT
        / "Archive"
        / "Drafts"
        / "draft_000365"
        / "platesolve"
        / "NoFilter_60_2"
        / "photometry",
    )
    ap.add_argument(
        "--xval-results",
        type=Path,
        default=ROOT / "xval_out" / "xval_results.csv",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "xval_out" / "trust_per_target.csv")
    args = ap.parse_args()

    qa = pd.read_csv(args.comp_qa_targets, dtype=str)
    qa["target_catalog_id"] = qa["target_catalog_id"].map(norm_id)
    qa["n_comps"] = pd.to_numeric(qa.get("n_comps"), errors="coerce")
    qa["n_flagged"] = pd.to_numeric(qa.get("n_flagged"), errors="coerce")
    if "n_clean" in qa.columns:
        qa["n_clean"] = pd.to_numeric(qa["n_clean"], errors="coerce")
    else:
        qa["n_clean"] = qa["n_comps"] - qa["n_flagged"]

    summ_path = args.photometry_dir / "photometry_summary.csv"
    if not summ_path.is_file():
        raise SystemExit(f"missing {summ_path}")
    summ = pd.read_csv(summ_path, dtype=str)
    summ["catalog_id"] = summ["catalog_id"].map(norm_id)
    summ["lc_quality_flag"] = summ.get("lc_quality_flag", pd.Series(dtype=str)).astype(str)
    summ["vsx_name"] = summ.get("vsx_name", pd.Series(dtype=str)).astype(str)

    rows: list[dict] = []
    for _, q in qa.iterrows():
        cid = norm_id(q["target_catalog_id"])
        if not cid:
            continue
        n_clean = int(q["n_clean"]) if np.isfinite(q["n_clean"]) else 0
        sep_conf = load_sep_confidence(args.xval_results, cid)
        srow = summ[summ["catalog_id"] == cid]
        vsx = str(q.get("target_vsx_name", "") or "")
        lc_quality = "—"
        if not srow.empty:
            vsx = str(srow["vsx_name"].iloc[0] or vsx)
            lc_quality = str(srow["lc_quality_flag"].iloc[0] or "—").strip().lower()
        chk_sc = check_star_scatter(args.photometry_dir, cid)

        w = 0
        if sep_conf != "confirmed":
            w += 1
        if 3 <= n_clean <= 4:
            w += 1
        if lc_quality == "noisy":
            w += 1
        if np.isfinite(chk_sc) and chk_sc >= 0.02:
            w += 1

        trust = trust_level(n_clean, w)
        reason = build_reason(trust, n_clean, sep_conf, lc_quality, chk_sc, int(q["n_comps"] or 0))
        rows.append({
            "catalog_id": cid,
            "vsx_name": vsx,
            "trust": trust,
            "n_clean": n_clean,
            "sep_confidence": sep_conf,
            "lc_quality": lc_quality,
            "check_scatter": chk_sc if np.isfinite(chk_sc) else "",
            "reason": reason,
        })

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print(f"Wrote {args.out} ({len(out_df)} targets)")
    for level in ("GREEN", "YELLOW", "RED"):
        sub = out_df[out_df["trust"] == level]
        print(f"  {level}: {len(sub)}")
    red = out_df[out_df["trust"] == "RED"]
    if not red.empty:
        print("\nRED targets:")
        for _, r in red.iterrows():
            print(f"  {r['vsx_name']} ({r['catalog_id'][-6:]}): {r['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
