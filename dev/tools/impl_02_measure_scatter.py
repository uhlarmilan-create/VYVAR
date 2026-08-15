"""Snapshot before/after production scatter for IMPL-02 Part D/E."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

PHOT = (
    ROOT
    / "Archive"
    / "Drafts"
    / "draft_000514"
    / "platesolve"
    / "NoFilter_60_2"
    / "photometry"
)
LC = PHOT / "lightcurves"
OUT = ROOT / "dev" / "results" / "IMPL_02_production_scatter.json"

TARGETS = {
    "BO_CVn": "1498613634033133184",
    "FW_CVn": "1497343732462852864",
}


def _std_mmag(series: pd.Series) -> float | None:
    s = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 5:
        return None
    return float(np.std(s, ddof=1) * 1000.0)


def _mad_mmag(series: pd.Series) -> float | None:
    s = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 5:
        return None
    med = float(np.median(s))
    return float(np.median(np.abs(s - med)) * 1.4826 * 1000.0)


def _check_scatter_mmag(cid: str) -> float | None:
    p = LC / f"check_kmag_{cid}.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    if "kmag" in df.columns:
        return _std_mmag(df["kmag"])
    if "check_scatter" in df.columns:
        v = float(pd.to_numeric(df["check_scatter"], errors="coerce").iloc[0])
        return v * 1000.0 if math.isfinite(v) else None
    return None


def measure_target(cid: str) -> dict:
    lc_path = LC / f"lightcurve_{cid}.csv"
    out: dict = {"catalog_id": cid, "lc_exists": lc_path.is_file()}
    out["check_scatter_mmag"] = _check_scatter_mmag(cid)
    if not lc_path.is_file():
        return out
    df = pd.read_csv(lc_path)
    out["n_rows"] = int(len(df))
    for col in ("delta_mag", "mag_calib", "mag_calib_ac", "mag_calib_ct"):
        if col in df.columns:
            out[f"{col}_std_mmag"] = _std_mmag(df[col])
            out[f"{col}_mad_mmag"] = _mad_mmag(df[col])
    if "aperture_r_px" in df.columns:
        out["aperture_r_px_median"] = float(
            pd.to_numeric(df["aperture_r_px"], errors="coerce").median()
        )
    elif "aperture_px" in df.columns:
        out["aperture_r_px_median"] = float(
            pd.to_numeric(df["aperture_px"], errors="coerce").median()
        )
    # Colour columns for Part E
    for col in ("ct_ok", "ct_c1", "ct_correction", "ct_bp_rp_target", "ct_bp_rp_comp_med"):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            out[col] = float(v.dropna().iloc[0]) if v.notna().any() else None
    return out


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "after"
    # Also pick quiet non-VSX targets from summary as check-star proxies
    summary = PHOT / "photometry_summary.csv"
    quiet: list[str] = []
    if summary.is_file():
        sdf = pd.read_csv(summary)
        ok = sdf[pd.to_numeric(sdf.get("n_frames"), errors="coerce").fillna(0) > 50].copy()
        if "lc_rms" in ok.columns:
            ok = ok.sort_values("lc_rms")
            for _, row in ok.head(8).iterrows():
                quiet.append(str(int(row["catalog_id"])))

    payload = {
        "label": label,
        "targets": {name: measure_target(cid) for name, cid in TARGETS.items()},
        "quiet_targets": {cid: measure_target(cid) for cid in quiet},
    }
    prev = {}
    if OUT.is_file():
        try:
            prev = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prev = {}
    if not isinstance(prev, dict):
        prev = {}
    prev[label] = payload
    OUT.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
