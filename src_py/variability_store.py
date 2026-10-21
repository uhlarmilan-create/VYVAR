"""Files of the Variability tab that the PDF report reads too (VAR-TAB-2026-10).

Kept out of ``ui_variability`` so the report does not import Streamlit.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPORT_INCLUDE_JSON = "variability_report_include.json"
ADOPTED_PERIODS_JSON = "variability_periods.json"


def _read_json(p: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def load_report_include(phot: Path) -> list[str]:
    """Candidates whose TESS result goes into the SUMMARY MEASURE REPORT."""
    return [str(x) for x in _read_json(Path(phot) / REPORT_INCLUDE_JSON).get("catalog_ids", [])]


def save_report_include(phot: Path, ids: list[str]) -> None:
    (Path(phot) / REPORT_INCLUDE_JSON).write_text(json.dumps({"catalog_ids": sorted(set(ids))}, indent=2), encoding="utf-8")


def load_adopted_periods(phot: Path) -> dict[str, dict[str, float]]:
    """``{catalog_id: {"period_d", "t0_bjd"}}`` the observer adopted in the phase view."""
    raw = _read_json(Path(phot) / ADOPTED_PERIODS_JSON)
    out: dict[str, dict[str, float]] = {}
    for cid, v in raw.items():
        try:
            out[str(cid)] = {"period_d": float(v["period_d"]), "t0_bjd": float(v["t0_bjd"])}
        except (KeyError, TypeError, ValueError):
            continue
    return out


def save_adopted_period(phot: Path, cid: str, period_d: float, t0_bjd: float) -> None:
    cur = load_adopted_periods(phot)
    cur[str(cid)] = {"period_d": float(period_d), "t0_bjd": float(t0_bjd)}
    (Path(phot) / ADOPTED_PERIODS_JSON).write_text(json.dumps(cur, indent=2, sort_keys=True), encoding="utf-8")


def tess_summary(phot: Path, cid: str) -> dict[str, Any]:
    from tess_runner import tess_dir_for  # noqa: PLC0415

    return _read_json(tess_dir_for(phot, cid) / "tess_result.json")


def tess_cell(res: dict[str, Any]) -> str:
    if not res:
        return "not run"
    v = str(res.get("verdict") or "")
    b = res.get("best") or {}
    if v == "RELIABLE" and b.get("period_d"):
        if str(b.get("note", "")).startswith("equal minima"):
            return f"P = {b['period_d']:.6f} d or 2P = {2 * b['period_d']:.6f} d if EW/ELL (reliable)"
        return f"P = {b['period_d']:.6f} d (reliable)"
    return {"NEIGHBOUR": "signal of a neighbour", "AMBIGUOUS": "ambiguous", "NO_DATA": "no data"}.get(v, v or "-")


def peranso_text(lc_csv: Path) -> str:
    """Night light curve of the variability search as ``time dmag err`` lines."""
    d = pd.read_csv(lc_csv)
    lines = ["# VYVAR night light curve: time (see variability_meta.json time_column), dmag, err"]
    for _, r in d.iterrows():
        if pd.notna(r["dmag"]):
            lines.append(f"{float(r['time']):.6f} {float(r['dmag']):.4f} {float(r['err']):.4f}")
    return "\n".join(lines) + "\n"


def phase_of(t: np.ndarray, period_d: float, t0_bjd: float) -> np.ndarray:
    """Phase 0..1 of each time; phase 0 at ``t0_bjd``."""
    return np.mod((np.asarray(t, float) - float(t0_bjd)) / float(period_d), 1.0)


def pdm_theta(phase: np.ndarray, mag: np.ndarray, n_bins: int = 20) -> float:
    """Phase dispersion Theta (Stellingwerf 1978): pooled bin variance / total variance."""
    ph = np.asarray(phase, float)
    m = np.asarray(mag, float)
    ok = np.isfinite(ph) & np.isfinite(m)
    ph, m = ph[ok], m[ok]
    if m.size < 3:
        return float("nan")
    b = np.minimum((ph * n_bins).astype(int), n_bins - 1)
    num = 0.0
    dof = 0
    for k in range(n_bins):
        x = m[b == k]
        if x.size > 1:
            num += float(np.sum((x - x.mean()) ** 2))
            dof += x.size - 1
    vt = float(np.var(m, ddof=1))
    return (num / dof) / vt if dof > 0 and vt > 0 else float("nan")


def epoch_of_minimum(t: np.ndarray, mag: np.ndarray, period_d: float, n_bins: int = 50) -> float:
    """Time of the faintest phase bin (phase 0 for the phase curve)."""
    t = np.asarray(t, float)
    m = np.asarray(mag, float)
    ref = float(np.nanmin(t))
    ph = phase_of(t, period_d, ref)
    b = np.minimum((ph * n_bins).astype(int), n_bins - 1)
    means = np.array([np.nanmean(m[b == k]) if np.any(b == k) else -np.inf for k in range(n_bins)])
    return ref + (int(np.argmax(means)) + 0.5) / n_bins * float(period_d)


def phased_text(t: np.ndarray, mag: np.ndarray, err: np.ndarray, period_d: float, t0_bjd: float, label: str) -> str:
    """Peranso-readable phased light curve (phase, dmag, err, BJD)."""
    ph = phase_of(t, period_d, t0_bjd)
    lines = [f"# {label}  P = {period_d:.7f} d  T0 = {t0_bjd:.6f} BJD_TDB", "# phase dmag err bjd_tdb"]
    for p, m, e, tt in zip(ph, mag, err, t):
        if np.isfinite(m):
            lines.append(f"{p:.5f} {m:.4f} {e:.4f} {tt:.6f}")
    return "\n".join(lines) + "\n"
