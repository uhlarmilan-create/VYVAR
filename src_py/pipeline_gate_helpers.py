"""Moved from pipeline.py (CONSOLIDATE-01E1). Facade re-exports these names."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
from config import AppConfig
from fits_meta import extract_fits_metadata
from proc_frame_store import proc_csv_path_for_aligned_fits
from utils import iter_fits_paths_recursive as _iter_fits_recursive

from pipeline import LOGGER

def validate_comparison_ensemble_flatness(
    *,
    frames_root: Path,
    comparison_stars_csv: Path,
    flux_col: str = "flux",
    name_col: str = "name",
    max_relative_rms: float = 0.03,
    min_frames_per_star: int = 5,
    min_stars_per_frame: int = 3,
    output_report_csv: Path | None = None,
) -> dict[str, Any]:
    """Check that comparison stars stay flat vs the **per-frame** ensemble (median flux of comps on that frame).

    Uses DAO ``flux`` from sidecar CSVs next to each aligned FITS under ``frames_root`` (same layout as
    ``export_per_frame_catalogs``). For each exposure, builds relative flux ``f_i / median(f_all comps present)``.
    A good comparison star has low RMS of that ratio over time (instrument / transparency drifts divide out).

    Catalog non-variable filtering (VSX / Gaia) is handled earlier when building ``comparison_stars.csv``; this
    step is the **photometric** sanity check among the selected ensemble.

    Returns summary counts and per-star metrics; optionally writes ``output_report_csv``.
    """
    import numpy as np

    comp_path = Path(comparison_stars_csv)
    if not comp_path.is_file():
        return {"error": f"missing {comp_path}", "rows": []}

    _hdr_comp = pd.read_csv(comp_path, nrows=0)
    _dtype_comp: dict[str, type] = {}
    _nc = str(name_col)
    if _nc in _hdr_comp.columns:
        _dtype_comp[_nc] = str
    if "catalog_id" in _hdr_comp.columns:
        _dtype_comp["catalog_id"] = str
    if "name" in _hdr_comp.columns:
        _dtype_comp["name"] = str
    comp_df = pd.read_csv(comp_path, dtype=_dtype_comp)
    if name_col not in comp_df.columns:
        return {"error": f"comparison table missing column {name_col!r}", "rows": []}
    names = [str(x).strip() for x in comp_df[name_col].dropna().astype(str).unique() if str(x).strip()]
    if not names:
        return {"error": "no comparison star names", "rows": []}

    by_jd: dict[float, dict[str, float]] = {}
    root = Path(frames_root)
    files_n = 0
    _cfg_for_workers = AppConfig()
    for fp in sorted(_iter_fits_recursive(root)):
        sidecar = proc_csv_path_for_aligned_fits(fp)
        if not sidecar.is_file():
            continue
        meta = extract_fits_metadata(fp, app_config=_cfg_for_workers)
        jd = float(meta.get("jd_start") or 0.0)
        if jd <= 0.0:
            continue
        try:
            _hdr_sc = pd.read_csv(sidecar, nrows=0)
            _dtype_sc: dict[str, type] = {str(name_col): str}
            if "catalog_id" in _hdr_sc.columns:
                _dtype_sc["catalog_id"] = str
            if "name" in _hdr_sc.columns:
                _dtype_sc["name"] = str
            dff = pd.read_csv(sidecar, dtype=_dtype_sc)
        except Exception as exc:  # noqa: BLE001
            from except_fix_counters import get_except_fix_counters

            get_except_fix_counters().stress_sidecar_skip += 1
            LOGGER.error("[STRESS] sidecar CSV read skip %s: %s", sidecar.name, exc)
            continue
        if name_col not in dff.columns or flux_col not in dff.columns:
            continue
        files_n += 1
        rowmap: dict[str, float] = {}
        for nm in names:
            m = dff.loc[dff[name_col].astype(str).str.strip() == nm]
            if m.empty:
                continue
            fl = float(m.iloc[0][flux_col])
            if np.isfinite(fl) and fl > 0:
                rowmap[nm] = fl
        if len(rowmap) >= int(min_stars_per_frame):
            by_jd[jd] = rowmap

    rel_lists: dict[str, list[float]] = {nm: [] for nm in names}
    for _jd, rowmap in sorted(by_jd.items(), key=lambda t: t[0]):
        vals = np.array(list(rowmap.values()), dtype=np.float64)
        med = float(np.median(vals))
        if not np.isfinite(med) or med <= 0:
            continue
        for nm, fl in rowmap.items():
            rel_lists[nm].append(float(fl / med))

    rows_out: list[dict[str, Any]] = []
    n_pass = 0
    n_fail = 0
    for nm in names:
        arr = np.array(rel_lists[nm], dtype=np.float64)
        n_fr = int(len(arr))
        if n_fr < int(min_frames_per_star):
            rows_out.append(
                {
                    "name": nm,
                    "n_frames": n_fr,
                    "relative_rms": None,
                    "flatness_ok": False,
                    "reason": "too_few_frames",
                }
            )
            n_fail += 1
            continue
        rms = float(np.sqrt(np.mean((arr - 1.0) ** 2)))
        ok = rms <= float(max_relative_rms)
        rows_out.append(
            {
                "name": nm,
                "n_frames": n_fr,
                "relative_rms": rms,
                "flatness_ok": ok,
                "reason": "" if ok else "high_rms",
            }
        )
        if ok:
            n_pass += 1
        else:
            n_fail += 1

    rep_path: str | None = None
    if output_report_csv is not None:
        outp = Path(output_report_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_out).to_csv(outp, index=False)
        rep_path = str(outp)

    return {
        "comparison_stars_csv": str(comp_path),
        "frames_sampled": int(files_n),
        "frames_used_ensemble": int(len(by_jd)),
        "n_comparison_names": int(len(names)),
        "n_pass_flatness": int(n_pass),
        "n_fail_flatness": int(n_fail),
        "max_relative_rms_threshold": float(max_relative_rms),
        "min_frames_per_star": int(min_frames_per_star),
        "rows": rows_out,
        "report_csv": rep_path,
    }
