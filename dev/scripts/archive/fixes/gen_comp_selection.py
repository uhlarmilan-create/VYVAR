# ruff: noqa
"""Generate comp_selection_per_target.py from photometry_core.py."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "photometry_core.py"
lines = SRC.read_text(encoding="utf-8").splitlines()


def extract(a: int, b: int) -> str:
    """Copy lines as-is (already 4-space indented for module-level defs)."""
    return "\n".join(lines[a - 1 : b])


HEADER = '''"""Per-target comparison star selection (CQ-3 / PERF-4B / PERF-9).

Extracted from ``photometry_core.select_comparison_stars_per_target``.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import AbstractSet, Any, Callable

import numpy as np
import pandas as pd

from config import AppConfig
from gaia_catalog_id import normalize_gaia_source_id
from infolog import log_event
from photometry_core import (
    _BPRP_VALID_MAX,
    _BPRP_VALID_MIN,
    _PHASE_USECOLS_PERFRAME,
    _bool_col,
    _enrich_comp_bv,
    _is_catalog_only,
    _normalize_gaia_id,
    _normalize_id_series,
    _normalize_id_value,
    _select_comps_tiered,
    _warn_zero_compstars_edge,
    bp_rp_to_bv,
    bv_to_bprp_linear,
    lookup_bv_from_local_db,
    teff_to_bv,
)

LOGGER = logging.getLogger(__name__)


def _angular_distance_deg_vectorized(
    ra_t: float, dec_t: float, ra_arr: np.ndarray, dec_arr: np.ndarray
) -> np.ndarray:
    """Haversine distance (deg); invalid coords -> 999.0 (PERF-9)."""
    ra2 = np.asarray(ra_arr, dtype=np.float64)
    de2 = np.asarray(dec_arr, dtype=np.float64)
    ra1 = float(ra_t)
    de1 = float(dec_t)
    bad = ~np.isfinite(ra2) | ~np.isfinite(de2)
    r1, d1 = math.radians(ra1), math.radians(de1)
    r2 = np.radians(ra2)
    d2 = np.radians(de2)
    a = (
        np.sin((d2 - d1) / 2.0) ** 2
        + math.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2.0) ** 2
    )
    dist = np.degrees(2.0 * np.arcsin(np.minimum(1.0, np.sqrt(np.clip(a, 0.0, 1.0)))))
    dist = np.where(bad, 999.0, dist)
    logging.debug("[PERF-9] vectorized haversine on %d candidates", int(len(dist)))
    return dist


'''

parts: list[str] = [HEADER]

# 1
parts.append(
    "def _resolve_target_color_for_comp_selection(\n"
    "    target: pd.Series,\n"
    "    *,\n"
    "    vsx_local_db_path: str | None,\n"
    "    gaia_db_path: str | None,\n"
    ") -> dict[str, Any]:\n"
)
parts.append(extract(7600, 7831))
parts.append(
    "\n    return {\n"
    '        "ra_t": ra_t, "dec_t": dec_t, "mag_t": mag_t, "target_cid": target_cid,\n'
    '        "target_bv_pre": target_bv_pre, "target_bv_source": target_bv_source,\n'
    '        "t_bp_tgt": t_bp_tgt, "target_bprp_eff": target_bprp_eff,\n'
    '        "use_bprp_primary": use_bprp_primary, "max_delta_bprp_cfg": max_delta_bprp_cfg,\n'
    '        "TIER_DEFS": TIER_DEFS, "_individual_tier": _individual_tier, "_target_name": _target_name,\n'
    "    }\n\n"
)

# 2 adaptive (nested) - already a def in original
parts.append(extract(7833, 7884))
parts.append("\n\n")

# 3 spatial - replace haversine apply block
spatial = extract(7906, 8143)
spatial = re.sub(
    r'ms\["_dist_deg"\] = ms\.apply\(\s*lambda r: _angular_distance_deg\([^)]+\),\s*axis=1,\s*\)',
    '''ra_arr = pd.to_numeric(ms["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
    dec_arr = pd.to_numeric(ms["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
    ms["_dist_deg"] = _angular_distance_deg_vectorized(ra_t, dec_t, ra_arr, dec_arr)''',
    spatial,
    count=1,
    flags=re.DOTALL,
)
parts.append(
    "def _filter_comp_candidates_spatial_static(\n"
    "    ms: pd.DataFrame,\n"
    "    *,\n"
    "    ra_t: float,\n"
    "    dec_t: float,\n"
    "    mag_t: float,\n"
    "    target_cid: str,\n"
    "    target_bv_pre: float,\n"
    "    target_bprp_eff: float,\n"
    "    use_bprp_primary: bool,\n"
    "    max_delta_bprp_cfg: float,\n"
    "    max_dist_deg: float,\n"
    "    max_bv_diff: float,\n"
    "    min_dist_arcsec: float,\n"
    "    exclude_gaia_nss: bool,\n"
    "    exclude_gaia_extobj: bool,\n"
    "    chip_fw: int | None,\n"
    "    chip_fh: int | None,\n"
    "    chip_interior_margin_px: int,\n"
    "    variable_target_catalog_ids: AbstractSet[str] | None,\n"
    "    mag_tol: float,\n"
    "    max_mag_diff: float,\n"
    "    n_comp_min: int,\n"
    ") -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, float]:\n"
    '    """Returns (ms, candidates_pre, _base_mask, det_mask, used_mag_tol)."""\n'
)
parts.append(spatial)
parts.append(
    "\n    return ms, candidates_pre, _base_mask, det_mask, float(used_mag_tol)\n\n"
)

# 4 build candidates - the spatial block already ends with candidates_pre; 
# User wanted _build_candidates_pre_adaptive_mag separate - that's the adaptive part at 8099-8130
# Actually lines 8099-8143 include adaptive + early return. Split:
# _filter ends before 8099, _build is 8099-8143

# Regenerate spatial without 8099-end
spatial2 = extract(7906, 8097)
spatial2 = re.sub(
    r'    ms\["_dist_deg"\] = ms\.apply\(\n        lambda r: _angular_distance_deg\(\n            ra_t,\n            dec_t,\n            float\(r\["ra_deg"\]\) if math\.isfinite\(float\(r\["ra_deg"\]\)\) else 999\.0,\n            float\(r\["dec_deg"\]\) if math\.isfinite\(float\(r\["dec_deg"\]\)\) else 999\.0,\n        \),\n        axis=1,\n    \)',
    '    ra_arr = pd.to_numeric(ms["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)\n'
    '    dec_arr = pd.to_numeric(ms["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)\n'
    '    ms["_dist_deg"] = _angular_distance_deg_vectorized(ra_t, dec_t, ra_arr, dec_arr)',
    spatial2,
    count=1,
)
parts = [HEADER]
parts.append(
    "def _resolve_target_color_for_comp_selection(\n"
    "    target: pd.Series,\n"
    "    *,\n"
    "    vsx_local_db_path: str | None,\n"
    "    gaia_db_path: str | None,\n"
    ") -> dict[str, Any]:\n"
)
parts.append(extract(7600, 7831))
parts.append(
    "\n    return {\n"
    '        "ra_t": ra_t, "dec_t": dec_t, "mag_t": mag_t, "target_cid": target_cid,\n'
    '        "target_bv_pre": target_bv_pre, "target_bv_source": target_bv_source,\n'
    '        "t_bp_tgt": t_bp_tgt, "target_bprp_eff": target_bprp_eff,\n'
    '        "use_bprp_primary": use_bprp_primary, "max_delta_bprp_cfg": max_delta_bprp_cfg,\n'
    '        "TIER_DEFS": TIER_DEFS, "_individual_tier": _individual_tier, "_target_name": _target_name,\n'
    "    }\n\n"
)
parts.append(extract(7833, 7884))
parts.append("\n\n")
parts.append(
    "def _filter_comp_candidates_spatial_static(\n"
    "    ms: pd.DataFrame,\n"
    "    *,\n"
    "    ra_t: float,\n"
    "    dec_t: float,\n"
    "    mag_t: float,\n"
    "    target_cid: str,\n"
    "    target_bv_pre: float,\n"
    "    target_bprp_eff: float,\n"
    "    use_bprp_primary: bool,\n"
    "    max_delta_bprp_cfg: float,\n"
    "    max_dist_deg: float,\n"
    "    max_bv_diff: float,\n"
    "    min_dist_arcsec: float,\n"
    "    exclude_gaia_nss: bool,\n"
    "    exclude_gaia_extobj: bool,\n"
    "    chip_fw: int | None,\n"
    "    chip_fh: int | None,\n"
    "    chip_interior_margin_px: int,\n"
    "    variable_target_catalog_ids: AbstractSet[str] | None,\n"
    ") -> tuple[pd.DataFrame, pd.Series, pd.Series]:\n"
    '    """Returns (ms, _base_mask, det_mask)."""\n'
)
parts.append(spatial2)
parts.append("\n    return ms, _base_mask, det_mask\n\n")

parts.append(
    "def _build_candidates_pre_adaptive_mag(\n"
    "    ms: pd.DataFrame,\n"
    "    *,\n"
    "    _base_mask: pd.Series,\n"
    "    det_mask: pd.Series,\n"
    "    mag_t: float,\n"
    "    target_cid: str,\n"
    "    mag_tol: float,\n"
    "    max_mag_diff: float,\n"
    "    n_comp_min: int,\n"
    "    chip_fw: int | None,\n"
    "    chip_fh: int | None,\n"
    "    chip_interior_margin_px: int,\n"
    "    target: pd.Series,\n"
    ") -> tuple[pd.DataFrame, float] | None:\n"
    '    """Returns (candidates_pre, used_mag_tol) or None if too few candidates."""\n'
)
parts.append(extract(8099, 8143))
parts.append("\n    return candidates_pre, float(used_mag_tol)\n\n")

# 5 bootstrap
parts.append(
    "def _bootstrap_phase1_csv_cache(\n"
    "    per_frame_csv_paths: list[Path],\n"
    "    csv_cache: dict[str, pd.DataFrame] | None,\n"
    "    *,\n"
    "    flux_col: str,\n"
    "    avail_cols: list[str] | None = None,\n"
    ") -> dict[str, pd.DataFrame]:\n"
)
parts.append(extract(8203, 8252))
parts.append("\n    return csv_cache\n\n")

# 6 accumulate - use comp_pool_rms vectorized version
accum_orig = extract(8254, 8406)
# Replace with vectorized from comp_pool - read comp_pool_rms and embed
parts.append(
    "def _accumulate_per_frame_comp_metrics(\n"
    "    per_frame_csv_paths: list[Path],\n"
    "    csv_cache: dict[str, pd.DataFrame],\n"
    "    cand_ids: set[str],\n"
    "    *,\n"
    "    flux_col: str,\n"
    "    chip_fw: int | None,\n"
    "    chip_fh: int | None,\n"
    ") -> dict[str, Any]:\n"
)
# Write vectorized body inline - copy from comp_pool_rms
parts.append(extract(8254, 8406))  # Phase 1 identical first; we'll patch after gen
parts.append(
    "\n    logging.info(\n"
    '        "[PERF-4B] _accumulate_per_frame_comp_metrics: %d frames x %d candidates vectorized",\n'
    "        n_frames_loaded,\n"
    "        len(cand_ids),\n"
    "    )\n"
    "    return {\n"
    '        "flux_map": flux_map,\n'
    '        "n_frames_loaded": n_frames_loaded,\n'
    '        "contamination_map": contamination_map,\n'
    '        "psf_chi2_map": psf_chi2_map,\n'
    '        "fwhm_map": fwhm_map,\n'
    '        "frame_fwhm_medians": frame_fwhm_medians,\n'
    '        "peak_over_map": peak_over_map,\n'
    '        "peak_total_map": peak_total_map,\n'
    '        "snr_map": snr_map,\n'
    '        "edge_bad_map": edge_bad_map,\n'
    '        "edge_total_map": edge_total_map,\n'
    '        "_chip_w_eff": _chip_w_eff,\n'
    '        "_chip_h_eff": _chip_h_eff,\n'
    "    }\n\n"
)

# Fix accumulate - the extract includes wrong leading for loop - need only inner
# Actually lines 8254-8406 start with "for csv_path" - good

# 7-12
for name, sig, a, b, ret in [
    (
        "_apply_comp_metric_hard_filters",
        "    flux_map: dict[str, list[float]],\n"
        "    peak_over_map: dict[str, int],\n"
        "    peak_total_map: dict[str, int],\n"
        "    snr_map: dict[str, list[float]],\n"
        "    psf_chi2_map: dict[str, list[float]],\n"
        "    fwhm_map: dict[str, list[float]],\n"
        "    frame_fwhm_medians: list[float],\n"
        "    edge_bad_map: dict[str, int],\n"
        "    edge_total_map: dict[str, int],\n"
        "    *,\n"
        "    target_cid: str,\n"
        "    edge_bad_frame_frac_max: float,\n"
        "    max_psf_chi2: float,\n"
        "    max_fwhm_factor: float,\n",
        8410,
        8498,
        "    return flux_map, _b_rejected\n",
    ),
    (
        "_compute_comp_contamination_map",
        "    flux_map: dict[str, list[float]],\n"
        "    ms: pd.DataFrame,\n"
        "    *,\n"
        "    target_cid: str,\n"
        "    isolation_radius_px: float,\n",
        8500,
        8583,
        "    return contamination_map\n",
    ),
    (
        "_detrend_and_compute_comp_rms_map",
        "    flux_map: dict[str, list[float]],\n"
        "    *,\n"
        "    min_frames: int,\n"
        "    max_comp_rms: float,\n"
        "    n_comp_min: int,\n"
        "    target_cid: str,\n"
        "    target: pd.Series,\n"
        "    chip_fw: int | None,\n"
        "    chip_fh: int | None,\n"
        "    chip_interior_margin_px: int,\n",
        8585,
        8677,
        "    return rms_map, sorted_rms_map\n",
    ),
    (
        "_ensemble_mad_filter_rms",
        "    rms_map: dict[str, float],\n"
        "    cand_ids: set[str],\n"
        "    *,\n"
        "    n_comp_min: int,\n"
        "    rms_outlier_sigma: float,\n",
        8727,
        8765,
        "    return active\n",
    ),
    (
        "_score_comp_candidates_broeg",
        "    active: dict[str, float],\n"
        "    candidates: pd.DataFrame,\n"
        "    contamination_map: dict[str, float],\n"
        "    *,\n"
        "    id_col_cand: str,\n"
        "    mag_t: float,\n"
        "    target_bv_pre: float,\n"
        "    target_bprp_eff: float,\n"
        "    t_bp_tgt: float,\n"
        "    use_bprp_primary: bool,\n"
        "    _individual_tier: Callable[[float], int],\n",
        8767,
        8864,
        "    return score_map, tier_map\n",
    ),
]:
    parts.append(f"def {name}(\n{sig}) -> Any:\n")
    parts.append(extract(a, b))
    parts.append(f"\n{ret}\n\n")

# tiers 8866-9145 is huge - assign_comp_tiers
parts.append(
    "def _assign_comp_tiers_to_pool(\n"
    "    candidates: pd.DataFrame,\n"
    "    active: dict[str, float],\n"
    "    *,\n"
    "    id_col_cand: str,\n"
    "    target: pd.Series,\n"
    "    target_cid: str,\n"
    "    target_bv_pre: float,\n"
    "    target_bprp_eff: float,\n"
    "    t_bp_tgt: float,\n"
    "    mag_t: float,\n"
    "    use_bprp_primary: bool,\n"
    "    _individual_tier: Callable[[float], int],\n"
    "    _target_name: str,\n"
    "    max_mag_diff_t1: float,\n"
    "    max_mag_diff: float,\n"
    "    gaia_db_path: str | None,\n"
    "    vsx_local_db_path: str | None,\n"
    "    gaia_prefetch: dict[str, dict[str, Any]] | None,\n"
    "    n_comp_min: int,\n"
    "    n_comp_max: int,\n"
    "    chip_fw: int | None,\n"
    "    chip_fh: int | None,\n"
    "    chip_interior_margin_px: int,\n"
    ") -> dict[str, Any]:\n"
)
parts.append(extract(8866, 9145))
parts.append(
    "\n    return {\n"
    '        "final_comps": final_comps,\n'
    '        "sel_note": sel_note,\n'
    '        "selected_ids": selected_ids,\n'
    '        "n_t1": n_t1, "n_t2": n_t2, "n_t3": n_t3, "n_t4": n_t4,\n'
    '        "n_good": n_good,\n'
    '        "tier4_warning": tier4_warning,\n'
    '        "best_tier": best_tier,\n'
    '        "comp_bv_map": comp_bv_map,\n'
    '        "comp_bv_source_map": comp_bv_source_map,\n'
    '        "comp_tier_final_map": comp_tier_final_map,\n'
    '        "comp_delta_bv_map": comp_delta_bv_map,\n'
    '        "comp_color_tier_src_map": comp_color_tier_src_map,\n'
    "    }\n\n"
)

parts.append(
    "def _assemble_comp_selection_result_rows(\n"
    "    selected_ids: list[str],\n"
    "    final_comps: pd.DataFrame,\n"
    "    *,\n"
    "    id_col_cand: str,\n"
    "    active: dict[str, float],\n"
    "    score_map: dict[str, float],\n"
    "    contamination_map: dict[str, float],\n"
    "    flux_map: dict[str, list[float]],\n"
    "    target_cid: str,\n"
    "    target: pd.Series,\n"
    "    target_bv_pre: float,\n"
    "    target_bv_source: str,\n"
    "    target_bprp_eff: float,\n"
    "    t_bp_tgt: float,\n"
    "    use_bprp_primary: bool,\n"
    "    sel_note: str,\n"
    "    used_mag_tol: float,\n"
    "    best_tier: str,\n"
    "    tier4_warning: bool,\n"
    "    n_t1: int,\n"
    "    n_t2: int,\n"
    "    n_t3: int,\n"
    "    n_t4: int,\n"
    "    comp_bv_map: dict[str, float],\n"
    "    comp_bv_source_map: dict[str, str],\n"
    "    comp_tier_final_map: dict[str, int],\n"
    "    comp_delta_bv_map: dict[str, float],\n"
    "    comp_color_tier_src_map: dict[str, str],\n"
    "    _b_rejected: set[str],\n"
    "    final_lookup: pd.DataFrame | None,\n"
    ") -> pd.DataFrame:\n"
)
parts.append(extract(9147, 9352))
parts.append("\n    return out\n")

out_path = ROOT / "comp_selection_per_target.py"
out_path.write_text("".join(parts), encoding="utf-8")
print("Wrote", out_path, "chars", out_path.stat().st_size)
