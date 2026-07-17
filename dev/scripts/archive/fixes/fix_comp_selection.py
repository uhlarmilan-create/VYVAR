# ruff: noqa
"""Post-process generated comp_selection_per_target.py."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "comp_selection_per_target.py"
text = path.read_text(encoding="utf-8")

# 1) Add ra/dec/mag/cid at start of _resolve
needle = ") -> dict[str, Any]:\n    target_bv_pre"
if needle in text:
    text = text.replace(
        needle,
        ") -> dict[str, Any]:\n"
        "    ra_t = float(target[\"ra_deg\"])\n"
        "    dec_t = float(target[\"dec_deg\"])\n"
        "    mag_t = float(\n"
        "        pd.to_numeric(\n"
        "            target.get(\n"
        '                "mag",\n'
        "                target.get(\n"
        '                    "phot_g_mean_mag",\n'
        '                    target.get("g_mag", target.get("gaia_g_mag", float("nan"))),\n'
        "                ),\n"
        "            ),\n"
        '            errors="coerce",\n'
        "        )\n"
        "    )\n"
        '    target_cid = str(target.get("catalog_id", ""))\n'
        "    target_bv_pre",
        1,
    )

# 2) Remove dead nested _adaptive after return in _resolve
dead_start = "    }\n\n    def _adaptive_mag_filter("
dead_end = "        return pool, float(mag_abs)\n\n\ndef _filter_comp_candidates_spatial_static"
if dead_start in text and dead_end in text:
    i0 = text.index(dead_start)
    i1 = text.index(dead_end) + len("        return pool, float(mag_abs)\n\n")
    text = text[: i0 + len("    }\n\n")] + text[i1:]

# 3) Insert module-level _adaptive before _filter if missing
if "\ndef _adaptive_mag_filter(" not in text:
    adaptive = '''
def _adaptive_mag_filter(
    all_candidates: pd.DataFrame,
    target_mag: float,
    mag_diff_start: float,
    mag_diff_absolute: float,
    n_comp_min: int,
    *,
    max_mag_diff: float,
    mag_diff_step: float = 0.25,
) -> tuple[pd.DataFrame, float]:
    """Postupne uvoľňuje Δmag limit kým nie je dostatok kandidátov."""
    if all_candidates is None or getattr(all_candidates, "empty", True):
        return pd.DataFrame(), float(mag_diff_start)
    target = float(target_mag)
    if not math.isfinite(target):
        return all_candidates, float(mag_diff_start)
    try:
        mag_abs = float(mag_diff_absolute)
    except Exception:  # noqa: BLE001
        mag_abs = 3.0
    if not math.isfinite(mag_abs) or mag_abs <= 0:
        mag_abs = 3.0
    mag_tol = float(mag_diff_start)
    if not math.isfinite(mag_tol) or mag_tol <= 0:
        mag_tol = float(max_mag_diff)
    if "mag" not in all_candidates.columns:
        all_candidates = all_candidates.copy()
        all_candidates["mag"] = pd.to_numeric(
            all_candidates.get("phot_g_mean_mag", pd.Series(index=all_candidates.index, dtype=float)),
            errors="coerce",
        )
    mags = pd.to_numeric(all_candidates["mag"], errors="coerce")
    while mag_tol <= mag_abs + 1e-9:
        pool = all_candidates[(mags - target).abs() <= mag_tol]
        if int(len(pool)) >= int(n_comp_min) * 2:
            return pool, float(mag_tol)
        if mag_tol >= mag_abs - 1e-9:
            return pool, float(mag_tol)
        mag_tol = min(float(mag_tol) + float(mag_diff_step), float(mag_abs))
    pool = all_candidates[(mags - target).abs() <= float(mag_abs)]
    return pool, float(mag_abs)


'''
    text = text.replace("\ndef _filter_comp_candidates_spatial_static", adaptive + "def _filter_comp_candidates_spatial_static", 1)

# 4) Fix _build early return
text = text.replace(
    "        return pd.DataFrame()\n    return candidates_pre, float(used_mag_tol)",
    "        return None\n    return candidates_pre, float(used_mag_tol)",
    1,
)

# 5) Fix _adaptive call to pass max_mag_diff
text = text.replace(
    "            n_comp_min=int(n_comp_min),\n            mag_diff_step=0.25,\n        )",
    "            n_comp_min=int(n_comp_min),\n            max_mag_diff=float(max_mag_diff),\n            mag_diff_step=0.25,\n        )",
    1,
)

# 6) Fix _detrend failure return
text = text.replace(
    "        return pd.DataFrame()\n    return rms_map, sorted_rms_map",
    "        return None, None\n    return rms_map, sorted_rms_map",
    1,
)

# 7) Fix _ensemble signature and failure return
text = text.replace(
    "def _ensemble_mad_filter_rms(\n    rms_map: dict[str, float],\n    cand_ids: set[str],\n    *,\n    n_comp_min: int,\n    rms_outlier_sigma: float,\n) -> Any:",
    "def _ensemble_mad_filter_rms(\n    rms_map: dict[str, float],\n    candidates: pd.DataFrame,\n    *,\n    target_cid: str,\n    target: pd.Series,\n    n_comp_min: int,\n    rms_outlier_sigma: float,\n    chip_fw: int | None,\n    chip_fh: int | None,\n    chip_interior_margin_px: int,\n) -> dict[str, float] | None:",
    1,
)
text = text.replace(
    "        return pd.DataFrame()\n    for _iter in range(10):",
    "        return None\n    for _iter in range(10):",
    1,
)

# 8) Fix _assign empty return
text = text.replace(
    "        return pd.DataFrame()\n    candidate_pool_df[\"comp_rms\"]",
    "        return {\"final_comps\": None, \"sel_note\": \"\", \"selected_ids\": [], \"n_t1\": 0, \"n_t2\": 0, \"n_t3\": 0, \"n_t4\": 0, \"n_good\": 0, \"tier4_warning\": False, \"best_tier\": \"TIER4\", \"comp_bv_map\": {}, \"comp_bv_source_map\": {}, \"comp_tier_final_map\": {}, \"comp_delta_bv_map\": {}, \"comp_color_tier_src_map\": {}}\n    candidate_pool_df[\"comp_rms\"]",
    1,
)

path.write_text(text, encoding="utf-8")
print("Patched", path)
