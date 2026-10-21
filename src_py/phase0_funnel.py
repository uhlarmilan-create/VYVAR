"""Phase 0 funnel fingerprints for anchor gate and observability."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def _counter_dict(series: pd.Series) -> dict[str, int]:
    c = Counter(str(v).strip() if pd.notna(v) else "" for v in series.tolist())
    return {str(k): int(v) for k, v in sorted(c.items())}


def compute_phase0_funnel_fingerprint(
    variable_targets_csv: Path,
    active_targets_csv: Path | None = None,
) -> dict[str, Any]:
    """Row counts and histograms for anchor / invariant gates."""
    vt_path = Path(variable_targets_csv)
    out: dict[str, Any] = {
        "variable_targets_rows": 0,
        "gaia_match_source_histogram": {},
        "active_targets_rows": 0,
        "skip_reason_histogram": {},
        "skip_photometry_true": 0,
        "zone_flag_histogram": {},
    }
    if not vt_path.is_file():
        return out

    vt = pd.read_csv(vt_path, low_memory=False)
    out["variable_targets_rows"] = int(len(vt))
    if "gaia_match_source" in vt.columns:
        out["gaia_match_source_histogram"] = _counter_dict(vt["gaia_match_source"])

    if active_targets_csv is None:
        return out
    at_path = Path(active_targets_csv)
    if not at_path.is_file():
        return out

    at = pd.read_csv(at_path, low_memory=False)
    out["active_targets_rows"] = int(len(at))
    if "skip_reason" in at.columns:
        out["skip_reason_histogram"] = _counter_dict(at["skip_reason"])
    if "skip_photometry" in at.columns:
        out["skip_photometry_true"] = int(at["skip_photometry"].astype(bool).sum())
    if "zone_flag" in at.columns:
        out["zone_flag_histogram"] = _counter_dict(at["zone_flag"])
    return out


def compare_phase0_funnel_fingerprints(
    observed: dict[str, Any],
    expected: dict[str, Any],
) -> list[str]:
    """Return human-readable mismatch messages (empty = match)."""
    issues: list[str] = []
    for key in ("variable_targets_rows", "active_targets_rows", "skip_photometry_true"):
        obs = int(observed.get(key) or 0)
        exp = int(expected.get(key) or 0)
        if obs != exp:
            issues.append(f"{key}: observed={obs} expected={exp}")
    for hist_key in ("gaia_match_source_histogram", "skip_reason_histogram", "zone_flag_histogram"):
        obs_h = dict(observed.get(hist_key) or {})
        exp_h = dict(expected.get(hist_key) or {})
        if obs_h != exp_h:
            issues.append(f"{hist_key}: observed={obs_h} expected={exp_h}")
    return issues
