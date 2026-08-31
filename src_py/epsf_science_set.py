"""Shared ePSF science-set builder (measurement + dashboard scope).

Science set per DECISIONS / EPSF-VALID-02 P1-C: active targets excluding
catalog_only, per-target SELECTED comps, check stars, and blended sample.
One definition consumed by F1 (dashboard) and F3 (PSF photometry IDs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from gaia_catalog_id import normalize_gaia_source_id, read_vyvar_csv


def _coerce_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    t = str(raw).strip().lower()
    return t in ("1", "true", "t", "yes", "y")


def _norm_id_gaia_or_raw(raw: Any) -> str:
    """Gaia canonical id; on exception return stripped raw. Differs from astrometry's direct normalize."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    try:
        return str(normalize_gaia_source_id(s)).strip()
    except Exception:  # noqa: BLE001
        return s


@dataclass(frozen=True)
class EpsfScienceSetResult:
    """Census of the ePSF science measurement set."""

    catalog_ids: frozenset[str]
    n_targets: int = 0
    n_per_target_comps: int = 0
    n_check_stars: int = 0
    n_blended: int = 0
    empty_reason: str | None = None
    composition: dict[str, int] = field(default_factory=dict)

    @property
    def n_total(self) -> int:
        return len(self.catalog_ids)

    def to_meta_dict(self) -> dict[str, Any]:
        return {
            "n_total": self.n_total,
            "n_targets": self.n_targets,
            "n_per_target_comps": self.n_per_target_comps,
            "n_check_stars": self.n_check_stars,
            "n_blended": self.n_blended,
            "composition": dict(self.composition),
            "empty_reason": self.empty_reason,
        }


def build_epsf_science_set(platesolve_dir: Path | str) -> EpsfScienceSetResult:
    """Build the science set for ePSF measurement and dashboard scope."""
    ps = Path(platesolve_dir)
    phot = ps / "photometry"

    targets: set[str] = set()
    at_p = phot / "active_targets.csv"
    if at_p.is_file():
        try:
            at = read_vyvar_csv(at_p, low_memory=False)
            if "catalog_id" in at.columns:
                for _, row in at.iterrows():
                    z = str(row.get("zone_flag", row.get("zone", "")) or "").strip().lower()
                    if z == "catalog_only":
                        continue
                    cid = _norm_id_gaia_or_raw(row.get("catalog_id"))
                    if cid:
                        targets.add(cid)
        except Exception as exc:  # noqa: BLE001
            return EpsfScienceSetResult(
                frozenset(),
                empty_reason=f"active_targets.csv unreadable: {exc}",
            )
    else:
        return EpsfScienceSetResult(
            frozenset(),
            empty_reason=f"missing {at_p}",
        )

    per_target_comps: set[str] = set()
    cpt = phot / "comparison_stars_per_target.csv"
    if cpt.is_file():
        try:
            cdf = read_vyvar_csv(cpt, low_memory=False)
            if "catalog_id" in cdf.columns:
                for raw in cdf["catalog_id"].fillna("").astype(str):
                    cid = _norm_id_gaia_or_raw(raw)
                    if cid:
                        per_target_comps.add(cid)
        except Exception as exc:  # noqa: BLE001
            return EpsfScienceSetResult(
                frozenset(),
                empty_reason=f"comparison_stars_per_target.csv unreadable: {exc}",
            )
    else:
        return EpsfScienceSetResult(
            frozenset(),
            empty_reason=f"missing {cpt}",
        )

    check_stars: set[str] = set()
    ps_p = phot / "photometry_summary.csv"
    if ps_p.is_file():
        try:
            psdf = read_vyvar_csv(ps_p, low_memory=False)
            if "is_check_star" in psdf.columns and "catalog_id" in psdf.columns:
                chk = psdf[psdf["is_check_star"].map(_coerce_bool)]
                for raw in chk["catalog_id"].fillna("").astype(str):
                    cid = _norm_id_gaia_or_raw(raw)
                    if cid:
                        check_stars.add(cid)
        except Exception:  # noqa: BLE001
            pass

    blended: set[str] = set()
    ms_p = ps / "masterstars_full_match.csv"
    if ms_p.is_file():
        try:
            ms = pd.read_csv(ms_p, low_memory=False, dtype={"catalog_id": str})
            blend_mask = pd.Series(False, index=ms.index)
            if "source_state" in ms.columns:
                blend_mask |= ms["source_state"].astype(str).str.upper().eq("BLENDED")
            if "is_blended" in ms.columns:
                blend_mask |= ms["is_blended"].map(_coerce_bool)
            for raw in ms.loc[blend_mask, "catalog_id"].fillna("").astype(str):
                cid = _norm_id_gaia_or_raw(raw)
                if cid:
                    blended.add(cid)
        except Exception:  # noqa: BLE001
            pass

    all_ids = targets | per_target_comps | check_stars | blended
    comp_only = per_target_comps - targets
    check_only = check_stars - targets - per_target_comps
    blend_only = blended - targets - per_target_comps - check_stars

    composition = {
        "targets": len(targets),
        "per_target_comps_only": len(comp_only),
        "check_stars_only": len(check_only),
        "blended_only": len(blend_only),
    }

    empty_reason: str | None = None
    if not all_ids:
        empty_reason = "no catalog_ids after merging targets/comps/checks/blended"

    return EpsfScienceSetResult(
        catalog_ids=frozenset(all_ids),
        n_targets=len(targets),
        n_per_target_comps=len(per_target_comps),
        n_check_stars=len(check_stars),
        n_blended=len(blended),
        empty_reason=empty_reason,
        composition=composition,
    )
