"""The stars RUN ePSF fits: every target with an aperture light curve and its comparison stars
(``build_epsf_fit_set``; EPSF-ALL-TARGETS-2026-10, S6-FIX EPSF-SCOPE-2026-10: no target selection)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from gaia_catalog_id import normalize_gaia_source_id, read_vyvar_csv
from stats_core import coerce_bool


@dataclass(frozen=True)
class EpsfFitSet:
    """The stars RUN ePSF fits - every target with an aperture light curve and their comparison stars."""

    target_ids: frozenset[str]
    comp_ids: frozenset[str]
    target_types: dict[str, str] = field(default_factory=dict)
    target_names: dict[str, str] = field(default_factory=dict)
    n_candidates: int = 0
    empty_reason: str | None = None

    @property
    def catalog_ids(self) -> frozenset[str]:
        return self.target_ids | self.comp_ids

    def to_meta_dict(self) -> dict[str, Any]:
        return {
            "rule": "targets with aperture photometry (VSX, exoplanet hosts, manual; not out of scope, not skipped) "
            "+ their comparison stars",
            "n_candidates": self.n_candidates,
            "n_targets": len(self.target_ids),
            "n_comps": len(self.comp_ids - self.target_ids),
            "n_total": len(self.catalog_ids),
            "target_types": dict(self.target_types),
            "target_names": dict(self.target_names),
            "empty_reason": self.empty_reason,
        }


def epsf_target_candidates(platesolve_dir: Path | str, out_of_scope_types: Iterable[str] | None) -> pd.DataFrame:
    """Targets RUN ePSF can measure: every target with an aperture light curve - VSX variables, exoplanet
    hosts, manual targets (EPSF-ALL-TARGETS-2026-10, Milan 2026-10-23: the photometry completeness gate
    wants a PSF light curve for every aperture light curve, and the ePSF joint fit matters most for a
    target next to a brighter star). Targets skipped for S/N or saturation have no aperture light curve and
    are not candidates; nor is the ``catalog_only`` zone. ``out_of_scope_types`` is the run's
    ``vsx_out_of_scope_types``: a target it excludes (``vsx_type_scope.target_is_out_of_scope``, the predicate of
    the aperture target selection) is never a candidate, also when an aperture light curve of it exists (S6-FIX
    EPSF-SCOPE-2026-10). An exoplanet host without a VSX type is typed ``EXOPLANET``. Columns: catalog_id, name,
    vsx_type, mag."""
    from vsx_type_scope import target_is_out_of_scope  # noqa: PLC0415
    from report_methods import lc_csv_path  # noqa: PLC0415

    phot = Path(platesolve_dir) / "photometry"
    at_p = phot / "active_targets.csv"
    cols = ["catalog_id", "name", "vsx_type", "mag"]
    if not at_p.is_file():
        return pd.DataFrame(columns=cols)
    at = read_vyvar_csv(at_p, low_memory=False)
    rows = []
    for _, row in at.iterrows():
        if coerce_bool(row.get("skip_photometry")):
            continue
        if str(row.get("zone_flag", row.get("zone", "")) or "").strip().lower() == "catalog_only":
            continue
        if target_is_out_of_scope(row, out_of_scope_types):
            continue
        cid = normalize_gaia_source_id(row.get("catalog_id"))
        if not cid or not lc_csv_path(phot / "lightcurves", cid, "aperture").is_file():
            continue
        vt = str(row.get("vsx_type") or "").strip()
        if vt.lower() in ("", "nan") and str(row.get("catalog") or "").strip().upper() == "EXOPLANET":
            vt = "EXOPLANET"
        name = str(row.get("vsx_name") or row.get("name") or "").strip()
        rows.append({"catalog_id": cid, "name": name if name.lower() != "nan" else cid,
                     "vsx_type": "" if vt.lower() == "nan" else vt, "mag": pd.to_numeric(row.get("mag"), errors="coerce")})
    return pd.DataFrame(rows, columns=cols).drop_duplicates("catalog_id").reset_index(drop=True)


def build_epsf_fit_set(platesolve_dir: Path | str, out_of_scope_types: Iterable[str] | None) -> EpsfFitSet:
    """Every target of ``epsf_target_candidates`` and its comparison stars."""
    phot = Path(platesolve_dir) / "photometry"
    cpt = phot / "comparison_stars_per_target.csv"
    if not cpt.is_file():
        return EpsfFitSet(frozenset(), frozenset(), empty_reason=f"missing {cpt}")
    cand = epsf_target_candidates(platesolve_dir, out_of_scope_types)
    types: dict[str, str] = {}
    names: dict[str, str] = {}
    for _, r in cand.iterrows():
        cid = str(r["catalog_id"])
        types[cid], names[cid] = str(r["vsx_type"]), str(r["name"])
    from comp_table import load_comp_table, used  # noqa: PLC0415

    cdf = used(load_comp_table(cpt))
    comps: set[str] = set()
    if {"catalog_id", "target_catalog_id"}.issubset(cdf.columns):
        tcol = cdf["target_catalog_id"].map(normalize_gaia_source_id)
        for raw in cdf.loc[tcol.isin(types), "catalog_id"]:
            cid = normalize_gaia_source_id(raw)
            if cid:
                comps.add(cid)
    return EpsfFitSet(
        frozenset(types), frozenset(comps), target_types=types, target_names=names, n_candidates=len(cand),
        empty_reason=None if types else "no target with an aperture light curve",
    )
