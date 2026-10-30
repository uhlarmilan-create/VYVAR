"""The stars RUN ePSF fits (EPSF-TARGETS-2026-10): the selected catalogue-variable targets with aperture
photometry and their comparison stars (``build_epsf_fit_set``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from gaia_catalog_id import normalize_gaia_source_id, read_vyvar_csv
from stats_core import _coerce_bool


EPSF_SELECT_ALL = "ALL"


def parse_epsf_selection(selection: str | None) -> list[str]:
    """Selection tokens: comma / semicolon / newline separated (star names contain spaces). Empty or ``ALL``
    -> [] (every candidate)."""
    raw = str(selection or "").strip()
    if not raw or raw.upper() == EPSF_SELECT_ALL:
        return []
    import re  # noqa: PLC0415

    return [t.strip() for t in re.split(r"[,;\n]+", raw) if t.strip()]


def _token_matches(token: str, vsx_type: str, names: list[str], cid: str) -> bool:
    """A token selects a target by VSX variability type (``EA`` matches ``EA``, ``EA:``, ``EA|RS``; ``*`` and
    ``?`` are wildcards on the TYPE only: ``E*`` = EA, EB, EW, ...), or by the exact star name / VSX name /
    catalog id (Milan 2026-10-29, G09-42: a wildcard never selects by name - ``E*`` must not pick EZ Lyn).
    Case-insensitive."""
    from fnmatch import fnmatchcase  # noqa: PLC0415

    from vsx_type_scope import normalize_vsx_type_token, tokenize_vsx_type  # noqa: PLC0415

    t = token.strip().upper()
    if not t:
        return False
    if any(fnmatchcase(tok, normalize_vsx_type_token(t)) for tok in tokenize_vsx_type(vsx_type)):
        return True
    if any(ch in t for ch in "*?["):
        return False
    return any(n and n.strip().upper() == t for n in [*names, cid])


@dataclass(frozen=True)
class EpsfFitSet:
    """EPSF-TARGETS-2026-10: the stars RUN ePSF fits - the selected catalogue-variable targets and their
    comparison stars."""

    target_ids: frozenset[str]
    comp_ids: frozenset[str]
    target_types: dict[str, str] = field(default_factory=dict)
    target_names: dict[str, str] = field(default_factory=dict)
    selection: str = EPSF_SELECT_ALL
    unmatched_tokens: tuple[str, ...] = ()
    n_candidates: int = 0
    empty_reason: str | None = None

    @property
    def catalog_ids(self) -> frozenset[str]:
        return self.target_ids | self.comp_ids

    def to_meta_dict(self) -> dict[str, Any]:
        return {
            "rule": "targets with aperture photometry (VSX, exoplanet hosts, manual; not out of scope, not skipped) matching the selection "
            "+ their comparison stars",
            "selection": self.selection,
            "unmatched_tokens": list(self.unmatched_tokens),
            "n_candidates": self.n_candidates,
            "n_targets": len(self.target_ids),
            "n_comps": len(self.comp_ids - self.target_ids),
            "n_total": len(self.catalog_ids),
            "target_types": dict(self.target_types),
            "target_names": dict(self.target_names),
            "empty_reason": self.empty_reason,
        }


def epsf_target_candidates(platesolve_dir: Path | str) -> pd.DataFrame:
    """Targets RUN ePSF can measure: every target with an aperture light curve - VSX variables, exoplanet
    hosts, manual targets (EPSF-ALL-TARGETS-2026-10, Milan 2026-10-23: the photometry completeness gate
    wants a PSF light curve for every aperture light curve, and the ePSF joint fit matters most for a
    target next to a brighter star). Targets out of the configured VSX type scope
    (``vsx_out_of_scope_types``) or skipped for S/N or saturation have no aperture light curve and are not
    candidates; nor is the ``catalog_only`` zone. An exoplanet host without a VSX type is typed
    ``EXOPLANET`` for the type selection. Columns: catalog_id, name, vsx_type, mag."""
    from report_methods import lc_csv_path  # noqa: PLC0415

    phot = Path(platesolve_dir) / "photometry"
    at_p = phot / "active_targets.csv"
    cols = ["catalog_id", "name", "vsx_type", "mag"]
    if not at_p.is_file():
        return pd.DataFrame(columns=cols)
    at = read_vyvar_csv(at_p, low_memory=False)
    rows = []
    for _, row in at.iterrows():
        if _coerce_bool(row.get("skip_photometry")):
            continue
        if str(row.get("zone_flag", row.get("zone", "")) or "").strip().lower() == "catalog_only":
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


def build_epsf_fit_set(platesolve_dir: Path | str, selection: str | None = EPSF_SELECT_ALL) -> EpsfFitSet:
    """Selected targets (``epsf_target_candidates`` filtered by ``selection``: ``ALL`` or comma-separated VSX
    types, names, catalog ids, ``*`` wildcards; Milan 2026-10-19) and their comparison stars."""
    phot = Path(platesolve_dir) / "photometry"
    sel = str(selection or "").strip() or EPSF_SELECT_ALL
    cpt = phot / "comparison_stars_per_target.csv"
    if not cpt.is_file():
        return EpsfFitSet(frozenset(), frozenset(), selection=sel, empty_reason=f"missing {cpt}")
    cand = epsf_target_candidates(platesolve_dir)
    tokens = parse_epsf_selection(sel)
    types: dict[str, str] = {}
    names: dict[str, str] = {}
    hit_tokens: set[str] = set()
    for _, r in cand.iterrows():
        cid, vt, nm = str(r["catalog_id"]), str(r["vsx_type"]), str(r["name"])
        if tokens:
            hits = [t for t in tokens if _token_matches(t, vt, [nm], cid)]
            if not hits:
                continue
            hit_tokens.update(hits)
        types[cid], names[cid] = vt, nm
    from comp_table import load_comp_table, used  # noqa: PLC0415

    cdf = used(load_comp_table(cpt))
    comps: set[str] = set()
    if {"catalog_id", "target_catalog_id"}.issubset(cdf.columns):
        tcol = cdf["target_catalog_id"].map(normalize_gaia_source_id)
        for raw in cdf.loc[tcol.isin(types), "catalog_id"]:
            cid = normalize_gaia_source_id(raw)
            if cid:
                comps.add(cid)
    if types:
        empty = None
    elif cand.empty:
        empty = "no target with an aperture light curve"
    else:
        empty = f"no target matches the selection '{sel}'"
    return EpsfFitSet(
        frozenset(types), frozenset(comps), target_types=types, target_names=names, selection=sel,
        unmatched_tokens=tuple(t for t in tokens if t not in hit_tokens), n_candidates=len(cand), empty_reason=empty,
    )
