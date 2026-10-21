"""COMP-SMART-2026-10 (Milan 2026-10-11): replace suspect comparison stars by clean ones.

Phase 1 ranks the comparison candidates of a target (RMS -> colour -> distance, single
source, photometry_comp._select_comps_by_rms_then_color) and admits the first
``n_comp_max``; the next ones are kept as a reserve (``p1_role = reserve`` rows of
comparison_stars_per_target.csv, see comp_table).
Phase 2A measures all of them on the aperture grid. Before the ensemble is built the
stability test of Phase 2A (photometry_lightcurve.check_comparison_stability: Abbe
point-to-point scatter and night-long slope on the common-mode-detrended residuals) is run
on primary + reserve together, and the ensemble becomes

* the clean stars in rank order, at most ``n_comp_max`` (fewer is fine - "6 clean are better
  than 8 with suspects");
* only when fewer than ``n_comp_min`` are clean: the least suspect ones (smallest slope /
  p2p) are added up to ``n_comp_min`` and stay marked suspect (typical for very red targets
  with few comparison stars).

COMP-CATOFF-2026-10 (Milan 2026-10-21): before the stability test a candidate whose catalogue
magnitude disagrees with the others is excluded (and so replaced from the reserve like a suspect):
offset_j = median over frames of [(cat_j - m_j) - median_i(cat_i - m_i)] over all candidates with a
valid band magnitude (at least 3); |offset_j| > ``catalog_offset_max_mag`` (config
``comp_catalog_offset_max_mag``, 0 = off) -> excluded. One comparison with a fixed limit, no
iteration. Draft 537: Gaia DR3 1423532039421321728 measured 0.60 mag off its catalogue V and set
the zero point of four targets.

Literature: comparison stars must be constant; suspect ones are rejected and replaced from
the candidate list (Henden & Kaitchuck 1982; AAVSO CCD Photometry Guide ch. 5; Broeg et al.
2005 weights handle noise, not trends).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

#: reserve candidates kept per target in Phase 1
N_RESERVE = 8
#: fewest candidates with a valid band magnitude for the catalogue-offset test (median reference)
CATOFF_MIN_COMPS = 3


def catalog_offsets(
    comp_lc: Mapping[str, np.ndarray],
    cat_mag: Mapping[str, float],
) -> dict[str, float]:
    """COMP-CATOFF-2026-10: offset of each candidate's (catalogue - instrumental) magnitude from the
    median of all candidates, per frame, then the median over frames. {} with fewer than
    CATOFF_MIN_COMPS candidates that have a band magnitude and a light curve."""
    ids = [c for c in comp_lc if math.isfinite(float(cat_mag.get(c, float("nan"))))]
    if len(ids) < CATOFF_MIN_COMPS:
        return {}
    d = np.vstack([float(cat_mag[c]) - np.asarray(comp_lc[c], dtype=float) for c in ids])
    with np.errstate(all="ignore"):
        ref = np.nanmedian(d, axis=0)
        n_ok = np.sum(np.isfinite(d), axis=0)
    ref = np.where(n_ok >= CATOFF_MIN_COMPS, ref, np.nan)
    out: dict[str, float] = {}
    for i, c in enumerate(ids):
        r = d[i] - ref
        r = r[np.isfinite(r)]
        if r.size:
            out[c] = float(np.median(r))
    return out


def _suspect_badness(info: Mapping[str, Any]) -> float:
    """Order for keeping suspects when too few are clean: smaller is better."""
    s = float(info.get("slope_mmag_hr", float("nan")) or float("nan"))
    p = float(info.get("rms_p2p", float("nan")) or float("nan"))
    return (s if math.isfinite(s) else 0.0) + 1000.0 * (p if math.isfinite(p) else 1.0)


def smart_select(
    primary: Sequence[str],
    reserve: Sequence[str],
    comp_lc: Mapping[str, np.ndarray],
    comp_bjd: Mapping[str, np.ndarray],
    *,
    n_comp_min: int,
    n_comp_max: int,
    stability_kwargs: Mapping[str, Any],
    comp_rms_map: Mapping[str, float] | None = None,
    cat_mag: Mapping[str, float] | None = None,
    catalog_offset_max_mag: float = 0.0,
) -> tuple[list[str], dict[str, Any]]:
    """Return (final comp ids in rank order, log).

    log = {"verdict": {id: quality}, "replaced": [ids dropped], "added": [reserve ids used],
    "kept_suspect": [ids], "n_clean": int}."""
    from photometry_lightcurve import check_comparison_stability  # noqa: PLC0415

    ranked = [str(c) for c in list(primary) + list(reserve)]
    ranked = list(dict.fromkeys(c for c in ranked if c in comp_lc))
    log: dict[str, Any] = {"verdict": {}, "replaced": [], "added": [], "kept_suspect": [], "n_clean": 0}
    if not ranked:
        return list(primary), log
    # COMP-CATOFF-2026-10: catalogue-offset exclusion before the stability test
    cat_excluded: dict[str, float] = {}
    thr = float(catalog_offset_max_mag or 0.0)
    if thr > 0 and cat_mag:
        offs = catalog_offsets({c: comp_lc[c] for c in ranked}, cat_mag)
        cat_excluded = {c: v for c, v in offs.items() if abs(v) > thr}
        log["catalog_offset"] = {"max_mag": thr, "offsets": {c: round(v, 4) for c, v in offs.items()},
                                 "excluded": [c for c in ranked if c in cat_excluded]}
    all_ranked = list(ranked)
    ranked = [c for c in ranked if c not in cat_excluded]
    if not ranked:
        log["verdict"] = {c: "excluded" for c in all_ranked}
        log["replaced"] = [str(c) for c in primary]
        log["notes"] = {c: f"catalogue offset {v:+.2f} mag" for c, v in cat_excluded.items()}
        return [], log
    def _check(ids: list[str]) -> dict[str, Any]:
        return check_comparison_stability(
            {c: np.asarray(comp_lc[c], dtype=float) for c in ids},
            comp_rms_map=dict(comp_rms_map or {}),
            comp_bjd={c: np.asarray(comp_bjd[c], dtype=float) for c in ids if c in comp_bjd},
            **dict(stability_kwargs),
        )

    # Iterative: a trending star leaks into the frame-ensemble residual of the others, so
    # the worst suspect is removed and the rest is tested again (one at a time).
    active = list(ranked)
    removed: list[str] = []
    q_first: dict[str, Any] = {}
    while True:
        q = _check(active)
        for c in active:
            q_first.setdefault(c, q.get(c) or {})
        bad = [c for c in active if str((q.get(c) or {}).get("quality", "")) != "good"]
        if not bad or len(active) <= 2:
            break
        worst = max(bad, key=lambda c: (_suspect_badness(q.get(c) or {}), -ranked.index(c)))
        q_first[worst] = q.get(worst) or {}
        active.remove(worst)
        removed.append(worst)
    log["verdict"] = {c: ("good" if c in active and str((q.get(c) or {}).get("quality", "")) == "good"
                          else str((q_first.get(c) or {}).get("quality", "") or "suspect")) for c in ranked}
    clean = [c for c in ranked if log["verdict"][c] == "good"]
    final = clean[: int(n_comp_max)]
    # The ensemble that is used is the final set, and the residual of each comp depends on the
    # set (frame-ensemble residual): verify the final set on its own (draft 526: a comp good
    # among 16 candidates was suspect among the final 8). A comp suspect in the final set is
    # replaced by the next clean candidate, or dropped while more than n_comp_min remain.
    pool = [c for c in clean if c not in final]
    final_suspect: list[str] = []
    while len(final) >= 2:
        qf = _check(final)
        badf = [c for c in final if str((qf.get(c) or {}).get("quality", "")) != "good"]
        if not badf:
            break
        worst = max(badf, key=lambda c: (_suspect_badness(qf.get(c) or {}), -ranked.index(c)))
        if pool:
            final.remove(worst)
            final.append(pool.pop(0))
        elif len(final) > int(n_comp_min):
            final.remove(worst)
        else:
            final_suspect = badf
            break
        log["verdict"][worst] = str((qf.get(worst) or {}).get("quality", "") or "suspect") + " (final set)"
        q_first[worst] = qf.get(worst) or {}
        final = [c for c in ranked if c in set(final)]
    if final_suspect:
        log["kept_suspect"] = [c for c in ranked if c in set(final_suspect)]
    clean = [c for c in ranked if log["verdict"][c] == "good"]
    if len(final) < int(n_comp_min):
        sus = [c for c in ranked if c not in clean and log["verdict"][c] != "excluded"]
        sus.sort(key=lambda c: (_suspect_badness(q_first.get(c) or {}), ranked.index(c)))
        need = int(n_comp_min) - len(final)
        log["kept_suspect"] = list(dict.fromkeys(list(log.get("kept_suspect") or []) + sus[:need]))
        final = [c for c in ranked if c in set(final) | set(log["kept_suspect"])]
    log["n_clean"] = len(clean)
    for c in cat_excluded:
        log["verdict"][c] = "excluded"
    ranked = all_ranked
    prim = [str(c) for c in primary]
    log["replaced"] = [c for c in prim if c not in final]
    log["added"] = [c for c in final if c not in prim]
    log["notes"] = {c: str((q_first.get(c) or {}).get("note", "")) for c in ranked if (q_first.get(c) or {}).get("note")}
    for c, v in cat_excluded.items():
        log["notes"][c] = f"catalogue offset {v:+.2f} mag"
    return final, log
