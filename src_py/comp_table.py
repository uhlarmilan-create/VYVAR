"""The comparison-star table of a setup: one file, every comparison candidate of every target.

OUTPUT-REVIEW-2026-10 step 5 (Milan 2026-10-21). ``photometry/comparison_stars_per_target.csv``
holds one row per (target, comparison star) that comparison-star selection admitted or ranked as a reserve:

* ``p1_role``   ``admitted`` (the comparison-star selection) or ``reserve`` (COMP-SMART candidates);
* ``p1_rank``   order within the target and role (admitted: selection order, reserve: rank);
* ``used_role`` set by aperture photometry on the comparison stars of the final ensemble:
  ``primary`` (admitted and used), ``replacement`` (a reserve star used), ``kept_suspect``
  (flagged by COMP-SMART but kept); empty = not used;
* ``used_rank`` order of the used stars within the target (the ensemble order).

Earlier drafts have three files (``comparison_stars_per_target.csv`` = used sets,
``comparison_stars_per_target_phase1.csv`` = comparison-star selection sets when COMP-SMART changed one,
``comparison_reserve_per_target.csv`` = reserve); :func:`load_comp_table` builds the same view
from them. Readers take :func:`used`, :func:`admitted` or :func:`reserve` of that view.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from photometry_defs import BLEND_K_COLUMNS, COMP_QUALITY_JSON_META_KEYS

COMP_TABLE_CSV = "comparison_stars_per_target.csv"
LEGACY_COMP_SELECTION_CSV = "comparison_stars_per_target_phase1.csv"
LEGACY_RESERVE_CSV = "comparison_reserve_per_target.csv"

ROLE_ADMITTED = "admitted"
ROLE_RESERVE = "reserve"
USED_PRIMARY = "primary"
USED_REPLACEMENT = "replacement"
USED_KEPT_SUSPECT = "kept_suspect"

# The columns readers take (column audit 2026-10-21). The MASTERSTAR row of every comparison star
# (~50 columns, gates constant by construction) and the duplicates (phot_g_mean_mag / _mag = mag,
# name / _norm_cid_vt = catalog_id, comp_rms_loo_mag = comp_rms, blend_contamination = blend_k100,
# _delta_bprp_abs, sigma_eff_mag = 1/sqrt(comp_weight)) are not written.
COMP_TABLE_COLUMNS: tuple[str, ...] = (
    "target_catalog_id",
    "catalog_id",
    "p1_role",
    "p1_rank",
    "used_role",
    "used_rank",
    "x",
    "y",
    "ra_deg",
    "dec_deg",
    "mag",
    "bp_rp",
    "comp_rms",
    "comp_n_frames",
    "comp_score",
    *BLEND_K_COLUMNS,
    "_dist_deg",
    "delta_bprp_abs",
    "comp_tier",
    "color_tier_src",
    "comp_weight",
    # per target, repeated on its rows (read from the first row of the target)
    "target_bp_rp",
    "selected_tier",
    "tier4_warning",
    "n_tier1",
    "n_tier2",
    "n_tier3",
    "n_tier4",
    "selection_note",
)


def _norm(s: pd.Series) -> pd.Series:
    from gaia_catalog_id import normalize_gaia_source_id_series  # noqa: PLC0415

    return normalize_gaia_source_id_series(s.astype(str))


def _with_target_order(df: pd.DataFrame) -> pd.DataFrame:
    order = {t: i for i, t in enumerate(pd.unique(df["target_catalog_id"].astype(str)))}
    return df.assign(_tord=df["target_catalog_id"].astype(str).map(order))


def _sorted(df: pd.DataFrame, rank_col: str) -> pd.DataFrame:
    if df.empty:
        return df.reset_index(drop=True)
    d = _with_target_order(df)
    d = d.assign(_rk=pd.to_numeric(d[rank_col], errors="coerce"))
    return d.sort_values(["_tord", "_rk"], kind="mergesort").drop(columns=["_tord", "_rk"]).reset_index(drop=True)


def comp_selection_table(admitted_rows: pd.DataFrame, reserve_rows: pd.DataFrame | None) -> pd.DataFrame:
    """The comparison-star selection table: admitted rows (in their order) + reserve rows (by ``reserve_rank``)."""
    parts = []
    if admitted_rows is not None and not admitted_rows.empty:
        a = admitted_rows.copy()
        a["p1_role"] = ROLE_ADMITTED
        a["p1_rank"] = a.groupby(a["target_catalog_id"].astype(str), sort=False).cumcount()
        parts.append(a)
    if reserve_rows is not None and not reserve_rows.empty:
        r = reserve_rows.copy()
        r["p1_role"] = ROLE_RESERVE
        if "reserve_rank" in r.columns:
            r["p1_rank"] = pd.to_numeric(r["reserve_rank"], errors="coerce")
        else:
            r["p1_rank"] = r.groupby(r["target_catalog_id"].astype(str), sort=False).cumcount()
        parts.append(r)
    if not parts:
        return pd.DataFrame(columns=list(COMP_TABLE_COLUMNS))
    df = pd.concat(parts, ignore_index=True, sort=False)
    df["used_role"] = ""
    df["used_rank"] = pd.NA
    return df


def write_comp_table(df: pd.DataFrame, path: Path | str) -> Path:
    """Write the table (known columns only, IDs normalised, ``_dist_deg`` as a 9-decimal token)."""
    from sky_separation import persist_dist_deg_column  # noqa: PLC0415

    out = df.copy()
    for col in COMP_TABLE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[list(COMP_TABLE_COLUMNS)]
    for col in ("catalog_id", "target_catalog_id"):
        out[col] = _norm(out[col])
    persist_dist_deg_column(out)
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(p)
    return p


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, dtype={"catalog_id": str, "target_catalog_id": str,
                                                    "p1_role": str, "used_role": str})
    for col in ("catalog_id", "target_catalog_id"):
        if col in df.columns:
            df[col] = _norm(df[col])
    return df


def load_comp_table(path_or_dir: Path | str) -> pd.DataFrame:
    """The table of a setup (``photometry`` dir or the CSV path); older drafts: built from 3 files."""
    p = Path(path_or_dir)
    main = p if p.suffix.lower() == ".csv" else p / COMP_TABLE_CSV
    if not main.is_file():
        return pd.DataFrame(columns=list(COMP_TABLE_COLUMNS))
    df = _read(main)
    if not df.empty and "target_catalog_id" not in df.columns:
        if any(col in df.columns for col in ("comp_id", "role")):
            raise ValueError(
                "comparison_stars_csv must be comparison_stars_per_target.csv "
                f"(missing target_catalog_id; got comparison pool file): {main}"
            )
        raise ValueError(
            "comparison_stars_csv missing target_catalog_id column "
            f"(required for aperture-photometry per-target comp routing): {main}"
        )
    if df.empty and "target_catalog_id" not in df.columns:
        return pd.DataFrame(columns=list(COMP_TABLE_COLUMNS))
    if "p1_role" in df.columns:
        for col in ("used_role", "p1_role"):
            df[col] = df[col].fillna("").astype(str).str.strip()
        return df
    return _legacy_table(df, main.parent)


def _legacy_table(main: pd.DataFrame, phot: Path) -> pd.DataFrame:
    p1_path = phot / LEGACY_COMP_SELECTION_CSV
    res_path = phot / LEGACY_RESERVE_CSV
    admitted_rows = _read(p1_path) if p1_path.is_file() else main.drop(columns=["comp_role"], errors="ignore")
    reserve_rows = _read(res_path) if res_path.is_file() else None
    table = comp_selection_table(admitted_rows, reserve_rows)
    used = main.copy()
    role = used["comp_role"] if "comp_role" in used.columns else pd.Series("", index=used.index)
    used["used_role"] = role.fillna("").astype(str).str.strip().replace("", USED_PRIMARY)
    used["used_rank"] = used.groupby(used["target_catalog_id"].astype(str), sort=False).cumcount()
    key = ["target_catalog_id", "catalog_id"]
    table = table.drop(columns=["used_role", "used_rank"]).merge(
        used[key + ["used_role", "used_rank"]], on=key, how="left")
    table["used_role"] = table["used_role"].fillna("").astype(str)
    return table


def admitted(df: pd.DataFrame) -> pd.DataFrame:
    """The comparison-star selection, per target in selection order."""
    return _sorted(df[df["p1_role"].astype(str) == ROLE_ADMITTED], "p1_rank")


def reserve(df: pd.DataFrame) -> pd.DataFrame:
    """The reserve candidates, per target by rank."""
    return _sorted(df[df["p1_role"].astype(str) == ROLE_RESERVE], "p1_rank")


def used(df: pd.DataFrame) -> pd.DataFrame:
    """The comparison stars of the final ensembles, per target in ensemble order."""
    return _sorted(df[df["used_role"].fillna("").astype(str) != ""], "used_rank")


def with_used_sets(df: pd.DataFrame, used_by_target: dict[str, Iterable[tuple[str, str]]]) -> pd.DataFrame:
    """``df`` with ``used_role`` / ``used_rank`` from ``{target: [(catalog_id, role), ...]}`` (in order).
    Targets not in the mapping keep no used rows."""
    out = df.copy()
    out["used_role"] = ""
    out["used_rank"] = pd.NA
    pos = {(str(t), str(c)): i for i, (t, c) in enumerate(zip(out["target_catalog_id"].astype(str),
                                                             out["catalog_id"].astype(str)))}
    for t, items in used_by_target.items():
        for rank, (cid, role) in enumerate(items):
            i = pos.get((str(t), str(cid)))
            if i is None:
                continue
            out.iat[i, out.columns.get_loc("used_role")] = str(role or USED_PRIMARY)
            out.iat[i, out.columns.get_loc("used_rank")] = rank
    return out


# --- moved from aperture_photometry (LAYER-2026-10) ---


def parse_comp_quality_json_map(raw: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Return ``catalog_id`` -> ``{quality, note}`` from ``comp_quality_*.json`` (strip metadata keys).

    Accepts legacy flat strings (``"good"``) and structured objects
    (``{"quality": "suspect", "note": "..."}``).
    """
    out: dict[str, dict[str, str]] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        ks = str(k).strip()
        if ks in COMP_QUALITY_JSON_META_KEYS:
            continue
        if isinstance(v, str) and v.strip().lower() in ("good", "suspect", "excluded"):
            out[ks] = {"quality": v.strip().lower(), "note": ""}
        elif isinstance(v, dict):
            q = str(v.get("quality", "") or "").strip().lower()
            if q in ("good", "suspect", "excluded"):
                out[ks] = {"quality": q, "note": str(v.get("note", "") or "").strip()}
    return out


# --- moved from photometry_gate_helpers (LAYER-2026-10) ---


def comp_quality_quality_strings(
    qmap: dict[str, dict[str, str]] | dict[str, str] | None,
) -> dict[str, str]:
    """Flatten parsed comp-quality map to ``catalog_id`` -> quality string (for w_rel / export helpers)."""
    if not qmap:
        return {}
    out: dict[str, str] = {}
    for k, v in qmap.items():
        if isinstance(v, dict):
            out[str(k)] = str(v.get("quality", "") or "").strip().lower()
        else:
            out[str(k)] = str(v).strip().lower()
    return out
