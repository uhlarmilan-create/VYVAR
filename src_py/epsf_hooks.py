"""Moved from pipeline.py and photometry_core.py (CONSOLIDATE-01E1). Facades re-export these names."""
from __future__ import annotations
import numpy as np
import pandas as pd

from pathlib import Path
from gaia_catalog_id import normalize_gaia_source_id, read_vyvar_csv

import logging
LOGGER = logging.getLogger("pipeline")

def _add_catalog_ids_from_csv(ids: set[str], comp_p: Path) -> None:
    """Merge ``catalog_id`` values from a comparison-star CSV into *ids*."""
    if not comp_p.is_file():
        return
    try:
        cdf = read_vyvar_csv(comp_p, low_memory=False)
        if "catalog_id" not in cdf.columns or cdf.empty:
            return
        for raw in cdf["catalog_id"].fillna("").astype(str).str.strip():
            if not raw or raw.lower() in ("nan", "none"):
                continue
            try:
                ids.add(str(normalize_gaia_source_id(raw)).strip())
            except Exception:  # noqa: BLE001
                ids.add(raw)
    except Exception as _comp_exc:  # noqa: BLE001
        LOGGER.warning("[ePSF] comparison star load failed (%s): %s", comp_p.name, _comp_exc)

def _epsf_lc_catalog_ids(platesolve_dir: Path) -> set[str] | None:
    """Full LC star set: active targets + comp pool + all per-target comps (gated PSF coverage)."""
    phot = Path(platesolve_dir) / "photometry"
    ids: set[str] = set()
    at_p = phot / "active_targets.csv"
    if at_p.is_file():
        try:
            at = read_vyvar_csv(at_p, low_memory=False)
            if "catalog_id" in at.columns:
                for _, row in at.iterrows():
                    z = str(row.get("zone_flag", row.get("zone", "")) or "").strip().lower()
                    if z == "catalog_only":
                        continue
                    raw = row.get("catalog_id")
                    if raw is None:
                        continue
                    s = str(raw).strip()
                    if not s or s.lower() in ("nan", "none"):
                        continue
                    try:
                        ids.add(str(normalize_gaia_source_id(s)).strip())
                    except Exception:  # noqa: BLE001
                        ids.add(s)
        except Exception:  # noqa: BLE001
            pass
    for comp_p in (phot / "comparison_stars.csv", Path(platesolve_dir) / "comparison_stars.csv"):
        _add_catalog_ids_from_csv(ids, comp_p)
        if comp_p.is_file():
            break
    _add_catalog_ids_from_csv(ids, phot / "comparison_stars_per_target.csv")
    if ids:
        LOGGER.debug("[ePSF] LC catalog_ids loaded: %d (targets+full comp pool)", len(ids))
    return ids if ids else None

def load_epsf_metrics_for_draft(
    per_frame_csv_dir: Path,
    active_targets_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate PSF fit metrics from ``proc_*.csv`` (vectorized groupby).

    Returns per-``catalog_id`` stats: frame counts, PSF OK %, chi^2, mean fluxes, PSF/DAO ratio.
    """
    proc_dir = Path(per_frame_csv_dir)
    proc_files = sorted(proc_dir.glob("proc_*.csv"))
    if not proc_files:
        return pd.DataFrame()

    usecols = ["catalog_id", "psf_flux", "psf_fit_ok", "psf_chi2", "dao_flux"]
    _cid_dtype = {"catalog_id": str}
    chunks: list[pd.DataFrame] = []
    for csv_path in proc_files:
        try:
            df = pd.read_csv(
                csv_path, usecols=usecols, low_memory=False, dtype=_cid_dtype
            )
            chunks.append(df)
        except Exception as exc:  # noqa: BLE001
            logging.error("[EXC-0163] One frame's psf_fit CSV unreadable - ePSF metrics aggregate omits that frame's stars: %s", exc)
            try:
                df = pd.read_csv(csv_path, low_memory=False)
                if "psf_fit_ok" not in df.columns:
                    continue
                keep = [c for c in usecols if c in df.columns]
                chunks.append(df[keep])
            except Exception:  # noqa: BLE001
                continue

    if not chunks:
        return pd.DataFrame()

    combined = pd.concat(chunks, ignore_index=True)
    if "catalog_id" not in combined.columns or "psf_fit_ok" not in combined.columns:
        return pd.DataFrame()

    try:
        from gaia_catalog_id import normalize_gaia_source_id_series  # noqa: PLC0415

        combined["catalog_id"] = normalize_gaia_source_id_series(combined["catalog_id"])
    except Exception:  # noqa: BLE001
        combined["catalog_id"] = combined["catalog_id"].astype(str).str.strip()
    combined = combined[
        combined["catalog_id"].astype(bool)
        & ~combined["catalog_id"].str.lower().isin(("nan", "none", ""))
    ]
    if combined.empty:
        return pd.DataFrame()

    combined["psf_fit_ok"] = combined["psf_fit_ok"].fillna(False).astype(bool)
    combined["psf_chi2"] = pd.to_numeric(combined["psf_chi2"], errors="coerce")
    combined["psf_flux"] = pd.to_numeric(combined["psf_flux"], errors="coerce")
    combined["dao_flux"] = pd.to_numeric(combined["dao_flux"], errors="coerce")

    grp = combined.groupby("catalog_id", sort=False)
    result = pd.DataFrame(
        {
            "n_frames": grp["psf_fit_ok"].count(),
            "n_psf_ok": grp["psf_fit_ok"].sum(),
            "mean_chi2": grp["psf_chi2"].mean(),
            "median_chi2": grp["psf_chi2"].median(),
            "min_chi2": grp["psf_chi2"].min(),
            "mean_psf_flux": grp["psf_flux"].mean(),
            "mean_dao_flux": grp["dao_flux"].mean(),
        }
    ).reset_index()

    result["pct_psf_ok"] = (100.0 * result["n_psf_ok"] / result["n_frames"]).round(1)
    for col in ("mean_chi2", "median_chi2", "min_chi2"):
        result[col] = pd.to_numeric(result[col], errors="coerce").round(2)
    result["psf_dao_ratio"] = (
        result["mean_psf_flux"] / result["mean_dao_flux"].replace(0, np.nan)
    ).round(4)

    if (
        not active_targets_df.empty
        and "catalog_id" in active_targets_df.columns
        and "vsx_name" in active_targets_df.columns
    ):
        meta = active_targets_df[["catalog_id", "vsx_name"]].copy()
        try:
            meta["catalog_id"] = normalize_gaia_source_id_series(meta["catalog_id"])
        except Exception:  # noqa: BLE001
            meta["catalog_id"] = meta["catalog_id"].astype(str).str.strip()
        meta = meta.drop_duplicates(subset=["catalog_id"], keep="first")
        result = result.merge(meta, on="catalog_id", how="left")

    # Sort: known targets (vsx_name not null) first, then by pct_psf_ok desc
    if "vsx_name" in result.columns:
        result["_has_name"] = result["vsx_name"].notna() & (
            result["vsx_name"].astype(str).str.strip() != ""
        )
        result = (
            result.sort_values(["_has_name", "pct_psf_ok"], ascending=[False, False])
            .drop(columns="_has_name")
            .reset_index(drop=True)
        )
    else:
        result = result.sort_values("pct_psf_ok", ascending=False).reset_index(drop=True)
    return result
