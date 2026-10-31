from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from variability_store import tess_dir_for  # defined in the lower module; re-exported here

LOGGER = logging.getLogger(__name__)


def run_tess_for_star(photometry_dir: Path, catalog_id: str, cfg: Any, progress: Any = None) -> Any:
    """TESS analysis of one star of a draft (UI button and night run share this).

    Independent of ``tess_enabled``: the switch only decides the automatic run.
    The VYVAR night light curve (varcand_<id>.csv) is folded on the TESS period.
    """
    from tess_analysis import run_tess  # noqa: PLC0415

    phot = Path(photometry_dir)
    cid = str(catalog_id).strip()
    ra = dec = gmag = None
    for name in ("variability_field.csv", "variability_candidates.csv", "active_targets.csv"):
        p = phot / name
        if not p.is_file():
            continue
        df = pd.read_csv(p, dtype={"catalog_id": str}, low_memory=False)
        hit = df[df["catalog_id"].astype(str) == cid]
        if not hit.empty and "ra_deg" in hit and "dec_deg" in hit:
            ra, dec = float(hit.iloc[0]["ra_deg"]), float(hit.iloc[0]["dec_deg"])
            if "mag" in hit:
                gmag = pd.to_numeric(hit.iloc[0]["mag"], errors="coerce")
                gmag = float(gmag) if pd.notna(gmag) else None
            break
    if ra is None:
        raise ValueError(f"no coordinates for {cid}")
    vy = None
    lcp = phot / "lightcurves" / f"varcand_{cid}.csv"
    if lcp.is_file():
        d = pd.read_csv(lcp)
        vy = (d["time"].to_numpy(float), d["dmag"].to_numpy(float))
    else:
        from report_methods import lc_csv_path  # noqa: PLC0415
        lc = lc_csv_path(phot / "lightcurves", cid)
        if lc.is_file():
            d = pd.read_csv(lc)
            from photometry_lightcurve import published_mag_column  # noqa: PLC0415

            # G09-41: the published magnitude, or delta_mag for a light curve without a band zero point
            # (UNCAL-LC-2026-10: mag_calib all NaN) - the overlay only needs the shape
            col = published_mag_column(d)
            if col is None or not np.isfinite(pd.to_numeric(d[col], errors="coerce")).any():
                col = "delta_mag"
            vy = (pd.to_numeric(d["bjd"], errors="coerce").to_numpy(float), pd.to_numeric(d[col], errors="coerce").to_numpy(float))
    res = run_tess(cid, ra, dec, tess_dir_for(phot, cid), cfg=cfg, vyvar_lc=vy, progress=progress, target_mag=gmag)
    LOGGER.info("[TESS] %s: verdict %s %s", cid, res.verdict, "; ".join(res.reasons))
    return res


def auto_tess_for_candidates(photometry_dir: Path, cfg: Any) -> int:
    """Night run: TESS for every new candidate, only when ``tess_enabled`` is on."""
    if not bool(getattr(cfg, "tess_enabled", False)):
        LOGGER.info("[TESS] automatic analysis off (tess_enabled = false)")
        return 0
    from variability_field import load_variability_candidates  # noqa: PLC0415

    df = load_variability_candidates(Path(photometry_dir))
    n = 0
    for cid in df.get("catalog_id", pd.Series(dtype=str)).astype(str):
        # a star analysed before keeps its result (re-run from the UI to refresh it)
        try:
            prev = json.loads((tess_dir_for(photometry_dir, cid) / "tess_result.json").read_text(encoding="utf-8"))
            if str(prev.get("verdict")) not in ("", "NO_DATA"):
                continue
        except (OSError, ValueError):
            pass
        try:
            run_tess_for_star(photometry_dir, cid, cfg)
            n += 1
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[TESS] %s failed: %s", cid, exc)
    return n
