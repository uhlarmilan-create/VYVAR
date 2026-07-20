"""Re-run Phase 1 comp selection (gs11 on) using existing active_targets.csv."""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from config import AppConfig, config_json_path  # noqa: E402
from photometry_core import (  # noqa: E402
    ProcFrameStore,
    _batch_enrich_targets_bp_rp_from_gaia_db,
    _enrich_target_bp_rp_from_gaia_db,
    _phase0_effective_frame_hw_px,
    _target_row_is_catalog_only,
    build_global_comp_pool,
    select_comparison_stars_per_target,
)
from ui_aperture_photometry import _load_fwhm  # noqa: E402
import pandas as pd  # noqa: E402
from photometry_core import _GAIA_ID_DTYPE, _normalize_id_series  # noqa: E402

DRAFT = _ROOT / "Archive/Drafts/draft_000342"
SETUP = "NoFilter_60_2"
PHOT = DRAFT / "platesolve" / SETUP / "photometry"
PS = DRAFT / "platesolve" / SETUP
PF = DRAFT / "detrended_aligned" / "lights" / SETUP


def main() -> int:
    buf: list[str] = []

    class _H(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            buf.append(record.getMessage())

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger().addHandler(_H())
    path = config_json_path(_ROOT)
    data = json.loads(path.read_text(encoding="utf-8"))
    import os

    restore_gs11 = os.environ.get("GS11_RESTORE", "0") != "1"
    data["gs11_dilution_enabled"] = not restore_gs11
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cfg = AppConfig()
    plate = float(cfg.phase01_plate_scale_arcsec_per_px or 1.3) or 1.3
    fwhm = float(_load_fwhm(PS / "MASTERSTAR.fits"))
    flux_col = "dao_flux"

    active = pd.read_csv(PHOT / "active_targets.csv", low_memory=False, dtype=_GAIA_ID_DTYPE)
    ms_df = pd.read_csv(PS / "masterstars_full_match.csv", low_memory=False, dtype=_GAIA_ID_DTYPE)
    for c in ("catalog_id", "name"):
        if c in ms_df.columns:
            ms_df[c] = _normalize_id_series(ms_df[c])

    store = ProcFrameStore.build(PF, glob_pattern="proc_*.csv")
    csv_paths = [Path(k) for k in store.keys()]
    _vt_cid = frozenset(str(x).strip() for x in active["catalog_id"].tolist() if str(x).strip())
    fw, fh = _phase0_effective_frame_hw_px(active, ms_df, frame_w_px=2082, frame_h_px=1397, edge_margin_px=100)
    gaia_db = str(cfg.gaia_db_path or "").strip()
    gaia_batch = _batch_enrich_targets_bp_rp_from_gaia_db(
        [str(r.get("catalog_id", "")) for _, r in active.iterrows()], gaia_db
    )

    gs11_events = 0
    gs11_comps = 0
    t0 = time.time()
    rows: list[pd.DataFrame] = []
    for _, target_row in active.iterrows():
        if _target_row_is_catalog_only(target_row):
            continue
        tr = _enrich_target_bp_rp_from_gaia_db(
            target_row, gaia_db_path=gaia_db, gaia_prefetch=gaia_batch
        )
        comps = select_comparison_stars_per_target(
            tr,
            ms_df,
            csv_paths,
            csv_cache=store,
            fwhm_px=fwhm,
            flux_col=flux_col,
            chip_fw=fw,
            chip_fh=fh,
            chip_interior_margin_px=100,
            gaia_db_path=gaia_db,
            variable_target_catalog_ids=_vt_cid,
            cfg=cfg,
            plate_scale_arcsec=plate,
        )
        if comps is not None and not comps.empty:
            rows.append(comps)

    comp_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out = PHOT / "comparison_stars_per_target.csv"
    comp_df.to_csv(out, index=False)
    print(f"phase1-only elapsed={time.time()-t0:.1f}s rows={len(comp_df)}")

    gs11_rej = [ln for ln in buf if "GS11 dilution filter vylucil" in ln]
    n_comps = sum(
        int(m.group(1))
        for ln in gs11_rej
        for m in [re.search(r"vylucil (\d+)", ln)]
        if m
    )
    print(f"GS11 reject events={len(gs11_rej)} total_comps_rejected={n_comps}")

    blended = {
        "1499974726349018112",
        "1498688469542868992",
        "1498072743031629824",
        "1497368849430107904",
        "1499187269863874304",
    }
    for cid in blended:
        n = len(comp_df[comp_df["catalog_id"].astype(str).str.strip() == cid])
        print(f"  blended {cid} in comp csv: {n} rows")

    if not restore_gs11:
        data["gs11_dilution_enabled"] = False
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("gs11_dilution_enabled restored to False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
