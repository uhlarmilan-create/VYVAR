"""Phase 2A replay draft_000321 — TODO-44 role-aware aperture verify."""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from io import StringIO
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from photometry_core import run_phase2a  # noqa: E402
from ui_aperture_photometry import _load_fwhm  # noqa: E402

DRAFT = _ROOT / "Archive" / "Drafts" / "draft_000321"
SETUP = "NoFilter_60_2"


def main() -> int:
    ps = DRAFT / "platesolve" / SETUP
    aligned = DRAFT / "detrended_aligned" / "lights" / SETUP
    phot = ps / "photometry"
    log_path = _ROOT / "todo44_verify.log"

    buf = StringIO()
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.StreamHandler(buf)):
        h.setLevel(logging.DEBUG)
        h.setFormatter(fmt)
        root.addHandler(h)
    logging.getLogger("photometry_core").setLevel(logging.DEBUG)

    cfg = AppConfig()
    print(
        f"aperture_variable_factor={cfg.aperture_variable_factor} "
        f"aperture_comp_factor={cfg.aperture_comp_factor}"
    )
    fw = float(_load_fwhm(ps / "MASTERSTAR.fits"))
    t0 = time.perf_counter()
    rc = 0
    try:
        out = run_phase2a(
            masterstar_fits_path=ps / "MASTERSTAR.fits",
            active_targets_csv=phot / "active_targets.csv",
            comparison_stars_csv=phot / "comparison_stars_per_target.csv",
            per_frame_csv_dir=aligned,
            detrended_aligned_dir=aligned,
            output_dir=phot,
            fwhm_px=fw,
            cfg=cfg,
            draft_id=321,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        rc = 1
        out = {}

    elapsed = time.perf_counter() - t0
    log = buf.getvalue()
    log_path.write_text(log, encoding="utf-8")

    print(f"exit={rc} elapsed_s={elapsed:.1f}")
    if rc != 0:
        return rc

    import pandas as pd

    print(f"n_lightcurves={out.get('n_lightcurves')}")
    s = pd.read_csv(phot / "photometry_summary.csv", dtype={"catalog_id": str})
    for name in ("BO CVn", "FW CVn"):
        m = s["vsx_name"].astype(str).str.strip() == name
        if m.any():
            r = s.loc[m].iloc[0]
            print(f"{name}: lc_rms={r.get('lc_rms')} aperture_px={r.get('aperture_px')}")

    t44 = [ln for ln in log.splitlines() if "[TODO-44]" in ln]
    print(f"todo44_log_lines={len(t44)}")
    for ln in t44:
        print(ln)

    errs = [ln for ln in log.splitlines() if "Traceback" in ln or "[ERROR]" in ln]
    print(f"error_lines={len(errs)}")

    # Spot-check BO target vs its comp stars (aperture from summary / comp table rms proxy)
    bo = s[s["vsx_name"].astype(str).str.strip() == "BO CVn"]
    if bo.empty:
        return rc
    bo_cid = str(bo["catalog_id"].iloc[0]).strip()
    bo_ap = float(bo["aperture_px"].iloc[0]) if "aperture_px" in bo.columns else float("nan")
    bo_lc_ap = float("nan")
    lc_bo = phot / "lightcurves" / f"lightcurve_{bo_cid}.csv"
    if lc_bo.is_file():
        dfb = pd.read_csv(lc_bo, usecols=["aperture_r_px"])
        bo_lc_ap = float(dfb["aperture_r_px"].iloc[0])
        print(f"BO CVn summary aperture_px={bo_ap:.3f} LC aperture_r_px={bo_lc_ap:.3f}")

    comp_df = pd.read_csv(
        phot / "comparison_stars_per_target.csv",
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    comp_df["target_catalog_id"] = comp_df["target_catalog_id"].astype(str).str.strip()
    comp_df["catalog_id"] = comp_df["catalog_id"].astype(str).str.strip()
    comps_bo = comp_df.loc[comp_df["target_catalog_id"] == bo_cid, "catalog_id"].drop_duplicates().tolist()
    print(f"BO CVn comp count={len(comps_bo)}")
    # Comp apertures: read from first proc CSV row per comp (if column exists)
    # Expected radii from saved SNR table × role factors
    ap_json = DRAFT / "aperture_snr_table.json"
    if ap_json.is_file() and math.isfinite(bo_lc_ap):
        from photometry_core import (  # noqa: E402
            _aperture_radius_from_snr_table,
            _phase2a_star_mag_lookup,
        )

        snr_tbl = json.loads(ap_json.read_text(encoding="utf-8"))
        mag_map = _phase2a_star_mag_lookup(
            pd.read_csv(phot / "active_targets.csv", dtype={"catalog_id": str}, low_memory=False),
            comp_df,
            ps / "MASTERSTAR.fits",
        )
        _apt_fw = float(cfg.aperture_fwhm_factor)
        comp_aps = []
        for cc in comps_bo[:12]:
            m = float(mag_map.get(cc, float("nan")))
            if math.isfinite(m):
                comp_aps.append(
                    _aperture_radius_from_snr_table(
                        m, snr_tbl, aperture_fwhm_factor=_apt_fw, fwhm_px=fw
                    )
                    * float(cfg.aperture_comp_factor)
                )
        if comp_aps:
            print(
                f"BO comp expected aperture (SNR×{cfg.aperture_comp_factor}): "
                f"median={pd.Series(comp_aps).median():.3f} min={min(comp_aps):.3f} max={max(comp_aps):.3f}"
            )
            print(
                f"BO target LC aperture_r_px={bo_lc_ap:.3f} vs comp median={pd.Series(comp_aps).median():.3f} "
                f"ratio={pd.Series(comp_aps).median()/bo_lc_ap:.3f} (expect ~1.1 if same mag bin differs)"
            )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
