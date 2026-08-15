"""Rebuild draft-514 SNR aperture table after IMPL-02 CoG/bkg fixes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from config import AppConfig  # noqa: E402
from photometry_core import (  # noqa: E402
    discover_aligned_science_fits,
    precompute_and_save_snr_aperture_table_for_draft,
)


def main() -> None:
    draft = ROOT / "Archive" / "Drafts" / "draft_000514"
    setup = "NoFilter_60_2"
    lights = draft / "detrended_aligned" / "lights" / setup
    ms = draft / "platesolve" / setup / "MASTERSTAR.fits"
    cfg = AppConfig()
    fits = discover_aligned_science_fits(lights)
    print(f"n_science_fits={len(fits)}")
    out = precompute_and_save_snr_aperture_table_for_draft(
        draft,
        masterstar_fits_path=ms,
        aligned_fits_paths=fits[:24],
        database_path=cfg.database_path,
        draft_id=514,
        cfg=cfg,
    )
    dest = ROOT / "dev" / "results" / "IMPL_02_aperture_cog.json"
    if out is None:
        rejected = draft / "aperture_snr_table_REJECTED.json"
        print("REFUSED - see", rejected)
        if rejected.is_file():
            dest.write_text(rejected.read_text(encoding="utf-8"), encoding="utf-8")
        sys.exit(2)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    tbl = out.get("table") or {}
    print("ee_path", out.get("ee_path"))
    print("r90", out.get("ee_r90_px"), "flat_outer", out.get("ee_flatness_outer_over_norm"))
    print("bkg_var", out.get("bkg_var_adu2_per_px"), "sky", out.get("sky_adu_per_px"))
    print("fwhm", out.get("fwhm_px"), "gain", out.get("gain"))
    print("gates_ok", (out.get("impl02_gates") or {}).get("ok"))
    for mag in (8.0, 10.0, 12.0, 14.0, 16.0):
        r = tbl.get(mag, tbl.get(str(mag)))
        ee = (out.get("ee_at_opt_by_mag") or {}).get(mag)
        bh = (out.get("bound_hit_by_mag") or {}).get(mag)
        print(f"  mag {mag}: r={r} ee={ee} bound={bh}")
    print("wrote", dest)


if __name__ == "__main__":
    main()
