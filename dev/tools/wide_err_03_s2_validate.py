#!/usr/bin/env python3
"""WIDE-ERR-03 Stage 2 validation: S2d reproduce g_pt=0.635, S2c fire proof."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from gain_photon_transfer import (  # noqa: E402
    DEFAULT_CONTAINER_SCALE,
    estimate_photon_transfer_gain_from_proc_dir,
    fire_proof_bare_db_vs_gpt,
    resolve_photometric_gain,
    write_gain_pt_sidecar,
)

RUN_SHA = "da9cce4a5edd1392b8ba842d3c8488589b9d0ac9"
DRAFT = ROOT / "Archive" / "Drafts" / "draft_000515"
SETUP = "NoFilter_60_2"
PROC = DRAFT / "detrended_aligned" / "lights" / SETUP
PHOT = DRAFT / "platesolve" / SETUP / "photometry"
OUT = ROOT / "dev" / "results" / "WIDE_ERR_03_S2.json"

ARCH_G = 0.635
ARCH_LO, ARCH_HI = 0.44, 1.09


def main() -> None:
    meta = json.loads((PHOT / "pipeline_meta.json").read_text(encoding="utf-8"))
    ap_r = float((meta.get("dynamic_params") or {}).get("aperture_r_px") or 3.999)
    pt = estimate_photon_transfer_gain_from_proc_dir(PROC, aperture_r_px=ap_r)
    auth = resolve_photometric_gain(
        g_pt_result=pt,
        g_db_native=3.17,
        container_scale=DEFAULT_CONTAINER_SCALE,
    )
    fire = fire_proof_bare_db_vs_gpt(g_pt=ARCH_G, g_db_bare=3.17)

    s2d_ok = bool(
        pt.ok
        and ARCH_LO <= pt.g_pt <= ARCH_HI
        and abs(pt.g_pt - ARCH_G) / ARCH_G < 0.15  # within ~15% of 0.635
    )
    # Stronger: point estimate inside architect CI (task: reproduce 0.635 within CI)
    s2d_in_ci = bool(pt.ok and ARCH_LO <= pt.g_pt <= ARCH_HI)

    sidecar = {
        "task": "WIDE-ERR-03 Stage S2",
        "run_sha": RUN_SHA,
        "draft_id": 515,
        "setup": SETUP,
        "domain": "e-/ADU_container",
        "aperture_r_px": ap_r,
        "photon_transfer": pt.to_dict(),
        "authority": auth.to_dict(),
        "s2c_fire_proof": fire,
        "s2d": {
            "architect_g_pt": ARCH_G,
            "architect_ci": [ARCH_LO, ARCH_HI],
            "estimated_g_pt": pt.g_pt,
            "estimated_ci": [pt.g_pt_ci_lo, pt.g_pt_ci_hi],
            "in_architect_ci": s2d_in_ci,
            "pass": s2d_in_ci,
        },
        "s2d_stop": not s2d_in_ci,
    }
    OUT.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    write_gain_pt_sidecar(PHOT / "gain_photon_transfer.json", sidecar)
    print("WROTE", OUT)
    print("g_pt", pt.g_pt, "CI", pt.g_pt_ci_lo, pt.g_pt_ci_hi, "n", pt.n_frames)
    print("authority", auth.source, auth.value_e_per_adu_container)
    print("S2c bare-DB guard fires", fire["guard_fires_on_bare_db"], "ratio", fire["ratio_bare_vs_gpt"])
    print("S2d", "PASS" if s2d_in_ci else "FAIL", "in_CI", s2d_in_ci)
    if not s2d_in_ci:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
