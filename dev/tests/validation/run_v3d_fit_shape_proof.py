"""CLI: fit_shape enlargement proof + fallback truth-sky diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.validation.v3d_bias_decomposition_v2 import (
    run_fallback_truth_sky_noiseless,
    run_v3d_bias_decomposition_v2,
)
from tests.validation.v3d_fine_scale import (
    V3dFineConfig,
    run_v3d_fine_scale,
    write_v3d_report,
)
from tests.validation.v3d_bias_decomposition_v2 import _shared_assets

DATA_DIR = Path(__file__).resolve().parent / "data" / "tier_v3d"

# Recorded enlargement attempts (harness re-runs during this task).
_ENLARGEMENT_ATTEMPTS = {
    "2xFWHM_baseline": {
        "rule": "odd(2xFWHM+1) ~15px",
        "mid_mag_post_ac_pct": 4.57,
        "drift_mag16_minus_12_pp": 3.48,
    },
    "3xFWHM_attempt": {
        "rule": "odd(3xFWHM+1) ~19px",
        "mid_mag_post_ac_pct": 9.77,
        "drift_mag16_minus_12_pp": 8.28,
    },
    "4xFWHM_attempt": {
        "rule": "odd(4xFWHM+1) ~27px",
        "mid_mag_post_ac_pct": 16.11,
        "drift_mag16_minus_12_pp": 14.36,
    },
}


def write_fit_shape_proof_report(
    out_dir: Path,
    *,
    v3d_result: dict,
    fallback: dict,
) -> Path:
    from psf_photometry import _fit_shape_for_cutout

    cfg = V3dFineConfig()
    shape = _fit_shape_for_cutout(cfg.cutout_size(), fwhm_px=cfg.fwhm_px)
    post = {s["mag"]: s for s in v3d_result.get("mag_stats", [])}
    lines = [
        "# V3d fit_shape enlargement proof + fallback",
        "",
        "## STEP 0 -- fit_shape is global / uniform",
        "",
        (
            "`_fit_shape_for_cutout` uses **global** `fwhm_px` from ePSF meta JSON "
            "(same for every star in the frame). It is **not** per-star measured FWHM. "
            "Fix path: enlarge window (uniform truncation fraction); not per-star unify."
        ),
        "",
        f"Production fit_shape (reverted): **{shape[0]}x{shape[1]} px**.",
        "",
        "## STEP 1-2 -- enlargement attempts (FAILED proof)",
        "",
        "Success criterion: post-AC **mag-drift** (mag16-mag12) shrinks toward 0; mid-mag <1-2%.",
        "",
        "| rule | mid-mag post-AC % | bright->mid drift (pp) |",
        "|:-----|------------------:|-----------------------:|",
    ]
    for key, row in _ENLARGEMENT_ATTEMPTS.items():
        lines.append(
            f"| {row['rule']} | {row['mid_mag_post_ac_pct']:+.2f} | {row['drift_mag16_minus_12_pp']:+.2f} |"
        )
    lines.extend(
        [
            "",
            (
                "**Enlargement worsens drift** in the full noisy V3d sweep (unlike noiseless T3 harness). "
                "Likely cause: fit window consumes the 31px cutout, admitting edge/sky pixels in iterative fit. "
                "**Reverted** to 2xFWHM+1 in production."
            ),
            "",
            "## Fallback -- noiseless truth sky vs annulus (production psf_photometry_stars)",
            "",
            f"Drift vanishes with truth sky: **{fallback.get('drift_vanishes_with_truth_sky')}**",
            f"Annulus drift (mag16-mag12 post-AC): **{fallback.get('annulus_post_ac_drift_mag16_minus_12_pp', float('nan')):+.2f} pp**",
            f"Truth-sky drift: **{fallback.get('truth_post_ac_drift_mag16_minus_12_pp', float('nan')):+.2f} pp**",
            "",
            "| mag | annulus post-AC % | truth-sky post-AC % |",
            "|----:|------------------:|--------------------:|",
        ]
    )
    for row in fallback.get("table", []):
        lines.append(
            f"| {row['mag']} | {row['annulus_post_ac_pct']:+.3f} | {row['truth_post_ac_pct']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Current V3d (reverted 2xFWHM, annulus sky, noisy)",
            "",
            "| mag | post-AC PSF bias % | APER % |",
            "|----:|-------------------:|-------:|",
        ]
    )
    for mag in sorted(post):
        if mag > 17:
            continue
        s = post[mag]
        lines.append(
            f"| {mag} | {s['psf_bias_pct']:+.2f} | {s.get('aper_bias_pct', float('nan')):+.2f} |"
        )
    drift = float(post[16]["psf_bias_pct"] - post[12]["psf_bias_pct"]) if 12 in post and 16 in post else float("nan")
    lines.extend(
        [
            "",
            f"Bright->mid drift: **{drift:+.2f} pp**.",
            "",
            "## VERDICT",
            "",
        ]
    )
    if fallback.get("drift_vanishes_with_truth_sky"):
        lines.append(
            "fit_shape enlargement is **not** the fix (worsens noisy V3d drift). Fallback shows "
            "**mag-drift vanishes** with truth sky (drift 0 pp) but ~+7% uniform offset remains -> "
            "**sky-annulus-wing contamination** is the drift source. Next: push annulus farther out "
            "or harder sigma-clip in sky estimate; do not enlarge fit_shape blindly."
        )
    else:
        lines.append(
            "fit_shape enlargement **failed**; truth-sky fallback **did not** flatten mag-drift. "
            "PSF not publication-grade; further diagnosis required."
        )
    lines.append("")
    lines.append("PSF remains gated OFF in production LC.")
    mp = out_dir / "v3d_fit_shape_proof.md"
    mp.write_text("\n".join(lines), encoding="ascii")
    return mp


def main() -> None:
    ap = argparse.ArgumentParser(description="V3d fit_shape proof + fallback")
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--n-real", type=int, default=30)
    args = ap.parse_args()
    cfg = V3dFineConfig(n_real=max(5, int(args.n_real)))
    out = Path(args.out)
    v3d = run_v3d_fine_scale(cfg, work_dir=out / "_work_proof")
    write_v3d_report(out, v3d)
    epsf_path, psf_ac, meta = _shared_assets(cfg, out / "_work_proof_fb")
    fallback = run_fallback_truth_sky_noiseless(cfg, epsf_path, psf_ac, meta)
    mp = write_fit_shape_proof_report(out, v3d_result=v3d, fallback=fallback)
    jp = out / "v3d_fit_shape_proof.json"
    with open(jp, "w", encoding="ascii") as f:
        json.dump({"v3d": v3d, "fallback": fallback, "attempts": _ENLARGEMENT_ATTEMPTS}, f, indent=2)
    print(f"proof: {mp}")
    print(f"drift_vanishes_truth_sky: {fallback.get('drift_vanishes_with_truth_sky')}")


if __name__ == "__main__":
    main()
