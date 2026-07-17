"""CLI: residual-annulus sky fix proof (option C) + option sweep summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.validation.sky_option_sweep import run_sky_option_sweep
from tests.validation.v3d_bias_decomposition_v2 import (
    run_fallback_truth_sky_noiseless,
    run_t1_noiseless,
    _shared_assets,
)
from tests.validation.v3d_fine_scale import V3dFineConfig, run_v3d_fine_scale

OUT = Path(__file__).resolve().parent / "data" / "tier_v3d"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-real", type=int, default=30)
    args = parser.parse_args()

    cfg = V3dFineConfig(n_real=args.n_real)
    work = OUT / "_work_clean_sky"
    epsf, ac, meta = _shared_assets(cfg, work)

    t1 = run_t1_noiseless(cfg, epsf, ac)
    fallback = run_fallback_truth_sky_noiseless(cfg, epsf, ac, meta)
    sweep = run_sky_option_sweep(cfg, work_dir=work / "sweep")
    v3d = run_v3d_fine_scale(cfg, work_dir=work / "v3d")

    ann_drift = float(fallback.get("annulus_post_ac_drift_mag16_minus_12_pp", float("nan")))
    t1_drift = float("nan")
    if t1.get("table"):
        rows = [r for r in t1["table"] if r["mag"] in (12, 16)]
        if len(rows) == 2:
            t1_drift = float(rows[1]["post_ac_bias_pct"] - rows[0]["post_ac_bias_pct"])

    post_by_mag = {s["mag"]: s["psf_bias_pct"] for s in v3d.get("mag_stats", [])}
    noisy_drift = float(post_by_mag.get(16, float("nan")) - post_by_mag.get(12, float("nan")))

    lines = [
        "# V3d clean-sky fix proof (option C: residual annulus)",
        "",
        "## Option sweep (noiseless, harness)",
        "",
        "Synthetic Moffat inject: annulus median sky equals truth (300 ADU) at all mags;",
        "wings do not reach r_in=4.75 FWHM. Options A/B/C change sky negligibly.",
        "",
        "| variant | noiseless drift (pp) | mid-mag post-AC % |",
        "|:--------|-------------------:|------------------:|",
    ]
    for k, v in sweep["variants"].items():
        lines.append(
            f"| {k} | {v.get('post_ac_drift_mag16_minus_12_pp', float('nan')):+.2f} | "
            f"{v.get('mid_mag_post_ac_mean_pct', float('nan')):+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Production path (psf_photometry_stars + error map)",
            "",
            f"- Noiseless annulus drift (pre-fix baseline): **{ann_drift:+.2f} pp**",
            f"- Noiseless with residual_annulus sky (current): **{t1_drift:+.2f} pp**",
            f"- Noisy V3d post-AC drift mag16-mag12: **{noisy_drift:+.2f} pp**",
            "",
            "| mag | noisy post-AC PSF % |",
            "|----:|--------------------:|",
        ]
    )
    for mag in sorted(post_by_mag):
        if mag <= 17:
            lines.append(f"| {mag} | {post_by_mag[mag]:+.2f} |")

    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            "Truth-sky fallback used plain PSFPhotometry **without** the production error map;",
            "with IterativePSFPhotometry **and** error map, truth sky still shows ~+5.3 pp drift.",
            "Mid-mag drift in V3d is dominated by **flux-dependent fit weights + scalar AC**,",
            "not annulus sky bias on isolated synthetic frames (annulus median = 300 ADU).",
            "",
            "## VERDICT",
            "",
            "**NOT publication-grade** for accuracy (<1-2% mid-mag, sub-% drift).",
            "Residual-annulus sky (option C) shipped for crowded-field correctness;",
            "next task: PSF-fit noise from residual annulus and/or mag-aware AC (separate).",
            "",
        ]
    )

    payload = {
        "t1_noiseless": t1,
        "fallback_truth_sky": fallback,
        "sweep": sweep,
        "v3d": {"mag_stats": v3d.get("mag_stats"), "status": v3d.get("status")},
        "noisy_drift_pp": noisy_drift,
    }
    jp = OUT / "v3d_clean_sky_proof.json"
    mp = OUT / "v3d_clean_sky_proof.md"
    with open(jp, "w", encoding="ascii") as f:
        json.dump(payload, f, indent=2)
    mp.write_text("\n".join(lines), encoding="ascii")
    print(f"proof: {mp}")
    print(f"noisy_drift_pp: {noisy_drift:.2f}")


if __name__ == "__main__":
    main()
