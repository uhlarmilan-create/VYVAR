#!/usr/bin/env python3
"""WIDE-ERR-03 Stage 5: fit s/sigma_r, weighted-SEM note, acceptance, fire proofs."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from err_calibration import (  # noqa: E402
    ERR_CALIB_SIDECAR,
    apply_calibration_mmag,
    bins_to_dicts,
    calibrate_bins,
    write_sidecar,
)
from sigma_floor_core import (  # noqa: E402
    ensemble_sem_mag_from_residuals,
    ensemble_sem_mag_from_residuals_weighted,
)

RUN_SHA = "da9cce4a5edd1392b8ba842d3c8488589b9d0ac9"
S4 = ROOT / "dev" / "results" / "WIDE_ERR_03_S4.json"
PHOT = ROOT / "Archive" / "Drafts" / "draft_000515" / "platesolve" / "NoFilter_60_2" / "photometry"
OUT = ROOT / "dev" / "results" / "WIDE_ERR_03_S5.json"
SCINT_FLOOR_MMAG = 2.2


def med(vals: list[float]) -> float:
    a = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
    return float(np.median(a)) if a.size else float("nan")


def main() -> None:
    s4 = json.loads(S4.read_text(encoding="utf-8"))
    # Rebuild per-star rows from LC bins is lossy; re-read unique comps from S4 by
    # reconstructing from by_G_bin is insufficient. Use offline re-dump: store in S4?
    # S4 did not persist per-star unique_rows. Re-run minimal extract from S4 bright + bins.
    # Instead: recompute calibration input from S4 JSON by expanding is wrong.
    # Load from a companion if present, else reconstruct from re-running S4 unique via
    # reading the S4 file's lc frame medians only for gate - for full calib re-call S4 logic.
    # Practical: run a slim extraction script inline by importing S4 numbers from
    # re-measuring with saved cache - fastest path: re-exec unique_rows dump.

    # Re-load unique rows by re-running the measurement file's JSON if we add them.
    # Patch: read WIDE_ERR_03_S4_stars.json if we write it now from a quick re-call.
    stars_path = ROOT / "dev" / "results" / "WIDE_ERR_03_S4_stars.json"
    if not stars_path.is_file():
        # Derive synthetic per-bin representative stars from bin medians (weak).
        # Prefer calling the S4 tool's cached approach: write stars in a micro-pass.
        print("Rebuilding per-star table via S4 remeasure import...", flush=True)
        # Minimal: parse from running a small helper
        import subprocess

        subprocess.check_call(
            [sys.executable, str(ROOT / "dev" / "tools" / "wide_err_03_s4_export_stars.py")],
            cwd=str(ROOT),
        )
    stars = json.loads(stars_path.read_text(encoding="utf-8"))
    rows = [
        {
            "G": float(s["G"]),
            "scatter_mmag": float(s["lc_frame_scatter_mad_mmag"]),
            "err_model_mmag": float(s["err_model_new_mmag"]),
            "catalog_id": s["catalog_id"],
        }
        for s in stars
        if math.isfinite(float(s.get("G", float("nan"))))
        and math.isfinite(float(s.get("err_model_new_mmag", float("nan"))))
        and float(s["err_model_new_mmag"]) > 0
    ]

    bins = calibrate_bins(rows, min_n=2)
    # Pre chi2 fire at G8-9
    g89 = [r for r in rows if 8.0 < r["G"] <= 9.0]
    pre_r = med([r["scatter_mmag"] / r["err_model_mmag"] for r in g89]) if g89 else float("nan")
    chi2_fires = bool(math.isfinite(pre_r) and not (0.9 <= pre_r <= 1.1))

    # Post apply
    post_rows = []
    for r in rows:
        ecal = apply_calibration_mmag(r["err_model_mmag"], r["G"], bins)
        post_rows.append(
            {
                **r,
                "err_exported_mmag": ecal,
                "ratio_post": r["scatter_mmag"] / ecal if ecal and ecal > 0 else float("nan"),
            }
        )

    bin_labels = [(8 + 0.5 * i, 8 + 0.5 * (i + 1)) for i in range(15)]
    s5e = {}
    fail_bins = []
    for lo, hi in bin_labels:
        lab = f"({lo:.1f}, {hi:.1f}]"
        sub = [r for r in post_rows if lo < r["G"] <= hi]
        if len(sub) < 2:
            s5e[lab] = {"n": len(sub)}
            continue
        mr = med([r["ratio_post"] for r in sub])
        me = med([r["err_exported_mmag"] for r in sub])
        ok = 0.9 <= mr <= 1.1
        if lo < 9.0 and hi > 8.0:
            ok = ok and me >= SCINT_FLOOR_MMAG
        s5e[lab] = {
            "n": len(sub),
            "median_ratio": mr,
            "median_err_exported_mmag": me,
            "in_window": ok,
        }
        if not ok:
            fail_bins.append({"bin": lab, **s5e[lab]})

    g89_post = [r for r in post_rows if 8.0 < r["G"] <= 9.0]
    g89_ratio = med([r["ratio_post"] for r in g89_post])
    g89_err = med([r["err_exported_mmag"] for r in g89_post])

    # Weighted SEM fire: equal w -> identity; unequal -> differs
    x = [0.01, -0.02, 0.015, -0.005, 0.0]
    w_eq = [1.0] * 5
    w_uneq = [1.0, 4.0, 1.0, 0.25, 1.0]
    sem_u = ensemble_sem_mag_from_residuals(x)
    sem_we = ensemble_sem_mag_from_residuals_weighted(x, w_eq)
    sem_wu = ensemble_sem_mag_from_residuals_weighted(x, w_uneq)
    sem_proof = {
        "equal_weights_match_unweighted": abs(sem_u - sem_we) < 1e-12,
        "unequal_weights_differ": abs(sem_wu - sem_u) > 1e-6,
        "sem_unweighted": sem_u,
        "sem_weighted_equal_w": sem_we,
        "sem_weighted_unequal_w": sem_wu,
        "ratio_unequal_over_unweighted": (sem_wu / sem_u) if sem_u else float("nan"),
    }

    # Legacy model byte-identity: applying s=1, sigma_r=0 leaves err unchanged
    legacy = [
        abs(apply_calibration_mmag(e, 12.0, []) - e) < 1e-15
        for e in [5.0, 10.0, 20.0]
    ]
    # Also empty bins -> identity
    legacy_ok = all(legacy)

    var_guard = s4.get("variable_guard") or {}

    accept = len(fail_bins) == 0 and math.isfinite(g89_ratio) and 0.9 <= g89_ratio <= 1.1 and g89_err >= SCINT_FLOOR_MMAG

    calib_payload = {
        "task": "WIDE-ERR-03",
        "run_sha": RUN_SHA,
        "draft_id": 515,
        "form": "err_exported^2 = (s * err_model)^2 + sigma_r^2",
        "domain_sigma_r": "rel-flux",
        "gain_authority": s4.get("gain_new_container"),
        "bins": bins_to_dicts(bins),
        "n_calib_stars": len(rows),
    }
    write_sidecar(PHOT / ERR_CALIB_SIDECAR, calib_payload)

    payload = {
        "task": "WIDE-ERR-03 Stage S5",
        "run_sha": RUN_SHA,
        "s5_scope": s4.get("s4b_gate", {}).get("stage5_scope"),
        "calibration": calib_payload,
        "fire_proofs": {
            "chi2_pre_g8_9_median_ratio": pre_r,
            "chi2_gate_fires_pre": chi2_fires,
            "chi2_post_g8_9_median_ratio": g89_ratio,
            "variable_guard": var_guard,
            "weighted_sem": sem_proof,
            "legacy_identity_empty_calib": legacy_ok,
        },
        "s5e_acceptance": {
            "pass": accept,
            "by_G_bin": s5e,
            "bins_outside_window": fail_bins,
            "G8_9": {
                "median_ratio": g89_ratio,
                "median_err_exported_mmag": g89_err,
                "scint_floor_mmag": SCINT_FLOOR_MMAG,
            },
        },
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("WROTE", OUT)
    print("calib bins", len(bins))
    print("chi2 pre", pre_r, "fires", chi2_fires, "post", g89_ratio)
    print("S5e", "PASS" if accept else "FAIL", "outside", fail_bins)
    print("sem_proof", sem_proof)
    if not accept:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
