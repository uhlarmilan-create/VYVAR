"""INV-CAL-01 pre-registered validation (P1-P3) with stage-aware P2.

Run from repo root:
  python dev/tools/inv_cal01_validate.py
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from cal_diag import (  # noqa: E402
    CAL_PED_BOOTSTRAP_N,
    CAL_PED_SUBSAMPLE_N,
    apply_calibrated_stage_for_compare,
    calibrated_compare_refused,
    calibrated_stage_from_header,
    cal_diag_gate_for_obs_group,
)
from cal_stage import resolve_calibrated_stage  # noqa: E402
from config import AppConfig  # noqa: E402
from pipeline import (  # noqa: E402
    _match_and_crop_pair,
    calibrate_lights_to_calibrated,
)


@dataclass
class P2Result:
    draft: str
    n_frames: int
    n_identical: int
    max_abs_diff: float
    stage: str
    passed: bool


def _library_masters() -> tuple[Path, Path]:
    lib = ROOT / "CalibrationLibrary"
    md = sorted(lib.glob("Dark_60s*Bin1*.fits"))[0]
    mf = sorted(lib.glob("Flat*.fits"))[0]
    return md, mf


def _recalibrate_draft(draft: str, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    md, mf = _library_masters()
    raw = (ROOT / f"Archive/Drafts/draft_{int(draft):06d}/Raw/lights").resolve()
    calibrate_lights_to_calibrated(
        lights_root=raw,
        calibrated_root=out,
        master_dark_path=md,
        masterflat_by_filter={"NoFilter": mf},
        pipeline_config=AppConfig(),
    )


def validate_p1(draft: str = "435") -> dict[str, object]:
    out = ROOT / "tmp/_inv_cal01_p1"
    _recalibrate_draft(draft, out)
    arch = ROOT / f"Archive/Drafts/draft_{int(draft):06d}/calibrated/lights"
    files = sorted(arch.rglob("BO_CVn_Light_*.fits"))
    identical = 0
    max_diff = 0.0
    for af in files:
        rel = af.relative_to(arch)
        ff = out / rel
        with fits.open(af) as ha, fits.open(ff) as hf:
            da = np.asarray(ha[0].data)
            dn = np.asarray(hf[0].data)
        if np.array_equal(da, dn):
            identical += 1
        max_diff = max(max_diff, float(np.max(np.abs(da.astype(np.float64) - dn.astype(np.float64)))))
    hdr0 = fits.getheader(files[0])
    return {
        "draft": draft,
        "n_frames": len(files),
        "n_identical": identical,
        "max_abs_diff": max_diff,
        "passed": identical == len(files) and max_diff == 0.0,
        "VY_DKRSMP": hdr0.get("VY_DKRSMP"),
        "VY_DKRSMP_SRC": hdr0.get("VY_DKRSMP_SRC"),
        "VY_CDSKY": float(hdr0.get("VY_CDSKY", float("nan"))),
    }


def validate_p2(draft: str, fresh_root: Path | None = None) -> P2Result:
    """P2: recalibrated pure (L-D)/F + archived stage match vs on-disk archive."""
    arch = ROOT / f"Archive/Drafts/draft_{int(draft):06d}/calibrated/lights"
    if fresh_root is None:
        fresh_root = ROOT / f"tmp/_inv_cal01_p2_{draft}"
        _recalibrate_draft(draft, fresh_root)
    files = sorted(arch.rglob("BO_CVn_Light_*.fits"))
    identical = 0
    max_diff = 0.0
    stage_label = "PURE"
    for af in files:
        rel = af.relative_to(arch)
        ff = fresh_root / rel
        with fits.open(af) as ha, fits.open(ff) as hf:
            ah = ha[0].header
            archive_res = resolve_calibrated_stage(ah)
            stage_label = archive_res.stage
            if archive_res.is_indeterminate:
                return P2Result(
                    draft=draft,
                    n_frames=len(files),
                    n_identical=0,
                    max_abs_diff=float("nan"),
                    stage=stage_label,
                    passed=False,
                )
            refuse = calibrated_compare_refused(ah)
            if refuse:
                return P2Result(
                    draft=draft,
                    n_frames=len(files),
                    n_identical=0,
                    max_abs_diff=float("nan"),
                    stage=stage_label,
                    passed=False,
                )
            fresh_staged = apply_calibrated_stage_for_compare(
                np.asarray(hf[0].data, dtype=np.float32),
                ah,
            )
            arch_d = np.asarray(ha[0].data, dtype=np.float32)
        if np.array_equal(arch_d, fresh_staged):
            identical += 1
        max_diff = max(max_diff, float(np.max(np.abs(arch_d.astype(np.float64) - fresh_staged.astype(np.float64)))))
    return P2Result(
        draft=draft,
        n_frames=len(files),
        n_identical=identical,
        max_abs_diff=max_diff,
        stage=stage_label,
        passed=identical == len(files) and max_diff == 0.0,
    )


def _pixel_bootstrap_pedestal(d60: Path, d120: Path) -> tuple[float, float, float]:
    rng = np.random.default_rng(42)
    with fits.open(d60) as h0, fits.open(d120) as h1:
        y60 = np.asarray(h0[0].data, dtype=np.float64).ravel()
        y120 = np.asarray(h1[0].data, dtype=np.float64).ravel()
    n = min(CAL_PED_SUBSAMPLE_N, y60.size)
    idx = rng.choice(y60.size, size=n, replace=False)
    y60 = y60[idx]
    y120 = y120[idx]
    t60, t120 = 60.0, 120.0
    b = float(np.mean(y120 - y60) / (t120 - t60))
    a = float(np.mean(y60) - b * t60)
    boots: list[float] = []
    for _ in range(CAL_PED_BOOTSTRAP_N):
        j = rng.choice(n, size=n, replace=True)
        bb = float(np.mean(y120[j] - y60[j]) / (t120 - t60))
        boots.append(float(np.mean(y60[j]) - bb * t60))
    return a, float(np.std(boots)), b


def validate_p3(draft: str = "435") -> dict[str, object]:
    lib = ROOT / "CalibrationLibrary"
    d60 = sorted(lib.glob("Dark_60s*Bin1*.fits"))[0]
    d120 = sorted(lib.glob("Dark_120s*Bin1*.fits"))[0]
    light = ROOT / f"Archive/Drafts/draft_{int(draft):06d}/Raw/lights/NoFilter_60_2/BO_CVn_Light_001.fits"
    gr = cal_diag_gate_for_obs_group(
        repr_light_path=light,
        dark_path=d60,
        obs_group_key="NoFilter|60|2",
        light_binning=2,
        master_binning=1,
        pedestal_dark_paths=[d60, d120],
        match_and_crop_pair=_match_and_crop_pair,
        saturation_adu=65535.0,
    )
    px_p, px_sigma, px_k = _pixel_bootstrap_pedestal(d60, d120)
    bf = 2
    delta_pred_gate = (bf * bf - 1) * gr.pedestal_p
    delta_meas = gr.delta_dark
    return {
        "gate_P": gr.pedestal_p,
        "gate_sigma_p": gr.pedestal_sigma_p,
        "pixel_bootstrap_P": px_p,
        "pixel_bootstrap_sigma_p": px_sigma,
        "pixel_bootstrap_k": px_k,
        "Delta_meas": delta_meas,
        "Delta_pred_gate": delta_pred_gate,
        "Delta_pred_pixel": (bf * bf - 1) * px_p,
        "R": gr.ratio_r,
        "convention": gr.convention,
        "convention_src": gr.convention_src,
        "status": gr.status,
        "P_methods_agree_3sigma": abs(gr.pedestal_p - px_p) <= 3 * max(px_sigma, 1e-6),
        "Delta_consistent_5pct": abs(delta_meas - delta_pred_gate) / max(abs(delta_pred_gate), 1e-6) <= 0.05,
        "passed": gr.convention == "SUM"
        and gr.convention_src == "DERIVED"
        and gr.status == "PASS"
        and abs(delta_meas - delta_pred_gate) / max(abs(delta_pred_gate), 1e-6) <= 0.05,
    }


def main() -> int:
    print("INV-CAL-01 validation\n")
    p1 = validate_p1("435")
    print("P1", json.dumps(p1, indent=2))
    p2_509 = validate_p2("509")
    p2_510 = validate_p2("510")
    print(
        "P2",
        json.dumps(
            {
                "predicate": (
                    "Recalibrate pure (L-D)/F; apply archived VY_SKYSF/VYSKYORD "
                    "before pixel compare (stage-aware)."
                ),
                "509": p2_509.__dict__,
                "510": p2_510.__dict__,
                "passed": p2_509.passed and p2_510.passed,
            },
            indent=2,
        ),
    )
    p3 = validate_p3("435")
    print("P3", json.dumps(p3, indent=2))
    ok = bool(p1["passed"] and p2_509.passed and p2_510.passed and p3["passed"])
    print("\nOVERALL", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
