#!/usr/bin/env python3
"""Per-draft err calibration: err_exported^2 = (s * err_model)^2 + sigma_r(G)^2.

WIDE-ERR-03B: smooth form (constant s, sigma_r constant or linear in G),
s >= 1 clamp (never deflate physical model). Legacy per-bin table kept for
sidecar read compatibility only.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


from mag_constants import MAG_ERR_SCALE

ERR_CALIB_SIDECAR = "err_calibration.json"
FormName = Literal["constant_sigma_r", "linear_sigma_r_G"]


@dataclass
class SmoothCalibration:
    """Global smooth calibration for one draft x rig."""

    s: float
    sigma_r0_mmag: float
    sigma_r_slope_mmag_per_G: float
    form: FormName
    n_stars: int
    s_clamped: bool
    median_ratio_pre: float
    median_ratio_post: float

    def sigma_r_mmag_at_G(self, g: float) -> float:
        if not math.isfinite(g):
            return max(0.0, float(self.sigma_r0_mmag))
        if self.form == "constant_sigma_r":
            return max(0.0, float(self.sigma_r0_mmag))
        # Linear in G; clamp non-negative. Monotone non-increasing in flux ~=
        # non-decreasing in G for a positive floor at the bright end.
        return max(0.0, float(self.sigma_r0_mmag) + float(self.sigma_r_slope_mmag_per_G) * float(g))

    def sigma_r_rel_at_G(self, g: float) -> float:
        return float(self.sigma_r_mmag_at_G(g)) / 1000.0 / MAG_ERR_SCALE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- Legacy per-bin (WIDE-ERR-03) kept for reading old sidecars ---


@dataclass
class BinCalibration:
    g_lo: float
    g_hi: float
    s: float
    sigma_r_rel: float
    n: int
    median_ratio_pre: float
    median_ratio_post: float


def apply_smooth_rel(err_model_rel: float, g: float, cal: SmoothCalibration) -> float:
    if not math.isfinite(err_model_rel) or err_model_rel <= 0:
        return float("nan")
    sr = cal.sigma_r_rel_at_G(g)
    return float(math.sqrt((cal.s * err_model_rel) ** 2 + sr * sr))


# --- Legacy API (03) ---


def apply_calibration_rel(
    err_model_rel: float,
    g: float,
    bins: list[BinCalibration] | SmoothCalibration,
) -> float:
    if isinstance(bins, SmoothCalibration):
        return apply_smooth_rel(err_model_rel, g, bins)
    if not math.isfinite(err_model_rel) or err_model_rel <= 0:
        return float("nan")
    match = None
    for b in bins:
        if b.g_lo < g <= b.g_hi:
            match = b
            break
    if match is None:
        return err_model_rel
    return float(
        math.sqrt((match.s * err_model_rel) ** 2 + match.sigma_r_rel * match.sigma_r_rel)
    )


def write_sidecar(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_sidecar(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def smooth_from_sidecar(data: dict[str, Any]) -> SmoothCalibration | None:
    sm = data.get("smooth")
    if not isinstance(sm, dict):
        return None
    return SmoothCalibration(
        s=float(sm["s"]),
        sigma_r0_mmag=float(sm.get("sigma_r0_mmag", 0.0)),
        sigma_r_slope_mmag_per_G=float(sm.get("sigma_r_slope_mmag_per_G", 0.0)),
        form=str(sm.get("form", "constant_sigma_r")),  # type: ignore[arg-type]
        n_stars=int(sm.get("n_stars", 0)),
        s_clamped=bool(sm.get("s_clamped", False)),
        median_ratio_pre=float(sm.get("median_ratio_pre", float("nan"))),
        median_ratio_post=float(sm.get("median_ratio_post", float("nan"))),
    )


def bins_from_sidecar(data: dict[str, Any]) -> list[BinCalibration]:
    out = []
    for b in data.get("bins") or []:
        out.append(
            BinCalibration(
                g_lo=float(b["g_lo"]),
                g_hi=float(b["g_hi"]),
                s=float(b["s"]),
                sigma_r_rel=float(b["sigma_r_rel"]),
                n=int(b["n"]),
                median_ratio_pre=float(b.get("median_ratio_pre", float("nan"))),
                median_ratio_post=float(b.get("median_ratio_post", float("nan"))),
            )
        )
    return out


