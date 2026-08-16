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

import numpy as np

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


def identity_smooth_calibration(*, n_stars: int = 0) -> SmoothCalibration:
    """Physical-model identity: s=1, sigma_r=0 (WIDE-ERR-04 default / C2 winner)."""
    return SmoothCalibration(
        s=1.0,
        sigma_r0_mmag=0.0,
        sigma_r_slope_mmag_per_G=0.0,
        form="constant_sigma_r",
        n_stars=int(n_stars),
        s_clamped=False,
        median_ratio_pre=float("nan"),
        median_ratio_post=float("nan"),
    )


def write_identity_sidecar(path: Path, *, draft_id: int | None = None, extra: dict[str, Any] | None = None) -> None:
    """Write s=1, sigma_r=0 sidecar (documented hook; no empirical floor)."""
    payload: dict[str, Any] = {
        "form": "err_exported^2 = (s * err_model)^2 + sigma_r(G)^2",
        "note": "Identity physical model (WIDE-ERR-04): s=1, sigma_r=0",
        "smooth": identity_smooth_calibration().to_dict(),
        "s_min_clamp": 1.0,
    }
    if draft_id is not None:
        payload["draft_id"] = int(draft_id)
    if extra:
        payload.update(extra)
    write_sidecar(path, payload)


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


def _bin_label(lo: float, hi: float) -> str:
    return f"({lo:.1f}, {hi:.1f}]"


def _median_ratio(scat: np.ndarray, err: np.ndarray) -> float:
    ok = np.isfinite(scat) & np.isfinite(err) & (err > 0)
    if not np.any(ok):
        return float("nan")
    return float(np.median(scat[ok] / err[ok]))


def _apply_smooth_mmag(err_mmag: np.ndarray, g: np.ndarray, cal: SmoothCalibration) -> np.ndarray:
    out = np.full_like(err_mmag, float("nan"), dtype=float)
    for i in range(err_mmag.size):
        e = float(err_mmag[i])
        if not (math.isfinite(e) and e > 0):
            continue
        sr = cal.sigma_r_mmag_at_G(float(g[i]))
        out[i] = math.sqrt((cal.s * e) ** 2 + sr * sr)
    return out


def _fit_constant_sigma_r(
    scat: np.ndarray,
    err: np.ndarray,
    *,
    s_min: float = 1.0,
) -> tuple[float, float, bool]:
    """Fit constant s>=s_min and constant sigma_r_mmag to median chi2~1."""
    excess = np.maximum(scat * scat - err * err, 0.0)
    sigma_r = float(math.sqrt(float(np.median(excess)))) if np.any(excess > 0) else 0.0
    err_f = np.sqrt(err * err + sigma_r * sigma_r)
    med_rf = _median_ratio(scat, err_f)
    s_raw = float(med_rf) if math.isfinite(med_rf) and med_rf > 0 else 1.0
    s_clamped = s_raw < s_min
    s = max(s_min, min(s_raw, 3.0))
    # Re-solve sigma_r after clamping s: scat^2 ~= (s*err)^2 + sigma_r^2
    excess2 = np.maximum(scat * scat - (s * err) ** 2, 0.0)
    sigma_r = float(math.sqrt(float(np.median(excess2)))) if np.any(excess2 > 0) else 0.0
    return s, sigma_r, s_clamped


def _fit_linear_sigma_r_G(
    scat: np.ndarray,
    err: np.ndarray,
    g: np.ndarray,
    *,
    s_min: float = 1.0,
) -> tuple[float, float, float, bool]:
    """Constant s>=s_min; sigma_r(G) = max(0, a + b*G) with b>=0 (floor grows toward faint)."""
    # Start from constant fit
    s, sig0, s_clamped = _fit_constant_sigma_r(scat, err, s_min=s_min)
    # Residuals vs G: target sigma_r_i^2 = max(0, scat^2 - (s*err)^2)
    y = np.maximum(scat * scat - (s * err) ** 2, 0.0)
    ok = np.isfinite(y) & np.isfinite(g)
    if int(np.sum(ok)) < 5:
        return s, sig0, 0.0, s_clamped
    gg = g[ok]
    yy = y[ok]
    # Fit yy = (a + b*G)^2 approximately via sqrt then linear
    z = np.sqrt(yy)
    # Least squares z = a + b*G with b >= 0, a >= 0
    A = np.column_stack([np.ones_like(gg), gg])
    try:
        coef, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        a, b = float(coef[0]), float(coef[1])
    except np.linalg.LinAlgError:
        return s, sig0, 0.0, s_clamped
    if b < 0:
        # Force non-decreasing in G: fall back to constant
        return s, sig0, 0.0, s_clamped
    a = max(0.0, a)
    return s, a, b, s_clamped


def calibrate_smooth(
    rows: list[dict[str, Any]],
    *,
    form: FormName | None = None,
    s_min: float = 1.0,
) -> SmoothCalibration:
    """Fit smooth calibration on all rows (scatter_mmag, err_model_mmag, G)."""
    scat = np.asarray([float(r["scatter_mmag"]) for r in rows], dtype=float)
    err = np.asarray([float(r["err_model_mmag"]) for r in rows], dtype=float)
    g = np.asarray([float(r["G"]) for r in rows], dtype=float)
    ok = np.isfinite(scat) & np.isfinite(err) & (err > 0) & np.isfinite(g)
    scat, err, g = scat[ok], err[ok], g[ok]
    n = int(scat.size)
    if n < 3:
        return SmoothCalibration(
            s=1.0,
            sigma_r0_mmag=0.0,
            sigma_r_slope_mmag_per_G=0.0,
            form="constant_sigma_r",
            n_stars=n,
            s_clamped=False,
            median_ratio_pre=float("nan"),
            median_ratio_post=float("nan"),
        )
    pre = _median_ratio(scat, err)

    candidates: list[SmoothCalibration] = []
    forms: list[FormName] = ["constant_sigma_r", "linear_sigma_r_G"] if form is None else [form]
    for f in forms:
        if f == "constant_sigma_r":
            s, sig0, clamped = _fit_constant_sigma_r(scat, err, s_min=s_min)
            cal = SmoothCalibration(
                s=s,
                sigma_r0_mmag=sig0,
                sigma_r_slope_mmag_per_G=0.0,
                form=f,
                n_stars=n,
                s_clamped=clamped,
                median_ratio_pre=pre,
                median_ratio_post=float("nan"),
            )
        else:
            s, a, b, clamped = _fit_linear_sigma_r_G(scat, err, g, s_min=s_min)
            cal = SmoothCalibration(
                s=s,
                sigma_r0_mmag=a,
                sigma_r_slope_mmag_per_G=b,
                form=f,
                n_stars=n,
                s_clamped=clamped,
                median_ratio_pre=pre,
                median_ratio_post=float("nan"),
            )
        post_err = _apply_smooth_mmag(err, g, cal)
        cal.median_ratio_post = _median_ratio(scat, post_err)
        candidates.append(cal)

    if form is not None:
        return candidates[0]

    # Choose by |median_ratio_post - 1| (in-sample tie-break; B2 held-out decides)
    def score(c: SmoothCalibration) -> float:
        r = c.median_ratio_post
        if not math.isfinite(r):
            return float("inf")
        return abs(r - 1.0)

    candidates.sort(key=score)
    return candidates[0]


def choose_form_by_heldout(
    calib_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    s_min: float = 1.0,
) -> SmoothCalibration:
    """Fit both forms on calib_rows; pick by eval held-out |median_ratio-1|."""
    best: SmoothCalibration | None = None
    best_score = float("inf")
    for f in ("constant_sigma_r", "linear_sigma_r_G"):
        cal = calibrate_smooth(calib_rows, form=f, s_min=s_min)  # type: ignore[arg-type]
        if not eval_rows:
            score = abs((cal.median_ratio_post or 1.0) - 1.0)
        else:
            scat = np.asarray([float(r["scatter_mmag"]) for r in eval_rows], dtype=float)
            err = np.asarray([float(r["err_model_mmag"]) for r in eval_rows], dtype=float)
            g = np.asarray([float(r["G"]) for r in eval_rows], dtype=float)
            ecal = _apply_smooth_mmag(err, g, cal)
            score = abs(_median_ratio(scat, ecal) - 1.0)
        if score < best_score:
            best_score = score
            best = cal
    assert best is not None
    return best


def apply_smooth_mmag(err_model_mmag: float, g: float, cal: SmoothCalibration) -> float:
    if not math.isfinite(err_model_mmag) or err_model_mmag <= 0:
        return float("nan")
    sr = cal.sigma_r_mmag_at_G(g)
    return float(math.sqrt((cal.s * err_model_mmag) ** 2 + sr * sr))


def apply_smooth_rel(err_model_rel: float, g: float, cal: SmoothCalibration) -> float:
    if not math.isfinite(err_model_rel) or err_model_rel <= 0:
        return float("nan")
    sr = cal.sigma_r_rel_at_G(g)
    return float(math.sqrt((cal.s * err_model_rel) ** 2 + sr * sr))


# --- Legacy API (03) ---


def calibrate_bins(
    rows: list[dict[str, Any]],
    *,
    bins: list[tuple[float, float]] | None = None,
    min_n: int = 2,
) -> list[BinCalibration]:
    """Deprecated per-bin fit (WIDE-ERR-03). Prefer calibrate_smooth."""
    if bins is None:
        bins = [(8 + 0.5 * i, 8 + 0.5 * (i + 1)) for i in range(15)]
    out: list[BinCalibration] = []
    for lo, hi in bins:
        sub = [
            r
            for r in rows
            if math.isfinite(float(r.get("G", float("nan"))))
            and lo < float(r["G"]) <= hi
            and math.isfinite(float(r.get("scatter_mmag", float("nan"))))
            and math.isfinite(float(r.get("err_model_mmag", float("nan"))))
            and float(r["err_model_mmag"]) > 0
        ]
        if len(sub) < min_n:
            continue
        scat = np.asarray([float(r["scatter_mmag"]) for r in sub], dtype=float)
        err = np.asarray([float(r["err_model_mmag"]) for r in sub], dtype=float)
        med_r = _median_ratio(scat, err)
        s, sig_mmag, _ = _fit_constant_sigma_r(scat, err, s_min=1.0)
        err_cal = np.sqrt((s * err) ** 2 + sig_mmag * sig_mmag)
        ratio_post = _median_ratio(scat, err_cal)
        sigma_r_rel = (sig_mmag / 1000.0) / MAG_ERR_SCALE if sig_mmag > 0 else 0.0
        out.append(
            BinCalibration(
                g_lo=lo,
                g_hi=hi,
                s=s,
                sigma_r_rel=sigma_r_rel,
                n=len(sub),
                median_ratio_pre=med_r,
                median_ratio_post=ratio_post,
            )
        )
    return out


def apply_calibration_mmag(
    err_model_mmag: float,
    g: float,
    bins: list[BinCalibration] | SmoothCalibration,
) -> float:
    if isinstance(bins, SmoothCalibration):
        return apply_smooth_mmag(err_model_mmag, g, bins)
    if not math.isfinite(err_model_mmag) or err_model_mmag <= 0:
        return float("nan")
    match = None
    for b in bins:
        if b.g_lo < g <= b.g_hi:
            match = b
            break
    if match is None:
        return err_model_mmag
    sigma_r_mmag = match.sigma_r_rel * MAG_ERR_SCALE * 1000.0
    return float(math.sqrt((match.s * err_model_mmag) ** 2 + sigma_r_mmag * sigma_r_mmag))


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


def bins_to_dicts(bins: list[BinCalibration]) -> list[dict[str, Any]]:
    return [asdict(b) | {"bin": _bin_label(b.g_lo, b.g_hi)} for b in bins]
