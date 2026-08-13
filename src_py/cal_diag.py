"""CAL-DIAG v2 / INV-CAL-01: derived calibration convention gate."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from astropy.io import fits

from calibration import DarkResampleMode, get_processed_master
from infolog import log_event

LOGGER = logging.getLogger(__name__)

# Implementation constants (VYVAR_CAL_DIAG_V2_SPEC.md -- not user config).
CAL_PED_INTERCEPT_MIN_EXPTIMES = 2
CAL_PED_BOOTSTRAP_N = 200
CAL_PED_SUBSAMPLE_N = 100_000
CAL_PED_CONSISTENCY_REL = 0.05
CAL_CONV_SUM_SCALE = 0.85
CAL_CONV_MEAN_SCALE = 1.15
CAL_SKY_HARD_SIGMA = 5.0
CAL_SKY_SAT_WARN_FRAC = 0.90
CAL_MAD_SIGMA = 1.4826
CAL_DARK_CURRENT_NEGLIGIBLE_FRAC = 0.05

CalDiagConvention = Literal["SUM", "MEAN", "NONE", "PASSTHROUGH"]
CalDiagConventionSrc = Literal[
    "DERIVED",
    "INDETERMINATE_NEGLIGIBLE",
    "INDETERMINATE_UNMEASURED",
    "PASSTHROUGH",
]
CalDiagStatus = Literal["PASS", "WARN", "ABORT"]

MatchCropFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class PedestalResult:
    p_adu: float
    sigma_p: float
    k_adu_per_s: float
    k_status: Literal["FITTED", "NEGLIGIBLE", "UNKNOWN"]
    method: Literal["INTERCEPT", "SINGLE_MASTER_MEDIAN", "NONE"]
    n_exptimes: int = 0
    pedestal_measurable: bool = True
    check_p_consistent: bool = True


@dataclass
class CalDiagGateResult:
    """Outcome of one CAL-DIAG gate run per (obs_group, dark_path, light_binning)."""

    obs_group_key: str
    dark_path: str
    light_binning: int
    status: CalDiagStatus
    convention: CalDiagConvention
    convention_src: CalDiagConventionSrc = "DERIVED"
    aborted: bool = False
    abort_reason: str | None = None
    m_L: float | None = None
    m_D_sum: float | None = None
    m_D_mean: float | None = None
    block_factor: int = 1
    pedestal_p: float | None = None
    pedestal_sigma_p: float | None = None
    delta_dark: float | None = None
    delta_pred: float | None = None
    ratio_r: float | None = None
    s_sum: float | None = None
    s_mean: float | None = None
    sky_median: float | None = None
    sigma_r: float | None = None
    saturation_adu: float | None = None
    resolv_limit_adu: float | None = None
    messages: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CalDiagSession:
    gate_results: dict[str, CalDiagGateResult] = field(default_factory=dict)
    aborted_groups: set[str] = field(default_factory=set)
    dark_cache: dict[str, np.ndarray] = field(default_factory=dict)
    checked_keys: set[str] = field(default_factory=set)

    def json_export(self) -> dict[str, Any]:
        return {
            "keys": {k: v.to_json_dict() for k, v in self.gate_results.items()},
            "aborted_groups": sorted(self.aborted_groups),
            "spec_version": "CAL-DIAG-v2",
        }


def cal_diag_gate_key(obs_group_key: str, dark_path: Path | str, light_binning: int) -> str:
    return f"{obs_group_key}|{Path(dark_path).resolve()}|b{int(light_binning)}"


def dark_np_cache_key(
    dark_path: Path,
    light_binning: int,
    master_binning: int | None,
    dark_resample_mode: DarkResampleMode,
) -> str:
    _mb_key = "hdr" if master_binning is None else str(int(master_binning))
    return f"{Path(dark_path).resolve()}|b{int(light_binning)}|mb{_mb_key}|{dark_resample_mode}"


def convention_to_dark_mode(convention: CalDiagConvention) -> DarkResampleMode:
    if convention == "MEAN":
        return "mean"
    return "sum"


def resolv_limit_adu(*, sigma_p: float, block_factor: int) -> float:
    bf = max(1, int(block_factor))
    if bf <= 1:
        return 1.0
    return max(3.0 * float(sigma_p) * float(bf * bf - 1), 1.0)


def _mad_sigma_r(arr: np.ndarray) -> float:
    d = np.asarray(arr, dtype=np.float64)
    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return 0.0
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    return CAL_MAD_SIGMA * mad


def _read_dark_median_and_exptime(path: Path) -> tuple[float, float] | None:
    try:
        with fits.open(path, memmap=False) as hdul:
            data = hdul[0].data
            hdr = hdul[0].header
        if data is None:
            return None
        ex = float(hdr.get("EXPTIME") or hdr.get("EXPOSURE") or 0.0)
        med = float(np.nanmedian(np.asarray(data, dtype=np.float64)))
        if not math.isfinite(med):
            return None
        return med, ex
    except (OSError, TypeError, ValueError):
        return None


def derive_pedestal_from_masters(
    dark_paths: list[Path],
    *,
    light_exptime: float | None = None,
) -> PedestalResult:
    """Check P: derive per-bin1-pixel pedestal from master dark(s)."""
    uniq: dict[Path, tuple[float, float]] = {}
    for p in dark_paths:
        pp = Path(p)
        if not pp.is_file():
            continue
        got = _read_dark_median_and_exptime(pp)
        if got is None:
            continue
        uniq[pp.resolve()] = got

    if not uniq:
        return PedestalResult(
            p_adu=0.0,
            sigma_p=0.0,
            k_adu_per_s=0.0,
            k_status="UNKNOWN",
            method="NONE",
            n_exptimes=0,
            pedestal_measurable=False,
            check_p_consistent=False,
        )

    if len(uniq) >= CAL_PED_INTERCEPT_MIN_EXPTIMES:
        exps = sorted({ex for _, ex in uniq.values() if ex > 0})
        if len(exps) >= 2:
            rng = np.random.default_rng(42)
            paths_list = list(uniq.keys())
            samples_p: list[float] = []
            samples_k: list[float] = []
            for pth in paths_list:
                with fits.open(pth, memmap=False) as hdul:
                    arr = np.asarray(hdul[0].data, dtype=np.float64)
                n = arr.size
                take = min(CAL_PED_SUBSAMPLE_N, n)
                idx = rng.choice(n, size=take, replace=False)
                samples_p.append(float(np.nanmedian(arr.ravel()[idx])))
            # Pairwise slope from medians (robust for stacked masters).
            items = [(ex, med) for med, ex in uniq.values() if ex > 0]
            items.sort(key=lambda x: x[0])
            t0, m0 = items[0]
            t1, m1 = items[-1]
            if t1 > t0:
                k = (m1 - m0) / (t1 - t0)
            else:
                k = 0.0
            p = m0 - k * t0
            boots: list[float] = []
            for _ in range(CAL_PED_BOOTSTRAP_N):
                j = rng.integers(0, len(items))
                ex_b, med_b = items[j]
                boots.append(med_b - k * ex_b)
            sigma_p = float(np.std(boots)) if boots else 0.0
            k_status: Literal["FITTED", "NEGLIGIBLE", "UNKNOWN"] = "FITTED"
            t_ref = float(light_exptime or t0)
            if abs(k) * max(t_ref, t0, t1) < CAL_DARK_CURRENT_NEGLIGIBLE_FRAC * max(abs(p), 1e-6):
                k_status = "NEGLIGIBLE"
            return PedestalResult(
                p_adu=float(p),
                sigma_p=sigma_p,
                k_adu_per_s=float(k),
                k_status=k_status,
                method="INTERCEPT",
                n_exptimes=len(exps),
                pedestal_measurable=True,
                check_p_consistent=True,
            )

    # Single-exposure fallback.
    pth = next(iter(uniq.keys()))
    med, ex = uniq[pth]
    return PedestalResult(
        p_adu=float(med),
        sigma_p=max(float(med) * 1e-4, 0.01),
        k_adu_per_s=0.0,
        k_status="UNKNOWN",
        method="SINGLE_MASTER_MEDIAN",
        n_exptimes=1,
        pedestal_measurable=True,
        check_p_consistent=False,
    )


def _load_resampled_dark(
    *,
    dark_path: Path,
    light_binning: int,
    master_binning: int | None,
    light_shape: tuple[int, int],
    light_filename: str,
    dark_resample_mode: DarkResampleMode,
) -> tuple[np.ndarray, int]:
    pm = get_processed_master(
        dark_path,
        int(light_binning),
        kind="dark",
        master_binning=master_binning,
        light_shape=light_shape,
        light_filename=light_filename,
        dark_resample_mode=dark_resample_mode,
    )
    return pm.data, int(pm.block_factor)


def _check_b_sky(
    *,
    diff: np.ndarray,
    hard_sigma: float,
    sat_frac: float,
    saturation_adu: float | None,
    obs_group_key: str,
    messages: list[str],
    status: CalDiagStatus,
) -> tuple[CalDiagStatus, float, float]:
    s = float(np.nanmedian(diff))
    sigma_r = _mad_sigma_r(diff)
    if s < -hard_sigma * sigma_r:
        return "ABORT", s, sigma_r
    st = status
    if -hard_sigma * sigma_r <= s < 0:
        st = "WARN"
        wmsg = (
            f"INV-CAL-01 WARN [{obs_group_key}]: post-dark sky median {s:.4g} < 0 "
            f"(sigma_r={sigma_r:.4g})"
        )
        log_event(wmsg)
        messages.append(wmsg)
    if saturation_adu is not None and math.isfinite(float(saturation_adu)) and float(saturation_adu) > 0:
        if s > sat_frac * float(saturation_adu):
            st = "WARN"
            wmsg = (
                f"INV-CAL-01 WARN [{obs_group_key}]: sky median {s:.4g} > "
                f"{sat_frac:.2f}x saturation {float(saturation_adu):.4g}"
            )
            log_event(wmsg)
            messages.append(wmsg)
    return st, s, sigma_r


def cal_diag_gate_for_obs_group(
    *,
    repr_light_path: Path,
    dark_path: Path,
    obs_group_key: str,
    light_binning: int,
    master_binning: int | None,
    pedestal_dark_paths: list[Path] | None,
    match_and_crop_pair: MatchCropFn,
    saturation_adu: float | None,
    ui_error: Callable[[str], None] | None = None,
) -> CalDiagGateResult:
    """Run Check P, Check C, Check B on representative frame."""
    eps = CAL_PED_CONSISTENCY_REL

    with fits.open(repr_light_path, memmap=False) as hdul:
        light = np.array(hdul[0].data, dtype=np.float32, copy=True)
        light_ex = float(hdul[0].header.get("EXPTIME") or hdul[0].header.get("EXPOSURE") or 0.0)

    light_shape = (int(light.shape[0]), int(light.shape[1]))
    m_L = float(np.nanmedian(light))
    status: CalDiagStatus = "PASS"
    messages: list[str] = []

    dark_sum, bf = _load_resampled_dark(
        dark_path=dark_path,
        light_binning=light_binning,
        master_binning=master_binning,
        light_shape=light_shape,
        light_filename=repr_light_path.name,
        dark_resample_mode="sum",
    )
    dark_mean, _bf2 = _load_resampled_dark(
        dark_path=dark_path,
        light_binning=light_binning,
        master_binning=master_binning,
        light_shape=light_shape,
        light_filename=repr_light_path.name,
        dark_resample_mode="mean",
    )
    _, dark_sum_c = match_and_crop_pair(light, dark_sum)
    _, dark_mean_c = match_and_crop_pair(light, dark_mean)
    m_D_sum = float(np.nanmedian(dark_sum_c))
    m_D_mean = float(np.nanmedian(dark_mean_c))
    delta_dark = m_D_sum - m_D_mean

    light_b, dark_sum_b = match_and_crop_pair(light, dark_sum)
    _, dark_mean_b = match_and_crop_pair(light, dark_mean)
    s_sum = float(np.nanmedian(light_b - dark_sum_b))
    s_mean = float(np.nanmedian(light_b - dark_mean_b))

    ratio_r = m_L / m_D_mean if m_D_mean > 0 else float("inf")
    ratio_q = m_L / m_D_sum if m_D_sum > 0 else float("inf")

    ped_paths = list(pedestal_dark_paths or [dark_path])
    ped = derive_pedestal_from_masters(ped_paths, light_exptime=light_ex)
    res_limit = resolv_limit_adu(sigma_p=ped.sigma_p, block_factor=bf)
    k_t = ped.k_adu_per_s * light_ex if ped.k_status != "UNKNOWN" else 0.0
    delta_pred = (float(bf * bf - 1) * (ped.p_adu + k_t)) if bf > 1 else 0.0

    check_p_ok = True
    if bf > 1 and ped.pedestal_measurable and ped.method == "INTERCEPT":
        denom = max(abs(delta_dark), 1.0)
        rel = abs(delta_dark - delta_pred) / denom
        check_p_ok = rel <= eps
        if not check_p_ok:
            wmsg = (
                f"INV-CAL-01 WARN [{obs_group_key}]: Check P inconsistent "
                f"delta_dark={delta_dark:.4g} delta_pred={delta_pred:.4g} rel={rel:.4g}"
            )
            log_event(wmsg)
            messages.append(wmsg)
            status = "WARN"

    def _abort(reason: str, msg: str) -> CalDiagGateResult:
        full = (
            f"INV-CAL-01 ABORT [{obs_group_key}] {reason}: {msg} | "
            f"P={ped.p_adu:.4g}+/-{ped.sigma_p:.4g} bf={bf} "
            f"Delta_meas={delta_dark:.4g} Delta_pred={delta_pred:.4g} "
            f"R={ratio_r:.4g} s_SUM={s_sum:.4g} s_MEAN={s_mean:.4g}"
        )
        log_event(full)
        if ui_error is not None:
            ui_error(full)
        return CalDiagGateResult(
            obs_group_key=obs_group_key,
            dark_path=str(Path(dark_path).resolve()),
            light_binning=int(light_binning),
            status="ABORT",
            convention="SUM",
            convention_src="DERIVED",
            aborted=True,
            abort_reason=reason,
            m_L=m_L,
            m_D_sum=m_D_sum,
            m_D_mean=m_D_mean,
            block_factor=bf,
            pedestal_p=ped.p_adu,
            pedestal_sigma_p=ped.sigma_p,
            delta_dark=delta_dark,
            delta_pred=delta_pred,
            ratio_r=ratio_r,
            s_sum=s_sum,
            s_mean=s_mean,
            saturation_adu=saturation_adu,
            resolv_limit_adu=res_limit,
            messages=[full],
        )

    convention: CalDiagConvention = "SUM"
    convention_src: CalDiagConventionSrc = "DERIVED"

    if bf <= 1:
        convention = "NONE"
        convention_src = "DERIVED"
        if m_D_sum > m_L * (1.0 + eps):
            return _abort(
                "WRONG_MASTER",
                f"bf=1 dark median {m_D_sum:.4g} > light {m_L:.4g}",
            )
        light_u, dark_u = match_and_crop_pair(
            light,
            _load_resampled_dark(
                dark_path=dark_path,
                light_binning=light_binning,
                master_binning=master_binning,
                light_shape=light_shape,
                light_filename=repr_light_path.name,
                dark_resample_mode="sum",
            )[0],
        )
        st_b, sky, sig = _check_b_sky(
            diff=light_u - dark_u,
            hard_sigma=CAL_SKY_HARD_SIGMA,
            sat_frac=CAL_SKY_SAT_WARN_FRAC,
            saturation_adu=saturation_adu,
            obs_group_key=obs_group_key,
            messages=messages,
            status=status,
        )
        if st_b == "ABORT":
            return _abort(
                "CHECK_B_FAIL",
                f"sky_median={sky:.4g} sigma_r={sig:.4g}",
            )
        status = st_b
        if status == "PASS":
            log_event(
                f"INV-CAL-01 PASS [{obs_group_key}]: bf=1 sky={sky:.4g} "
                f"convention=NONE"
            )
        return CalDiagGateResult(
            obs_group_key=obs_group_key,
            dark_path=str(Path(dark_path).resolve()),
            light_binning=int(light_binning),
            status=status,
            convention=convention,
            convention_src=convention_src,
            m_L=m_L,
            m_D_sum=m_D_sum,
            m_D_mean=m_D_mean,
            block_factor=bf,
            pedestal_p=ped.p_adu,
            pedestal_sigma_p=ped.sigma_p,
            delta_dark=delta_dark,
            delta_pred=delta_pred,
            ratio_r=ratio_r,
            s_sum=s_sum,
            s_mean=s_mean,
            sky_median=sky,
            sigma_r=sig,
            saturation_adu=saturation_adu,
            resolv_limit_adu=res_limit,
            messages=messages,
        )

    # bf > 1: convention resolution.
    sum_supported = m_D_sum <= m_L * (1.0 + eps) and ratio_r >= float(bf * bf) * CAL_CONV_SUM_SCALE
    mean_supported = (
        m_D_sum > m_L * (1.0 + eps)
        and (m_D_sum / float(bf * bf)) <= m_L * (1.0 + eps)
    )

    if delta_dark < res_limit:
        convention = "SUM"
        convention_src = "INDETERMINATE_NEGLIGIBLE"
        wmsg = (
            f"INV-CAL-01 [{obs_group_key}]: convention separation {delta_dark:.4g} ADU "
            f"< resolv_limit {res_limit:.4g} ADU; SUM default (difference negligible)"
        )
        log_event(wmsg)
        messages.append(wmsg)
        status = "WARN"
    elif not ped.pedestal_measurable or (bf > 1 and ped.method == "INTERCEPT" and not check_p_ok):
        convention = "SUM"
        convention_src = "INDETERMINATE_UNMEASURED"
        wmsg = (
            f"INV-CAL-01 WARN [{obs_group_key}]: pedestal not fully measured "
            f"(method={ped.method}, check_p_ok={check_p_ok}); applying SUM with caution"
        )
        log_event(wmsg)
        if ui_error is not None:
            ui_error(wmsg)
        messages.append(wmsg)
        status = "WARN"
    elif sum_supported and not mean_supported:
        convention = "SUM"
        convention_src = "DERIVED"
    elif mean_supported and not sum_supported:
        convention = "MEAN"
        convention_src = "DERIVED"
    elif sum_supported and mean_supported:
        convention = "SUM"
        convention_src = "DERIVED"
    else:
        return _abort(
            "CONFLICT",
            f"SUM/MEAN both rejected: m_D_sum={m_D_sum:.4g} m_L={m_L:.4g} "
            f"R={ratio_r:.4g} bf={bf}",
        )

    mode = convention_to_dark_mode(convention)
    dark_use, _ = _load_resampled_dark(
        dark_path=dark_path,
        light_binning=light_binning,
        master_binning=master_binning,
        light_shape=light_shape,
        light_filename=repr_light_path.name,
        dark_resample_mode=mode,
    )
    light_c, dark_c = match_and_crop_pair(light, dark_use)
    st_b, sky, sig = _check_b_sky(
        diff=light_c - dark_c,
        hard_sigma=CAL_SKY_HARD_SIGMA,
        sat_frac=CAL_SKY_SAT_WARN_FRAC,
        saturation_adu=saturation_adu,
        obs_group_key=obs_group_key,
        messages=messages,
        status=status,
    )

    # CCD on-chip binning signature: SUM convention derived but SUM sky fails while MEAN counterfactual passes.
    if (
        convention == "SUM"
        and convention_src == "DERIVED"
        and st_b == "ABORT"
    ):
        st_mean, sky_m, sig_m = _check_b_sky(
            diff=light_b - dark_mean_b,
            hard_sigma=CAL_SKY_HARD_SIGMA,
            sat_frac=CAL_SKY_SAT_WARN_FRAC,
            saturation_adu=saturation_adu,
            obs_group_key=obs_group_key,
            messages=messages,
            status="PASS",
        )
        if st_mean != "ABORT" and delta_dark >= res_limit:
            return _abort(
                "CCD_LINEAR_INCONSISTENT",
                f"SUM sky fail (s={sky:.4g}) but MEAN counterfactual ok (s={sky_m:.4g}); "
                f"linear SUM resample invalid for this sensor class",
            )
        _ = sig_m

    if st_b == "ABORT":
        return _abort("CHECK_B_FAIL", f"sky_median={sky:.4g} sigma_r={sig:.4g}")

    status = st_b
    if status == "PASS" and convention_src.startswith("INDETERMINATE"):
        status = "WARN"

    if status in ("PASS", "WARN"):
        log_event(
            f"INV-CAL-01 {status} [{obs_group_key}]: convention={convention} "
            f"src={convention_src} P={ped.p_adu:.4g} bf={bf} "
            f"Delta={delta_dark:.4g} R={ratio_r:.4g} sky={sky:.4g}"
        )

    return CalDiagGateResult(
        obs_group_key=obs_group_key,
        dark_path=str(Path(dark_path).resolve()),
        light_binning=int(light_binning),
        status=status,
        convention=convention,
        convention_src=convention_src,
        m_L=m_L,
        m_D_sum=m_D_sum,
        m_D_mean=m_D_mean,
        block_factor=bf,
        pedestal_p=ped.p_adu,
        pedestal_sigma_p=ped.sigma_p,
        delta_dark=delta_dark,
        delta_pred=delta_pred,
        ratio_r=ratio_r,
        s_sum=s_sum,
        s_mean=s_mean,
        sky_median=sky,
        sigma_r=sig,
        saturation_adu=saturation_adu,
        resolv_limit_adu=res_limit,
        messages=messages,
    )


def ensure_cal_diag_gate(
    session: CalDiagSession,
    *,
    obs_group_key: str,
    repr_light_path: Path,
    dark_path: Path | None,
    light_binning: int,
    master_binning: int | None,
    pedestal_dark_paths: list[Path] | None,
    match_and_crop_pair: MatchCropFn,
    saturation_adu: float | None,
    ui_error: Callable[[str], None] | None = None,
) -> CalDiagGateResult | None:
    if dark_path is None or not Path(dark_path).is_file():
        return None
    gkey = cal_diag_gate_key(obs_group_key, dark_path, light_binning)
    if gkey in session.checked_keys:
        return session.gate_results.get(gkey)
    session.checked_keys.add(gkey)
    result = cal_diag_gate_for_obs_group(
        repr_light_path=repr_light_path,
        dark_path=Path(dark_path),
        obs_group_key=obs_group_key,
        light_binning=light_binning,
        master_binning=master_binning,
        pedestal_dark_paths=pedestal_dark_paths,
        match_and_crop_pair=match_and_crop_pair,
        saturation_adu=saturation_adu,
        ui_error=ui_error,
    )
    session.gate_results[gkey] = result
    if result.aborted:
        session.aborted_groups.add(obs_group_key)
    return result


def dark_np_for_cal_diag(
    session: CalDiagSession,
    *,
    master_binning: int | None,
    dark_path: Path | None,
    light_binning: int,
    light_shape: tuple[int, int] | None,
    light_filename: str,
    gate_result: CalDiagGateResult | None,
) -> np.ndarray | None:
    if dark_path is None or not dark_path.is_file():
        return None
    convention: CalDiagConvention = "SUM"
    if gate_result is not None:
        convention = gate_result.convention
    mode = convention_to_dark_mode(convention)
    key = dark_np_cache_key(dark_path, light_binning, master_binning, mode)
    if key not in session.dark_cache:
        pm = get_processed_master(
            dark_path,
            int(light_binning),
            kind="dark",
            master_binning=master_binning,
            light_shape=light_shape,
            light_filename=light_filename,
            dark_resample_mode=mode,
        )
        session.dark_cache[key] = pm.data
    return session.dark_cache[key]


def apply_cal_diag_headers(hdr: fits.Header, gate_result: CalDiagGateResult | None) -> None:
    if gate_result is None or gate_result.aborted:
        return
    conv = gate_result.convention
    if conv == "NONE":
        hdr["VY_DKRSMP"] = ("SUM", "CAL-DIAG: no resample; dark applied at matched binning")
    else:
        hdr["VY_DKRSMP"] = (conv, "CAL-DIAG dark resample convention applied")
    hdr["VY_DKRSMP_SRC"] = (
        gate_result.convention_src,
        "CAL-DIAG how resample convention was established",
    )
    if gate_result.sky_median is not None and math.isfinite(float(gate_result.sky_median)):
        hdr["VY_CDSKY"] = (
            float(gate_result.sky_median),
            "CAL-DIAG post-dark sky median ADU (repr frame)",
        )
    hdr["VY_CDSTAT"] = (gate_result.status, "CAL-DIAG gate outcome for obs_group")
    if gate_result.pedestal_p is not None and math.isfinite(float(gate_result.pedestal_p)):
        hdr["VY_CPED"] = (float(gate_result.pedestal_p), "CAL-DIAG derived bin1 pedestal ADU")


def passthrough_cal_diag_headers(hdr: fits.Header) -> None:
    hdr["VY_DKRSMP"] = ("PASSTHROUGH", "CAL-DIAG: no dark applied")
    hdr["VY_DKRSMP_SRC"] = ("PASSTHROUGH", "CAL-DIAG how resample convention was established")
    hdr["VY_CDSTAT"] = ("PASS", "CAL-DIAG gate outcome for obs_group")


def group_lights_by_obs_key(
    files: list[Path],
    *,
    obs_group_key_from_path: Callable[[Path], str],
) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for fp in files:
        gk = obs_group_key_from_path(fp)
        groups.setdefault(gk, []).append(fp)
    for gk in groups:
        groups[gk] = sorted(groups[gk], key=lambda p: str(p).casefold())
    return groups


def discover_pedestal_dark_paths(dark_path: Path) -> list[Path]:
    """Sibling library darks for pedestal intercept (same folder, Dark*.fits)."""
    dp = Path(dark_path).resolve()
    parent = dp.parent
    out: list[Path] = []
    for f in sorted(parent.glob("Dark*.fits")):
        if f.is_file():
            out.append(f)
    if dp not in [p.resolve() for p in out]:
        out.insert(0, dp)
    return out


def run_cal_diag_pregate(
    files: list[Path],
    *,
    obs_group_key_from_path: Callable[[Path], str],
    resolve_dark_path: Callable[[Path, str, int], Path | None],
    light_binning_from_path: Callable[[Path], int],
    master_binning: int | None,
    match_and_crop_pair: MatchCropFn,
    saturation_for_light: Callable[[Path], float | None],
    pedestal_paths_for_dark: Callable[[Path], list[Path]] | None = None,
    ui_error: Callable[[str], None] | None = None,
) -> CalDiagSession:
    session = CalDiagSession()
    groups = group_lights_by_obs_key(files, obs_group_key_from_path=obs_group_key_from_path)
    ped_fn = pedestal_paths_for_dark or discover_pedestal_dark_paths
    for og, paths in sorted(groups.items()):
        if not paths:
            continue
        repr_path = paths[0]
        lb = light_binning_from_path(repr_path)
        dark_p = resolve_dark_path(repr_path, og, lb)
        if dark_p is None:
            continue
        ensure_cal_diag_gate(
            session,
            obs_group_key=og,
            repr_light_path=repr_path,
            dark_path=dark_p,
            light_binning=lb,
            master_binning=master_binning,
            pedestal_dark_paths=ped_fn(dark_p),
            match_and_crop_pair=match_and_crop_pair,
            saturation_adu=saturation_for_light(repr_path),
            ui_error=ui_error,
        )
    return session


def write_cal_diag_json(archive_root: Path | str, session: CalDiagSession) -> Path | None:
    if not session.gate_results:
        return None
    root = Path(archive_root)
    out = root / "cal_diag.json"
    try:
        out.write_text(json.dumps(session.json_export(), indent=2), encoding="utf-8")
    except OSError as exc:
        log_event(f"CAL-DIAG: failed to write {out}: {exc}")
        return None
    return out


def load_cal_diag_json_for_meta(archive_path: Path | str) -> dict[str, Any] | None:
    p = Path(archive_path) / "cal_diag.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def gate_result_for_frame(
    session: CalDiagSession,
    *,
    obs_group_key: str,
    dark_path: Path | None,
    light_binning: int,
) -> CalDiagGateResult | None:
    if dark_path is None:
        return None
    gkey = cal_diag_gate_key(obs_group_key, dark_path, light_binning)
    return session.gate_results.get(gkey)


class CalStageCompareRefusedError(RuntimeError):
    """Raised when a pixel compare cannot proceed because stage is unknown."""


def calibrated_stage_from_header(hdr: fits.Header) -> tuple[str, int | None]:
    """Return ``(stage, sky_order)`` for an archived ``calibrated/lights`` FITS.

    Uses :func:`cal_stage.resolve_calibrated_stage`. Indeterminate resolutions return
    the indeterminate token with ``sky_order=None``.
    """
    from cal_stage import parse_cal_stage_token, resolve_calibrated_stage

    res = resolve_calibrated_stage(hdr)
    if res.is_indeterminate:
        return res.stage, None
    order, _pass_n = parse_cal_stage_token(res.stage)
    return res.stage, order


def apply_calibrated_stage_for_compare(
    data: np.ndarray,
    hdr: fits.Header,
    *,
    default_sky_order: int = 2,
) -> np.ndarray:
    """Match archived ``calibrated/lights`` processing stage before pixel compare.

    Recalibration produces pure ``(L-D)/F``. Archives may carry in-place preprocess
    sky-surface subtract(s). Refuses when stage is indeterminate (INV-CAL-02).
    """
    from cal_stage import resolve_calibrated_stage

    archive_res = resolve_calibrated_stage(hdr)
    if archive_res.is_indeterminate:
        raise CalStageCompareRefusedError(
            f"INV-CAL-02: refuse calibrated compare - archive {archive_res.confidence.value}: "
            f"{archive_res.reason or archive_res.stage}"
        )
    if not archive_res.stage.startswith("SKYSF"):
        return np.asarray(data, dtype=np.float32)
    sky_order = archive_res.sky_order if archive_res.sky_order is not None else int(default_sky_order)
    pass_n = max(1, int(archive_res.sky_pass))
    from pipeline import _fit_subtract_preprocess_sky_surface  # noqa: PLC0415

    out = np.asarray(data, dtype=np.float32)
    for _ in range(pass_n):
        out, _stats = _fit_subtract_preprocess_sky_surface(out, order=int(sky_order))
    return np.asarray(out, dtype=np.float32)


def calibrated_compare_refused(
    archive_hdr: fits.Header,
    *,
    fresh_hdr: fits.Header | None = None,
) -> str | None:
    """Return refusal reason when a calibrated pixel compare must not run."""
    from cal_stage import CalStageConfidence, CalStageResolution, refuse_calibrated_compare, resolve_calibrated_stage

    archive_res = resolve_calibrated_stage(archive_hdr)
    if fresh_hdr is None:
        fresh_res = CalStageResolution(
            stage="PURE",
            confidence=CalStageConfidence.LEGACY_INFERRED,
            reason="synthetic fresh recalibration",
        )
    else:
        fresh_res = resolve_calibrated_stage(fresh_hdr)
    return refuse_calibrated_compare(archive_res, fresh_res)


def is_obs_group_aborted(session: CalDiagSession, obs_group_key: str) -> bool:
    return obs_group_key in session.aborted_groups
