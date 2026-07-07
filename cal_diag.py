"""CAL-DIAG calibration-time radiometry gate (VYVAR_CAL_DIAG_SPEC v1.1)."""

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

CalDiagConvention = Literal["SUM", "MEAN_AUTOCORRECTED", "PASSTHROUGH"]
CalDiagStatus = Literal["PASS", "WARN", "ABORT"]

MatchCropFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class CalDiagGateResult:
    """Outcome of one CAL-DIAG gate run per (obs_group, dark_path, light_binning)."""

    obs_group_key: str
    dark_path: str
    light_binning: int
    status: CalDiagStatus
    convention: CalDiagConvention
    aborted: bool = False
    m_L: float | None = None
    m_S: float | None = None
    m_M: float | None = None
    block_factor: int = 1
    sky_median: float | None = None
    sigma_r: float | None = None
    saturation_adu: float | None = None
    messages: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CalDiagSession:
    """Per-calibrate-run gate state (single source of truth for convention + abort)."""

    gate_results: dict[str, CalDiagGateResult] = field(default_factory=dict)
    aborted_groups: set[str] = field(default_factory=set)
    dark_cache: dict[str, np.ndarray] = field(default_factory=dict)
    checked_keys: set[str] = field(default_factory=set)

    def json_export(self) -> dict[str, Any]:
        return {
            "keys": {k: v.to_json_dict() for k, v in self.gate_results.items()},
            "aborted_groups": sorted(self.aborted_groups),
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
    if convention == "MEAN_AUTOCORRECTED":
        return "mean"
    return "sum"


def _clamp_cfg_float(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return max(lo, min(hi, v))


def cal_diag_config_from_app(cfg: Any) -> dict[str, Any]:
    """Read CAL-DIAG knobs from AppConfig with spec clamps."""
    return {
        "enabled": bool(getattr(cfg, "cal_diag_gate_enabled", True)),
        "autocorrect": bool(getattr(cfg, "cal_diag_autocorrect_enabled", True)),
        "rel_tol": _clamp_cfg_float(getattr(cfg, "cal_diag_rel_tol", 0.02), 0.0, 0.2, 0.02),
        "hard_sigma": _clamp_cfg_float(getattr(cfg, "cal_diag_hard_sigma", 5.0), 3.0, 10.0, 5.0),
        "sat_warn_frac": _clamp_cfg_float(getattr(cfg, "cal_diag_sat_warn_frac", 0.90), 0.5, 1.0, 0.90),
    }


def _mad_sigma_r(arr: np.ndarray) -> float:
    d = np.asarray(arr, dtype=np.float64)
    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return 0.0
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    return 1.4826 * mad


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


def _cal_diag_gate_for_obs_group(
    *,
    repr_light_path: Path,
    dark_path: Path,
    obs_group_key: str,
    light_binning: int,
    master_binning: int | None,
    gate_cfg: dict[str, Any],
    match_and_crop_pair: MatchCropFn,
    saturation_adu: float | None,
    ui_error: Callable[[str], None] | None = None,
) -> CalDiagGateResult:
    """Check A + Check B on representative frame; does not cache dark arrays."""
    rel_tol = float(gate_cfg["rel_tol"])
    autocorrect = bool(gate_cfg["autocorrect"])
    hard_sigma = float(gate_cfg["hard_sigma"])
    sat_frac = float(gate_cfg["sat_warn_frac"])

    with fits.open(repr_light_path, memmap=False) as hdul:
        light = np.array(hdul[0].data, dtype=np.float32, copy=True)

    m_L = float(np.nanmedian(light))
    convention: CalDiagConvention = "SUM"
    status: CalDiagStatus = "PASS"
    messages: list[str] = []

    dark_sum, bf = _load_resampled_dark(
        dark_path=dark_path,
        light_binning=light_binning,
        master_binning=master_binning,
        light_shape=(int(light.shape[0]), int(light.shape[1])),
        light_filename=repr_light_path.name,
        dark_resample_mode="sum",
    )
    light_a, dark_a = match_and_crop_pair(light, dark_sum)
    m_S = float(np.nanmedian(dark_a))
    m_M: float | None = None

    def _fail(msg: str) -> CalDiagGateResult:
        log_event(f"CAL-DIAG ABORT [{obs_group_key}]: {msg}")
        if ui_error is not None:
            ui_error(f"CAL-DIAG ABORT [{obs_group_key}]: {msg}")
        return CalDiagGateResult(
            obs_group_key=obs_group_key,
            dark_path=str(Path(dark_path).resolve()),
            light_binning=int(light_binning),
            status="ABORT",
            convention=convention,
            aborted=True,
            m_L=m_L,
            m_S=m_S,
            m_M=m_M,
            block_factor=bf,
            saturation_adu=saturation_adu,
            messages=[msg],
        )

    if m_S <= m_L * (1.0 + rel_tol):
        convention = "SUM"
    else:
        if bf > 1:
            m_M = m_S / float(bf * bf)
            if m_M <= m_L * (1.0 + rel_tol) and autocorrect:
                convention = "MEAN_AUTOCORRECTED"
                status = "WARN"
                msg = (
                    f"CAL-DIAG AUTO-CORRECT [{obs_group_key}]: dark SUM median {m_S:.4g} > light "
                    f"{m_L:.4g}; applying MEAN resample (bf={bf}, m_M={m_M:.4g})"
                )
                log_event(msg)
                if ui_error is not None:
                    ui_error(msg)
                messages.append(msg)
                dark_mean, bf2 = _load_resampled_dark(
                    dark_path=dark_path,
                    light_binning=light_binning,
                    master_binning=master_binning,
                    light_shape=(int(light.shape[0]), int(light.shape[1])),
                    light_filename=repr_light_path.name,
                    dark_resample_mode="mean",
                )
                _, dark_a = match_and_crop_pair(light, dark_mean)
                m_S = float(np.nanmedian(dark_a))
                bf = bf2
                if m_S > m_L * (1.0 + rel_tol):
                    return _fail(
                        f"MEAN auto-correct still fails: m_S={m_S:.4g} m_L={m_L:.4g} bf={bf}"
                    )
            else:
                return _fail(
                    f"convention mismatch: m_S={m_S:.4g} m_L={m_L:.4g} m_M={m_M:.4g} bf={bf}"
                )
        else:
            return _fail(
                f"dark median exceeds light at bf=1 (wrong master pairing, hot dark, or "
                f"scaling error): m_S={m_S:.4g} m_L={m_L:.4g}"
            )

    dark_use, _ = _load_resampled_dark(
        dark_path=dark_path,
        light_binning=light_binning,
        master_binning=master_binning,
        light_shape=(int(light.shape[0]), int(light.shape[1])),
        light_filename=repr_light_path.name,
        dark_resample_mode=convention_to_dark_mode(convention),
    )
    light_b, dark_b = match_and_crop_pair(light, dark_use)
    diff = light_b - dark_b
    s = float(np.nanmedian(diff))
    sigma_r = _mad_sigma_r(diff)

    if s < -hard_sigma * sigma_r:
        return _fail(
            f"Check B HARD FAIL: sky_median={s:.4g} sigma_r={sigma_r:.4g} "
            f"(floor {-hard_sigma * sigma_r:.4g})"
        )

    if -hard_sigma * sigma_r <= s < 0:
        status = "WARN"
        wmsg = (
            f"CAL-DIAG WARN [{obs_group_key}]: post-dark sky median {s:.4g} < 0 "
            f"(sigma_r={sigma_r:.4g})"
        )
        log_event(wmsg)
        messages.append(wmsg)

    if saturation_adu is not None and math.isfinite(float(saturation_adu)) and float(saturation_adu) > 0:
        if s > sat_frac * float(saturation_adu):
            status = "WARN"
            wmsg = (
                f"CAL-DIAG WARN [{obs_group_key}]: sky median {s:.4g} > "
                f"{sat_frac:.2f}x saturation {float(saturation_adu):.4g}"
            )
            log_event(wmsg)
            messages.append(wmsg)

    if status == "PASS":
        log_event(
            f"CAL-DIAG PASS [{obs_group_key}]: convention={convention} m_L={m_L:.4g} "
            f"m_S={m_S:.4g} bf={bf} sky={s:.4g} sigma_r={sigma_r:.4g}"
        )

    return CalDiagGateResult(
        obs_group_key=obs_group_key,
        dark_path=str(Path(dark_path).resolve()),
        light_binning=int(light_binning),
        status=status,
        convention=convention,
        aborted=False,
        m_L=m_L,
        m_S=m_S,
        m_M=m_M,
        block_factor=bf,
        sky_median=s,
        sigma_r=sigma_r,
        saturation_adu=saturation_adu,
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
    gate_cfg: dict[str, Any],
    match_and_crop_pair: MatchCropFn,
    saturation_adu: float | None,
    ui_error: Callable[[str], None] | None = None,
) -> CalDiagGateResult | None:
    """Run gate once per key; return None when gate disabled or no dark."""
    if not gate_cfg.get("enabled", True):
        return None
    if dark_path is None or not Path(dark_path).is_file():
        return None
    gkey = cal_diag_gate_key(obs_group_key, dark_path, light_binning)
    if gkey in session.checked_keys:
        return session.gate_results.get(gkey)
    session.checked_keys.add(gkey)
    result = _cal_diag_gate_for_obs_group(
        repr_light_path=repr_light_path,
        dark_path=Path(dark_path),
        obs_group_key=obs_group_key,
        light_binning=light_binning,
        master_binning=master_binning,
        gate_cfg=gate_cfg,
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
    gate_enabled: bool,
) -> np.ndarray | None:
    """Cached dark array honoring CAL-DIAG convention (single source of truth)."""
    if dark_path is None or not dark_path.is_file():
        return None
    if gate_enabled and gate_result is not None:
        convention = gate_result.convention
    elif gate_enabled:
        convention = "SUM"
    else:
        convention = "SUM"
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


def apply_cal_diag_headers(hdr: fits.Header, gate_result: CalDiagGateResult | None, *, gate_enabled: bool) -> None:
    """Write VY_DKRSMP / VY_CDSKY / VY_CDSTAT on calibrated lights."""
    if not gate_enabled:
        return
    if gate_result is None:
        return
    if gate_result.aborted:
        return
    hdr["VY_DKRSMP"] = (gate_result.convention, "CAL-DIAG dark resample convention applied")
    if gate_result.sky_median is not None and math.isfinite(float(gate_result.sky_median)):
        hdr["VY_CDSKY"] = (float(gate_result.sky_median), "CAL-DIAG post-dark sky median ADU (repr frame)")
    hdr["VY_CDSTAT"] = (gate_result.status, "CAL-DIAG gate outcome for obs_group")


def passthrough_cal_diag_headers(hdr: fits.Header, *, gate_enabled: bool) -> None:
    if not gate_enabled:
        return
    hdr["VY_DKRSMP"] = ("PASSTHROUGH", "CAL-DIAG dark resample convention applied")
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


def run_cal_diag_pregate(
    files: list[Path],
    *,
    obs_group_key_from_path: Callable[[Path], str],
    resolve_dark_path: Callable[[Path, str, int], Path | None],
    light_binning_from_path: Callable[[Path], int],
    master_binning: int | None,
    gate_cfg: dict[str, Any],
    match_and_crop_pair: MatchCropFn,
    saturation_for_light: Callable[[Path], float | None],
    ui_error: Callable[[str], None] | None = None,
) -> CalDiagSession:
    """Parent pre-gate for MP variant (a): evaluate every obs_group key before workers."""
    session = CalDiagSession()
    if not gate_cfg.get("enabled", True):
        return session
    groups = group_lights_by_obs_key(files, obs_group_key_from_path=obs_group_key_from_path)
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
            gate_cfg=gate_cfg,
            match_and_crop_pair=match_and_crop_pair,
            saturation_adu=saturation_for_light(repr_path),
            ui_error=ui_error,
        )
    return session


def write_cal_diag_json(archive_root: Path | str, session: CalDiagSession, *, gate_enabled: bool) -> Path | None:
    """Write ``cal_diag.json`` under draft archive root when gate ran."""
    if not gate_enabled or not session.gate_results:
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
    """Load cal_diag block for pipeline_meta merge (missing file => None)."""
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


def is_obs_group_aborted(session: CalDiagSession, obs_group_key: str) -> bool:
    return obs_group_key in session.aborted_groups
