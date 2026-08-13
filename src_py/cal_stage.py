"""INV-CAL-02: calibrated product stage stamp, resolve, and verify."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

LOGGER = logging.getLogger(__name__)

CAL_STAGE_SPEC_VERSION = "CAL-STAGE-v1"
CAL_STAGE_COHERENCE_TOL_ADU = 2.0

_STAGE_SKYSF_RE = re.compile(r"^SKYSF_(\d+)(?:_R(\d+))?$")


class CalStageConfidence(str, Enum):
    AUTHORITATIVE = "AUTHORITATIVE"
    MANIFEST_VERIFIED = "MANIFEST_VERIFIED"
    LEGACY_INFERRED = "LEGACY_INFERRED"
    INDETERMINATE_LEGACY = "INDETERMINATE_LEGACY"
    INDETERMINATE_UNKNOWN = "INDETERMINATE_UNKNOWN"


@dataclass(frozen=True)
class CalStageResolution:
    stage: str
    confidence: CalStageConfidence
    sky_order: int | None = None
    sky_pass: int = 1
    reason: str = ""

    @property
    def is_compare_ready(self) -> bool:
        return self.confidence in (
            CalStageConfidence.AUTHORITATIVE,
            CalStageConfidence.MANIFEST_VERIFIED,
            CalStageConfidence.LEGACY_INFERRED,
        )

    @property
    def is_indeterminate(self) -> bool:
        return self.confidence in (
            CalStageConfidence.INDETERMINATE_LEGACY,
            CalStageConfidence.INDETERMINATE_UNKNOWN,
        )


@dataclass
class CalStageVerifyFrame:
    path: str
    outcome: str
    stage: str | None = None
    confidence: str | None = None
    detail: str = ""


@dataclass
class CalStageVerifyReport:
    frames_total: int = 0
    pass_n: int = 0
    warn_coherence: int = 0
    fail_stamp: int = 0
    fail_corrupt: int = 0
    indeterminate_legacy: int = 0
    indeterminate_unknown: int = 0
    frames: list[CalStageVerifyFrame] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.fail_stamp == 0 and self.fail_corrupt == 0


def compute_fits_datasum(data: np.ndarray) -> str:
    """Registered FITS DATASUM string for a float32 image array."""
    from astropy.io.fits import PrimaryHDU

    arr = np.asarray(data, dtype=np.float32)
    hdu = PrimaryHDU(data=arr)
    hdu.add_datasum()
    return str(hdu.header["DATASUM"]).strip()


def verify_fits_datasum(data: np.ndarray, expected: str | None) -> bool:
    if expected is None or str(expected).strip() == "":
        return False
    try:
        return compute_fits_datasum(data) == str(expected).strip()
    except Exception:  # noqa: BLE001
        return False


def parse_cal_stage_token(stage: str) -> tuple[int | None, int]:
    """Return ``(sky_order, sky_pass)`` for stage tokens."""
    token = str(stage or "").strip().upper()
    if token in ("PURE", "PASSTHROUGH", ""):
        return None, 1
    m = _STAGE_SKYSF_RE.match(token)
    if not m:
        return None, 1
    order = int(m.group(1))
    pass_n = int(m.group(2)) if m.group(2) else 1
    return order, max(1, pass_n)


def skysf_stage_token(*, order: int, pass_n: int = 1) -> str:
    order_i = int(order)
    pass_i = max(1, int(pass_n))
    if pass_i <= 1:
        return f"SKYSF_{order_i}"
    return f"SKYSF_{order_i}_R{pass_i}"


def _header_bool(hdr: fits.Header, key: str) -> bool | None:
    if key not in hdr:
        return None
    v = hdr.get(key)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "1", "t", "yes"):
        return True
    if s in ("false", "0", "f", "no"):
        return False
    return None


def _header_has_vy_skysf(hdr: fits.Header) -> bool:
    v = _header_bool(hdr, "VY_SKYSF")
    return bool(v)


def _legacy_pure_evidence(hdr: fits.Header) -> bool:
    """Positive evidence of calibrate-only (no preprocess sky) on legacy frames."""
    if _header_has_vy_skysf(hdr):
        return False
    if hdr.get("VYSKYP2P") is not None:
        return False
    if _header_bool(hdr, "VYVARPR") is True:
        return False
    if str(hdr.get("VY_CALIB") or "").strip().upper() == "PASSTHROUGH":
        return False
    if hdr.get("VYVARCAL") is not None:
        return True
    if _header_bool(hdr, "VY_DARK") is not None or _header_bool(hdr, "VY_FLAT") is not None:
        return True
    if hdr.get("VY_CFLAG") is not None:
        return True
    if hdr.get("VY_QCBG") is not None:
        return True
    if hdr.get("VY_DKRSMP") is not None:
        return True
    return False


def _prior_skypass(hdr: fits.Header) -> int:
    if "VY_SKYPASS" in hdr:
        try:
            return max(1, int(hdr["VY_SKYPASS"]))
        except (TypeError, ValueError):
            pass
    stage = str(hdr.get("VY_CALSTAGE") or "").strip().upper()
    if stage:
        _, pass_n = parse_cal_stage_token(stage)
        return pass_n
    if _header_has_vy_skysf(hdr):
        return 1
    return 0


def compute_skysf_apply_stage(
    hdr: fits.Header,
    *,
    sky_order: int,
    force_reapply: bool,
) -> tuple[str, int]:
    """Stage token and pass count when applying (or re-applying) sky surface."""
    prior_pass = _prior_skypass(hdr)
    if force_reapply and prior_pass >= 1:
        pass_n = prior_pass + 1
    else:
        pass_n = 1
    return skysf_stage_token(order=int(sky_order), pass_n=pass_n), pass_n


def stamp_cal_stage_headers(
    hdr: fits.Header,
    data: np.ndarray,
    *,
    stage: str,
    pstbg: float | None = None,
    skypass: int | None = None,
) -> str:
    """Write ``VY_CALSTAGE`` + ``VY_CALDATASUM`` (+ optional post-sky QC). Returns datasum."""
    datasum = compute_fits_datasum(data)
    hdr["VY_CALSTAGE"] = (str(stage), "INV-CAL-02 calibrated product stage")
    hdr["VY_CALDATASUM"] = (datasum, "FITS DATASUM of primary data array")
    if pstbg is not None and math.isfinite(float(pstbg)):
        hdr["VY_PSTBG"] = (float(pstbg), "Post-stage sigma-clipped sky median [ADU]")
    if skypass is not None and int(skypass) > 0:
        hdr["VY_SKYPASS"] = (int(skypass), "Sky-surface subtract pass count on these pixels")
    return datasum


def resolve_calibrated_stage(
    hdr: fits.Header,
    *,
    manifest_row: dict[str, Any] | None = None,
    data: np.ndarray | None = None,
) -> CalStageResolution:
    """Resolve processing stage without assuming PURE on ambiguous legacy frames."""
    stage_kw = str(hdr.get("VY_CALSTAGE") or "").strip()
    if stage_kw:
        order, pass_n = parse_cal_stage_token(stage_kw)
        return CalStageResolution(
            stage=stage_kw.upper(),
            confidence=CalStageConfidence.AUTHORITATIVE,
            sky_order=order,
            sky_pass=pass_n,
            reason="VY_CALSTAGE",
        )

    if manifest_row and manifest_row.get("cal_stage"):
        m_stage = str(manifest_row["cal_stage"]).strip().upper()
        m_sum = manifest_row.get("cal_datasum")
        if data is not None and verify_fits_datasum(data, m_sum):
            order, pass_n = parse_cal_stage_token(m_stage)
            return CalStageResolution(
                stage=m_stage,
                confidence=CalStageConfidence.MANIFEST_VERIFIED,
                sky_order=order,
                sky_pass=pass_n,
                reason="manifest+cal_datasum",
            )

    if hdr.get("VYSKYP2P") is not None and not _header_has_vy_skysf(hdr):
        return CalStageResolution(
            stage="INDETERMINATE_LEGACY",
            confidence=CalStageConfidence.INDETERMINATE_LEGACY,
            reason="VYSKYP2P without VY_SKYSF",
        )

    if _header_has_vy_skysf(hdr):
        order: int | None = None
        if "VYSKYORD" in hdr:
            try:
                order = int(hdr["VYSKYORD"])
            except (TypeError, ValueError):
                order = None
        pass_n = _prior_skypass(hdr) or 1
        token = skysf_stage_token(order=order if order is not None else 0, pass_n=pass_n)
        if order is None:
            token = f"SKYSF_?"
        return CalStageResolution(
            stage=token,
            confidence=CalStageConfidence.LEGACY_INFERRED,
            sky_order=order,
            sky_pass=pass_n,
            reason="VY_SKYSF+VYSKYORD",
        )

    if str(hdr.get("VY_CALIB") or "").strip().upper() == "PASSTHROUGH":
        return CalStageResolution(
            stage="PASSTHROUGH",
            confidence=CalStageConfidence.LEGACY_INFERRED,
            reason="VY_CALIB=PASSTHROUGH",
        )

    if _legacy_pure_evidence(hdr):
        return CalStageResolution(
            stage="PURE",
            confidence=CalStageConfidence.LEGACY_INFERRED,
            reason="calibration markers without preprocess sky markers",
        )

    return CalStageResolution(
        stage="INDETERMINATE_UNKNOWN",
        confidence=CalStageConfidence.INDETERMINATE_UNKNOWN,
        reason="no stage evidence",
    )


def refuse_calibrated_compare(res_a: CalStageResolution, res_b: CalStageResolution) -> str | None:
    """Return refusal reason when a pixel compare must not proceed."""
    for label, res in (("archive", res_a), ("candidate", res_b)):
        if res.is_indeterminate:
            return f"{label} stage {res.confidence.value}: {res.reason or res.stage}"
        if not res.is_compare_ready:
            return f"{label} stage not compare-ready: {res.stage}"
    return None


def verify_cal_stage_frame(
    hdr: fits.Header,
    data: np.ndarray,
    *,
    manifest_row: dict[str, Any] | None = None,
    path: str = "",
) -> CalStageVerifyFrame:
    """Verify one calibrated frame; legacy indeterminate frames WARN, not FAIL."""
    resolution = resolve_calibrated_stage(hdr, manifest_row=manifest_row, data=data)
    if resolution.is_indeterminate:
        outcome = (
            "INDETERMINATE_LEGACY"
            if resolution.confidence == CalStageConfidence.INDETERMINATE_LEGACY
            else "INDETERMINATE_UNKNOWN"
        )
        return CalStageVerifyFrame(
            path=path,
            outcome=outcome,
            stage=resolution.stage,
            confidence=resolution.confidence.value,
            detail=resolution.reason,
        )

    stamped_sum = str(hdr.get("VY_CALDATASUM") or "").strip()
    if stamped_sum:
        if not verify_fits_datasum(data, stamped_sum):
            return CalStageVerifyFrame(
                path=path,
                outcome="FAIL_CORRUPT",
                stage=resolution.stage,
                confidence=resolution.confidence.value,
                detail="VY_CALDATASUM mismatch",
            )
    elif resolution.confidence == CalStageConfidence.AUTHORITATIVE:
        return CalStageVerifyFrame(
            path=path,
            outcome="FAIL_STAMP",
            stage=resolution.stage,
            confidence=resolution.confidence.value,
            detail="VY_CALSTAGE without VY_CALDATASUM",
        )

    if manifest_row and manifest_row.get("cal_stage"):
        m_stage = str(manifest_row.get("cal_stage")).strip().upper()
        m_sum = manifest_row.get("cal_datasum")
        if m_stage != resolution.stage.upper() or (
            m_sum and stamped_sum and str(m_sum).strip() != stamped_sum
        ):
            return CalStageVerifyFrame(
                path=path,
                outcome="FAIL_STAMP",
                stage=resolution.stage,
                confidence=resolution.confidence.value,
                detail="manifest cal_stage/cal_datasum disagree with FITS",
            )

    if resolution.stage.startswith("SKYSF") and hdr.get("VY_PSTBG") is not None:
        med = float(np.nanmedian(data))
        pst = float(hdr["VY_PSTBG"])
        if math.isfinite(med) and math.isfinite(pst) and abs(med - pst) > CAL_STAGE_COHERENCE_TOL_ADU:
            return CalStageVerifyFrame(
                path=path,
                outcome="WARN_COHERENCE",
                stage=resolution.stage,
                confidence=resolution.confidence.value,
                detail=f"|median-VY_PSTBG|={abs(med - pst):.4g} ADU",
            )

    return CalStageVerifyFrame(
        path=path,
        outcome="PASS",
        stage=resolution.stage,
        confidence=resolution.confidence.value,
    )


def verify_calibrated_tree(
    root: Path | str,
    *,
    manifest_files: list[dict[str, Any]] | None = None,
    glob_pattern: str = "**/*.fits",
) -> CalStageVerifyReport:
    """Verify all calibrated light FITS under ``root``."""
    root_p = Path(root)
    manifest_by_cal: dict[str, dict[str, Any]] = {}
    if manifest_files:
        for row in manifest_files:
            if not isinstance(row, dict):
                continue
            cal_p = row.get("calibrated_path") or row.get("cal_path")
            if cal_p:
                manifest_by_cal[str(Path(cal_p).as_posix())] = row
            rel = row.get("calibrated_rel")
            if rel:
                manifest_by_cal[str((root_p / rel).resolve())] = row

    report = CalStageVerifyReport()
    for fp in sorted(root_p.glob(glob_pattern)):
        if not fp.is_file():
            continue
        report.frames_total += 1
        with fits.open(fp, memmap=False) as hdul:
            data = np.asarray(hdul[0].data)
            hdr = hdul[0].header
        man = manifest_by_cal.get(str(fp.resolve())) or manifest_by_cal.get(str(fp.as_posix()))
        frame = verify_cal_stage_frame(hdr, data, manifest_row=man, path=str(fp))
        report.frames.append(frame)
        outcome = frame.outcome
        if outcome == "PASS":
            report.pass_n += 1
        elif outcome == "WARN_COHERENCE":
            report.warn_coherence += 1
            report.pass_n += 1
        elif outcome == "FAIL_STAMP":
            report.fail_stamp += 1
        elif outcome == "FAIL_CORRUPT":
            report.fail_corrupt += 1
        elif outcome == "INDETERMINATE_LEGACY":
            report.indeterminate_legacy += 1
        elif outcome == "INDETERMINATE_UNKNOWN":
            report.indeterminate_unknown += 1
    return report


def archive_stage_census(root: Path | str, *, glob_pattern: str = "**/*.fits") -> dict[str, int]:
    """Count resolver outcomes under a directory tree."""
    counts: dict[str, int] = {}
    root_p = Path(root)
    for fp in sorted(root_p.glob(glob_pattern)):
        if not fp.is_file():
            continue
        with fits.open(fp, memmap=False) as hdul:
            hdr = hdul[0].header
            data = np.asarray(hdul[0].data)
        res = resolve_calibrated_stage(hdr, data=data)
        key = f"{res.stage}|{res.confidence.value}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def write_cal_stage_json(
    draft_dir: Path | str,
    report: CalStageVerifyReport,
    *,
    draft_id: int | None = None,
) -> Path:
    """Write draft-level ``cal_stage.json`` summary."""
    draft_dir = Path(draft_dir)
    stages: dict[str, int] = {}
    for fr in report.frames:
        if fr.stage and not str(fr.stage).startswith("INDETERMINATE"):
            stages[fr.stage] = stages.get(fr.stage, 0) + 1
    payload = {
        "schema": "vyvar_cal_stage_v1",
        "spec_version": CAL_STAGE_SPEC_VERSION,
        "draft_id": draft_id,
        "frames_total": report.frames_total,
        "stages": stages,
        "verify_last": {
            "ut": datetime.now(timezone.utc).isoformat(),
            "pass": report.pass_n,
            "warn_coherence": report.warn_coherence,
            "fail_stamp": report.fail_stamp,
            "fail_corrupt": report.fail_corrupt,
            "indeterminate_legacy": report.indeterminate_legacy,
            "indeterminate_unknown": report.indeterminate_unknown,
        },
    }
    out = draft_dir / "cal_stage.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
