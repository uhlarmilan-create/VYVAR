"""Per-frame ePSF accounting and INV-PSF-FRAME-01 enforcement."""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from invariants_runtime import InvariantViolation, inv_check

LOGGER = logging.getLogger(__name__)

INV_PSF_FRAME_01 = "INV-PSF-FRAME-01"
DEFAULT_ZERO_OK_FRAME_FRACTION_FAIL = 0.20
_TRACEBACK_TAIL_LINES = 8
_JOB_SUMMARY_NAME = "epsf_photometry_job_summary.json"


def make_empty_frame_record(
    *,
    frame_name: str = "",
    frame_index: int | None = None,
) -> dict[str, Any]:
    return {
        "frame_name": frame_name,
        "frame_index": frame_index,
        "n_fit": 0,
        "n_ok": 0,
        "exception_class": None,
        "exception_message": None,
        "traceback_tail": None,
    }


def record_from_worker(raw: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not raw or not isinstance(raw, Mapping):
        return None
    return dict(raw)


def summarize_epsf_frame_job(
    records: Sequence[Mapping[str, Any]],
    *,
    science_set_meta: Mapping[str, Any] | None = None,
    fail_fraction: float = DEFAULT_ZERO_OK_FRAME_FRACTION_FAIL,
) -> dict[str, Any]:
    """Aggregate per-frame PSF records into a job summary dict."""
    recs = [dict(r) for r in records if r]
    frames_total = len(recs)
    zero_ok = [r for r in recs if int(r.get("n_ok") or 0) == 0]
    frames_with_zero_ok = len(zero_ok)
    fraction = (frames_with_zero_ok / frames_total) if frames_total else 0.0

    exc_hist: Counter[str] = Counter()
    first_fail: dict[str, Any] | None = None
    for r in recs:
        if int(r.get("n_ok") or 0) == 0 and r.get("exception_class"):
            cls = str(r["exception_class"])
            exc_hist[cls] += 1
            if first_fail is None:
                first_fail = {
                    "frame_name": r.get("frame_name"),
                    "frame_index": r.get("frame_index"),
                    "exception_class": cls,
                    "exception_message": r.get("exception_message"),
                }

    n_fit_total = sum(int(r.get("n_fit") or 0) for r in recs)
    n_ok_total = sum(int(r.get("n_ok") or 0) for r in recs)

    summary: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "frames_total": frames_total,
        "frames_with_zero_ok": frames_with_zero_ok,
        "frames_with_zero_ok_fraction": round(fraction, 6),
        "inv_psf_frame_01_threshold": float(fail_fraction),
        "n_fit_total": n_fit_total,
        "n_ok_total": n_ok_total,
        "first_failing_frame": first_fail,
        "exception_histogram_by_class": dict(exc_hist),
        "per_frame_records": recs,
    }
    if science_set_meta:
        summary["science_set"] = dict(science_set_meta)
    return summary


def enforce_inv_psf_frame_01(
    summary: Mapping[str, Any],
    *,
    pipeline_meta: dict[str, Any] | None = None,
    fail_fraction: float = DEFAULT_ZERO_OK_FRAME_FRACTION_FAIL,
) -> str:
    """Apply INV-PSF-FRAME-01. Returns policy used: ``ok``, ``WARN``, or raises."""
    frames_total = int(summary.get("frames_total") or 0)
    if frames_total == 0:
        return "ok"

    zero_ok = int(summary.get("frames_with_zero_ok") or 0)
    fraction = zero_ok / frames_total
    detail = (
        f"frames_with_zero_ok={zero_ok}/{frames_total} "
        f"fraction={fraction:.4f} threshold={fail_fraction}"
    )

    meta = pipeline_meta if pipeline_meta is not None else {}
    if fraction > fail_fraction:
        inv_check(meta, INV_PSF_FRAME_01, False, policy="FAIL", detail=detail)
        raise InvariantViolation(f"{INV_PSF_FRAME_01}: {detail}")

    if zero_ok > 0:
        inv_check(meta, INV_PSF_FRAME_01, True, policy="WARN", detail=detail)
        LOGGER.warning("[%s] below FAIL threshold but %s", INV_PSF_FRAME_01, detail)
        return "WARN"

    inv_check(meta, INV_PSF_FRAME_01, True, policy="FAIL", detail=detail)
    return "ok"


def persist_epsf_job_summary(summary: Mapping[str, Any], platesolve_dir: Path | str) -> Path:
    out = Path(platesolve_dir) / _JOB_SUMMARY_NAME
    out.write_text(json.dumps(dict(summary), indent=2), encoding="utf-8")
    return out


def finalize_epsf_frame_job(
    records: Sequence[Mapping[str, Any]],
    *,
    platesolve_dir: Path | str,
    science_set_meta: Mapping[str, Any] | None = None,
    pipeline_meta: dict[str, Any] | None = None,
    fail_fraction: float = DEFAULT_ZERO_OK_FRAME_FRACTION_FAIL,
    persist: bool = True,
) -> dict[str, Any]:
    """Summarize, persist, and enforce INV-PSF-FRAME-01."""
    summary = summarize_epsf_frame_job(
        records,
        science_set_meta=science_set_meta,
        fail_fraction=fail_fraction,
    )
    summary["inv_psf_frame_01_policy"] = enforce_inv_psf_frame_01(
        summary,
        pipeline_meta=pipeline_meta,
        fail_fraction=fail_fraction,
    )
    if persist:
        summary["summary_json"] = str(
            persist_epsf_job_summary(summary, platesolve_dir)
        )
    return dict(summary)
