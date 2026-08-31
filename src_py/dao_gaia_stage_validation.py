"""DAO-GAIA-ERA-01 A-fix 2 / XFER-01: STAGE-01 iter4 sandbox drift gate.

XFER-01 (2026-08-24): the sandbox is always re-scored with
``ValidationParams.hand_validated()``. Draft-derived centroid tols stay
on the certificate for production photometry of the current set and are
never substituted into this compare.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

from dao_gaia_calibration import DerivedTolerances, ValidationGateResult

MAX_REGRESSION_PP = 0.005  # 0.5 percentage points
G2_PASS_MAX = 0.01  # STAGE-01 G2: empty-sky false-accept <= 1%
SANDBOX_DRAFT_ID = 516
HAND_CSV_REL = (
    Path("dev")
    / "results"
    / "context"
    / "session_20260819_daostage01_iter4"
    / "final_scores.csv"
)
HAND_MATCH_RADIUS_PX = 3.0
HAND_PASS2_CENTER_TOL_PX = 2.0
HAND_SEED_CENTROID_MAX_PX = 2.0
HAND_PASS1_SIGMA = 4.5
HAND_PASS2_SIGMA = 4.0
TOL_DRIFT_RATIO_WARN = 2.0
IDENTITY_STAMP_KEYS = (
    "gaia_fingerprint",
    "vsx_fingerprint",
    "sandbox",
    "hand_csv",
    "lock_rig",
    "production_tolerances",
    "tol_drift_warn",
)

HIGHER_IS_BETTER = (
    "g1_strict_le13",
    "g1_strict_le145",
    "g1_eye_le13",
    "g1_eye_le145",
    "g1_eye_seed_le13",
    "g1_eye_seed_le145",
)
LOWER_IS_BETTER = ("g2", "g3_g18")
SCORE_METRICS = (*HIGHER_IS_BETTER, *LOWER_IS_BETTER)


def _repo_root(repo_root: Path | str | None) -> Path:
    if repo_root is not None:
        p = Path(repo_root)
        if (p / "src_py").is_dir():
            return p
        if (p.parent / "src_py").is_dir():
            return p.parent
    return Path(__file__).resolve().parent.parent


def hand_csv_path(repo_root: Path | str | None = None) -> Path:
    return _repo_root(repo_root) / HAND_CSV_REL


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _identity_fail(reason: str) -> None:
    from invariants_runtime import InvariantViolation

    raise InvariantViolation("DAO-GAIA-IDENTITY", reason)


def _load_hand_baseline(repo_root: Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    csv_path = hand_csv_path(repo_root)
    if not csv_path.is_file():
        _identity_fail(f"hand-validated iter4 baseline missing: {csv_path}")
    by_frame: dict[str, dict[str, Any]] = {}
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            frame = str(row["frame"])
            parsed: dict[str, Any] = {"frame": frame}
            for key in SCORE_METRICS:
                val = row.get(key)
                if val in (None, "", "nan"):
                    parsed[key] = float("nan")
                else:
                    parsed[key] = float(val)
            g4 = row.get("g4_ok")
            parsed["g4_ok"] = str(g4).strip().lower() in {"true", "1", "yes"}
            by_frame[frame] = parsed
    if not by_frame:
        _identity_fail(f"hand-validated iter4 baseline empty: {csv_path}")
    return csv_path, by_frame


def _import_iter4(repo_root: Path):
    src = repo_root / "src_py"
    tmp = repo_root / "tmp"
    for p in (src, tmp):
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)
    import dao_gaia_stage_01_iter4 as iter4  # noqa: PLC0415

    return iter4


def sandbox_frame_paths(repo_root: Path | str | None = None) -> list[tuple[str, Path]]:
    """Frames actually scored by STAGE-01 iter4 (draft 516 WIDE sandbox)."""
    root = _repo_root(repo_root)
    _import_iter4(root)
    from dao_gaia_stage_01 import FRAMES  # noqa: PLC0415

    out: list[tuple[str, Path]] = []
    for label, fpath in FRAMES:
        out.append((str(label), Path(fpath)))
    if len(out) != 4:
        _identity_fail(f"sandbox FRAMES expected 4 entries, got {len(out)}")
    return out


def _score_rows_to_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r["frame"]): r for r in rows}


def _compare_scores(
    hand: dict[str, dict[str, Any]],
    sandbox: dict[str, dict[str, Any]],
) -> tuple[dict[str, float], list[str], bool]:
    regressions: dict[str, float] = {}
    failures: list[str] = []
    g2_pass = True

    for frame, hand_row in hand.items():
        sb_row = sandbox.get(frame)
        if sb_row is None:
            failures.append(f"{frame}: missing sandbox score row")
            continue

        if hand_row.get("g4_ok") and not sb_row.get("g4_ok"):
            failures.append(f"{frame}: g4_ok regressed True -> False")

        for metric in HIGHER_IS_BETTER:
            h = float(hand_row.get(metric, float("nan")))
            d = float(sb_row.get(metric, float("nan")))
            if not (np.isfinite(h) and np.isfinite(d)):
                continue
            reg = h - d  # positive = sandbox worse
            regressions[f"{frame}/{metric}"] = reg
            if reg > MAX_REGRESSION_PP:
                failures.append(
                    f"{frame}/{metric}: sandbox {d:.4f} vs hand {h:.4f} "
                    f"(regression {reg * 100:.2f} pp > {MAX_REGRESSION_PP * 100:.1f} pp)"
                )

        for metric in LOWER_IS_BETTER:
            h = float(hand_row.get(metric, float("nan")))
            d = float(sb_row.get(metric, float("nan")))
            if not np.isfinite(d):
                continue
            if metric == "g2":
                if d > G2_PASS_MAX:
                    g2_pass = False
                    failures.append(f"{frame}/g2: {d:.4f} > {G2_PASS_MAX:.2f} (G2 audit FAIL)")
            if not np.isfinite(h):
                continue
            reg = d - h  # positive = sandbox worse
            regressions[f"{frame}/{metric}"] = reg
            if reg > MAX_REGRESSION_PP:
                failures.append(
                    f"{frame}/{metric}: sandbox {d:.4f} vs hand {h:.4f} "
                    f"(regression {reg * 100:.2f} pp > {MAX_REGRESSION_PP * 100:.1f} pp)"
                )

    return regressions, failures, g2_pass


def _tol_ratio(derived_px: float, hand_px: float) -> float | None:
    d = float(derived_px)
    h = float(hand_px)
    if not (np.isfinite(d) and np.isfinite(h)):
        return None
    if d <= 0.0 or h <= 0.0:
        return None
    return max(d, h) / min(d, h)


def build_tol_drift_warn(derived: DerivedTolerances) -> dict[str, Any]:
    """Informational WARN when current-set derived tols differ from hand by >= 2x."""
    checks = (
        ("pass2_center_tol_px", float(derived.pass2_center_tol_px), HAND_PASS2_CENTER_TOL_PX),
        (
            "forced_seed_centroid_max_px",
            float(derived.forced_seed_centroid_max_px),
            HAND_SEED_CENTROID_MAX_PX,
        ),
        ("match_radius_px", float(derived.match_radius_px), HAND_MATCH_RADIUS_PX),
    )
    trips: list[dict[str, Any]] = []
    for name, der, hand in checks:
        ratio = _tol_ratio(der, hand)
        if ratio is None:
            continue
        if ratio >= TOL_DRIFT_RATIO_WARN:
            trips.append(
                {
                    "name": name,
                    "derived_px": der,
                    "hand_px": hand,
                    "ratio": ratio,
                }
            )
    if not trips:
        return {
            "status": "OK",
            "blocks": False,
            "message": None,
            "checks": [],
        }
    bits = [
        f"{t['name']} derived {t['derived_px']:.2f} px vs hand {t['hand_px']:.2f} px "
        f"(ratio {t['ratio']:.2f}x >= {TOL_DRIFT_RATIO_WARN:.0f}x)"
        for t in trips
    ]
    return {
        "status": "WARN",
        "blocks": False,
        "message": (
            "cross-rig / scale drift vs STAGE-01 hand tols (informational, not a gate): "
            + "; ".join(bits)
        ),
        "checks": trips,
        "current_set_plate_scale_arcsec_per_px": float(derived.plate_scale_arcsec_per_px),
        "current_set_fwhm_px": float(derived.fwhm_px),
        "lock_hand_pass2_center_tol_px": HAND_PASS2_CENTER_TOL_PX,
        "lock_hand_forced_seed_centroid_max_px": HAND_SEED_CENTROID_MAX_PX,
        "lock_hand_match_radius_px": HAND_MATCH_RADIUS_PX,
    }


def build_certificate_identity_stamps(
    derived: DerivedTolerances,
    *,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Mandatory identity stamps for a reference-bearing certificate. Fail loud."""
    root = _repo_root(repo_root)
    from config import AppConfig  # noqa: PLC0415
    from catalog_provenance import fingerprint_gaia_db, fingerprint_vsx_db  # noqa: PLC0415
    from dao_gaia_calibration import plate_scale_arcsec_per_px_from_wcs  # noqa: PLC0415

    cfg = AppConfig()
    gaia_fp = fingerprint_gaia_db(cfg.gaia_db_path)
    vsx_fp = fingerprint_vsx_db(cfg.vsx_local_db_path)
    if not gaia_fp or not gaia_fp.get("fingerprint_sha256"):
        _identity_fail(f"gaia fingerprint unavailable: {cfg.gaia_db_path!r}")
    if not vsx_fp or not vsx_fp.get("fingerprint_sha256"):
        _identity_fail(f"vsx fingerprint unavailable: {cfg.vsx_local_db_path!r}")

    csv_path = hand_csv_path(root)
    if not csv_path.is_file():
        _identity_fail(f"hand-validated iter4 baseline missing: {csv_path}")
    hand_sha = _sha256_file(csv_path)
    if not hand_sha:
        _identity_fail(f"hand CSV sha256 empty: {csv_path}")

    frames = sandbox_frame_paths(root)
    sha_by_label: dict[str, str] = {}
    for label, fpath in frames:
        if not fpath.is_file():
            _identity_fail(f"sandbox frame missing ({label}): {fpath}")
        sha_by_label[str(label)] = _sha256_file(fpath)
    need = ("MASTERSTAR", "Light_001", "Light_076", "Light_148")
    missing = [k for k in need if k not in sha_by_label]
    if missing:
        _identity_fail(f"sandbox SHA missing labels: {missing}")

    ms_path = dict(frames)["MASTERSTAR"]
    try:
        from astropy.io import fits  # noqa: PLC0415
        from astropy.wcs import WCS  # noqa: PLC0415
        from warnings import catch_warnings, simplefilter  # noqa: PLC0415
        from astropy.wcs import FITSFixedWarning  # noqa: PLC0415

        with fits.open(ms_path, memmap=False) as hdul:
            hdr = hdul[0].header
        fwhm = hdr.get("VY_FWHM", hdr.get("FWHM"))
        if fwhm is None:
            _identity_fail(f"sandbox MASTERSTAR missing VY_FWHM/FWHM: {ms_path}")
        fwhm_px = float(fwhm)
        if not np.isfinite(fwhm_px) or fwhm_px <= 0.0:
            _identity_fail(f"sandbox MASTERSTAR FWHM not finite: {fwhm!r}")
        with catch_warnings():
            simplefilter("ignore", FITSFixedWarning)
            wcs = WCS(hdr)
        plate_scale = plate_scale_arcsec_per_px_from_wcs(wcs)
        if not np.isfinite(plate_scale) or plate_scale <= 0.0:
            _identity_fail(f"sandbox MASTERSTAR plate scale unavailable: {ms_path}")
    except Exception as exc:  # noqa: BLE001
        from invariants_runtime import InvariantViolation  # noqa: PLC0415

        if isinstance(exc, InvariantViolation):
            raise
        _identity_fail(f"sandbox lock-rig stamp failed: {exc}")

    production = {
        "scope": "production_photometry_current_set",
        "derived_pass2_center_tol_px": float(derived.pass2_center_tol_px),
        "derived_forced_seed_centroid_max_px": float(derived.forced_seed_centroid_max_px),
        "derived_match_radius_px": float(derived.match_radius_px),
        "derived_plate_scale_arcsec_per_px": float(derived.plate_scale_arcsec_per_px),
        "derived_fwhm_px": float(derived.fwhm_px),
    }
    stamps = {
        "gaia_fingerprint": str(gaia_fp["fingerprint_sha256"]),
        "vsx_fingerprint": str(vsx_fp["fingerprint_sha256"]),
        "sandbox": {
            "draft_id": int(SANDBOX_DRAFT_ID),
            "masterstar_sha256": sha_by_label["MASTERSTAR"],
            "light_001_sha256": sha_by_label["Light_001"],
            "light_076_sha256": sha_by_label["Light_076"],
            "light_148_sha256": sha_by_label["Light_148"],
        },
        "hand_csv": {
            "path": str(csv_path.as_posix()),
            "sha256": hand_sha,
        },
        "lock_rig": {
            "draft_id": int(SANDBOX_DRAFT_ID),
            "plate_scale_arcsec_per_px": float(plate_scale),
            "fwhm_px": float(fwhm_px),
            "note": "STAGE-01 iter4 sandbox (draft 516 WIDE)",
        },
        "production_tolerances": production,
        "tol_drift_warn": build_tol_drift_warn(derived),
        "sandbox_params": "hand_validated",
        "sandbox_params_detail": {
            "match_radius_px": HAND_MATCH_RADIUS_PX,
            "pass2_center_tol_px": HAND_PASS2_CENTER_TOL_PX,
            "forced_seed_centroid_max_px": HAND_SEED_CENTROID_MAX_PX,
            "pass1_sigma": HAND_PASS1_SIGMA,
            "pass2_sigma": HAND_PASS2_SIGMA,
        },
    }
    for key in IDENTITY_STAMP_KEYS:
        if stamps.get(key) in (None, "", {}):
            _identity_fail(f"identity stamp {key} empty")
    return stamps


def run_validation_gate(
    derived: DerivedTolerances,
    *,
    pass1_sigma: float,
    pass2_sigma: float,
    seed_snr_min: float,
    repo_root: Path | str | None = None,
) -> ValidationGateResult:
    """Re-score the 516 sandbox with hand-validated params; compare to hand CSV.

    ``derived`` / ``pass1_sigma`` / ``pass2_sigma`` / ``seed_snr_min`` are the
    current-set production knobs. They are intentionally unused for scoring
    (DAO-GAIA-XFER-01). Kept on the signature so callers stay stable.
    """
    del derived, pass1_sigma, pass2_sigma, seed_snr_min
    root = _repo_root(repo_root)
    iter4 = _import_iter4(root)
    from config import AppConfig  # noqa: PLC0415

    gaia_db = Path(AppConfig().gaia_db_path)
    if not gaia_db.is_file():
        _identity_fail(f"gaia db missing for sandbox rescore: {gaia_db}")
    csv_path, hand_baseline = _load_hand_baseline(root)

    hand_params = iter4.ValidationParams.hand_validated()
    rng = np.random.default_rng(51604)
    sandbox_rows = iter4.score_validation_params(hand_params, gaia_db=gaia_db, rng=rng)
    sandbox_map = _score_rows_to_map(sandbox_rows)

    regressions, failures, g2_pass = _compare_scores(hand_baseline, sandbox_map)

    max_reg = float(max(regressions.values(), default=0.0))
    status = "PASS" if not failures else "FAIL"
    fail_reason = "; ".join(failures) if failures else None

    return ValidationGateResult(
        status=status,
        fail_reason=fail_reason,
        max_regression_pp=max_reg,
        hand_scores={"baseline_csv": hand_baseline, "csv_path": str(csv_path.as_posix())},
        derived_scores=sandbox_map,
        regressions=regressions,
        g2_pass=g2_pass,
    )


