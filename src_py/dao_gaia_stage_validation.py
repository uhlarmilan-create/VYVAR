"""DAO-GAIA-ERA-01 A-fix 2: STAGE-01 iter4 validation gate for derived tolerances."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from dao_gaia_calibration import DerivedTolerances, ValidationGateResult

MAX_REGRESSION_PP = 0.005  # 0.5 percentage points
G2_PASS_MAX = 0.01  # STAGE-01 G2: empty-sky false-accept <= 1%

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


def _load_hand_baseline(repo_root: Path) -> dict[str, dict[str, Any]]:
    csv_path = (
        repo_root
        / "dev"
        / "results"
        / "context"
        / "session_20260819_daostage01_iter4"
        / "final_scores.csv"
    )
    if not csv_path.is_file():
        raise FileNotFoundError(f"hand-validated iter4 baseline missing: {csv_path}")
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
    return by_frame


def _import_iter4(repo_root: Path):
    src = repo_root / "src_py"
    tmp = repo_root / "tmp"
    for p in (src, tmp):
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)
    import dao_gaia_stage_01_iter4 as iter4  # noqa: PLC0415

    return iter4


def _params_from_derived(
    derived: DerivedTolerances,
    *,
    pass1_sigma: float,
    pass2_sigma: float,
    seed_snr_min: float,
    iter4_mod: Any,
) -> Any:
    return iter4_mod.ValidationParams(
        pass1_sigma=float(pass1_sigma),
        pass2_sigma=float(pass2_sigma),
        match_radius_px=float(derived.match_radius_px),
        pass2_center_tol_px=float(derived.pass2_center_tol_px),
        seed_centroid_max_px=float(derived.forced_seed_centroid_max_px),
        seed_snr_min=float(seed_snr_min),
    )


def _score_rows_to_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r["frame"]): r for r in rows}


def _compare_scores(
    hand: dict[str, dict[str, Any]],
    derived: dict[str, dict[str, Any]],
) -> tuple[dict[str, float], list[str], bool]:
    regressions: dict[str, float] = {}
    failures: list[str] = []
    g2_pass = True

    for frame, hand_row in hand.items():
        der_row = derived.get(frame)
        if der_row is None:
            failures.append(f"{frame}: missing derived score row")
            continue

        if hand_row.get("g4_ok") and not der_row.get("g4_ok"):
            failures.append(f"{frame}: g4_ok regressed True -> False")

        for metric in HIGHER_IS_BETTER:
            h = float(hand_row.get(metric, float("nan")))
            d = float(der_row.get(metric, float("nan")))
            if not (np.isfinite(h) and np.isfinite(d)):
                continue
            reg = h - d  # positive = derived worse
            regressions[f"{frame}/{metric}"] = reg
            if reg > MAX_REGRESSION_PP:
                failures.append(
                    f"{frame}/{metric}: derived {d:.4f} vs hand {h:.4f} "
                    f"(regression {reg * 100:.2f} pp > {MAX_REGRESSION_PP * 100:.1f} pp)"
                )

        for metric in LOWER_IS_BETTER:
            h = float(hand_row.get(metric, float("nan")))
            d = float(der_row.get(metric, float("nan")))
            if not np.isfinite(d):
                continue
            if metric == "g2":
                if d > G2_PASS_MAX:
                    g2_pass = False
                    failures.append(f"{frame}/g2: {d:.4f} > {G2_PASS_MAX:.2f} (G2 audit FAIL)")
            if not np.isfinite(h):
                continue
            reg = d - h  # positive = derived worse
            regressions[f"{frame}/{metric}"] = reg
            if reg > MAX_REGRESSION_PP:
                failures.append(
                    f"{frame}/{metric}: derived {d:.4f} vs hand {h:.4f} "
                    f"(regression {reg * 100:.2f} pp > {MAX_REGRESSION_PP * 100:.1f} pp)"
                )

    return regressions, failures, g2_pass


def run_validation_gate(
    derived: DerivedTolerances,
    *,
    pass1_sigma: float,
    pass2_sigma: float,
    seed_snr_min: float,
    repo_root: Path | str | None = None,
) -> ValidationGateResult:
    """Re-score derived tolerances with STAGE-01 iter4; compare to hand baseline."""
    root = _repo_root(repo_root)
    iter4 = _import_iter4(root)
    from config import AppConfig  # noqa: PLC0415

    gaia_db = Path(AppConfig().gaia_db_path)
    hand_baseline = _load_hand_baseline(root)

    hand_params = iter4.ValidationParams.hand_validated()
    derived_params = _params_from_derived(
        derived,
        pass1_sigma=pass1_sigma,
        pass2_sigma=pass2_sigma,
        seed_snr_min=seed_snr_min,
        iter4_mod=iter4,
    )

    rng = np.random.default_rng(51604)
    hand_rows = iter4.score_validation_params(hand_params, gaia_db=gaia_db, rng=rng)
    derived_rows = iter4.score_validation_params(derived_params, gaia_db=gaia_db, rng=rng)

    hand_map = _score_rows_to_map(hand_rows)
    derived_map = _score_rows_to_map(derived_rows)

    regressions, failures, g2_pass = _compare_scores(hand_baseline, derived_map)

    max_reg = float(max(regressions.values(), default=0.0))
    status = "PASS" if not failures else "FAIL"
    fail_reason = "; ".join(failures) if failures else None

    return ValidationGateResult(
        status=status,
        fail_reason=fail_reason,
        max_regression_pp=max_reg,
        hand_scores={"baseline_csv": hand_baseline, "recomputed": hand_map},
        derived_scores=derived_map,
        regressions=regressions,
        g2_pass=g2_pass,
    )


def write_validation_artifact(
    result: ValidationGateResult,
    out_dir: Path | str,
    *,
    derived_params: dict[str, float],
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "validation_gate.json"
    payload = {
        "status": result.status,
        "fail_reason": result.fail_reason,
        "max_regression_pp": result.max_regression_pp,
        "g2_pass": result.g2_pass,
        "derived_params": derived_params,
        "derived_scores": result.derived_scores,
        "regressions": result.regressions,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
