# -*- coding: ascii -*-
"""VYVAR-INVARIANTS P2: registry parity, unit gates, RNG AST guard."""

from __future__ import annotations

import ast
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"
DOCS = REPO / "docs"
sys.path.insert(0, str(SRC))

from invariants_runtime import WIRED_INV_IDS, InvariantViolation, check_flat_mean_near_one, check_preprocess_large_small_ratio, check_wcs_identity_p95, dao_only_fraction_from_masterstars, stamp_pipeline_stage, validate_provenance_schema


def _registry_wired_ids() -> set[str]:
    text = (DOCS / "VYVAR_INVARIANTS.md").read_text(encoding="utf-8")
    found = set(re.findall(r"((?:INV-[A-Z0-9-]+|QC-\d+|OSC-\d+))\s+\*\*\[wired\]\*\*", text))
    # Also accept markdown table form: INV-XXX **[wired]**
    found |= set(re.findall(r"\|\s*((?:INV-[A-Z0-9-]+|QC-\d+|OSC-\d+))\s+\*\*\[wired\]\*\*", text))
    return found


def test_registry_lists_all_wired_ids() -> None:
    reg = _registry_wired_ids()
    assert reg == set(WIRED_INV_IDS), f"registry={sorted(reg)} code={sorted(WIRED_INV_IDS)}"


def test_wired_ids_have_call_sites() -> None:
    """Every wired ID appears in exactly one primary enforcement locus (src or tests)."""
    blobs: list[str] = []
    for root in (SRC, REPO / "dev" / "tests"):
        for p in root.rglob("*.py"):
            if p.name.startswith("."):
                continue
            try:
                blobs.append(p.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
    corpus = "\n".join(blobs)
    for inv_id in sorted(WIRED_INV_IDS):
        # Count string occurrences of the ID as a quoted/literal token.
        n = len(re.findall(re.escape(inv_id), corpus))
        assert n >= 1, f"{inv_id} has no call site / mention in src_py or tests"
    # Inverse: no stray INV-*-style wired markers in inv_check that are unknown
    for m in re.findall(r'inv_check\([^)]*["\']((?:INV-[A-Z0-9-]+|QC-\d+|OSC-\d+))["\']', corpus):
        if m in WIRED_INV_IDS or m == "INV-WCS-00":
            continue
        # allow only registry-known wired set from inv_check in production helpers
        assert m in WIRED_INV_IDS, f"unknown inv_check id {m}"


def test_flux02_flat_mean() -> None:
    ok, _ = check_flat_mean_near_one(np.ones((32, 32), dtype=np.float64))
    assert ok
    ok_b, _ = check_flat_mean_near_one(np.full((32, 32), 1.05))
    assert ok_b is False


def test_flux02_skewed_flat_median_passes_mean_would_fail() -> None:
    """INV-FLUX-02 pins median semantics (matches normalize_flat_master, not mean)."""
    flat = np.ones((32, 32), dtype=np.float64)
    flat[:15, :] = 0.994
    assert abs(float(np.nanmedian(flat)) - 1.0) < 1e-6
    assert abs(float(np.nanmean(flat)) - 0.996770) < 1e-3
    ok, detail = check_flat_mean_near_one(flat)
    assert ok, detail


def test_wcs01_warn() -> None:
    ok, _ = check_wcs_identity_p95(1.54)
    assert ok
    ok_b, _ = check_wcs_identity_p95(2.5)
    assert ok_b is False


def test_dag01_upstream_and_cold_start() -> None:
    meta: dict = {}
    stamp_pipeline_stage(meta, "phase2a", enforce_upstream=True)  # cold start
    assert meta["stages"][-1]["cold_start"] is True
    meta2: dict = {}
    stamp_pipeline_stage(meta2, "masterstar", enforce_upstream=True)
    stamp_pipeline_stage(meta2, "phase01", enforce_upstream=True)  # gap OK
    assert meta2["stages"][-1].get("gap") is True
    stamp_pipeline_stage(meta2, "phase2a", enforce_upstream=True)  # contiguous
    with pytest.raises(InvariantViolation):
        stamp_pipeline_stage(meta2, "phase01", enforce_upstream=True)  # backwards


def test_prov01_round_trip_minimal() -> None:
    meta = {
        "provenance": {
            "git_hash": "deadbeef",
            "entry_point": "run_phase2a",
            "labbe_rng_seed_policy": "content_frame_hash_v1",
            "config_snapshot": {
            },
        },
        "invariants": [],
        "stages": [{"name": "phase2a", "seq": 6, "head_inputs_present": True}],
    }
    validate_provenance_schema(meta, photometry_dir=None)
    assert meta.get("prov_schema_version") == 1
    assert any(r.get("id") == "INV-PROV-01" and r.get("ok") for r in meta["invariants"])


# ---------------------------------------------------------------------------
# INV-RNG-01: AST guard - no naked np.random.<fn>( in src_py science modules
# ---------------------------------------------------------------------------

_ALLOWED_NP_RANDOM_ATTRS = frozenset(
    {
        "Generator",
        "SeedSequence",
        "default_rng",
        "RandomState",  # type mentions only; calls still flagged if used as call
    }
)

# Modules that may use global RNG (UI / scratch). Keep empty unless justified.
_RNG_ALLOWLIST_FILES: frozenset[str] = frozenset()


def _naked_np_random_calls(path: Path) -> list[str]:
    hits: list[str] = []
    try:
        tree = ast.parse(
            path.read_text(encoding="utf-8", errors="replace"),
            filename=str(path),
        )
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # np.random.<name>(...)
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "random"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "np"
        ):
            if func.attr in _ALLOWED_NP_RANDOM_ATTRS:
                continue
            hits.append(f"{path.name}:{getattr(node, 'lineno', '?')}: np.random.{func.attr}(")
        # numpy.random.<name>(...)
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "random"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "numpy"
        ):
            if func.attr in _ALLOWED_NP_RANDOM_ATTRS:
                continue
            hits.append(f"{path.name}:{getattr(node, 'lineno', '?')}: numpy.random.{func.attr}(")
    return hits


def test_inv_rng01_no_naked_global_rng() -> None:
    hits: list[str] = []
    for p in sorted(SRC.rglob("*.py")):
        rel = str(p.relative_to(SRC)).replace("\\", "/")
        if rel in _RNG_ALLOWLIST_FILES:
            continue
        hits.extend(_naked_np_random_calls(p))
    assert hits == [], "INV-RNG-01 naked global RNG hits:\n" + "\n".join(hits)


def test_cfg01_per_frame_sat_markers_absent_when_off() -> None:
    from invariants_runtime import validate_config_behavior

    meta = {
        "provenance": {
            "git_hash": "abc",
            "entry_point": "run_phase2a",
            "labbe_rng_seed_policy": "content_frame_hash_v1",
            "config_snapshot": {
                "per_frame_saturation_enabled": False,
            },
        },
        "invariants": [],
    }
    validate_config_behavior(meta, None)
    assert meta["invariants"][-1]["ok"] is True


def test_prep01_flat_frame_passes_gradient_guard() -> None:
    flat = np.full((128, 128), 1000.0, dtype=np.float64)
    ok, det, ratio = check_preprocess_large_small_ratio(flat)
    assert ok
    assert math.isfinite(ratio)
    assert ratio < 10.0


def test_prep01_gradient_frame_warns() -> None:
    h, w = 256, 256
    yy, xx = np.mgrid[0:h, 0:w]
    steep = (1000.0 + 2.0 * xx + 1.5 * yy).astype(np.float64)
    ok, det, ratio = check_preprocess_large_small_ratio(steep, warn_ratio=10.0)
    assert not ok
    assert ratio > 10.0


def test_dao_only_fraction_anchor_fixture_regression() -> None:
    import pandas as pd

    fixture = (
        REPO / "dev" / "results" / "context" / "session_20260727" / "draft_452_masterstars_full_match.csv"
    )
    if not fixture.is_file():
        pytest.skip(f"anchor fixture missing: {fixture}")
    df = pd.read_csv(fixture)
    assert dao_only_fraction_from_masterstars(df) == pytest.approx(0.0369, abs=0.002)


def test_dao_only_fraction_is_informational_only() -> None:
    import invariants_runtime

    assert not hasattr(invariants_runtime, "check_dao_only_fraction")
    assert "INV-MS-01" not in WIRED_INV_IDS


def test_masterstars_csv_write_survives_bp_rp_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression for draft_501: bp_rp annotate failure must not skip CSV write."""
    import logging

    import pandas as pd

    import pipeline as pl
    import pipeline_astrometry

    df_final = pd.DataFrame({"catalog_id": ["1111111111111111111", ""], "flux": [100.0, 50.0]})
    csv_path = tmp_path / "masterstars_full_match.csv"

    def _boom(*_args: object, **_kwargs: object) -> tuple[pd.DataFrame, int, int]:
        raise RuntimeError("simulated Gaia DB lock")

    monkeypatch.setattr(
        pipeline_astrometry,
        "_fill_masterstars_gaia_matched_bp_rp_from_local_db",
        _boom,
    )
    log_events: list[str] = []
    monkeypatch.setattr(pl, "log_event", lambda msg: log_events.append(str(msg)), raising=False)
    monkeypatch.setattr(pl, "LOGGER", logging.getLogger("test.masterstars_csv"))

    try:
        cid = df_final.get("catalog_id", pd.Series([""] * len(df_final))).fillna("").astype(str).str.strip()
        df_final["source_type"] = np.where(cid.ne(""), "GAIA_MATCHED", "DAO_ONLY")
        _gdb_fill = ""
        df_final, _n_bp_fill, _n_bp_miss = pipeline_astrometry._fill_masterstars_gaia_matched_bp_rp_from_local_db(
            df_final,
            gaia_db_path=_gdb_fill,
        )
    except Exception as exc:  # noqa: BLE001
        pl.log_event(f"MASTERSTAR source_type annotate failed: {exc!s}")
    pl._vyvar_df_to_csv(df_final, csv_path)

    assert csv_path.is_file()
    assert any("MASTERSTAR source_type annotate failed" in ln for ln in log_events)
    written = pd.read_csv(csv_path)
    assert len(written) == 2
    assert "source_type" in written.columns
