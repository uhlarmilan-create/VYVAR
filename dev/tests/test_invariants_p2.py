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

from invariants_runtime import (  # noqa: E402
    COG_META_KEYS,
    PER_FRAME_SAT_META_KEYS,
    WIRED_INV_IDS,
    InvariantViolation,
    check_dao_only_fraction,
    check_dark_resample_flux_conservation,
    check_flat_mean_near_one,
    check_preprocess_large_small_ratio,
    check_residual_flatness,
    check_wcs_identity_p95,
    dao_only_fraction_from_masterstars,
    inv_check,
    preprocess_large_small_ratio,
    stamp_pipeline_stage,
    uniform_sum_preserving_upscale,
    validate_config_behavior,
    validate_provenance_schema,
)


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


def test_flux01_downscale_and_upscale() -> None:
    rng = np.random.default_rng(0)
    src = rng.uniform(10.0, 100.0, size=(64, 64)).astype(np.float64)
    bf = 2
    # block-sum downscale
    h, w = src.shape
    v = src.reshape(h // bf, bf, w // bf, bf)
    out = np.sum(v, axis=(1, 3))
    ok, det = check_dark_resample_flux_conservation(src, out, block_factor=bf, mode="sum")
    assert ok, det
    # uniform upscale
    up = uniform_sum_preserving_upscale(src, bf)
    ok_u, det_u = check_dark_resample_flux_conservation(src, up, block_factor=bf, mode="upscale")
    assert ok_u, det_u
    # violation
    bad = out * 1.01
    ok_b, _ = check_dark_resample_flux_conservation(src, bad, block_factor=bf, mode="sum")
    assert ok_b is False
    meta: dict = {}
    with pytest.raises(InvariantViolation):
        inv_check(meta, "INV-FLUX-01", False, policy="FAIL", detail="boom")


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


def test_flat01_warn_band() -> None:
    flat = np.zeros((64, 64), dtype=np.float64)
    ok, det, p99 = check_residual_flatness(flat, p99_max_adu=400.0)
    assert ok, det
    yy, xx = np.mgrid[0:64, 0:64]
    steep = (xx.astype(float) * 20.0).astype(np.float64)
    ok2, det2, p99_2 = check_residual_flatness(steep, p99_max_adu=50.0)
    assert ok2 is False
    assert p99_2 > 50.0
    meta: dict = {"invariants": []}
    inv_check(meta, "INV-FLAT-01", False, policy="WARN", detail=det2)
    assert meta["invariants"][-1]["policy"] == "WARN"
    assert meta["invariants"][-1]["ok"] is False


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


def test_cfg01_cog_keys_absent_when_disabled() -> None:
    meta = {
        "provenance": {
            "git_hash": "abc",
            "entry_point": "run_phase2a",
            "labbe_rng_seed_policy": "content_frame_hash_v1",
            "config_snapshot": {
                "cog_aperture_correction_enabled": False,
                "psf_photometry_enabled": False,
                "temporal_binning_enabled": False,
            },
        },
        "invariants": [],
    }
    validate_config_behavior(meta, None)
    assert meta["invariants"][-1]["ok"] is True
    meta_bad = dict(meta)
    meta_bad["cog_night_fallback"] = False
    meta_bad["invariants"] = []
    with pytest.raises(InvariantViolation):
        validate_config_behavior(meta_bad, None)


def test_prov01_round_trip_minimal() -> None:
    meta = {
        "provenance": {
            "git_hash": "deadbeef",
            "entry_point": "run_phase2a",
            "labbe_rng_seed_policy": "content_frame_hash_v1",
            "config_snapshot": {
                "cog_aperture_correction_enabled": False,
                "psf_photometry_enabled": False,
                "temporal_binning_enabled": False,
            },
        },
        "invariants": [],
        "stages": [{"name": "phase2a", "seq": 6, "head_inputs_present": True}],
    }
    validate_provenance_schema(meta, photometry_dir=None)
    assert meta.get("prov_schema_version") == 1
    assert any(r.get("id") == "INV-PROV-01" and r.get("ok") for r in meta["invariants"])


def test_prov01_requires_cog_key_when_enabled() -> None:
    meta = {
        "provenance": {
            "git_hash": "x",
            "entry_point": "run_phase2a",
            "labbe_rng_seed_policy": "content_frame_hash_v1",
            "config_snapshot": {
                "cog_aperture_correction_enabled": True,
                "psf_photometry_enabled": False,
                "temporal_binning_enabled": False,
            },
        },
        "invariants": [],
    }
    with pytest.raises(InvariantViolation):
        validate_provenance_schema(meta, photometry_dir=None)
    meta["cog_night_fallback"] = False
    meta["invariants"] = []
    validate_provenance_schema(meta, photometry_dir=None)


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


def test_cog_meta_keys_constant_matches_cfg_gate() -> None:
    assert "cog_night_fallback" in COG_META_KEYS


def test_cfg01_per_frame_sat_markers_absent_when_off() -> None:
    from invariants_runtime import validate_config_behavior

    meta = {
        "provenance": {
            "git_hash": "abc",
            "entry_point": "run_phase2a",
            "labbe_rng_seed_policy": "content_frame_hash_v1",
            "config_snapshot": {
                "cog_aperture_correction_enabled": False,
                "psf_photometry_enabled": False,
                "temporal_binning_enabled": False,
                "per_frame_saturation_enabled": False,
            },
        },
        "invariants": [],
    }
    validate_config_behavior(meta, None)
    assert meta["invariants"][-1]["ok"] is True
    meta_bad = dict(meta)
    meta_bad["per_frame_sat_enabled"] = True
    meta_bad["invariants"] = []
    with pytest.raises(InvariantViolation):
        validate_config_behavior(meta_bad, None)
    assert "per_frame_sat_enabled" in PER_FRAME_SAT_META_KEYS


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


def test_ms01_dao_only_fraction_warn_and_fail() -> None:
    import pandas as pd

    df_ok = pd.DataFrame({"catalog_id": ["1", "2", "3"], "source_type": ["GAIA_MATCHED"] * 3})
    ok, det, frac, pol = check_dao_only_fraction(df_ok)
    assert ok and pol == "ok"
    assert dao_only_fraction_from_masterstars(df_ok) == 0.0

    df_warn = pd.DataFrame(
        {
            "catalog_id": ["", ""] + ["2"] * 13,
            "source_type": ["DAO_ONLY", "DAO_ONLY"] + ["GAIA_MATCHED"] * 13,
        }
    )
    ok_w, _, frac_w, pol_w = check_dao_only_fraction(df_warn)
    assert not ok_w and pol_w == "WARN"
    assert abs(frac_w - (2.0 / 15.0)) < 1e-9

    df_fail = pd.DataFrame({"catalog_id": [""] * 4, "source_type": ["DAO_ONLY"] * 4})
    ok_f, _, frac_f, pol_f = check_dao_only_fraction(df_fail)
    assert not ok_f and pol_f == "FAIL"
    assert frac_f == 1.0
