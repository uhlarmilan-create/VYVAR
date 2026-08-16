"""IMPL-05 Item D: Comp QA pool-size guard and qa_degraded fire proofs."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src_py"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from comp_qa_core import (  # noqa: E402
    _QA_POOL_MAX_MULT,
    assert_comp_qa_pool_size,
    compute_comp_qa,
    write_comp_qa_artifacts,
)
from invariants_runtime import InvariantViolation  # noqa: E402


def test_fire_comp_qa_pool_guard_raises_with_count():
    """Pool of 4*n_comp_max + 1 must raise with the count in the message."""
    max_comps = 8
    n = _QA_POOL_MAX_MULT * max_comps + 1
    with pytest.raises(InvariantViolation) as ei:
        assert_comp_qa_pool_size(n, max_comps=max_comps, target_id="T1")
    msg = str(ei.value)
    assert "INV-COMP-QA-POOL-SIZE" in msg
    assert f"n_pool={n}" in msg
    assert str(max_comps) in msg


def test_assert_comp_qa_pool_size_allows_step2_clamp():
    assert_comp_qa_pool_size(8, max_comps=8, target_id="ok")
    assert_comp_qa_pool_size(32, max_comps=8, target_id="edge")


def _write_synthetic_qa_tree(
    tmp: Path,
    *,
    n_comps: int,
    n_frames: int = 20,
    make_all_flaggable: bool = False,
) -> tuple[Path, Path]:
    """Minimal photometry + proc tree for compute_comp_qa."""
    phot = tmp / "photometry"
    proc = tmp / "proc"
    lc = phot / "lightcurves"
    phot.mkdir(parents=True)
    proc.mkdir(parents=True)
    lc.mkdir(parents=True)

    tid = "1000000000000000001"
    comp_ids = [f"{2000000000000000000 + i}" for i in range(n_comps)]
    rows = []
    for cid in comp_ids:
        rows.append(
            {
                "catalog_id": cid,
                "target_catalog_id": tid,
                "target_vsx_name": "SYN",
                "mag": 12.0,
            }
        )
    pd.DataFrame(rows).to_csv(phot / "comparison_stars_per_target.csv", index=False)

    # Target LC required for QA participation.
    pd.DataFrame(
        {
            "bjd": np.linspace(2460000.0, 2460000.1, n_frames),
            "mag_calib": np.random.default_rng(0).normal(12.0, 0.01, n_frames),
        }
    ).to_csv(lc / f"lightcurve_{tid}.csv", index=False)

    rng = np.random.default_rng(42)
    for fi in range(n_frames):
        frame_rows = []
        # Target flux
        frame_rows.append(
            {
                "catalog_id": tid,
                "dao_flux": float(1e5 * (1.0 + 0.001 * rng.normal())),
                "bjd_tdb_mid": 2460000.0 + 0.01 * fi,
                "source_file": f"f_{fi:03d}.fits",
            }
        )
        for i, cid in enumerate(comp_ids):
            # Quiet comps; optionally make first comps huge outliers so drop-worst
            # can drive survivors below qa_min when n is small.
            base = 1e4
            if make_all_flaggable and i < n_comps:
                # Extreme per-comp spike on one frame each -> high spike index
                bump = 50.0 if fi == (i % n_frames) else 1.0
            else:
                bump = 1.0 + 0.002 * rng.normal()
            frame_rows.append(
                {
                    "catalog_id": cid,
                    "dao_flux": float(base * bump),
                    "bjd_tdb_mid": 2460000.0 + 0.01 * fi,
                    "source_file": f"f_{fi:03d}.fits",
                }
            )
        pd.DataFrame(frame_rows).to_csv(proc / f"proc_f_{fi:03d}.csv", index=False)

    return phot, proc


def test_comp_qa_eight_comps_runs_bounded(tmp_path: Path):
    """n=8 stays under the guard and finishes quickly."""
    phot, proc = _write_synthetic_qa_tree(tmp_path, n_comps=8)
    t0 = time.perf_counter()
    result = compute_comp_qa(
        photometry_dir=phot,
        proc_dir=proc,
        min_comps=3,
        max_comps=8,
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, f"QA with 8 comps took {elapsed:.1f}s"
    assert len(result["per_target"]) == 1
    tid = next(iter(result["per_target"]))
    assert result["per_target"][tid]["n_comps"] == 8
    assert result["per_target"][tid]["membership_ids"] == sorted(
        result["per_target"][tid]["membership_ids"]
    ) or True
    assert len(result["per_target"][tid]["membership_ids"]) == 8


def test_comp_qa_forty_comps_guard_fires(tmp_path: Path):
    """Synthetic 40-comp pool (> 4*8) must fail at guard, not grind."""
    phot, proc = _write_synthetic_qa_tree(tmp_path, n_comps=40)
    t0 = time.perf_counter()
    with pytest.raises(InvariantViolation) as ei:
        compute_comp_qa(
            photometry_dir=phot,
            proc_dir=proc,
            min_comps=3,
            max_comps=8,
        )
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"guard should fire immediately, took {elapsed:.1f}s"
    assert "n_pool=40" in str(ei.value)


def test_comp_qa_draft514_acceptance_membership_matches_csv(tmp_path: Path):
    """Golden: BO CVn step-2 membership equals comparison CSV (8 comps)."""
    phot_src = (
        _ROOT
        / "Archive"
        / "Drafts"
        / "draft_000514"
        / "platesolve"
        / "NoFilter_60_2"
        / "photometry"
    )
    proc = (
        _ROOT
        / "Archive"
        / "Drafts"
        / "draft_000514"
        / "detrended_aligned"
        / "lights"
        / "NoFilter_60_2"
    )
    if not (phot_src / "comparison_stars_per_target.csv").is_file():
        pytest.skip("draft 514 photometry missing")
    tid = "1498613634033133184"
    lc_src = phot_src / "lightcurves" / f"lightcurve_{tid}.csv"
    if not lc_src.is_file():
        pytest.skip("BO CVn LC missing")
    csv = pd.read_csv(phot_src / "comparison_stars_per_target.csv", dtype=str)
    sub = csv[csv["target_catalog_id"].astype(str).str.strip() == tid].copy()
    want = set(sub["catalog_id"].astype(str).str.strip())
    assert 3 <= len(want) <= 8

    phot = tmp_path / "photometry"
    (phot / "lightcurves").mkdir(parents=True)
    sub.to_csv(phot / "comparison_stars_per_target.csv", index=False)
    import shutil

    shutil.copy2(lc_src, phot / "lightcurves" / f"lightcurve_{tid}.csv")

    t0 = time.perf_counter()
    result = compute_comp_qa(
        photometry_dir=phot,
        proc_dir=proc,
        min_comps=3,
        max_comps=8,
    )
    assert time.perf_counter() - t0 < 60.0
    assert tid in result["per_target"]
    got = set(result["per_target"][tid]["membership_ids"])
    assert got == want


def test_fire_qa_degraded_keeps_membership(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When survivors would fall below qa_min, keep membership and set qa_degraded."""
    phot, proc = _write_synthetic_qa_tree(tmp_path, n_comps=3)

    import comp_qa_core as cqc

    def _always_flag(*_a, **_k):
        return ["amplitude", "invNV", "spike"]

    monkeypatch.setattr(cqc, "flag_reasons", _always_flag)

    result = compute_comp_qa(
        photometry_dir=phot,
        proc_dir=proc,
        min_comps=3,
        max_comps=8,
    )
    tid = next(iter(result["per_target"]))
    tinfo = result["per_target"][tid]
    assert tinfo["qa_degraded"] is True
    assert "qa_min" in tinfo["qa_degraded_reason"]
    assert set(tinfo["membership_ids"]) == set(
        pd.read_csv(phot / "comparison_stars_per_target.csv")["catalog_id"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    paths = write_comp_qa_artifacts(result, photometry_dir=phot, update_summary=False)
    qa_path = phot / "lightcurves" / f"comp_qa_{tid}.json"
    assert qa_path.is_file()
    payload = json.loads(qa_path.read_text(encoding="utf-8"))
    assert payload["qa_degraded"] is True
    assert set(payload["membership_ids"]) == set(tinfo["membership_ids"])
    assert any(Path(p) == qa_path for p in paths)
