"""CQ-C: fix-once comp_qa magnitude locus — order independence + bounded diff."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from comp_qa_core import compute_comp_qa
from config import AppConfig
from tests.comp_qa_legacy_iterative import compute_comp_qa_iterative_locus
from trust_flag_core import comp_thresholds_from_config, trust_level, classify_warnings

_DRAFT = Path(__file__).resolve().parents[1] / "Archive" / "Drafts" / "draft_000366"
_PHOT = _DRAFT / "platesolve" / "NoFilter_60_2" / "photometry"
_PROC = _DRAFT / "detrended_aligned" / "lights" / "NoFilter_60_2"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (_PHOT / "comparison_stars_per_target.csv").is_file(),
        reason="draft_000366 photometry tree not present",
    ),
]

# Bounded-diff record (draft_000366, 2026-06-09): iterative vs fix-once locus.
_BOUNDED_FLAG_FLIPS = 1
_BOUNDED_N_CLEAN_CHANGES = 1
_BOUNDED_TRUST_CHANGES = 0


def _qa_kwargs() -> dict:
    cfg = AppConfig()
    return {
        "photometry_dir": _PHOT,
        "proc_dir": _PROC,
        "min_comps": int(cfg.phase01_comparison_n_comp_min),
        "max_comps": int(cfg.phase01_comparison_n_comp_max),
    }


def _canonical_payload(result: dict) -> bytes:
    rows = sorted(
        result["per_comp_rows"],
        key=lambda r: (r["target_catalog_id"], r["catalog_id"]),
    )
    targets = {
        tid: {
            "n_clean": v["n_clean"],
            "n_flagged": v["n_flagged"],
            "n_comps": v["n_comps"],
        }
        for tid, v in sorted(result["per_target"].items())
    }
    blob = json.dumps(
        {"rows": rows, "targets": targets, "stats": result.get("stats", {})},
        sort_keys=True,
        ensure_ascii=False,
    )
    return blob.encode("utf-8")


def _target_ids(result: dict) -> list[str]:
    return sorted(result["per_target"].keys())


def _trust_map(result: dict, *, cfg: AppConfig) -> dict[str, str]:
    th = comp_thresholds_from_config(cfg)
    out: dict[str, str] = {}
    for tid, tinfo in result["per_target"].items():
        hard, soft = classify_warnings(
            n_clean=int(tinfo["n_clean"]),
            check_scatter=float("nan"),
            lc_quality="good",
            thresholds=th,
        )
        out[tid] = trust_level(int(tinfo["n_clean"]), hard, soft, th)
    return out


def test_fix_once_order_independence_across_shuffles():
    """Fix-once locus: byte-identical QA across >=5 target order permutations."""
    kwargs = _qa_kwargs()
    base = compute_comp_qa(**kwargs)
    ref = hashlib.sha256(_canonical_payload(base)).hexdigest()
    ids = _target_ids(base)
    assert len(ids) >= 5

    rng = random.Random(366)
    for i in range(8):
        order = ids.copy()
        rng.shuffle(order)
        out = compute_comp_qa(**kwargs, _target_processing_order=order)
        digest = hashlib.sha256(_canonical_payload(out)).hexdigest()
        assert digest == ref, f"order shuffle {i} diverged from reference"


def test_iterative_locus_was_order_coupled():
    """Record: legacy iterative locus differed under permuted target order."""
    kwargs = _qa_kwargs()
    ids = _target_ids(compute_comp_qa(**kwargs))
    rng = random.Random(366)
    digests: set[str] = set()
    for _ in range(6):
        order = ids.copy()
        rng.shuffle(order)
        out = compute_comp_qa_iterative_locus(**kwargs, _target_processing_order=order)
        digests.add(hashlib.sha256(_canonical_payload(out)).hexdigest())
    assert len(digests) > 1, "expected legacy behavior to be order-coupled"


def test_bounded_diff_old_vs_new_on_draft_366():
    """CQ-C re-baseline: small bounded flag / n_clean / trust churn only."""
    kwargs = _qa_kwargs()
    cfg = AppConfig()
    old = compute_comp_qa_iterative_locus(**kwargs)
    new = compute_comp_qa(**kwargs)

    old_by = {(r["target_catalog_id"], r["catalog_id"]): r for r in old["per_comp_rows"]}
    new_by = {(r["target_catalog_id"], r["catalog_id"]): r for r in new["per_comp_rows"]}
    assert old_by.keys() == new_by.keys()

    flag_flips = sum(
        1 for k in old_by if bool(old_by[k]["FLAG"]) != bool(new_by[k]["FLAG"])
    )
    n_clean_changes = 0
    n_clean_deltas: list[int] = []
    for tid in old["per_target"]:
        o = int(old["per_target"][tid]["n_clean"])
        n = int(new["per_target"][tid]["n_clean"])
        if o != n:
            n_clean_changes += 1
            n_clean_deltas.append(n - o)

    old_trust = _trust_map(old, cfg=cfg)
    new_trust = _trust_map(new, cfg=cfg)
    trust_flips = sum(1 for tid in old_trust if old_trust[tid] != new_trust[tid])

    assert flag_flips == _BOUNDED_FLAG_FLIPS
    assert n_clean_changes == _BOUNDED_N_CLEAN_CHANGES
    assert trust_flips == _BOUNDED_TRUST_CHANGES
    assert max(abs(d) for d in n_clean_deltas) <= 2 if n_clean_deltas else True
