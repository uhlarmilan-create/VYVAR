# CURSOR RESULT - IMPL-05 Item D (Comp QA runaway)

Date: 2026-08-16
Baseline: b2ae3b7 / stamp ac51e84 (Item A)
Tip: 3927afd
Push: NO

## Status (B/C) at addendum start

- Tip `ac51e84`, branch `main`, clean index; 32 untracked (unrelated).
- Items B and C: **not started** (blocked on missing IMPL-04 per-star flux
  ladders; idle since Item A stamp `2026-08-16 08:26:53`).
- Nothing re-photometering. Streamlit `app.py` only.

## Diagnosis (verified)

Architect read confirmed at tip.

Evidence on draft 514 photometry:

| Artifact | Count / size |
|---|---|
| `comp_qa_*.json` before this fix | **0** (QA never finished a hung run) |
| `comp_quality_*.json` | 49 (Phase-2A stability sidecars, not Sokolovsky QA) |
| Keys in old quality files | median **1299**, max 1299 (39/49 files) |
| Current `comparison_stars_per_target.csv` | 734 pairs, 3-8/target (post COMP-ASSIGN-02) |

Hung Phase 2A stopped at "Comp QA (Sokolovsky locus)..." after all LCs were
written. Zero `comp_qa_*.json` = the drop-worst loop never completed. Pool size
that QA reads is the CSV groupby; when the CSV carried the COMP-ADMIT uncapped
pool (~1292/target), the loop is O(N^3 F) ~ days. **Diagnosis stands.**

(Note: large `comp_quality_*.json` maps are from Phase-2A
`check_comparison_stability` over the same oversized CSV era, not from Comp QA
writes.)

## Fix

1. Docstring + behaviour: QA already reads step-2 CSV; now **guards** if
   `n_pool > 4 * n_comp_max` (limit=32 at defaults). Chosen 4x: allows thin /
   sparse sets, refuses pool-era hundreds. Raises `InvariantViolation`
   `INV-COMP-QA-POOL-SIZE` with `n_pool=` in the message.
2. Replaced silent `surv_final = list(td["pool"])` restore-after-drop with
   `qa_degraded=true` + reason; **membership unchanged** (step-2 set).
3. Artifacts: `comp_qa_*.json` carries `qa_degraded` / `membership_ids`; merges
   metadata into existing `comp_quality_*.json`.

## Remeasure (QA-only over existing LCs)

`compute_comp_qa` + write on draft 514: **275 s**, 49 targets with LCs,
`comp_qa_*.json` for all 49. BO/FW: n=8, 1 flagged each, not degraded. Two
other targets marked `qa_degraded` (flagging left 2 < qa_min=3).

## Tests

- `dev/tests/test_comp_qa_pool_guard.py` (6): guard fire proof, 40-comp raise,
  8-comp bounded, BO membership golden, qa_degraded membership untouched.
- `--fast` (see tip)

## Files

- `src_py/comp_qa_core.py`
- `src_py/photometry_core.py` (meta keys for qa_degraded)
- `dev/tests/test_comp_qa_pool_guard.py`
- this result
