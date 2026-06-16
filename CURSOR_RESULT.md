CURSOR RESULT — 2026-06-16 (Phase-1b: per-target comp_rms gate authoritative for N_good)

What I did
Closed known-issue (b): made the per-target `phase01_comparison_max_comp_rms` (=0.1) gate authoritative
for N_good. Verified the two reported loci against the current tree + ground-truth matrix `164157`,
implemented the minimal fix, added unit tests, re-ran the validation matrix, updated docs. Separate
commit on top of Phase-1 (`1c80219`).

## Output / findings

### Verify-before-fix (loci re-confirmed)
- `comp_selection_per_target.py` `_detrend_and_compute_comp_rms_map`: RMS fallback steps
  `[max_comp_rms, 0.08, 0.15]` admitted comps above the gate (the `0.15` step). Confirmed SS Cam reached
  `default` with its 0.134 comp only via this relaxation.
- `photometry_core.py` auto-routing: treated `len(result) >= 1` as success, not gate-passer count.

### The fix
- **Locus 1:** RMS fallback now selects only among gate-passers (`comp_rms <= max_comp_rms`); the
  above-gate `0.15` relaxation is removed. Never exceeds the gate. Phase-1 thin-set keeping for
  gate-passers is untouched.
- **Locus 2:** added `_count_gate_passing_comps(result, per_target_rms_map, max_comp_rms, id_col)`.
  Auto-routing routes on the gate-passer count; zero gate-passers -> `sparse_fallback`.

### Matrix re-run (`tmp/comp_degradation_validate/matrix_20260616_185831.json` vs baseline `164157`)
| Target | OLD (path/trust/N) | NEW (path/trust/N) | result |
|--------|--------------------|--------------------|--------|
| 409 V0612 `...526912` | default/RED/1 @0.034 | default/RED/1 @0.034 | UNCHANGED (gate-passer thin set) |
| 409 SS Cam `...992064` | default/RED/1 @**0.134** | **sparse_fallback/YELLOW/3** @~0.35 | FLIP; 0.134 no longer a good default comp |
| 410 BO CVn `...133184` | default/GREEN/4 @0.0086 | default/GREEN/4 @0.0086 | UNCHANGED (regression guard) |
| 411 V0842 Her `...714240` | default/YELLOW/8 @0.0124 | default/YELLOW/8 @0.0124 | UNCHANGED (regression guard) |
| 409 V0611 / degenerate | sparse/YELLOW/8 | sparse/YELLOW/8 | UNCHANGED |

Only SS Cam changed. All regression guards hold.

### SS Cam trust: expected RED, got YELLOW (reconciliation — NOT a defect, NOT forced)
The (b) fix correctly flips SS Cam `default -> sparse_fallback` (its single default comp `comp_rms 0.134`
fails the gate -> 0 gate-passers). The predicted "stays RED (hard check 0.053)" assumed a
path-independent check-star scatter. Check-star scatter is ensemble-dependent: against the new sparse
ensemble it is **0.043 < `_CHECK_HARD_LO` = 0.05** (trust_flag_core.py:34), so it is a soft warning ->
YELLOW. This matches the grounded trust model (spec section 5: `sparse_fallback` lands at YELLOW at
most). SS Cam's sparse comps (~0.35 mag field-wide) are only catchable by the **Phase-2 sparse-comp
sanity ceiling**, which is explicitly out of scope here. RED was not forced (no threshold re-tuning, no
Phase-2 / check-star-selector changes). Surfaced the conflict; accepted the grounded YELLOW outcome.

### Tests / lint
- New: `tests/test_comp_rms_gate_authoritative.py` (8 cases).
- **pytest:** 330 passed, 15 skipped.
- **ruff:** all checks passed.

## Errors (if any)
None. One DoD deviation (SS Cam YELLOW vs predicted RED) explained above; root-caused to the Phase-2
sanity ceiling boundary, not the (b) fix.

## Files changed
- `comp_selection_per_target.py` (RMS fallback no above-gate relaxation)
- `photometry_core.py` (`_count_gate_passing_comps` + gate-passer routing)
- `tests/test_comp_rms_gate_authoritative.py` (new)
- `docs/VYVAR_JOURNAL.md`, `docs/VYVAR_STATE.md`, `docs/VYVAR_ROADMAP.md`
- `CURSOR_RESULT.md`

**HEAD:** this commit `fix(comp): per-target comp_rms gate authoritative for N_good`
(separate commit on top of Phase-1 `1c80219`; pushed to origin/main)
**pytest:** 330 passed, 15 skipped
**ruff:** all checks passed
