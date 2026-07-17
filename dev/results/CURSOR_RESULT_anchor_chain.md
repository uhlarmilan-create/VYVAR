CURSOR RESULT -- 2026-07-13 (ANCHOR-CHAIN-ACCEPT)

What I did
Made the draft_424 anchor baseline chain explicit, cut the missing intermediate baseline
at origin/main (b5364e6) without c4/floor, validated the c4-only delta exactly per epoch,
then accepted the sigma-floor snapshot as the anchor (wiring + docs + comparator hardening).
No push performed.

## Inventory + 07-10 verdict (Part A)

Draft_424 snapshots present under Archive/Drafts:

| Snapshot | git_hash (from pipeline_meta) | core_sha (prefix) | extended_sha (prefix) |
|----------|-------------------------------|-------------------|-----------------------|
| draft_000424_snapshot_20260708_full | 750c856 | 92939fab | 76642318 |
| draft_000424_snapshot_20260708_hybrid_deprecated | (none) | ba8e78af | 1faaa1dc |
| draft_000424_snapshot_sigma_floor_20260713 | 8fb21b3 | bf3743a1 | dec5c637 |

Inventory artifact: tmp/anchor_chain/inventory_424.json.

Missing 2026-07-10 reference snapshot:

- session_baseline_check.py references: draft_000424_snapshot_sigma_floor_20260713 (core bf3743a1..., ext dec5c637...).
- validation ledger VL-ANCHOR-424 references the same accepted snapshot (git 8fb21b3).
- No draft_424 snapshot dated 2026-07-10 exists on disk; prior F-BINGAIN result files were deleted from the repo
  (git status showed D CURSOR_RESULT_bingain_* at session start). Verdict: process gap (snapshot was never physically cut
  or was not retained).

## Intermediate baseline (Part B)

Cut on origin/main commit b5364e6 (detached worktree) with two independent full pipeline runs:

- Snapshot: Archive/Drafts/draft_000424_snapshot_intermediate_b5364e6_20260713
- core SHA: 373e8235... (n=357)
- extended SHA: 0243f719... (n=535)
- Repro: byte-identical run_a/run_b = True

## Stepwise validation (Part C)

### C.1 0708 -> intermediate (approved changes; document)

Per-tertile err ratios (old 07-08 anchor -> intermediate):

| Tertile | err_old med | err_int med | ratio |
|---------|-------------|-------------|-------|
| faint | 0.0150195 | 0.0225710 | 1.503 |
| mid | 0.0376410 | 0.0634100 | 1.685 |
| bright | 0.0857560 | 0.1436660 | 1.675 |

Overall (23542 epochs): ratio median 1.619, p25 1.428, p75 1.836.

Artifact: tmp/anchor_chain/c1_0708_to_intermediate.json.

This matches the already-measured mechanism: the dominant increase vs 07-08 comes from the
approved F-BINGAIN-1 empirical photon term (bingain/howell median ~1.634) plus the SEM unit fix
(ensemble carrier -8%).

### C.2 intermediate -> sigma-floor snapshot (unapproved delta; EXACT)

Validated that the accepted snapshot err is exactly the intermediate err with c4 applied to the
ensemble SEM and floor=0 on eq1:

Tolerance: abs(err_pred - err_snapshot) <= 2e-6 (rel-flux).

Result:

- epochs compared: 23542
- median abs diff: 2.93e-7
- max abs diff: 9.97e-7
- outliers: 0
- n_comps histogram: {3: 2116, 4: 1226, 5: 552, 6: 798, 7: 4252, 8: 14598}

Artifact: tmp/anchor_chain/c2_exact_c4_validation.json.

## Acceptance + wiring (Part D)

Accepted anchor snapshot (no change to contents; now accepted by explicit chain + exact validation):

- Archive/Drafts/draft_000424_snapshot_sigma_floor_20260713
- git_hash 8fb21b3
- core bf3743a1... (n=357)
- extended dec5c637... (n=535)

Wiring updated:

- validation/VYVAR_VALIDATION_LEDGER.json: VL-ANCHOR-424 notes/verification now include intermediate baseline + exact validation.
- docs/VYVAR_SIGMA_FLOOR_SPEC.md: anchor chain table + exact validation stats.
- CURSOR_RESULT_sigma_floor.md attribution corrected (no longer \"c4-only\" vs 07-08).

Pending local verification step required by task: scripts/session_baseline_check.py --full OVERALL PASS on the accepted anchor.

**Verified (2026-07-13):** `session_baseline_check.py --full` **OVERALL PASS** (pytest 796 passed, 15 skipped; photometry SHA core/extended match accepted snapshot).

## Comparator hardening (Part E)

Implemented designed-err acceptance plumbing in tests.photometry_sha.compare_photometry_science_meaningful:

- If err_designed=True, caller must supply err_accept mode:
  - mode=envelope: bounded per-tertile median ratios must lie within [min_ratio, max_ratio]
  - mode=exact_pred: per-epoch predicted err JSON must match within abs_tol

Tests added:

- tests/test_photometry_sha_err_designed.py: synthetic 1.6x inflation vs envelope [0.96, 1.05] -> FAIL
- tests/test_photometry_sha_err_designed.py: exact predictor path -> PASS

## Patch confirmation (Part F)

CURSOR_RESULT_anchor_err_verify.md updated with:

- per-frame n_comps histogram (23542 epochs)
- per-target median n_comps histogram (178 LCs)
- photon ratio stats (median 1.634, p25 1.475, p75 1.806)

## Errors

None (c4 exact validation PASS; no outliers).

## Files changed

- tests/photometry_sha.py (153dc96)
- tests/test_photometry_sha_err_designed.py (153dc96)
- CURSOR_RESULT_sigma_floor.md, CURSOR_RESULT_anchor_err_verify.md (9a6cf07)
- CURSOR_RESULT_anchor_chain.md (9a6cf07)
- docs/VYVAR_SIGMA_FLOOR_SPEC.md, VYVAR_STATE.md, VYVAR_ROADMAP.md, VYVAR_JOURNAL.md (9a6cf07)
- validation/VYVAR_VALIDATION_LEDGER.json (9a6cf07)

Artifacts (untracked, tmp/):

- tmp/anchor_chain/inventory_424.json
- tmp/anchor_chain/c1_0708_to_intermediate.json
- tmp/anchor_chain/c2_exact_c4_validation.json

## pytest / ruff

- ruff check .: PASS (touched files clean)
- pytest tests/: **796 passed**, 15 skipped (+2 new comparator tests)
- session_baseline_check.py --full: **OVERALL PASS**

## READY-FOR-PUSH

Local chain ready for Milan review; **no push performed**.

