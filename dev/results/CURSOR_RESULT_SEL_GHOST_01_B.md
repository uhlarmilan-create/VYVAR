CURSOR RESULT - 2026-08-25 (SEL-GHOST-01 Part B, B-STOP-1)

What I did
B1 INV-MATCH-IDENTITY-01 and B2 provenance are on `main` (ahead of
origin, not pushed). B-STOP-1 sandbox MASTERSTAR rerun of 520 `g_60_4`
and 516 `NoFilter_60_2` after the skip-solve WCS hotfix. Live 520/516
bytes unchanged. B3-B5 not started: waiting on Milan D1-D3.

HEAD `01f6f77` = B1 `d8c18a7` + B2 `e2a0a84` + hotfix `01f6f77`.
Base of the series: `c592ecf` (Part A docs) on `origin/main` `b1f5b8c`.
Session: `dev/results/session_20260825_sel_ghost_01_b/` plus Rule 0.2
copy `dev/results/context/session_20260825_sel_ghost_01_b/`.
Push: NO.

## Premise (Rule 0.1)

**What is compared:** sandbox MASTERSTAR tables produced by B1+B2
against the Part B predictions P-B1..P-B5, and against live draft 520
`g_60_4` / 516 `NoFilter_60_2` SHA guards. The 520 prediction "gate-out
~61" assumed a re-solve from pixels onto a healthy match-time WCS.
That re-solve failed (VYVAR triangle match); skip-solve catalogs
against the live FITS header, which is the destroyed post-optimizer
WCS (Part A F3). Numbers below are that skip-solve, not a 61-pair
honest rematch.

**How they differ:** Part A live table reported 347/692 catalog IDs
into the optimizer after name-export rehydration. B1 must make
optimizer entry equal gate-out, must not restore the 8 ghost IDs by
export, and must leave 516 quiet (fail=0, widen not fired).

## Gates

| Gate | Result | Evidence |
|------|--------|----------|
| G1 live 520 CSV SHA before == after | PASS | `5ce9b07fe0490103b2e16f6fbe3b18ffc7cd987fbee8a334722cc2fd46c6a683` |
| G2 live 516 CSV SHA | PASS | `bfa24039778f437b2bf7ed37056b6b507e068d52d2c7b4a222a73002125b250a` |
| G3 live 516 ePSF SHA | PASS | `172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20` |
| G4 `--fast` | PASS | 1536 passed, 32 skipped, OVERALL PASS at HEAD `01f6f77` |
| G5 no live draft writes | PASS | `draft_id=None`; sandbox under session dir |

## Commits (B1+B2+hotfix)

| Hash | What |
|------|------|
| `c592ecf` | docs: SEL-GHOST-01 A measured |
| `d8c18a7` | B1 INV-MATCH-IDENTITY-01: gate clears full identity; export does not copy `name` onto empty `catalog_id`; optimizer gates existing pairs on DAO FWHM; born-owned lock vs Gaia xy |
| `e2a0a84` | B2: INFO per 0.95 widen iter; `pipeline_meta` match_sep / identity-gate / optimizer_refit stamps; `time_base` in `_LC_OVERVIEW_COLS` (P-B8 display) |
| `01f6f77` | skip-solve must not shadow module `WCS` (`UnboundLocalError` on post-merge gate) |

B1a Gaia-derived columns cleared on fail (from `_assign_catalog_at_threshold`
and later copies): `name`, `mag`, `b_v`, `catalog`, `catalog_id`,
`gaia_nss`/`qso`/`gal`, `bp_rp`, `phot_g_mean_mag`, `catalog_mag`,
`match_sep_arcsec`. Stamps: `vy_identity_gate`, `gaia_dao_resid_px`.

B1b export callers: `pipeline.py` ~5149, ~5940, ~6656;
`astrometry_optimizer.py` ~1212. Reverse rename remains
`_proc_rename_det_names_to_catalog_id` (catalog_id -> name).

## 520 sandbox (skip-solve, post-hotfix)

Infolog `session_20260825_sel_ghost_01_b/infolog_20260825_145143.txt`.
DAO after SNR filter: **685/712**. First identity gate:
`ok=0 warn=7 fail=339` (FWHM=1.25 px). Post-merge and optimizer-entry
gates: `ok=0 warn=7 fail=0` (hotfix: post-merge **ran**; pre-hotfix
infolog `144555` skipped with `UnboundLocalError: WCS`).
Optimizer `matched_nonempty=7/685`. Skipped: need >=50 pairs, got 7.
No `optimizer_refit` stamp. `match_sep_effective=96.0` (0.95 loop to
cap; requested 10, floor would have been 12). Refine iters=0.

Final table n=791 (685 detections + 106 catalog-derived membership).
`vy_identity_gate`: fail=339, warn=7, empty=445.
Nonempty `catalog_id`=113 (7 gated warn + 105 un-gated DETECTED_P1
injects + 1 EDGE).

### Predictions

| ID | Verdict | Evidence |
|----|---------|----------|
| P-B1 | **TRUE** on the identity contract (7==7, not 347). Gate-out **7 not 61** because skip-solve used the poisoned header WCS. | log `matched_nonempty=7/685`; `identity_gate.n_matched_out=7` |
| P-B2 | **FALSE** (1/8). 7/8 ghost IDs absent. `1111922300852743808` remains: `DETECTED_P1` / `locked` / `vy_identity_gate` empty / d_px=0 (sits on Gaia xy). Post-gate membership/lock inject, **not** name-export rehydration. | `bstop1_measure.json` P-B2 |
| P-B3 | **N/A (skipped)**. No Grip. B4 not wired. The 81/84 px Grip-on-347 path did not run. | optimizer skipped, 7 pairs |
| P-B4 | **TRUE as written** (11/11 ok/warn vs final WCS) and **misleading**. 10/11 G<12 IDs are post-gate rows at Gaia xy (d_px=0, gate empty). Only `1111749157833870208` is a gated DAO match (`warn`, 3.37 px). Original DAO detections of the other G<12 stars are `DAO_ONLY` / `fail` (`DET_*`). | P-B4 rows in measure JSON |
| Honest vs reported (Part A 2.5, DAO n=685) | honest **7/685 = 0.010**; reported nonempty cid on DAO-origin rows **112/685 = 0.164**; full table 113/791 = 0.143 | Part A was 0.088 vs 0.522 |

P-B1 would read "61" only after a successful pixel re-solve onto
pre-optimizer WCS. That attempt failed (`retry_520_sandbox.py`).
Do not treat gate-out=7 as the B1e no-op; it is WCS quality.

## 516 sandbox (skip-solve, post-hotfix)

DAO 2643. Match rate 95.46% at first pass (already above 0.95).
`match_sep_requested=10`, **`match_sep_effective=12.0`**,
`wcs_gaia_pixel_refine_iters=0` -> 0.95 widen **did not fire**
(12" floor only; 12"/9.77"/px ~ 1.23 px). Identity gate fail=0
(quiet; INFO line omitted when warn=fail=0, `wcs_invertibility.py`).
`n_matched_out=2523`. Optimizer ran: `rms_lin=0.659`, `rms_sip=0.846`,
`n_pairs=2618`. Post-merge gate ran (passes=2; ok accumulated 5046 =
2523 x 2).

| ID | Verdict | Evidence |
|----|---------|----------|
| P-B5 fail count quiet | **TRUE** | fail=0 |
| P-B5 widen not fired | **TRUE** | effective=12.0" |
| P-B5 catalog_id set byte-identical | **FALSE** | sandbox 3571 vs live 3584; only_sandbox 1; only_live 14 |
| P-B5 DETECTED-only cid set | **FALSE** | 3501 vs 3509; only_sandbox 1; only_live 9 |
| B1e `n_lock_geometry_reject==0` | **FALSE** | **10** (pass2 born-owned is not a guaranteed no-op on 516) |

516 catalog_id delta is small and likely the B1e geometry rejects plus
membership-expansion jitter under skip-solve. Not a widen-loop effect.

## B1/B2 behaviour that held

- Optimizer entry nonempty cid == gate-out (P-B1 contract).
- Export rehydration closed: stripped rows are `DET_%04d` with empty
  `catalog_id`; the 7/8 missing ghosts are not restored from `name`.
- Widen loop logged/stamped: 520 effective 96"; 516 effective 12".
- `pipeline_meta.json` carries match_sep, identity-gate thresholds,
  optimizer_refit (516 only).
- Post-merge gate works on skip-solve after `01f6f77`.

## Open after B-STOP-1 (do not wire)

1. Catalog-derived membership / leftover lock can re-attach a Gaia ID
   at Gaia xy after the gate (`vy_identity_gate` empty). That is the
   remaining P-B2 path. B5 (comp candidacy) is the intended filter;
   it is waiting on D3. Label `DETECTED_P1` on a d_px=0 inject is a
   B5 input-quality issue.
2. 520 skip-solve cannot test Grip-on-honest-pairs (P-B3) or a 61-pair
   gate-out. A healthy WCS re-solve is a different experiment.
3. B3-B5 wait on D1-D3. No recut. No push.

## Decisions needed (unchanged; STOP)

D1  B3: remove the widening loops (architect default) or cap them.
D2  B4: rms_sip > max(3 x FWHM, 3 px) reject; n_pairs >= 10.
D3  B5: SNR floor 10; residual max(3 FWHM, 2 solve rms).
    Do not change `phase01_comparison_max_comp_rms`.

Architect retraction (still for DECISIONS at Part B close):
"H-MATCH-WIDEN (2026-08-25) named WCS-refine rematch (H2) as the path
that returned stripped IDs to the table. Measured: refine was rejected
3x and never rematched on 520 g_60_4. The return path is name-based
rehydration on CSV export (F3). H2 is retracted as cause; the code path
remains real and is closed by B1 as defence, not as the fix."

## Errors

520 pixel re-solve failed (expected for this destroyed WCS). Skip-solve
used instead. First skip-solve post-merge gate hit `UnboundLocalError`
(fixed `01f6f77`, sandbox re-run). tmp/ 4-tuple lock unpacks for
`--fast` are gitignored local fixes, not committed.

## Files changed

Code/docs already committed: B1/B2/hotfix as above.
This STOP: `dev/results/CURSOR_RESULT_SEL_GHOST_01_B.md`;
`docs/VYVAR_STATE.md` / `docs/VYVAR_JOURNAL.md` one-liners;
untracked session dir + Rule 0.2 context copy.
No B3-B5. No anchor recut. No push.

## Docs impact

STATE / JOURNAL one-liners for B-STOP-1. DECISIONS (F1-F5 + retraction
+ D1-D3) and ROADMAP (close REG-520-S2) wait for B-STOP-2 after D1-D3.
INVARIANTS already has INV-MATCH-IDENTITY-01 from B1.

## Recurrence

existing `dev/tests/test_inv_match_identity_01.py` (B1). Sandbox
prediction misses (P-B2 leftover inject, B1e=10 on 516) are new
observations, not yet tests.

## `--fast`

`python dev/scripts/session_baseline_check.py --fast` at HEAD `01f6f77`:
**OVERALL PASS** (1536 passed, 32 skipped). git-untracked WARN is
pre-existing session dirt plus this STOP. db-quick-check WARN waived.
git-origin-main WARN: local `main` is ahead of `origin/main` `b1f5b8c`
(expected; Push: NO).
