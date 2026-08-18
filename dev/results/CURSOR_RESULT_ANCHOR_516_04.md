CURSOR RESULT - 2026-08-18

What I did
ANCHOR-516-04 on tip `4a65675`: clean 516 rebuild wholly generated on
the fixed tip (MASTERSTAR zones re-annotated at saturate_limit_fraction
0.80, comparison pool regenerated, proc sat-limit columns rewritten),
then freeze, --full + P1 recut, retirement list, docs.

Premise (Rule 0.1): compare a 516 product generated on the fixed tip
against `be6191e0`, which reused a 0.85-annotated
`masterstars_full_match.csv`. The 435 `--full` golden is retired by
design (ROT out-of-scope policy), not repaired.

## Output / findings

### E1-E5

| ID | State | Evidence |
|----|-------|----------|
| E1 | PASS | Saturated 23 -> 24; star `1497853802778923392` linear->saturated and left `comparison_stars.csv`. |
| E2 | PASS | ct_n_comp 2346 -> 2345; ct_c1 unchanged **-0.373** / **-0.373**. |
| E3 | PASS | MAG 48/48 vs 515/`de6f7c8` (STOP gate did not fire). AAVSO MAG 134/134. |
| E4 | PASS | ERR empirical; sem/scint/sys max abs vs 515 = 0; photon-only ERR diffs (max 6.589 mmag). |
| E5 | PASS | Fresh sat-limit unique 52428.0; leftover 55704.75 = 0 (proc rewrite also cleared `linearity_limit_adu`). |

### Part A - clean rebuild

Tip: `4a65675eba04`. Harness `tmp/anchor_516_04_clean_rebuild.py`.
Resume was idempotent after a first pass that did the 23->24 flip;
provenance `n_stale=0` (all product mtimes post run-start).

| Phase | Runtime |
|-------|---------|
| Phase 0 | 0.37 s |
| Phase 1 | 4298 s |
| Phase 2A | 517 s |
| **Total** | **5047 s** |

Final product SHA (supersedes de6f7c8 -> be6191e0):

- core `477dc8cfc292ed63910ecca6ea1dacfda279fee2850422229739a5cf7db90956` n=97
- extended `f71e07226893a6b07e24999927bad0da8c16e6407656fc97ee02e0d57494be5d` n=145

Raw: `dev/results/context/session_20260818_anchor_516_04/part_a_clean_rebuild.json`

### Part B - exports + acceptance

Gating: 218 active, **48 LCs**, 48 comp_qa. Skip: 48 ok, 45 zone_noise,
1 `per_frame_saturation` (CV CVn `1497007144465726080`), 121
`vsx_type_out_of_scope`, 3 `below_target_depth`.

- BO median err **8.532 mmag**
- 01B MAD identical to 515: BO **7.1506**, FW **8.2010** mmag (check `1497613731286514432`)
- Per-term median (BO): photon 5.511, sem 4.914, scint 3.684, sys 0.0 mmag
- AAVSO MAGERR 82/134 rows change at 3-decimal; median MAGERR still 0.009
- ERR_MODEL `gain=g_pt=0.6371`; depth 0.470
- SUBMIT-01 PASS (OBSCODE=UMIA, VYVAR/1.0, CV, KNAME/NOTES in data rows,
  BJD, 134 epochs). C5: Milan submits; nothing was previously uploaded.
  New files supersede 515-era on-disk exports.

Raw: `.../part_b.json`

### Part C - freeze + recut + fire-proof

Frozen snapshot: `Archive/Drafts/draft_000516_snapshot_cleanrebuild_20260818`
(SHA match to live 516). `--full` copies snapshot inputs into
`tmp/session_baseline/<ts>/` (never live 516; never mutates freeze).

P1 mini: `draft_000516_p1mini` (16-frame stride; `cal_diag.json` on draft
root, `platesolve/`, and `platesolve/NoFilter_60_2/` for INV-CAL-01).
Manifest `f788270d...`. Headless lock 651.6 s: core `41604319` n=97,
ext `65e64210` n=145, 48 LCs.

Windows mkdir crash on first `--full` (colon timestamp
`20260818T15:35:17Z`, WinError 123). Fixed `_full_work_stamp` to
`%Y%m%dT%H%M%SZ`. Work dir `tmp/session_baseline/20260818T161323Z`.

| Gate | Result | Runtime |
|------|--------|---------|
| `--fast` | OVERALL PASS; pytest 1460 passed, 21 skipped | 1664 s |
| `--full` | OVERALL PASS; pytest 1461 passed, 21 skipped; pipeline 6135 s; SHA 477dc8cf/f71e0722; plan-regen 873; science-compare 48/0 | 8334 s wall |
| P1 golden | headless SHA + UI identity + census + physics PASS; `test_mini_present_or_buildable` PASS after skipping rewriteable `alignment_report.csv` | 651 s lock + 1333 s prior pytest |

Stale-golden tests pass, not xfail. First `--fast` FAIL was
order-dependent caplog on `test_sigma_sys_explicit_zero_for_equipment_1`
(`_LOGGED_UNFLOORED` once-per-process); test now discards the key.

### Part D - retirement + docs

DELETE-OK (Milan deletes manually; Cursor did not delete):
`dev/results/context/session_20260818_anchor_516_04/DELETE_OK.md`

Keep: 516, `draft_000516_snapshot_cleanrebuild_20260818`, `draft_000516_p1mini`.

Delete-OK: 435, 435_p1mini, 435_snapshot_skysurface, 436, 437, 509, 513,
514, 515 (no submission hold).

Live gates (`session_baseline_check.py`, P1 tests, `build_p1_golden_mini.py`,
seed, `test_cal_stage_gate.py`) point at 516. `5bccd85a` remains only as
superseded-with-pointer in the ledger. Historical `wide_err_*` /
`closure_step*` / `audit_stage3_*` still name 435 as a measurement target;
they are not live `--fast`/`--full` gates.

Docs: STATE, ROADMAP (P1-RECUT, A-1-435-RECUT, FULL-ANCHOR-RECUT,
ANCHOR-GATE-SEED, ANCHOR-CLEAN-BUILD closed), DECISIONS (canonical mode,
INV-ERR-MODE-01, one-authority including MASTERSTAR writer,
anchor-from-clean-rebuild), JOURNAL, RUNBOOK, VALIDATION, PARAMS already
0.80 at `4a65675`, CHANGELOG, ledger.

Push: not authorized (single close authorization expected for
96aa0d6..anchor-recut).

## Errors (if any)

- First `--full`: `OSError: [WinError 123]` colon in work-root stamp.
  Fixed; re-run OVERALL PASS.
- First `--fast`: pytest FAIL `test_sigma_sys_explicit_zero_for_equipment_1`
  (order-dependent once-log). Fixed; re-run OVERALL PASS.
- P1 `test_mini_present_or_buildable` SHA drift on `alignment_report.csv`
  (pipeline rewrites it). Test skips that input; re-run PASS.

## Files changed

Code/gates: `dev/scripts/session_baseline_check.py`,
`dev/tests/test_session_baseline_check.py`,
`dev/tests/test_invariants_p1_golden.py`,
`dev/tests/test_invariants_p1_seed.py`,
`dev/tests/test_cal_stage_gate.py`,
`dev/tests/test_wide_err_03_gain.py`,
`dev/tools/build_p1_golden_mini.py`,
`dev/validation/VYVAR_VALIDATION_LEDGER.json`

Docs/results: STATE, ROADMAP, DECISIONS, JOURNAL, RUNBOOK, VALIDATION,
CHANGELOG, this file, `CURSOR_TASK_ANCHOR_516_04.md`,
`dev/results/context/session_20260818_anchor_516_04/`

Untracked Archive freeze/mini are gitignored (not committed). Leftover
2026-08-17 tracked diffs excluded from commits.
