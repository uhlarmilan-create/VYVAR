# CURSOR RESULT - GOLDEN ASSET RE-CUT (2026-07-28)

Session data: `dev/results/context/session_20260728_golden_recut/`

---

## Part 0 - Premise confirmed

Four stale P1 mini light curves map to the four identity-gate dropped targets (matched Gaia LC IDs):

| VSX target (dropped) | LC file on stale mini |
|----------------------|------------------------|
| 1499883638682689920 | lightcurve_1499883638682689408.csv |
| 1500410236033012352 | lightcurve_1500410613990135296.csv |
| 1498513166158147968 | lightcurve_1498513269237363456.csv |
| 1499064433800590592 | lightcurve_1499064399440851968.csv |

333 - 325 = 8 = 4 x (lightcurve + comp_quality). Root cause: `headless_mini` skipped pipeline
when mini matched stale VL-P1-GOLD SHA; not UI/headless parity (F-431 CLOSED).

---

## Part 1 - Fixture fix

- `headless_mini`: no silent PASS; `VYVAR_P1_REUSE_FROZEN=1` -> visible `pytest.skip`.
- Default under `VYVAR_INVARIANTS_P1=1`: always execute (`VYVAR_P1_FORCE=1` also forces).
- `session_baseline_check.py --full` sets `VYVAR_P1_FORCE=1`.
- Short-circuit audit: **only** `test_invariants_p1_golden.py` (no other SHA-reuse skips found).

**Run time:** part of P1 pytest (~815 s per full golden run).

---

## Part 2 - VL-P1-GOLD re-cut

Two independent wipe+rebuild+headless runs (run A 443.2 s, run B 449.1 s) **byte-identical**.

| Field | New | Superseded |
|-------|-----|------------|
| core_sha | `e7976de18e4197e85e0120dcadf6bdae5ac0be73238be92f83c7cd87fa0fedee` | `074ae881...` |
| core_n | **325** | 333 |
| extended_sha | `d0e8f64b55806400aa7e305d97c79c7d4d03cb99b96e48ac44c81d7e087b3deb` | `66285d3f...` |
| extended_n | **485** | 497 |
| n_summary_targets | 165 | 169 |
| n_lightcurves | 162 | 166 |

Justification: identity gate (4 mis-associated targets) + `delta_mag_sysrem` schema column.

`test_invariants_p1_golden.py`: **5/5 PASS** with both chains executed (incl. `test_ui_chain_byte_identity`).

**Run time:** ~897 s recut script + ~815 s verify pytest.

---

## Part 3 - VL-ANCHOR-WCSINV schema re-cut

Export leak check: **PASS** -- AAVSO writes explicit columns (BJD, mag, err); VarAstro uses
`delta_mag`, not `delta_mag_sysrem`.

Two agreeing `--full` runs from POST-453 fixes (2409 s / 2352 s); science columns bit-identical.

| Field | New | Superseded |
|-------|-----|------------|
| core_sha | `b7f980c09e238b855c2ee1b9518061777934d8f0a61eaec7431cda4f537aed52` | `1c48d9fc...` |
| extended_sha | `2c43bbbf06921fbef46fb6a4ed1f8afccdabacaa5827b8ec50372de0e3816205` | `744bce94...` |
| core_n | 325 | 325 |
| extended_n | 487 | 487 |

Justification: schema harmonisation (`delta_mag_sysrem` always written); science bit-identical.

Local anchor snapshot photometry patched from agreeing run1 (`tmp/session_baseline/20260728T065730Z`);
snapshot SHA now matches ledger. `test_invariants_p1_seed.py` updated (325/487; census via ledger fingerprint).

**Run time:** prior session runs (not re-executed this session).

---

## Part 4 - Decisions / process

- `VYVAR_DECISIONS.md`: F-431 recurrence note (SKIPPROC); SKY-SURFACE-RESTORE second occurrence + INV-PREP-01 guard.
- `VYVAR_PROCESS.md`: closing without regression guard lesson.

---

## Part 5 - Frame 001 residual

- Raw 451 vs 452 frame 001: **0.0 ADU** (identical).
- Calibrated 451 vs 452: **659.6 ADU** (frame 001) vs **121.2 ADU** (frame 002).
- 451 preprocess -> 452 cal: frame 001 **533.5 ADU** residual; frames 002-010 **0.0**.
- **Not** current calibration non-determinism: draft **451 frame-001 calibrated product** carries a
  structured defect beyond missing VYSKYORD (~5x larger cal delta than sibling frames with identical raw).
- Root cause of 451 cal anomaly **not on disk** (would need 451 calibration run logs / per-frame masters).

---

## Acceptance

| Item | Status |
|------|--------|
| Part 1 | PASS |
| Part 2 | PASS (core_n=325 as expected) |
| Part 3 | PASS (no export leak; ledger updated) |
| Part 4 | PASS |
| Part 5 | Explained; root cause of 451 cal defect needs missing logs |
| `--fast` | PASS |
| `ruff` | PASS |

Recurrence: existing `test_invariants_p1_golden.py` (fixture fix strengthens gate).

Docs impact: `VYVAR_DECISIONS.md`, `VYVAR_PROCESS.md`, `VYVAR_VALIDATION_LEDGER.json`, `session_baseline_check.py`.
