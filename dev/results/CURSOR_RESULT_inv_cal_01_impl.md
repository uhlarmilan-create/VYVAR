CURSOR RESULT - 2026-08-13 INV-CAL-01 (CAL-DIAG v2)

What I did
Implemented CAL-DIAG v2 per `dev/results/specs/VYVAR_CAL_DIAG_V2_SPEC.md` with Decision 2
amendment (`INDETERMINATE_NEGLIGIBLE` / `INDETERMINATE_UNMEASURED`). Wired pipeline pregate,
headers, `cal_diag.json`, photometry meta merge, **INV-CAL-01** runtime gate. Added 11 unit
tests. Updated INVARIANTS, DECISIONS, STATE, ROADMAP, JOURNAL, SAT-DIAG spec pedestal note.
**Not pushed** (per authorization).

---

## Pre-registered predictions

| ID | Prediction | Result | Measured |
|----|------------|--------|----------|
| **P1** | Draft 435: SUM, DERIVED, Check B PASS, calibrated **byte-identical** | **PASS** | 150/150 frames `np.array_equal`; max abs diff **0.0**; `VY_DKRSMP=SUM`, `VY_DKRSMP_SRC=DERIVED`, `VY_CDSTAT=PASS`, sky **2399.5293 ADU** |
| **P2** | Drafts 509/510: SUM, DERIVED, calibrated unchanged | **FAIL** (archive compare) | Recalibrate vs on-disk archive: **150/150 pixel mismatch** both drafts (max diff ~123 ADU on 510). Gate resolves **SUM/DERIVED**; fresh cal **matches manifest QC** background (e.g. 510 Light_001 median **2414.373** = manifest `qc.background`). Archive FITS appear **stale** vs manifest from recent RAM cal |
| **P3** | P **24.548 +/- 0.011**, Delta **~73.65**, R **~102** | **PARTIAL** | Gate on 435 Light_001: P **24.470589**, sigma_p **0.000000** (intercept on 2 library exptimes); Delta_meas **73.5882**, Delta_pred **73.4118**, R **101.7554**, resolv_limit **1.0 ADU** |
| **P4** | Synthetic MEAN: MEAN, R near 1 | **PASS** | `test_gate_mean_driver_derived`: MEAN/DERIVED, R **2.5** (driver-averaged light 25 vs dark mean 10) |
| **P5** | Synthetic CCD: ABORT `CCD_LINEAR_INCONSISTENT` | **PASS** | `test_gate_ccd_linear_inconsistent_aborts` |
| **P6** | NEGLIGIBLE vs UNMEASURED distinguishable | **PASS** | `test_gate_indeterminate_negligible` -> `INDETERMINATE_NEGLIGIBLE`; `test_gate_indeterminate_unmeasured` -> `INDETERMINATE_UNMEASURED`; headers stamp `VY_DKRSMP_SRC` |
| **P7** | `--fast` OVERALL PASS | **PASS** | **1312 passed**, 27 skipped |
| **P8** | Gate check-only: SUM/DERIVED does not change calibration arithmetic | **PASS** | P1 150/150 pixel identity on anchor 435 proves gate + SUM path matches pre-removal hardcoded SUM |

**P8 (agent):** If implementation changed dark resample arithmetic, draft 435 pixels would move. They do not.

---

## Case table (spec section 7)

| Case | Test / evidence | Outcome |
|------|-----------------|--------|
| CMOS SUM (QHY294MM) | Draft 435 gate + P1 | **SUM, DERIVED, PASS** |
| CMOS MEAN driver | `test_gate_mean_driver_derived` | **MEAN, DERIVED, PASS** |
| CMOS MEAN light + SUM dark misconfig | Not isolated (would ABORT Check B or CONFLICT) | Covered by v1 garbage-dark pattern; v2 CONFLICT path in code |
| CCD on-chip bin | `test_gate_ccd_linear_inconsistent_aborts` | **ABORT CCD_LINEAR_INCONSISTENT** |
| Matched binning bf=1 | Not dedicated test | Code path: convention NONE, Check B only |
| Single master exptime | Draft 435 (one 60s master + library siblings for intercept) | **DERIVED SUM** |
| Single master warm k confounded | `test_gate_indeterminate_unmeasured` (pedestal not measurable) | **INDETERMINATE_UNMEASURED, WARN, SUM** |
| No dark | `test_passthrough_headers` | **PASSTHROUGH** |
| P below resolvability | `test_gate_indeterminate_negligible` | **INDETERMINATE_NEGLIGIBLE, WARN, SUM** |
| Wrong master | Not synthetic in suite | ABORT paths in code (WRONG_MASTER, CHECK_B_FAIL) |

---

## Resolvability floor (implementation, real data)

On draft 435 Light_001 (`bf=2`, library intercept P):

```
resolv_limit_adu = max(3 * sigma_p * (bf^2 - 1), 1.0) = 1.0 ADU
```

(Pedestal indistinguishable when P < **0.33 ADU** at bf=2 per spec; Milan P ~ 24.5 >> floor.)

---

## `--fast` (raw)

```
SESSION BASELINE CHECK (fast)
pytest                       PASS   1312 passed, 27 skipped
OVERALL: PASS
```

---

## Draft 435 hash compare (P1)

| Metric | Value |
|--------|-------|
| Frames compared | 150 |
| Pixel mismatches | **0** |
| max abs diff | **0.0** |
| cal_diag_aborted_groups | 0 |

Headers on new cal: `VY_DKRSMP=SUM`, `VY_DKRSMP_SRC=DERIVED`, `VY_CDSKY=2399.5293`, `VY_CPED=24.470589`.

---

## Draft 510 photometry (3.4)

**Not re-run** (full Phase 2A would take ~40+ min). On-disk calibrated frames for 509/510 were **not modified** by this implementation commit. Existing photometry (check scatter **0.008629**, GREEN, 134 pts, 5 comps) remains valid until user recalibrates. Recalibrate-harness shows archive 510 FITS stale vs manifest QC; photometry re-validation after recalibrate is a separate step.

---

## Case B recommendation (Decision 2 amendment)

**Recommend: proceed with SUM + loud WARN** (`INDETERMINATE_UNMEASURED`, `ui_error`, `VY_DKRSMP_SRC=INDETERMINATE_UNMEASURED`, `VY_CDSTAT=WARN`) — **not fail-closed**.

**Reasoning:**

1. Fail-closed on first run with a single master dark blocks users who have done nothing wrong; QHY294MM at -10 C is pedestal-dominated and convention is still resolved via Check C (`Delta_meas`, `R`).
2. Case B is the **residual risk when k/P are confounded**, not "single master" alone. Single-master median P is a valid measurement when dark current is negligible (Milan data).
3. Fail-closed is appropriate for **ABORT** cases (Check B hard fail, CONFLICT, CCD_LINEAR_INCONSISTENT) where the numeric counterfactuals are shown — already implemented.
4. Downstream consumers can key on `VY_DKRSMP_SRC=INDETERMINATE_UNMEASURED` for stricter policy if Milan wants tier-2 exclusion later.

**Implemented behaviour:** SUM + WARN + loud log (matches recommendation). Milan to confirm vs fail-closed.

---

## Open across session

| Item | Status |
|------|--------|
| Exposure ramp linearity measurement | Open (SAT-DIAG Tier 2/3; DEFAULT_FRAC with loud WARN) |
| Generic checksum tool | Open |
| INV-DAG-01 re-stamp friction | Open (postprocess stamp / cold-start) |
| P2 archive vs manifest for 509/510 | Investigate: recalibrate-on-disk vs manifest QC drift |
| Push | Awaiting Milan after this report |

---

## Errors (if any)

None blocking implementation or P1/P7.

---

## Files changed

| File | Change |
|------|--------|
| `src_py/cal_diag.py` | v2 module (new/restored) |
| `src_py/pipeline.py` | Pregate, headers, MP session, RAM QC, `calibrate_batch` |
| `src_py/photometry_core.py` | `cal_diag` block merge |
| `src_py/invariants_runtime.py` | `check_cal_diag`, INV-CAL-01 wired |
| `dev/tests/test_cal_diag_v2_gate.py` | 11 tests |
| `docs/VYVAR_INVARIANTS.md` | INV-CAL-01 row |
| `docs/VYVAR_DECISIONS.md` | INV-CAL-01 decisions + findings 11.1/11.2 |
| `docs/VYVAR_STATE.md`, `ROADMAP.md`, `JOURNAL.md` | Status |
| `dev/results/specs/VYVAR_CAL_DIAG_V2_SPEC.md` | IMPLEMENTED |
| `dev/results/specs/VYVAR_SAT_DIAG_SPEC.md` | Section 4.1.1 pedestal note |

Not committed. Not pushed.
