CURSOR RESULT - 2026-08-13

What I did
Filed updated saturation memo; added SAT-DIAG decision stub and ROADMAP item; investigated CAL-DIAG removal (`967f835`); wrote `docs/VYVAR_SAT_DIAG_SPEC.md` and planned invariant INV-SAT-01. No implementation. Not pushed.

---

## Part A - Memo filed

| Item | Path |
|------|------|
| Memo (updated ?7 confirmed, rec 6, authoritative table) | `dev/results/MEMO_saturation_limit_literature.md` |
| DECISIONS stub (OPEN - awaiting Milan) | `docs/VYVAR_DECISIONS.md` ? **SAT-DIAG** |
| ROADMAP open item + measured consequence | `docs/VYVAR_ROADMAP.md` ? **OPEN - SAT-DIAG** |

---

## Part B - CAL-DIAG removal (`967f835`, 2026-08-11)

### B.1 What was removed

| Category | Removed |
|----------|---------|
| **Module** | `src_py/cal_diag.py` (486 lines) |
| **Tests** | `dev/tests/test_cal_diag_gate.py` (409 lines, 14 tests) |
| **Pipeline wiring** | `run_cal_diag_pregate`, `ensure_cal_diag_gate`, `dark_np_for_cal_diag`, `apply_cal_diag_headers`, `passthrough_cal_diag_headers`, `write_cal_diag_json`, `load_cal_diag_json_for_meta`, MP session export/import, `_saturation_adu_for_cal_diag`, per-obs_group abort logic |
| **Photometry** | `cal_diag` block merge in `photometry_core.py` (12 lines) |
| **UI** | `cal_diag_gate_enabled` exposure in `ui_settings.py` |
| **Config keys** | `cal_diag_gate_enabled`, `cal_diag_autocorrect_enabled`, `cal_diag_rel_tol`, `cal_diag_hard_sigma`, `cal_diag_sat_warn_frac` |
| **Registry** | Five keys removed (287 ? 282) |
| **Provenance flag** | `dark_resample = SUM \| MEAN_AUTOCORRECTED \| PASSTHROUGH` (runtime) |
| **Headers** | `VY_DKRSMP`, `VY_CDSKY`, `VY_CDSTAT` (no longer written) |
| **Sidecar** | `archive/<draft>/cal_diag.json` (no longer written) |

### B.2 Quoted reason

Commit message (`967f835`):

> Remove CAL-DIAG gate and its five config parameters
>
> Delete cal_diag.py and all pregate/header/provenance wiring. Calibration stays dark SUM / flat MEAN resample only. Registry 287 to 282 keys. P1 core SHA byte-identical.

No entry in `docs/VYVAR_JOURNAL.md`, `docs/VYVAR_DECISIONS.md`, or task files records the removal rationale beyond this commit message. Prior param-budget audit (`dev/results/param_budget_audit.csv`) listed all five `cal_diag_*` keys as **HARDCODE candidates** for config reduction.

### B.3 Deliberate or side effect?

**Deliberate removal during configuration-parameter reduction**, not an accidental regression. The commit explicitly deletes the gate and states calibration remains **dark SUM / flat MEAN only**. P1 byte-identity was the acceptance criterion.

### B.4 Verification today

| Check | Survives? |
|-------|-----------|
| CAL-DIAG pre-subtraction `median(light)` vs `median(dark)` | **No** |
| Post-dark sky median sanity (`median < -k*sigma`) | **No** |
| AUTO-CORRECT SUM ? MEAN on mismatch | **No** |
| **INV-FLUX-01** flux conservation on block resample | **Yes** (`invariants_runtime.py`, unit tests) |
| Hardcoded `dark_resample_mode="sum"` | **Yes** (`pipeline.py:760`, `15649`; `calibration.py`) |

Nothing today verifies that a master dark resampled to different binning does not **over-subtract radiometrically**. The `median < 0` sign-flip heuristics in `pipeline.py:1447`, `15387` serve DAO roundness helpers only - not a calibration gate.

### B.5 `VY_DKRSMP` on existing drafts

| Draft | `VY_DKRSMP` on calibrated Light_001 | `cal_diag.json` |
|-------|-------------------------------------|-----------------|
| 435 | **SUM** (+ `VY_CDSKY`, `VY_CDSTAT=PASS`) | **present** |
| 509 | absent | absent |
| 510 | absent | absent |

---

## Part C - SAT-DIAG spec

**Path:** `docs/VYVAR_SAT_DIAG_SPEC.md`

Summary: invariant-backed gate (`INV-SAT-01` planned in `docs/VYVAR_INVARIANTS.md`); derive ceiling from raw pile-up; resolve limits header ? equipment ? derived; two levels (saturation + linearity); raw peaks with self-check; flags separated from consumer policy; provenance `VY_SATSRC` / `sat_diag.json`; survives config reduction unlike CAL-DIAG.

**Decisions left open for Milan:**

1. Interim limit source and CONFLICT handling (fail-closed vs adapt)
2. Equipment table schema for `(readmode, binning)` keys
3. Exposure-ramp linearity measurement schedule
4. Default consumer policies (pool / AC / PSF / export) per flag type

---

## Part D

See sections above.

**DECISIONS REQUIRED (before SAT-DIAG implementation):**

1. Interim limit: derive from draft pile-up vs trust equipment/header; CONFLICT policy
2. Target structure: two levels keyed by equipment + readmode + binning
3. Exposure-ramp linearity measurement at telescope (or accept DEFAULT_FRAC with loud WARN)
4. Consumer policies: which flags auto-exclude vs warn-only

## Files changed

- `dev/results/MEMO_saturation_limit_literature.md` (new)
- `docs/VYVAR_SAT_DIAG_SPEC.md` (new)
- `docs/VYVAR_DECISIONS.md` (SAT-DIAG stub; CAL-DIAG status note)
- `docs/VYVAR_ROADMAP.md` (SAT-DIAG open; CAL-DIAG superseded)
- `docs/VYVAR_INVARIANTS.md` (INV-SAT-01 planned)
- `dev/results/CURSOR_RESULT_sat_diag_spec_and_caldiag.md` (this file)

Not pushed.
