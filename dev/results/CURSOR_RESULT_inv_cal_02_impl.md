CURSOR RESULT - 2026-08-13 (INV-CAL-02 implementation)

What I did
Implemented INV-CAL-02 Option A with Milan amendments 0.1-0.3: stage stamp + FITS DATASUM
verify, honest legacy resolver, compare refusal, force-reapply stage tokens, preprocess rename.
Did not push (awaiting Milan).

## 0.1 Decision: keep `preprocess_sky_surface_force_reapply`

**Reasoning:** Removing the key would force full recal-from-raw after algorithm fixes (P-10
class recovery). The defect was silent double-subtract being indistinguishable from single
subtract, not the existence of a reapply path. **Kept** the key; a forced second subtract now
stamps `SKYSF_N_R2` (and `VY_SKYPASS=2`) so pixels carrying two subtracts are never labeled
as one.

## Predictions

| ID | Result | Measured |
|----|--------|----------|
| P1 | **PASS** | 435/509/510 calibrated sha256: 150/150 each, mismatch 0 (435/510 manifests; 509 session baseline) |
| P2 | **PASS** | Resolver: 435 150x PURE; 509/510 150x SKYSF_2 (LEGACY_INFERRED) |
| P3 | **PASS** | Sample `proc_BO_CVn_Light_001.fits` (435 processed): INDETERMINATE_LEGACY; compare refused |
| P4 | **PASS** | Fresh cal: VY_CALSTAGE=PURE, DATASUM verifies |
| P5 | **PASS** | Force reapply: VY_CALSTAGE=SKYSF_2_R2 |
| P6 | **PASS** | trust_1498613634033133184.json unchanged: scatter 0.008629278, GREEN, n_clean=5 (134 pts in prior report) |
| P7 | **PASS** | `--fast` OVERALL PASS (1322 passed, 27 skipped) |
| P8 | **PASS** | pytest `test_calibrate_stamps_pure`: cal arithmetic unchanged; stamp is post-compute header only |

## Archive stage census (calibrated + processed trees)

| Category | Frames |
|----------|--------|
| PURE \| LEGACY_INFERRED | 316 |
| SKYSF_2 \| LEGACY_INFERRED | 300 |
| INDETERMINATE_LEGACY | 278 |
| INDETERMINATE_UNKNOWN | 0 |

Note: 577 FITS archive-wide with `VYSKYP2P` and no `VY_SKYSF` includes detrended_aligned and
other trees; 278 are under `calibrated/` + `processed/` where INV-CAL-02 resolver runs.

## INV-CAL-01 P2 (3.3)

Still **PASS** after INV-CAL-02: 509/510 150/150 max diff 0.0, stage SKYSF_2.

## Rename (0.3)

`preprocess_calibrated_to_processed` -> **`qc_enrich_calibrated_lights_in_place`** (primary);
deprecated alias retained. `app.py` / `night_run.py` call new name. No behaviour change
(same in-place `_qc_enrich_calibrated_in_place` body).

## `--fast` raw

```
pytest                       PASS   1322 passed, 27 skipped
OVERALL: PASS
```

## Commit stack (local, not pushed)

1. `feat(cal-stage): INV-CAL-02 stamp, verify, and legacy resolver`
2. `refactor(pipeline): rename preprocess entry to qc_enrich_calibrated_lights_in_place`

## Files changed

- `src_py/cal_stage.py` (new)
- `src_py/pipeline.py`, `src_py/cal_diag.py`, `src_py/invariants_runtime.py`
- `src_py/draft_provenance.py`, `src_py/database.py`
- `src_py/app.py`, `src_py/night_run.py`
- `dev/tests/test_cal_stage_gate.py` (new)
- `dev/tools/inv_cal02_validate.py` (new)
- `dev/tools/inv_cal01_validate.py`
- `docs/VYVAR_INVARIANTS.md`, `DECISIONS.md`, `STATE.md`, `ROADMAP.md`, `JOURNAL.md`
- `dev/results/specs/VYVAR_CAL_STAGE_SPEC.md`
- ascii migrate on 4 tracked result/spec files (P7 gate)

## Open across session

- Push authorization (Milan)
- Optional read-only `cal_stage.json` backfill for legacy drafts (not required for science)
- Exposure-ramp linearity (SAT-DIAG Tier 2/3)
- Generic checksum tool (roadmap)
- INV-DAG-01 re-stamp friction
- Anchor 435 optional post-restore checksum compare (restore arc)
