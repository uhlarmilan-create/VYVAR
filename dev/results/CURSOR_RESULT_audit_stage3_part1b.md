CURSOR RESULT - 2026-07-30 AUDIT STAGE 3 PART 1b

What I did
Added `photometer_check_star_production_path()` (production Phase 2A diagnostic), measured
check-star ?²_red on anchor snapshot `draft_000435_snapshot_skysurface_20260716`, and
reconciled Part 1's 40.98 vs 0.108 discrepancy.

## Provenance

| Field | Value |
|-------|-------|
| `git_hash` | `47007f56cb71777ee65f1d129814103ce31bceed` (+ uncommitted 1b code) |
| `git_dirty` | `true` (WIP 1b/2 code) |
| Data | Anchor snapshot (162 check-star sidecars) |
| Script | `dev/scripts/audit_stage3_part1b_check_chi2.py` |
| JSON | `tmp/audit_stage3_part1b_results.json` |

## 1b.3 — Production-path check-star ?²_red

| Metric | Value |
|--------|------:|
| Check-star fields | 162 |
| Median ?²_red (production path) | **649.0** |
| Median ?²_red (target LC, same targets) | **187.6** |
| Typical N epochs | 139 |

Err-budget fractions on production check LCs (median over fields):

| Component | Fraction of variance |
|-----------|---------------------:|
| `err_photon` | ~1–26% |
| `err_sem_rel` | ~74–99% |
| `err_sigma_sys_rel` | ~0% |

`sigma_scint` (context only, not in production `err`): median **0.00183** rel flux
(Osborn/Young, D=0.2 m, 60 s, airmass~1).

## Reconciliation: 40.98 vs 0.108 — explicit verdict

| Method | Median ?²_red | Valid constant-star test? |
|--------|--------------:|---------------------------|
| Part 1 **flawed** (`kmag` + reconstructed check err) | **40.98** | **No** |
| Part 1 **“target_err_proxy”** (`kmag` + target `err` at same epochs) | **0.108** | **No** (mislabeled) |
| Part 1b **production path** (check promoted to target, production `err`) | **649** | **Yes** |

**Both Part 1 numbers were wrong as audit conclusions:**

1. **40.98** — Used `check_kmag_*` sidecar magnitudes with **reconstructed** uncertainty
   (check photon + field ensemble SEM). The `kmag` series is ensemble-corrected but the
   reconstructed err did not match the production error budget; dividing raw-ish scatter by
   too-small err inflates ?² (consistent with ~95% SEM fraction in decomposition).

2. **0.108** — **Not** ?²_red of the target light curve. Part 1 computed
   ?²(`kmag_check`, `err_target` per frame) — check-star kmag divided by **target** errors.
   That pairs two different objects and artificially yields ?² ? 1 when target err is sized
   for variable targets, not check-star scatter. It is **not comparable** to 40.98 or to
   production check-star ?².

3. **649 (production path)** — Correct pairing: `mag_calib_final` and `err` from the same
   `_phase2a_process_one_target` run with the parent target's comparison ensemble. ?² ? 1
   indicates production `err` underestimates check-star scatter (heteroscedastic small-err
   frames dominate; example target `1485540612577549568`: mag ?=0.033 mag, err median
   0.058 mag but min 0.006 mag ? ?²_red=110).

**Decision 3 stands:** scintillation (~0.002 rel) is negligible vs production err; not wired.

## 1b.4 — D1-2 deferred

Linearity vs peak-ADU drift deferred until Part 3 (CV?CR). Recorded reason: ?0.30 mag drift
at high peak ADU is likely **colour** (Gaia G vs unfiltered red-sensitive instrument;
Stage 2.2 colour slope ?0.386 mag/mag BP?RP), not sensor non-linearity. Re-run with CR
magnitudes or narrow BP?RP bin + quality cuts.

---

**STOP GATE 1b** — awaiting Milan review.
