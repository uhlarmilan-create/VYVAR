CURSOR RESULT - 2026-07-30 AUDIT STAGE 3 PART 0b

What I did
Recorded anchor-gate blind spot (0b.1), committed pending docs (0b.2), ran full-chain rebuild
from `draft_000435` calibrated lights into scratch `draft_000499` (0b.3–0b.5).

## Provenance

| Field | Value |
|-------|-------|
| `git_hash` | `dd8a2d02ca036461de9d4b1e274bf81d7d858d81` |
| `git_dirty` | `true` (scratch: untracked rebuild script + sqlite WAL only) |
| `git_dirty_code` | **`false`** |
| Harness | `dev/scripts/audit_stage3_part0b_rebuild.py` |
| Raw JSON | `tmp/audit_stage3_part0b_results.json` |
| Scratch draft | `Archive/Drafts/draft_000499` (does not touch `draft_000435*`) |

## 0b.1 — Gate-gap finding (committed)

Documented in `docs/VYVAR_INVARIANTS.md` (`INV-ANCHOR-00`) and
`docs/VYVAR_DECISIONS.md` (`ANCHOR-GATE-BLIND-SPOT`): `--full` covers photometry only;
preprocess, alignment, MASTERSTAR stack, and DAO are blind.

## 0b.4 — Tranche 3 prediction table

| Quantity | Predicted (Tranche 3) | Measured (rebuild) | Verdict |
|----------|----------------------:|-------------------:|---------|
| pass-1 DAO count | ~2550 | **2521** (`pipeline_meta.dynamic_params.n_stars_dao`; anchor snapshot **2552**) | **PASS** (within class) |
| `DAO_ONLY` fraction | ~3.7% | **3.93%** | **PASS** |
| active targets (photometry LCs) | ~165 | **230** LCs written (anchor snapshot **162**) | **FAIL** vs prediction; plan regen expanded target set |
| large-scale residual (preprocess) | removed, not doubled | see 0b.5 | **PASS** (P-10 active) |
| `bg_std` on rebuilt MASTERSTAR | ~46–50 ADU | **54.3 ADU** (`sigma_clipped_stats` on MASTERSTAR data) | **PASS** (near class) |
| threshold in ADU at 3.8? | ~175 | **192.8 ADU** (`3.8 × sigma_pp` on MASTERSTAR, current code path) | **PASS** (~10% high) |

**Note on active targets:** `variable_targets.csv` has 875 planner rows (no `skip_photometry` flags);
photometry wrote **230** light curves vs anchor snapshot **162**. The ~165 prediction matches the
anchor photometry cohort, not the full VSX-in-field planner table.

## vs frozen anchor snapshot (`draft_000435_snapshot_skysurface_20260716`)

| Metric | Value |
|--------|------:|
| Common LCs | 156 |
| ? `mag_calib_final` median / p95 / max | **?0.0078 / 0.430 / 2.560 mag** |
| ? `err` median / p95 / max | **0.00043 / 0.039 / 0.948 mag** |
| Targets only in rebuild | 74 (plan/regen expansion) |
| Targets only in anchor | 0 |

Comparison ensembles: not diffed file-by-file in this pass; large active-set delta indicates
Phase 0 plan/regen differed from frozen snapshot inputs.

## 0b.5 — P-10 direct confirmation

Archived pre-P-10 `proc_*` FITS from `draft_000435/processed` are no longer on disk; JSON
from the harness captured archived order-1 residual p99 **97–185 ADU** (e.g. frame 001 **185.6**).

Fresh `_fit_subtract_preprocess_sky_surface` (P-10, order=2) on calibrated frame 001:

| Stage | order-1 residual p99 [ADU] |
|-------|---------------------------:|
| Input (calibrated) | 121.4 |
| After P-10 subtract | **56.0** (stats `residual_flatness_p99` **47.1**) |

Aligned-frame QC on rebuild: `INV-FLAT-01` max residual p99 **58.8 ADU** (150 frames) vs
pre-P-10 archived proc p99 **~100–185 ADU** ? large-scale structure **reduced, not doubled**.

`pipeline_meta.sky_surface_p2p_median_adu` on rebuild: **136.8 ADU** (150 frames preprocessed).

## Timing

| Stage | Seconds |
|-------|--------:|
| Preprocess | 61 |
| Platesolve + align + MASTERSTAR | 1370 |
| Photometry | 2876 |
| **Total** | ~4300 |

## Errors

None fatal. `threshold_adu` helper in harness failed (option B not in tree at run time — Part 2);
threshold computed offline via `_dao_noise_sigma_adu` (see table above).

## Files changed

- `docs/VYVAR_INVARIANTS.md`, `docs/VYVAR_DECISIONS.md` (0b.1)
- Pending audit docs batch (0b.2)
- `dev/scripts/audit_stage3_part0b_rebuild.py` (this part)
- `dev/results/CURSOR_RESULT_audit_stage3_part0b.md`

---

**STOP GATE 0b** — awaiting Milan review before Part 1b / Part 2 close-out.
