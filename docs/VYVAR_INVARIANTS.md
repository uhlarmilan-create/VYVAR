# VYVAR - machine-enforced invariants registry

Human registry for VYVAR-INVARIANTS (DECISIONS 2026-07-16). Runtime helpers:
`src_py/invariants_runtime.py`. Results land in `pipeline_meta.json` under
`invariants` / `stages` (outside photometry SHA / science compare scope).

Policy: **FAIL** = FAIL-CLOSED (raises `InvariantViolation`); **WARN** = log +
record, run continues. Wired gates are check-only (never mutate science arrays).

| ID | Contract statement | Enforced | Policy | Evidence / history |
|----|-------------------|----------|--------|--------------------|
| INV-FLUX-01 **[wired]** | `dark_resample` conserves total flux: block-sum downscale and uniform upscale preserve SUM within relative 1e-6. | both (runtime in `get_processed_master` / resample path; unit tests for upscale) | FAIL | Calibration masters; broken resample = broken dark. |
| INV-FLUX-02 **[wired]** | Normalized master flat has mean ? 1.0 within relative 1e-3. | both | FAIL | `normalize_flat_master` / `get_processed_master`. |
| INV-FLUX-03 | Sky-surface preprocess subtracts the **full** fitted surface including the constant term (pedestal convention; **not** mean-preserving). Provenance sky-surface stats mandatory when applied. | provenance schema (INV-PROV-01) + FLOW ch 4.5 wording | FAIL (schema) | T3-PREPROCESS-SKY-SURFACE; FLOW 4.5 doc-drift fixed 2026-07-19 (was wrongly "flux-conserving"). |
| INV-FLAT-01 **[wired]** | Post-preprocess residual large-scale flatness: order-2 refit on the processed frame has p99 \|surface\| below a generous band (default 400 ADU; known honest gap vs oracle - T3). | both | WARN | Value recorded in invariants detail / sky_stats. |
| INV-WCS-00 | WCS invertibility / round-trip gate (F-428). | already enforced (F-428) | FAIL | Reference only; not re-wired in P2. |
| INV-WCS-01 **[wired]** | MASTERSTAR matched world?pix identity p95 ? 2.0 px (band around 1.54 px draft_435 baseline). | both | WARN | `evaluate_matched_world2pix_identity_px`; meta key `matched_world2pix_identity_p95_px`. |
| INV-DAG-01 **[wired]** | Stage ordering: each stage stamps `pipeline_meta.stages` with `(name, seq, head_inputs_present)`; a stage refuses to run if its declared upstream stamp is missing (cold-start mid-pipeline entry allowed when `stages` empty). DAG: calibrate ? preprocess ? align ? masterstar ? perframe ? phase01 ? phase2a ? postprocess. | both | FAIL | `stamp_pipeline_stage` / `stamp_stage_on_disk`. |
| INV-RNG-01 **[wired]** | Determinism: science path has no naked global-RNG calls (`np.random.<fn>(` without a `Generator` seeded via `SeedSequence`). Seeds/policy recorded in provenance (`labbe_rng_seed_policy`). | test (AST/grep over `src_py`) + schema | FAIL (test/schema) | LABBE-DET pattern; allowlist empty on 2026-07-19 tree. |
| INV-PROV-01 **[wired]** | Provenance schema: `pipeline_meta.json` validates a minimal schema (`prov_schema_version`, provenance keys incl. `labbe_rng_seed_policy`, sky_surface stats when applied, `cog_night_fallback` iff COG enabled, `invariants` block, census keys when masterstar stamped). | runtime (end-of-run) | FAIL | `validate_provenance_schema`. |
| INV-CFG-01 **[wired]** | Config?behavior no-op: when a gating flag is OFF, its provenance markers are ABSENT. Wired set: `psf_photometry_enabled=False` ? no method=="psf" LC rows; `temporal_binning_enabled=False` ? no binning-applied markers; `cog_aperture_correction_enabled=False` ? no cog meta keys; `per_frame_saturation_enabled=False` ? no `per_frame_sat_*` meta keys and no `sat_clean_frac` / `skip_reason=per_frame_saturation` in `photometry_summary.csv`. | runtime (end-of-run) | FAIL | `validate_config_behavior`. |
| INV-SHA-01 | Double-photometry SHA determinism (core + extended). | already enforced (`session_baseline_check.py --full` + P1 golden) | FAIL | VL-ANCHOR-WCSINV / VL-P1-GOLD; registry pointer only. |

## Meta / SHA scope (P2 finding)

`pipeline_meta.json` is **not** part of core/extended photometry SHA or the
`--full` science comparator file set (`dev/tests/photometry_sha.py`). Invariants
and stage stamps may be written to `pipeline_meta.json` without breaking
byte-identity of science outputs.

## Growth rule

Broader enforcement expands under P3 recurrence discipline. Do not gold-plate
beyond the **[wired]** set without a new ROADMAP phase.
