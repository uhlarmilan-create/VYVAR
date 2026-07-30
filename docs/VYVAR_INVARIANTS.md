# VYVAR - machine-enforced invariants registry

Human registry for VYVAR-INVARIANTS (DECISIONS 2026-07-16). Runtime helpers:
`src_py/invariants_runtime.py`. Results land in `pipeline_meta.json` under
`invariants` / `stages` (outside photometry SHA / science compare scope).

Policy: **FAIL** = FAIL-CLOSED (raises `InvariantViolation`); **WARN** = log +
record, run continues. Wired gates are check-only (never mutate science arrays).

| ID | Contract statement | Enforced | Policy | Evidence / history |
|----|-------------------|----------|--------|--------------------|
| INV-FLUX-01 **[wired]** | `dark_resample` conserves total flux: block-sum downscale and uniform upscale preserve SUM within relative 1e-6. | both (runtime in `get_processed_master` / resample path; unit tests for upscale) | FAIL | Calibration masters; broken resample = broken dark. |
| INV-FLUX-02 **[wired]** | Normalized master flat has median ? 1.0 within relative 1e-3 (matches ``normalize_flat_master``). | both | FAIL | `normalize_flat_master` / `get_processed_master`. |
| INV-FLUX-03 | Sky-surface preprocess subtracts the **full** fitted surface including the constant term (pedestal convention; **not** mean-preserving). Provenance sky-surface stats mandatory when applied. | provenance schema (INV-PROV-01) + FLOW ch 4.5 wording | FAIL (schema) | T3-PREPROCESS-SKY-SURFACE; FLOW 4.5 doc-drift fixed 2026-07-19 (was wrongly "flux-conserving"). |
| INV-FLAT-01 **[wired]** | Post-preprocess residual large-scale flatness: order-2 refit on the processed frame has p99 \|surface\| below a generous band (default 400 ADU; known honest gap vs oracle - T3). | both | WARN | Value recorded in invariants detail / sky_stats. |
| INV-WCS-00 | WCS invertibility / round-trip gate (F-428). | already enforced (F-428) | FAIL | Reference only; not re-wired in P2. |
| INV-WCS-01 **[wired]** | MASTERSTAR matched world?pix identity p95 ? 2.0 px (band around 1.54 px draft_435 baseline). | both | WARN | `evaluate_matched_world2pix_identity_px`; meta key `matched_world2pix_identity_p95_px`. |
| INV-DAG-01 **[wired]** | Stage ordering: each stage stamps `pipeline_meta.stages` with `(name, seq, head_inputs_present)`; a stage refuses to run if its declared upstream stamp is missing (cold-start mid-pipeline entry allowed when `stages` empty). DAG: calibrate ? preprocess ? align ? masterstar ? perframe ? phase01 ? phase2a ? postprocess. | both | FAIL | `stamp_pipeline_stage` / `stamp_stage_on_disk`. |
| INV-RNG-01 **[wired]** | Determinism: science path has no naked global-RNG calls (`np.random.<fn>(` without a `Generator` seeded via `SeedSequence`). Seeds/policy recorded in provenance (`labbe_rng_seed_policy`). | test (AST/grep over `src_py`) + schema | FAIL (test/schema) | LABBE-DET pattern; allowlist empty on 2026-07-19 tree. |
| INV-PROV-01 **[wired]** | Provenance schema: `pipeline_meta.json` validates a minimal schema (`prov_schema_version`, provenance keys incl. `labbe_rng_seed_policy`, sky_surface stats when applied, `cog_night_fallback` iff COG enabled, `invariants` block, census keys when masterstar stamped). | runtime (end-of-run) | FAIL | `validate_provenance_schema`. |
| INV-CFG-01 **[wired]** | Config?behavior no-op: when a gating flag is OFF, its provenance markers are ABSENT. Wired set: `psf_photometry_enabled=False` ? no method=="psf" LC rows; `temporal_binning_enabled=False` ? no binning-applied markers; `cog_aperture_correction_enabled=False` ? no cog meta keys; `per_frame_saturation_enabled=False` ? no `per_frame_sat_*` meta keys and no `sat_clean_frac` / `skip_reason=per_frame_saturation` in `photometry_summary.csv`; `vsx_out_of_scope_types=[]` ? no `skip_reason=vsx_type_out_of_scope` markers. **Reverse (INV-CFG-01R, WARN):** non-empty `vsx_out_of_scope_types` with matching VSX types in-frame must produce `skip_reason=vsx_type_out_of_scope` markers. | runtime (end-of-run) | FAIL / WARN | `validate_config_behavior`. |
| INV-CFG-01R **[wired]** | Reverse of INV-CFG-01: non-empty `vsx_out_of_scope_types` with matching in-frame VSX types must produce `skip_reason=vsx_type_out_of_scope` on active targets. | runtime (end-of-run) | WARN | `validate_config_behavior`. |
| QC-01 **[wired]** | Every frame entering alignment appears in ``qc_metrics.csv`` with ``status=ok`` (exact match); violation = FAIL. | runtime (alignment collection) | FAIL | ``check_qc01_skipproc_alignment`` after allowlist join in ``astrometry_align_and_build_masterstar``. |
| OSC-01 **[wired]** | OSC equipment (``EQUIPMENTS.BAYERMASK`` set): no raw Bayer mosaic may enter alignment/photometry; every aligned FITS must carry ``VY_CHANNEL`` (extraction complete). | runtime (alignment collection) | FAIL | ``check_osc01_channel_extraction_required`` in ``astrometry_align_and_build_masterstar``. |
| OSC-02 **[wired]** | OSC draft: the four channel obs-groups (oneRGGB/R/G/B) share an **identical** post-allowlist frame ID set; violation = FAIL. | runtime (alignment orchestration) | FAIL | ``check_osc02_unified_frame_sets`` before OSC alignment bundle run. |
| OSC-03 **[wired]** | AAVSO/VarAstro export writers must not emit **oneRGGB** rows; R/G/B exports must use TR/TG/TB FILT codes respectively. | runtime (export pre-write) | FAIL | ``check_osc03_export_eligibility`` in ``export_lightcurve_reports``; oneRGGB skipped before write. |
| INV-PHASE0-ID **[wired]** | Active `catalog_id` must equal planner `catalog_id` for the same `vsx_name` (identity join; no positional adoption). | runtime (end-of-run) | FAIL | `validate_config_behavior` after Phase 0. |
| INV-PREP-01 **[wired]** | Post-preprocess large-scale gradient guard: ``large_small_ratio = var(blur(sigma=30)) / var(frame-blur)`` on one QC frame per obs_group; WARN above **10x** (threshold constant ``PREPROCESS_LARGE_SMALL_RATIO_WARN``). **Measured (draft 454, 2026-07-28):** healthy **0.03x** on BO CVn wide rig. **SKIPPROC regression (draft 450 era):** **20-60x**. Margin at threshold 10x: healthy is **~330x below** warn; regression is **2-6x above** -- threshold left at 10 because separation is enormous; not anchored on the original 1-5x estimate (that estimate was ~2 orders high vs measured healthy). | runtime (preprocess QC) | WARN | ``check_preprocess_large_small_ratio`` in ``_qc_enrich_calibrated_in_place``. |
| INV-MS-01 **[wired]** | Masterstar export DAO_ONLY fraction: WARN above 0.10, FAIL above 0.25 (anchor ~3.7%). | runtime (masterstar CSV write) | WARN / FAIL | ``check_dao_only_fraction`` before ``masterstars_full_match.csv`` write. |
| INV-VSXGAIA-DEGEN | VSX->Gaia plan-time mixture fit must fail loud when ``sigma_broad`` exceeds chance-scale guards; no fixed-radius fallback. | plan export (`vsx_gaia_crossmatch`) | FAIL | ``VsxGaiaCrossmatchDegenerateError``; see ``docs/VYVAR_DECISIONS.md`` VSX-GAIA-MATCHER-TWO-STEP. |
| INV-VSXGAIA-OUTCOME | Plan-time cross-match WARN when ``masterstars_accepted / masterstars_eligible < 80%`` (G3 recovery check). | plan export (log) | WARN | ``vsx_gaia_crossmatch`` INFO/WARNING; ``outcome_check=warn_masterstars_low``. |
| INV-SHA-01 | Double-photometry SHA determinism (core + extended). | already enforced (`session_baseline_check.py --full` + P1 golden) | FAIL | VL-ANCHOR-WCSINV / VL-P1-GOLD; registry pointer only. |
| INV-ANCHOR-00 **[audit finding]** | **Anchor `--full` gate coverage boundary.** `session_baseline_check.py --full` copies frozen inputs from the anchor snapshot and runs photometry only. It does **not** exercise calibration, preprocess, alignment, stacking, MASTERSTAR build, DAO detection, or catalogue construction. Preprocess fixes (e.g. P-10 sky-surface sign), DAO threshold/noise changes, and MASTERSTAR census changes are **invisible** to this gate unless a separate end-to-end rebuild harness is run. | reference (not wired as FAIL) | — | `CURSOR_RESULT_masterstar_count_diag.md`; Audit Stage 3 Part 0b (2026-07-30). |

### Anchor `--full` stage coverage (INV-ANCHOR-00 detail)

| Pipeline stage | Covered by `--full`? | Notes |
|----------------|---------------------|-------|
| Calibration (flat/dark/bias) | **No** | Not in snapshot path |
| Preprocess (sky surface, etc.) | **No** | Uses frozen `detrended_aligned/lights` (aligned lights already preprocessed upstream) |
| Alignment / detrending | **No** | Frozen aligned lights copied from snapshot |
| MASTERSTAR stack + build | **No** | Frozen `MASTERSTAR.fits` copied |
| DAO pass-1 / pass-2 detection | **No** | Frozen `masterstars_full_match.csv` copied |
| Catalogue / variable-target plan | **Partial** | Plan regen fingerprint checked; catalogue CSV frozen |
| Per-frame DAO + photometry (Phase 0/2A) | **Yes** | `run_full_photometry_pipeline` on copied inputs |
| Postprocess / export | **Yes** | Same photometry path |

End-to-end verification of preprocess, detection, and MASTERSTAR requires a **full-chain rebuild harness** from calibrated lights (Audit Stage 3 Part 0b), not `--full` alone.

## Meta / SHA scope (P2 finding)

`pipeline_meta.json` is **not** part of core/extended photometry SHA or the
`--full` science comparator file set (`dev/tests/photometry_sha.py`). Invariants
and stage stamps may be written to `pipeline_meta.json` without breaking
byte-identity of science outputs.

## Growth rule

Broader enforcement expands under P3 recurrence discipline. Do not gold-plate
beyond the **[wired]** set without a new ROADMAP phase.
