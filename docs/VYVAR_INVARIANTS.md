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
| INV-VSXGAIA-DEGEN | VSX->Gaia plan-time mixture fit must fail loud when ``sigma_broad`` exceeds chance-scale guards; no fixed-radius fallback. | plan export (`vsx_gaia_crossmatch`) | FAIL | ``VsxGaiaCrossmatchDegenerateError``; see ``docs/VYVAR_DECISIONS.md`` VSX-GAIA-MATCHER-TWO-STEP. |
| INV-VSXGAIA-OUTCOME | Plan-time cross-match WARN when ``masterstars_accepted / masterstars_eligible < 80%`` (G3 recovery check). | plan export (log) | WARN | ``vsx_gaia_crossmatch`` INFO/WARNING; ``outcome_check=warn_masterstars_low``. |
| INV-SHA-01 | Double-photometry SHA determinism (core + extended). | already enforced (`session_baseline_check.py --full` + P1 golden) | FAIL | VL-ANCHOR-WCSINV / VL-P1-GOLD; registry pointer only. |
| INV-ANCHOR-00 **[audit finding]** | **Anchor `--full` gate coverage boundary.** `session_baseline_check.py --full` copies frozen inputs from the anchor snapshot and runs photometry only. It does **not** exercise calibration, preprocess, alignment, stacking, MASTERSTAR build, DAO detection, or catalogue construction. Preprocess fixes (e.g. P-10 sky-surface sign), DAO threshold/noise changes, and MASTERSTAR census changes are **invisible** to this gate unless a separate end-to-end rebuild harness is run. | reference (not wired as FAIL) | - | `CURSOR_RESULT_masterstar_count_diag.md`; Audit Stage 3 Part 0b (2026-07-30). |
| INV-COMP-MEMBERSHIP **[wired]** | Comparison-star membership in the ensemble is decided ONCE per draft by a selection rule, never per frame. A star is either a valid comparison for the observation or it is not. Membership that changes frame to frame converts a smooth systematic into a step function in the zeropoint and is forbidden regardless of the criterion used to decide it. **Scope:** ``photometry_core.ensemble_normalize`` (Phase 2A ensemble combine). **Out of scope:** Phase-1 candidate rejection, SAT/edge gates, per-frame finite-mag skip. | test ``dev/tests/test_iron_gates.py::test_inv_comp_membership_ensemble_normalize`` | FAIL | Draft 509 BO CVn: per-frame 3xMAD ZP clip at N>=4 produced ~50 mmag two-state ZP; see DECISIONS ZP-CLIP-REMOVAL-2026-08-12. |
| INV-PIN-01 **[wired]** | Pinned target reproduces exact comp membership from ``pinned_ensembles.csv`` when all members pass data-derived re-validation (no silent substitution). Drops with named reasons are WARN-only. | runtime (``pinned_ensembles.verify_inv_pin_01``) | FAIL | DAO-GAIA ERA-03 anchor 48-target pin path. |
| INV-PIN-02 **[wired]** | Rule-violating pinned members must drop with a non-empty named reason (never silent empty drop). | runtime (``pinned_ensembles.verify_inv_pin_02``) | FAIL | Pair with INV-PIN-01. |
| INV-PIN-03 **[wired]** | ``pinned_ensembles.csv`` SHA256 must appear in ``pipeline_meta.pinned_ensembles_sha256`` when pin mode active. | runtime (``pinned_ensembles.verify_inv_pin_03``) | FAIL | Provenance gate for pin file drift. |
| INV-PIN-04 **[wired]** | Catalog-derived colour cannot newly fail on re-validation: pin-time tier pass implies current tier-limit pass; catalog ``bp_rp`` delta must be stable vs pin-time comp_pt. Tier limits only (no global ``max_delta_bprp_cfg`` ceiling). | runtime (``pinned_ensembles.verify_inv_pin_04``) | FAIL | Panel-red R1; supersedes ad-hoc colour ceiling misuse. |
| INV-NOCLIP-01 **[wired]** | No sigma-clipping, kappa-sigma or outlier rejection in the science data production path. **Scope:** ``src_py/photometry_core.py``, ``pipeline.py``, ``comp_*.py``, ``check_star_kmag.py``, ``calibration.py``, ``importer.py``, ``trust_flag_core.py``, ``method_lc_output.py``, ``export_reports.py``, ``sat_diag.py``, ``cal_diag.py``, ``cal_stage.py``, ``vyvar_alignment_frame.py``, ``psf_photometry.py``, ``psf_neighbor_sub.py``, ``plain_stats.py``. **Out of scope:** ``xval_*``, ``tess_verify.py``, ``validate_lc_crossval.py``, ``hrd_*``, ``variability_detector.py``, UI modules, ``dev/`` diagnostics. | test ``dev/tests/test_iron_gates.py`` (static scan + fire proof) | FAIL | SKY-CLIP-01 2026-08-14; iron rules previously policy-only. |
| INV-NOCOSMIC-01 **[wired]** | No cosmic-ray cleaning on science data. **Scope:** same production module list as INV-NOCLIP-01. **Out of scope:** DAO anti-CR concentration checks (peak/sum), SAT flags, offline harness. | test ``dev/tests/test_iron_gates.py::test_inv_nocosmic01_production_scope_clean`` | FAIL | LaCosmic removed 2026-08-12 (DECISIONS). |
| INV-PIXELS-01 **[wired]** | Science pixels are never modified, masked, zeroed or interpolated; saturated stars are flagged in metadata only. **Scope:** same production module list. **Known exception under review:** non-finite pixel fill with ``nanmedian`` before photometry (``photometry_core``, ``pipeline``, ``psf_photometry``). **Out of scope:** BPM bad-column flags, export/reporting masks. | test ``dev/tests/test_iron_gates.py::test_inv_pixels01_known_sites_only`` | FAIL (review) | Milan adjudication pending CLOSE-IRON-GATES 2026-08-14. |
| INV-MASTER-01 **[wired]** | Master dark and flat built by plain combine (mean or median), no rejection. **Scope:** ``importer.py``, ``calibration.py``, ``pipeline.py`` master-build paths. | test ``dev/tests/test_iron_gates.py::test_inv_master01_plain_combine_only`` | FAIL | ``_combine_stack_mean`` / ``_combine_stack_median`` in ``importer.py``. |
| INV-MS-CENSUS-01 **[wired]** | Gaia-complete MASTERSTAR census CSV has one row per on-chip Gaia source; ``source_state`` is one of DETECTED_P1/P2, FORCED_SEED, SEED_REJECTED, TOO_FAINT, BLENDED, SATURATED, EDGE. Census is always written; mismatch FAIL-closed. | runtime (MASTERSTAR catalog write + overlay) | FAIL | ``write_gaia_census_and_verify`` / ``masterstar_gaia_accounting.py``; MASTERSTAR-GAIA-01. |
| INV-SAT-01 **[wired]** | Saturation and linearity limits are expressed in **image ADU** for the active `(equipment, readout mode, XBINNING, YBINNING)`; star peaks for limit comparison are measured on **raw** frames via **placed aperture** (aligned DAO `(x,y)` on raw grid + optional 11 px COM centroid; no brightest-pixel search on comps/targets). `sat_limit_source` provenance is recorded. Aligned/resampled frames must not be the sole saturation authority when raw exists. Tier 3 (`DEFAULT_FRAC`) must not trigger exclusion. | both (`sat_diag.py` + `invariants_runtime.check_sat_diag`) | FAIL | SAT-DIAG spec 2026-08-13; placed-aperture 2026-08-13. See `dev/results/specs/VYVAR_SAT_DIAG_SPEC.md`. |
| INV-SAT-LIMIT | An unresolved saturation clip (header/equipment/BITPIX/sat_diag all missing) must not silently admit stars. Conservative default: GAIN-DOMAIN-01 container 65535 ADU, peak-test 0.80 x clip when the D1-2 knee is unmeasured, with a WARN naming value and source. | `dev/tests/test_masterstar_zone_classifier.py` | WARN (named default) | SAT-LIMIT-01 2026-08-17. Not in `WIRED_INV_IDS` (code contract in `pipeline.py`). |
| INV-CAL-01 **[wired]** | When master dark is resampled to match light binning, CAL-DIAG v2 derives SUM vs MEAN from Check P/C/B (pedestal `P`, `Delta_meas`, `R`, resolvability floor); stamps `VY_DKRSMP` + `VY_DKRSMP_SRC` on calibrated output and `cal_diag.json`. Gate is always on (zero config keys). ABORT skips obs_group fail-closed; INDETERMINATE split by cause (`NEGLIGIBLE` vs `UNMEASURED`). Check-only when convention unchanged (vs `967f835` removal). | both (`cal_diag.py` + `invariants_runtime.check_cal_diag`; pipeline pregate) | FAIL | CAL-DIAG v2 spec 2026-08-13; restores verification removed at `967f835`. INV-GATE-REMOVAL: byte-identity cannot prove gate retained. |
| INV-CAL-02 **[wired]** | Calibrated product stage integrity: ``VY_CALSTAGE`` + FITS ``VY_CALDATASUM`` stamped in the same flush as pixel mutation; legacy frames resolved honestly (`INDETERMINATE_*` when ambiguous); compare gates refuse when stage unknown; force reapply stamps ``SKYSF_N_R{pass}`` (never reuses single-pass token). Zero config keys. | both (`cal_stage.py` + write-path self-check; `invariants_runtime.check_cal_stage`) | FAIL (stamp/hash); WARN (legacy indeterminate) | CAL-STAGE spec 2026-08-13 Option A; INV-GATE-REMOVAL applies. |
| INV-GATE-REMOVAL | A verification gate may not be removed or disabled on byte-identity evidence alone. A passing gate produces identical outputs by construction, so byte-identity cannot distinguish a removed gate from a retained one. Removing a gate requires either a demonstration that the condition it checked is now impossible, or an explicit recorded decision to accept the unverified condition, with the risk stated. | policy (documented; not runtime FAIL) | - | CAL-DIAG removal `967f835`; same structural limitation as INV-ANCHOR-00. Recorded 2026-08-13 with SAT-DIAG implementation. |
| INV-NO-SILENT-EMPTY **[wired at COMP_POOL_DERIVED_ADMIT]** | A science-path population filter must not silently continue when it empties a previously non-empty population. When ``n_in > 0`` and ``n_out == 0``, raise with ``rule_id``, ``threshold``, ``unit``, ``population``, ``n_in`` so attribution does not require a log file. Helper: ``invariants_runtime.assert_population_nonempty``. First wire site: ``build_global_comp_pool`` derived admission (GATE-REGIME-01). Broader wiring follows GATE-OWNERSHIP-01 inventory. | runtime (raise ``PopulationEmptiedError``) + test fire proof | FAIL | GATE-REGIME-01 2026-08-15. Rationale: empty pool produces no usable light curve; a silent empty loses the explanation. Silent fallback to a different admission regime is forbidden. |
| INV-ERR-SIGMA-ACCT-01 **[wired]** | Empirical Labbe transport accounting: if at least one radius in ``_sigma_by_r`` carries source ``empirical`` after the per-radius loop, row assignment must emit at least one ``err_bkg_source=empirical`` row. Zero measured + zero assigned is legal (crowded field). Raises ``InvariantViolation`` (ERR-518-02; catches global_fixed key mismatch class). | runtime in ``enhance_catalog_dataframe_aperture_bpm`` + unit test | FAIL | ERR-518-01/02; would have caught draft 518 before INV-ERR-MODE-01. |
| INV-PSF-FRAME-01 **[wired]** | RUN ePSF per-frame accounting: if more than 20% of frames in a PSF photometry export job have zero ``psf_fit_ok``, the job FAILS LOUDLY (``InvariantViolation``). Below threshold with any zero-ok frames: WARN with persisted per-frame table. Exception class and full message captured on every swallowed frame failure. | runtime in ``epsf_frame_accounting.finalize_epsf_frame_job`` + unit test | FAIL (>20%); WARN (<=20% with any zero-ok) | EPSF-VALID-02 F2; draft 516 incident was 74% zero-ok frames. |
| INV-PSF-ADDITIVE-01 **[wired]** | RUN ePSF PSF-only merge: existing proc sidecar non-``psf_*`` columns (row set, order, values) must be byte-identical in memory before write. Missing sidecar = FAIL LOUDLY (no catalog fabrication). UI RUN ePSF uses ``run_epsf_psf_merge_job`` only; full ``export_per_frame_catalogs`` requires explicit ``full_catalog_export=True``. | runtime in ``epsf_psf_merge.assert_inv_psf_additive_01`` + unit tests | FAIL | EPSF-VALID-02 F6/S6; R5 accept rerun rewrote aperture columns on draft 516. |
| INV-PSF-SUBMIT-01 **[wired]** | AAVSO and VarAstro submission writers must hard-fail when the effective ``lc_method`` is ``psf`` or ``adaptive``. No config escape hatch. Internal diagnostic PSF light curves (``lightcurve_*_psf.csv``) must not route through those writers. Science exports remain aperture-only; PSF relative photometry is an internal product pending EPSF-SHAPE-01. | runtime in ``export_reports.export_lightcurve_reports`` (raise ``InvariantViolation``) + ``dev/tests/test_psf_internal_lc.py`` | FAIL | EPSF-LC-LOG-01; SESSION-CLOSE 2026-08-23 trust boundary. |
| INV-PSF-LC-PIN-01 **[wired]** | Internal PSF LC ensemble ZP uses the full pinned (or resolved) comparison set or the epoch is NaN. If any pinned comp lacks ``psf_fit_ok`` or finite ``psf_flux``, that epoch gets NaN ``psf_delta_mag`` and ``psf_epoch_drop_reason=comp_psf_fail:<gaia_id>``. Aperture columns stay filled. No partial-ensemble fallback and no substitution. | writer in ``psf_internal_lc.write_one_internal_psf_lc`` + ``dev/tests/test_psf_internal_lc.py`` | FAIL | EPSF-AC-02; analogue of INV-PIN / ERA-03 same-membership discipline on the PSF branch. |
| INV-EPSF-BUILD-GUARD-01 **[wired]** | Production ``build_epsf_model``: on non-finite EPSFBuilder result, drop edge-nearest gated star (logged in ``masterstar_epsf_meta.json`` -> ``build_guard``); FAIL LOUDLY if >10% of gated pool would be dropped. | ``psf_photometry.build_epsf_model`` + ``dev/tests/test_epsf_build_guard.py`` | FAIL | EPSF-VALID-02 S6 Addendum 1; D1b-a odd-half build failure. |

### INV-PSF-SUBMIT-01 (EPSF-LC-LOG-01)

- **Definition:** AAVSO and VarAstro submission writers must hard-fail when the
  effective ``lc_method`` is ``psf`` or ``adaptive``. Internal diagnostic PSF
  light curves (``lightcurve_*_psf.csv``) must not enter those writers.
- **Rationale:** SESSION-CLOSE 2026-08-23 trust boundary: PSF is production-usable
  for relative photometry only; absolute PSF flux on bright stars is untrusted
  pending EPSF-SHAPE-01. Science exports stay aperture-only. No config escape
  hatch -- the guard is unconditional.
- **Trigger:** ``export_reports.export_lightcurve_reports`` raises
  ``InvariantViolation("INV-PSF-SUBMIT-01", ...)`` before any AAVSO/VarAstro
  bytes are written. ``export_all_method_lightcurve_reports`` iterates aperture
  only.
- **Test:** ``dev/tests/test_psf_internal_lc.py`` (T3: psf and adaptive raise
  with the invariant name in the message; aperture still writes).

### INV-PSF-LC-PIN-01 (EPSF-AC-02)

- **Definition:** The internal PSF light-curve ensemble zero-point for a target
  uses the full pinned (or resolved) comparison set or the epoch is NaN. If any
  pinned comparison star lacks ``psf_fit_ok`` or a finite ``psf_flux`` on that
  epoch, ``psf_delta_mag`` is NaN and ``psf_epoch_drop_reason`` is
  ``comp_psf_fail:<gaia_id>`` for the first missing member. Aperture
  ``delta_mag`` / ``err`` stay filled. There is no partial-ensemble fallback
  and no substitution of a spare comparison star.
- **Rationale:** EPSF-AC-01 A3: BO CVn PSF-vs-aperture delta_mag RMS 614 mmag
  was dominated by per-epoch membership drift. When a pinned comp failed PSF
  fit, the ZP silently renormalized from the remaining comps and the per-comp
  mag-slope leaked into ``delta_mag`` as jumps. Same-membership discipline as
  INV-PIN / ERA-03, applied to the PSF branch.
- **Trigger:** ``psf_internal_lc.write_one_internal_psf_lc`` after
  ``ensemble_normalize``.
- **Test:** ``dev/tests/test_psf_internal_lc.py`` (synthetic epoch with one
  failed comp yields NaN + reason, not a renormalized value).

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

## Removed invariants

`INV-MS-01` (masterstar DAO_ONLY purity, WARN 0.10 / FAIL 0.25) -- **removed from runtime
2026-08-05.** Thresholds were seeded on a single configuration (wide rig, VYVAR-calibrated,
measured 0.0369) and are not portable across rigs or calibration modes; the same quantity is
0.417 on a legitimate Newton/eq4 pre-calibrated run (draft_501). The fraction is retained as
an informational log line, and the anchor-regression sense is retained as a fixture test
(`dev/tests/test_invariants_p2.py`). See `docs/VYVAR_LIMITATIONS.md` (INV-MS-01-REMOVED).

## Growth rule

Broader enforcement expands under P3 recurrence discipline. Do not gold-plate
beyond the **[wired]** set without a new ROADMAP phase.
