# VYVAR -- Known limitations (methods-paper section)

**Date:** 2026-08-02
**Purpose:** Referee-facing summary of documented and deferred audit items with measured
magnitude, fix status, and literature reference. Updated at audit closure batches A and B.

---

## A-1 -- SNR-table differential aperture (no curve-of-growth correction)

The SNR aperture table assigns magnitude-dependent radii (`aperture_r_px`, clamp 1.916 px at
the faint end) with **no curve-of-growth correction**, so the enclosed-flux fraction differs
between target and comparison stars and does not fully cancel in the differential. Verdict
robust (all 5x3 proxy cells exceed the 10 mmag gate, Step 1d). The physics expectation from
`dev/tools/closure_a1_reference_fixture.py` is **~144 mmag** for G 8-9 comparisons over the
anchor r50 span. **Recommended fix:** enable `cog_aperture_correction_enabled` (Stetson 1990
growth-curve aperture correction). Deferred to numeric batch D. Evidence:
`dev/results/CURSOR_RESULT_closure_step1g.md`, register item 16.

## A-9 -- Absolute PSF / FWHM scale unresolved

FWHM estimators disagree (`VY_FWHM_GAUSS` 2.395 px, header `VY_FWHM` 3.207 px, COG identities
4.0-4.9 px) because they measure different quantities (Gaussian core fit vs moment vs
curve-of-growth on a non-Gaussian PSF). **Not blocking** differential photometry. **Required
before any absolute claim** (enclosed-flux fraction, absolute SNR, detection completeness,
D1-2 `fwhm_ratio` test). Use curve-of-growth r50 as scale proxy, not a fitted FWHM. Status:
**DOCUMENTED**. Evidence: register item 31, Step 1f.

## D1-1 -- Cosmic-ray rejection

**FIXED (batch E, 2026-08-04).** L.A.Cosmic via astroscrappy in preprocessing (`enable_lacosmic`).
Physical re-cut: 365810 pixels cleaned on 150 frames. Evidence:
`dev/results/CURSOR_RESULT_batch_E_physical_recut.md`.

## D1-2 -- Detector linearity correction

**DEFERRED** to observing plan. Per-sensor linearity curve (Howell 2006 sec 4.4) requires a
dome-flat ramp measurement per sensor. **Not** the chosen fix for D5-2.

## D5-2 -- Production flux vs catalogue magnitude compression

**FIXED (2026-08-04, batch E).** Mechanism: saturation / detector non-linearity G 8-9.
Saturation admission gate at **70%** full well (C-1/C-2). Physical re-cut G 8-9 slope
**-0.318 -> -0.491**. Evidence: `dev/results/CURSOR_RESULT_batch_B_revised.md`,
`dev/results/CURSOR_RESULT_batch_E_physical_recut.md`.

## WIDE-ERR -- Wide-rig quoted error underquoted

Wide-rig (equipment_id 1) quoted error underquoted vs check-star scatter (H1-global).
Affects **error bars only**, not fluxes. Investigation record as of **2026-08-04**
(restored July anchor `draft_000435_snapshot_skysurface_20260716`). Evidence chain:
`dev/results/CURSOR_RESULT_wide_error_diag.md` (E0), W1/W2, A/A2b, E1-E4, AUDIT,
AUDIT-2.

### Measured (facts on this rig, restored July anchor)

- sigma_total_robust/err = **1.83** (W1, 163 fields, check star 1499906247391001088)
- Per-comp excess distribution (E4.1): median **20.18 mmag** across 1167 comp instances;
  IQR 15.77 -- 20.18 -- 25.50 mmag; rises with fainter G (11.9 -> 30.5 mmag from
  G 8-10 to G 13-14); rises with lower peak ADU (14.8 -> 25.2 mmag from ADU
  quintile q80-100 to q0-20)
- Check-star excess (E4.2): median **17.62 mmag** across 163 fields
- Ratio check/bright-comp excess: **1.48** (check star is ~50% noisier than
  brightness-matched comps -- neither generic rig floor nor clean special)
- sigma_sys_mag = **0** for equipment_id 1 (config.json)

### Eliminated (with the evidence file)

- Check star variability (E0 STEP 5: VSX no match within 10 arcsec; Gaia var_flag
  NOT_AVAILABLE) -- `CURSOR_RESULT_wide_err_step0_checkstar.md`
- Outliers (E4/W1 robust estimators consistent with non-robust)
- Trend (W1: linear/quadratic detrend does not collapse ratio) --
  `CURSOR_RESULT_wide_err_w1w2.md`
- N_eff / flux-weight mismatch (W2: predicted 1.04, measured 1.83; Spearman -0.23) --
  `CURSOR_RESULT_wide_err_w1w2.md`
- Iterative comp clipping / D12-1 (E1.3: SEM_unclipped/SEM_production = 1.00) --
  `CURSOR_RESULT_wide_err_e1.md`
- Flux-set vs SEM-set difference (E1.2: diff = 0 every frame) --
  `CURSOR_RESULT_wide_err_e1.md`
- Photon term (M4: faint-end k = 1.12, not 1.83; photon-dominated stars are OK)
- Common mode c(t) (E2.2: check_pc1_corr = 0.003 -> c(t) cancels between target
  and ensemble as expected) -- `CURSOR_RESULT_wide_err_e2.md`
- Spatial gradient (E2.2: 9% of fields significant at p<0.05, not enough to
  support EPD-style correction) -- `CURSOR_RESULT_wide_err_e2.md`
- Scintillation (A3: closes ~9% of gap even with corrected APTDIA) --
  `CURSOR_RESULT_wide_err_a.md`

### Untested (open)

- Detector non-linearity D1-2: A1 had peak_max_adu range 25774-32452 ADU (1.26x span,
  ~40-50% of full scale), insufficient lever arm to detect non-linearity. NOT refuted.
- Honeycutt (1992) primary eq. 3-5: AUDIT-2 could not retrieve. Whether VYVAR's
  std/(c4*sqrt(n)) on m - comp_ref_map approximates Honeycutt's derived error formula
  cannot be determined without the primary source. --
  `CURSOR_RESULT_wide_err_audit2.md`
- Cross-rig comparison: no Newton/Dablice or Boyden draft with check-star LCs on
  disk. Whether 20 mmag excess is a wide-rig characteristic or a VYVAR-pipeline
  characteristic is not established.

### Decision status: DEFERRED

- Voluntary batch D policy: 15-17 mmag floor sits outside Everett & Howell 2-5 mmag
  band. That band applies to well-sampled PSF; this rig has 9.55"/px and APTDIA=70 mm
  (see ROADMAP **DB-DEFECT-DIAMETER** below).
- E4.1 measured the same 15-20 mmag range independently. Two methods, same result;
  not fitted to chi2=1.
- Applying it as sigma_sys_mag = 17.6 mmag would collapse chi2 for check star from
  3.35 to 1.0 by definition, closing the measurement. Deferred until Newton or Boyden
  cross-check (ROADMAP **WIDE-ERR-CROSSRIG**).

### Retractions (do not resurrect these hypotheses without re-testing)

- **WIDE-ERR-CORRELATED-COMPS** (E2.1 predicted factor 1.90 was coincidence; the
  across-frame correlation captures c(t) which cancels) -- `CURSOR_RESULT_wide_err_e2.md`
- **WIDE-ERR-SEM-ARITH** (E1.4 measured brightness spread, not measurement scatter) --
  `CURSOR_RESULT_wide_err_e1.md`
- **WIDE-ERR-MISSING-TARGET-TERM** (E3.2 sigma_eps included photon; monotonic G rise
  is photon, not systematic) -- `CURSOR_RESULT_wide_err_e3.md`
- Multiplicative gain model (A2b M2 sky PTC on unflat-fielded frames; g_eff 0.96
  violates the science-data lower bound g >= 2.50) -- `CURSOR_RESULT_wide_err_a2b.md`
- Additive sigma_sys floor of exactly 15 mmag from batch D chi2 fit (predicts
  constant excess in mmag; A2 measured 7.8 mmag at G~10 vs 56 mmag at G~14) --
  `CURSOR_RESULT_wide_err_a.md`

Status: **DEFERRED** (decision on sigma_sys_mag pending cross-rig check). Direct
measurement: `dev/results/CURSOR_RESULT_wide_err_e4.md`.

## I-12 -- Proper motion when pmra/pmdec absent

Gaia proper-motion correction is a no-op when PM columns are missing; logging fixed (WARNING).
Not a photometry bias on the anchor; deferred to Gaia DR4 for full PM coverage. Status: **FIXED**
(logging). Evidence: `CURSOR_RESULT_audit_t2.md`.

## D11-1 -- Dilution / crowding (G proxy)

Crowding and blend dilution are not fully propagated into the reported error budget; Gaia G used
as a proxy where crowding metrics are incomplete. Affects faint-end comp selection context, not
the anchor differential aperture closure. Status: **DOCUMENTED**. Evidence: Stage 3 forensics.

## D12-1 -- Sigma-clip bias in ensemble statistics

Iterative sigma clipping on comparison-star ensembles introduces a small bias toward lower
scatter (standard in Honeycutt 1992 pipelines). Magnitude not re-measured on anchor; acknowledged
for crowded fields. Status: **DOCUMENTED**. Literature: Honeycutt (1992).

## U-09 -- DATE-OBS convention per rig

BO CVn wide rig: DATE-OBS = shutter-open, **verified**. Other rigs: convention not verified per
rig; a +EXPTIME/2 error is invisible in light-curve shape but fatal for times of minimum. QHY294
has no DATE-END/EXPMID header; driver convention **UNVERIFIED** before timing-critical submission.
Status: **MEASURED** (home rig), **DOCUMENTED** (others). Evidence: `CURSOR_RESULT_audit_stage2.md`.

## aperture_snr_sizing -- partially wired (closure Step 1, 2026-07-31)

`aperture_snr_sizing` (`config.py:727`, default `{small: 1.5, large: 4.0}`) is **partially
wired**, not dead or orphan:

- **Live path:** `pipeline.py:10051-10052` and `10713-10714` unpack `small`/`large` into
  `aperture_fwhm_factor_small` / `_large` in the per-frame settings dict; `pipeline.py:187-188`
  consumes those into `r_small_px` / `r_large_px` on the aperture-bounds path.
- **Ignored path:** `photometry_core.py:1297-1298` `compute_snr_optimal_aperture_table` keeps
  hardcoded `r_min_fwhm=0.8`, `r_max_fwhm=2.5` and no caller passes the config values -- the
  SNR-optimal sweep does not read `aperture_snr_sizing`.

**Flow doc note (Step 1b, V7):** `build_flow_doc.py:391` documents the hardcoded sweep bounds
(`r_min=0.8 x FWHM` .. `r_max=2.5 x FWHM`) correctly; `flow_doc_facts.py:60` tracks
`compute_snr_optimal_aperture_table`. The split is in config wiring, not in the FLOW PDF text.
Evidence: closure Step 1 + batch A (S1 DOCUMENTED). This note was removed from
`docs/VYVAR_PARAMS.md` in commit `faa1782` because that file is generator output only.

## INV-MS-01-REMOVED

**What it was.** Runtime invariant `INV-MS-01` compared the masterstars `dao_only_fraction`
(fraction of rows without a Gaia crossmatch) against WARN > 0.10 / FAIL > 0.25 before writing
`masterstars_full_match.csv`. Thresholds were seeded from the wide-rig VYVAR-calibrated BO CVn
anchor.

**Measured numbers (not comparable).**
- Anchor fixture `dev/results/context/session_20260727/draft_452_masterstars_full_match.csv`:
  n=2951, DAO_ONLY=109, fraction **0.0369** (wide rig, VYVAR-calibrated).
- draft_501 (Newton/eq4, pre-calibrated V lights): fraction **0.417** on 1668 detections.

Plate scale, calibration mode, Gaia cone depth, and field crowding differ between these runs.
A single threshold derived from one configuration is not a health criterion for the other.

**Design rule.** A runtime FAIL gate may only encode an invariant that holds across every
supported rig, calibration mode, and observing site. Configuration-dependent health metrics
belong in tests, reports, and logs -- never in fail-closed runtime.

**Structural defect fixed with removal.** When `INV-MS-01` failed, the broad exception handler
swallowed the error and skipped `_vyvar_df_to_csv`, leaving a pre-annotate CSV on disk and
producing 0 light curves with a misleading *"source_type annotate failed"* log line (draft_501).

**Residual risk.** VYVAR no longer hard-stops on catalogue inflation. If DAO detection inflates,
the symptom surfaces in LC quality and in the informational `MASTERSTAR DAO_ONLY census` log line
(per-class breakdown with derived `confirmable_depth_g`; implied-G deciles in `pipeline_meta`).
**DAO_ONLY class counts depend on the local Gaia DB build and must not be compared across
installations or wired to any gate.** The transferable quantity is the implied-G distribution
(`implied_g_mag`, `implied_g_minus_depth` per row), not the raw counts. No runtime gate (A-6/A-6b).

Operators should treat wide-rig `unmatched_in_range` populations as undecidable at detection
stage on current evidence (DAO-CLOSE confusion-blend test refuted local Gaia blend hypothesis on
drafts 435/500). See `docs/VYVAR_DAO_DETECTION.md`.

## MASTERSTAR architecture (enhancement thread)

Single-frame MASTERSTAR copy is scientifically usable but non-standard vs stacked reference
(Stetson 1994). Frame-selection metric `I_j`, median stack, provenance (register Steps 1-6),
admission gate C-1/C-2, and proper coaddition (TODO-B) are **enhancements**, not audit-correctness
blockers. See `docs/VYVAR_TODO_MASTERSTAR_REFERENCE.md`, `docs/VYVAR_MASTERSTAR_REFERENCE_ARCHITECTURE.md`.
