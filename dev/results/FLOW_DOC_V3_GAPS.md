# FLOW doc v3 - GAPS: follow-up work surfaced while writing the documentation

Writing the full-depth FLOW doc forced a claim-by-claim pass over the whole
pipeline. Everything stated in the doc is code-verified as of HEAD; the items
below are the places where the honest sentence had to be "prepared but not
validated", "planned", or "historical figure not re-benchmarked". Grouped by
type. Priorities are suggestions only.

## A. Program (code to write)

A1. AUTO-VSX-LIMIT - derive vsx_variable_targets_mag_limit from measured depth.
    The doc (ch 8.3, 13.5, quick-reference) says the limiting magnitude
    (SNR=5 crossing, crowding_index) is "the planned basis of an automatic
    VSX query limit". Today the limit is a static 14.5. Task: report-level
    suggestion first (print G_lim_90 and SNR5 limit next to the configured
    value + warn on mismatch), config automation later (opt-in flag).
    Effort: small. Validation: existing drafts, no new data needed.

A2. SIGMA-SYS-PER-RIG - sigma_sys_mag has only band "4" (wide, 18 mmag).
    Newton and Brno filtered bands have no calibrated floor, so their err
    bars silently omit the systematic term. Task: calibrate floors from
    check-star scatter per rig/band (procedure already defined - same as
    wide) once enough nights exist; until then consider a report warning
    "no sigma_sys floor for this band".
    Effort: small code, needs per-rig nights.

A3. DOCS-REVISION-RITUAL - PROCESS.md should list "regenerate FLOW PDF via
    builder" in the docs-revision checklist (the doc promises this ritual).
    Verify it is there; add if missing. Effort: trivial.

## B. Verify on REAL data (blocked on observations)

B1. NEWTON-DENSE-FIELD (the big gate; already on ROADMAP). One dense-field
    draft (h/chi Per class) unlocks three validations at once:
    - PSF enablement cross-validation (aperture vs PSF on isolated stars;
      acceptance: center agreement within a few mmag, no trend with mag),
    - crowding classifier ON/OFF comparison,
    - dense-profile density adaptation sanity (does "tighten" behave?).

B2. K2-DATA-BLOCKER / NIGHT_FIT v2 - per-night k'' fit stays gated OFF for
    lack of a night with sufficient detectability (color spread x airmass
    range). Task: identify/plan such a night (wide rig, long run through
    X ~ 1.1 -> 2.0, field with broad BP-RP range), then evaluate fit vs
    literature consistency gates. See also C1 (synthetic pre-validation).

B3. F-BINGAIN-1 - bin2 gain scaling latent finding; photon transfer on field
    lights was inconclusive. Not live on the production wide path (scaled-db
    path fires 0%), but the doc's calibration chapter (RN_eff = RN x bin)
    assumes the convention is right. Task: dedicated flat-pair photon
    transfer sequence at bin1 AND bin2 on the same camera, one session.

B4. VERIFY-MAG-LIMIT BENCHMARK - the doc repeats the historical result
    "verify_mag_limit=14 as reliable as 16 and ~28% faster". Cheap re-check
    post-REORG on one archived draft (blind solve both settings, compare
    wall time + verification outcome); if it drifted, soften the doc line.

B5. BRIGHT-LIMIT FIGURE - doc states practical wide-rig bright limit
    "~G 9-10 at 60 s" (limits chapter). Confirm against saturation-zone
    statistics from existing wide drafts; adjust wording if data says
    otherwise.

## C. Verify on SYNTHETIC data (Claude can generate these)

C1. K2-FIT-RECOVERY (pre-validation of NIGHT_FIT before B2 exists):
    inject a known k'' into synthetic comp mag_inst series (given airmass
    curve + BP-RP distribution + realistic noise from the error model),
    run the v2 fit, assert recovery within consistency gates across a
    parameter sweep (k'' in 0..0.08, color spreads, noise levels). Proves
    the fitter itself; the real-night blocker then only concerns data.

C2. PSF-DEBLEND-INJECTION: synthetic frames with pairs at separations
    0.6..2.5 FWHM and dmag 0..4, known fluxes; run grouped ePSF fit and
    the neighbor-sub path; assert flux recovery and that every guard
    (refuse_sep, overmag, undermag, residual RMS, centroid shift) trips
    exactly where designed. Gives PSF enablement a machine-checkable
    regression suite independent of the Newton draft.

C3. DILUTION-CORRECTNESS: synthetic star + injected catalog neighbors;
    compare dilution.py catalog prediction D against the actual flux ratio
    in the synthetic aperture across separations/mag deltas; documents the
    prediction error envelope of the GS11 model (doc currently calls it
    "prediction, not measurement" - quantify how good).

C4. FLEMING-COMPLETENESS: synthetic field with known input magnitudes ->
    DAO detect -> dao_reconcile fit; assert recovered G_lim_50/90 vs truth.
    Closes the loop on DAO-RECONCILE (D1) methodology.

## D. Existing ROADMAP items the doc leans on (no new action, listed for order)

D1. DAO-RECONCILE completion (interpretation on dense fields; resolve the
    truncated "3.5% Gaia->DAO" figure). Doc ch 8.3 written to current state.
D2. BULK-2 optional pass (~98 recorded-but-not-applied except dispositions).
D3. APCORR-MIXEDFRAME (COG mixed-frame systematic) - doc states COG OFF with
    this as the reason; task already documented on ROADMAP.

## E. Documentation polish (optional, later)

E1. Replace the illustrative numbers in the "one night walkthrough" chapter
    with real numbers extracted from the anchor draft_435 run (authentic
    worked example; needs a small extraction script over the anchor outputs).
E2. Cross-check the FLOW quick-reference parameter table against the
    parameter handbook for wording consistency (parity ritual extension).
E3. If Milan wants the 50+ page target strictly: next expansion candidates
    are (a) real screenshots/figures from a report (needs image embedding in
    builder), (b) per-chapter "how to read this in the UI" walk-throughs,
    (c) an expanded worked example per rig. Content-first; no filler.
