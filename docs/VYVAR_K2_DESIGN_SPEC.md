# VYVAR -- K2: Band-aware second-order extinction (design spec, v1.0 DRAFT)

Status: **v1.1 -- Cursor live-tree review incorporated (2026-07-07); ready for implementation.**
v1.1 changes: defaults table -> converter module with explicit slope direction
(k2_bprp = k''_native * d(C_native)/d(C_bprp); review's division corrected to
multiplication); OSC tri-colour policy row (CT standard, k2 none); k2_mode declared
independent of apply_color_term; Section 4 marked target-vs-HEAD with duplicate-list
retirement; Section 5 dual-path (photometry_core + method_lc_output) and k''-before-c1-fit
ordering; CITATIONS additions (Smith 2002, Jordi 2010, Fukui 2016).
Decisions: Q1 `k2_mode=literature` default (correction ON for standard filters);
Q2 v1 scope = band_classify wiring + CV/CR flip + literature k'' (fit path + pre-gate = v2);
Q3 draft_425 validation rerun under snapshot discipline.
Date: 2026-07-07.
Grounding: parked band-aware k'' design (ROADMAP IN-FLIGHT section); `band_classify.py`
(`fe9b375`, shipped additive, not wired); today's measurement campaign
(`CURSOR_RESULT_k2_diag.md`, `_427_rerun.md`, `_k2_sigma_fix.md`, `_k2_fit427.md`,
`_k2_fit_verify.md`, `_k2_fit427_v2.md`); literature (Section 3).

---

## 1. What today's campaign established (evidence, not assumption)

1. **No local dataset can currently fit k''.** 425 BVR: dX=0.014 -> k''*C*X_bar degenerate
   with the colour term (only k''*C*dX identifiable, ~0.6 mmag vs 6 mmag noise). 426:
   monotonic X(t) (time aliasing unbreakable) + low leverage. 427: flux-based comp residual
   floor 71-89 mmag (Boyden flat/flip systematics class, June V0454 diagnosis) >> the
   Smith-scale signal; fitted values fail sign/tertile/arc consistency.
2. **Bad frames fake k''.** Ungated 427 r: k''=+56 mmag (6 sigma_boot) vs gated -1 mmag.
   Any k'' fit MUST run on QC-filtered frames.
3. **Fits must use flux-derived magnitudes.** Proc CSV `mag` = Gaia catalog G (constant per
   comp); science flows through `dao_flux` (`read_flux_from_csv`, photometry_core.py:1291).
4. **k'' and CT are entangled.** An unfitted k'' biases a per-night fitted colour term by
   -k''*X_bar (standard result; see Section 3 refs) -- so the two must be applied/fitted in
   a defined order, together, which matches the parked plan ("activate together").

## 2. Design principle

**Airmass is a CORRECTION, not a selection** (parked decision, kept). k'' is applied as a
deterministic, provenance-tagged correction whose VALUE comes from a source hierarchy:

```
NIGHT_FIT        (only when the feasibility pre-gate passes)   [v2 -- not in first ship]
  -> LITERATURE_DEFAULT  (band-keyed table, config-overridable) [v1]
    -> NONE              (clear/unfiltered/luminance/unknown)   [always for non-standard]
```

Every LC/report row carries `k2_source` in {night_fit, literature_default, none} and the
applied `k2_value`. VYVAR never silently invents a per-night coefficient it cannot defend.

## 3. Literature defaults (v1 table)

Primary source: Smith et al. 2002 (u'g'r'i'z' standard system) second-order coefficients:
u' -0.021, g' -0.016, r' -0.004, i' +0.006, z' +0.003 mag/airmass/mag(colour in native
Sloan indices). Johnson: k''_B ~ -0.03 per (B-V) (Henden & Kaitchuck 1982; AAVSO practice);
V/Rc/Ic ~ 0 (insignificant; AAVSO). Note r/z can flip positive for very red stars via TiO
(Fukui et al. 2016, K2-3d) -- defaults are population-mean values, adequate for
colour-matched comps.

**Unit convention:** VYVAR's native colour is Gaia BP-RP. Internal k'' is stored
**per BP-RP mag**. Literature values convert via the adopted colour-slope (Jordi et al.
2010; d(g-r)/d(BP-RP) ~ **1.054** at FGK anchor g-r=0.48 (Jordi Table 6; converter authoritative);
B-V slope analogous -- exact adopted slopes land in `CITATIONS.bib` + converter helper.

Default table (per BP-RP; **illustrative rounded values -- the authoritative numbers are
COMPUTED by a converter module `k2_extinction.py` from cited native-colour coefficients**;
config override `k2_defaults_bprp` wins over the converter):

**Converter contract (slope direction explicit):** each band's literature k'' is defined
per its NATIVE colour index (Smith: Sloan indices; Henden: B-V). Conversion:
`k2_bprp = k''_native * d(C_native)/d(C_bprp)` (multiplication; slopes < 1, so |k2_bprp| <
|k''_native|), with per-index slopes from Jordi et al. 2010 recorded as named constants
with the source equation. CITATIONS.bib additions: `smith2002`, `jordi2010`, `fukui2016`
(none currently present; `henden1982` exists).

| Band class token | k2 policy |
|------------------|-----------|
| u | literature (converted from Smith u': -0.021 per u'-g') |
| B | literature (converted from Henden k''_B ~ -0.03 per B-V) |
| g | literature (converted from Smith g': -0.016 per g'-r' -> ~ **-0.0169** per BP-RP @ slope 1.054) |
| V / R / Rc / Ic / I | 0.0 (insignificant; AAVSO) |
| r | literature (Smith r': -0.004 -> ~ **-0.0042** per BP-RP @ slope 1.054) |
| i | literature (Smith i': +0.006) |
| z | literature (Smith z': +0.003) |
| **OSC BLUE/GREEN/RED, TG/TB/TR** | **none** -- classify STANDARD_FILTER for CT (live-tree behavior, band_classify.py:130-132,176-177) but NO literature k'' exists for Bayer RGB bandpasses; applying Sloan/Johnson coefficients would be an uncited number ("trust in the numbers"). |
| CLEAR / CV / CR / L / UNKNOWN | none (no correction) |

**Independence note:** `k2_mode` is INDEPENDENT of `apply_color_term` (live default: CT
**off**, config.py:465). Literature k'' applies for STANDARD_FILTER regardless of whether
the CT gate (`should_apply_color_term`, comp-count/quality) passes -- the two do not flip
together.

**Implementation notes (Cursor sign-off, 2026-07-07):** (a) OSC k2=none needs a SECOND
check on the canonical filter token (`k2_none_tokens` in `k2_extinction.py`) -- band class
alone is insufficient, since `classify_photometric_band` returns STANDARD_FILTER for
BLUE/GREEN/RED/TG/TB/TR. (b) The B-band conversion needs an explicit, cited
`d(B-V)/d(BP-RP)` slope constant (Jordi et al. 2010 Johnson transformation, same pattern
as the Sloan slopes) -- named constant with the source equation in `k2_extinction.py`.

## 4. Band policy (wires `band_classify.py` -- the activation bundle)

| `classify_photometric_band` | CT (`apply_color_term=auto`) | k'' |
|------------------------------|------------------------------|-----|
| STANDARD_FILTER | eligible (existing comp-count gate `should_apply_color_term`, photometry_core.py:2823 still applies) | source hierarchy (Section 2) |
| CLEAR_UNFILTERED (incl. CV/CR flip) | **off** (behavioral flip vs legacy broadband list) | none -- tight colour-match remains the primary defense |
| LUMINANCE | off (as legacy) | none (distinct class kept for future policy) |
| UNKNOWN | fail-safe -> CLEAR path | none |

**TARGET STATE (not HEAD):** `resolve_apply_color_term` (photometry_core.py:3023 -- today
returns `_is_broadband_photometric_filter(obs_group)`, treating CV/CR as broadband,
:3009-3010) delegates to `color_term_auto_from_band(classify_photometric_band(...))`
(band_classify.py:337). Implementation must also: (a) pass `fits_filter`/`aavso_code` into
the classifier where Phase 2A has headers; (b) retire-or-delegate the duplicate filter
lists in `should_apply_color_term` (photometry_core.py:2841-2894) and
`_is_broadband_photometric_filter` to band_classify (single source, no drift). Wiring
entry point: Phase 2A group setup, photometry_core.py:6887-6909
(`resolve_apply_color_term` -> `_compute_group_color_term_fit`). The CV/CR flip activates
here (documented FLIP tests already exist, tests/test_band_classify.py:130-131) --
together with k'', per the parked plan.

## 5. Correction form and ordering

Per target t, frame f, applied at the CT stage (photometry_core.py:7654-7701), using the
same `bp_rp_comp_med` already computed there:

```
mag_k2corr_tf = mag_tf - k2 * (BPRP_t - bp_rp_comp_med) * X_f
```

**Ordering (mandatory, THREE insertion points -- the highest-risk implementation items):**

1. **Group CT fit:** `_compute_group_color_term_fit` (photometry_core.py:~3176) -- apply
   per-frame k'' to each comp's instrumental-mag array (comps flow from `dao_flux` via
   `_group_comp_mag_inst_from_proc_csvs`, :3110-3134 -- no proc-`mag` trap) **BEFORE**
   `fit_color_term_c1` (:2624-2670). Otherwise the fitted c1 absorbs k''*X_bar and the
   two double-count.
2. **Per-target, Phase 2A main path:** k'' correct the magnitude **before**
   `apply_color_term` at photometry_core.py:~7664 (insertion before, not inside, the CT
   block :7654-7701).
3. **Method/report LC builder:** `method_lc_output.py:~202-234` -- the SECOND production
   path; must receive the identical correction (shared helper, no duplicated formula).

Shared colour reference: k'' and CT use the SAME comp BP-RP median -- computed once
(definition of `apply_color_term`, photometry_core.py:2768) and reused.

Provenance: LC columns `k2_source`, `k2_value`, `k2_colour_ref` (=bp_rp_comp_med) added
alongside the existing `ct_bp_rp_target`/`ct_bp_rp_comp_med` columns (writer
photometry_core.py:3905+, column block ~4028); register the new columns as
provenance-class (not science) in `compare_photometry_science_meaningful`. PDF methods
section line; AAVSO NOTES unchanged (headroom rule from GS6b).

Differential common-mode is preserved: for a colour-matched ensemble the correction is
small by construction (tier ladder <=0.79 window bounds |BPRP - med| for comps; targets may
sit further out -- exactly the case the correction serves).

**Missing-data rules (per-row, fail-soft, provenance-honest):** a target or comp without a
finite Gaia BP-RP gets NO k'' correction and `k2_source=none` on that row (never a default
colour); a frame without a finite airmass gets no correction on that frame (count of
skipped frames logged once per obs_group). Both rules mirror how the CT stage handles the
same gaps today -- implementation verifies and reuses that behavior rather than inventing
a parallel one.

**Variable-star colour caveat:** BPRP_t is the Gaia epoch-mean colour; colour variation
through eclipse/pulsation (dC ~ 0.1-0.3) leaves a residual error ~ |k2|*dC*X ~ 1-7 mmag at
X<=2 -- below the current err model floor; documented, not corrected in v1.

## 6. Feasibility pre-gate (productionizes today's sigma_k2 machinery) -- v2 fit path

Ships as infrastructure with the fit path (v2), spec'd now so v1 lays columns/config
compatibly. NIGHT_FIT is accepted only if ALL hold:

1. **Inputs:** flux-derived Honeycutt residuals (never proc `mag`); fit frames =
   QC-clean subset computed READ-ONLY from always-on QC (`align_residual_px` in
   `alignment_report.csv` + B.2 quality metrics) -- the photometry frame set is NOT
   changed by the fit (gates stay user-controlled).
2. **Leverage:** sigma_k2_pred <= |k2_literature|/3 for the band (detectability), with
   sd(C*dX) and N from the actual night.
3. **Consistency:** colour-tertile and brightness-tertile k'' agree within 2*sigma_boot;
   if X(t) is non-monotonic, per-arc k'' agree within 2*sigma_boot; if monotonic,
   fit is REFUSED (time aliasing unbreakable) unless a future detrend-aware design says
   otherwise.
4. **Plausibility:** |k2_fit| <= 0.1 and sign/magnitude within a configurable factor
   (default 4x) of the literature default.

Any failure -> fall to LITERATURE_DEFAULT with the failure reason logged. Draft 427
becomes the permanent REFUSE regression fixture (its v2 numbers fail 3 and 4).

## 7. Config keys (register in VYVAR_PARAMS.md)

| Key | Default | UI | Notes |
|-----|---------|----|-------|
| `k2_mode` | see Section 9 Q1 | Settings (exposed) | off / literature / fit_else_literature |
| `k2_defaults_bprp` | Section 3 table | hidden | dict band->value, per BP-RP |
| `k2_ceiling` | 0.1 | hidden | hard plausibility bound |
| `k2_fit_enabled` | OFF (v1) | hidden | v2 flips with pre-gate |
| `k2_fit_min_detectability` | 3.0 | hidden | pre-gate item 2 |
| `k2_fit_consistency_sigma` | 2.0 | hidden | pre-gate item 3 |
| `k2_fit_lit_factor` | 4.0 | hidden | pre-gate item 4 |

## 8. Validation plan (Definition of Done, v1 activation bundle)

1. **draft_424 (NoFilter, home rig):** CLEAR path -- CT off, k'' none;
   **byte-identical** photometry to pre-activation baseline (the flip must be a no-op
   where legacy already had CT off). CV/CR-class obs_groups covered by unit tests
   (behavioral flip is intended, do-no-harm exempt there by design -- decision recorded).
2. **draft_425 (BVR, Newton) -- snapshot first, then rerun:** STANDARD path;
   k''=literature applied; validate (a) science-meaningful compare vs snapshot shows ONLY
   the k'' + CT-policy deltas, quantified per band; (b) with dX=0.014 the per-frame
   variable part is ~0 -- the observable effect is a colour-dependent ZP shift; report its
   size vs the 0.79 comp window; (c) `k2_source=literature_default` provenance present.
3. **draft_427:** Sloan STANDARD path with literature defaults; pre-gate (when v2 lands)
   REFUSES fit -- regression fixture.
4. Unit tests: band routing matrix (all four classes x CT x k''), ordering test (k'' then
   CT; no double-count vs analytic), variable-colour caveat bound, defaults override.
5. Suite green, ruff clean, PARAMS parity, 0 PDF overflow; JOURNAL + DECISIONS entries.

## 9. Design forks -- RESOLVED (Milan, 2026-07-07)

- **Q1 -- v1 default `k2_mode` = `literature`.** Correction ON for standard filters
  immediately: values are stable population means, the effect is small for colour-matched
  comps, and provenance is explicit. Matches AAVSO practice of applying mean k''_B without
  per-night fits.
- **Q2 -- v1 scope confirmed:** activation bundle = band_classify wiring + CV/CR flip + k''
  literature path. Fit path + feasibility pre-gate = **v2** (machinery exists in sandbox;
  productionize after a feasible dataset exists per the refined K2-DATA-BLOCKER).
- **Q3 -- draft_425 validation rerun under snapshot discipline** (as draft_427:
  full derived-artifact snapshot + manifest before rerun).

## 10. New/updated ledger items from today (land with the docs commit)

| ID | Sev | Item |
|----|-----|------|
| PROV-HEADLESS | MED | `merge_photometry_pipeline_meta` writes no `git_hash`/`config_snapshot` on any path; TODO-PIPELINE-VERSIONING not on HEAD. Git archaeology: regression vs never-wired; fix after diagnosis. |
| PROC-MAG-NAMING | LOW | proc CSV `mag` column holds Gaia catalog G (constant per star) -- rename or document; sandbox harnesses must use `dao_flux` (PROCESS note). |
| K2-DATA-BLOCKER (refined) | -- | k'' NIGHT_FIT needs a photometrically calibrated filtered draft with dX >= ~0.3 and comp residual floor << 15 mmag; home rig with flats qualifies; Boyden blocked on flat/flip systematics (existing flip-aware comp selection candidate). |
