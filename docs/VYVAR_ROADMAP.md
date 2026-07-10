# VYVAR — Roadmap (open work)

Single source of truth for **open** tasks. Closed work lives in `VYVAR_JOURNAL.md`;
durable rationale in `VYVAR_DECISIONS.md`; current architecture in `VYVAR_STATE.md`.

Reconciled against the full development log on **2026-06-09** — de-duplicated, and
stale-closed items removed (e.g. GS6b had been listed as open in a side register but was
closed 2026-05-20). **Session close 2026-06-25:** band-aware k'' design parked; blockers + deferred
findings recorded below.

Priority legend: **HIGH / MEDIUM / LOW / FUTURE**. Each item is a short status, not a history.

---

## IN-FLIGHT / ACTIVATED — band-aware k'' (second-order extinction) v1

**Status:** **ACTIVATED v1 (2026-07-07).** Spec: `docs/VYVAR_K2_DESIGN_SPEC.md`. Code: `k2_extinction.py`,
`k2_mode=literature` default, band_classify CT wiring + CV/CR flip, LC provenance columns.
**NIGHT_FIT = v2 open** (`k2_fit_enabled` OFF; pre-gate spec'd, sandbox machinery in `tmp/k2_*`).

**Rejected (do not port):** comp-select **grow-redesign** — population validation showed ~**45%**
regressions; sandbox-only; **nothing ported to production**.

**Policy (shipped):** airmass = **CORRECTION not selection**; literature k'' for STANDARD_FILTER
(excluding OSC RGB tokens); CLEAR/CV/CR/L/UNKNOWN → k2 none; CV/CR CT flip live.

**Shipped additive precursor:** `band_classify.py` + tests (`fe9b375`); now wired to
`resolve_apply_color_term`.

### K2 ledger (2026-07-07)

| ID | Sev | Item |
|----|-----|------|
| **PROV-HEADLESS** | — | **FIXED** (`e7ce7ea`): `pipeline_meta.json` `provenance` block (`git_hash`, `git_dirty`, full `config_snapshot`, `stamped_at_utc`, `entry_point`); last-writer-wins at `run_phase2a` + `generate_masterstar_and_catalog`. |
| **PROC-MAG-NAMING** | — | **FIXED** (`0913665`, documented): proc CSV `mag` = Gaia catalog G (constant per star at match time); science uses `dao_flux`. Docstring + `VYVAR_PIPELINE_CZ.md` schema note; PROCESS dao_flux rule verified. |
| **K2-DATA-BLOCKER** | — | NIGHT_FIT needs calibrated filtered draft, dX ≥ ~0.3, comp residual floor ≪ 15 mmag. Home rig + flats qualifies; Boyden blocked on flat/flip systematics. **Supersedes** old blockers (filtered draft + FILTER strings — both satisfied 2026-07-07). |
| **K2-SLOPE-TRACE** | — | **FIXED** (`8c44b71`): GR slope 0.859→1.054 (Jordi 2010 Table 6 inverse, FGK g-r=0.48); k2_g≈−0.0169, k2_r≈−0.0042. UG retained 1.091 as documented exception → **K2-SLOPE-UG** (FUTURE). |
| **K2-SLOPE-UG** | FUTURE | No Jordi u-g row; spec-anchored UG slope 1.091 retained pending citable source. |

---

## IN-FLIGHT / PARKED — band-aware k'' (second-order extinction) [SUPERSEDED — see ACTIVATED v1 above]

**Status:** design parked; first code piece shipped additive (`fe9b375`); production CT rewiring **PENDING**.

**Rejected (do not port):** comp-select **grow-redesign** — population validation showed ~**45%**
regressions; sandbox-only (`sandbox/comp_grow*.py`, `tmp/` metrics); **nothing ported to production**.

**Correct direction (when unblocked):** treat **airmass as CORRECTION, not selection** — derive k''
from **constant comps** (signal-safe); **band-aware** policy via `band_classify.py`:
- **STANDARD_FILTER** (B/V/R/etc.): k'' is reliable; apply second-order extinction correction.
- **CLEAR_UNFILTERED** / unfiltered: tight colour-match is primary; **do not** rely on k''.
- **LUMINANCE (L):** own class; distinct from clear for future policy.
- CV/CR → CLEAR_UNFILTERED (physically clear-transformed); flip activates **with** CT rewiring, not alone.

**Shipped additive (not wired):** `band_classify.py` + 52 tests (`fe9b375`); consolidates legacy
`_is_nofilter_obs_group` / broadband CT auto. Ledger: `VYVAR_AUDIT_LEDGER.md` BAND-DETECT (`5d6801c`).

### REAL BLOCKERS — resume band-aware k'' (Milan-side data, not code) [SATISFIED 2026-07-07 for v1 literature path]

1. **Filtered draft (V/B/R) for validation.** k'' only does real work on filtered data. Locally
   **NoFilter** draft **424** exists; **filtered drafts now exist on disk (Milan, 2026-07-09) —
   data availability to be re-verified when NIGHT_FIT v2 is scheduled** (do not unblock/schedule here).
2. **Newton/Brno literal FITS FILTER strings.** Needed so those rigs get k'' routing instead of
   fail-safe clear (UNKNOWN → CLEAR). Capture on dev PC / rigs: `FILTER`, `FILT`, `FILTER1`, `INSTFILT`
   on Chi_and_H **B/V/R/L** frames and Brno **`r_60_4`**; SQL on dev DB:
   `SELECT DISTINCT FILTER, ID_EQUIPMENTS FROM FITS_HEADER_CACHE`.

**Activate together (code, when data unblocked):** wire `band_classify` into production CT gating +
CV/CR→clear behavioral flip + band-aware k'' correction path.

---

## Deferred findings (carry forward — none blocking)

| ID | Sev | Notes |
|----|-----|-------|
| **F-BINGAIN-1** | MED | **LATENT (not live).** Stage A diag 2026-07-07: draft_424 bin2 **12/12** frames gain via `header_index_mapped` **3.17 e-/ADU**; scaled-db (`exponent=2`) path **0%**. **A4 (2026-07-09):** draft_426 bin4 check-star χ²=0.04–0.33; header GAIN=12.48 (session truth); σ_ratio≈1.13 → χ²_pred≈0.78. **SESSION-CLOSE-0709:** FITS headers unambiguously identify **eq4 C5A-150M/IMX411** (INSTRUME + 3552×2664 @ bin4 → 14208×10656); OBS_DRAFT already eq4 — **gain accounting EXCLUDED** as equipment-attribution error. Corrected-gain hypothesis (gain=16) **worsens** χ². Open candidates: **RN scaling mode** (RN=14.08 = 4× bin1 — sum-mode assumption to verify), **sky/area accounting** in binned pixels, **stack hypothesis** (IMAGETYP=OBJECT, NCOMBINE absent on sampled frames — not stacked subs). Forensics: `scripts/bin4_sigma_forensics.py`, `tmp/sigma_budget/bin4_sigma_forensics.json`. |
| **F-EXCEPT-TIER1** | — | **CLOSED (2026-07-08)** — 40 fixes + EXCEPT-BULK; census `docs/VYVAR_EXCEPT_CENSUS.md` all **625 EVIDENCE**. |
| **GAIA-ID-FLOAT-GUARD** | MED | **CLOSED** (verified 2x clone + live tree, 2026-07-07). |
| **F-HOWELL-3** | MED/HIGH | **FIXED (Stage C)** | `sky_adu_per_px_annulus`; draft_424 science byte-identical |
| **F-BJD-1** | LOW | **FIXED (Stage D)** | `time_base` LC column; numeric times unchanged |
| **F-AIRMASS-CITE** | LOW | **FIXED** (2026-07-07) | Kasten & Young (1989) attribution |
| **G7-F003c** | — | **FIXED** (`80aab21`): PDF reads `provenance.config_snapshot` from `pipeline_meta.json`; live `AppConfig` fallback footer-annotated. |
| **EQUIP-BINNING-ASYM** | LOW | Asymmetric binning (`XBINNING ≠ YBINNING`) warns but does not scale gain/RN; all current rigs symmetric. |
| **TIER1-OBSLOC-ZERO** | — | **FIXED** (`166cbf4`): `resolve_site` null-island guard (|lat|,|lon| < 0.01° → UNRESOLVED); airmass refuses, BJD `JD_FALLBACK`. |
| **TIER1-UI-DEBT** | LOW | 38 SAFE UI/plotly broad-except `pass` sites — cosmetic; subset of T3-UI in `docs/VYVAR_EXCEPT_CENSUS.md`. |
| **HRD-PLOT-TUPLE** | — | **FIXED** (EXCEPT-BATCH-S0): `sqlite3.Row` iterates values not keys (`hrd_analysis.py:113`); PDF now emits HRD page or explicit unavailable placeholder (`photometry_report.py`). |
| **299-defensive cluster** | — | **CLOSED (2026-07-08)** — EXCEPT batch complete; ledger `docs/VYVAR_EXCEPT_CENSUS.md`. |
| **PROV-HEADLESS** | — | **FIXED** (`e7ce7ea`) — see K2 ledger row. |
| **PROC-MAG-NAMING** | — | **FIXED** (`0913665`) — documented, not renamed. |
| **K2-DATA-BLOCKER** | — | NIGHT_FIT needs dX ≥ ~0.3, comp floor ≪ 15 mmag; Boyden blocked on flats/flip. |

---

## PUBLICATION — VYVAR methods paper + software DOI (workstream, opened 2026-07-08)

**Scheduling: LAST** — after functional work completes (Milan, 2026-07-09). Do not delete PUB items.

**Status:** venue research **DONE** (2026-07-08, Claude web research); venue decision **PENDING**
(Milan). Two-track strategy recommended: **JAAVSO** methods paper + **JOSS** software DOI;
**OEJV** (Masaryk Univ. Brno — SQUADRA home venue) for subsequent science papers citing the
pipeline paper; **Astronomy & Computing** as a possible expanded v2 paper later.

**Venue matrix (researched 2026-07-08):**

| Venue | Fit | Cost | Notes |
|-------|-----|------|-------|
| JAAVSO | methods paper, observer audience, ADS-indexed | free for AAVSO members ($100/page non-members) | 2 issues/yr, window closes 6 wks before Jun 15 / Dec 15; Word or LaTeX; journal@aavso.org |
| JOSS | software itself gets citable DOI; review = code+docs+tests via public GitHub issues | free | REQUIRES: OSI license, public repo, English README/docs; paper ~1000 words, must NOT contain science results; 2026 criteria value sustained commit history (VYVAR: months of history, 590+ tests) |
| OEJV | science-results papers using VYVAR; SQUADRA/MUNI home venue | free | PDF to oejv@physics.muni.cz; referee report ≤1 month; arXiv-mirrored, Crossref-indexed |
| Astronomy & Computing | professional pipeline-paper league (PHOTOMETRYPIPELINE, Photometry+) | OA $2220 or subscription route free | candidate for expanded v2 paper (post NIGHT-FIT v2) |

**Sub-items:**

- [ ] **PUB-VENUE** (Milan decision): confirm JAAVSO+JOSS two-track; optionally ask OEJV
      editors informally whether a methods paper is in scope (home-field option).
- [ ] **PUB-OUTLINE**: Claude drafts paper outline (Intro / Architecture / Photometric methods /
      QA & trust / Validation / Comparison with existing tools / Summary) — first writing task.
- [ ] **PUB-FIGS**: figure set from real drafts (RMS-vs-mag, example LC (BO CVn), comp-QA panel,
      architecture diagram, CAL-DIAG/k2 provenance example) — Cursor generates, Milan approves.
- [ ] **PUB-JOSS-PREREQS** (only if JOSS confirmed): OSI license file in repo, repo public,
      English README + install/user docs, tagged release + Zenodo DOI at acceptance.
- [ ] **PUB-POLICY**: check chosen venue's AI-assisted-writing policy before submission;
      acknowledge per policy (Milan authorial decision).
- [ ] **PUB-VALIDATION-SECTION**: existing threads feed here — **TODO-SEP-XVAL** longitudinal
      record ("mission + paper Validation section") and the single-night canonical unit
      (**TODO-GS8** / DECISIONS product-scope boundary) become the Validation section's backbone.

**Depends on / feeds:** nothing blocks writing the outline now; PUB-FIGS wants a
representative filtered draft (425 BVR qualifies today); JOSS track blocked only on repo
licensing decision.

---

## IN-FLIGHT — CAL-DIAG (calibration-time radiometry gate) — **IMPLEMENTED / VALIDATED (2026-07-07)**

**Agreed 2026-07-07 (Milan + Claude).** Spec: `docs/VYVAR_CAL_DIAG_SPEC.md` v1.1.
Grounding: `CURSOR_RESULT_caldiag_flow.md`. Implementation: `CURSOR_RESULT_caldiag_impl.md`.

Camera-agnostic calibration-time diagnostic gate (shipped):

1. **Post-dark-subtraction sky-median sanity** — median > 0 and plausible (reject negative/absurd
   calibrated sky).
2. **Pre-subtraction convention cross-check** — `median(light)` vs `median(resampled dark)`; detect
   SUM vs MEAN dark-resample mismatch with loud auto-correction (SUM -> MEAN retry) or fail-closed abort.
3. **Provenance** — `VY_DKRSMP` = `SUM` | `MEAN_AUTOCORRECTED` | `PASSTHROUGH`; `archive/<draft>/cal_diag.json`.

**Validation:** 14 pytest gate tests; full `pytest tests/` green; draft_424 regression PASS (150/150
`VY_DKRSMP=SUM`, 0 WARN/FAIL, calibrated arrays + photometry science byte-identical).

**Related ledger (NOT part of this gate — remain open):**

| ID | Sev | Summary |
|----|-----|---------|
| **CAL-AGE-CLOCK** | — | **FIXED**: import scan + get_calibration_status use `resolve_master_age` (header `VY_CDATE`→`DATE-OBS`→`DATEOBS`, naive→UTC); mtime fallback warns once per scan. |
| **RN-HEADER-NONE** | — | **FIXED** (`1830527`): SNR precompute passes MASTERSTAR header to `resolve_read_noise` (bin2 RN×bin parity with Phase 2A). draft_424 validated: max aperture shift 2.2%, LC byte-identical. |
| **CAL-PASSTHRU-DEAD** | — | **FIXED** (`21c20e3`): removed `allow_passthrough` from `get_processed_master` (tests-only dead branch). |

Flat-norm Stage 2 and other ROADMAP calibration items unchanged (D1/D2/D3 codify current behavior).

---

## IN-FLIGHT — EXCEPT batch (silent broad-except census) — **CLOSED (2026-07-08)**

**Ledger:** `docs/VYVAR_EXCEPT_CENSUS.md` — **625 EVIDENCE** (all tranches); **40 fix-batch** sites
surfaced; remainder **EXCEPT-BULK** processed. Scanner: `scripts/_except_census_scan.py`.

| Tranche | Scope | Sites | Status |
|---------|-------|------:|--------|
| 1 | science kernel | 144 | EVIDENCE + FIX-1 + EXC-0626 |
| 2 | pipeline.py | 170 | EVIDENCE + FIX-2 |
| 3+3b | astrometry/import/database | 98 | EVIDENCE + FIX-3 |
| 4+4b | report/export/UI + remaining | 213 | EVIDENCE + FIX-4 + BULK |
| bulk | delete-dead / log / comment / narrow | remainder | **DONE** |

**Fixes landed:** 40 (FIX-1..4 + companions/closeouts). **Gate: 623 passed**, 15 skipped.
**Empirical check:** run `except_fix_counters` snapshot on next healthy draft — expect all zeros.

**Next:** none (batch closed). See JOURNAL for history pointer.

---

### Phase-1 graceful comp degradation — DONE (2026-06-16)

Validated matrix `matrix_20260616_164157.json`. Spec: `docs/VYVAR_COMP_DEGRADATION_SPEC.md`.
Known-issue **(b)** deferred to immediate next item (comp_rms gate authoritative for N_good).

---

### Simple differential + reporting + trust cleanup — DONE (2026-06-16)

- **Workstream A** (defaults + Phase-1 tier-ladder selector): DoD-A PASS 2026-06-15; commit `2a8355b`.
- **Workstream B** (reporting column grounded decision): DoD-B PASS 2026-06-15; `apply_reporting_postprocess`.
- **draft_409 Fixes 1-3** (trust/consistency): comp stability on per-frame ensemble residual;
  measured aperture + observed-band SNR sizing; `lc_rms (OOE)` for variables. Cross-validated vs
  SIPS on V0612 (eclipse + shared frame anomaly at ~JD 2461200.385).

Grounded specs: `VYVAR_DECISION_GROUNDING_RULE.md`, `VYVAR_REPORTING_COLUMN_GROUNDED_DECISION.md`,
`VYVAR_CANONICAL_COMBINATION_LOGIC.md`, `VYVAR_SIGMA_BUDGET_SPEC.md` (PARKED).

---

### Workstream B — DONE (DoD-B PASS 2026-06-15)

Tier-2 comp-ensemble extinction k **parked** (wide-field delta-airmass; ROADMAP).

Evidence: ``tmp/phase11/dod_b_workstream_b.json``; ``apply_reporting_postprocess`` in
``photometry_core.py``.

---

### Round 1 — four known fixes — DONE (2026-06-17)

Verified g-only on draft_413. A-durable (MP reload robustness), B-cap (spatial-first
variable_targets; comp-purity coupling Milan-accepted), measurable completeness gate, NoDetections
log-flood summary. Writeup: `CURSOR_RESULT_round1.md`; decisions in `VYVAR_DECISIONS.md`.

### Round 2 — frame-quality gate (B.2) DONE; aperture-skirt (B.1) REFUTED (2026-06-17)

Diagnosed and measured in isolation on draft_413 g; writeup `CURSOR_RESULT_round2.md`. Push gated on Milan.
- **B.1 aperture-skirt / COG — REFUTED, not implemented.** COG confirms 5 px aperture captures only
  EE=0.65, but widening to the plateau does not reduce differential scatter (flat 24→27 mmag) and
  adaptive sizing is worse; the skirt swing is common-mode (cancelled differentially). Decision in
  `VYVAR_DECISIONS.md`.
- **B.2 transparency frame-quality gate — DONE (default-OFF).** `frame_quality_gate_enabled` (+
  `ratio_k`, `fwhm_factor`, `min_keep_frames`) rejects PSF-collapsed frames via `flux_large/flux`
  robust-outlier + FWHM guard; cuts bright-target LC scatter median −257 mmag on draft_413 g.

**Round 2 open follow-ups (HIGH):**
- **OUTSTANDING — UI-VYVAR live test of A-durable** (the one validation still pending from this
  session). In the Streamlit UI, reach the alignment MP stage, save a watched `.py` to trigger a
  mid-run reload, and confirm **no PicklingError** (MP continues via fresh-attr dispatch, or the
  single-process fallback engages). Headless never reproduces it — must be exercised in the UI.
- **Structural comp / check-star yield = next lever for GREEN on bright/sparse fields** — post-B.2,
  trust RED is **structural** (check-star gaps, thin comps, colour-term-off), **not** LC scatter
  (the gate already fixed scatter, median −257 mmag). `green_min=3` is literature-grounded → improve
  comp/check-star **yield**; do NOT relax the gate.
- **Flip-aware comp selection (candidate)** — when a meridian flip is detected, prefer **near-target**
  comps (per-side normalization as an alternative). Root cause = uncorrected flat-field on non-cal
  data under the 180° p→−p mapping; evidence = V0454 flip diagnostic (±0.1 mag position-dependent
  post-flip step; `docs/round2_figs/v0454_flip_diag.png`, DECISIONS). Connects to the comp follow-up.
- **B.2 frame-quality gate Phase-0+1 extension (future)** — gate collapsed frames at
  **comp-selection** time too, not only Phase 2A; may recover comps and feed the structural follow-up.

### run-414 fixes — Fix A DONE; Fix B DONE; Fix C DIAGNOSED → N/A; control-point-cap perf ticket OPEN (2026-06-18)

From the run-414 V0454 diagnostic (`CURSOR_RESULT_414_diag.md`) + the C1 diagnostic
(`CURSOR_RESULT_fixC_diag.md`). Fix A + Fix B pushed to origin/main 2026-06-18 (Milan-authorized).
- **Fix A — per-point `err` model — DONE (default).** `err` term-3 was std of comp instrumental mags
  (brightness spread, ~0.58 mag floor); now per-frame ensemble-ZP residual SEM. Decisions/JOURNAL.
- **Fix B — reject-on-alignment-residual gate — DONE (default-OFF), PERMANENT quality gate.**
  `frame_align_residual_gate_enabled` (+ `max_frac`=0.25 of aperture radius, `min_keep_frames`).
  Per-frame `align_residual_px` recorded always-on in `alignment_report.csv`. On run-414 g, ON drops the
  13 phase_correlation + 1 mis-aligned astroalign frame; OFF byte-identical. C1 confirmed Fix B + B.2 are
  the **correct permanent handling** of these frames (not a stop-gap awaiting Fix C).
- **Fix C — dense-field alignment recovery — DIAGNOSED → NOT APPLICABLE (2026-06-18, C1).** The premise
  ("good data, only alignment failed → recoverable") is **refuted**: the 14 frames are **PSF/FWHM-bloated**
  (median FWHM 8.60 px = 1.85× the good 4.64 px; concentration 13.1 vs 1.65; **corr(FWHM,residual)=0.95**).
  Bloated-donut centroid noise (~2.4 px) breaks astroalign (misalignment = symptom) and is what Fix-B/B.2
  measure. **Not recoverable to sub-px** (centroid floor ~2.4 px > 1.37 px gate; cap50→3/14, WCS absent
  0/162, translation-refine inapplicable). Likely **late-night focus drift on the defocused rig** (FWHM
  bloat, not a flux drop); post-flip-half-not-refocused is an observer question. **Closed** — recovery
  would be useless and risky. `CURSOR_RESULT_fixC_diag.md`, `tmp/phaseC1/fixC_root_cause.png`.
- **NEW — dense-field astroalign control-point cap (perf/robustness, OPEN, MED).** *Separate from Fix-C
  recovery.* astroalign at `mcp≈200` on dense fields is ~654 s/frame (and still fails); cap to ~50
  (astroalign's design point) → ~3–10 s. Two shapes: (i) **additive recovery rung** — cap tried only
  after the current attempts fail, before `phase_correlation` (hook: `vyvar_alignment_frame.`
  `_alignment_compute_one_frame`, between the `_attempts` loop end and the phase_correlation block);
  keeps the 147 byte-identical, does **not** fix slowness. (ii) **primary cap** — fixes slowness but
  changes the 147 transforms → requires a **cross-rig regression (home + narrow rigs)** before adoption.
  **Defer until cross-rig data is available.** **Watch-item:** mildly-bloated near-threshold frames kept
  by the gate (e.g. g_0231, FWHM 1.10×) — likely benign (differential photometry cancels common-mode
  FWHM), watch the LC near the good→bad transition.

---

## NEXT SESSION — open items

**Resume here when Milan data unblocks k'':** filtered V/B/R draft + Newton/Brno FITS FILTER capture
(see IN-FLIGHT section above). Until then, other HIGH items below remain valid but k'' is the designed
next calibration lever.

0. **Phase-1b — comp_rms gate authoritative for N_good — gate-authority DONE (2026-06-16).** RMS
   fallback no longer relaxes above `max_comp_rms` (the `0.15` step removed); auto-routing counts
   gate-passers via `_count_gate_passing_comps` instead of raw `len(result)`. Matrix re-run `185831`:
   SS Cam flips default->sparse_fallback; V0612 / BO CVn / V0842 Her unchanged. No threshold re-tuning.
   **STILL OPEN:** the SS Cam trust **band** (it landed YELLOW; check scatter 0.043 < 0.05) is
   **UNRESOLVED** — see item 1. Do not treat RED as the answer.

1. **Phase-2 comp degradation — sparse-comp sanity (DIAGNOSTICS DONE, gate redesign PROPOSED).** Sparse
   comp_rms is **field-wide-scale** (~0.41–1.0 mag headline on SS Cam/V0611) — offset structure that
   **~95% cancels** in differential (cancellation_factor ~0.02–0.08 on draft_426). **Temporal component**
   8–12 mmag (healthy). Check scatter 0.037 mag on SS Cam g_60_4 with CI [0.032, 0.040] — healthy vs
   gate. **PROPOSED (not enacted):** sparse trust should evaluate (a) check scatter vs its CI and
   (b) the **temporal** component of comp_rms — never the field-wide headline against the 0.1 per-target
   gate (different quantities). SS Cam stays **YELLOW**; band decision **OPEN** pending Milan review.
   **Do NOT reverse-engineer RED.** Diagnostic: `scripts/sparse_comp_diag.py`
   (`tmp/sigma_budget/sparse_comp_diag.json`).

2. ~~**Sigma budget Phase A (wide rig)**~~ **DONE (2026-07-09, wide rig; Newton bin4 open).**
   Validated model: photon (+) Honeycutt ensemble SEM (+) **6.5 mmag floor**; scintillation ~2 mmag
   negligible on D=0.2 m; f_resid → 0 (variant e). Attribution (draft_424): k2-signature pooled R²≈0 —
   **k'' would recover 0.0 mmag** (colour matching works; NIGHT_FIT v2 will not reduce this floor);
   phase/PRNU strongest candidate (6.5 → 4.5 mmag); **~4.5 mmag rig constant** remains. Committed:
   `sigma_budget.py`, `scripts/chi2_sigma_gate.py`, `scripts/select_constant_calibrators.py`,
   `scripts/sigma_floor_attribution.py`, `scripts/bin4_sigma_forensics.py`, `scripts/fix_telescope_diameter.py`,
   `scripts/fix_draft_equipment.py`. **Remaining scoped:** Newton bin4 chi2 deficit (F-BINGAIN-1).
   **Open decisions (Milan):**
   - **PROD-SIGMA-FLOOR:** add per-rig `sigma_sys` floor to production err (changes outputs → re-anchor;
     bright-star AAVSO error bars currently underestimated by floor) — separate session.
   - **SIGMA-NEWTON:** extend chi2 gate to Newton rigs once bin4 sigma is fixed.
   Blocks TODO-GS8 + TODO-MULTISET for IVW flip. **`delta_mag` flux-sum canonical until Newton gate passes.**

3. **EXTERNAL-XVAL — external validation campaign (MEDIUM).** VYVAR vs external tools (AstroImageJ
   et al.); includes the FWHM external-validation claim (confirm/deny "true ~7.7–8.6 px" before any
   factor retune). Feeds **TODO-SEP-XVAL** ledger and the future publication Validation section.
   Supersedes standalone "FWHM external validation" item.

4. **Frame-level CR / bad-pixel rejection (MEDIUM)** — shared V0612 anomaly (~JD 2461200.385) is a
   single-frame artifact matching SIPS; consider pixel-level CR rejection **before** photometry
   (not LC sigma-clip, which must preserve real eclipse features).

5. **Source_id exact-match audit (MEDIUM)** — verify loose `contains()` matching is confined to
   throwaway harnesses, NOT production (19-digit exact match; draft_409 audit neighbor contamination
   tied to Decimal-precision discipline in `build_gaia_catalog`).

6. **Optional fresh byte-identity anchor (LOW)** — cut new SHA + recipe on committed simple-diff path
   for regression on top of SIPS/AIJ empirical validation (Milan call).

7. **Brno / Newton characterization gate (finish)** — `g_60_4` solves on **production path**
   (draft_400: 75.5% brightest-N, WCS persists). **Per-set fault isolation shipped** (2026-06-14) —
   one bad filter no longer aborts RUN VYVAR; surviving sets reach photometry. **Open:** Milan
   **draft_401 UI sign-off + overlay**; Brno **r/i/z** end-to-end (draft_402: `r` hint_sep reject,
   `z` blocked by whole-run abort — now unblocked for survivors); gate `_brno_check` tail fixed
   (draft_400 / skip). **Plate-solver TASK 2 shipped (2026-06-14):** catalog-recovery gate +
   hint-as-prior; pending Milan `r` overlay + anchor/home-rig regression re-run.
8. **TODO-MULTISET** — per-telescope-set config (wide vs fine optics); blocks clean multi-rig
   production and crowding gating per rig.
9. ~~**Short-baseline LC quality (#3)**~~ **DONE (2026-06-10)** — `short_baseline` terminal class,
   config keys, trust YELLOW non-escalating, exportable. ~~Finding E~~ **re-checked (2026-06-11)**.
10. **Phase C catalog rebuild** — DR3 full-sky build completes on existing schema (G<=17.5).
   GAIA-1/GAIA-2 columns deferred to DR4 (~Dec 2026) -- see DECISIONS.
11. ~~**Chi_and_H baseline re-cut**~~ **DONE (2026-06-11)** — full zaloha anchor
   (core `3f7c9e7a...`, full `d5b72d08...`; draft_000387 re-cut ×2); completeness
   gate in `night_run.py`. See STATE + RUNBOOK.
12. ~~**Trust Findings A/B + CS-1..4 + comp trust floor (Option B)**~~ **DONE (2026-06-11)** —
   specs under `docs/`; trust baseline 1382/106 on draft_387 at `comp_trust_min_comps=5`.
13. ~~**Broad-except hygiene (BLE001/E722 regression guard)**~~ **DONE (2026-06-11)** — see
   DECISIONS + `pyproject.toml` / pre-commit / `tests/test_ble001_regression.py`.
14. ~~**DEV-PROCESS-A — JSON pass/fail validation ledger**~~ **DONE (2026-07-08)** —
    `validation/VYVAR_VALIDATION_LEDGER.json` + `tests/test_validation_ledger.py` (frozen
    required-ID guard). Rules: agents edit only `passes` / `last_verified` / `commit` / `notes`.
15. ~~**DEV-PROCESS-B — session-start baseline-check script**~~ **DONE (2026-07-08)** —
    `scripts/session_baseline_check.py` (`--fast` / `--full`); `--full` re-verifies draft_424
    science anchor + `except_fix_counters` zero-check; documented in RUNBOOK + CLAUDE_OPERATING_PRINCIPLES.
16. **INSTALL-MANUAL** — user install + catalog build guide; Lenovo T460 + TODO-LIB.
13. ~~**Per-frame proc export perf (DAO pre-filter + Moffat gate)**~~ **DONE (2026-06-12)** —
   `_proc_drop_unmatched_dao_rows` before aperture/PSF; Moffat gated on `_run_epsf` only; ~4.8× on
   `draft_000389` B_60_1 (171 → 36 s/frame). See JOURNAL + DECISIONS.
14. ~~**Sparse-only comp fallback**~~ **DONE (2026-06-11)** — default **ON**; anchor
    `3f7c9e7a` / `d5b72d08`; science-meaningful comparator for regression vs prior cut.
15. ~~**Comp-slope stability (B2+B1)**~~ **DONE (2026-06-11)** — common-mode detrend BJD sort +
    significance gate (`comp_slope_significance_k`); Honeycutt 1992 conditional. Anchor footprint
    LC-neutral on `draft_000387`.

---

## HIGH

- ~~**Sigma budget Phase A — validated per-measurement uncertainty (wide rig)**~~ **DONE (2026-07-09).**
  Howell + Honeycutt ensemble SEM + 6.5 mmag floor; scintillation negligible on wide rig; f_resid→0.
  Newton bin4 chi2 gate still open (F-BINGAIN-1). **Do not** flip ensemble combine to Broeg IVW until
  Newton gate passes. See `VYVAR_SIGMA_BUDGET_SPEC.md`, `scripts/chi2_sigma_gate.py`.

- **TODO-MULTISET — per-telescope-set config architecture.** One config per rig (wide
  Carl-Zeiss 200 mm + QHY294MM ≈ 9.77″/px vs Newton 300/1200 + C3-26000 ≈ 0.65″/px).
  Underpins per-set plate scale, aperture, and crowding gating; blocks clean multi-rig
  production.
- **TODO-GS8 — Multi-Night Global Matching + global ZP solver — FUTURE / nice-to-have
  (descoped from HIGH 2026-06-25).** Cross-night comp matching + inter-night zeropoint.
  NOT a priority: the canonical publishable unit is a single night (see DECISIONS,
  product-scope boundary). Build only if a long-baseline science case demands it.
  Dep: AAVSO validation (GS6b ✓). ~2–4 days. → **PUBLICATION** workstream.
- **APCORR-MIXEDFRAME — latent (COG default OFF); blocks enabling COG in production.** With
  `cog_aperture_correction_enabled=True`, `cog_ok=False` frames keep *uncorrected* flux
  alongside corrected ones → cross-frame step in the LC. The QA dashboard `IS_REJECTED` path
  is separate and clean (rejected frames are dropped before calibration, never enter the LC).
  Fix = drop `cog_ok=False` frames / wire the nightly `fallback_ee` / all-or-nothing per
  night. (Rationale in DECISIONS.)
- **PSF on fine-scale (Newton ≈ 0.65″/px) data.** Infrastructure is DONE and default **OFF**
  (wiring `psf_flux`→Phase 2A, adaptive selector, per-star quality + auto-fallback, spatial
  grid, grouper — all lose to aperture at 9.77″/px, correctly kept off). OPEN:
  - ~~**Crowding rule-2 bug**~~ **CLOSED (2026-06-09):** resolvable-blend->PSF (rule 2) was
    **removed** from adaptive routing (`photometry_core.py` ~5484); `is_blended` at 1.5 FWHM
    remains for crowding metrics only. Adaptive PSF uses faint-isolated rule only. NEIGHBOR-SUB
    is the blend path (not grouped PSF).
  - **TODO-PSF-NEIGHBOR-SUB** — subtract neighbour + aperture residual (gated OFF).
    **Steps 1-2a + pre-2b DONE**: A9 envelope, `psf_neighbor_sub.py`, fail-safe guards +
    `bright_close_regime` edge guard. Fine-scale draft 367: mismatch **~1.0**, HV **~83%**,
    FAIL-SILENT **0**; real crowding **sparse** (9 blended) -> **VALIDATED_FINE_SCALE_IDLE**;
    **2b deferred** until blended fine-scale field. Coarse bin2 remains **SAFE_LOW_YIELD**.
    Design: `docs/VYVAR_NEIGHBOR_SUB_DESIGN.md`; crowding: `docs/VYVAR_DRAFT367_CROWDING.md`.
  - **TODO-PSF-V3d-FINE-SCALE** — **DONE (2026-06-08)** harness `tests/validation/v3d_fine_scale.py`.
    Inject-and-recover at draft-367-like scale; PASS on accuracy/precision/calibration pillars.
    Report: `tier_v3d/v3d_fine_scale.md`. Production PSF still OFF; real-field enablement separate.
  - ~~**TODO-PSF-V3d-MIDMAG-BIAS**~~ **DONE (2026-06-09)** sky-only PSF fit weights
    (`psf_weight_mode=sky_only`; Astier 2013 / Lacroix 2025). V3d mid-mag <2%, drift sub-%.
    Report: `tier_v3d/v3d_weight_proof.md`. PSF arc ready to batch (pre-2b + V3d + sky + weights).
  - **TODO-PSF-MULTIFRAME** — multi-frame ePSF stacking (isolation part done).
  - **TODO-PSF-ASYMMETRY** — tracking-smear diagnostics (BO CVn right-tail PSF).
  - ~~**TODO-FWHM-CONSISTENCY**~~ **DONE (2026-06-09)** — `header_core_fwhm_px` in
    `masterstar_context.py`; `crowding_index._load_wcs_meta` + `psf_photometry.get_epsf_fwhm_from_context`
    prefer `VY_FWHM_GAUSS` -> `VY_FWHM_GAUSSIAN` -> `VY_FWHM`, matching aperture path
    (`pipeline.py:9206`). h & chi Per L: is_blended 77/87 -> 58/53; numeric SHA 770966c3 unchanged.
    See `docs/VYVAR_HCHIPER_CROWDING_RECOMPUTE.md`.
  - ~~**TODO-EPSF-1-FWHM-QC**~~ **DONE (2026-06-08)** — `_epsf_fwhm_native_from_profile`
    (azimuthally-binned radial profile); QC warning band [0.80, 1.25]. V3e PASS (NEW ratios
    1.038-1.049 on synthetic Moffat). Diagnostic only; numeric SHA 770966c3 unchanged. See
    `docs/VYVAR_EPSF_AUDIT.md`, `tier_v3e/v3e_epsf_fwhm.md`.
  - ~~**Realistic per-star PSF uncertainties**~~ **DONE (2026-06-09)** via sandwich variance
    (`psf_err_mode=sandwich_skyonly`; V3d P3 ~1 mag<=17). **2026-07-09 audit (HEAD f38e924):**
    four latent findings **FIXED** — PSF-ERR-DECOUPLED, PSF-AC-FALLBACK, PSF-FLAG-VESTIGIAL,
    PSF-LEGACY-SELECTOR. Enablement checklist: real-field Newton draft + ERR/AC gates DONE;
    production PSF still OFF (`psf_photometry_enabled=False`).
- **Per-frame saturation (not whole-star skip).** Whole-star `zone_flag=saturated` from the
  longest-exposure masterstar drops comps/targets with viable unsaturated frames (76 Green/49 Red
  on M67). Same "silent wrong drop" class as the cross-group MASTERSTAR bug.

## MEDIUM

- **TODO-COMP-P2P-RESIDUAL — evaluate p2p outlier on common-mode-removed residual.** The p2p
  stability check (`check_comparison_stability`) still fires on the raw comp LC; on nights with a
  shared airmass/transparency ramp, `rms_p2p` ≈ the ramp range (same methodological class as the
  pre-B2 slope bug). **Separate science-changing task:** mirror the Honeycutt common-mode removal
  before p2p scoring; own footprint + optional anchor re-cut. Comp-slope fix (B2+B1) is **DONE**
  pending Milan acceptance; p2p deferred deliberately.
- **Expose comp_qa / trust as Settings toggles** + write their defaults to
  `config.json`. These are **user-facing** QA features (the trust badge is observer-facing),
  currently config-only and UI-hidden. Add a "Data quality & validation" Settings section for
  the two. Keep the experimental `phase01_comparison_proximity_tiebreak` / `rms_bin_mag` and
  the PSF/COG/crowding flags hidden.
- **TODO-SEP-XVAL — Persist offline cross-val results to a tracked ledger.** Append one summary row per
  `xval_run.py` harness run to `validation/xval_ledger.csv` (in git):
  `date, draft_id, engine, n_targets, median_ratio, n_confirmed, n_review, n_indep_failed,
  commit, notes`. The offline `xval_out/` stays scratch; per-draft photometry tree is unchanged
  (no in-pipeline `sep_xval_*` artifacts). Gives a longitudinal validation record (mission +
  paper Validation section). *(Re-scoped 2026-06-03 after in-pipeline sep_xval retirement.)*
  → **PUBLICATION** workstream.
- **TODO-APCORR-COLOR — Extrapolation guard warn→block: DONE** (2026-06-03). Target BP-RP
  outside comp range → CT skipped, target kept uncorrected (`phase01_ct_extrapolation_tol`,
  default strict 0). **NoFilter CT enable: PARKED** — prototype on draft_000366 showed modest
  effect; **filtered drafts now exist (Milan, 2026-07-09) — revisit when scheduled; do not
  unblock here.** Revisit when filtered / Newton fine-scale data is re-verified.
- **Verify wide-field WCS distortion model (from SIPS comparison).** Confirm whether the
  plate solve fits higher-order distortion (SIP-style) or only a linear CD matrix. On the wide
  rig (~5.6°×3.8°, with a corrector) residual field curvature can reach several pixels at the
  edges → edge astrometry/photometry offsets. SIPS models this with a 3rd-order 2D polynomial
  (Monomial/Legendre). **Verify first:** if SIP is already fitted, close this; if linear-only,
  add higher-order distortion terms. (See DECISIONS: *What VYVAR deliberately does NOT adopt from SIPS* — WCS distortion scoped here.)
- **TODO-GS10 — AAVSO Direct API upload** ("Submit to AAVSO" button; WebObs API after the
  GS6b validation). Dep GS6b ✓.
- **TODO-45 — RGB camera support** (IMX533 RGGB → de-Bayer → G-channel photometry).
- **TODO-8-BOO — Bootes globular-cluster validation** (ePSF vs aperture on a dense ~2 h
  field). Pairs with the PSF/Newton work.
- ~~**TODO-FORCED-COMP — forced-aperture `catalog_only` without Phase-1 tier selection.**~~ **SUPERSEDED (`7f0dc86`)** — forced-aperture / catalog_only path removed; DAO+Gaia matched only.
- **TODO-LC-TREND — differential extinction + ALG audit** (partial; re-validate on a
  moonless night).
- **TODO-LIB — Cython `.pyd` compilation** (hide source, enable C translation).
- **TODO-CONFIG-CHURN — dedicated day.** The app rewrites session/UI state into the tracked
  `config.json` each run → perpetual git diff. **Zero functional effect** (the resolver
  ignores the config site; UI uses `LOCATION.IS_DEFAULT`). Fix = split a session-state store
  from the static overrides. **Do NOT gitignore `config.json`** — it holds real overrides.
- ~~**TODO-BROAD-EXCEPT-HYGIENE**~~ **DONE (2026-06-11).** Phase G critical-path logging
  (2026-06-08) + **BLE001/E722 regression guard** (pyproject, pre-commit, pytest); 168 sites
  grandfathered `# noqa: BLE001`; 4 bare excepts fixed; 8 `photometry_core` narrowings. ~1200
  pre-existing noqa sites remain opportunistic. MASTERSTAR writeto fail-closed + edge-ok
  fail-open (#4) unchanged.
- **Phase H cosmetic lint (2026-06-08, DONE):** value-filtered subset applied (SIM118 x11,
  RUF022 x2, RUF007 x2, RUF034 x3 dead-ternary); ProcFrameStore SIM118 x2 kept; ~89 style
  findings accepted per PROCESS. **Clean-code campaign Phases A–H COMPLETE.**
- ~~**TODO-GEO**~~ **CLOSED (2026-06-09)** — superseded by PARAM-PROVENANCE (`param_resolver.py`:
  per-draft `ID_LOCATION` -> header -> config; BJD/airmass config-independent). See DECISIONS.
- ~~**Comparison-star floor policy**~~ **DONE (2026-06-11, Option B).** Trust-only
  `comp_trust_min_comps=5`; Phase-1 selection min stays 3. Spec:
  `VYVAR_COMP_FLOOR_POLICY_SPEC.md`. Option A (selection floor 5) parked — moves anchor.
- **`classify_lc_quality` short-baseline (#3).** **DONE (2026-06-10)** — see
  `VYVAR_LC_QUALITY_SHORT_BASELINE_SPEC.md`. Follow-up: vsx_type-aware thresholds.
- ~~**Night-run false success (draft_383)**~~ **DONE (2026-06-11)** — completeness gate in
  `night_run.py` (>=90% summary/active per setup).
- ~~**Check-star trust audit CS-1..4**~~ **DONE (2026-06-11)** — specs +
  `VYVAR_CHECKSTAR_SELECTION_SPEC.md`; CS-3 ensemble exclusion via `ensemble_ids`.
- **Reserved check-star (hold-one-out).** PARKED — moves photometry anchor; see DECISIONS.
- **AAVSO-standard output (#4).** PARKED — G→standard B/V/Rc (Broeg band/colour point).

## LOW

- **GAIA-1 / GAIA-2 (pmra/pmdec, ruwe)** -- **DEFERRED to Gaia DR4 build** (~Dec 2026). Not
  restarting the DR3 rebuild. See DECISIONS + `VYVAR_GAIA_DR3_AUDIT.md`. DR4 migration hooks
  (epoch J2017.5, build columns, lite-table check) recorded in DECISIONS.
- ~~**comp_qa fix-once magnitude locus (CQ-C)**~~ **DONE (2026-06-09)** — fix-once pass-1 locus;
  order-independent flagging; bounded diff 1 flag / 1 n_clean / 0 trust on draft_000366; SHA
  `edbd97e7...` (426 files incl. comp_qa). Sibling ddof+threshold co-calibration remains open.
- **FITS-side proc glob consistency.** `pipeline.py` uses inline `aligned_dir.glob("proc_*.fits")`
  (~5578, 12604, 12683) rather than a shared helper; functionally correct (`proc_*` matches both
  naming styles). Optional consistency cleanup only.
- **Spatial term in calibration (from SIPS comparison) — only for a future whole-field absolute
  mode.** SIPS's ensemble adds x,y polynomial terms (`x1·X + y1·Y + x2·X² + y2·Y² + xy·XY`) for
  field gradient / vignetting. **Not a gap in VYVAR's current per-target differential path** —
  the local-comp ensemble already cancels most spatial systematics. Relevant only if/when VYVAR
  does whole-field absolute photometry (all targets against one frame-wide solution). Sibling of
  APCORR-COLOR (the colour term). (See DECISIONS: *What VYVAR deliberately does NOT adopt from SIPS* — spatial term scoped here.)
- ~~**TODO-WIDE-RIG-REPROCESS — clean re-run of 361/362.**~~ **CLOSED (OBE, 2026-07-09):** drafts
  361/362 deleted; item no longer actionable.
- ~~**B-V legacy removal — Stages 2–4.**~~ **Closed 2026-06-03** (scope A+B; commits in
  JOURNAL). Regenerate `vyvar_vsx_local.db` on the catalog machine after pulling.
- **TODO-10 — Settings-tab refactor + `CONFIG_GUIDE.md`.** Ties to `VYVAR_PARAMS.md` /
  config↔UI parity.
- **DAO-RECONCILE — Gaia↔DAO field accounting (reframes TODO-13).** **CLOSED
  (2026-07-09).** Arc: R-1 (`becc274`/`e9daec9`) population bug fix → R-2 (`bd6244a`/
  `4279a52`/`b7df7c6`) footprint+Fleming fit → R-2b (`78febea`) censoring+miss@G90 → close
  (`DAO-RECONCILE-CLOSE`) flat-curve no-crossing fix. **Outcome:** completeness 89.7–98.3%
  across rigs; G_lim characterized (wide ~15.0, Newton V ~16.7, B/R >=17.5 censored,
  narrow-band >=17.5 no-crossing); `completeness_50` + `missed_below_g90` live in QA dashboard
  as ongoing health signals; original TODO-13 fully superseded. Diagnostic:
  `scripts/dao_reconcile_diag.py --all-drafts`.
  **2-pass DAO recovery: CLOSED (not-worth-complexity, 2026-07-09).** Decision basis: after
  the reference-population fix, true anomalies (missed@G90) are 15 (424 wide), 6/18 (427 g/r),
  10 (425 V); 425 B/R misses (314/212 under censored depth) are the practical ceiling of
  catalog cross-match (Gaia near-bright-star spurious sources, sub-blend-radius pairs, proper
  motion) — 2-pass would harvest artifacts, not signal. **REOPEN CONDITION:** missed@G90
  becomes material (hundreds) on a new rig/config in the QA dashboard.
  **PUB-QC-MISSRESIDUAL (PARKED, LOW):** positional classification of the 425 B/R missed
  sample (near-bright-star distance, pair separation, PM) — explains the residual 2–3%, strong
  QC passage for future publication Validation section. Not scheduled.
  **Artificial-star injection tests** (photutils) = future gold-standard validation; candidate
  for EXTERNAL-XVAL / publication Validation section.
- **TODO-LC-QUALITY — LC classification filter.** `lc_quality_flag` exists and is consumed by
  the trust gate; verify the saturated/noisy export policy is complete.
- **TODO-14 — PDF size optimization** (29 MB → < 10 MB).
- ~~**TODO-MASTERSTAR-QA — FORCED_APERTURE cyan overlay** in the QA UI.~~ **SUPERSEDED (`7f0dc86`)** — `FORCED_APERTURE` proc rows no longer produced.
- **Misc LOW:** TODO-7 plate-solver refactor · TODO-11 auto-trigger watchdog (`night_run`
  foundation exists) · ~~TODO-12 HRD classification (after new DB)~~ **DONE (2026-07-10):** session-aware
  HRD extreme-object table; online Gaia TAP + SIMBAD enrichment for candidates only (lite DB teff/logg
  NULL); absolute thresholds (Pecaut & Mamajek 2013; Andrae et al. 2023). **TODO-12b (2026-07-10):**
  parallax gate 0.15 mas + SNR 5 (config-driven); per-category cap 3; NSS deprioritized in enrich budget;
  apparent-G legend/caption fix (chi Per / draft_425 cluster admitted to M_G plane). **TODO-12c (2026-07-10):**
  luminosity-first Stage-2 label priority (RSG before Very cool); Stage-1 per-net reserved slots
  (`hrd_min_per_net=4`). **TODO-12d (2026-07-10):** `hrd_nss_category_enabled=False` drops NSS/binary
  from table; annotated MASTERSTAR field image (scale guard, short labels, legend) wired to PDF HRD page.
  **TODO-12e (2026-07-10):** identification tiers (confirmed/likely/candidate); enrichment cache v2
  (+ SIMBAD sp_type, Gaia DSC); SIMBAD lum-class logg substitute (RS Per RSG fix).
  **TODO-12f (2026-07-10):** PDF/UI `Extreme objects -- details` per-object blocks; extended row
  payload (dist, parallax, raw SpT/otype, DSC p); validate summary stamps.
  Arc **DONE** — see `CURSOR_RESULT_todo12_hrd.md` … `_todo12f_hrd.md`.
  · TODO-20 mean-stack
  MASTERSTAR (improves WCS/FWHM only, not LC SNR) · TODO-CACHE-CENTRAL centralize `csv_cache`
  · TODO-PIXEL-XCHECK-BINNING binning-aware pixel cross-check (cosmetic log) ·
  **TODO-RECUT-HARNESS-FIDELITY** — **CLOSED (superseded, 2026-07-08):** draft_387 zaloha no longer
  exists (Milan confirmed); frozen-anchor premise dead. Current regression references: draft_424
  provenance anchor `e1a7a311…` + `session_baseline_check.py --full`; rebaseline cut1 SHAs where
  fixture present (`tests/test_photometry_sha_baseline.py`). Historical SHA constants in
  `tests/photometry_sha.py` retained (self-skipping).
  **TODO-INSTALL-MANUAL — inštalačný manuál + inštalátor pre nového užívateľa (vrátane
  katalógov)** · TODO-PLATESCALE-PERSET focal×pixel per-set plate-scale fallback.

### TODO-INSTALL-MANUAL (naviazané: **TODO-9**, **TODO-LIB**)

**Cieľ:** nový užívateľ (bez znalosti vnútra) nainštaluje celý VYVAR + sprevádzkuje katalógy
podľa manuálu; na referenčnom stroji **LENOVO T460** (Linux — potvrdiť presnú distro) to spraví
**jedným inštalátorom**. Inštalátor = realizácia; manuál = dokumentácia toho istého cieľa.
Prepojiť s Cython balíkom (**TODO-LIB**). Nahrádza/rozširuje starý one-liner **TODO-9**.

**Manuál (dokumentácia):**
1. Prostredie — Python, venv, `requirements.txt`, spustenie Streamlit dashboardu.
2. Aplikácia — získanie repa/balíka, prvé spustenie.
3. **Katalógy (kľúčové)** — 3 gitignored súbory: `vyvar_gaia_dr3.db` +
   `gaia_triangles_fine.pkl` + `gaia_triangles_wide.pkl`. Dve cesty: **(a) build**
   (`build_gaia_catalog.py` → ESA TAP, hodiny; potom `build_blind_index.py`) alebo **(b) stiahnuť
   hotové** (ak hostované). Na T460 je full-sky build @ G≤16.5 nepraktický → uprednostniť hotové
   (príp. slim podmnožina).
4. Konfigurácia — Settings: `GAIA_DB_PATH`, `BLIND_INDEX_FINE_PATH`, `BLIND_INDEX_WIDE_PATH`,
   `archive_root`, `calibration_library_root`, `database_path`; Test connection / Skontrolovať index.
5. Prvý beh — verifikačný checklist (test solve / krátky pipeline, trust gate GREEN).

**Inštalátor (T460):** skript/balík pripraví prostredie + app + nasmeruje na katalógy (download
hotových alebo build). Zohľadniť RAM (fine PKL ~1.3 GB + in-memory verify katalóg), disk (~10 GB DB),
CPU. Jedna zdrojová pravda s manuálom.

**Otvorené (rozhodnúť pri implementácii):** distribúcia katalógov (build vs download vs slim);
presná OS/distribúcia T460; forma inštalátora (shell / conda / pip / Cython wheel); min. RAM/disk v
manuáli.

**Definition of Done:** manuál → čistá inštalácia od nuly po GREEN trust na testovacom poli;
inštalátor → rovnaký výsledok na T460; manuál a inštalátor konzistentné.

### Plate-solver robustness backlog (future, lower priority)

**Future (lower priority):**
1. **Odds-based verification** — replace flat brightest-N % with confidence/odds vs chance (cone
   density aware; astrometry.net lesson).
2. **RANSAC / sigma-clipped SIP on inlier pairs** — fit distortion on inliers when ~35% spurious
   pairs prevent SIP adoption (`lin <= sip` guard).
3. **Geometry/position verification** — score full matched-set position residuals (band-agnostic)
   instead of brightness-ranked brightest-N alone.

## FUTURE

- **Blind index — 3rd rig tier (Noctutec 206/560).** When a validated draft exists, add a
  third PKL tier + config path; architecture is ready (`blind_index_fine_path` /
  `blind_index_wide_path` + `blind_index_select_mode=auto`).
- **TODO-GS7 — paper draft** (PASP / AN). Working title locked: *VYVAR: An Automated
  Differential Photometry Pipeline for Amateur Variable Star Observers*.
- **Comet photometry mode** — major parallel phase **after** the variable-star pipeline is
  finished (shared front-end calibrate→platesolve→star-stack→Gaia ZP; forked back-end:
  comet-rate stacking + extended coma photometry + ICQ/COBS export). Analysis only — do NOT
  start yet. (DECISIONS.)
- **TODO-SCENE-FORWARD-MODEL** — conditional on crowded-faint science (Brno / globular
  clusters); priority lowered after the grouper-negative result.

---

## Parked (next — Milan chooses, 2026-06-14)

| Item | Notes |
|------|-------|
| **CM-detrend differential** | ~10× lever; opt-in; needs transit injection-recovery test before opt-in |
| **Exoplanet / TOI catalog** | **DONE** — NASA Exoplanet Archive TAP (14,185 rows), TOI annotation live; see JOURNAL 2026-07-08 EXO entry |
| **Newton-V colour-term** | Per-rig c1 from field BP-RP |
| **Meridian-flip handling** | Qatar-8 class |
| **`[TODO-RECUT-HARNESS-FIDELITY]`** | **CLOSED (superseded 2026-07-08)** — draft_387 zaloha gone; use draft_424 + DEV-PROCESS-B --full |
| **Pre-filled camera catalog** | New-user onboarding — sensor-keyed research, camera-keyed rows; **PARKED** (see below) |

**Backlog (unchanged):** broad-except Tier-1 (~25); B-V legacy removal Stages 2–4; TODO-46 (skip
airmass detrend for known VSX variables); TODO-LC-QUALITY; TODO-LC-TREND; TODO-GEO; GS8–GS11.

### TODO (PARKED) — Pre-filled camera catalog for new-user onboarding

**Idea:** Ship VYVAR with a curated catalog of monochrome astronomy cameras (the ones amateurs /
professionals actually use). A new user picks their camera from the list to pre-populate their
EQUIPMENTS row and set it as default, instead of hand-entering specs. Lowers the onboarding barrier
(fits the non-expert / "trust in the numbers" mission).

**Status:** PARKED — discuss scope + format with Milan before any implementation.

**Design decisions already reached (do not relitigate without discussion):**

- **Two-tier table.**
  - *Static / authoritative (safe to ship as fixed):* sensor model, pixel size, sensor dimensions,
    bit depth, nominal saturation / full-well. These are genuinely fixed per camera/sensor.
  - *Setting-dependent (ship as "base defaults", NOT authority):* gain (e-/ADU) and read noise.
    These are **not fixed** — they vary with the gain setting and binning. Store them **per gain
    mode** (e.g. gain-0/LCG, unity, HCG/high), clearly marked "nominal, verify against your FITS
    header / measurement." A single value per camera would give a false sense of correctness.
- **Integrates with the header-first gain logic (already implemented).** Catalog = base/fallback;
  the FITS header overrides gain per-session when it carries e-/ADU (Moravian). Read noise has no
  header source in tested cameras, so the catalog/DB per-mode RN is what's used → per-mode RN
  matters most.
- **"Trust in the numbers":** recommend the user measure their own gain/RN (photon-transfer) or rely
  on the header value; the catalog is a starting point, not a substitute.
- **Research sensor-keyed, deliver camera-keyed.** Many cameras share one Sony BSI sensor
  (IMX571 = ASI2600/QHY268/C3-26000; IMX455 = ASI6200/QHY600; etc.). Research gain/RN at the sensor
  level, deliver rows that map to EQUIPMENTS columns (CAMERANAME, SENSORTYPE, SENSORSIZE, PIXELSIZE,
  SATURATE_ADU, GAIN_ADU, READNOISE_E; FOCAL stays null = per telescope).
- **Suggested first scope:** dominant Sony BSI mono family + the cameras using each — IMX571, IMX455,
  IMX411, IMX492/294, IMX533, IMX585, IMX461, plus legacy IMX174/178/290 and Panasonic MN34230
  (ASI1600). Covers ~90% of real usage. Not an exhaustive "every camera" list (scope + accuracy).
- **Sources:** manufacturer spec pages + peer-reviewed characterizations (e.g. Alarcon et al. for
  IMX455/IMX411), not memory.

**Open questions to settle before implementing:**

1. Scope breadth (start with the Sony BSI family above? wider/narrower?).
2. Format: seed data the app offers for selection (CSV/SQL) vs a reference document for review first.
3. Schema: does EQUIPMENTS need a gain-mode dimension (per-mode gain/RN rows), or one base row per
   camera + reliance on header-first gain for the per-session value?

---

## Parked (round 2 — refinements, not blocking)

- **Magnitude-aware check-star threshold for the trust gate.** The flat `0.02` / `0.05`
  cutoffs carry the same magnitude-dependence the comp_qa locus fixed; extend the
  Sokolovsky/locus treatment to the check-star axis.
- **PSF cross-validation** — needs a PSF-heavy/faint draft + per-frame ePSF (the aperture
  cross-val is CLOSED).

## Dropped / resolved (do not re-open)

- **TODO-GS9 — Ground-LC period analysis in the PDF** — **closed: descoped 2026-06-09.**
  Lomb-Scargle/BLS + folded diagram on VYVAR's own Phase-2A LC as a PDF science product is
  out of scope; period finding/classification is downstream (Peranso, VStar, Period04). See
  DECISIONS (product scope boundary). LS/BLS citations remain for `tess_verify` TESS cross-check.
- **Blind solver in dense fields + index series + rig-prior** — **RESOLVED 2026-06-04** (Newton):
  mag14 tiers, `vyvar_blind_series`, solve-rate harness, scale/FOV hard gates (`blind_use_rig_prior`,
  `blind_scale_tol_frac`). ~~**Wide-rig blind HIT**~~ **CLOSED (2026-07-09):** believed resolved by
  later solver work (Milan); reopen on the next wide-rig blind-solve failure.
- **V/R re-run (draft_375)** — **RESOLVED 2026-06-04** via draft_380 clean full run (all filters).
- **Trust `n_clean=0` diagnosis** — **RESOLVED 2026-06-04:** root cause = pre-cal proc-CSV glob in
  `load_proc_pivot` (draft-specific, not a cleaning regression); folded into canonical pre-cal
  proc-CSV resolution (HIGH).
- **Canonical pre-cal proc-CSV resolution** — **CLOSED 2026-06-08:** `load_proc_pivot` uses
  `list_proc_csvs` / `PROC_CSV_GLOB="proc_*.csv"`; verified `tests/test_proc_csv_glob.py` +
  calibrated draft_000366 n_clean populated.
- **ProcFrameStore pre-cal naming** — folded into canonical pre-cal proc-CSV resolution (HIGH,
  2026-06-04); do not fix per-consumer.
- **MASTERSTAR-EPSF-ALL** — dropped 2026-06-02: plate scale is WCS-derived; affected drafts
  311/321/358 are deleted; 361/362 ePSF already ≈ 9.77. No recurrence risk.
- **GS6b** — DONE 2026-05-20 (`scripts/validate_aavso_export.py`). Residual delta only: add a
  headroom check so the new `trust=` AAVSO NOTES field stays within Extended-Format limits.
- **IRAF / PyRAF cross-val** (and TODO-32 EPADU) — closed as unnecessary; two independent
  engines (sep matches VYVAR to 0.2 %) already validate extraction; not feasible on
  Py3.12/Ubuntu24.
- **TODO-WEIGHTED-LC, TODO-SKY-PLANE** — tested negative, closed.
- ~~**TODO-DEV-PROCESS**~~ **DONE (2026-07-08)** — **DEV-PROCESS-A** (validation ledger) +
  **DEV-PROCESS-B** (`session_baseline_check.py`); Definition-of-Done discipline in
  `VYVAR_PROCESS.md`. Charter: `VYVAR_CLAUDE_OPERATING_PRINCIPLES.md`.
- The full **TODO-1…45 / PERF-1…10 / ALG-1…5 / CQ-1…7 / GS1–GS5** series — closed; see
  `VYVAR_JOURNAL.md`.

---

## Parked for next session (2026-06-11)

| Item | Notes |
|------|-------|
| Reserved check-star | Hold-one-out by design — **moves photometry anchor** |
| AAVSO-standard output #4 | G→B/V/Rc (Broeg band/colour) |
| TODO-MULTISET | Per-rig config (wide vs fine) |
| TODO-GS8 | Phase-3 global ZP |
| DR4 build | ~Dec 2026; J2017.5 epoch hook `vyvar_platesolver.py:63` |
| PSF / NEIGHBOR-SUB | Needs bin1 ~0.65"/px data (Brno gate) |
| `build_gaia_catalog.py` adaptive-split | Next full-sky build only (not this commit) |
| **D1-combination (Broeg-weighted vs flux-sum)** | Re-test weighted `ens_med` after **sigma budget + χ² gate** — **moves anchor**; blocked on sigma completeness |
| **D3 extinction/colour physics** | Second-order extinction + standard system → ties **AAVSO #4** |
| **C second pass (Howell + aperture corr.)** | CCD error budget, curve-of-growth / APCORR; citation-integrity follow-up |
