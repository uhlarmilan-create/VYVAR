# VYVAR -- Development State

Last updated: **2026-06-22** — **Forced-aperture / catalog_only removed; DAO+Gaia photometry only.**
Variable targets measured **only on direct DAO `catalog_id` hit** (miss → nondetection/NaN; no XY
fallback). Unmatched VSX excluded in Fáza 0. Validated do-no-harm vs draft 419: `mag_inst` 360/360 B;
`mag_calib` 357/360 B (one target +~30 mmag uniform zeropoint from comp_qa excluding intermittent
forced-only comp — accepted). See DECISIONS.

Prior: **2026-06-19** — Stage B held pending validation (forced-aperture removal draft).

Prior: **2026-06-18** — **Fix C / Phase C1: dense-field alignment DIAGNOSED → root = PSF/FWHM
bloat; recovery NOT APPLICABLE.** The 14 late-night (post-flip, back-half) frames Fix B drops are **not**
"good data that only failed alignment" — they are **PSF-degraded**: median **FWHM 8.60 px = 1.85× the
good baseline 4.64 px**, concentration flux_large/flux **13.1 vs 1.65**, **corr(FWHM,
alignment-residual)=0.95** (161 frames; `tmp/phaseC1/fixC_root_cause.png`). The bloated-donut centroid
noise (~2.4 px) is the single root — it breaks astroalign (misalignment is the *symptom*) and is what
B.2 (concentration) + Fix-B (residual) measure. Likely **late-night focus drift on the defocused rig**
(a transparency/flux drop alone would not bloat FWHM); post-flip-half-not-refocused is an observer
question. **Not recoverable to sub-px** (centroid floor ~2.4 px > 1.37 px gate; cap50→3/14, WCS absent
0/162, translation-refine inapplicable). **Fix B + B.2 are the correct PERMANENT quality gate** — not a
stop-gap awaiting Fix C. Logged a SEPARATE control-point-cap perf ticket (astroalign mcp≈200 → ~654
s/frame on dense fields; cap ~50 → ~3–10 s; ROADMAP). **A.B.: Fix A `005716d` + Fix B `fa03410` pushed
to origin/main this session (Milan-authorized).** `CURSOR_RESULT_fixC_diag.md`. See DECISIONS/JOURNAL.
Prior: 2026-06-18 — **Fix B: reject-on-alignment-residual frame gate** (default-OFF;
`frame_align_residual_gate_enabled`). Two additive pieces: (1) **always-on QC** — a per-frame
**alignment residual** (median deviation of bright matched sources from their across-night median
position) is computed at the Phase-2A frame-selection point and recorded as `align_residual_px` in
`alignment_report.csv` (additive metadata → photometry byte-identical); it reproduces the run-414
diagnostic separation (astroalign med **0.358**/max **1.648** px vs phase_corr min **1.450**/med
**2.130** px). (2) **gate (default-OFF)** — rejects frames whose residual exceeds
`frame_align_residual_max_frac × science-aperture-radius-px` (**rig-agnostic** fraction, default
**0.25** → 1.37 px, in the 1.206→1.450 px good/bad gap; safety floor `min_keep_frames`). Verified on
run-414 g: **OFF byte-identical** (70 targets, V0454 `mag_calib`/`delta_mag`/`err` max|diff|=0); **ON
drops 14 frames = all 13 phase_correlation + 1 mis-aligned astroalign** (dr=1.648, itself an LC
outlier) — V0454 outliers 22→10, the catastrophic +3.7 mag/NaN points gone (clean SIPS-grade egress;
`tmp/fixB_v0454.png`). **B.2 cross-check:** residual gate ⊇ B.2 (overlap 13, residual-only the 1
astroalign, B.2-only 0) — cause-correct (alignment) superset of B.2's aperture-integrity symptom; both
kept distinct. **[C1 correction: PERMANENT gate, not "self-deactivating once Fix C fixes alignment" —
the frames are PSF/FWHM-bloated and unrecoverable.]** See DECISIONS/JOURNAL.
Prior: 2026-06-18 — **Fix A: per-point error model bug fixed** (default; no flag). The LC
`err` term-3 was `np.std(comp instrumental mags)/√n` (`photometry_core.py:2567`) — for a sparse/
brightness-spread ensemble this is the comps' brightness *spread* (a fixed ~0.58 mag floor on V0454,
23× the empirical 0.025), not a per-point uncertainty. Replaced with the per-frame **ensemble-ZP
standard error from comp residuals** (each comp vs its own across-night median → brightness/colour
cancels; Honeycutt 1992); the redundant `comp_rms/√n` term-2 was dropped (no double-count); photon
term-1 (incl. SNR-blowup on bad frames) kept. Verified on run-414 g: centres `mag_calib`/`delta_mag`
**byte-identical**, V0454 err 0.581→0.013 (≈empirical), faint targets photon-dominated, the 13
mis-aligned frames still flagged (Fix B). `err` does NOT feed trust/lc_rms/production-Broeg-combine;
it does feed SysRem IVW weights (default-OFF) — improved, not broken. See DECISIONS/JOURNAL.
Prior: 2026-06-17 (end-of-day) — clean committed **+ pushed** baseline at `955b850`
(8 commits: `1eea2d2` masterstar recovery, `e042bc1` A-durable, `d222eb7` B-cap, `2cc2b76`
completeness gate, `63e57c0` log-flood, `a126980` B.2 gate, `15c699e`/`955b850` docs). `draft_413` =
Boyden V454 CrA non-cal sandbox (g+r; **g fully validated** this session). Validated this session:
non-cal ingest, headless run, meridian-flip handling, Brno gate, B-cap, B.2 (default-OFF). **V0454 CrA
flip diagnostic:** the 0.45 mag rise = real eclipse egress (~0.37 mag, comp-invariant, SIPS-corroborated)
dominating ~4:1 over a ~+0.1 mag position-dependent meridian-flip step (explains the 0.45-vs-SIPS-0.548
gap as comp choice, not pixels; see DECISIONS + `docs/round2_figs/v0454_flip_diag.png`). **Pending:
UI-VYVAR live test of A-durable** (ROADMAP).
Prior: 2026-06-17 (Part A clean baseline committed [6 commits, push gated]; Round 2:
B.1 aperture-skirt **refuted** by COG/scatter diagnostic [not implemented]; B.2 transparency
**frame-quality gate** implemented behind default-OFF `frame_quality_gate_enabled` -- isolated
measurement on draft_413 g cuts bright-target LC scatter by median -257 mmag, trust still RED
[structural check-star/comp]. See `CURSOR_RESULT_round2.md`).
Prior: 2026-06-17 (Round 1 four known fixes verified on draft_413 g: A-durable MP-reload
robustness, B-cap spatial-first variable_targets [+comp-purity coupling, Milan-accepted], measurable
completeness gate, NoDetections log-flood summary; simple-differential PRODUCTION).
Prior: 2026-06-16 (Phase-1 graceful comp degradation committed + matrix `164157` validated;
known-issue (b) closed).

This is the **entry point**: a snapshot of what is true *now* + an index. It deliberately holds
no history and no open-task detail -- those live in the linked files.

| File | Holds |
|------|-------|
| `docs/VYVAR_ROADMAP.md` | Open work (the only place to look for "what's next"). |
| `docs/VYVAR_DECISIONS.md` | Durable design decisions + *why* they hold. |
| `docs/VYVAR_JOURNAL.md` | Chronological session log (history, append-only). |
| `docs/VYVAR_PROCESS.md` | How we work: Definition of Done, validation discipline, config<->UI parity, tests. |
| `docs/VYVAR_CLAUDE_OPERATING_PRINCIPLES.md` | Claude operating charter (session-init required read; governs reasoning and answers). |
| `docs/VYVAR_PARAMS.md` | Config-key <-> default <-> clamp <-> UI-location registry. |
| `docs/VYVAR_DECISION_GROUNDING_RULE.md` | Adopted rule: cite physics/literature/practice before design forks. |
| `docs/VYVAR_REPORTING_COLUMN_GROUNDED_DECISION.md` | Workstream B reporting column (supersedes B1/B2). |
| `docs/VYVAR_CANONICAL_COMBINATION_LOGIC.md` | Flux-sum vs Broeg IVW -- conditional hold until sigma budget. |
| `docs/VYVAR_SIGMA_BUDGET_SPEC.md` | PARKED sigma-budget work item (Howell + scintillation + chi-squared gate). |
| `docs/VYVAR_VALIDATION.md` | Inject-and-recover synthetic validation harness (matrix, FAIL policy). |
| `docs/VYVAR_PIPELINE_CZ.md` | Czech pipeline manual for the paper (ASCII, rev. 2026-06-09). |
| `docs/VYVAR_GAIA_DR3_AUDIT.md` | Gaia DR3 ingest audit (build schema, match, ref mag; 2026-06-10). |
| `docs/VYVAR_LC_QUALITY_SHORT_BASELINE_SPEC.md` | `short_baseline` LC-quality spec #3 (rev b, ready; 2026-06-10). |
| `docs/VYVAR_RUNBOOK.md` | Chi_and_H zaloha-only night-run procedure (alias → baseline runbook). |
| `docs/VYVAR_CHIANDH_BASELINE_RUNBOOK.md` | Chi_and_H baseline re-cut procedure (byte-identity anchor; 2026-06-11). |
| `docs/VYVAR_TRUST_CHECKSTAR_HARDENING_SPEC.md` | Trust Findings A/B + CS-1 hardening (2026-06-11). |
| `docs/VYVAR_CHECKSTAR_SELECTION_SPEC.md` | Check-star selection CS-2..4 (2026-06-11). |
| `docs/VYVAR_COMP_FLOOR_POLICY_SPEC.md` | Comp trust floor policy; Option B adopted. |
| `docs/VYVAR_MATH_PHYS_AUDIT.md` | Math/physics audit (first pass; citation scoping landed). |
| `docs/VYVAR_COMP_DEGRADATION_SPEC.md` | Phase-1 graceful comp degradation spec (committed 2026-06-16). |

---

## Mission

A high-automation differential-photometry pipeline that lets amateur astronomers contribute
science with confidence: **trust in the numbers** (comp-stability QA via comp_qa + per-target
trust gate) and **guardrails for non-experts**. Independent extraction cross-validation lives
in the offline `xval_run.py` harness (not in-pipeline). **Aperture photometry remains the
validated workhorse** on the wide rig; **PSF is validated publication-grade at fine scale on
synthetic truth** (draft-367-like, mismatch ~0) but **gated OFF** in production LC until a
real Newton / dense-field draft passes the characterization gate.

## Pipeline (current)

```
Raw -> Calibrate -> QC (in-place) -> Align -> MASTERSTAR + Gaia DR3 catalog
   -> Phase 0+1: tier-ladder comp selection (colour window + RMS rank; bounds 3/8)
   -> Phase 2A: simple differential photometry
        (flux-sum ensemble; SNR-opt per-star aperture; NO temporal comp binning)
        -> reporting postprocess (ensemble ZP mag; NO per-target airmass detrend;
           mask-first outlier guard for known variables)
        -> comp stability QA (per-frame ensemble residual p2p)
   -> comp_qa (Sokolovsky LOO QA, read-only)
   -> trust gate (GREEN/YELLOW/RED; comp-health + check-star + lc_quality + stability)
   -> reports/exports (PDF SUMMARY MEASURE REPORT, AAVSO, VarAstro)
```

Plate scale is **WCS-derived** (~9.77 arcsec/px on the wide rig). Ensemble combine is **flux-sum**
(`delta_mag` canonical; AIJ/SIPS validated). Broeg inverse-variance ensemble combine is **PARKED**
until sigma budget validates (`docs/VYVAR_SIGMA_BUDGET_SPEC.md`). Comp selection ranks colour tier
first, stability second (both gated), proximity as a distance gate only. (See DECISIONS.)

## Production defaults (feature flags)

| Area | Flag / behaviour | Default |
|------|------------------|---------|
| Comp temporal binning | `temporal_binning_enabled` | **OFF** (ALG-3 breaks common-mode) |
| Color term | `apply_color_term` | **OFF** (colour-matched comps) |
| Comp selection | tier ladder 0.15/0.30/0.55, cap 0.79; bounds 3/8 | `_select_comps_by_color_then_rms` |
| Comp RMS floor | `comp_select_rms_floor` | **1e-6** (drop isolated-bin artefact) |
| Reporting | `apply_reporting_postprocess` | ensemble ZP `mag_calib`; no target airmass detrend; mask-first outliers |
| Comp stability | `check_comparison_stability` | p2p on **per-frame ensemble residual** (not raw `mag_inst`) |
| LC precision display | `lc_rms_ooe` on card | brightest-tertile scatter for variables |
| Aperture on card | `aperture_px` | **measured** proc `aperture_r_px` (not Phase-2A replan) |
| Comp QA | `comp_qa_enabled` | **ON** |
| Trust gate | `trust_flag_enabled` | **ON** (GREEN observed on draft_409 V0612) |
| Proximity tie-break | `phase01_comparison_proximity_tiebreak` | OFF |
| PSF (all) | `psf_photometry_enabled`, `psf_adaptive_enabled`, `psf_grouper_enabled`, `psf_spatial_enabled` | OFF |
| NEIGHBOR-SUB | `psf_neighbor_sub_enabled` | OFF |
| COG aperture corr. | `cog_aperture_correction_enabled` | OFF |
| Crowding classifier | `crowding_classifier_enabled` | OFF (wide rig) |
| Sparse comp fallback | `comp_sparse_fallback_enabled` | **ON** (per-target; inert on rich anchor) |
| Detrend | `sysrem_enabled`, `savgol_detrend_enabled` | OFF |
| Skip processed/ | `skip_processed_directory` | OFF |

PSF flags stay **OFF** on the wide rig (correct). The PSF path is now **validated-but-gated**:
enable only on characterized fine-scale data after the Brno / Newton characterization gate
(see DECISIONS + ROADMAP).

Comp bounds (user-configurable): Phase-1 selection `phase01_comparison_n_comp_min/max` = **3 / 8**
(unchanged). Trust-only floor `comp_trust_min_comps` = **3** (Phase-1; GREEN requires >=3 clean T1/T2
comps + check); `check_star_min_epochs` = **5**; CS-2 artefact floor `check_select_rms_floor` =
**1e-4**; CS-4 uses `aperture_correction_max_contamination` = **0.15** when
`contamination_idx` is present. `max_comp_rms` = 0.1; colour cut <= 0.79.

### Phase-1 graceful comp degradation (2026-06-16)

**Status:** committed + structurally validated (matrix `164157`; check-star preselect active).

Graded routing keeps 1-N good comps on default path (sparse only at 0); honest `comp_rms` /
`comp_rms_fieldwide` split; `comp_path` on summary + PDF; sigma scales with N; SS Cam fold
(pool attach, check-star field preselect).

### Phase-1b: per-target comp_rms gate authoritative for N_good (2026-06-16)

**Status:** committed (gate-authority part of known-issue (b) **CLOSED**). The per-target
`max_comp_rms`=0.1 gate is now the hard quality bar for N_good. RMS fallback no longer relaxes above the
gate (the `0.15` step is gone); auto-routing counts gate-passers (`_count_gate_passing_comps`), not raw
`len(result)`. Matrix re-run `185831`: SS Cam flips **default -> sparse_fallback** (its 0.134 comp fails
the gate, no longer a good default comp); V0612 + BO CVn + V0842 Her unchanged.

**OPEN -- SS Cam trust band (RED vs YELLOW) is UNRESOLVED, not closed.** SS Cam came out **YELLOW**, not
the predicted RED, but whether YELLOW is the grounded-correct band is **not yet decided**. The tension:
the sparse comp_rms (~0.35 mag) is a **field-wide-scale** quantity (different definition from the 0.1
per-target gate), and the check-star scatter (0.043 < 0.05 hard line) is **ensemble-dependent** -- comps
look bad, check looks OK, and neither has been verified. Resolve **diagnostic-first** in Phase-2 (does
field-wide sparse comp_rms cancel in the differential? is check-0.043 reliable given N points /
baseline?) **before** setting any sanity-ceiling threshold. Do NOT reverse-engineer RED. No threshold
re-tuning was done here.

## Rigs (known sets)

| ID | Telescope | Camera | Scale | Site |
|----|-----------|--------|-------|------|
| 1 | Carl-Zeiss 200 mm | QHY294MM | ~9.77 arcsec/px (wide) | Jirny |
| 2 | Newton 300/1200 | C3-26000 | ~0.65 arcsec/px (fine) | Dablice |
| 3 | Noctutec 206/560 f/2.72 | C3-26000 | TBD | TBD |

Per-set config architecture is still pending (ROADMAP: TODO-MULTISET).

## Status snapshot

### Gaia DR3 catalog integration

PM (`pmra`/`pmdec`) and `ruwe` are **NOT** in the DR3 catalog; **deferred to the DR4 build**
(~Dec 2026). Platesolver PM propagation is present but a no-op against DR3. Fine-scale dense
fields carry the GAIA-1 mis-association caveat until DR4 (DECISIONS).

### Brno AZ800 / C5A-150M (production solver — 2026-06-14)

**Brno AZ800 / C5A-150M onboarded.** Production solver uses **catalog-recovery verification**
(Gaia-in-frame / DAO at 2.5 px) as the MASTERSTAR accept gate; detection match% is informational.
Stale FITS pointing (`VY_TARG`) → **`hint_sep_warn`** when VERIFIED (Lang et al. 2010 prior), not
hard reject. Cone recenter at solved center when hint offset **≥ 0.05°** unchanged.
`generate_masterstar_and_catalog` passes `app_config` + scoped flags.

**Brno `r_60_4`:** catalog recovery tight **~84%** → **VERIFIED** under new gate (was rejected on
`hint_sep` + detection-denominated metrics). **`z_90_4`:** recovery **~34%** → stays rejected.
**Open:** Milan overlay sign-off on `tmp/diag_overlay_r.png`; anchor + home-rig regression re-run.

### Comp sparse-only fallback (2026-06-11 lock)

**Sparse-only fallback live (default ON).** Historical byte-identity anchor `3f7c9e7a` / `d5b72d08`
retired by simple-differential algorithm change; regression now uses empirical SIPS/AIJ cross-validation
on V0612 plus `compare_photometry_science_meaningful` for archaeology vs the zaloha cut.

### Reference draft and validation (not byte-identity)

The simple-differential algorithm change **retired** the old photometry SHA byte-identity anchors
(by design). Validation is now **empirical cross-validation** vs AIJ/SIPS on V0612:

- Out-of-eclipse RMS ~0.011 mag; eclipse shape correlation ~0.95+.
- draft_409 (2026-06-16): eclipse + single shared bright outlier at ~JD 2461200.385 matches SIPS
  -> frame-level artifact (cosmic-ray-like on target), not a VYVAR reduction bug.

Historical SHA anchors (`3f7c9e7a` / `d5b72d08`, Chi_and_H chi Per zaloha cut) remain documented for
regression archaeology; current code does not byte-reproduce them. Optional fresh anchor cut after
Milan sign-off (see ROADMAP / JOURNAL).

**Regeneration recipe (historical zaloha anchor)** (`docs/VYVAR_CHIANDH_BASELINE_RUNBOOK.md`):

1. **Source data (must retain):** `Archive/Chi_and_H` — pre-calibrated FITS (only non-regenerable input).
2. **Catalog + blind index:** `GAIA_DR3/zaloha/vyvar_gaia_dr3.db` (G<=16), zaloha blind PKLs
   (`gaia_triangles_fine.pkl`, `gaia_triangles_wide.pkl`). **Do not read** in-progress
   `GAIA_DR3/vyvar_gaia_dr3.db`.
3. **Run:** `python scripts/chiandh_night_run_bvr.py` (#3 code; Newton bin2 ~1.30"/px).
4. **Verify:** `compute_photometry_sha(draft_root)` core + full vs recorded SHAs (`3f7c9e7a…` /
   `d5b72d08…`). For regression vs the historical cut, use
   `compare_photometry_science_meaningful` (PROCESS) — excludes provenance/`err` QC drift.

**Setups (filter-wheel labels):** `B_20_2`, `V_20_2`, `R_20_2`, `L_20_2` — **B/V/R/L** are wheel
positions. **V** = visual/green (`G/` folder); **L** = clear/broadband (`L_20_2` in anchor).

**Provenance at anchor cut (2026-06-11, zaloha):**

| Item | Value |
|------|-------|
| `git_commit` | `7317ece87944b749461a7b6abca6615f1a30dc72` (re-baseline lock) |
| Catalog | `GAIA_DR3/zaloha/vyvar_gaia_dr3.db` (G<=16) + zaloha blind PKLs |
| Rig | Newton 300/1200 + C3-26000, bin2 ~1.30"/px |
| Ephemeral draft | `draft_000386` (~1401 LCs; deletable) |
| Completeness gate | `night_run.audit_photometry_completeness` — >=90% summary/active per setup |

Scoped trust at cut (`comp_trust_min_comps=5`, floor-5 baseline): **1382 YELLOW / 106 RED**
(1488 summary rows; re-trust on draft_387). Pre-floor-5 counts were 1400/88 — superseded.
Anchor photometry SHA is numeric and trust-independent.
- **Last science-validated wide draft:** `draft_000365` (V842 Her, 127 frames, 143 targets).
- **CT science locked:** h & chi Per `draft_000380` (Johnson-Cousins B/V/Rc); details in JOURNAL.

### PSF photometry (gated OFF in production LC)

Infrastructure complete. Validated **publication-grade on synthetic fine-scale** truth
(draft-367-like, ePSF-vs-star mismatch ~0):

| pillar | result (V3d harness, mag 12-17) |
|--------|----------------------------------|
| ACCURACY | mid-mag bias **<~2%** via brightness-independent **sky-only fit weights** (`psf_weight_mode=sky_only`; Astier 2013 / Lacroix 2025) |
| PRECISION | PSF scatter wins from ~mag 13 |
| UNCERTAINTY | P3 ~1 via **sandwich** variance (`psf_err_mode=sandwich_skyonly`) |
| ePSF FWHM QC | robust azimuthal-profile estimator (EPSF-1); warning band [0.80, 1.25] (diagnostic only) |

Sky estimate: aperture-consistent **annulus** / **residual_annulus** (`psf_sky_method` column).
Real-field enablement **blocked on a Newton / dense-field draft** (incoming Brno data will
unblock after the characterization gate).

### NEIGHBOR-SUB

**VALIDATED_FINE_SCALE_IDLE** -- works at fine scale (A9 HV ~83%, FAIL-SILENT 0), fail-safe
guards + full provenance, gated OFF; no current real use case (draft 367 sparse crowding).
Coarse / under-sampled fields fall back to SAFE_LOW_YIELD (correct REFUSE, not silent deblend).

### Fail-safety #4 (2026-06-08)

- MASTERSTAR WCS persist: **fail-closed** (draft solve fails; Phase 2A blocked for that draft).
- Edge-ok check: **fail-open + loud flag** (`edge_filter_failed` on `variability_candidates.csv`).
- Dead UI modules removed (`ui_photometry_results`, `ui_suspected_lightcurves`).

### Citations (PSF arc)

Astier et al. 2013, Lacroix et al. 2025, Guy et al. 2010, Stetson 1987, Mighell 1999 wired in
`CITATIONS.bib` where the methods run.

### Cross-validation, trust, tests, reporting

- **Cross-validation:** CLOSED for the aperture path (offline `xval_run.py`: sep reproduces
  VYVAR to 0.2 %/frame); in-pipeline `sep_xval` retired 2026-06-03; PSF cross-val deferred.
- **Trust distribution (draft_000365 baseline):** GREEN 69 / YELLOW 59 / RED 15.
- **Tests:** **261 passed / 14 skipped** (last full `tests/` run; incl. BLE001 + mid-exposure JD).
- **Lint:** `ruff check . --select BLE001,E722` clean (`pyproject.toml` + pre-commit + pytest).
- **Reporting:** R1 overflow guarantee holds (0 violations); R3 (aperture-vs-PSF overlay) pending.

---

## Top of mind

**Simple differential photometry is PRODUCTION** and cross-validated vs SIPS on V0612 (draft_409):
clean eclipse + shared single-frame anomaly at ~JD 2461200.385 (matches SIPS -> frame artifact).

**Trust/consistency cleanup landed (2026-06-16):** comp stability on per-frame ensemble residual;
measured aperture on card; `lc_rms (OOE)` for variables; trust GREEN on draft_409.

**Canonical column:** `delta_mag` flux-sum (AIJ/SIPS parity). Reporting `mag_calib` via
`apply_reporting_postprocess`. Broeg IVW ensemble combine **PARKED** until sigma budget validates
(`docs/VYVAR_SIGMA_BUDGET_SPEC.md`).

**Parked (see ROADMAP):** sigma budget; FWHM external validation; frame-level CR rejection;
source_id exact-match audit; TODO-MULTISET; Brno/Milan overlay; PSF / NEIGHBOR-SUB.
