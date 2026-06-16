# VYVAR -- Development State

Last updated: **2026-06-16** (simple-differential PRODUCTION; draft_409 trust/consistency cleanup;
SIPS cross-validation on V0612).

This is the **entry point**: a snapshot of what is true *now* + an index. It deliberately holds
no history and no open-task detail -- those live in the linked files.

| File | Holds |
|------|-------|
| `docs/VYVAR_ROADMAP.md` | Open work (the only place to look for "what's next"). |
| `docs/VYVAR_DECISIONS.md` | Durable design decisions + *why* they hold. |
| `docs/VYVAR_JOURNAL.md` | Chronological session log (history, append-only). |
| `docs/VYVAR_PROCESS.md` | How we work: Definition of Done, validation discipline, config<->UI parity, tests. |
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
| `CITATIONS.bib` | Single source of truth for all algorithm/software citations. |

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
(unchanged). Trust-only floor `comp_trust_min_comps` = **5** (`strong = min+2` → **7** at
defaults); `check_star_min_epochs` = **5**; CS-2 artefact floor `check_select_rms_floor` =
**1e-4**; CS-4 uses `aperture_correction_max_contamination` = **0.15** when
`contamination_idx` is present. `max_comp_rms` = 0.1; colour cut <= 0.79.

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
