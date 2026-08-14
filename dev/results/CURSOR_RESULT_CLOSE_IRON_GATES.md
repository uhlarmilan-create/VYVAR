CURSOR RESULT - CLOSE-IRON-GATES (2026-08-14)

Register IDs: **ENC-STALE-01**, **IRON-GATES-01**, **SKY-CLIP-01**
Base: `origin/main @ 4a3e855` (implementation uncommitted on working tree)
Author: Cursor implementation pass

---

## What I did

1. **Part 1 (ENC-STALE-01):** Ran `ascii_migrate.process_file` on 7 offending docs; recorded SHA-on-gate rule in `docs/VYVAR_PROCESS.md`.
2. **Part 2 (IRON-GATES-01):** Implemented static iron-rule scanner (`dev/tools/iron_gates_scan.py`), wired tests (`dev/tests/test_iron_gates.py`), invariant rows + `WIRED_INV_IDS` parity.
3. **Part 3 (SKY-CLIP-01):** Replaced one-sided annulus sky clip with plain median; unified batch and single-star paths via `_sky_pp_from_annulus_mask`; measured draft 510 FITS impact; documented decision in `VYVAR_DECISIONS.md`.

---

## Section 0 re-verification

| Finding | Reproduced? |
|---------|-------------|
| F1 `--fast` red (7 non-ASCII docs) | YES at task start; fixed |
| F2 one-sided clip `_sky_pp_from_annulus_image` | YES at 4a3e855; removed |
| F3 batch clip vs single-star median split | YES; unified |
| F4 iron rules not wired | YES; wired |

---

## Part 1 -- ENC-STALE-01

### ASCII fix

| File | Non-ASCII chars rewritten |
|------|---------------------------|
| `docs/VYVAR_AUDIT_2026_CLOSURE.md` | 38 |
| `docs/VYVAR_AUDIT_2026_REGISTER.md` | 7 |
| `docs/VYVAR_DECISIONS.md` | 7 |
| `docs/VYVAR_HANDOFF_2026-08-14.md` | 22 |
| `docs/VYVAR_JOURNAL.md` | 7 |
| `docs/VYVAR_ROADMAP.md` | 2 |
| `docs/VYVAR_STATE.md` | 2 |

Transliteration spot-check: table alignment and numeric tokens preserved (em-dashes/arrows to ASCII equivalents).

### `--fast` after fix (commit `4a3e855`, uncommitted tree)

```
SESSION BASELINE CHECK (fast)
------------------------------------------------------------------------
Check                        Status Detail
------------------------------------------------------------------------
git-branch                   PASS   main
git-head                     PASS   4a3e855
git-staged                   PASS   none
git-untracked-known          WARN   4 known untracked
git-untracked                WARN   dev/results/REGISTER_DIFF_U_SCATTER_DEF_A.md; dev/tests/_tmp_batch_e_lc/; dev/te
config-paths                 PASS   all present
pytest                       PASS   1331 passed, 27 skipped
manifest-db-parity           PASS   draft_id=435
ledger                       PASS   v1 15 items
ledger-todo                  WARN   VL-ANCHOR-424, VL-ANCHOR-DQ-430
deps-outdated                WARN   numpy 2.4.4->2.5.2 (+96 other) - gated upgrade, see docs/DEPS_POLICY.md
------------------------------------------------------------------------
OVERALL: PASS
```

Process note added to `docs/VYVAR_PROCESS.md`: gate results valid only for the commit SHA they were run on.

---

## Part 2 -- IRON-GATES-01

### 2.1 Scope (in invariant rows)

Scopes are explicit in `docs/VYVAR_INVARIANTS.md` for INV-NOCLIP-01, INV-NOCOSMIC-01, INV-PIXELS-01, INV-MASTER-01, INV-COMP-MEMBERSHIP. Scanner module list: `dev/tools/iron_gates_scan.py` (`PRODUCTION_SCOPE`, `OUT_OF_SCOPE_MODULES`, `MASTER_SCOPE`).

**Deliberately out of scope:** `xval_*`, `tess_verify.py`, `validate_lc_crossval.py`, `hrd_*`, `variability_detector.py`, UI modules, all of `dev/`.

### 2.2 Enforcement choice

| Rule | Method | Justification |
|------|--------|---------------|
| INV-NOCLIP-01 | Static grep/AST over production modules | Structural property (forbidden API patterns); precedent INV-RNG-01 |
| INV-NOCOSMIC-01 | Static grep | Same |
| INV-PIXELS-01 | Static grep for `np.where(isfinite(d), d, fill)` | Known sites under Milan review |
| INV-MASTER-01 | Static grep for ccdproc/sigma combine | Structural |
| INV-COMP-MEMBERSHIP | AST parse of `ensemble_normalize` | Per-frame selection is structural |

Allowlists: empty (PIXELS-01 test uses fixed known-site set only for interim FAIL-review).

### 2.3 Fire proofs

| Gate | Test | Demonstration |
|------|------|---------------|
| INV-NOCLIP-01 | `test_inv_noclip01_fire_proof_detects_annulus_clip` | Fixture with clip pattern flagged |
| INV-NOCLIP-01 | `test_inv_noclip01_production_scope_clean` | **Would have FAILED at 4a3e855** on `_sky_pp_from_annulus_image` clip; **PASS** after Part 3 |
| INV-NOCOSMIC-01 | `test_inv_nocosmic01_production_scope_clean` | PASS |
| INV-PIXELS-01 | `test_inv_pixels01_known_sites_only` | PASS (4 known fill sites) |
| INV-MASTER-01 | `test_inv_master01_plain_combine_only` | PASS |
| INV-COMP-MEMBERSHIP | `test_inv_comp_membership_ensemble_normalize` | PASS (AST: `good_ids` before frame loop; no per-frame ZP clip) |

### 2.4 Violation inventory (at base 4a3e855 scan)

| Location | Description | Outcome |
|----------|-------------|---------|
| `photometry_core.py:12120` (pre-fix) | One-sided annulus sky clip `sky_pixels < sky_med + 2*std` | **Fix authorized (Part 3)** -- removed |
| `photometry_core.py:2678,12235,12470` | Non-finite pixel fill with frame `nanmedian` before photometry | **Milan decision** -- see below |
| `psf_photometry.py:1993` | Same nanmedian fill before PSF fit | **Milan decision** |
| `photometry_core.py:4719-4777` `detect_outliers` | MAD-based outlier **flags** on calibrated mags (reporting) | **Scope refined** -- metadata flags only; not pixel modification; outside annulus/flux production path intent |
| `comp_selection_per_target.py` `_iterative_ensemble_clip_cm_residual` | Name contains "clip"; body is passthrough stub | **Scope refined** -- scanner matches `def` only; stub excluded when "Passthrough: no ensemble sigma-clip" present |
| `xval_harness_core.py` `sclip_std` | Sigma-clip scatter helper | **Out of scope** (xval harness) |
| `hrd_colorfield.py` `_sigma_clipped_median_sigma` | HRD analysis | **Out of scope** |
| `tess_verify.py` `_iterative_sigma_clip_lc` | TESS verification | **Out of scope** |

#### Milan decision: INV-PIXELS-01 nanmedian fill

**Sites:** `_annulus_sky_subtracted_flux` (2678), COG ladder path (12235), batch aperture BPM (12470), PSF (1993).

**Alternatives:** (A) propagate NaN -> photutils/aperture often returns NaN flux; flag star/frame. (B) exclude star from measurement. (C) keep global fill (current).

**External practice:** IRAF often masks bad pixels; photutils operates on finite arrays; SExtractor uses flag maps. Substitution with global median is a **choice**, not derived.

**This task:** no behaviour change.

---

## Part 3 -- SKY-CLIP-01

### 3.1 Physics

One-sided upper clip removes high annulus pixels only -> retained median biased **low** -> sky underestimated -> flux **overestimated**. Scale uses `std` on unclipped sample (contaminated). Q1 arm P1: `phot_plain - vyvar = -0.000585` (VYVAR higher flux) matches direction.

### 3.2 Literature survey

| Tool | Annulus sky estimator | Rejection on annulus sample? | Citation |
|------|----------------------|------------------------------|----------|
| **DAOPHOT** | Sky plane / annulus mode fit | CR rejection separate from sky sample asymmetric clip | Stetson 1987 PASP 99, 191 |
| **IRAF apphot** | `mode`, `median`, `centroid` selectable | Optional symmetric `nsigma` rejection in `skyvalue`; not one-sided upper-only | IRAF apphot package manual |
| **SExtractor** | Mesh background; local annulus stats | Optional sigma rejection on background mesh | Bertin & Arnouts 1996 A&AS 117, 393 |
| **photutils** | `Background2D`, annulus median/mean via `ApertureStats` | Optional `sigma_clip` (symmetric); default None | Bradly et al., photutils docs |
| **sep / SExtractor API** | Background subtracted before photometry | Mesh sigma clip configurable | Barbaro et al. SEP docs |
| **AstroImageJ** | Annulus median (user annulus) | No standard asymmetric upper clip | AIJ documentation |
| **C-Munipack** | Annulus aggregation for sky | Standard pipeline uses robust estimators without VYVAR-style one-sided cut | Munipack docs |
| **VaST** | SExtractor background | Uses SExtractor conventions | Sokolovsky & Lebedev 2017 |

**Finding:** No surveyed tool uses VYVAR's one-sided `median + 2*std` upper cut on the annulus pixel list.

### 3.2 Synthetic truth measurement (5000 trials, sky=100 ADU, sigma=2, N=80 annulus pixels)

Bias of estimator minus true sky (ADU):

| Condition | median | mean | clip (2-sigma upper) |
|-----------|--------|------|----------------------|
| Clean Gaussian | -0.002 | +0.000 | **-0.058** |
| PSF-wing contamination (8 high pixels) | +0.276 | +0.999 | +0.069 |
| Hot pixel (+50 ADU) | +0.038 | +0.632 | +0.006 |
| Cosmic ray (+200 ADU) | +0.035 | +2.503 | +0.003 |

**Pre-registered rule:** (a) unbiased on clean; (b) smallest bias under contamination among (a)-satisfiers without rejection; (c) no rejection step.

- **Clip** fails (a) on clean sky (bias -0.058 ADU) and violates (c).
- Among no-rejection: **median** beats **mean** on all contamination cases while tied on clean.
- **Decision: plain median.** No conflict with iron rule 1 case (c).

### 3.3 Path unification (F3)

| Path | Before | After |
|------|--------|-------|
| Batch per-star (`_aperture_flux_sky_per_star`) | `_sky_pp_from_annulus_image` with 2-sigma upper clip | `_sky_pp_from_annulus_mask` -> plain median |
| Single-star DAO/PSF (`_annulus_sky_subtracted_flux`) | Plain median via `get_values` | Same `_sky_pp_from_annulus_mask` (handles `to_image` and legacy masks) |

Draft 510 was produced with **clipped batch** path (stored proc CSVs match old clip recompute exactly).

### 3.4 Annulus weighting asymmetry

- **Aperture:** photutils default `exact` (fractional overlap) -- correct for integrated flux.
- **Annulus sky:** `method="center"` -- pixel counted in/out by center position.

**Justification:** Per-pixel statistics (median) require whole-pixel samples; fractional weights do not define a unique median. photutils `ApertureStats` with `sum_zeropoint` weighting is for sums, not location medians. IRAF/SExtractor likewise use discrete pixel lists for local sky. **Derived justification, not accidental.**

### 3.5 Draft 510 consequences

**FITS-level recomputation** (`dev/tools/sky_clip_510_impact.py`, `tmp/sky_clip_510_flux_delta.csv`):

| Metric | Value |
|--------|-------|
| Rows (star x frame) | 804 (134 frames, 6 stars) |
| Stored `dao_flux` vs old clip recompute | max rel diff **0.0** |
| Median fractional flux change (new median - old clip) | **-0.058%** all |
| Target BO CVn (`1498613634033133184`) | **-0.027%** |
| Comparison stars | **-0.070%** |
| Per-star median frac | see CSV / summary JSON in tool output |

**Approximate instrumental mag shift** (small-flux limit): target ~**+0.29 mmag** (flux down -> mag up).

**Stored light curve** (`lightcurve_1498613634033133184.csv`): **not updated** -- full Phase 2A re-export not run.

**Anchor re-cut:** **NOT EXECUTED.** Requires Milan authorization + full draft 510 photometry re-run from aligned FITS (Wave 7 procedure). Manifest builder ready: `dev/tools/build_anchor_checksum_manifest.py --label sky_median_20260814 --out dev/validation/anchor_510_checksums_sky_median_20260814.json`.

Prior manifest: `dev/validation/anchor_510_checksums_a1_dao_fwhm_20260814.json`.

### Retraction

**Q1-XVAL-MATCHED arm P1** (clip vs median offset at matched geometry): **retracted**; replacement is SKY-CLIP-01 fix + FITS recomputation above. See `docs/VYVAR_DECISIONS.md` SKY-CLIP-01 block.

---

## Files changed

| Path | Change |
|------|--------|
| `docs/VYVAR_*.md` (7 files) | ASCII migration + register/process/invariants/decisions updates |
| `src_py/photometry_core.py` | SKY-CLIP-01 sky estimator + path unify |
| `src_py/invariants_runtime.py` | `WIRED_INV_IDS` +5 iron rules |
| `dev/tools/iron_gates_scan.py` | NEW scanner |
| `dev/tests/test_iron_gates.py` | NEW wired + fire-proof tests |
| `dev/tools/sky_clip_510_impact.py` | NEW impact measurement |
| `dev/tools/build_anchor_checksum_manifest.py` | NEW manifest helper |
| `dev/results/REGISTER_DIFF_CLOSE_IRON_GATES.md` | NEW authorization diff |
| `dev/results/CURSOR_RESULT_CLOSE_IRON_GATES.md` | this memo |

**Not committed.** Nothing pushed (per task stop condition).

---

## Errors / gaps

| Item | Status |
|------|--------|
| Draft 510 physical anchor re-cut + new checksum manifest | **Not done** -- needs Milan authorization and long pipeline run |
| BO CVn final LC before/after from stored archive | **Not done** -- only FITS-level flux delta measured |
| INV-PIXELS-01 nanmedian fill | **Pending Milan** |

---

## Authorization checklist for Milan

- [ ] Register diff: `dev/results/REGISTER_DIFF_CLOSE_IRON_GATES.md`
- [ ] DECISIONS SKY-CLIP-01 block in `docs/VYVAR_DECISIONS.md`
- [ ] INV-PIXELS-01 nanmedian fill adjudication
- [ ] Authorize draft 510 re-cut + push when `--fast` green on commit to push
