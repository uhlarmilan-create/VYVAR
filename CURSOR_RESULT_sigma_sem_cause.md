CURSOR RESULT -- 2026-07-13 13:30 UTC+2

What I did
Built ``scripts/sigma_sem_cause.py`` + ``scripts/sem_cause_core.py`` to test whether ensemble
SEM inflation on draft_426 i/r is caused by smooth comp-residual trends (colour x airmass /
field structure) rather than independent per-epoch noise. Ran D1-D4 on draft_426 g/i/r (V0611 +
5 pooled check stars per setup, SS Cam separate) and draft_424 wide-rig control (4 stars).
Artifacts: ``tmp/sigma_sem_cause/``. **No production code changes.**

Script location: committed under ``scripts/`` (not gitignored sandbox) for reuse.

## C0 equivalence (MANDATORY)

Setup i_70_4 V0611, all 25 frames:

| check | max abs diff |
|-------|----------------|
| ensemble_normalize vs manual std/sqrt(n) | **0.0** |
| vs recomputed helper | **0.0** |
| vs LC-implied ensemble (mag quadrature) | 0.168 |

**C0 PASS** on production formula replication (``photometry_core.py:3113-3115``).
LC-implied ensemble differs because ``_combine_err_with_ensemble_scatter_keyed`` (~3205)
joins mag-domain ``ensemble_scatter`` with relative-flux ``err_photon`` without conversion
(~3233) -- documented as separate finding below.

## D1 -- Trend content of comp residuals

Pooled lag-1 autocorrelation (per comp, per star; SS Cam excluded):

| setup | lag-1 median | trend fraction (AM linear) median |
|-------|--------------|----------------------------------|
| g_60_4 | **0.642** | 0.190 |
| i_70_4 | **0.560** | **0.274** |
| r_60_4 | **0.558** | **0.274** |

White-noise expectation ~0. Observed **substantially positive** on all setups -- supports
smooth correlated structure in comp residuals, not independent epoch noise.

**D1.3 colour-slope (C4):** per-target comp colour-offset vs airmass-slope correlations vary;
pooled setups show both |r|>0.5 cases (differential-extinction signature on some anchors) and
near-zero cases. **Present on subset, not universal.**

**g vs i/r:** g has **similar lag-1** (~0.64) but **lower airmass trend fraction** (0.19 vs 0.27).
Underdispersed g pair (chi2 0.11/0.33) shows lag-1 **0.64** -- same autocorrelation mechanism as i/r.

## D2 -- Detrended SEM vs production

Three-way median SEM (mag), V0611:

| setup | ensemble_normalize | LC-implied ensemble | detrend AM | split-half |
|-------|-------------------|---------------------|------------|------------|
| g | 0.0059 | 0.0154 | 0.0047 | 0.230 |
| i | 0.0067 | **0.0482** | 0.0034 | 0.136 |
| r | 0.0054 | **0.0203** | 0.0047 | 0.215 |

**Key finding:** LC-implied ensemble term is **4-7x larger** than ``ensemble_normalize`` output
(ratio i: **7.46x**, g: 2.73x, r: 4.19x). This is evidence that the err assembly path inflates
the effective ensemble contribution beyond the Honeycutt SEM scalar itself.

Airmass-linear detrend **reduces** normalize-level SEM (~0.49-0.78x) but **does not** reconcile
LC-implied ensemble (detrend/LC ratio ~0.07 on i).

**Predicted chi2 (V0611, check_kmag + offline recomposition):**

| setup | chi2 LC err actual | chi2 detrend AM | chi2 split-half |
|-------|-------------------|-----------------|-----------------|
| g | 1.24 | 4.06 (worse) | 0.011 |
| i | 0.24 | 2.16 (worse) | 0.047 |
| r | 0.26 | 1.33 (worse) | 0.006 |

Simple per-comp airmass detrend **does not** move i/r toward chi2=1. Split-half flux-sum test
over-corrects (different estimand than comp-residual SEM).

## D3 -- Split-half ZP test

Split-half empirical SEM (flux-sum halves, sqrt(n/n_half) scaling) is **10-40x larger** than
ensemble_normalize SEM and **3-5x larger** than LC-implied ensemble. It measures a different
quantity (absolute ensemble mag differences between random halves) than comp-residual SEM
(referenced to per-comp night medians). Not directly comparable for chi2 repair without
redefining the estimand.

## D4 -- Cross-cohort

- **g chi2~1 zone:** V0611 g chi2=1.24 healthy; lower AM trend fraction explains partial
  insulation, but lag-1 still high -- g chi2~1 is **not** because residuals are white noise.
- **Underdispersed g pair:** same lag-1 ~0.64 as i/r; same mechanism, stronger LC-implied
  ensemble inflation on those anchors.
- **draft_424 wide:** detrend chi2 0.293 vs LC 0.275 (within tolerance); healthy case preserved.
- **SS Cam (separate):** chi2=122, lag-1=0.67, 2 comps only -- sparse-path anchor mismatch;
  excluded from pooled medians.

## Gates C0-C4

| Gate | Result |
|------|--------|
| **C0** | **PASS** -- manual SEM matches ensemble_normalize to 1e-12 |
| **C1** | **PASS** -- lag-1 + trend fraction distributions per setup |
| **C2** | **PASS** -- three-way comparison reported; detrend does NOT reconcile LC ensemble |
| **C3** | **FAIL** -- neither AM detrend nor split-half moves i/r chi2 toward 1; wide rig OK |
| **C4** | **PASS** -- colour-slope tested; present on subset |

**Verdict:** ``hypothesis_partial_trend_present_am_detrend_insufficient``

Trend/autocorrelation in comp residuals is **confirmed** (supports working hypothesis), but
**simple airmass detrend is not the repair**. Dominant measurable cause of i/r underdispersion
is the **inflation of the LC ensemble term relative to ensemble_normalize** (4-7x on i),
consistent with mag/flux unit inconsistency at err assembly
(``photometry_core.py:3233`` / ``_combine_err_with_ensemble_scatter_keyed`` ~3205).

**Rejected shortcut:** blanket ~0.5x ensemble scale on i/r (reverse-engineering chi2=1).

## Production recommendation (Milan decision -- C3 did NOT pass cleanly)

Do **not** adopt a setup-specific scale factor. Next production investigation should target
**unit-consistent err assembly**: convert ensemble SEM to relative-flux before quadrature with
``err_photon``, or store/combine entirely in magnitude domain. Evaluate impact on draft_424
wide-rig (detrend chi2 0.293 vs 0.275 -- small) before re-anchor. Trend-aware SEM (common-mode
removal before std/sqrt(n)) is a separate follow-up if unit fix alone is insufficient.

## Errors (if any)

None. RuntimeWarnings on empty split-half slices for stars with <4 comps (handled as NaN).

## Files changed

- ``scripts/sem_cause_core.py`` -- NEW pure helpers
- ``scripts/sigma_sem_cause.py`` -- NEW diagnostic runner
- ``tests/test_sigma_sem_cause.py`` -- NEW (6 tests)
- ``docs/VYVAR_ROADMAP.md``, ``docs/VYVAR_STATE.md``, ``docs/VYVAR_JOURNAL.md``
- ``CURSOR_RESULT_sigma_sem_cause.md`` (this file)

Artifacts (not committed): ``tmp/sigma_sem_cause/``

## pytest count

**769 passed**, 15 skipped (was 763 + 6 new tests). ``ruff check`` clean on touched files.
