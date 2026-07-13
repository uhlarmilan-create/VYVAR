CURSOR RESULT -- 2026-07-13 12:42 UTC+2

What I did
Part A: wired empirical proc ``sigma_bkg_ap`` into ``sigma_arrays_from_lc_and_proc`` (Howell-family
variants) and added acceptance-authoritative ``production_lc_err`` variant. Part B: ran
``scripts/sigma_newton_run.py`` on draft_426 (g/i/r, V0611 + 6 pooled check stars per setup);
produced decomposition-driven Newton baseline and anomaly attribution artifacts under
``tmp/sigma_newton/``.

## Part A -- SIGMA-BUDGET-EMPIRICAL harness

- ``sigma_arrays_from_lc_and_proc`` now reads ``sigma_bkg_ap`` + ``err_bkg_source`` from proc rows
  and calls ``_photometric_error_with_bkg_mode`` (empirical) when finite; analytic Howell fallback
  when absent, recorded as ``bkg_term_source.primary = analytic_fallback``.
- New variant ``production_lc_err``: ``sigma_mag = _MAG_ERR_SCALE * lc_df['err']`` (same as
  ``bingain_fix_validate._chi2_lc_err``).
- Module docstring documents authoritative vs attribution variants.
- Tests: 3 new cases in ``tests/test_sigma_budget.py`` (empirical bkg, analytic fallback,
  production_lc_err hand chi2).

## Part B -- per-setup tables (production_lc_err, draft_426)

| setup   | V0611 chi2/dof [CI]      | pooled check chi2 median | V0611 ensemble share |
|---------|--------------------------|--------------------------|----------------------|
| g_60_4  | 1.23 [0.90, 1.51]        | 4.67 (2.95 excl. SS Cam outlier) | 84% |
| i_70_4  | 0.238 [0.17, 0.28]       | 0.81                     | 91% |
| r_60_4  | 0.253 [0.19, 0.30]       | 0.63                     | 85% |

Harness ``bkg_term_source``: 100% ``empirical`` on all three setups (168/175/150 frame rows).

Artifacts: ``tmp/sigma_newton/sigma_newton_summary.json``, per-star JSON/plots under
``tmp/sigma_newton/<setup>/``, ``underdispersion_ir.json``, ``g60_heterogeneity.json``.

## Anomaly 1 -- i/r underdispersion (V0611 chi2 ~ 0.25, ensemble ~91%)

EVIDENCE (V0611 i_70_4, ``tmp/sigma_newton/underdispersion_ir.json``):

1. **Empirical vs modeled:** LC ``err`` median mag = 0.0554 vs empirical kmag scatter = 0.0265;
   ratio **2.09** (``err_over_empirical_scatter``). Predicted chi2 if err were 2x too large:
   **0.228** vs observed **0.238** -- fully explains underdispersion.
2. **SEM vs scatter:** ``sem_over_raw_scatter_median`` = ``inv_sqrt_n_comps_median`` = 0.378
   (n_comps ~ 7). Production implements SEM correctly as std/n^(1/2)
   (``photometry_core.py:3113-3115``), not raw scatter.
3. **Double-count:** comp photon share of ensemble term median **0.32%** -- negligible.
4. **Decomposition:** ensemble_share median **91.0%** (i), **84.6%** (r); background < 11%.

**Attribution:** dominant cause is **ensemble SEM term magnitude in LC err** (~2x empirical
epoch scatter on i/r check-star cohort). Implementation is SEM-not-scatter (not a sqrt(n) bug).

**Milan decision (PROD, not implemented):** candidate minimal fix = scale per-frame ensemble
scatter in ``_combine_err_with_ensemble_scatter_keyed`` by setup-specific factor ~0.48-0.5 on
i/r (or revisit comp_ref_map / comp count weighting). Requires re-anchor if adopted.

## Anomaly 2 -- g_60_4 pooled heterogeneity (pooled ~2.95 vs V0611 g ~1.23)

Per-star ``production_lc_err`` chi2 on g_60_4 (6 check stars + V0611):

| catalog_id (tail) | chi2/dof | ensemble share | saturation fill_p95 |
|-------------------|----------|----------------|---------------------|
| 1112113066119992064 (SS Cam) | **122.1** | 98% | 0.088 |
| 1111749368289526912 | 5.34 | 84% | 0.031 |
| 1112130898824233216 | 4.67 | 96% | 0.042 |
| V0611 | 1.23 | 84% | 0.026 |
| 1111931646701447424 | 0.33 | 84% | 0.025 |
| 1112121175018240768 | 0.11 | 84% | 0.025 |

**Verdict:** **bimodal subset** -- not broad-based. Pooled median **2.95** (matches regate seed)
is elevated by **one extreme overdispersed outlier (SS Cam, chi2=122)** plus two moderate
overdispersed stars (chi2 4.7-5.3). Underdispersed tail (chi2 0.11-0.33) mirrors i/r ensemble
SEM inflation pattern. V0611 g is healthy (chi2 ~1.23). SS Cam overdispersion likely sparse-comp /
anchor mismatch (98% ensemble share, not saturation-limited).

Figure: ``tmp/sigma_newton/g60_heterogeneity_hist.png``.

## Gate verdict N1-N4

| Gate | Result |
|------|--------|
| **N1** Newton baseline | **DEFINED.** production_lc_err chi2 per setup above (with bootstrap CI). No pass/fail threshold applied. |
| **N2** i/r underdispersion attribution | **PASS (conclusive).** Ensemble SEM ~2x empirical scatter; chi2_predicted 0.228 vs observed 0.238 (i). |
| **N3** g_60_4 heterogeneity | **PASS (localized).** SS Cam + 2 moderate stars drive high pooled chi2; underdispersed pair separate cause. |
| **N4** harness sanity | **PASS.** draft_424 NoFilter_60_2 check star 1485540612577549568: harness chi2=0.275 matches ``bingain_fix_validate`` ref exactly; within CI. |

## Errors (if any)

None. Initial V0611 run used raw LC mags (chi2~133 on g); fixed to prefer ``check_kmag`` sidecar
(same path as regate). Re-run confirmed seed numbers.

## Files changed

- ``sigma_budget.py`` -- ``SIGMA_VARIANT_PRODUCTION_LC_ERR``
- ``scripts/chi2_sigma_gate.py`` -- empirical bkg, production_lc_err, bkg_term_source meta
- ``scripts/sigma_newton_run.py`` -- NEW Part B runner + diagnostics
- ``tests/test_sigma_budget.py`` -- 3 new tests
- ``docs/VYVAR_ROADMAP.md``, ``docs/VYVAR_STATE.md``, ``docs/VYVAR_JOURNAL.md``
- ``CURSOR_RESULT_sigma_newton.md`` (this file)

## pytest count

**763 passed**, 15 skipped (was 760 + 3 new tests). ``ruff check`` clean on touched files.
