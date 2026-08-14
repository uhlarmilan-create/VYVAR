CURSOR RESULT - U-SCATTER-DEF (2026-08-14)

Register ID: U-SCATTER-DEF (new)
Related: U-XVAL-COMP-RMS (retracted), D12-1, Q1-XVAL-MATCHED
Register diff for Part A: `dev/results/REGISTER_DIFF_U_SCATTER_DEF_A.md` (awaiting authorization)

---

## Part A -- Register corrections

See `REGISTER_DIFF_U_SCATTER_DEF_A.md` for exact diff. Summary:

- **A1:** U-XVAL-COMP-RMS -> RETRACTED with superseding pointer to Q1-XVAL-MATCHED; add Q1 row CLOSED.
- **A2:** Correction note on conflated metrics (check_scatter vs comp_rms_dao; fleet median vs per-target gaps).
- **A3:** Open U-SCATTER-DEF.

---

## Part B1 -- Scatter / RMS inventory

Quantities reported, logged, exported, or gated. **Publication-relevant** rows marked [PUB]. **Gate** rows marked [GATE].

| ID | Name / column | Formula | Space | Estimator | Stars | Code | Consumers |
|----|---------------|---------|-------|-----------|-------|------|-----------|
| S1 | `check_scatter` | `nanstd(kmag, ddof=1)` on check-star sidecar epochs | mag (check kmag) | sample std | check star (not comps) | `trust_flag_core.py:84-97` | [GATE] trust JSON, sparse-trust path |
| S2 | `comp_rms_dao` / `comp_rms_phot` | LOO: for each comp c, `sclip_std(diff_series(w,c,others))`; median over comps | flux -> diff mag | iter 3-sigma clip then population std | comps per target | `xval_harness_core.py:24-93` | xval CSV only (offline) |
| S3 | `comp_qa` LOO series | `m_focus - (-2.5*log10(sum 10^{-0.4 m_others})))`; demeaned | **mag** | indices on series (sigma_iqr, inv_nv) | comps | `comp_qa_core.py:76-97` | comp QA JSON, pipeline QA |
| S4 | `robust_comp_rms` / Phase-1 `comp_rms` | `1.4826 * median(|f_i - median(f)|)` on detrended relative flux | relative flux | MAD-scaled | comp candidates | `comp_frame_normalize.py:153-179`; `comp_selection_per_target.py:1482` | [GATE] comp selection, Broeg weights, [PUB] export `comp_rms` column |
| S5 | `comp_rms_fieldwide` | Same pipeline as S4 on field-wide pool (no per-target detrend) | relative flux | MAD-scaled | field pool | `comp_selection_per_target.py:2077-2081` | sparse-trust labelling |
| S6 | `lc_rms` | `std(mag_calib)` on target calibrated LC | mag | sample std (ddof=0 implicit) | target | `photometry_core.py:10405-10435` | [PUB] photometry_summary, reports |
| S7 | `lc_rms_ooe` | RMS excluding flagged epochs | mag | std on subset | target | `compute_lc_rms_ooe` `3096+` | summary column |
| S8 | `rms_p2p` / stability p2p | peak-to-peak or std of comp LC after stability check | mag | varies | comps | `photometry_core.py:3148-3193` | comp quality JSON, stability gate |
| S9 | `differential_lc_rms` | Broeg diff mag per frame; 3-sigma MAD clip; std | mag | clipped std | target vs ensemble | `validate_lc_crossval.py:350-389` | crossval script only |
| S10 | PyTICS inflation | Iterative inflation of S4 map from ensemble residuals | relative flux | iterative | comps | `pytics_iterative_weights` `2875+` | ensemble weights |
| S11 | `sigma_iqr` / Sokolovsky | `(P90-P10)/1.349` on LOO diff mag series | mag | IQR-based | per comp in QA | `comp_qa_core.py:100+` | comp QA |
| S12 | global pool RMS | Same as S4 on nightly relative flux per star | relative flux | MAD | field | `comp_pool_rms.py:356-380` | Phase-1 pool ranking |
| S13 | `_robust_scatter_mad` | MAD-based scatter helper | generic | MAD | internal | `photometry_core.py:728` | color-term diagnostics |
| S14 | `comp_loo_std` (Q1 T4) | LOO diff mag; `std(ddof=1)` no clip | mag | plain std | comps | diagnostic | Q1 memo only |

**Additional non-scatter but often confused:**

- `sky_adu_per_px_annulus` -- per-star sky level (ADU/px), not scatter.
- `sigma_bkg_ap` -- background noise for error model, not comp precision.

**Completeness note:** Grep covered `src_py/*.py` for scatter/rms/dispersion/p2p/MAD/sclip patterns. UI display columns mirror export names. No additional production **comparison-star scatter gate** beyond S1 (check star), S4 (comp selection), S8 (stability).

---

## Part B2 -- External implementations

| Tool | Comp / ensemble scatter | Space | Estimator | Rejection | LOO? | Source |
|------|-------------------------|-------|-----------|-----------|------|--------|
| **Honeycutt 1992** | Ensemble reference light curve; scatter of comps not uniquely standardized | mag | often std of O-C | varies | no fixed LOO | PASP 104,435 |
| **Broeg 2005** | Weight by comp stability `1/rms^2` | flux/mag | RMS of comp vs ensemble | iterative | no | Broeg et al. 2005 |
| **VaST** | Per-object lightcurve sigma, clipped sigma, MAD, IQR, RoMS | mag | multiple indices | clipped variants | no | Sokolovsky & Lebedev 2017 |
| **AstroImageJ** | Manual ensemble; user std of comp differential mags | mag | std | optional | no | AIJ docs; SNU AO tutorial |
| **C-Munipack** | Ensemble photometry scatter in validation | mag | RMS-like | configurable | no | C-Munipack manual |
| **SExtractor/sep** | Background RMS map; not comp-scatter | counts | RMS of background | sigma-clips in BG | N/A | Bertin & Arnouts 1996 |
| **IRAF apphot** | No standard comp scatter | N/A | N/A | N/A | N/A | IRAF apphot |
| **photutils docs** | No comp scatter; tutorials use `np.std(..., ddof=1)` on ZP samples | mag | std | user choice | no | photutils + SNU AO |

**Disagreements among tools:** VaST offers >10 scatter indices; AIJ/C-Munipack typically use plain std; VYVAR S4 uses MAD-scaled estimator; xval S2 uses clipped std on LOO flux-space diffs. **No majority convention** for LOO vs fixed ensemble.

**VYVAR-specific:** None of the external tools use exactly S2 (flux-space LOO + sclip_std) or S4 (MAD on detrended relative flux) as the primary published precision metric.

---

## Part B3 -- Estimator properties at N=134 frames, 5 comps

| Estimator | Bias at small N | Efficiency vs normal std | At our N | D12-1 link |
|-----------|-----------------|--------------------------|----------|------------|
| `std` (ddof=1) | unbiased for i.i.d. normal | 1.0 (UMVUE under normality) | n=134: SE ~ s/sqrt(133) ~ 0.087s | none |
| MAD x 1.4826 | biased low for normal (~0.86-0.95 factor at small n) | ~37% efficiency vs std | n=134 per star: stable | project already uses for S4 |
| `sclip_std` 3-sigma | **biased low** (truncated tail) | not defined (adaptive) | Q1 T4: **~0.5 mmag** vs plain std at matched flux | **D12-1: uncorrected clip bias** |
| LOO vs fixed ensemble | LOO removes self-influence; slightly wider | comparable | 5 comps: LOO vs full ensemble ~20% df effect | method choice |
| flux-space vs mag-space LOO | nonlinear transform; differ at large scatter | N/A | not quantified here | S2 vs S3 structural difference |

**References:** Hodges & Lehmann 1952 (MAD efficiency); Croux & Rousseeuw 1992 (breakdown); standard error of std ~ s/sqrt(2(n-1)) for normal.

**Correlated frames:** All estimators assume epoch series; 134 frames are sequential (not independent). Effective DOF < 134 for scatter SE; block bootstrap (Q1 T3) appropriate for comparing estimators on same frames.

---

## Part B4 -- Recommendation (proposal only; no implementation)

### Publication metric (single reported comparison-star precision)

**Recommend: S3-style LOO differential magnitude scatter with plain sample standard deviation (ddof=1), demeaned, median over comparison stars.**

Formula: for each comp c, compute LOO diff mag series (mag space, as `comp_qa_core.loo_diff_series`); take `std(ddof=1)`; report **median over comps** and **N frames used**.

**Justification:**

1. **B2:** Closest to AIJ / tutorial practice (std of differential mags) while using LOO (standard for comp self-exclusion).
2. **B3:** Unbiased std at n=134; avoids D12-1 clip bias from S2.
3. **Separates roles:** Target precision remains `lc_rms` (S6) for science; comp precision is explicitly ensemble stability.

### Keep as internal diagnostics (distinct names)

| Current | Rename / label | Role |
|---------|----------------|------|
| S1 `check_scatter` | `check_star_scatter` | check-star trust only |
| S2 xval metrics | `xval_comp_rms_*` | offline harness |
| S4 `comp_rms` | `comp_stability_mad` | Phase-1 selection gate |
| S8 p2p | `comp_p2p_stability` | stability QA |
| S2 sclip variant | `comp_loo_scatter_clipped` | sensitivity diagnostic only |

### Defensible alternative

If referees expect robustness to outliers: report **MAD-scaled LOO scatter** (S3 series + MAD estimator) alongside plain std, with explicit label. S4 MAD on relative flux remains valid for **selection** but measures a different quantity (detrended flux stability, not differential mag LOO).

---

## Files changed

- `dev/results/CURSOR_RESULT_U_SCATTER_DEF.md` (this memo)
- `dev/results/REGISTER_DIFF_U_SCATTER_DEF_A.md` (register diff for authorization)

No production code modified.
