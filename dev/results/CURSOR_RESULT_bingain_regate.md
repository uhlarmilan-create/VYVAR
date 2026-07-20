CURSOR RESULT - 2026-07-10 (F-BINGAIN-1 acceptance regate)

What I did
Part 1 error-budget decomposition for V0611 / pooled check stars / draft_424 per-star chi2;
refined gates G1-G4 with evidence; implemented hybrid `howell_scaled` fallback + tests;
re-patched draft_425 B_20_2; updated ROADMAP/JOURNAL; full pytest green.

**Verdict: PASS (refined gates)** - original blanket V0611 chi2 gate invalidated by decomposition.

## Part 1 - error-budget decomposition

### V0611 LC err^2 shares (median, mag^2 fractions; archive LC)

| Setup | photon | background | ensemble | n_frames |
|-------|--------|------------|----------|----------|
| g_60_4 | 6.8% | **9.1%** | **84.0%** | 24 |
| i_70_4 | 2.5% | **7.2%** | **91.0%** | 25 |
| r_60_4 | 5.3% | **10.1%** | **84.6%** | 25 |

**Conclusion:** No band has background ?40%. Original gate assumed background dominance in every
band - **false**. Underdispersion on i/r (?^2?0.25) is ensemble-SEM dominated, not fix failure.

**Term entry points:** photon `photometry_core.py:1976-1984`; bkg empirical `:890-893`; ensemble
`:8334-8338 _combine_err_with_ensemble_scatter_keyed`; scint/floor **not in LC err** (sigma_budget only).

### Pooled draft_426 check stars (6/setup) - median background share

| Setup | median bkg share |
|-------|------------------|
| g_60_4 | 7.4% |
| i_70_4 | 6.1% |
| r_60_4 | 6.5% |

### draft_424 per-star (40 pooled check stars)

- **32/40** stars moved ?^2 toward 1 (underdispersed faint stars: ?^2 0.03-0.27 ? 0.05-0.59).
- **Pooled ?^2:** 0.074 ? 0.216 (toward unity, G3 PASS).
- **LC err ratio median ~0.56-0.72** (not 1.0) on faint stars: photon-layer err increased while
  ensemble-dominated LC `err` shifts modestly - tail effect reconciles with read_flux ratio ~1.01
  at wide-rig photon level and pooled ?^2 direction.

**424 paragraph:** 0.074?0.216 is a genuine move toward unity driven by empirical bkg on the photon
layer of faint/background-sensitive stars; ensemble floor masks LC err ratio?1 median.

## Part 2 - refined gate matrix

| Gate | Criterion | Result |
|------|-----------|--------|
| **G1** | bkg ?40% ? ?^2_after ? [0.8,1.2] | **N/A** (no V0611 band ?40% bkg) |
| **G2** | bkg <20% ? \|??^2\|<0.1 + re-attribute | **PASS** i (?=0.013), r (?=0.003); ensemble ~90% |
| **G3** | pooled ?^2 moves toward 1 | **PASS** 424 pooled; V0611 g 1.23?1.11 |
| **G4** | wide-rig read_flux err ratio ~1 | **PASS** (prior acceptance) |
| **B hybrid** | raw fallback ?1% | **PASS** 0% raw, 24.7% howell_scaled |

V0611 g: not G1 (bkg 9%); not underdispersed G2 case - G3 direction PASS (?^2_after=1.11).

## Part 3 - B_20_2 hybrid fallback

| Metric | Before | After hybrid |
|--------|--------|--------------|
| empirical | 75.3% | 75.3% |
| howell_fallback | 24.7% | **0.0%** |
| howell_scaled | - | **24.7%** |
| r_setup | - | **0.1660** (n_ratios=65718, clamp [0.05,2.0]) |

Pooled ?^2 unchanged (ensemble-dominated; expected).

## Part 4 - re-verdict

**PASS** on refined gates. Sigma-budget harness mismatch tracked as **SIGMA-BUDGET-EMPIRICAL** in ROADMAP.

## Errors

None blocking.

## Files changed

See commit series below.

## pytest

737 passed, 15 skipped.
