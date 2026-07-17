CURSOR RESULT -- 2026-07-14 K2-COHORT

What I did
Verified session_baseline PASS on ed59dfd. Built full-cohort k'' signature test (report-only):
k2_cohort_core.py + scripts/k2_cohort_run.py + tests. Ran on archive drafts 424/425/426.
Applied pre-registered FDR rule mechanically. Updated ROADMAP/STATE/JOURNAL/K2_BAND_AWARE_SPEC.

## Pre-registered rule (verbatim, frozen before run)

Family of tests: one per (rig, band). Multiple-testing control: Benjamini-Hochberg FDR
at q = 0.05 across the whole family (both T1 and T2 below).

- k'' priority UP if ANY (rig, band) shows the T1 signature with |rho| >= 0.3,
  q <= 0.05, and the physically expected sign (slope magnitude increasing with colour
  offset; per-filter sign conventions stated in the result).
- k'' priority DOWN only if ALL tested (rig, band) cells are null AND each cell had
  >= 80% power to detect rho = 0.4 at alpha = 0.05 (Spearman power: n >= ~46 per cell).
  Underpowered nulls do not count toward DOWN.
- Otherwise UNCHANGED, with the per-cell power stated.

## Cohort yields per cell

| cell | draft/setup | n_cand | n_stars | N epochs | notes |
|------|-------------|--------|---------|----------|-------|
| wide_CLEAR | 424/NoFilter_60_2 | 149 | 148 | 139 | delta_mag LOO per host target; 87 hosts |
| wide_V | 425/V_20_2 | 0 | -- | max 12 | excluded: N_epochs < 20 gate |
| wide_B | 425/B_20_2 | 0 | -- | max 12 | excluded: N_epochs < 20 gate |
| wide_R | 425/R_20_2 | 0 | -- | max 12 | excluded: N_epochs < 20 gate |
| Newton_g | 426/g_60_4 | 23 | 23 | varies | underpowered (power~0.47) |
| Newton_i | 426/i_70_4 | 19 | 19 | varies | underpowered (power~0.40) |

Mag quantity: delta_mag LOO per host target (consistent within cell). SS Cam excluded.
Sparse-path targets excluded via comp_path=sparse_fallback (0 on draft_424).

## T1 table (primary: b_X vs signed colour offset)

Sign conventions (negative k2_lit filters): expect rho < 0 (redder offset -> more negative b_X).
CLEAR/V: no strong literature sign (|rho| gate only for UP).

| cell | n_T1 | rho | p | q_FDR | k2_eff | k2_eff_se | lever_excl |
|------|------|-----|---|-------|--------|-----------|------------|
| wide_CLEAR | 147 | -0.013 | 0.877 | 0.877 | -0.040 | 1.08e-6 | 0 stars |
| Newton_g | 23 | -0.044 | 0.840 | 0.877 | -0.057 | 2.34e-5 | -- |
| Newton_i | 19 | -0.325 | 0.175 | 0.350 | -0.036 | 3.13e-5 | -- |

95% CI [mag/airmass/mag_colour] from WLS (k2_eff +/- 1.96*k2_eff_se): wide_CLEAR
[-0.040002, -0.039998]; Newton_g [-0.057494, -0.057362]; Newton_i [-0.036010, -0.035888].

**RETRACTED (K2-STATS-FIX):** naive WLS CIs above are invalid (photon-weight SE ignores
overdispersion; 37,000-sigma slope incompatible with rho=-0.013). Superseded by bootstrap CIs
in CURSOR_RESULT_k2_stats_fix.md.

Figures: tmp/k2_cohort/figures/t1_wide_CLEAR.png (and per-cell PNGs where n>=2).

## T2 table (secondary: sigma_r vs |colour| x airmass_range)

| cell | n_T2 | rho | p | q_FDR | sigma_r median [CI] mag |
|------|------|-----|---|-------|-------------------------|
| wide_CLEAR | 148 | -0.193 | 0.019 | 0.114 | 0.00473 [0.00448,0.00500] |
| Newton_g | pzq_ok | -0.137 | 0.533 | 0.799 | -- |
| Newton_i | pzq_ok | +0.470 | 0.043 | 0.128 | -- |

Lag-1 autocorr columns in k2_cohort_summary.json per star (correlated-structure cross-check).

## FDR summary

Family size: 6 tests (3 cells x T1+T2; 425 B/V/R excluded). q=0.05 BH: no rejections.
T2 wide_CLEAR raw p=0.019 but q=0.114 (does not trigger UP; T1 is physics gate).

## Verdict

**k'' priority UNCHANGED** (verbatim pre-registered rule).

Retraction: the initial DOWN verdict misapplied the frozen rule by treating underpowered nulls
as excluded from the "each cell >=80% power" requirement; corrected here without re-running
anything.

Three tested cells: wide_CLEAR (power 0.999, null), Newton_g (power 0.47), Newton_i (power 0.40).
DOWN clause requires each tested cell >=80% power; Newton g/i fail -> **UNCHANGED** with per-cell
power stated (not DOWN).

wide_CLEAR T1 null: rho=-0.013, q=0.877. T2 null after FDR: q=0.114.
Newton g/i underpowered (n=23/19) -- suggestive probes only (see per-rig record in ROADMAP).

Next step (if priority were UP): K2 design review vs K2_BAND_AWARE_SPEC; coefficient fit
awaits Milan BVR night dX>=0.3. Empirical k2_eff seeds recorded in spec (not significant).

## Errors (if any)

None. Runtime ~12 min for full cohort (148 wide stars x host-target LOO).

## Files changed

k2_cohort_core.py (new)
scripts/k2_cohort_run.py (new)
tests/test_k2_cohort_core.py (new)
docs/K2_BAND_AWARE_SPEC.md
docs/VYVAR_ROADMAP.md
docs/VYVAR_STATE.md
docs/VYVAR_JOURNAL.md
CURSOR_RESULT_k2_cohort.md

## pytest count

820 passed, 15 skipped (session_baseline_check --fast on HEAD after commits).

NOT PUSHED -- Milan review first.
