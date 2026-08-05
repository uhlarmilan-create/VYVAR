CURSOR RESULT - 2026-08-04 (WIDE-ERR W1+W2)

What I did
Ran the pre-registered W1 discriminator and W2 N_eff test on the restored July
generation of draft_000435_snapshot_skysurface_20260716, check star
1499906247391001088 (G=8.743, 164 sidecars). Production-path LCs recomputed via
photometer_check_star_production_path with all writes under tmp/wide_err_w1w2/.
Harness: dev/tools/wide_err_w1w2.py; machine JSON: tmp/wide_err_w1w2/wide_err_w1w2.json.

## Scope statement

ensemble_normalize and sigma_floor_core are functionally identical between git
10d610c (July snapshot producer) and origin/main; repository diff on those modules
is ASCII-migration / hygiene only. Conclusions about the ERROR MODEL therefore carry
to current code. Conclusions about the comp POPULATION do not -- that is
WIDE-ERR-POP-DELTA (July 333 core LCs / 166 check sidecars vs mutated August
1121 / 248).

## Guard rails

| check | result |
|-------|--------|
| anchor_manifest_check.py (pre-run) | PASS (exit 0) |
| wide_error_budget_diag.py:186 -> tmp/ | confirmed |
| wide_err_step0_checkstar.py:40 -> tmp/ | confirmed |
| anchor_manifest_check.py (post-run) | PASS (exit 0) |

No snapshot mutation detected. All diagnostic LC output under
tmp/wide_err_w1w2/diag_check_lc/ (163 fields cached).

## W1 -- Pre-registered discriminator

Substrate: 164 sidecars with check_catalog_id 1499906247391001088; 163 fields yielded
valid production-path LCs (>=3 epochs with mag_calib_final and err); 1 field skipped
(photometry path error on comp stability).

| metric | median | IQR (25 / 50 / 75) |
|--------|--------|---------------------|
| sigma_total / err | 1.97 | 1.71 / 1.97 / 2.29 |
| sigma_total_robust / err | 1.83 | 1.54 / 1.83 / 2.10 |
| sigma_p2p / err | 1.83 | 1.59 / 1.83 / 2.06 |
| sigma_p2p_robust / err | 1.69 | 1.48 / 1.69 / 1.96 |
| sigma_total (mmag) | 20.3 | -- |
| sigma_total_robust (mmag) | 17.8 | -- |
| quoted err (mmag) | 9.4 | -- |
| n_epochs per field | 139 | -- |

Outlier census (5-sigma MAD clip on mag_calib_final): **37** points rejected across
163 fields (18 fields with >=1 outlier). Non-robust and robust ratios differ modestly
(total 1.97 vs 1.83; p2p 1.83 vs 1.69) -- not the OUTLIERS split pattern (robust ~1).

Detrending (per field, median across fields):

| metric | median ratio / err after detrend |
|--------|----------------------------------|
| linear trend removed | 1.94 |
| quadratic trend removed | 1.93 |
| median linear slope (mag/day) | -0.030 |
| median linear F-test p (vs constant) | 0.135 |

Detrending does **not** collapse the ratio toward 1 (1.97 -> 1.94 linear, 1.93 quad).

### W1 verdict (pre-registered list)

**WHITE -- error model underquotes.** Both robust ratios sit near ~1.7-1.8 (IQR spans
~1.5-2.1), matching the original ~1.96 figure on this generation. The robust pair
does **not** show p2p~1 with total~2 (RED), and robust ratios do **not** collapse to
~1 while non-robust stay ~2 (OUTLIERS). Detrending leaves the ratio elevated -> not
TREND-driven on this sample.

## W2 -- Flux-weighted ZP vs equal-weight SEM (N_eff)

Per target, per frame: N comps entering the flux sum, F_i = 10**(-0.4 m_i), N_eff,
predicted factor sqrt(N/N_eff). Comp inst mags from proc CSVs (read-only on Archive).

| metric | median | IQR (25 / 50 / 75) |
|--------|--------|---------------------|
| N (comps per frame) | 8 | 8 / 8 / 8 |
| N_eff | 7.30 | 6.14 / 7.30 / 7.81 |
| sqrt(N / N_eff) predicted | 1.04 | 1.01 / 1.04 / 1.08 |
| comp mag spread (max-min, mag) | 0.92 | 0.53 / 0.92 / 1.34 |
| comp mag spread p90 (mag) | 0.95 | 0.60 / 0.95 / 1.39 |

Per-field Spearman rho( predicted sqrt(N/N_eff), observed sigma_total_robust/err ):
**rho = -0.23, p = 0.003, n = 163** (weak **negative** correlation, not positive).

Median predicted factor **1.04** vs median observed robust ratio **1.83**. Residual gap
(unexplained after N_eff): **~0.79 ratio units** (~18 mmag at median err 9.4 mmag).

Rough sanity (8 comps): spread ~0.9 mag -> N_eff ~7.3, factor ~1.04; spread ~2 mag would
give factor ~1.15-1.40. A factor of 2 would require one comp to dominate the flux sum --
not seen (median spread 0.92 mag, N_eff close to N).

### W2 verdict (pre-registered list)

**Hypothesis rejected.** Predicted sqrt(N/N_eff) ~1.04 does **not** approximate the
observed ~1.8-2.0 underquote; Spearman correlation is weak and **negative**. The
flux-sum / equal-weight SEM mismatch is a real structural difference in the code but is
**not** the dominant mechanism for this check star on the July generation. N_eff
explains **none** of the gap (at most ~4% of the ratio).

## Combined line

**WIDE-ERR-ERRORMODEL** -- white underquote on the restored July tree; the ~1.96
sigma_total/err figure stands for 1499906247391001088 (163/164 fields). N_eff /
flux-weight mismatch explains none of it; detrending and outlier removal do not
dissolve it. Root cause remains in the error model assembly, not check-star variability,
bad epochs, or comp-count weighting alone.

## Errors

One field skipped (ValueError in check_comparison_stability: mismatched comp LC lengths).
163/164 successful.

## Files created

- dev/results/CURSOR_RESULT_wide_err_w1w2.md (this file)
- dev/tools/wide_err_w1w2.py (harness; output under tmp/ only)
- tmp/wide_err_w1w2/wide_err_w1w2.json

## Note on diag writes

This run uses photometer_check_star_production_path with lc_dir under tmp/ only.
Post-run manifest PASS confirms no Archive photometry mutation. This does **not**
account for historical 10:20-11:47 masterstars / comparison_stars / check_kmag writes
(quarantined tree); it accounts only for ensuring this diagnostic did not repeat them.
