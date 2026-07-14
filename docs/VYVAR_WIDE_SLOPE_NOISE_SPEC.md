# VYVAR WIDE-SLOPE-NOISE SPEC -- what drives the 0.094 mag/airmass slope scatter on the wide rig

Status: APPROVED (Milan, 2026-07-14). Author: Claude, 2026-07-14. ASCII-only.
Data: 100% archival (draft_424; k2-cohort b_X values in tmp/k2_cohort/ + per-epoch proc
x/y/FWHM/flux + LC err). Report-only; no production changes.

## 1. Question and why it matters

K2-COHORT measured per-star airmass slopes b_X with star-to-star scatter SD = 0.094
mag/airmass on the wide rig (n = 147, N ~ 139 epochs). Colour explains <= 0.031 of it.
The dominant source is unidentified and large. Three previously separate numbers may be
faces of the same thing: this slope scatter, the PZQ sigma_r ~ 5.5 mmag [4.7, 6.5], and
the ~4.5 mmag unexplained rig constant (SIGMA-A3/A4). If the source is identified and
actionable (e.g. flat residuals x field drift), it defines the next real precision gain
on the wide rig; if it is measurement noise, the mystery dissolves and we close it.

## 2. Hypotheses (pre-registered, with physically calibrated effect sizes)

H0 -- **Measurement noise.** b_X is a regression slope; its sampling error from per-epoch
photometric noise is sigma_b ~ sigma_pt / (sqrt(N) * SD(X)). With tertile err medians
0.015-0.086 mag (post-floor-era values from the anchor), faint-star slope SEs alone reach
~0.03-0.04 mag/airmass. PREDICTION: excess (systematic) slope variance is computed per
brightness tertile as SD_obs^2 - median(SE^2); H0 is the fraction attributable to noise.
This is computed FIRST -- every later test runs on the excess, not the raw scatter.

H1 -- **Flat/vignetting residuals x field drift.** If a star drifts D pixels over the
night along a flat-error gradient of amplitude eps, and the drift correlates with airmass
(alt-az trajectory does), the star acquires a spurious airmass slope
~ eps_grad * D * corr(pos, X). Effect size is COMPUTED from **detector-frame** positions
on calibrated pre-alignment lights (aligned proc x/y are alignment residuals, not physical
drift). Per-epoch detector positions via cutout DAO centroid chain (path c when alignment
shifts are not archived).

H2 -- **FWHM/aperture coupling.** Focus/seeing varies with time (and correlates with X);
aperture losses depend on FWHM and on brightness (fixed-radius apertures). PREDICTION:
b_X correlates with each star's sensitivity d(mag)/d(FWHM) (estimable per star from
epoch-level mag-vs-FWHM partial regression at fixed X), stronger for stars where
aperture_r / FWHM is small.

H3 -- **Brightness-dependent background/nonlinearity.** PREDICTION: |b_X| or excess
variance trends with magnitude beyond the H0 noise prediction, and mag-vs-sky-level
partials are nonzero.

H4 -- **Colour (known, bounded).** Included as a control regressor; contribution already
bounded <= 0.031 (K2-STATS-FIX). Its fitted share must come out consistent with that
bound -- a consistency check on the whole method.

## 3. Method

P1 **Noise floor (H0):** per star, slope SE two ways -- (a) propagated from per-epoch LC
err via sqrt(1 / sum(w * (x - xbar_w)^2)) with w = 1/err^2 (NOT the WLS residual SE,
which is deflated on mean-detrended mags); (b) residual bootstrap of the b_X fit
(>= 1000 draws). se_use = max(analytic, bootstrap) when both finite. Per brightness
tertile (lower mag_g = brighter): excess variance = SD_obs^2 - median(SE^2), with a
bootstrap CI on the excess. Everything downstream uses excess; a tertile whose excess
CI includes zero is CLOSED as noise-dominated.

P2 **Regression on excess-bearing stars:** weighted model
b_X ~ colour + x + y + x^2 + y^2 + xy + r^2 + drift_X_corr + det_drift_X_corr +
det_drift_span + fwhm_sens + mag,
weights 1/SE^2 with no ad-hoc SE floor (WSN-FIX: removed SD/5 policy); overdispersion-scaled
Type-II partial SS + 10-fold CV R^2 per term group alongside in-sample shares.

P3 **Physical effect-size table (pre-registered before P2 runs):** attainable b_X
contribution per hypothesis from measured inputs -- drift spans, FWHM ranges, flat-error
scenarios {0.3%, 1%}. Any term whose attainable maximum is below the measurement floor
is marked untestable-here (not "absent").

P4 **Cross-checks:** (a) integrate the fitted positional/FWHM terms over the night ->
predicted per-point correlated noise; compare against PZQ sigma_r 5.5 mmag and the
4.5 mmag rig constant (order-of-magnitude consistency, stated numerically). (b) The
colour term must respect the K2 bound (H4).

P5 **Pre-registered outcomes:**
- A term group explains >= 50% of excess variance with q <= 0.05 -> named dominant source;
  action recommendation follows its physics (flat quality -> Milan's bin2-flats data task
  gains a concrete "why"; positional/FWHM -> a literature-grounded EPD/SysRem-style
  decorrelation becomes a candidate workstream, decision NOT taken here).
- Excess mostly noise (H0) -> mystery closed; record and stop.
- Excess real but unattributed (all groups < 50%) -> record honestly; park with the
  measured bounds per hypothesis.

## 4. Deliverables

tmp/wide_slope_noise/ artifacts (per-star table, decomposition JSON, figures: b_X field
map, excess-by-tertile, per-term shares); result MD with the P3 table printed BEFORE the
P2 results; ROADMAP/STATE/JOURNAL updates. New pure helpers tested with hand-computed
cases; K2 stats machinery reused, not duplicated.

Implementation: `wide_slope_noise_core.py`, `scripts/wide_slope_noise_run.py`.

## 5. Non-goals

No production changes; no detrending of any science output; no new photometry; no
threshold tuning. Newton is out of scope (underpowered; revisit with more nights).
