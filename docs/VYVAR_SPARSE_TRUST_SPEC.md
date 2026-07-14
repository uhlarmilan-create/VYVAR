# VYVAR SPARSE-TRUST SPEC -- check-star ensembles and trust bands for sparse comp fields

Status: **IMPLEMENTED (2026-07-14).** Arc CLOSED. Author: Claude, approved direction
by Milan 2026-07-13; SS Cam YELLOW confirmed Milan 2026-07-14. ASCII-only. All symbols in magnitude domain unless stated; conversions
via canonical MAG_ERR_SCALE only.

Changelog:
- 2026-07-14 Amendment 1 (SPARSE-CHECK-POOL): external check-star K sourcing (section 2.1),
  revised n semantics (K external; n=1 branch with 2-star R test), scope guard, S3 revision.

## 1. Scope and motivation

Current behavior: `compute_check_ensemble_mag_calib` returns None when the good-comp pool
has n < n_comp_min (=3), so sparse fields (draft_426 r_60_4: n=2, legitimately sparse per
COMP-POOL-R) get NO check-star validation and no trust evaluation beyond field-wide
heuristics. Separately, the sparse trust heuristic historically quoted field-wide comp_rms
(0.35-1.0 mag) against per-target gates -- physically wrong, since ~95% of field-wide
structure cancels in the differential (sparse-comp diag, 2026-07).

This spec defines: (a) a check-star ensemble valid at n >= 2 with honest uncertainty,
(b) a per-target, CI-based trust band replacing field-wide headlines, (c) the statistics
required for small n (comps) and moderate N (epochs).

Grounding: Howell, Warnock & Mitchell 1988 (AJ 95, 247) -- variance analysis of
differential CCD time series with photon-noise correction for unequal-brightness stars;
Honeycutt 1992 (ensemble ZP); standard small-sample statistics (c4 bias, chi-square CI
for variance, F ratio). Red-noise context: Pont, Zucker & Queloz 2006 (sigma_r does not
average down; see SIGMA_FLOOR_SPEC).

## 2. Definitions

Per setup and night: target V (variable; excluded from all estimators below), check star K,
good comps C_1..C_n (n >= 2), N epochs (frames) after standard gating.

- m_X(t): calibrated instrumental magnitude series of star X.
- d_XY(t) = m_X(t) - m_Y(t): differential series; s2_XY = sample variance of d_XY over the
  night (ddof=1).
- p_X(t): per-epoch photon+bkg sigma of star X in mag (from proc: rel-flux err *
  MAG_ERR_SCALE). pbar2_X = mean over epochs of p_X(t)^2.
- ZP(t): flux-sum ensemble zero point from C_1..C_n; weights w_i(t) = F_i(t)/sum_j F_j(t).
  For small errors, sigma_ZP(t)^2 = sum_i w_i(t)^2 * sigma_i(t)^2 where sigma_i is the
  TOTAL per-epoch noise of comp i (photon + its share of unmodeled noise).
- kmag(t) = m_K(t) - ZP(t) (existing definition).

### 2.1 Check-star sourcing (Amendment 1)

K is selected from stars that are: (a) constant (VSX-negative and low variability index
per existing check-star criteria), (b) Phase-2A p2p quality good, (c) above the SNR floor
used for comps, (d) NOT a member of the comp ensemble C_1..C_n. The Phase-1 colour window
and tier membership are NOT required for K; tier-excluded stars are eligible.

Selection order among eligible candidates: closest in brightness to the target (comparable
photon noise), tie-break by lowest p2p. The sidecar records: `k_source`
(`comp_pool_external` / `tier_excluded` / ...), `k_colour_offset` (BP-RP offset from the
ensemble median), and `k_tier_excluded` flag.

Colour caveat: an external K with large colour offset can carry a differential-extinction
trend of its own; an elevated R (3.4) may then reflect K's colour term, not ZP failure.
Therefore: (i) `k_colour_caveat = true` when |k_colour_offset| exceeds the comp colour
window AND the night's airmass range > 0.2; (ii) report R both raw and after a linear
airmass detrend of kmag -- the BAND uses raw R (no silent detrending of evidence), the
detrended value is a diagnostic column only.

n counts ENSEMBLE COMPS ONLY (K is external and never counted):

- n >= 2: full sparse branch (triangulation 3.1-3.3, tests 3.4-3.5, band 4) -- reachable
  on an n=2 pool when K is external.
- n == 1: triangulation impossible (single difference series K-C1 cannot separate
  sigma_K from sigma_C1). kmag = m_K - m_C1 IS produced, with the 2-star model test:
  R = var(d_KC1) / mean_t(p_K^2 + p_C1^2 + sigma_sys_rig^2 [+ both stars' floor share]),
  chi2 CI as in 3.4. Band capped at YELLOW (`single_comp`), but R [R_lo, R_hi] is
  recorded -- numbers instead of NaN. RED is still possible downward-only in reporting
  language ("YELLOW, evidence consistent with RED") but the band value stays YELLOW.
- n == 0: no check output; trust capped YELLOW with reason `no_comps` only if the target
  LC itself exists (edge case; do not fabricate).

Scope guard (Amendment 1): external-K sourcing applies to the SPARSE branch (n <= 2
ensemble) in this iteration. The n >= 3 path keeps its current check-star selection
unchanged (unifying K sourcing across all n is a FUTURE item -- changing wide-rig check
stars would invalidate S2 comparisons and is not needed now).

## 3. Estimators (n = 2 branch; generalizes to small n)

### 3.1 Pairwise variance triangulation (Howell 1988)

With three constant-star series K, C1, C2 and independent per-star noise:

    s2_KC1  ~ sig2_K + sig2_C1
    s2_KC2  ~ sig2_K + sig2_C2
    s2_C1C2 ~ sig2_C1 + sig2_C2

    sig2_K_hat  = (s2_KC1 + s2_KC2 - s2_C1C2) / 2
    sig2_C1_hat = (s2_KC1 + s2_C1C2 - s2_KC2) / 2
    sig2_C2_hat = (s2_KC2 + s2_C1C2 - s2_KC1) / 2

Guards: (i) sampling noise can make a hat negative -> clip at 0 and set flag
`triangulation_clipped`; (ii) independence is an ASSUMPTION -- common-mode (transparency,
first-order extinction) cancels in each difference, but shared red noise between nearby
stars partially cancels too, so hats are lower bounds on isolated-star noise; state this
in outputs. K must be a constant star (existing check-star selection).

### 3.2 Photon correction (the Howell 1988 point)

Comps of unequal brightness contribute unequal photon variance. Excess (non-photon)
variance per star:

    x2_X = max(sig2_X_hat - pbar2_X, 0)     with flag if clipped

x2 is the star's unmodeled noise (systematics + floor + red component at epoch scale).
Comparisons between stars and against gates use x2, never raw sig2, when brightness
differs.

### 3.3 ZP noise for the sparse ensemble

    sigma_ZP(t)^2 = sum_i w_i(t)^2 * ( p_i(t)^2 + x2_i )

For n=2 this replaces the per-frame cross-comp std entirely. The per-frame
std-across-comps estimator is DEPRECATED for n <= 2 (a 2-point ddof=1 std is |r1-r2|/sqrt2
with c4(2)=0.798 bias and enormous per-frame variance); the night-pooled triangulation
above is the estimator of record. For n in {3,4}, compute BOTH (triangulation-based
sigma_ZP and the production per-frame SEM path with c4) and record the ratio as a
diagnostic column; production err keeps the SEM path (changing it is out of scope here).

### 3.4 Check-star model test

Model per epoch: sig2_model_K(t) = p_K(t)^2 + sigma_ZP(t)^2 + sigma_sys_rig^2 (floor in
mag; 0.018 eq4, 0.0 eq1 per SIGMA_FLOOR_SPEC). Pooled model variance
V_model = mean_t sig2_model_K(t). Observed: V_obs = robust variance of kmag(t)
(ddof=1; report both plain and outlier-trimmed with trim count).

Test statistic: R = V_obs / V_model. Under H0 (model correct, Gaussian),
(N-1) * V_obs / V_model ~ chi2_{N-1}, so the 95% CI of R is

    R_lo = R * (N-1) / chi2_{0.975, N-1}
    R_hi = R * (N-1) / chi2_{0.025, N-1}

(N ~ 25 -> CI half-width ~ +-30% on R; N ~ 139 (wide) -> ~ +-12%.)
Point estimates of sigma from s use the c4(N) correction where a sigma (not variance) is
reported.

### 3.5 Comp mutual stability test

For the pair (or all pairs at small n): T = (N-1) * s2_C1C2 / mean_t(p_C1(t)^2 + p_C2(t)^2)
~ chi2_{N-1} under photon-only H0. One-sided p-value; excess x2_pair = photon-corrected
excess per 3.2.

## 4. Trust band logic (per target, CI-based; NO field-wide inputs)

Inputs: R with [R_lo, R_hi] (3.4); comp mutual stability p and x2_pair (3.5);
n (comp count); flags.

    GREEN:  R_hi <= T_green            AND stability p >= 0.01  AND n >= 2
    YELLOW: not GREEN and not RED      (includes: CI straddles T_green; n == 2 with
            marginal stability 0.001 <= p < 0.01; any clipped-flag present; n == 1
            with band capped at YELLOW but R recorded)
    RED:    R_lo >= T_red              OR stability p < 0.001 with
            x2_pair > X2_RED (comps mutually unstable at a level that invalidates ZP)

Defaults (config, all overridable): T_green = 1.5, T_red = 4.0, X2_RED = (0.02 mag)^2.
Rationale: T_green tolerates the PZQ red component not captured by the white model at
single-epoch scale; T_red = 4 corresponds to err underestimated 2x -- beyond honest.
n == 1: band capped YELLOW (`single_comp`); kmag and R [R_lo, R_hi] ARE produced.
Sparse outputs carry `check_sparse = (n<=2)`.

Existing rule preserved: bands never RED-reversed by any field-wide quantity; check
scatter is always judged against its own CI, not a bare threshold.

## 5. Wiring changes

1. `compute_check_ensemble_mag_calib`: accept n >= 2; sparse branch (n <= 2) computes
   sigma_ZP per 3.3 and returns kmag + per-epoch sigma_model_K + flags. n >= 3 path
   unchanged except emitting the same model columns for the trust test.
2. New module `sparse_trust_core.py`: pure functions -- triangulation (3.1), photon
   correction (3.2), chi2 CI (3.4), stability test (3.5), band logic (4). Every function
   unit-tested against hand-computed values AND synthetic injections (Section 6).
3. Trust evaluation consumes the new statistics; field-wide comp_rms headline removed
   from the sparse trust path (kept as an informational diagnostic only, clearly labeled).
4. Sidecar and LC columns: `check_sparse`, `trust_R`, `trust_R_lo`, `trust_R_hi`,
   `trust_R_detrend` (diagnostic), `comp_stability_p`, `x2_pair_mag2`, `k_source`,
   `k_colour_offset`, `k_tier_excluded`, `k_colour_caveat`, flags. PDF/report: sparse
   targets show the CI-based verdict with n and N.

## 6. Validation protocol (mandatory before enabling)

S1 Synthetic: Gaussian injections with known sig_K, sig_C1, sig_C2 (unequal photon
   levels), N in {15, 25, 139}: triangulation recovers inputs within CI in >= 93% of 500
   trials per config (95% nominal, tolerance for finite trials); band logic hits the
   designed band in constructed GREEN/YELLOW/RED cases; negative-variance clip rate
   reported.
S2 Real GREEN control: draft_424 wide targets with n >= 5 -- new trust bands must agree
   with existing verdicts (no regressions to RED on healthy targets; report any flips).
S3 Real sparse: draft_426 r_60_4 (n=2 ensemble) -- sidecars with external K must produce
   kmag and a computed band on >= 1 target; SS Cam gets computed R [R_lo, R_hi]. Bands are
   whatever the numbers say. Baseline chi2 row filled (production_lc_err alongside spec-3.4).
S2 re-verify after Amendment 1: wide path bit-for-bit unaffected (zero band changes on
   n>=5 targets vs pre-amendment run, not only zero GREEN->RED flips).
S4 No production err change: LC `err` column byte-identical on draft_424 anchor
   (this feature adds columns and trust logic only). Anchor SHA must NOT move.

## 7. Explicit non-goals

- No change to production err assembly (SEM path, floor) -- SIGMA_FLOOR_SPEC governs.
- No comp-selection/threshold changes (COMP-POOL-R verdict stands).
- No multi-night statistics (single night is the canonical unit).
- Red-noise correction of kmag is NOT attempted here; sigma_r is reported by the PZQ
  diagnostic (SIGMA_FLOOR_SPEC Part B) and interpreted in the k'' workstream.
