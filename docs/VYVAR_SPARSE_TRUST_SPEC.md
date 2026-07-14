# VYVAR SPARSE-TRUST SPEC -- check-star ensembles and trust bands for sparse comp fields

Status: DRAFT for implementation (task SPARSE-TRUST). Author: Claude, approved direction
by Milan 2026-07-13. ASCII-only. All symbols in magnitude domain unless stated; conversions
via canonical MAG_ERR_SCALE only.

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
            marginal stability 0.001 <= p < 0.01; any clipped-flag present)
    RED:    R_lo >= T_red              OR stability p < 0.001 with
            x2_pair > X2_RED (comps mutually unstable at a level that invalidates ZP)

Defaults (config, all overridable): T_green = 1.5, T_red = 4.0, X2_RED = (0.02 mag)^2.
Rationale: T_green tolerates the PZQ red component not captured by the white model at
single-epoch scale; T_red = 4 corresponds to err underestimated 2x -- beyond honest.
n == 1: no check ensemble possible (triangulation needs 3 series) -> trust capped YELLOW
with reason `single_comp`, kmag not produced. Sparse outputs carry `check_sparse = (n<=2)`.

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
   `comp_stability_p`, `x2_pair_mag2`, flags. PDF/report: sparse targets show the CI-based
   verdict with n and N.

## 6. Validation protocol (mandatory before enabling)

S1 Synthetic: Gaussian injections with known sig_K, sig_C1, sig_C2 (unequal photon
   levels), N in {15, 25, 139}: triangulation recovers inputs within CI in >= 93% of 500
   trials per config (95% nominal, tolerance for finite trials); band logic hits the
   designed band in constructed GREEN/YELLOW/RED cases; negative-variance clip rate
   reported.
S2 Real GREEN control: draft_424 wide targets with n >= 5 -- new trust bands must agree
   with existing verdicts (no regressions to RED on healthy targets; report any flips).
S3 Real sparse: draft_426 r_60_4 (n=2) -- sidecars produced, baseline chi2 row filled,
   band computed. SS Cam -- band from fresh chi2 evidence; expected YELLOW or RED by the
   numbers, not by decree.
S4 No production err change: LC `err` column byte-identical on draft_424 anchor
   (this feature adds columns and trust logic only). Anchor SHA must NOT move.

## 7. Explicit non-goals

- No change to production err assembly (SEM path, floor) -- SIGMA_FLOOR_SPEC governs.
- No comp-selection/threshold changes (COMP-POOL-R verdict stands).
- No multi-night statistics (single night is the canonical unit).
- Red-noise correction of kmag is NOT attempted here; sigma_r is reported by the PZQ
  diagnostic (SIGMA_FLOOR_SPEC Part B) and interpreted in the k'' workstream.
