CURSOR RESULT - 2026-08-01 17:30 UTC+2

**Outcome L-a** (aperture / magnitude-dependent radius; Step 1k K-a withdrawn).
`pearson(residual, r50)` production **-0.024**; aperture-controlled G slope (joint OLS) **+0.048**;
peak trend after G+aperture control **-0.023** (was +0.362).

What I did
Ran L1-L4 diagnostics on harness F(12) and production flux (4083 / 4058 star-frames); fixed
fixture annulus so `r_in > r_norm`; re-ran G0-G5. Diagnostic only; no production code change.

## L1 -- discriminating statistic

Source: `dev/tools/closure_step1l_discriminate_slope.py`, `tmp/closure_step1f_ee_cache.npz`,
`proc_BO_CVn_Light_*.csv`. Residual from G-only fit of log10(flux).

### Production flux

| correlation | pearson | spearman | n |
|-------------|---------|----------|---|
| residual vs **r50_frame** | **-0.024** | -0.036 | 4058 |
| residual vs fwhm_estimate_px | -0.496 | -0.438 | 4058 |
| residual vs VY_FWHM (header) | -0.022 | -0.023 | 4058 |
| residual vs peak_max_adu | **+0.362** | +0.599 | 4058 |
| partial(residual, peak \| r50) | **+0.362** | -- | 4058 |

Pre-registered reading: `|pearson(residual, r50)| = 0.024 < 0.2` -- does **not** show the
simulation's -0.999 seeing signature on pooled star-frames (r50 spans only **1.63-1.91 px**
across 139 frames; most residual variance is cross-star at fixed frame).

**Residual binned by r50_frame** (production): mean residual flat near **+0.04 dex** for
r50 1.70-1.85; **-0.022 dex** at r50 1.90-1.95 (n=30). No monotonic seeing trend.

**Residual binned by peak ADU** (production, G-only fit): identical to Step 1k
(-0.421 to +0.123 dex across bins). **After G + aperture_r_px fit**, `pearson(residual, peak)`
falls to **-0.023** -- peak trend does **not** survive aperture control.

### Harness F(12)

| correlation | pearson |
|-------------|---------|
| residual vs r50_frame | -0.007 |
| residual vs peak_max_adu | +0.359 |
| partial(residual, peak \| r50) | +0.360 |

Same pattern: r50 weak; peak trend present under G-only fit, confounded with aperture/magnitude
structure on production path.

**Note on fwhm_estimate_px (r = -0.50):** per-star constant from draft SNR table (one value
per star, not per-frame seeing). Correlation is a magnitude/aperture proxy, not frame seeing.

## L2 -- colour term, tested properly

Split at median G = **10.11**. Source: same script.

| half | n | slope G only | slope G + BP-RP | BP-RP coef | pearson(G, BP-RP) |
|------|---|--------------|-----------------|------------|-------------------|
| faint (production) | 1946 | **-0.408** | **-0.428** | -0.175 | -0.202 |
| bright (production) | 2112 | **-0.194** | **-0.224** | -0.673 | +0.072 |
| full (production) | 4058 | -0.296 | -0.339 | -0.245 | -0.232 |

BP-RP coefficient **inconsistent across halves** (-0.175 faint vs -0.673 bright; signs of
pearson(G, BP-RP) flip -0.20 / +0.07). Adding BP-RP moves the **faint half away** from -0.4
(-0.408 to -0.428), not toward it. **L-c excluded** -- omitted-variable bias from
colour-magnitude correlation in a magnitude-limited field, not a band effect (D10-1 not implicated).

Harness F(12): same pattern (faint -0.404 -> -0.414 with BP-RP; bright -0.593 BP-RP coef).

## L3 -- what else produces -0.285 / -0.296

Source: production `proc_*.csv`, n = 4058.

| test | G slope | notes |
|------|---------|-------|
| log10(flux) vs mag only | **-0.296** | D5-2 baseline |
| + aperture_r_px regressor | **+0.048** | coef(aperture) = **+1.631**; G and aperture_r correlated **r = -0.95** |
| + sky_subtracted_adu | -0.296 | negligible (coef ~ 7e-8) |
| log10(flux_large) vs mag | -0.269 | still compressed |
| log10(flux_small) vs mag | -0.271 | still compressed |
| partial slope mag \| aperture_r | **-0.002** | near zero |

Joint G + aperture_r OLS does not yield G slope -0.4 (collinearity). But **peak residual trend
vanishes** once aperture_r is in the model (L1), and the partial G slope after removing
aperture dependence is ~0. The -0.296 compression is consistent with **SNR-table aperture
increasing with brightness** (aperture_r vs mag r = -0.95) while `flux` is measured in that
aperture with **no curve-of-growth normalisation to a fixed radius**. This is D5-1 / A-1 seen
from the flux-magnitude side, not detector non-linearity.

Neither pure seeing (r50) nor pure non-linearity at plausible strength reproduces -0.296 alone;
the dominant identifiable contributor is **aperture-radius vs magnitude coupling**.

## L4 -- fixture annulus corrected

Formula: `r_in = max(r_norm + 2.0, 4.75 * fw) = 14.0 px`, `r_out = 21.555 px` (`r_norm = 12`).

**G0-G5: all PASS.**

| geometry | r_in / r_out (px) | step | G_gt_11 @ r50=1.87 (mmag) |
|----------|-------------------|------|---------------------------|
| harness | 25.0 / 45.0 | 1b-1j | 71.66 |
| production (overlap bug) | 11.376 / 21.555 | 1k | 71.66 |
| corrected (r_in > r_norm) | **14.0 / 21.555** | **1l** | **71.66** |

Range over r50 span G_gt_11 unchanged at **14.8 mmag** across all three (< 0.1 mmag per row).

## D5-2 -- consolidated statement

**Confirmed.** On anchor draft_435, pipeline aperture flux does not scale with catalogue
magnitude at slope -0.4 mag^-1:

- log10(flux) vs mag: **-0.296** (n = 4058, se = 0.0022)
- log10(dao_flux) vs mag: **-0.296** (identical in this draft)
- Harness F(12): **-0.285** (n = 4083)

Reproduced on two independent production flux columns and harness recomputation. **Not a harness
artefact.**

**Mechanism (Step 1l):** magnitude-dependent `aperture_r_px` from the SNR table (r vs mag
-0.95) with flux measured in that aperture and no COG correction to a fixed reference radius.
Links D5-2 to **D5-1** and the A-1 differential-aperture thread. **Not** detector non-linearity
(D1-2).

## Step 1k K-a verdict

**Withdrawn.** Residual-vs-peak was confounded (peak ~ F/FWHM^2; OLS zeroes residual vs G by
construction). Step 1l shows:

1. `pearson(residual, r50)` is weak (-0.024), not the discriminating seeing signature on pooled data.
2. Peak trend **does not survive** G + aperture_r_px control (r: +0.362 -> -0.023).
3. Colour term fails L2 consistency test.
4. D1-2 must **not** be recorded as MEASURED on the Step 1k statistic.

## What Step 1m must change

1. Any flux-vs-G test must control for `aperture_r_px` or normalise to fixed radius (e.g. COG to
   r = 12 px) before interpreting slope.
2. Do not consolidate `delta_ap` until the aperture-normalisation path is explicit.
3. Revisit D5-2 remediation jointly with D5-1 / A-1 (SNR-table radius policy).

## Docs impact

- Register: K-a withdrawn; D5-2 mechanism = aperture; D1-2 reverted to prior state
- AUDIT_FINAL: revert D1-2 "first valid measurement"; D5-1 cross-ref; D5-2 consolidated
- Fixture: annulus 14.0-21.555 px (Step 1l)

## Errors (if any)

None.

## Files changed

- `dev/tools/closure_step1l_discriminate_slope.py` (new)
- `dev/tools/closure_a1_reference_fixture.py` (annulus r_in = 14.0 px)
- `dev/results/CURSOR_RESULT_closure_step1l.md` (this file)
- `docs/VYVAR_AUDIT_CLOSURE_REGISTER.md`
- `docs/VYVAR_AUDIT_FINAL.md`
- `docs/VYVAR_STATE.md`
- `docs/VYVAR_ROADMAP.md`
