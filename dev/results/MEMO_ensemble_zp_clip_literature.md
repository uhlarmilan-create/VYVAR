# MEMO: Literature basis for removing the per-frame ensemble zeropoint clip

**Date:** 2026-08-12
**Purpose:** Decision record companion for `docs/VYVAR_DECISIONS.md` (ZP-CLIP-REMOVAL)
and methods-paper text.
**Question:** Is per-frame rejection of a comparison star from the ensemble
zeropoint an accepted practice in differential photometry?

---

## 1. Broeg et al. 2005 - the method VYVAR already cites

Broeg, Fernandez & Neuhaeuser, Astron. Nachr. 326, 134 (2005) construct an
artificial comparison star as the weighted average of many field stars, with
weights derived from each star's variability determined self-consistently over
the whole series.

The key structural point: **the variability weight and the decision to drop a
star are properties of the star across the entire time series, not of a single
frame.** Implementations state this explicitly.

- SPECULOOS-South pipeline (Murray et al., arXiv:2005.02423), which implements
  Broeg: the iterative algorithm weights all comparison stars by their
  variability and removes those that are clearly variable. Removal is of a
  *star*, over the series.
- Jena University Observatory pipeline (Broeg's own `chphot`, arXiv:0905.1833):
  after the weighting stage, stars with low weights are sorted out - those not
  present on every image, those with low S/N, and variable stars. Again a
  per-star, whole-series decision.
- Same structure in Cep-Cas rotation-period work (arXiv:0905.1837) and REM Orion
  monitoring (arXiv:0911.0760).

Broeg-family implementations also consistently state that it is statistically
optimal to use **as many comparison stars as possible**, appropriately weighted,
rather than a small hand-picked set.

**Bearing on VYVAR:** the clip at `ensemble_normalize` made a per-frame
in/out decision on a star that Broeg treats as a fixed member with a fixed
weight. That is not the cited method.

## 2. Honeycutt 1992 - the one place variable membership is legitimate

Honeycutt, PASP 104, 435 (1992), "CCD ensemble photometry on an inhomogeneous
set of exposures", explicitly handles the case where the number and identity of
comparison stars vary across exposures.

But note *why*, and *how*:

- **Why:** stars are genuinely missing from some exposures - different fields,
  different depths, long-term heterogeneous archives. Not because a star's
  residual looked large on that frame.
- **How:** by a single global linear least-squares solution over all stars and
  all exposures simultaneously, solving jointly for each star's magnitude and
  each frame's zeropoint offset, with the error treatment derived for exactly
  that varying membership.

**Bearing on VYVAR:** varying membership is not forbidden in the literature -
but a method that allows it must account for it in the estimator and in the
errors. VYVAR's clip varied the membership per frame and then took a simple
weighted mean as though membership were fixed. That combination appears nowhere
in the literature and is the actual defect.

This also converges with the parked WIDE-ERR item, which is already routed to
Honeycutt's leave-one-out treatment. A future Honeycutt-style global ensemble
solution would fix the error bars and would make membership variation
legitimate - but it is a separate, larger change and is not what is being
decided now.

## 3. What comparable tools do

| tool | ensemble membership | per-frame rejection from the ZP? |
|---|---|---|
| **AstroImageJ** (Collins et al. 2017) | fixed set of apertures applied to the whole series; ensemble can be changed by the user without re-running photometry | no. Offers ensemble optimisation over the series and optional light-curve outlier removal at the plotting/fitting stage - both whole-series and user-controlled |
| **C-Munipack / Muniwin** (the AAVSO amateur standard) | user selects comparison stars from VSP; ENSEMBLE mode uses the same set on every frame, validated by a check star | no |
| **Broeg-family pipelines** (SPECULOOS, chphot, STARSKY) | weights per star over the series; low-weight stars dropped once | no |
| **VaST** (Sokolovsky & Lebedev 2018) | per-frame magnitude calibration by a robust fit **against hundreds to thousands of matched stars** | robust fitting yes, but at N in the hundreds - a different statistical regime entirely |
| **Honeycutt 1992** | membership may vary per exposure | not rejection; absence, handled by a global least-squares solution |

No tool in this set performs automatic per-frame ejection of a comparison star
from a small ensemble zeropoint.

**Honest qualification:** the literature does not ban robust rejection in
general. VaST clips per frame and is right to. What distinguishes it is N. With
hundreds of stars a robust scale estimate is well determined; the rejection
decision is stable and the estimator is not the dominant noise source. That
condition is not met at N = 5.

## 4. Why N = 5 is the decisive fact

At N = 5, the MAD is effectively a single order statistic of five numbers, and
its behaviour is poor in two separate ways.

**Efficiency.** The MAD has an asymptotic Gaussian efficiency of about 37%,
against 58% for the Rousseeuw-Croux Sn and 82% for Qn (Akinshin,
arXiv:2209.12268). Finite-sample efficiency is lower still than these asymptotic
figures. A low-efficiency scale estimator at N = 5 has very large frame-to-frame
variance - which is directly what was measured: MAD ranged from 0.002 to 0.037
across frames while the star's absolute deviation stayed near 100 mmag.

**Bias.** For a standard normal sample at n = 5, the MAD underestimates the
population scale; consistency requires a correction factor near 1.72 rather than
the asymptotic 1.4826, i.e. roughly 16% relative bias. VYVAR applied the
uncorrected 1.4826. The rejection boundary was therefore systematically **too
tight** at exactly the N where the clip activated.

So the clip was not merely unnecessary at N = 5. It was biased toward
over-rejection, and its threshold was dominated by estimator noise rather than
by the data.

**Measured confirmation** (draft 509, 134 frames): on the 37 rejected frames the
star's absolute deviation from the ensemble median was about 110 mmag, versus
99 mmag on kept frames - essentially unchanged. What changed was the MAD, 0.018
versus 0.037, shrinking the boundary from about 166 to 80 mmag. 33 of 37
rejections had MAD below the overall 25th percentile. On rejected frames the
star's residual against the other comparison stars was *quieter* (0.007) than on
kept frames (0.010): the clip preferentially discarded the star when it was
behaving best.

## 5. Comparison-star admission (Decision 2)

Broeg-family sources support using as many comparison stars as possible with
appropriate weighting, rather than restricting to a few. Astrokit (Burdanov et
al., arXiv:1408.0664) gives the practical envelope: keep the magnitude
difference within about 2 mag of the target and the ensemble within a few arcmin
of it.

`phase01_comparison_max_mag_diff = 2.0` sits exactly at that stated limit, so
the current setting is defensible and there is no literature reason to revert it
to 1.5.

**Caveat that the counterfactual matrix could not test.** The matrix measured
scatter. The risk from loosening comparison-star admission is not scatter but a
colour-dependent systematic, which appears as a smooth airmass-correlated drift
and is invisible to a scatter metric. `comp_max_delta_bprp` moved 0.79 -> 0.99
in the same config generation. On the BO CVn field this did not bite - all five
admitted comparisons have dBP-RP <= 0.15 - but that is a property of this field,
not evidence about the parameter. Any future decision on colour tolerance needs
a residual-versus-airmass and residual-versus-colour test, not this matrix.

## 6. Conclusion

Per-frame rejection of a comparison star from a small-ensemble zeropoint has no
support in the differential-photometry literature and is not implemented by any
comparable tool. At N = 5 the MAD is the wrong instrument for the job on both
efficiency and bias grounds. Removing the clip returns VYVAR to the Broeg 2005
structure it already cites: fixed membership decided once per draft, weighted by
quality, with every admitted star contributing to every frame.

## References

- Broeg, Ch., Fernandez, M., Neuhaeuser, R. 2005, Astron. Nachr. 326, 134
- Honeycutt, R. K. 1992, PASP 104, 435
- Collins, K. A., Kielkopf, J. F., Stassun, K. G., Hessman, F. V. 2017, AJ 153, 77 (AstroImageJ)
- Sokolovsky, K. V., Lebedev, A. A. 2018, Astronomy & Computing 22, 28 (VaST)
- Murray, C. A. et al. 2020, MNRAS (SPECULOOS-South photometry), arXiv:2005.02423
- Burdanov, A., Krushinsky, V., Popov, A. 2014 (Astrokit), arXiv:1408.0664
- Akinshin, A. 2022, "Finite-sample Rousseeuw-Croux scale estimators", arXiv:2209.12268
- Rousseeuw, P. J., Croux, C. 1993, JASA 88, 1273
