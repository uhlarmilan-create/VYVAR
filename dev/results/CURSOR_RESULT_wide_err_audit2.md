CURSOR RESULT - 2026-08-04 (WIDE-ERR AUDIT-2)

What I did
Attempted direct retrieval of Honeycutt (1992, PASP 104:435) error-section equations.
When the primary PDF/full text (numbered eq. 3-5) was unavailable, collected secondary
sources that quote Honeycutt's post-LS error treatment. Compared those to VYVAR code.
Read-only; no measurements.

## R1 -- Retrieve Honeycutt (1992) directly

### Channel results (in task order)

1. NASA ADS (https://ui.adsabs.harvard.edu/abs/1992PASP..104..435H/abstract)
   NOT OBTAINED. Page returned JavaScript bot verification; no PDF or full text.

2. arXiv
   Honeycutt (1992): NOT FOUND (1992 predates arXiv).
   Kundic et al. 1995 (astro-ph/9508145): PDF obtained. Quotes equation of condition
   and beta minimization only (see verbatim below). Refers to sigma[m0(s)] and
   sigma[em(e)] from the Honeycutt reduction but does not print closed-form error
   equations.
   Vazquez et al. 2015 (arXiv:1502.01977): PDF obtained. Appendix quotes A1-A2
   (equation of condition + beta) only; states "empirical estimate of the uncertainty"
   without printing Honeycutt's sigma formulas.

3. IOP direct (https://iopscience.iop.org/article/10.1086/133015/pdf)
   NOT OBTAINED. HTTP 200 but Radware Bot Manager captcha HTML (14371 bytes), not PDF.
   Partial HTML/text fetch (prior session + search snippets) recovered Section 2 only:
   equation of condition and beta; text cuts before the error-evaluation section.

4. Masaryk University / institutional access
   NOT AVAILABLE on this side.

5. Secondary sources with actual post-LS error formulas attributed to Honeycutt (1992)

   (a) Rengstorf et al. 2004, AJ, "QUEST1 Variability Survey. II." (Honeycutt co-author),
       Section 4.3, eq. (6)-(13). States: "See Honeycutt (1992) for a detailed description
       of the original algorithm." Gives Bevington & Robinson (1992) weighted-average-variance
       forms for sigma_m0 and sigma_em and quadrature combination for light-curve error.

   (b) Richmond solvepht reference implementation (ensemble-1.1.tar.gz from
       http://spiff.rit.edu/ensemble/ ; documents Honeycutt 1992). calc_sigmas() in
       solvematrix.c implements post-solution RMS for em(e), m0(s), and M(e,s).

   (c) Fernandez et al. 2012, PASP 124:507 -- cites Honeycutt (1992) ensemble method;
       does NOT quote Honeycutt per-frame/per-star error formulas.

   (d) Broeg et al. 2005, AN 326:134 -- iterative 1/sigma^2 weights; no Honeycutt
       sigma_em / sigma_m0 formulas found in abstract or search snippets.

### Primary outcome

Honeycutt (1992) primary PDF or full text including numbered equations 3, 4, 5 and the
per-frame error sigma_em(e) section: NOT OBTAINED from any direct channel.

Partial primary text (Section 2 only, IOP snippet / search extraction):

  m(e,s) = m0(s) - em(e)

  beta = Sum_e Sum_s [ m(e,s) - m0(s) - em(e) ]^2 * w(e,s)

  w(e,s) = 1 / sigma^2(m(e,s))

  (sign convention: Honeycutt writes minus before em; Kundic/Vazquez/Bruzual use plus)

### Verbatim secondary equations (post-LS error treatment attributed to Honeycutt 1992)

From Rengstorf et al. 2004, Section 4.3 (after LS solution for m0, em):

  m(e;s) = m0(s) + em(e)                                    (eq. 6)

  sigma_m0(s) = sqrt( N * Sum_e { [m(e;s) - em(e) - m0(s)]^2 / sigma(m(e,s))^2 }
                      / { (N-1) * Sum_e [ 1 / sigma(m(e,s))^2 ] } )    (eq. 7)

  sigma_lc(e;s) = sqrt( sigma_m(e,s)^2 + sigma_em(e)^2 )    (eq. 12)

  sigma_em(e) = sqrt( N * Sum_s { [m(e;s) - em(e) - m0(s)]^2 / sigma(m(e,s))^2 }
                      / { (N-1) * Sum_s [ 1 / sigma(m(e,s))^2 ] } )    (eq. 13)

  (Summations over N appearances of star s on exposures, or N stars on exposure e.)

From Richmond solvematrix.c calc_sigmas() (Honeycutt 1992 solvepht implementation),
after LS: Mes(e,s) = m(e,s) - em(e) [corrected magnitude]; w = input weight:

  residual for exposure e:  (Mes(e,s) - m0(s))

  sigma_em(e) = sqrt( (n * Sum_s w * (Mes - m0)^2) / ((n-1) * Sum_s w) )

  sigma_em(e) / sqrt(n)  stored as sigemn  [per-image diagnostic]

  sigma_m0(s) = sqrt( (m * Sum_e w * (Mes - m0)^2) / ((m-1) * Sum_e w) )

  sigma(M(e,s)) = sqrt( sigma_inst(e,s)^2 + sigemn(e)^2 )
    where sigma_inst from 1/sqrt(w) or brightness-matched sigma_m0 neighbors

From Kundic et al. 1995 (astro-ph/9508145), Section 2:

  m(e,s) = m0(s) + em(e)                                    (eq. 1)

  beta = Sum_e Sum_s [ m(e,s) - m0(s) - em(e) ]^2 * w(e,s) (eq. 2)

  (No closed-form sigma_em printed; refers to sigma[m0(s)] vs m0 and sigma[em(e)] vs em
   from the Honeycutt reduction.)

From Vazquez et al. 2015, Appendix A:

  m(e,s) = m0(s) + em(e)                                    (A1)

  beta = Sum_e Sum_s [ m(e,s) - m0(s) - em(e) ]^2 * w(e,s) (A2)

  w(e,s) = sigma(m(e,s))^(-2)

  "yields ... an empirical estimate of the uncertainty" (formula not printed).

Richmond ensemble.html diagnostic (quality metric, not formal error bar):

  z = corrected_mag(i,j) - true_mag(i)

  z1 = sqrt( Sum(z^2 * w) / Sum(w) )

  z2 = z1 / sqrt(N)

## R3 -- Primary error formula not obtained; what CAN be said

Honeycutt 1992 primary error formula not obtained from any source.

The following is grounded in secondary sources and code reads only.

### Equation of condition (Kundic 1995; Vazquez 2015; partial Honeycutt 1992)

  m(e,s) = m0(s) + em(e)   [plus-sign convention in secondary papers]

  beta = Sum_e Sum_s [ m(e,s) - m0(s) - em(e) ]^2 * w(e,s)

LS residual for constant stars at solution: m(e,s) - m0(s) - em(e).

VYVAR ensemble SEM residual (photometry_core.py:3430-3434):

  comp_resid = m - comp_ref_map[cid]

  where comp_ref_map[cid] = median(m_i across night) (lines 3382-3390).

VYVAR excludes em(e). This is a documented difference in the residual definition.
AUDIT 1 row 3 remains DIFFERS.

### Secondary-attributed Honeycutt error treatment vs VYVAR (not primary eq. 3-5)

| Item | Literature (secondary, attributed to Honeycutt 1992) | VYVAR code | MATCH / DIFFERS / CANNOT DETERMINE |
|------|------------------------------------------------------|------------|-------------------------------------|
| 1. Residual for scatter/error | m(e,s) - em(e) - m0(s) after LS; or Mes - m0 with Mes = m - em (Rengstorf eq. 7, 13; solvematrix.c calc_sigmas) | m - comp_ref_map[cid] (photometry_core.py:3430-3434) | DIFFERS (no em(e); m0 approximated by night median) |
| 2. Per-frame ZP error sigma_em(e) | sqrt( N * Sum_s w*(Mes-m0)^2 / ((N-1)*Sum_s w) ) (Rengstorf eq. 13; solvematrix.c:422-423) | std_ddof1(comp_resid)/c4/sqrt(n) on m-median only (sigma_floor_core.py:48-49 via photometry_core.py:3438) | DIFFERS (different residual, unweighted std, c4 correction, no input weights w) |
| 3. Per-star error sigma_m0(s) | sqrt( M * Sum_e w*(Mes-m0)^2 / ((M-1)*Sum_e w) ) (Rengstorf eq. 7; solvematrix.c:469-470) | not computed; comp_ref_map is median only (photometry_core.py:3387-3390) | DIFFERS (VYVAR has no sigma_m0) |
| 4. Differential magnitude error bar | sqrt( sigma_m(e,s)^2 + sigma_em(e)^2 ) quadrature of instrumental + exposure terms (Rengstorf eq. 12; solvematrix.c:486) | ensemble_scatter joined with photon err in combine_production_err_rel (sigma_floor_core.py:64-86; photometry_core.py:3584-3586) | CANNOT DETERMINE vs primary Honeycutt (secondary uses LS-derived sigma_em, not VYVAR's per-frame comp std) |

Whether Honeycutt's derived error formula is well-approximated by VYVAR's
std(residual)/c4/sqrt(n) on m - comp_ref_map cannot be determined without the
primary Honeycutt (1992) error-evaluation section (numbered eq. 3-5 not retrieved).

Do not close AUDIT 1 row 3 as DIFFERS-BAD. It remains DIFFERS.

This audit reports what was retrieved from the literature. It does not close AUDIT 1 row 3.
