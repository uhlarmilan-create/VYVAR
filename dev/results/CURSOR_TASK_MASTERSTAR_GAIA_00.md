# MASTERSTAR-GAIA-00: Investigate + prototype the DAO-Gaia accounting
# fix (assignment / admission / detection axes). Decision input only.

Date issued: 2026-08-18
Type: investigation + sandbox prototype. NO production code change,
NO config change. Prototypes live under tmp/, never imported by
src_py. Push: only on Milan's authorization.

Premise check: DAO-COMPLETENESS-01 established: bright-end match
91% (55 stars G<=13 missing from MASTERSTAR); 69 unique holes at
G<=13.5, 68/69 caused by a neighbour DAO inside the 10 px match
floor + greedy 1-1 assignment (not by failed detection); 26 of them
PSF-isolated (nn >= FWHM 5.30 px) and cleanly measurable; edge50 =
15.0 = target_depth_g; pass 2 fills 3314/3681; astrometric identity
residuals p50 0.54 / p95 1.78 / p99 2.91 px. Literature direction
(architect): catalog-driven star list (Stetson 1987/1994 master-list
practice; Lang et al. 2016 Tractor forced photometry; SExtractor
dual-image; DOLPHOT reference-list photometry). SIPS-style
dual-aperture detection discussed and classified as detection-axis;
our measured losses are on assignment+admission axes. This task
tests those conclusions on data before any design is frozen.

Baseline for all deltas: anchor product 477dc8cf (draft 516,
NoFilter_60_2, MASTERSTAR of the clean rebuild).

Pre-registered expectations (PASS/DEVIATE each):
E1. Assignment prototype (radius ~3 px + optimal 1-1) recovers the
    majority of the 69 holes without any new detection.
E2. Catalog-seeded admission closes bright-end completeness G<=13
    from 91% to >= 99% on the same frames.
E3. Centroid QA at 2.0 px (p95-derived) rejects < 5% of legitimate
    forced seeds (false-reject rate measured on known-good MS
    members).
E4. No change to any existing MS member identity: current 3621
    members keep their catalog_id 1-1 under the new assignment
    (collisions resolved in favour of the same pairing) - deviations
    listed star-by-star.

## Part A - Assignment prototype (sandbox)
1. Rebuild the DAO<->Gaia assignment for the 5 reference frames
   (001/037/076/109/148) and for MASTERSTAR:
   a) match radius sweep: {2.0, 2.5, 3.0, 4.0} px (vs current
      10 px floor); report pairing count, hole count, false-pair
      proxy (pairs with |G_pred - inst_mag_scaled| outlier) per
      radius;
   b) optimal 1-1 assignment via scipy.optimize.linear_sum_assignment
      with distance cost (variant: distance + mag-consistency term);
      compare against current greedy on identical inputs;
   c) output: how many of the 69 holes obtain a correct detection;
      E4 identity check on existing members.
2. Blend axis: Gaia pairs with separation < FWHM (5.30 px) in FOV -
   count them, and report how current assignment treats each
   (which star owns the detection; is the loser in MS at all).
   No fix; accounting table only.

## Part B - Admission prototype (sandbox)
3. Catalog-seeded MASTERSTAR admission: for every on-chip Gaia star
   with G <= 15.0 lacking a DAO owner after Part A assignment,
   forced aperture measurement at the propagated position on
   MASTERSTAR: local background, SNR, centroid offset.
4. Position-QA calibration for E3: distribution of centroid offsets
   for (i) known-good MS members re-measured the same way (truth
   sample) and (ii) the new seeds. Report false-reject at bounds
   {1.5, 2.0, 2.5, 3.0} px. Recommend the bound from data.
5. Rung census after A+B on the 5 frames + MASTERSTAR:
   DETECTED(p1) / DETECTED(p2) / FORCED_SEED / BLENDED / TOO_FAINT /
   SATURATED / EDGE - one table, no star unaccounted. Compare
   bright-end completeness vs the 91% baseline (E2).

## Part C - Detection axis (measurement only, no prototype)
6. Quantify the pass1/pass2 imbalance mechanism: on the 5 frames,
   report rms_conv (global threshold basis) vs local annulus std
   distribution (pass 2 basis) in comparable units, and the
   resulting effective sigma of each pass. One paragraph: is pass 1
   mis-thresholded (T3-1/T4-1 class) or measuring something else?
   Note: NO retuning, no threshold change - diagnosis only. State
   whether a SIPS-style second detection scale would add anything
   the current pass 2 does not already cover (data-based answer).

## Part D - Impact projection + decision input
7. Project the science impact of A+B if productionized: expected
   comp-pool candidate delta (how many new admissible comps pass
   current zone/sat/color gates), CT pool delta, dilution accounting
   delta (BLENDED groups vs current silent drops). Target MAG is
   expected UNMOVED (membership adds only); state the mechanism by
   which each consumer could move and whether gates block it.
8. Runtime cost projection: assignment (Hungarian) and seeded
   admission per frame and per MASTERSTAR build.
9. One page verdict: which combination (A only / A+B / A+B+
   detection rebalance) reaches Milan's target "Gaia-complete
   MASTERSTAR above depth with honest states", at what complexity,
   and what the migration risk is (anchor re-cut needed: yes -
   membership changes are Stage-4 class; name every frozen artifact
   that would move).

## Constraints
- Sandbox only (tmp/ harnesses); production src_py and config
  untouched. Anchor 477dc8cf and all goldens untouched.
- Raw numbers under dev/results/context/session_2026081x_msgaia00/
  (Rule 0.2); tables as CSV.
- Standing authority: reject on premise contradiction.
- Runtime per part (Rule 0.3).

## Docs impact
None (investigation). Output feeds MASTERSTAR-GAIA-01 design
decision (Milan + architect).
