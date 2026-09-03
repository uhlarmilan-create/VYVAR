# DAO-GAIA-STAGE-01: Standalone detection+match stage tuning with
# visual overlay loop. Nothing downstream runs.

Date issued: 2026-08-19
Type: sandbox stage harness + iteration. NO production pipeline
runs, NO Phase 1/2A, NO anchor/golden contact. Production code
untouched until Milan approves a config visually and numerically.
Push: only on Milan's authorization.

Goal (Milan's acceptance, verbatim intent): on the overlay image,
every visually present star has a green DAO circle, with the blue
Gaia dot in it; no empty green circles. Quantified:
G1. Completeness vs Gaia G: >= 99% for G <= 13; >= 95% for
    G <= 14.5; report the full curve to G=16.
G2. Empty-sky false-accept <= 1% (the INV-DET-FALSEFILL-01 set).
G3. Spurious circles: DAO detections with no Gaia within 3 px,
    excluding corners: <= 1% of detections.
G4. Every Gaia G <= depth without a circle is explained by a named
    state (BLENDED / SATURATED / EDGE / TOO_FAINT), listed.

## Part A - Stage harness (build once, iterate fast)
1. tmp/ harness: input = one platesolved frame (MASTERSTAR and
   Light_001/076/148); runs detection + match ONLY; outputs:
   a) overlay PNG (stretch fixed asinh; green circles = accepted
      detections, blue dots = on-chip Gaia to G=16, red X =
      Gaia G<=14 without detection) - full frame + 3 fixed
      500x500 crops (center, mid, corner) for eye comparison
      across iterations;
   b) metrics JSON: G1-G4 numbers per iteration;
   c) runtime per iteration (target: seconds).
2. Iteration log: one row per config tried (params, G1-G4, verdict).
   Every iteration's PNGs kept under the session context dir so
   Milan can flip through them.

## Part B - Detection built right (the hypothesis to beat)
3. Single-pass DAOStarFinder with:
   a) threshold from SKY sigma, not scene rms_conv: background
      via annulus-free global sky estimate (sigma-clipped or
      SExtractor-style mode) on the star-masked image - report
      the value vs the 40 ADU local annulus median from GAIA-00;
   b) kernel FWHM from the frame's measured FWHM (5.3 px), not a
      default;
   c) sharpness/roundness windows opened enough not to reject
      bright/saturated flat-top stars - measure how many bright
      stars current cuts kill (this may be a big chunk of Milan's
      missing bright circles - verify, report);
   d) threshold sweep {3.0, 3.5, 3.8, 4.5, 5.0} x sky sigma,
      each iteration scored G1-G4.
4. If single-pass at sky sigma covers G<=14.5 at >=95% with G2/G3
   green: pass 2 becomes UNNECESSARY for detection (keep only as
   fallback or delete later). If not: report exactly which
   magnitude range needs a second local pass and tune only that.
5. Match: radius 3 px greedy (GAIA-00: == optimal), on the full
   detection list. BLENDED = Gaia pairs < FWHM (state, both dots
   marked violet in overlay so Milan sees blends are known, not
   missed).

## Part C - Milan's eye loop
6. Deliver the best config's overlays (full + crops) + the metrics
   table + the iteration log. Milan judges visually; architect
   judges G1-G4. Iterate on feedback until both accept. Only then
   does a separate task wire the winning config into production
   (with anchor implications handled there, not here).

## Constraints
- Frames: draft 516 platesolved products only (MASTERSTAR +
  Light_001/076/148). Read-only access to the draft.
- Empty-sky set from GAIA-01 reused for G2.
- Raw under dev/results/context/session_2026081x_daostage01/
  (Rule 0.2). Runtime per part (Rule 0.3).
- Standing authority: reject on premise contradiction.

## Docs impact
None (sandbox). Winning config + evidence feed the production
wiring task.
