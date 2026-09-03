# DAO-COMPLETENESS-01: Measure the DAO-Gaia gap; account for every
# Gaia star in FOV; decision input for detection-ladder architecture

Date issued: 2026-08-18
Type: measurement only. No code change, no parameter change.
Push: only on Milan's authorization.

Premise check: UI overlay on a current frame shows GAIA_MATCHED 3516,
FORCED_APERTURE 0, DAO_ONLY 0, with many Gaia positions lacking DAO
detections - some visually coincident with real PSFs. Goal is NOT
100% DAO detection (stars below the frame limit legitimately cannot
be detected); goal is 100% accounting: every Gaia star in FOV gets a
named state. This task measures the current gap and feeds the
architect's ladder proposal (DETECTED / FORCED_OK / TOO_FAINT /
BLENDED / SATURATED / EDGE). Report which draft and frames were used
(do not assume 516; state it).

Pre-registered expectations (state PASS/DEVIATE each):
E1. Bright-end completeness (>= 2 mag above the 50% edge) is ~100%;
    isolated bright undetected stars are rare or zero.
E2. The 50% completeness edge is consistent with the operative
    detection/depth limit of the rig (compare against
    below_target_depth machinery values).
E3. A significant fraction of unmatched-but-visible cases are blends
    (Gaia neighbour < 2 px at 9.77 arcsec/px).
E4. Pass 2 (targeted local search at unmatched Gaia positions)
    contributes ~0 on current config - and the reason is identifiable
    in code (gate too strict, disabled, or not wired on this path).

## Part A - Current two-pass accounting (code + counters)
1. Read the actual Pass 2 implementation: entry conditions, threshold,
   local sigma estimate, acceptance gate. Report why the overlay
   shows FORCED_APERTURE 0: not run / ran and rejected all / ran on
   zero candidates / display-layer filter. Numbers from code and logs,
   not UI.
2. Map where DAO-match status gates science on the current tip: does
   unmatched exclude a star from masterstar catalog admission, comp
   pool, CT pool, target photometry, dilution neighbour accounting?
   One table: consumer -> behaviour on unmatched.

## Part B - Completeness curve
3. On >= 5 frames spanning the night (include twilight and dark; the
   T4-1 sigma spread 52->30 is expected to move the edge): match rate
   vs Gaia G in 0.5 mag bins. Report per-frame curves + the night
   median curve; 50% edge per frame.
4. Compare the edge against E2's reference value. Report the delta.

## Part C - Classify every unmatched Gaia star (forced measurement)
5. For each unmatched Gaia star brighter than (50% edge - 1.5 mag) on
   the sampled frames: forced aperture photometry at the propagated
   Gaia position (PM-corrected, local background, local sigma).
   Classify:
   a) flux consistent with background -> TOO_FAINT (stretch illusion)
   b) measurable flux + Gaia neighbour < 2 px -> BLENDED
   c) measurable flux, isolated, SNR above the nominal detection
      threshold -> DETECTION HOLE (real defect): report position,
      G mag, local sigma, distance to nearest frame edge and to
      nearest bright star
   Report counts a/b/c per frame and the c-list in full.
6. Reverse check on the same frames: DAO detections with no Gaia
   match within tolerance - count, sigma distribution, spatial
   pattern (corners vs uniform). Artifact class or real?
7. Spatial map of category-c stars (if any): uniform, edge-clustered,
   or optics-correlated (vignetting zone)?

## Part D - Decision input (no implementation)
8. One page: given A-C numbers, evaluate the ladder proposal.
   Specifically: how many stars per frame would land on each rung
   (DETECTED / FORCED_OK / TOO_FAINT / BLENDED); what position-QA
   would FORCED_OK need (centroid check feasibility with measured
   astrometric residuals); which science consumers should accept
   which rungs (comp pool = DETECTED only is the architect's
   presumption - confirm or refute from data); and whether current
   Pass 2 should be repaired or absorbed into the ladder.
9. Runtime per part (Rule 0.3).

## Constraints
- Measurement only; no config or code edits, even where Part A finds
  a defect - name it, do not fix it.
- Raw numbers under dev/results/context/session_2026081x_dao_comp_01/
  (Rule 0.2). Curves as CSV, not only plots.
- Standing authority: reject on premise contradiction.

## Docs impact
None yet (measurement). Findings feed a future DAO-LADDER design
task; JOURNAL entry only if a category-c defect is confirmed.
