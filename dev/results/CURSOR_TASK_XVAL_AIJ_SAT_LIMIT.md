# CURSOR TASK - XVAL-AIJ-01 + SAT-LIMIT-01

Date issued: 2026-08-16 (late)
Baseline: origin/main a217e1d; draft 515 photometry SHA da9cce4.
Push: NOT authorized (next session close covers it).

## Part A - XVAL-AIJ-01: record the external cross-validation (docs only)

Milan ran AstroImageJ 6.0.10 on the same BO CVn raw night with the SAME five
comparison stars as VYVAR's BO ensemble (verified by RA/DEC match: AIJ
C2..C6 = 1500748301498613248, 1497771992240531712, 1499200223486564608,
1497974027502858240, 1497368849430107904; T1 = BO CVn). The architect
compared epoch-by-epoch (identical unweighted flux-sum form, both normalized
by median):

- 134/134 epochs matched; per-epoch difference AIJ - VYVAR: median +0.15
  mmag, RMS 3.28 mmag, MAD 3.24 mmag, max |diff| 10.0 mmag.
- Light-curve amplitude: AIJ 474 vs VYVAR 475 mmag ptp - identical.
- No FWHM or airmass correlation of the difference (r = +0.08 / -0.12),
  despite AIJ using a fixed 7 px aperture vs VYVAR per-mag 3.5-8.5 px.
- Errors: AIJ CCD-equation median 6.2 mmag vs VYVAR exported 9.1 mmag
  (CT+SEM+scint) - consistent with the WIDE-ERR-04 conservative closure.

A1. Commit the evidence: Milan supplies Table.tbl (AIJ measurements) and
    XVAL_AIJ_01_bo_compare.csv (architect's epoch table) - place under
    dev/results/ with dev/results/CURSOR_RESULT_XVAL_AIJ_01.md summarizing
    the numbers above (they are measured; do not re-derive unless a number
    fails a sanity read).
A2. Register: the July roadmap item "independent-tool cross-check" -> CLOSED
    with this evidence. JAAVSO methods queue gets the sentence: VYVAR and
    AstroImageJ agree to 3.3 mmag RMS per epoch over 134 frames of a
    0.47-mag-amplitude eclipser using identical comparison ensembles.
A3. Cross-validation evidence chain is now: photutils/sep 3 mmag RMS
    (library level) -> architect independent reconstruction 0.0001 mmag
    (product formula) -> AIJ 3.3 mmag RMS (full chain, external tool).
    One DECISIONS line linking the three.

## Part B - SAT-LIMIT-01: the saturation gate is structurally disabled

Finding (architect, from masterstars_full_match.csv of draft 515):
`saturate_limit_adu` and `saturate_limit_adu_85pct` are NaN; comparisons
against NaN are silently False, so `is_saturated`/`likely_saturated` can
never fire. Consequence measured: comp C2 (G=7.99, VYVAR peak_max_adu
64350 = 98.2 percent of the 65535 container clip; AIJ warned saturation on
its own C1 and C2 peaks cross 64k) sits in the BO CVn ensemble as
zone=linear. Independent evidence it hurts: architect's flux-sum LOO drops
10.33 -> 7.82 mmag when C2 is removed; pytics already down-weights it
(rms 0.0081 -> 0.0123). Likely the same resolver-hole pattern as the
sigma_sys map (equipment 1 missing).

B1. Trace the saturate-limit resolver chain (header SATURATE/MAXLIN/
    LINLIMIT/MAXADU -> equipment DB -> config) for equipment 1 on this
    draft: where exactly does NaN enter? Is the whole catalog NaN (report
    the count)? Cite the resolver lines and the DB row.
B2. Fire proof of the silent-pass defect: a guard test where limit=NaN and
    peak=65000 must FAIL the current code (prove the hole exists), then
    pass after the fix. An unresolved limit must never silently admit -
    INV-SAT-LIMIT: NaN/missing limit = hard error at draft build, or an
    explicit conservative default with a WARN naming the value and source.
B3. The limit value itself, data-derived per house principle: the container
    clip is 65532/65535 (14-bit grid, proven in GAIN-DOMAIN-01). The
    linearity knee is measurable: D1-2's cheap check - instrumental minus
    catalogue magnitude residual vs peak_max_adu on 515 clean stars; the
    ADU level where the trend departs is the knee. Run it; if the data do
    not resolve a knee, adopt limit = 0.8 * container clip with the WARN
    (state the choice; do not silently hardcode).
B4. Impact and re-read (measurement, then one decision for Milan):
    - Re-classify all 515 catalog stars under the fixed limit; report how
      many change zone, and which comps of which targets are affected
      (C2/BO is the known one).
    - Recompute the BO ensemble without C2 (layer-2 re-select on the fixed
      pool) and the fixed-meter check MAD; report old vs new. The 01B
      same-meter acceptance numbers may shift - if they do, the acceptance
      table gets a superseded-with-pointer note, not a silent overwrite.
    - Do NOT change other drafts; 515 is the evidence draft.
B5. Cross-links in the register: SAT-LIMIT-01 <-> BIN-8-9 (bright-end LOO
    excess plausibly partly nonlinearity) <-> D1-2 (the knee measurement
    partially answers it - record what remains open there).

## Report

dev/results/CURSOR_RESULT_XVAL_AIJ_01.md and
dev/results/CURSOR_RESULT_SAT_LIMIT_01.md + JSON (B1 counts, B3 knee
measurement, B4 before/after). Every number: quantity, units, domain, SHA.
session_baseline_check.py --fast OVERALL PASS required. Defects in this
spec: name them. Physics outranks the spec.

ASCII only. English.
