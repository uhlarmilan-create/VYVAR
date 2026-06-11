# VYVAR — Comparison-star Floor Policy (min_comp) — Decision Spec

Question: what minimum comparison-star count should VYVAR require (current default
`phase01_comparison_n_comp_min = 3`; the "7 vs 5" question). Grounded in the primary
literature rather than a single-field sweep, because all archived drafts are the same field
(h & chi Per) and a true cross-field experiment is not available.

## Literature basis (author-year)

- **Broeg, Fernandez & Neuhauser 2005** (the algorithm VYVAR uses): builds an artificial
  comparison star as the variability-weighted average of *all* field stars, determining each
  star's variability self-consistently and iteratively down-weighting / removing variables.
  There is **no hard minimum** in the method itself; achievable accuracy depends strongly on
  the photometric band and on spectral-type (colour) differences between target and comps.
- **AAVSO Guide to Photometric Uncertainty / CCD Guide:** practitioners typically use ~12-20
  ensemble stars, or every star in the frame if properly weighted (Honeycutt-style). Quality
  beats quantity: comps with above-peer uncertainty/variability should be rejected; a few poor
  comps need not beat one good comp. The **check star is separate and must be independent**
  (C - K should be flat) — consistent with the CS-3 fix (2026-06-11).
- **Empirical optimum studies:** per-target optimal ensemble size is small and data-chosen
  (Burke et al. NGC 1245: median ~4, chosen by minimising LC scatter); transit pipelines find
  ~4-10 references reduce RMS until correlated noise dominates (Croll et al.); scintillation
  cancellation saturates around **6-8** comps (single comp amplification ~1.8 -> ~1); deep
  surveys require >=10 (SDSS Stripe 82).

**Take-away:** with Broeg weighting (VYVAR), the comp count is **not a precision knob** — the
weighted ensemble over all good stars sets precision. `min_comp` is a **robustness / trust
floor**: enough *good* stars that (a) single-star systematics cancel, (b) variable-comp
rejection can actually work (impossible with 2-3), and (c) sequence-magnitude offsets average
out. The literature puts that floor at **>= ~5**, with **~7-10 preferred** where the field
allows. The current floor of 3 is below where iterative variable rejection is reliable.

## Policy (maps onto VYVAR's existing two-threshold structure)

VYVAR already separates a hard floor (`min_comps`, n_clean below -> RED) from a "strong"
preferred level (`thin comp set` soft -> YELLOW between the two). Keep the Broeg "use all good
weighted" ensemble unchanged; only adjust the thresholds:

1. **Raise the hard floor** `phase01_comparison_n_comp_min` / trust `min_comps` from **3 -> 5**
   (below 5, variable rejection + systematics cancellation are unreliable per the literature).
2. **Keep / set the preferred level** `strong` at **~10** (AAVSO 12-20; conservative). Targets
   with 5-9 good comps -> existing `thin comp set` soft -> YELLOW (graceful degradation), NOT a
   hard reject — so sparse fields are flagged, not discarded.
3. Record the rationale + citations in `VYVAR_DECISIONS.md`.

This is **graceful degradation by design**, exactly the AAVSO/Broeg posture: prefer many good
weighted comps, accept fewer with a caution flag, reject only below the robustness floor.

## CRITICAL — this is NOT byte-identity-neutral (unlike A/B, CS-1..4)

Changing `min_comps` / `n_comp_min` changes which targets are accepted and how the ensemble is
populated -> it changes `lightcurve_*.csv` and `comparison_stars_per_target.csv` -> the
**photometry SHA moves** (`203254fd...` / `95a5515a...`). So any change to the floor requires
the **anchor re-cut discipline** we established:

1. Apply the threshold change.
2. Fresh zaloha-only run -> new draft; then a second fresh run; confirm the two are
   byte-identical (as draft_386 == draft_387).
3. Only then re-cut the anchor to the new SHA, retiring `203254fd / 95a5515a`, and record the
   recipe + threshold values.

Do not change the floor and the anchor in an unconfirmed single run.

## Coupling — do not starve the check star

Higher `min_comp` -> richer ensemble -> fewer leftover independent stars for the check
(CS-3 footprint on draft_387: only **60 / 1392** independent checks at h & chi Per). Weigh the
floor against check-star availability and the parked "reserved check-star (hold-one-out by
design)" item. Recommendation: a **moderate floor (5)** plus the reserved-check design is a
better balance than maximising the floor.

## Optional empirical probe on draft_387 (dense end only)

If empirical confirmation is wanted, sweep `n_comp_min in {3, 5, 7}` **on scratch drafts**
(never the locked config/anchor). Report, per setting: per-target LC scatter distribution,
number of accepted targets, and independent-check availability. Expectation: little change at
dense h & chi Per (comps are plentiful) — which itself confirms that the floor matters mainly
in *sparse* fields, the regime the literature already covers. The probe does not, by itself,
justify changing the live default; if a change is adopted, follow the anchor re-cut discipline.

## Decisions for Milan

- Hard floor value (recommend **5**) and `strong` preferred (recommend **~10**).
- Run the 387 dense-end probe first, or adopt the literature-grounded policy directly?
- Accept the anchor re-cut that a floor change entails?

## Out of scope

- Changing the floor in production config until Milan decides (this spec is decision-only).
