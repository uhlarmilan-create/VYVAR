# VYVAR -- Graceful comp degradation + honest sparse path + trust model

Status: DRAFT for Milan review. Grounded design spec; a Cursor implementation task follows on
approval. English/ASCII. No code changed by this document.

---

## 1. Problem (from the draft_409 diagnosis)

The comp-selection path is **fragile** and the fallback **misreports**:

- A modest, normal data change (V0612 LC RMS 0.0089 -> 0.0141, ~1.6x) flipped V0612 from 8 clean
  `default` comps + GREEN to 8 `sparse_fallback` comps + YELLOW.
- The default path **found 2 good colour-matched comps** (per-target comp_rms 0.017-0.034) but
  **discarded them** because `2 < n_comp_min=3`, then took 8 worse comps ranked on **field-wide**
  comp_rms (0.13-2.67). The pipeline traded good comps for more-but-worse ones.
- The sparse path writes field-wide comp_rms into the **same `comp_rms` column** as per-target
  values, unlabeled; PDF omits `comp_path`; `pipeline_meta.comp_sparse_fallback_used` is not
  per-target; the check-star selector ranks on the mislabeled values. (Trust itself **does** fire
  YELLOW via `comp_path` -- that layer works.)

Goal: make comp selection **degrade gracefully** (keep a few good comps rather than fall to a
worse path), make the fallback **honest** (every number has an unambiguous scale and the path is
visible), and grade **trust** by validated evidence rather than a hard cliff -- all grounded so the
outputs are publication-defensible (AAVSO / VarAstro).

---

## 2. Design principles (grounded)

**P1 -- Quality over quantity. A few good colour-matched comps beat many mixed ones.**
- Broeg et al. (2005): the optimal artificial comparison star weights each comp by 1/sigma^2;
  noisy comps are down-weighted toward zero, so they add almost no information. Under VYVAR's
  current **flux-sum** ensemble (equal weight, not IVW), noisy comps are *not* down-weighted and
  actively corrupt the reference -- so clean-comp gating is even more essential now than it will be
  after the parked Broeg-IVW sigma-budget work.
- Honeycutt (1992) / AAVSO: "more is better" applies to *random-error reduction with good comps*;
  a single good colour-matched comp + a check star is already valid, publishable differential
  photometry (AAVSO CCD guide; confirmed in our trust-floor grounding).
- Random-error arithmetic: going 8 -> 3 *equal-quality* comps raises the reference noise by only
  sqrt(8/3) ~ 1.6x. But an "8" that is 2 good + 6 field-wide-noisy comps, combined by flux-sum, is
  *worse* than the 3 good comps it replaced.

**P2 -- The per-target differential RMS is the selection/ranking metric. Field-wide RMS is never
used to select or rank comps.** Field-wide RMS measures scatter against the field zero-point, not a
star's usefulness as a reference for *this* target; selecting by it is selecting on the wrong axis.

**P3 -- sparse_fallback is a flagged last resort, not a routine path.** It runs only when per-target
selection genuinely yields ~0 usable comps, and when it runs it is loudly marked.

**P4 -- Trust grades by validated evidence (comp count + check star), per the AAVSO minimum -- not a
hard cliff.** (Grounded trust model, section 5.)

**P5 -- Every reported number has an unambiguous scale and provenance.** Publication requirement: an
AAVSO/VarAstro reviewer (or Milan) must never have to guess whether a comp_rms of 0.3 means "noisy
comp" or "different scale."

---

## 3. Graceful routing (the core change)

Replace the "default < n_comp_min -> sparse_fallback" cliff with a graded decision on the count of
**per-target-validated** comps (N_good = comps passing the colour ladder + per-target comp_rms gate):

| N_good (per-target, validated) | action | trust band (sec. 5) |
|--------------------------------|--------|---------------------|
| >= green_min (e.g. 5) | use them | GREEN-eligible |
| 1 <= N_good < green_min | **use them (do NOT fall to sparse)** | YELLOW |
| 0 | sparse_fallback as last resort, loudly flagged | RED / low-YELLOW |

Key: when the per-target path yields a few good comps, **keep them**. V0612 on the morning proc
would then report its 2-3 good colour-matched comps (YELLOW), not 8 field-wide sparse comps.

**Grounded refinement (recommended, Cursor to assess feasibility):** the per-target gate is an
**absolute** `max_comp_rms=0.1`, which a ~1.6x noisier night trips wholesale. Physically, a comp's
usefulness is *relative to the achievable precision* that night (photon + scintillation + seeing
floor), not a fixed number -- this is exactly the principle comp_qa already encodes in its
magnitude-dependent expected-RMS locus (Sokolovsky). Gating relative to that locus (or to the best
comps' RMS) instead of a fixed 0.1 would remove the cliff at its source. If reusing the comp_qa
locus during selection is a large reordering, defer it; the section-3 routing fixes the symptom
regardless.

---

## 4. Honest sparse path

When sparse_fallback does run:
- **Segregate the scale.** Do not write field-wide values into the per-target `comp_rms` column.
  Either keep `comp_rms` per-target-only (NaN/blank on the sparse path) and put field-wide values in
  a clearly named separate column, or tag the scale explicitly. Same column name must mean the same
  metric everywhere.
- **Surface `comp_path`** in `photometry_summary.csv` and the PDF report (currently per-row in the
  comp CSV only). The reader must see "this target used sparse_fallback."
- **Fix `pipeline_meta.comp_sparse_fallback_used`** to be a per-target signal, or remove it as a
  health indicator (it is currently a config flag and was `true` even on the all-`default` overnight
  run).
- **Check-star selector** must rank on a per-target metric, never field-wide comp_rms.

---

## 5. Trust model (grounded; supersedes the >=5 hard RED floor)

From the trust-floor grounding (Honeycutt/Broeg/AAVSO): the AAVSO *mandatory* minimum is >=1 comp +
a check star in the FOV; a single good colour-matched comp + check is valid science.

- **RED** = below the AAVSO minimum: 0 usable per-target comps **or** no check star.
- **YELLOW** = 1-4 good colour-matched comps + check star + comp_qa OK (valid, fewer comps, less
  averaging -- use with caution). sparse_fallback, if used, lands here at most and is marked.
- **GREEN** = >= green_min (e.g. 5) good comps + check star + comp_qa OK.

`comp_trust_min_comps=5` becomes the **GREEN** threshold, not the RED gate. Final band numbers are
Milan's config call; the *structure* is the grounded part.

---

## 6. Deferred / optional (noted, not in this fix's core)

- **Shared cross-filter comp core** (from the C-hypothesis finding: stored g/i/r Jaccard ~0.50
  because ranking is per-filter-independent). Selecting a consistent colour-matched core across
  filters, pruning only genuinely-bad-per-filter stars, would improve multi-filter consistency and
  target colour reliability. Worth doing **before** multi-filter colour science, but separable from
  this robustness fix.
- **z-band 0-comps** (filter-specific failure for all shared targets) -- separate investigation.

---

## 7. Publication alignment (next workstream after this)

This fix is the prerequisite for clean AAVSO / VarAstro / Summary Measure Report outputs:
- AAVSO ensemble reporting needs the **comp list** (or ENSEMBLE + comps in notes), the **check
  star** (KNAME/KMAG), and a defensible **uncertainty** -- all of which depend on honest comp_rms
  scale + a real check star + the trust band.
- The trust band (GREEN/YELLOW/RED) maps naturally to a data-quality note for submission.
Building the comp/check/uncertainty data cleanly *here* means the output-sync workstream is a
formatting/mapping job, not a re-derivation.

---

## 8. Validation / DoD (for the implementation task)

- V0612 on current proc: keeps its good per-target comps (YELLOW if only 2-4 pass on this proc, or
  GREEN if >=5), **not** an 8-comp field-wide sparse set. No field-wide value in the per-target
  `comp_rms` column. `comp_path` visible in summary + PDF.
- SS Cam: 3 good tier-1 comps + check star -> YELLOW (not RED, not a discarded-for-sparse set).
- No regression on a well-populated control (re-run; confirm comps/delta_mag stable).
- Process discipline added: **archive the proc snapshot before any re-proc**, so a validated
  reduction stays reproducible (we lost the overnight proc this way).

---

## 9. Grounding references

- Broeg et al. 2005 (optimal artificial comparison star; 1/sigma^2 weighting).
- Honeycutt 1992, PASP 104, 435 (ensemble photometry; "more is better" with good comps).
- AAVSO CCD Observing Manual / Guide to CCD Photometry (>=1 comp + check minimum; single good
  colour-matched comp + check valid).
- Osborn et al. 2015, MNRAS 452, 1707 (scintillation -> the per-night achievable floor that a
  relative gate should track).
- Sokolovsky (magnitude-dependent expected-RMS locus; already in VYVAR comp_qa).
