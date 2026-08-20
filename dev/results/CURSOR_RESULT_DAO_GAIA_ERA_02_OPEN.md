CURSOR RESULT - 2026-08-19 (DAO-GAIA-ERA-02-OPEN)

What I did
Open investigation of DAO-Gaia era divergence vs baseline 477dc8cf using
retry #3 artifacts (`part_c_rebuild_l1_l6.json`), baseline snapshot,
XVAL tables, and a cheap MS-only pool comparison (~2 min, no full rebuild).
Live draft_000516 verified at 477dc8cf n=97 on exit.

---

## Executive summary

| Question | Answer |
|----------|--------|
| **(a) Mechanism** | **Ensemble zero-point rebase** from comparison-star pool turnover (~19%) plus per-target comp re-ranking on detection-derived `comp_rms`/flux — not MAG measurement noise or census failure. **L4 BO 228 mmag is a harness bug** (baseline offset applied to new-era mags). |
| **(b) Product verdict** | **DIFFERENT**, not demonstrably WORSE. Internal scatter improved (BO check MAD **5.15** vs anchor **7.15** mmag). External shape vs AIJ preserved at **4.86 mmag RMS** once era-native offset is used. Large L2 deltas are mostly constant per-target ZP offsets (~0.1–0.4 mag). |
| **(c) Acceptance model** | **Retire L2 byte-continuity for era migration.** Replace with tiered gates: infrastructure (L5) + external absolute quality (XVAL, check MAD) + **shape preservation after ZP rebase** + optional **pinned-comp sensitivity** experiment. Keep strict L2 only for same-branch regression. |

---

## (a) Mechanism — named with evidence

### M1 — L4 “228 mmag” is an evaluation defect (PROVEN)

The Part C L4 harness computes AIJ offset from the **baseline** compare table
(`mag_calib_final` at 477dc8cf), then applies it to **candidate-era** BO
magnitudes.

Offline reproduction (Rule 0.2):

| Test | BO XVAL RMS |
|------|------------|
| Baseline LC, self-consistent offset | **4.858 mmag** (n=134) |
| Simulated +227.6 mmag constant shift, self-consistent offset | **4.858 mmag** |
| Same shift, **baseline offset (harness bug)** | **227.759 mmag** |
| Reported retry #3 L4 | **228.489 mmag** |

The 228 mmag signal is **exactly** the L2 median shift (227.6 mmag) applied
with the wrong zero-point. It carries **no independent photometric information**.

**Conclusion:** BO “228 mmag vs AIJ” must not be used in acceptance or
architecture decisions. Fix the harness or compute offset from candidate LCs.

### M2 — Primary physics: differential ensemble rebase (DOMINANT)

Differential photometry measures target flux **relative to a comparison ensemble**
(Honeycutt 1992, PASP 104, 435; already cited in VYVAR_DECISIONS ZP-CLIP /
FORCED-PHOT-01). When the comp pool or per-target comp subset changes, each
target’s `mag_calib_final` acquires a **new arbitrary zero-point** even if
per-frame flux ratios are unchanged.

**Pool turnover (MS-only experiment, 2026-08-19):**

| Metric | Baseline snapshot | Candidate MS rebuild |
|--------|-----------------|----------------------|
| Plan pool size | **2356** | **2240** |
| Pool ID overlap | — | **1899 / 2356 (80.6%)** |
| Only in baseline | — | **457** |
| Only in candidate | — | **341** |

Same **count** in retry #3 harness (2240) masked **457 star substitutions**.
Comp selection ranks by detection-derived **`comp_rms`**, color tier, and
distance — all sensitive to redetected centroids/flux even for “the same” Gaia ID.

**L2 delta morphology (retry #3, 46 targets with LCs):**

| Category | n | |median ?| |
|----------|---|------------|
| ?10 mmag (continuity) | 8 | good |
| 10–50 mmag | 9 | moderate rebase |
| >50 mmag | 29 | large rebase |

For **29 large-shift targets**, `max_abs_epoch_? / |median_?| ? 1.00–1.11` —
epoch patterns are **preserved within ~10–15 mmag** of a constant offset (BO:
227.6 mmag median, 240.7 mmag max). This is the signature of **ZP rebase**, not
random epoch noise.

Distribution is **bimodal**: overall median L2 shift **3.7 mmag**, mean **55 mmag**
(std 137) — most targets barely move; a subset (CVn variables, Gaia-only IDs,
BO/FW anchors) shift ~0.2–0.4 mag **as a block**.

### M3 — Detection-era inputs changed ranking surface (CONTRIBUTING)

Candidate era uses certified DAO-Gaia detection (2.5/2.5 px, ? 4.5/4.0) with
catalog expand (+967 rows at G?15). Expanded rows are **`zone=unknown`** and
**excluded** from the comp pool — they do not directly enter the pool but the
**detection table** feeding flux/`comp_rms`/centroids **did** change:

- MS Gaia overlap: 3146 shared, 400 baseline-only, 438 candidate-only IDs
- Centroid drift on shared IDs: median **0.25 px**, p95 **1209 px** (outliers =
  catalog-position rows vs detection centroids)

Pool admission still requires `zone=linear`, detection flux, and per-frame
`comp_rms` — i.e. the **477dc8cf-era detection overlay** and the **new-era
overlay** produce different sort keys even when spatial gates match.

### M4 — What did NOT cause the divergence

| Ruled out | Evidence |
|-----------|----------|
| Census / membership accounting | L5 **PASS** 100% (4990/4990); INV-MS-EXPAND-01 PASS |
| Empty-sky / certificate | PASS 2.5/2.5, ? 4.5/4.0 |
| Uniform measurement degradation | BO MAD **improved**; XVAL shape **unchanged** at 4.86 mmag |
| “228 mmag external failure” | Harness offset bug (M1) |

### M5 — Self-contradictory signals explained

| Signal | Reading |
|--------|---------|
| BO check MAD **5.15 mmag** (< anchor 7.15) | **Better internal differential scatter** in new era |
| L4 BO **228 mmag** | **Invalid metric** — wrong offset (M1) |
| L2 **227 mmag** median vs baseline | **Expected** ensemble rebase for BO’s new comp set |
| L3 BO below 0.85 band | Side effect of **tighter scatter**, not looseness |

---

## (b) Verdict — worse / better / different?

**DIFFERENT** (new differential reference frame per target).

**Not WORSE** on the metrics that matter for time-series differential work:

- **Internal precision:** BO check MAD 5.15 vs 7.15 mmag anchor ? **28% tighter**
- **External shape:** 4.86 mmag RMS vs AIJ (134 epochs) with era-native offset ?
  **unchanged** vs XVAL matrix reference
- **Epoch morphology:** For BO/FW and most CVn targets, post-rebase epoch
  residuals vs baseline are **O(10 mmag)**, not hundreds, once constant offset
  is recognized

**Not BETTER** in the sense of a free upgrade — the product is **not comparable
by absolute magnitude** to 477dc8cf without explicit ZP reconciliation. Science
products that assume cross-era absolute MAG identity (combined catalogs, legacy
overlay plots) **will break** until comp pools are pinned or transforms are
recomputed.

**Caveat:** 24/46 targets show **>10 mmag** epoch residual after subtracting
median offset (proxy “L2-shape” gate: 22/46 pass). Some are genuine secondary
effects (detrending/common-mode with new comps); **pinned-comp rerun** is needed
to bound how much is selection vs reduction. Not executed here (would need
preserved candidate tree or ~106 min rebuild with preservation).

---

## (c) Recommended acceptance model for pipeline-era changes

### Why L2 continuity is the wrong gate for era migration

L2 compares **`mag_calib_final` absolute values** across eras with **different
implicit comp ensembles**. That conflates:

1. **Legitimate differential rebase** (expected when comps change), with
2. **Photometric regression** (broken flux extraction, bad WCS, etc.)

Honeycutt (1992) and standard differential practice (e.g. Stetson’s packages,
AAVSO ensemble workflows) treat comp-star changes as **legitimate** provided
**light-curve shape** and **external checks** hold. Survey migrations (LSST DIA
injection tests; PhoPS validation separating calibration accuracy from repeatability)
gate **artifact rates, injection recovery, and error calibration** — not
byte-identical magnitudes across pipeline versions.

VYVAR’s own DECISIONS (XVAL-AIJ-02) already document that **4.86 mmag vs AIJ**
on production 4-comp ensemble is expected physics vs a brighter 5-comp sum — a
precedent that **absolute offset vs an external reference is not regression** if
RMS is bounded.

### Proposed tier model (ERA-ACCEPT)

| Tier | When | Gates | Retry #3 |
|------|------|-------|----------|
| **T0 Infrastructure** | Always | Certificate PASS, census 100%, invariants, empty-sky | **PASS** |
| **T1 External absolute** | Era acceptance | XVAL RMS vs independent photometry with **era-native offset**; check-star MAD within band or improved vs anchor | **PASS** (once L4 fixed) |
| **T2 Shape preservation** | Era acceptance | Per baseline LC target: subtract median offset; require max epoch residual ? ? (e.g. 10 mmag) vs baseline **or** detrended amplitude agreement | **Partial** (~22/46 proxy pass) — needs pinned-comp follow-up |
| **T3 Pool stability diagnostic** | Informational | Report pool Jaccard, per-target comp-set Jaccard for anchor targets; flag if <0.95 for BO/FW | **Fail** (80.6% pool overlap) |
| **T4 Continuity regression** | **Same branch / hotfix only** | Current L2 (|median|?2, max?10 mmag) | N/A for era migration |

**L1-final** (baseline 48 ? LC set) remains valid as a **coverage** gate, not
a magnitude gate.

### Pinned-comp experiment (recommended before anchor recut)

Run Phase 2A with **baseline `comparison_stars_per_target.csv` frozen** but
new-era MS/detection/certificate. Prediction:

- If L2 shifts **collapse** ? mechanism is **selection-only** (M2 confirmed)
- If shifts **persist** ? investigate flux extraction / forced-phot path

Cheaper than full rebuild if candidate MS layer is preserved once.

---

## Literature grounding (used, not decorative)

1. **Honeycutt (1992)** — differential ensemble method; variable comp membership
   is standard; absolute offset undefined without comp set specification.
2. **VYVAR XVAL-AIJ-02 decision** — 4.86 mmag BO RMS is accepted external bound;
   offset vs brighter ensemble is expected, not failure.
3. **STDWeb multi-night repeatability (arXiv:2608.10017)** — validates **scatter
   and error calibration** on constant field stars, not cross-pipeline mag identity.
4. **LSST DIA injection framework** — era acceptance via efficiency/artifact
   metrics on injected sources, not legacy product byte match.

AAVSO practice (ensemble check stars, transforms) aligns with **T1 external**
rather than cross-version MAG equality.

---

## Task list (follow-on)

| Priority | Task | Owner |
|----------|------|-------|
| P0 | Fix L4 harness: offset from **candidate** LC vs AIJ, not baseline compare table | Cursor |
| P0 | Document L2 as **same-branch regression only** in DECISIONS / acceptance spec | Architect |
| P1 | **Pinned-comp Phase-2A rerun** on one preserved candidate tree (BO/FW + 6 CVn) | Cursor |
| P1 | Add **T2 shape gate** + **T3 pool Jaccard** to era acceptance harness | Cursor |
| P2 | **MS-POOL-POLICY-01** — stabilize comp selection under catalog-derived membership | Architect |
| P2 | One **preserved full rebuild** if pinned-comp inconclusive | Pre-authorized |

---

## Artifacts

| Path | Content |
|------|---------|
| `dev/results/context/session_20260819_era01_part_c/part_c_rebuild_l1_l6.json` | Retry #3 L-table + per-target L2 |
| `dev/results/XVAL_AIJ_02_bo_compare.csv` | Baseline BO XVAL (477dc8cf) |
| `tmp/era02_*.py` | Offline diagnostics (L4 offset, L2 categorize, shape gate) |
| `dev/results/CURSOR_RESULT_DAO_GAIA_ERA_01_PART_C_RETRY3_STOP.md` | Retry #3 STOP record |

## Errors

None. Live draft restored **477dc8cf** n=97 verified on exit.

## Files changed

- `dev/results/CURSOR_TASK_DAO_GAIA_ERA_02_OPEN.md` (task stub)
- `dev/results/CURSOR_RESULT_DAO_GAIA_ERA_02_OPEN.md` (this report)
- `tmp/era02_l2_categorize.py`, `tmp/era02_bo_shape.py`, `tmp/era02_l2_shape_gate.py` (sandbox diagnostics)

No `src_py` production edits. No push.
