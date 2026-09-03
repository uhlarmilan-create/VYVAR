CURSOR RESULT - 2026-08-18 22:50 UTC+2

What I did
MASTERSTAR-GAIA-00 sandbox on draft 516 / NoFilter_60_2 (anchor
prefix 477dc8cf). No src_py or config edits. Harness
`tmp/masterstar_gaia_00.py` (not imported by production). Frames
001/037/076/109/148 plus MASTERSTAR.fits. Gaia set = on-chip G<=17.5
(n=11953), same as DAO-COMPLETENESS-01. Assignment uses the
**production detection list** (MS 3621 x,y including 75 DAO_ONLY;
per-frame unforced proc x,y) so E1 is not contaminated by new finder
peaks.

Premise (Rule 0.1): compare a 3 px Hungarian 1-1 (and a lock-existing
variant) plus catalog-seeded forced admission against the current
10 px greedy floor that left 69 holes / 91% bright-end match. The
69-hole list is from DAO-COMPLETENESS-01 unique IDs (G<=13.5).
Completeness G<=13 baseline is 91% on that same FOV set.

## Pre-registered expectations

| ID | State | Evidence |
|----|-------|----------|
| E1 | DEVIATE | Hungarian r=3 px recovers **24/69** holes (35%), not a majority. Greedy at the same radius recovers the same 24 (Hungarian == greedy at every radius on this field). The 26 FWHM-isolated holes recover **0**. Those have nearest existing DAO at median **7.43 px**, outside 3 px. The 24 recovered already had a leftover DAO at median 0.42 px (DAO_ONLY sitting on the star). |
| E2 | DEVIATE | After A (hung3): G<=13 match = 905/974 = **92.9%**. After A+B (FORCED_SEED with dcent<=2 px): **935/974 = 96.0%**, not >=99%. Remainder at G<=13: BLENDED 27, TOO_FAINT 8, SAT 2, EDGE 2. Mag-term Hungarian reaches 94.4% DETECTED-only, still short of 99%. |
| E3 | PASS | Known-good MS members (n=2400 stamps): centroid offset p50=0.44 px, p95=0.75 px. False-reject at 2.0 px = **0.04%** (<5%). Seeds p95=1.93 px; seed reject at 2.0 px = 4.6%. Recommend **2.0 px** (truth-safe; seeds near the 5% cap). 2.5 px if seed yield is preferred (seed FR 1.3%). |
| E4 | DEVIATE | Unconstrained hung3: **264** current MS Gaia IDs unmatched, **63** new IDs, **381** detection remaps (`e4_lost_members.csv`, `e4_pairing_diffs.csv`). Lock-existing then leftover hung3: **5** new IDs (DAO_ONLY promotions, list in `e4_gained_locked_ids.txt`); 194 "lost" are almost entirely MS catalog_ids **absent from the G<=17.5 on-chip query** (off-chip / mag cap), not remapped members. Unconstrained A does not preserve identity. |

## Output / findings

### Part A - assignment

Radius sweep on MASTERSTAR (production 3621 detections vs 11953 Gaia):

| r_px | n_pair greedy=hung | holes rec | bright G<=13 | mag-outliers |
|------|--------------------|-----------|--------------|--------------|
| 2.0 | 3263 | 21 | 0.924 | 183 |
| 2.5 | 3313 | 23 | 0.928 | 194 |
| 3.0 | 3345 | 24 | 0.929 | 202 |
| 4.0 | 3403 | 26 | 0.930 | 219 |
| 10.0 | 3453 | 26 | 0.930 | 240 |

Hungarian distance-only **equals** greedy at all five radii (same
n_pair, same holes). The bipartite graph inside r is already unique
1-1; optimal assignment adds no pairing. Mag-consistency cost is
the only differentiator (bright_frac 0.929 -> 0.944 at 3 px;
outliers 202 -> 161).

Hole recovery file: `hole_recovery.csv`. Isolated-not-MS (n=26):
hung3_owned=0. Mechanism confirmed: no leftover detection within
3 px. Shrinking the radius cannot invent a second peak; it only
assigns DAO_ONLY that already sit on the star (the 24).

**Blend axis** (sep < FWHM 5.30 px), MASTERSTAR, greedy-10:
2048 undirected pairs. Both own a detection: 187. Exactly one
owner: **810** (loser silently dropped; 789 of those losers are
not in MS). Neither: 1051 (both too faint / unmatched). Table:
`blend_pairs.csv`. No fix in this task.

Frames: hung3 drops pairing vs greedy-10 (e.g. Light_001 2658 ->
2010) because many current matches live at 3-10 px. Frame-level
A-only is a completeness **loss** unless admission fills the gap.

### Part B - admission

Catalog-seeded forced aperture on G<=15 lacking a hung3 owner.
MASTERSTAR rung census (`rung_census.csv`; all 11953 accounted):

| Rung | n |
|------|---|
| DETECTED (hung3) | 3345 |
| FORCED_SEED | 169 (156 not already in MS) |
| BLENDED nn<2 px | 91 |
| TOO_FAINT G<=15 | 192 |
| SATURATED | 5 |
| EDGE | 10 |
| TOO_FAINT G>15 | 8141 |
| **sum** | **11953** |

G<=13 unmatched after A: 69 = FORCED_SEED 30 + BLENDED 27 +
TOO_FAINT 8 + SAT 2 + EDGE 2. All 30 FORCED_SEED pass dcent<=2 px.
A+B DETECTED+FORCED at G<=13 = 96.0%. Honest-state accounting at
G<=13 is 100% (every star named). The 99% DETECTED+FORCED target
is blocked by 27 blends, not by missed isolated PSFs (those 26
become FORCED_SEED).

Centroid QA: `centroid_qa_bounds.csv`, `centroid_qa_truth.csv`,
`centroid_qa_seeds.csv`. Bound recommendation: **2.0 px**.

### Part C - pass1 / pass2 imbalance (diagnosis only)

| Image | rms_conv | local_std p16/p50/p84 | thr_p1 | n_p1 |
|-------|----------|-----------------------|--------|------|
| MS | 635 | 39 / 41 / 60 | 2402 | 307 |
| 001 | 634 | 38 / 44 / 61 | 2396 | 307 |
| 037 | 620 | 32 / 37 / 56 | 2343 | 315 |
| 076 | 622 | 31 / 36 / 55 | 2352 | 309 |
| 109 | 631 | 39 / 41 / 60 | 2385 | 307 |
| 148 | 620 | 29 / 34 / 53 | 2342 | 301 |

Pass1 effective sigma vs rms_conv is exactly N_equiv=**3.78**
(by construction). Pass2 uses 1.9 x **local annulus** std
(~40 ADU) => threshold ~76 ADU. That is **0.12 x rms_conv**, not
1.9 x rms_conv. rms_conv is convolved **scene** RMS (stars in the
kernel), not sky. Local annulus is sky-like. This is a T4-1-class
global-vs-local mismatch: pass1 is not "3.8 sigma of sky"; it is
3.8 sigma of a star-contaminated convolved image, so only ~300
peaks survive. Pass2 already is the local-threshold detection
scale (3314 extra on the MS build). A SIPS-style second detection
aperture would retread pass2, not recover the 26 isolated holes
(those fail because pass2 **skips** any Gaia with a DAO within
10 px, then 1-1 assignment gives the peak to the neighbour).
No retune in this task.

### Part D - impact + verdict

**Comp / CT:** 156 new MASTERSTAR FORCED_SEED not in MS, all inside
the current comp mag envelope (8.0-17.3) and nn>=2, edge>=10.
Upper bound on new pool candidates ~156 before zone/sat/variable
gates; a subset would pass `zone==linear` (peak vs bg_sigma). CT
uses the same pool; ct_c1 **can** move if new comps enter the
ensemble. Target MAG: VT list unchanged; `below_target_depth` still
G=15; MAG expected UNMOVED unless a target was missing from MS
(not the case for the 48 LC set). Dilution: GS11 still off;
naming the 810 one-owner blend pairs is the accounting delta
(silent drops -> BLENDED). Mechanism to move MAG would be a
change in comparison ensemble, blocked unless the new seeds pass
comp gates **and** are selected.

**Runtime:** greedy 0.03 s; Hungarian (connected components)
0.22-0.86 s per image; lock-then-leftover 0.005 s. Seeded
admission is cheap (hundreds of apertures). Full harness 36.5 s
including 6-image photometry. Production add: <<1 s/frame for
assignment, few seconds for MS seeded admission.

**Verdict: A+B, not A only, not detection-rebalance-first.**

A only does not reach Gaia-complete-above-depth: it recovers the
24 holes that already had a spare DAO and **cannot** recover the
26 isolated stars. Hungarian without a mag term is a no-op vs
greedy here. Detection rebalance (lower pass1) would multiply
peaks but the 10 px pass2-skip + 1-1 still hides the isolated
neighbour cases; it is the wrong axis for the measured losses.
Catalog-seeded admission (B) at 2.0 px centroid QA puts those 26
on FORCED_SEED, names BLENDED/TOO_FAINT/SAT/EDGE, and lifts
bright-end DETECTED+FORCED to 96%. Remaining 4% at G<=13 is
honest blends/edge/sat/faint, which is the "honest states" target
even if DETECTED+FORCED never hits 99%.

Complexity: assignment change is small (optional mag-cost; keep
greedy if mag-cost not wanted). Admission is the real design
(Stetson/Tractor/DOLPHOT catalog list; forced photometry already
exists for MS members - extend the seed set). Overlay
FORCED_APERTURE label remains a display bug (out of scope).

**Migration risk: Stage-4 membership change. Anchor recut yes.**
Frozen artifacts that would move: `masterstars_full_match.csv`
(3621 + ~156 seeds, minus none if E4 locked), `comparison_stars.csv`
/ per-target, all `proc_*.csv` (forced inject of new IDs),
`pipeline_meta.json` Gaia-DAO completeness, product SHA 477dc8cf,
P1 mini `draft_000516_p1mini`, freeze
`draft_000516_snapshot_cleanrebuild_20260818`, possibly CT
coefficients. MAG 48 LCs expected byte-identical if comp
selection does not pick the new seeds; not guaranteed. Goldens
and 435 retirement list untouched in this task.

## Errors (if any)

None after a one-line census scalar fix in the harness. First
run aborted on `int(boolean array)`; no production files touched.

## Files changed

- `dev/results/CURSOR_TASK_MASTERSTAR_GAIA_00.md`
- `dev/results/CURSOR_RESULT_MASTERSTAR_GAIA_00.md`
- `dev/results/context/session_20260818_msgaia00/` (CSV/JSON)
- sandbox only: `tmp/masterstar_gaia_00.py`

Push not authorized.

## Runtime (Rule 0.3)

| Part | seconds |
|------|---------|
| A assignment + sweep (6 images) | included in 35.7 |
| B seeded photometry + QA | included in 35.7 |
| C rms/local std | included (pass1 only) |
| **Wall** | **36.5** |

Hungarian 0.45 s (MS, r=3) / 0.22-0.70 s (frames). Greedy 0.03 s.
Locked leftover assign 0.005 s.
