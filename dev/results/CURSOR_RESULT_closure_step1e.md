CURSOR RESULT - 2026-08-01 CLOSURE STEP 1e (measurement-method repair, re-measure delta_ap)

**G6: FAIL.** **G7: PASS** (max err 0.005 mmag vs fixture L2). **No decisive number reported.**

---

## What I did

Pushed Steps 1c/1d to origin (`665c76a`). Implemented R1-R4 in new harness
`dev/tools/closure_step1e_differential_aperture.py`: photutils exact COG (R1), Gaussian2D+Const2D
centroid (R2), robust p95-p5 and slope*span ranges (R3), reject non-monotonic/EE>1 curves (R4).
Rebuilt EE cache on anchor draft_435 (139 frames, 35 stars). Ran gates G6/G7, G4 jitter re-check,
and M1-M4. Marked V8-V10 VOID in Step 1d report.

## Output / findings

### Gates (first three lines per task spec)

1. **G6 FAIL** -- five fixed proxies cannot produce admissible p95-p5 ranges on enough frames.
   All 139 frames excluded (>20% star-frame drop each). Four of five proxies have zero admissible
   frames; one proxy (G 12.03) retains 27 star-frames before frame exclusion, insufficient for G6.
2. **G7 PASS** -- harness `_curve_of_growth_photutils` matches fixture L2 table within **0.005 mmag**
   (limit 5 mmag) on all r50 x sub-ensemble cells.
3. **No decisive delta_ap number** -- G6 failure blocks M2 per pre-registered rule.

### G4 jitter (sub-pixel position sensitivity at r = 1.916 px)

| Method | Position jitter [mmag] |
|--------|------------------------|
| Before (integer centre, L3) | **77.3** |
| After (photutils + sub-pixel centre, R1/R2) | **4.3** |

### R1-R4 repairs (line numbers in `closure_step1e_differential_aperture.py`)

| Repair | Change |
|--------|--------|
| **R1** | L69-L93 `_curve_of_growth_photutils`: `CircularAperture` + `CircularAnnulus` (25-45 px), norm at 12 px, 0.25 px grid -- matches fixture L2 |
| **R2** | L96-L145 `_gaussian_centroid_or_none`: centroid from converged `Gaussian2D+Const2D`; star-frame dropped if fit fails (no silent proc CSV x,y) |
| **R3** | L248-L318 `_robust_delta_stats`: reports `range_p95_p5_mmag` and `slope_times_r50_span_mmag`; two-point min/max-r50 diff labelled explicitly, not as range |
| **R4** | L108-L116 `_cog_admissible`: drop star-frame on EE>1 or non-monotonic curve; no cumulative-max clip |

**R4 drop counts (4865 star-frame attempts = 35 stars x 139 frames):**

| Reason | Count |
|--------|------:|
| Gaussian fit fail | 2 |
| COG fail (edge/flux) | 239 |
| EE > 1 before norm radius | 2590 |
| Non-monotonic | 319 |
| **Kept admissible** | **1709** (35%) |

All **139 frames** exceed the 20% drop threshold and are excluded from delta_ap series.

### M1 (per proxy, blocked for headline by G6)

Fixed proxy IDs from Step 1d: G 12.03, 12.06, 12.10, 13.01, 14.50. With frame exclusion,
all proxies yield **n_frames_used = 0** for delta_ap series. Diagnostic (frame exclusion disabled,
R4 per-star rejection only): proxy G 12.03 alone retains 27 frames; p95-p5 range G 8-9 **205 mmag**,
G > 11 **165 mmag** -- but four other proxies have zero frames, so G6 cannot pass.

### M2 consolidated

**Blocked.** No `X +/- Y mmag` headline. Fixture expectation unchanged: G 8-9 **144.3 mmag**,
G > 11 **14.8 mmag** over r50 span.

### M3 real target

Focus star `1498135552633294976`: QC failed; **0 admissible frames** after R4 (133 validation
drops, 47 COG fails, 2 fit fails). Not headline.

### M4 B.5 / B.6 (repaired harness)

All NaN -- insufficient admissible star-frames after R4 + frame exclusion. Inputs to Milan's
option (i)/(iii)/(iv) choice remain from Step 1c until admissible real-data COG coverage improves.

### T4 (median across proxies, pre-registered)

| Per-proxy ratio G8-9 / G>11 | Value |
|-----------------------------|------:|
| (insufficient data) | -- |

**Median: NaN. PASS: false.** Fixture expectation: **9.74**.

### V8-V10 in Step 1d

Marked VOID in `dev/results/CURSOR_RESULT_closure_step1d.md` with pointer to this report.

## Register wording (section 4)

> **A-1b CONFIRMED.** Seeing-correlated, magnitude-dependent differential aperture systematic,
> magnitude of order 10^2 mmag. Target on the `r_min` clamp at 1.916 px against comparisons at
> larger magnitude-binned radii, with no curve-of-growth correction. Exact value open pending
> sub-pixel re-measurement (Step 1e). Physics expectation from
> `dev/tools/closure_a1_reference_fixture.py`: 144.3 mmag for G 8-9 comparisons over the
> anchor's measured r50 span.

Step 1e completed measurement-method repair; G7 validates the method on synthetic data. G6 failed
on real data because R4 rejection (mostly EE>1 from neighbor contamination) leaves insufficient
admissible coverage. **203 mmag remains superseded, not replaced.**

### VOID (Step 1f supersession)

**Pointer:** `dev/results/CURSOR_RESULT_closure_step1f.md`

The claim that **89% of R4 rejections are neighbour contamination** is **VOID**. Step 1f shows
the same rejection rate on perfect isolated Moffat stars with Poisson noise; the full-curve
admissibility rule (fixed by C1 in Step 1f) was the cause.

## What Step 2 inherits

- **G7 PASS:** measurement method is validated against known answer at the COG level (not just arithmetic).
- **G6 FAIL root cause:** strict R4 rejection exposes neighbor contamination the old monotone clip
  masked; 89% of rejected star-frames are EE>1 before norm radius. Frame exclusion then removes all
  139 frames (>57% drop rate per frame).
- **Verdict A-1b CONFIRMED** stands (minimum 5x3 proxy cell 12.0 mmag > 10 mmag gate from Step 1d
  broken-method table; not re-litigated).
- **Exact consolidated magnitude:** still open. Step 2 must address COG admissibility on crowded
  real fields (isolation, deblending, or neighbor-aware COG) before G6 can pass and M2 can land.
- **Milan patch choice** (options i/iii/iv): B.5/B.6 on repaired harness inconclusive (NaN).

## Docs impact

| Doc | Action |
|-----|--------|
| `VYVAR_AUDIT_CLOSURE_REGISTER.md` | A-1 row: 203 mmag superseded; exact value open post-Step 1e |
| `VYVAR_AUDIT_FINAL.md` | D5-1: Step 1e G7 pass; no real-data decisive number |
| `VYVAR_STATE.md` / `VYVAR_ROADMAP.md` | Step 1e done; Step 2 gated on COG admissibility |
| `closure_a1_reference_fixture.py` | unchanged; G0-G5 still pass |
| `VYVAR_DECISIONS.md` | no entry |

## Errors (if any)

None. G6 failure is an acceptable pre-registered outcome.

## Files changed

| File | Role |
|------|------|
| `dev/tools/closure_step1e_differential_aperture.py` | new Step 1e harness (R1-R4, G6/G7, M1-M4) |
| `dev/results/CURSOR_RESULT_closure_step1e.md` | this report |
| `dev/results/CURSOR_RESULT_closure_step1d.md` | V8-V10 VOID markers |
| `docs/VYVAR_AUDIT_CLOSURE_REGISTER.md` | A-1 row update |
| `docs/VYVAR_AUDIT_FINAL.md` | Step 1e paragraph |
| `docs/VYVAR_STATE.md` | Step 1e status |
| `docs/VYVAR_ROADMAP.md` | Step 1e status |

## Commands

```bash
git push origin main   # 665c76a (2531607, fc33672, 665c76a)

python dev/tools/closure_a1_reference_fixture.py
python dev/tools/closure_step1e_differential_aperture.py --gate-only

python dev/tools/closure_step1e_differential_aperture.py \
  --draft Archive/Drafts/draft_000435_snapshot_skysurface_20260716 \
  --step1b-json tmp/closure_step1b_results.json \
  --out tmp/closure_step1e_results.json \
  --cache tmp/closure_step1e_ee_cache.npz \
  --rebuild-cache
```
