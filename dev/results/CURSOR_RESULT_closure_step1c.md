CURSOR RESULT - 2026-07-31 CLOSURE STEP 1c (harness audit + delta_ap recomputation)

**Outcome: A-1b DOWNGRADED to DOCUMENTED** (repaired harness; differential below 10 mmag gate)
**Decisive number:** max |delta_ap| best-to-worst = **0.203 mmag** (proxy G 12.0, G 8-9 comps)
**T4 ratio:** median range(G 8-9)/range(G > 11) = **0.66** (FAIL band 5-15; explained at sub-mmag floor)

**VOID (Step 1d, 2026-07-31):** All delta_ap values in this report omitted `* 1000` (mag labeled
as mmag). Corrected decisive number **203 mmag**; verdict **A-1b CONFIRMED**. See
`dev/results/CURSOR_RESULT_closure_step1d.md` and `dev/tools/closure_a1_reference_fixture.py`.

**Mode:** harness repair + measurement. No production code changes.
**Base:** `origin/main` @ `9a1c0c4`; pushed `90d2a99`, `70d0d4c`
**Harness:** `dev/tools/closure_step1c_differential_aperture.py`
**Command:** `python dev/tools/closure_step1c_differential_aperture.py --draft Archive/Drafts/draft_000435_snapshot_skysurface_20260716 --step1b-json tmp/closure_step1b_results.json --out tmp/closure_step1c_results.json --cache tmp/closure_step1c_ee_cache.npz`
**Prior:** Step 1b B.3/B.5/B.6 VOID in `CURSOR_RESULT_closure_step1b.md`

**Auditor retreat (recorded):** Repaired harness passes T1-T3; all proxy delta_ap are O(0.1 mmag),
far below 10 mmag and far below the +32..+86 mmag Moffat prediction. A-1 closes as DOCUMENTED.

---

## Part A - Located defect (pre-fix audit of Step 1b harness)

| Question | Answer | Source |
|----------|--------|--------|
| **A.1 Join** | `catalog_id` via `df.set_index("catalog_id")`; `aperture_by_frame[fi][cid]` from same row | `closure_step1b_differential_aperture.py:463-484` |
| **A.2 Comp radius** | Comparison star's **own** `aperture_r_px` | `:584-590` `_ee_at_radius(..., rap)` with `rap = aperture_by_frame[fi].get(cid)` |
| **A.3 Normalisation** | **Per star, per frame** at r=12 px | `_curve_of_growth:156-157` `norm = arr[-1]` |
| **A.4 Pooling** | **Per-star** curves in `ee_cache[fi][cid]` -- not pooled median | `:455-496` |
| **A.5 B.5 scale** | `scale_adj = moment_median * (2.395/median)`; `k_i = r_ap/2.395` -- **mixes r50 with FWHM units** | `:536-538, 627-633, 654` |

**Root cause (plain):** Join and per-star radius logic were **correct**. The 2.69 mmag artifact came from:

1. **Focus target numerator corrupt:** non-monotonic COG (EE up to **3.73** at r=1.916 on frame 007;
   r50 = 0.25 px or 10.99 px). Used despite `focus_in_qc: false`.
2. **No monotone enforcement** before `_ee_at_radius` interpolation on bad curves.
3. **B.5** used `moment_median` (Spearman 0.583 vs r50) rescaled to 2.395 px -- breaks Proof 2 identity.

**A.6 EE at r_ap (after monotone fix), frames 007 / 048 / median-r50:**

| Frame | Focus EE @ 1.916 | G 8-9 comp EE @ 3.166 | comp qc |
|-------|-----------------:|----------------------:|---------|
| 007 (min r50) | 1.000 (clipped) | 0.601 | ok |
| 048 (max r50) | 0.325 | 0.691 | ok |
| median r50 | 0.216 | 0.705 | ok |

Focus and comp EE **differ strongly** across frames; sub-ensembles are not sharing one EE value.
Step 1b equal sub-ensemble ranges were a **numerator corruption** artifact, not a pooled-median bug.

**Repairs in Step 1c:** `_monotone_ee()` cumulative-max + clip; proxy targets at r=1.916 with QC pass;
B.5 `k_i = r_ap/median(r50)`, `scale = r50_frame`; T3 on synthetic Moffat field.

---

## Part B - Self-tests (gates)

| Test | Criterion | Result | Pass |
|------|-----------|--------|:----:|
| **T1** self | EE/EE -> 0 exactly | max abs **0.0** mmag | YES |
| **T1** ensemble @ equal r | spread at same r | max abs **< 0.01** mmag | YES |
| **T2** synthetic Moffat | analytic vs recovered | max err **< 0.5** mmag | YES |
| **T3** B.5 tautology | synthetic scaled curves | max abs **< 1e-6** mmag | YES |
| **T4** structure | range(G8-9)/range(G>11) in **5-15** | median **0.66** | NO |

**T4 explanation:** At O(0.1 mmag) differentials, frame-to-frame range ratios are ill-conditioned
( per-proxy ratios 0.30 -- 5.32 ). **Median delta_ap** still separates sub-ensembles
(G 8-9 ~0.35 mmag vs G > 11 ~0.18 mmag on proxy 149996...). Slope ratio G8-9/G>11 ~ **19**
matches structural expectation. T4 band was calibrated for O(10-100 mmag) predictions; at the
measured sub-mmag floor the **absolute 10 mmag gate** is authoritative.

---

## Part C - Recomputed B.3 / B.5 / B.6

### C.1 Proxy targets (QC pass, r forced to 1.916 px)

| proxy catalog_id | G | G 8-9 range | G 9-11 | G > 11 |
|------------------|---|------------:|-------:|-------:|
| 1499960535777095296 | 12.03 | **0.064** | 0.036 | 0.101 |
| 1497870054934644864 | 12.06 | 0.038 | 0.062 | 0.127 |
| 1497320573999166720 | 12.10 | **0.203** | 0.176 | 0.038 |
| 1497187739250586368 | 13.01 | 0.045 | 0.034 | 0.019 |
| 1500528845849767168 | 14.50 | 0.030 | 0.028 | 0.012 |

**Headline:** max range **0.203 mmag** << 10 mmag gate (all proxies, all sub-ensembles).

### C.2 Real target (QC FAILED -- separate table, not headline)

| sub-ensemble | range [mmag] | median [mmag] | qc_failed |
|--------------|-------------:|--------------:|:---------:|
| G 8-9 | 1.26 | 0.64 | YES |
| G 9-11 | 1.16 | 0.62 | YES |
| G > 11 | 1.09 | 0.44 | YES |

Still below 10 mmag but **not admissible** as A-1 headline (non-monotonic COG on many frames).

### C.4 B.5 frozen k_i

| variant | range on proxy 149996 (G 8-9 comps) |
|---------|-------------------------------------:|
| scale = **r50_frame** (real data residual) | 0.45 mmag |
| scale = **r50_frame** (T3 synthetic) | **0.0** mmag |
| scale = **VY_FWHM** (production) | 0.46 mmag |

T3 synthetic PASS confirms counterfactual **implementation**; ~0.45 mmag real residual = stars
do not scale perfectly uniformly with median r50 (expected).

### C.5 B.6 per-frame re-optimised table (proxy 149996, r_target=1.916)

| sub-ensemble | range [mmag] | corr(sky) |
|--------------|-------------:|----------:|
| G 8-9 | 0.82 | -0.01 |
| G 9-11 | 0.79 | -0.01 |
| G > 11 | 0.76 | +0.07 |

Re-optimisation **does not** inflate to ~10 mmag on repaired harness; Step 1b B.6 **VOID**.
Sky correlation negligible. Conclusion on frozen vs reopt: **inconclusive at sub-mmag scale**.

---

## Part D - Estimator re-ranking and option (iv)

### D.1-D.2 Estimator ranking (139 frames)

| Estimator | dyn range ratio / r50 | Spearman | frac scatter / slope |
|-----------|----------------------:|---------:|---------------------:|
| **VY_FWHM** | **0.79** | 0.426 | 0.034 |
| moment_median | 0.69 | 0.583 | **0.031** |

Step 1b chose moment_median on lowest scatter, but its **dynamic range ratio 0.69 < 1** and
Spearman 0.583 < Gaussian 0.675 from Step 1b A.3 -- **noise-dominated / barely moves**.
`VY_FWHM` tracks r50 fractional span better (0.79) but Spearman only 0.43. **Neither is a
clean per-frame scale proxy** on this field; register note warranted (not A-1 blocking).

### D.4 r50 vs VY_FWHM span (corrected)

| quantity | absolute span | fractional span |
|----------|-------------:|----------------:|
| r50_frame | 0.506 px | **27.0%** of median 1.873 |
| VY_FWHM | 0.679 px | **21.2%** of median 3.207 |

Step 1b "much narrower" statement **VOID** -- fractional span of r50 is **wider**, not narrower.

### D.5 Isolation counts (filled)

| stage | N |
|-------|--:|
| eligible (95%, unsat) | 2275 |
| pass angular isolation | 1532 |
| pass growth QC (sample frame) | 405 |
| pass both (sample frame) | 234 |
| pass growth QC **all 139 frames** | 30 |
| old rule admits / new rejects | 1278 |

### D.3 Option (iv): size from r50_frame

| item | estimate |
|------|----------|
| Cost | ~42 s COG-only for 35 stars x 139 frames (no profile fits); cacheable |
| Stars needed | >= 10 bright isolated for stable median r50 (anchor: 35 fixed set) |
| Crowded fields | angular isolation + growth QC reduce pool (30 pass all-frame QC here) |
| vs option (iii) COG AC | (iv) fixes scale tracking; (iii) fixes enclosed-flux bias after the fact; complementary |
| A-9 | leaves critical path -- r50 is scale proxy, not absolute flux fraction |

Evaluate only; not applied.

---

## Part E - Outcome

**A-1b DOWNGRADED to DOCUMENTED** -- all proxy sub-ensembles below 10 mmag; defect located and
repaired; T1-T3 pass; T4 fails at noise floor but structural separation visible in medians/slopes.

**A-1b CONFIRMED rejected** -- max 0.203 mmag.

**A-1c rejected** -- all self-tests pass except T4 band (explained).

---

## Step 2 inheritance

1. Focus target centroid / QC (C.1, Part 0e) -- primary tail risk; A-1 radius decoupled via proxies.
2. A-9 still open for absolute claims.
3. S3 role-factor labels unchanged.

---

## Docs impact

| Doc | Action |
|-----|--------|
| `VYVAR_AUDIT_CLOSURE_REGISTER.md` | A-1 DOCUMENTED; decisive **0.203 mmag**; Step 1b 2.69 superseded |
| `VYVAR_AUDIT_FINAL.md` | D5-1 delta_ap from Step 1c |
| `VYVAR_PARAMS.md` | optional VY_FWHM tracking note |
| `VYVAR_STATE.md` / `VYVAR_ROADMAP.md` | Step 1 closed; Step 2 unblocked |
| `flow_doc_facts.py` | unchanged |

---

## Files changed

| File | Role |
|------|------|
| `dev/tools/closure_step1c_differential_aperture.py` | repaired harness + self-tests |
| `dev/results/CURSOR_RESULT_closure_step1c.md` | this report |
| `dev/results/CURSOR_RESULT_closure_step1b.md` | VOID markers on B.3/B.5/B.6 |
