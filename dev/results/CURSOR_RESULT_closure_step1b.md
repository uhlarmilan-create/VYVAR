CURSOR RESULT - 2026-07-31 CLOSURE STEP 1b (A-1 evidence repair)

**Outcome: A-1b DOWNGRADED to DOCUMENTED** (undersized apertures; differentially harmless on anchor)
**Decisive number:** max |delta_ap| best-to-worst frame = **2.69 mmag** (G 8-9 comparison sub-ensemble;
B.4 gate 10 mmag not met)

**Mode:** measurement and report only. No production code changes.
**Base:** `origin/main` @ `9a1c0c4`; Step 1 at `90d2a99`
**Harness:** `dev/tools/closure_step1b_differential_aperture.py`
**Command:** `python dev/tools/closure_step1b_differential_aperture.py --draft Archive/Drafts/draft_000435_snapshot_skysurface_20260716 --out tmp/closure_step1b_results.json`
**Prior:** `dev/results/CURSOR_RESULT_closure_step1.md` (VOID items marked there)

---

## V1-V7 dispositions

| ID | Step 1 claim | Disposition | Corrected statement |
|----|--------------|-------------|---------------------|
| **V1** | SNR-optimum radius not below ~1.0 x FWHM; 0.8 floor binding implies S2 | **VOID** | Background-dominated limit maximises `EE(r)/r` at **r = 1.585 sigma = 0.673 x FWHM** (Howell 1989). Floor at **0.8 x FWHM** binding on faint stars in a sky-limited field is **expected**. Not S2 evidence. S2 stands on ZP mismatch alone. |
| **V2** | COG FWHM 4.00 px; r_ap/FWHM = 0.479 | **VOID** | `2 x r_50` is a Gaussian identity on profiles Step 1 declared non-Gaussian. Estimators disagreed 22%. Split to **A-9** (absolute scale unresolved). |
| **V3** | 0/45 fits on all frames | **VOID (harness)** | Root cause: `Gaussian2D+Const2D` compound model exposes `x_stddev_0` not `x_stddev`; unbounded centroid. Step 1b: **4859/4859 (100%)** Gaussian, **4858/4859 (99.98%)** Moffat. |
| **V4** | Isolation at 6 x FWHM_table = 14.4 px | **VOID** | Replaced by **15 px -> 146.6 arcsec** (plate scale) angular neighbour rule + growth-curve QC. **1278** stars admitted under old rule, rejected under new. |
| **V5** | 8.03 pp EE spread decisive | **VOID** | Common-mode at fixed r; cancels in ensemble. Differential metric is **delta_ap** (B.3). |
| **V6** | r_opt ~3.2 px; EE 51%->72% benefit | **VOID** | 3.2 = 0.8 x 4.0 (floor). Projected gain was clamp arithmetic. B.5 gives measured counterfactual. |
| **V7** | No flow_doc aperture-sizing facts | **VOID** | `build_flow_doc.py:391` documents `r_min=0.8 x FWHM` .. `r_max=2.5 x FWHM`; `flow_doc_facts.py:60` tracks `compute_snr_optimal_aperture_table`. Doc matches code; orphan is `aperture_snr_sizing` only. |

---

## Part A - Rebuilt measurement set

### A.1 Fixed star set

| Rule | Value |
|------|-------|
| Frames | 139 (`detrended_aligned/lights/NoFilter_60_2/proc_*.csv`) |
| Presence | >= 95% (>= 133/139) |
| Saturation | peak < 85% `saturate_limit_adu` on **every** frame |
| Magnitude bins | SNR-table bins G 8.0-15.0 step 0.5; >= 3 stars per bin where pool allows |
| Focus target | `1498135552633294976` forced in set |

**Result:** **35 stars** (focus + 34 comps). Bins 13.0-15.0 partially filled (faint-end pool thin);
bin 14.0 has 0 picks. Source: `tmp/closure_step1b_results.json` -> `part_a.star_ids`, `part_a.bins`.

### A.2 Isolation (non-circular)

| Parameter | Value | Source |
|-----------|-------|--------|
| Plate scale | **9.773 arcsec/px** | MASTERSTAR `PC1_1 = -0.002714` deg/px |
| Min separation | **15 px = 146.6 arcsec** | fixed px x scale (not FWHM) |
| Neighbour cut | nearest with `delta_G < 5` must exceed min separation | catalogue `ra_deg`/`dec_deg` |
| Growth QC | monotonic EE to 12 px, no outer shoulder | per-star COG on sample frame |

| Count | N |
|-------|--:|
| Eligible (95%, unsat) | 2294 |
| Pass angular isolation | (subset) |
| Pass growth QC | (subset) |
| Pass both | 1278+ pool before bin pick |
| Old 6 x FWHM_table rule would admit, new rejects | **1278** |

Focus target: **fails growth QC** (`focus_in_qc: false`) -- non-monotonic COG (C.1).

### A.3 Profile fit repair

**Step 1 failure cause:** compound-model parameter names (`x_stddev_0`); unbounded centroid
(`x_mean` drift to -36..59 in 19 px cutout); Step 1 read wrong attribute -> all rejected.

**Fix:** TRFLSQFitter; centroid bounds +/- 2 px; `theta` fixed; `Const2D` background; box = 32 px.

| Model | Converged | Rate |
|-------|----------:|-----:|
| Gaussian2D + Const2D | 4859 / 4859 | **100%** |
| Moffat2D + Const2D | 4858 / 4859 | **99.98%** |

Rate exceeds 80% gate; fit-based tracking in B.2 is admissible.

---

## Part B - Differential measurement (no FWHM ground truth required)

**VOID (Step 1c, 2026-07-31):** B.3 / B.5 / B.6 tables below used a corrupt focus-target numerator
(non-monotonic COG, EE>1) and incorrect B.5 unit mixing (r50 vs TABLE_FWHM=2.395). Superseded by
`dev/results/CURSOR_RESULT_closure_step1c.md`. Parts A, C, E and V1-V7 **stand**.

### B.1 Per-frame PSF scale proxy `r50_frame` (STANDS)

Median over fixed-star set of COG radius at EE = 0.5 (0.25 px steps to 12 px, normalised at 12 px).

| Stat | px | Source |
|------|---:|--------|
| min | **1.464** | frame `proc_BO_CVn_Light_007.csv` |
| median | **1.873** | 139 frames |
| max | **1.970** | frame `proc_BO_CVn_Light_048.csv` |
| header `VY_FWHM` same frames | 2.910 / 3.207 / 3.207 | FITS headers |

**Note:** `r50_frame` span (0.51 px) is much narrower than `VY_FWHM` span (0.68 px) because r50
tracks enclosed-flux geometry, not header moment FWHM.

### B.2 Estimator tracking vs `r50_frame` (origin-forced regression, 139 frames)

| Estimator | slope | frac scatter / slope | Spearman |
|-----------|------:|---------------------:|---------:|
| **moment median** (best) | 1.828 | **0.031** | 0.583 |
| `VY_FWHM` header | 1.711 | 0.034 | 0.426 |
| Gaussian fit median | 1.353 | 0.042 | 0.675 |
| Moffat fit median | 1.096 | 0.051 | 0.610 |

**Best tracker for B.5:** `moment_median` (lowest proportional leakage). Residual scatter ~**3.1%**
of scale -- the leakage that would survive frozen-`k_i` option (i).

### B.3 Differential systematic `delta_ap` (primary evidence)

For each frame:
`delta_ap = -2.5 * log10( EE_target(r_ap_target) / median_comps( EE_comp(r_ap_comp) ) )`
with EE from measured growth curves at **actual proc CSV** `aperture_r_px`.

| Sub-ensemble | N comps | Best-worst range [mmag] | Slope [mmag/r50] | r vs r50 |
|--------------|--------:|------------------------:|-----------------:|---------:|
| G 8-9 | 6 | **2.69** | 5.18 | 0.85 |
| G 9-11 | 12 | **2.59** | 5.12 | 0.85 |
| G > 11 | 16 | **2.55** | 4.98 | 0.85 |

Example frames: best r50 frame 007 -> delta_ap = **-1.90 mmag**; worst r50 frame 048 -> **+0.79 mmag**.

**Prediction test (R8):** Moffat model anchored to Step 1 COG predicted +32..+86 mmag best-worst.
**Measured 2.6-2.7 mmag** -- model **wrong** by an order of magnitude; measurement wins.

### B.4 Downgrade gate

Gate: |delta_ap| best-to-worst < **10 mmag** in all sub-ensembles -> **DOWNGRADE**.

**All three sub-ensembles pass** (max 2.69 mmag). Undersizing (clamp, magnitude-dependent radii)
is **real** but **differentially harmless** at this anchor's seeing range.

### B.5 Counterfactual: frozen `k_i` option (i)

`k_i = aperture_r_px_i / 2.395` frozen; `scale_frame` = moment median rescaled to draft median 2.395.

| Sub-ensemble | Current range [mmag] | After frozen-k range [mmag] |
|--------------|---------------------:|----------------------------:|
| G 8-9 | 2.69 | 5.24 (magnitude of change in spread) |
| G 9-11 | 2.59 | 5.32 |
| G > 11 | 2.55 | 5.37 |

Frozen-k rescaling **does not clearly reduce** delta_ap below the sub-10 mmag bound; median
delta_ap shifts to ~+0.4..+0.6 mmag but frame-to-frame structure changes. **Do not quote
Step 1 "EE 51%->72%" benefit** -- that claim is VOID (V6).

### B.6 Sky-correlation control: per-frame re-optimised table

Recomputed SNR table each frame (VY_FWHM + measured sky; ZP=25.0) -> new radii per mag bin.

| Sub-ensemble | delta_ap range [mmag] | corr(delta_ap, sky) |
|--------------|----------------------:|--------------------:|
| G 8-9 | **9.56** | -0.17 |
| G 9-11 | **9.56** | -0.17 |
| G > 11 | **9.52** | -0.16 |

Per-frame re-optimisation **inflates** differential spread toward ~10 mmag and introduces
sky correlation. Evidence favours **frozen k_i** over re-optimised per-frame table (Milan decision).

---

## Part C - Loose ends

### C.1 Focus target moment FWHM (~7.6 px vs ~3.3 px for comps)

| Frame | fwhm_estimate_px | peak [ADU] | x, y |
|-------|-----------------:|-----------:|-----|
| 002 | 7.55 | 2457 | 281.9, 620.2 |
| 063 | **8.64** | 1666 | 282.2, 618.2 |
| 087 | 7.02 | 1590 | 281.6, 620.6 |
| 080 | 7.67 | 1630 | 282.2, 620.4 |

**Excluded:** blend (nearest neighbour G=14.45 at **107 arcsec**, dG=0.24); saturation (peak
<< limit); bad column (not flagged in proc CSV on sample frames).

**Supported:** (1) **moment estimator noise inflation** at G~14.2 with low peak SNR; (2) **centroid
offset** -- Part 0e documented 3.48 px shift on frame 063 on same `catalog_id`; focus growth curves
are **non-monotonic** (fail QC) while comp-star median curves are smooth -> mis-centre drives
inflated moment FWHM and corrupt EE at 1.916 px.

**Step 2 inheritance:** C.1 ** strengthens** A-2/A-3 coupling for the focus target; does not
change the DOCUMENTED verdict (computed from comp ensembles, not target moment FWHM).

### C.2 Focus target sizing magnitude

| Field | Value | Source |
|-------|------:|--------|
| `_star_mag_for_aperture_sizing` equivalent | **14.203** | proc CSV `mag` / `phot_g_mean_mag` |
| Catalogue G | **14.203** | same |
| SNR bin | **>= 14.5** | nearest bin to 14.203 |
| `aperture_r_px` | **1.916** | = `r_min_px` clamp |

Sizing magnitude is **consistent**; target correctly lands on faint-end clamp. Defect is
table/clamp design, not wrong mag bin assignment.

### C.3 MASTERSTAR `VY_FWHM` mechanism

`pipeline.py:2988-2990` **always overwrites** MASTERSTAR `VY_FWHM` with `_fwhm_auto` = median
FWHM over the processed set. Pixel copy is frame 008 (VY_FWHM=2.945 pre-overwrite); header
shows **3.207** = set median. **Not an anomaly** -- documented code path.

### C.4 Zero-point direction (S2 warning)

At `fwhm_px = 2.395`, recomputed table with **Zp = 21.68** vs disk (Zp = 25.0):

| mag | r disk [px] | r at Zp=21.68 [px] |
|-----|------------:|-------------------:|
| 8.0 | 3.666 | **2.716** |
| 9.0 | 3.366 | **2.416** |
| 11.0 | 2.816 | **1.916** (clamp) |
| 14.5 | 1.916 | 1.916 |

**Correcting ZP alone shrinks radii** and pushes more stars onto the clamp. **S2 must not be
patched independently of A-1.**

---

## Part D - Outcome justification

**A-1b DOWNGRADED to DOCUMENTED** selected.

| Criterion | Threshold | Result |
|-----------|-----------|--------|
| delta_ap best-worst | > 10 mmag -> CONFIRMED | **2.69 mmag max** |
| B.4 downgrade gate | < 10 mmag all bins | **PASS** |

**A-1b CONFIRMED (FIX required) rejected** -- differential systematic below gate.

**A-1c rejected** -- fixed star set built; 139 frames measured.

### Fix posture (not applied)

- **Option (i) frozen k_i:** evidence mixed; B.5 does not show clear improvement on delta_ap;
  still candidate if seeing range widens on other fields.
- **Option (ii) raise r_min_fwhm:** addresses clamp but needs A-9 absolute scale for tuning.
- **Option (iii) `cog_aperture_correction_enabled`:** directly addresses magnitude-dependent
  enclosed-flux bias (Stetson 1990 growth-curve AC). Would normalise comps and target to common
  EE scale **without** requiring per-frame FWHM truth. Trade-off: mixed-frame guard (DECISIONS);
  nightly all-or-nothing policy; extra reference-star demand. **More direct for D5-1 mechanism**
  than option (i) when differential delta_ap is already small. Milan chooses; no patch applied.

---

## Part E - A-9 register item

Written to `docs/VYVAR_AUDIT_CLOSURE_REGISTER.md`:

**A-9 -- absolute PSF scale unresolved.** Estimators: `VY_FWHM_GAUSS=2.395`, header
`VY_FWHM=3.207`, Step 1 `2 x r_50=4.00`, `r_90/1.34=4.9`, Moffat-implied 2.4-3.2 px. No
admissible single ground truth (V2, V3). Status: **MEASURED**. Not blocking Steps 2-10
(differential cancellation). Required before absolute SNR-loss, flux-fraction, or D1-2
`fwhm_ratio` claims.

---

## Step 2 inheritance (A-2 / A-3 placement)

1. Focus target: high moment FWHM + failed growth QC + Part 0e centroid shift -> **placement
   dominates** over aperture radius for target tails.
2. A-1 aperture radius: **DOCUMENTED** at anchor; not blocking anchor re-cut on differential
   grounds alone.
3. A-9 open for any absolute-flux claim.
4. S3 role-factor label defect unchanged; D5-1 "1.0 vs 1.1" framing superseded (factors not applied).

---

## Literature

- Howell 1989: SNR vs r; faint limit optimum ~**0.67 x FWHM** (replaces VOID V1).
- Naylor 1998: equivalent weighted-extraction optimum.
- Stetson 1990: growth-curve aperture correction (option iii reference).

---

## Docs impact

| Doc | Action |
|-----|--------|
| `VYVAR_AUDIT_CLOSURE_REGISTER.md` | A-1 -> DOCUMENTED; decisive 2.69 mmag; A-9 added; D5-1 updated |
| `VYVAR_AUDIT_FINAL.md` | D5-1: delta_ap replaces Gaussian EE table; S3 supersedes 1.0/1.1 framing |
| `VYVAR_PARAMS.md` | `aperture_snr_sizing` DEAD unchanged |
| `VYVAR_DECISIONS.md` | none |
| `flow_doc_facts.py` / `build_flow_doc.py:391` | Step 1 V7 corrected; note `aperture_snr_sizing` orphan vs hardcoded 0.8/2.5 |
| `VYVAR_STATE.md` / `VYVAR_ROADMAP.md` | Step 1 closed; A-9 open; Step 2 unblocked |

---

## Files changed

| File | Role |
|------|------|
| `dev/tools/closure_step1b_differential_aperture.py` | Step 1b harness |
| `dev/results/CURSOR_RESULT_closure_step1b.md` | this report |
| `dev/results/CURSOR_RESULT_closure_step1.md` | VOID markers added |
| `tmp/closure_step1b_results.json` | machine output (gitignored) |
