# CURSOR RESULT - DAO-ONLY VERIFY (2026-07-26)

Read-only verification. No code changes, no commits. Scratch: `tmp/dao_only_verify.py`,
`tmp/dao_verify_bisect.py`, `tmp/dao_only_verify_results.json`, `tmp/dao_verify_bisect/bisect_results.json`.

---

## What I did

Ran tests V1-V5 on `draft_000435` vs `draft_000450` `masterstars_full_match.csv` to decide
whether the 2596 extra `DAO_ONLY` rows in 450 (2705 vs 109) are spurious noise (Reading A) or
real faint stars beyond the Gaia DB cap G=17.5 (Reading B).

Data used:

| Path | Role |
|------|------|
| `Archive/Drafts/draft_000435/platesolve/NoFilter_60_2/masterstars_full_match.csv` | Anchor (109 DAO_ONLY) |
| `Archive/Drafts/draft_000450/platesolve/NoFilter_60_2/masterstars_full_match.csv` | Friday re-run (2705 DAO_ONLY) |
| `Archive/Drafts/draft_000450/detrended_aligned/lights/NoFilter_60_2/proc_*.csv` | 139 per-frame CSVs (V2) |
| `GAIA_DR3/vyvar_gaia_dr3.db` | Local Gaia cap check (`MAX(g_mag)=17.5`) |

---

## V1 - Photometric test (decisive)

### Zeropoint fit (GAIA_MATCHED only)

Instrumental magnitude: `inst = -2.5 * log10(flux)`. Linear fit `G = slope * inst + intercept`.

| Draft | n_fit | slope | intercept | r | scatter (MAD) | G range fitted |
|-------|------:|------:|----------:|--:|--------------:|----------------|
| 435 | 2538 | 0.890 | 21.139 | 0.845 | 0.70 mag | 6.29 - 16.50 |
| 450 | 3349 | 0.850 | 20.853 | 0.806 | 0.74 mag | 6.47 - 17.44 |

Fit is stable over the matched population; scatter ~0.7 mag is typical for passband mismatch
(unfiltered vs Gaia G).

### Estimated G for DAO_ONLY (zeropoint from matched rows)

| Metric | draft_435 (109) | draft_450 (2705) |
|--------|----------------:|-----------------:|
| est G p50 | 13.96 | **14.69** |
| matched G p50 | 13.52 | 14.05 |
| **est G < 16** | **100 (91.7%)** | **2235 (82.6%)** |
| est G in [16, 17.5] | 9 | 162 |
| **est G > 17.5** | **0 (0%)** | **26 (1.0%)** |
| flux invalid (no est G) | -- | 282 |
| brightest est G | 7.80 | 8.12 |
| faintest est G | 16.10 | 20.67 |

Gaia DB cap confirmed: `MAX(g_mag) = 17.5` in `vyvar_gaia_dr3.db` (211.7M stars).

### Histogram (0.25 mag bins, same axis)

**draft_435 DAO_ONLY** -- sparse, peaks at G~13.6-14.1 (24+16 rows); extends to G~16.1.
No rows estimated fainter than 17.5.

**draft_450 DAO_ONLY** -- **strong peak at G~14.1-14.9** (199+536+523+353 = 1611 rows, 60%
of all DAO_ONLY). Secondary tail to G~17.5 (49+40+20+24+12+17 = 162 in 16-17.5) and **only 26
rows (1%) beyond 17.5** (faintest est G~20.7).

**draft_450 GAIA_MATCHED** -- similar shape shifted ~0.5 mag fainter (p50 14.05 vs DAO p50 14.69);
matched population has substantial counts to G~17.5 (76 rows in 17.0-17.5 bin).

### V1 reading

- **Reading B rejected for the bulk.** If DAO_ONLY were real stars beyond the catalogue cap, the
  histogram would pile up **fainter than 17.5** with its bright edge at the cap. Instead, 450
  DAO_ONLY spans the **same G~13-16 range as matched sources**, with peak **brighter** than the
  matched p50. Only **26/2705 (1%)** sit beyond 17.5.
- **2235 rows (82.6%) have est G < 16** -- Gaia is essentially complete there; these cannot be
  unknown real stars. This is the cleanest spurious count.
- **435 control:** even the anchor pipeline assigns est G < 16 to **100/109 (92%)** of its DAO_ONLY
  rows. The 450 excess is quantitatively larger (2235 vs 100) but **qualitatively the same failure
  mode** -- unmatched detections at magnitudes where Gaia counterparts must exist.

---

## V2 - Persistence across frames

**Skipped.**

139 per-frame CSVs exist under `draft_000450/detrended_aligned/lights/NoFilter_60_2/proc_*.csv`,
but every file contains **only `GAIA_MATCHED` rows** (`master_reference_locked` mode). Zero
`DET_*` / `DAO_ONLY` names. Raw per-frame detection positions for unmatched peaks are not exported;
no substitute was synthesised.

---

## V3 - Spatial structure

Image size 2082 x 1397 px (from MASTERSTAR).

### Distance to nearest bright star (G < 12, among GAIA_MATCHED)

| Group | n | p10 (px) | p50 (px) | p90 (px) | < 20 px | < 50 px |
|-------|--:|---------:|---------:|---------:|--------:|--------:|
| 435 matched | 2842 | 0.0 | 33.0 | 65.8 | 854 | 2143 |
| 435 DAO_ONLY | 109 | 8.1 | 29.5 | 62.4 | 30 | 89 |
| 450 matched | 3993 | 0.0 | 34.7 | 67.0 | 1049 | 2952 |
| 450 DAO_ONLY | 2705 | 14.5 | 37.2 | 68.4 | 525 | 1944 |

450 DAO_ONLY are **slightly farther** from bright stars on average (p10 14.5 px vs 0 for matched,
expected since bright stars are mostly matched). **525/2705 (19%)** lie within 20 px of a G<12
star -- some halo/ringing association, but not dominant.

### Edge distance and quadrants

| Group | edge p50 (px) | edge p10 (px) |
|-------|-------------:|-------------:|
| 450 matched | 246 | 42 |
| 450 DAO_ONLY | 284 | 54 |

No strong edge pile-up; DAO_ONLY are if anything **slightly more central**.

Quadrant fractions (450 DAO_ONLY vs matched):

| Q | matched | DAO_ONLY |
|---|--------:|---------:|
| 0 (TL) | 25.4% | 17.9% |
| 1 (TR) | 18.2% | 4.9% |
| 2 (BL) | 26.1% | 33.6% |
| 3 (BR) | 30.3% | 43.6% |

Mild excess in lower half (Q2+Q3: 70% vs 58% for matched). Not a single-corner artifact.

### 2D density (8 x 6 bins, 450 DAO_ONLY)

Counts per bin range 0-132 (max in central-right bins around x~780-1560, y~465-1164). Excess is
**spread across the field** with higher density in the lower-right quadrant -- consistent with
structured background/residual pattern rather than isolated hot pixels. No single bin dominates
(132/2705 = 5% max).

435 control: 109 DAO_ONLY in 31/48 nonzero bins (max 10/bin) -- same general field coverage.

---

## V4 - Detrend difference characterisation and commit locate

### Difference image statistics (435 proc vs 450 plain, same frame indices)

| Frame | max abs diff (ADU) | mean abs diff | diff mean | diff sigma | var_large / var_small |
|-------|-------------------:|--------------:|----------:|-----------:|----------------------:|
| 001 | 1008 | 28.9 | +6.6 | 35.7 | **59.8** |
| 050 | 1248 | 21.5 | +4.7 | 26.6 | **20.3** |
| 139 | (not on disk for 450 plain name check) | -- | -- | -- | -- |

The inter-draft difference is predominantly a **smooth large-scale component** (order-2 sky surface /
background model), not small-scale noise: large-scale variance dominates small-scale by 20-60x.

### MASTERSTAR stack background and DAO threshold (sigma = 2.1)

| Draft | bg_median (ADU) | **bg_std (ADU)** | **threshold ADU** |
|-------|----------------:|-----------------:|------------------:|
| 435 | 1955.1 | **83.5** | **175.4** |
| 450 | 1954.4 | **62.0** | **130.2** |

450's lower bg_std (likely from missing sky-surface subtract on the stack input) drops the DAO
threshold by **~26%** (175 -> 130 ADU). Same 2.1-sigma knob bites deeper in ADU on 450.

Per-frame example (050): 435 bg_std 69.1 -> thresh 145 ADU; 450 bg_std 48.7 -> thresh 102 ADU.

### Git bisect (preprocess on scratch copies, commits in `tmp/dao_verify_bisect/`)

Tested `preprocess_calibrated_to_processed` on frames 001/050/139 from calibrated masters at:

| Commit | Label | Output behaviour | vs 8815c45 (pre-SKIPPROC) frame 050 |
|--------|-------|------------------|-------------------------------------|
| `8815c45` | pre-SKIPPROC | writes `proc_*.fits`, sky-surface subtract | (reference) |
| **`013cb0c`** | **SKIPPROC** | **in-place QC only, no proc copy, no sky subtract on mono** | **first divergence: mean abs 21.7 ADU, max 101 ADU** |
| `263c6e7` | SKIPPROC-QC fix | restores proc_ copy + sky subtract | **identical to 8815c45 (0 ADU diff)** |
| `c055ac3` | pre-OSC | in-place skip-only again | same as 013cb0c |
| `0f1c07f` | OSC-1 | in-place skip-only | same as 013cb0c |
| `224c442` | OSC-2 | in-place skip-only | same as 013cb0c |

**First commit producing different preprocess output: `013cb0c` (SKIPPROC).** The OSC pair
(`0f1c07f`, `224c442`) does **not** change mono preprocess pixels relative to post-SKIPPROC skip-only.

Archived detrended 435-proc vs 450-plain frame 050: mean abs diff **21.5 ADU** -- matches the
preprocess sky-surface offset measured in bisect (21.7 ADU). The full-pipeline detrended delta is
therefore **consistent with SKIPPROC-era loss of order-2 sky-surface subtract on mono frames**, not
a separate OSC regression.

Note: `263c6e7` temporarily restored the old processed/ path in code, but `draft_450` at HEAD
(`cb78b25`) uses skip-only again; the Friday re-run path matches 013cb0c+ behaviour.

---

## V5 - Verdict

**Reading A (mostly noise / spurious detections) is supported. Reading B (real stars beyond G=17.5)
accounts for at most ~1% of the excess.**

### Quantified split (draft_450, 2705 DAO_ONLY)

| Category | Count | Fraction | Basis |
|----------|------:|---------:|-------|
| **Spurious (est G < 16)** | **2235** | **82.6%** | Gaia essentially complete; no physical unknown stars |
| Ambiguous near cap (16 <= est G <= 17.5) | 162 | 6.0% | Near local DB limit; some could be real, most likely spurious given V1 shape |
| Possibly real beyond cap (est G > 17.5) | 26 | 1.0% | Only population Reading B predicts; upper bound on real faint stars |
| Unestimated (invalid flux) | 282 | 10.4% | No reliable photometric G; treat as unknown/spurious |

**Net:** ~**2235-2517 spurious** (including unestimated), ~**0-26 real** beyond catalogue depth, ~**162** ambiguous.

The 2596-row excess over 435 (2705 - 109) is **not** explained by Gaia incompleteness above 17.5.
It is explained by (a) **lower effective DAO threshold** on a smoother background (130 vs 175 ADU),
(b) **large-scale detrend difference** from SKIPPROC loss of sky-surface subtract (first at
`013cb0c`), and (c) detections at **G~14-15 where Gaia is complete** -- the same false-positive
regime seen in 435's 109-row control sample.

### What would settle the ambiguous 16-17.5 tail

Per-frame **raw DAO detection lists** (not master-locked CSVs) to run V2 persistence, or external
cross-match against a deeper catalogue (e.g. PS1) for the 162 near-cap rows and 282 flux-invalid rows.

---

## Errors

None. Frame 139 archived detrended path not verified (450 uses `BO_CVn_Light_139.fits`; present in
bisect but omitted from early V4 triplet due to initial glob miss).

---

## Files changed

None (read-only). Scratch only under `tmp/`.

**STOP remains in force. No commits.**
