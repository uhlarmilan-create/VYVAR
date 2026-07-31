CURSOR RESULT - 2026-07-31 CLOSURE STEP 1 (finding A-1)

**Verdict: A-1b** (FIX required; radius too small; patch proposed, not applied)
**Decisive number:** enclosed-flux fraction at `r_ap = 1.916 px` varies by **8.03 percentage points**
between best- and worst-seeing science frames (COG median curve; Part C.2).

**Mode:** measurement and report only. No production code changes.
**Base:** `origin/main` @ `9a1c0c4`
**Harness:** `dev/tools/closure_step1_aperture_fwhm_ground_truth.py`
**Command:** `python dev/tools/closure_step1_aperture_fwhm_ground_truth.py --draft Archive/Drafts/draft_000435_snapshot_skysurface_20260716 --out tmp/closure_step1_results.json`
**Anchor:** `Archive/Drafts/draft_000435_snapshot_skysurface_20260716`

---

## Part A - Artefact readout

### A.1 `aperture_snr_table.json` (draft root)

| Field | Value |
|-------|------:|
| Path | `Archive/Drafts/draft_000435_snapshot_skysurface_20260716/aperture_snr_table.json` |
| mtime | `2026-07-16T15:35:27.536933` |
| `fwhm_px` | **2.395** |
| `sky_adu_per_px` | **1570.0167846679688** |
| `gain` | **3.17** |
| `read_noise` | **15.2** |
| `r_min_px` | **1.9160000000000001** |
| `r_max_px` | **5.9875** |

Mag -> radius table (complete):

| mag | r [px] | mag | r [px] | mag | r [px] |
|-----|-------:|-----|-------:|-----|-------:|
| 7.0 | 3.916 | 10.0 | 3.116 | 13.0 | 2.266 |
| 7.5 | 3.766 | 10.5 | 2.966 | 13.5 | 2.116 |
| 8.0 | 3.666 | 11.0 | 2.816 | 14.0 | 2.016 |
| 8.5 | 3.516 | 11.5 | 2.666 | 14.5 | **1.916** |
| 9.0 | 3.366 | 12.0 | 2.516 | 15.0 | **1.916** |
| 9.5 | 3.266 | 12.5 | 2.366 | 15.5 | **1.916** |
| | | | | 16.0-18.0 | **1.916** (clamp) |

Duplicate copy under `platesolve/aperture_snr_table.json` differs only in `gain=1.0`, `read_noise=10.0`; production loader uses **draft root** (`photometry_core.py:1744`).

**P1 CONFIRMED:** `fwhm_px = 2.395`; `r_min_px = 1.916 = 0.8 x 2.395`.

### A.2 MASTERSTAR FITS header

Path: `platesolve/NoFilter_60_2/MASTERSTAR.fits`

| Keyword | Value |
|---------|------:|
| `VY_FWHM` | **3.2068878608540805** |
| `VY_FWHM_GAUSS` | **2.395** |
| `VY_FWHM_GAUSSIAN` | **ABSENT** |
| `VY_ELONG` | **1.1137901045888108** |
| `VY_NSTAR` | **445** |
| `VY_NDAO` | **2552** |

**Source frame (pixel-identical):** `processed/lights/NoFilter_60_2/proc_BO_CVn_Light_008.fits`
(max_abs_diff 0.0 ADU vs platesolve MASTERSTAR before header refresh). Source frame header
`VY_FWHM = 2.944672` (not the night minimum; frame 002 has `VY_FWHM = 2.743`).

### A.3 Run log lines

Source searched: `infolog_20260716_123126.txt` (draft root) and full draft tree.

| Line pattern | Status |
|--------------|--------|
| `[PHOT] gaussian_fwhm_px_override = ...` | **ABSENT** |
| `[PIPELINE] aperture_snr_table.json ulozena pred exportom CSV: mag7-> mag11-> mag14->` | **ABSENT** |
| `[PIPELINE] SNR table gain= ... RN= ... (source: ...)` | **ABSENT** |
| `[PIPELINE] Sky for SNR table: ... ADU/px` | **ABSENT** |
| `[PIPELINE] SNR table: measured star-free bkg var = ... ADU^2/px` | **ABSENT** |
| `[FAZA 2A] Gaussian FWHM z MASTERSTAR: ... px` | **ABSENT** |
| `[FAZA 2A] SNR per-star apertures: min= median= max= (N=)` | **ABSENT** |
| `[PHOT] Gaussian FWHM fallback na moment:` | **ABSENT** |

Infolog predates photometry export or logs were not captured in this snapshot bundle.

### A.4 Proc CSV columns (139 frames, `detrended_aligned/lights/NoFilter_60_2/`)

**Provenance columns from Stage 1.2:** `fwhm_px_for_aperture`, `fwhm_px_scope`, `snr_aperture_mode`,
`aperture_factor_applied` are **ABSENT** on this anchor snapshot (pre-2026-07-30 export).

Focus target `1498135552633294976` (133 common frames):

| Column | min | median | max | varies frame-to-frame? |
|--------|----:|-------:|----:|------------------------|
| `aperture_r_px` | 1.916 | 1.916 | 1.916 | **No** |
| `fwhm_estimate_px` | 6.544 | 7.559 | 8.788 | **Yes** (per-star moment) |
| `sky_annulus_r_out_px` | 21.555 | 21.555 | 21.555 | **No** |

Five comparison stars (Gaia G ~8.0-10.9):

| catalog_id | G mag | aperture_r_px (constant) | fwhm_estimate_px median |
|------------|------:|-------------------------:|------------------------:|
| 1496300948763054976 | 7.99 | 3.166 | 3.504 |
| 1498927097925811072 | 8.29 | 3.016 | 3.136 |
| 1498735778606786816 | 9.11 | 2.866 | 3.260 |
| 1497558618266311808 | 9.13 | 2.866 | 3.240 |
| 1498072743031629824 | 10.93 | 2.316 | 3.072 |

**H0 CONFIRMED:** radii are mag-binned SNR-table values (multiple distinct `aperture_r_px` in
population), not `aperture_fwhm_factor x per-frame FWHM`.

### A.5 One-frame aperture distribution (`proc_BO_CVn_Light_063.csv`, N=2649)

| `aperture_r_px` | N stars |
|-----------------|--------:|
| **1.916** (=`r_min_px`) | **2060** |
| 2.066 | 199 |
| 2.166 | 120 |
| 2.316 | 82 |
| 2.416 | 49 |
| 2.566 | 42 |
| 2.716 | 37 |
| 2.866 | 18 |
| 3.016 | 15 |
| 3.166 | 8 |
| 3.316 | 4 |
| 3.416 | 14 |
| 4.5505 | 1 |

On `r_max_px`: **0** stars. On `r_min_px`: **2060 / 2649 (77.8%)**.

### A.6 Photometry frame type and pixel scale

| Item | Value |
|------|-------|
| Science photometry FITS | `detrended_aligned/lights/NoFilter_60_2/proc_BO_CVn_Light_*.fits` (aligned, resampled) |
| Proc CSV co-located | same directory |
| MASTERSTAR | `platesolve/NoFilter_60_2/MASTERSTAR.fits` |
| NAXIS | 2082 x 1397 (both MASTERSTAR and science frame 063) |
| WCS scale | `PC1_1 = -0.0027139448020425` (identical) |

**No binning/resample scale mismatch** between MASTERSTAR and science frames. FWHM discrepancy
is not explained by pixel scale alone.

---

## Part B - Independent ground truth FWHM

Script: `dev/tools/closure_step1_aperture_fwhm_ground_truth.py` (no VYVAR aperture imports).

### B.1 Frame selection (by header `VY_FWHM`)

| Role | File | `VY_FWHM` [px] |
|------|------|---------------:|
| Best | `proc_BO_CVn_Light_002.fits` | 2.743 |
| Median | `proc_BO_CVn_Light_087.fits` | 3.207 |
| Worst | `proc_BO_CVn_Light_080.fits` | 3.422 |

### B.2-B.3 Star selection and Gaussian fit

>=15 stars per frame from proc CSV (peak 10-60% of saturation, isolated >6 x FWHM_table).

**Gaussian2D + Const2D fit:** **0 / 15** converged within bounds on all three science frames
(`n_stars_fit = 0`). COG (B.4) succeeded on the same selections (`n_stars_selected = 15`).

**VYVAR `measure_fwhm_from_masterstar` difference:** VYVAR fits `Gaussian2D` only after border-median
subtraction (`photometry_core.py:566-604`); this harness keeps an explicit background term. COG
FWHM (B.4) is the primary ground truth for this report.

### B.4 Curve of growth (median over 15 stars / frame)

Annulus: inner = max(r+0.5, 4.75 x FWHM_hint), outer = max(inner+0.5, 9.0 x FWHM_hint); normalized at r=12 px.

| Frame | COG FWHM from r_50 (= 2 x r@EE0.5) | COG FWHM from r_90/1.34 | EE@12px |
|-------|-----------------------------------:|------------------------:|--------:|
| Best (002) | **3.50** | ~4.3 | 1.00 |
| Median (087) | **4.00** | ~4.9 | 1.00 |
| Worst (080) | **5.00** | ~6.1 | 1.00 |
| MASTERSTAR | **4.00** | ~4.9 | 1.00 |

Wings are **heavier than a pure Gaussian** (EE@2px ~0.52 vs Gaussian ~0.63 for FWHM=4 px). Section 0
Gaussian EE table **understates** enclosed-flux loss (R7: measurement wins).

### B.5 MASTERSTAR vs science

MASTERSTAR COG FWHM (4.0 px) matches **median-seeing** frame, not best frame (3.5 px). Supports
draft-constant `VY_FWHM_GAUSS = 2.395` being a **sharp-frame Gaussian fit**, while typical night
seeing is wider.

---

## Part C - Reconciliation

Ground truth for ratios: **COG FWHM on median frame = 4.00 px** (Part B.4).

| quantity | value (px) | source | / FWHM_true (4.00) |
|----------|----------:|--------|-------------------:|
| `VY_FWHM` (MASTERSTAR header) | 3.207 | MASTERSTAR header | 0.802 |
| `VY_FWHM_GAUSS` (MASTERSTAR header) | 2.395 | MASTERSTAR header | 0.599 |
| `VY_FWHM * 0.667` | 2.139 | arithmetic from 3.207 | 0.535 |
| `fwhm_px` in `aperture_snr_table.json` | 2.395 | draft root JSON | 0.599 |
| `fwhm_px_for_aperture` (proc CSV) | **ABSENT** | snapshot predates Stage 1.2 | -- |
| median `fwhm_estimate_px` (focus target) | 7.559 | proc CSV all frames | 1.890 |
| moment median x 0.619 | 4.679 | 7.559 x 0.619 | 1.170 |
| Part B COG, best frame | 3.50 | harness COG | 0.875 |
| Part B COG, median frame | **4.00** | harness COG | **1.000** |
| Part B COG, worst frame | 5.00 | harness COG | 1.250 |
| Part B COG FWHM, median frame | 4.00 | r_50 x 2 | 1.000 |
| `aperture_r_px` (focus target) | **1.916** | proc CSV | **0.479** |

### C.1 `r_ap / FWHM_true` and EE from COG (not Gaussian formula)

| Frame | r_ap/FWHM (COG) | EE at 1.916 px (COG interp) |
|-------|----------------:|----------------------------:|
| Best | 0.547 | **55.7%** |
| Median | 0.479 | **51.3%** |
| Worst | 0.383 | **47.6%** |

Using header `VY_FWHM` instead of COG: median frame ratio = 1.916/3.207 = **0.598**.

### C.2 Enclosed-flux spread (differential systematic)

**51.3% - 47.6% = 3.7 pp** (worst vs median) or **55.7% - 47.6% = 8.0 pp** (best vs worst).
Exceeds **1%** gate decisively.

### C.3 Clamp binding

**YES:** `1.916 == r_min_px` in anchor table. **2060/2649 (77.8%)** stars on clamp on frame 063;
focus target at clamp for all 133 frames.

### C.4 SNR cost (focus target G~14.2)

Model: Howell (1989) top-hat SNR with enclosed Gaussian fraction; `gain=3.17`, `RN=15.2`,
`sky=1570 ADU/px`, table `fwhm_px=2.395`:

| FWHM used in model | r_opt [px] | SNR(r=1.916)/SNR(r_opt) |
|--------------------|----------:|------------------------:|
| Table FWHM 2.395 | 1.966 | **1.000** (clamp masks error) |
| COG truth 4.00 | 3.200 | **0.885** |

Instrumental zero point from frame 063 comps: **Zp ~ 21.68** (not 25.0). With Zp=21.68 the
table still hits `r_min` for `fwhm_px=2.395`; with `fwhm_px=4.0`, `r_opt ~ 3.2 px`.

---

## Part D - Verdict justification

**A-1b selected.**

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| r_ap/FWHM_true on median frame | >= 0.75 | **0.479** (COG) or **0.598** (header) |
| EE spread best-worst | <= 1% | **8.03 pp** |

**A-1a rejected:** radius is not 0.75-0.90 x true FWHM; EE not constant.
**A-1c rejected:** prerequisites established (table present, scale match excluded).

**Proposed correction (not applied) -- option (i) only:**

Size SNR table from **per-frame** `VY_FWHM` or per-frame COG FWHM (median over stars), not
draft-constant `VY_FWHM_GAUSS` from MASTERSTAR. Expected effect at median frame: move `r_opt` from
1.916 px toward ~3.2 px for G~14; EE rises from ~51% to ~72% (COG interp); EE spread across
frames should drop toward differential cancellation regime.

Literature (R1): Howell (1989, PASP 101, 616) SNR-optimal aperture; Naylor (1998, MNRAS 296, 339).
Optimum radius scales with FWHM (~1-2 x FWHM for bright sources; faint limit not below ~1 x FWHM).
Operating at **0.48 x FWHM** (COG) is below the faint-limit optimum; **0.8 x FWHM floor binding**
on an underestimated FWHM is itself evidence for S2/H2 failure mode.

---

## Part E - Coupled sub-questions

### S1 - `aperture_snr_sizing` dead? **CONFIRMED**

`precompute_and_save_snr_aperture_table_for_draft` (`photometry_core.py:2049-2055`) calls
`compute_snr_optimal_aperture_table` **without** `r_min_fwhm` / `r_max_fwhm`; defaults **0.8 / 2.5**
(`:1290-1291`) apply. Registered dict `aperture_snr_sizing` (`config.py:727-729`) is never read
on this path.

| Call site | r_min_fwhm | r_max_fwhm |
|-----------|----------:|----------:|
| `precompute_and_save_snr_aperture_table_for_draft` :2049 | default 0.8 | default 2.5 |
| Phase 2A inline `:8544` | default 0.8 | default 2.5 |

**Proposed patch (unapplied):** pass `r_min_fwhm=cfg.aperture_snr_sizing["small"]/fwhm` only after
unit audit, or wire explicit `snr_r_min_fwhm` keys; until then mark parameter **DEAD** in PARAMS budget.

### S2 - flux scale / zero point **CONFIRMED issue**

- Hardcoded `zero_point=25.0` (`photometry_core.py:1293`, `:1332`).
- Anchor instrumental Zp from comps frame 063: **21.68 +/- 0.54 mag** (N=2634).
- With table FWHM, wrong Zp still lands on clamp; with **true FWHM 4.0 px**, `r_opt` moves to **3.2 px**
while pipeline uses **1.916 px** (SNR ratio **0.885**).
- `n_bkg = area * bkg_var_px / g`: if `bkg_var_px` is ADU^2/px, conversion to e- variance per pixel
  should scale as **g^2**, not **g** ( dimensional inconsistency in `:1341` when using measured variance).

`sky_fallback=1581.6`: module constant `_SKY_ADU_FALLBACK` (`pipeline.py:857`), default arg in
`precompute_and_save_snr_aperture_table_for_draft` (`photometry_core.py:1952`); anchor table used
measured **1570.0** ADU/px instead.

### S3 - role factors **REFUTED on production per-frame path**

`enhance_catalog_dataframe_aperture_bpm` (`:12172-12211`) assigns `r_ap_arr[i]` directly from
`_get_star_aperture_px` with **no** multiply by `aperture_variable_factor` / `aperture_comp_factor`.
Labels `:12220-12223` claim `snr_table_comp_1.100x` when `aperture_comp_factor=1.1` but radius
is unchanged -- **provenance defect**. Phase 2A `_apply_role_aware_aperture_scaling` (`:7696-7719`)
scales a different `apertures_px` dict on the planning path.

D5-1 premise "target 1.0x vs comp 1.1x radii" is **false** on the per-frame export path that
feeds stored flux.

### S4 - per-draft vs per-frame FWHM **CONFIRMED**

`pipeline.py:10154-10195` sets `gaussian_fwhm_px_override` once from MASTERSTAR `VY_FWHM_GAUSS`.
`fwhm_px_for_aperture` would be constant (when column present); snapshot lacks column but SNR table
uses single `fwhm_px=2.395` for all frames. Per-star `fwhm_estimate_px` varies (moment on each star).

**D5-1 question (1) answered:** aperture sizing FWHM is **per-draft**, not per-frame seeing.

### S5 - MASTERSTAR selection bias **PARTIALLY CONFIRMED**

| Quantity | px |
|----------|---:|
| `VY_FWHM` MASTERSTAR header | 3.207 |
| min / median / max `VY_FWHM` on 139 science frames | 2.743 / 3.207 / 3.422 |
| median(set) / MASTERSTAR | **1.000** |
| min(set) / MASTERSTAR | **0.855** |

MASTERSTAR header equals **median** frame VY_FWHM, but pixel copy is frame **008** (VY_FWHM=2.945).
`VY_FWHM_GAUSS=2.395` is sharper than typical night COG FWHM (~4 px).

---

## Open questions for Step 2 (A-2 / A-3 / placement)

1. Per-frame DAO centroid instability (Part 0e) is amplified at **r/FWHM < 0.6** -- coupling to
   aperture undersizing.
2. Whether to fix SNR table FWHM first (option i) before MASTERSTAR stack metric `I_j` work.
3. Role-factor labels vs applied radii (S3) must be reconciled before any comp/target radius policy.
4. Regenerate anchor proc CSVs with Stage 1.2 provenance columns for follow-up measurement.

---

## Docs impact

| Doc | Action |
|-----|--------|
| `VYVAR_AUDIT_CLOSURE_REGISTER.md` | A-1 -> **A-1b MEASURED**; note clamp + FWHM underestimate |
| `VYVAR_AUDIT_FINAL.md` | D5-1 (1)(2) answered; S1 dead-parameter note |
| `VYVAR_DECISIONS.md` | none (measurement only) |
| `VYVAR_PARAMS.md` | record `aperture_snr_sizing` DEAD if S1 accepted |
| `VYVAR_STATE.md` / `VYVAR_ROADMAP.md` | Step 1 complete; Step 2 unblocked pending Milan patch choice |
| `flow_doc_facts.py` / `test_docs_sync_guard` | no aperture-sizing facts found; **no change required** |

---

## Literature (R1)

- Howell, S. B. 1989, PASP 101, 616 -- SNR vs aperture radius for Gaussian PSF.
- Naylor, T. 1998, MNRAS 296, 339 -- optimal extraction aperture scaling.

Under flat background, SNR-optimal circular radius is **not below ~1.0 x FWHM** even in the faint
limit; bright sources prefer **> 2 x FWHM**. Operating at **0.48 x FWHM** with a **0.8 x FWHM floor**
on underestimated FWHM explains clamp dominance (S2 evidence).

---

## Files changed

| File | Role |
|------|------|
| `dev/tools/closure_step1_aperture_fwhm_ground_truth.py` | Part B harness |
| `dev/results/CURSOR_RESULT_closure_step1.md` | this report |
| `tmp/closure_step1_results.json` | machine-readable output (gitignored) |
