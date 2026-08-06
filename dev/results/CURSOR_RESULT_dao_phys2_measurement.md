CURSOR RESULT — 2026-08-06 (DAO-PHYS-2)

What I did
Extended `dev/tools/dao_phys_measure.py` with DAO-PHYS-2 measurements (noise scale, Q-statistic, persistence, SNR-floor curves). Re-ran to `dev/results/dao_phys_measure.json`. No changes to `src_py/`, `config.json`, or registry.

---

## 1. R recomputed both ways (M1a)

Per-pixel prediction: `sigma_pred = sqrt(median*gain + RN²)/gain` (pedestal = 0).

| Draft | R (rms_conv) | R (std_clip) | Pedestal conclusion (501) |
|-------|--------------|--------------|---------------------------|
| 501 | **0.044** | **0.082** | **Survives** — R ? 1 either way |
| 435 | 2.51 | **3.31** | n/a (VYVAR-calibrated) |
| 500 | 2.08 | **1.98** | n/a |

DAO-PHYS-1 compared convolved RMS to per-pixel Poisson — apples to oranges. **std_clip is the correct comparator.** Brief expected values (501 **0.082**, 435 **3.31**, 500 **1.98**) confirmed.

501: median ~33?479 ADU cannot be sky at gain 3.12 — inferred photon-limited ? would be ~104 ADU; measured ?_clip = 8.51 ADU. Additive calibration pedestal reading **unchanged**.

---

## 2. Binning slope — white vs correlated noise (M1b)

Source-masked sky-subtracted frame; `sigma_clip` at bin factors 1, 2, 4, 8, 16; fit slope of log ? vs log N.

| Draft | Slope | Verdict | Interpretation |
|-------|-------|---------|----------------|
| 501 | **?0.91** | **white** | Excess vs Poisson is not spatially correlated at scales tested; consistent with mis-scaled ?_pred (pedestal), not sky structure |
| 435 | **?0.05** | **correlated** | Variance barely falls with binning — unresolved-source / flat-residual dominated (expected at 9.77?/px) |
| 500 | **?0.39** | **correlated** | Same class as 435 |

**R-gate implication:** No universal R threshold across rigs. Wide-rig R > 1 reflects correlated confusion, not necessarily RN mis-calibration alone.

---

## 3. Gain / read-noise convention (M1c)

### C3-26000 (501, header XBINNING=2)

| Quantity | DB (bin1) | Resolved | Check |
|----------|-----------|----------|-------|
| Gain | 0.78 e?/ADU | **3.12** (header) | 0.78 × 4 = 3.12 ? |
| RN | 1.3 e? | **2.6** (DB scaled) | 1.3 × 2 = 2.6 ? |

Resolver behaviour matches 2×2 mean-binning physics. **Retract DAO-PHYS-1 read_noise_db_note** — not a bug for Newton.

### QHY294MM (435/500, header XBINNING=2, DAO bfac=1)

| Quantity | DB | Resolved | Issue |
|----------|-----|----------|-------|
| Gain | 3.17 | **3.17** (header index map) | No scaling (header wins) |
| RN | 7.6 | **15.2** (DB × binning) | **Likely double-count** |

JOURNAL records `gain=3.17, RN=7.6` from **draft_303 NoFilter_60_2** — already at **bin2 session values**. `param_resolver` treats DB RN as bin1 and multiplies by `XBINNING=2` again. Correct RN for R at these sky levels is probably **7.6 e?**, not 15.2. Effect on R is minor here (sky-dominated) but **must be fixed before any RN-sensitive R gate on short exposures**.

---

## 4. MASTERSTAR frame combination (M1d)

| Draft | NCOMBINE header | std_clip MASTERSTAR | std_clip single light | Ratio |
|-------|-----------------|---------------------|----------------------|-------|
| 501 | 1 (implicit) | 8.510 ADU | 8.510 ADU (23-05-09 light) | **0.981** |
| 435 | 1 | 83.82 ADU | n/a (MASTERSTAR is ref) | 1.0 |
| 500 | 1 | 53.02 ADU | n/a | 1.0 |

**Verdict:** MASTERSTAR.fits is a **single reference frame**, not a noise-reducing stack. R is not confounded by stack averaging.

501 calibration_mode: **pre_calibrated** (from `draft_manifest.json`).

---

## 5. BPM presence and Q-statistic (M2a)

### Bad-pixel maps

| Draft | `*_dark_bpm.json` in draft tree | `on_bad_column` in proc CSVs |
|-------|--------------------------------|------------------------------|
| 501 | **0** | **0%** (all False; pre-cal has no dark master ? no BPM sidecar) |
| 435 | **0** | **0%** |
| 500 | **0** | **0%** |

501 ran **without bad-pixel information** (consistent with `calibration_mode=pre_calibrated` nulling dark master).

### Q = (?8 neighbours ? 8·bg) / (central ? bg)

PSF-predicted Q from header FWHM (501: 2.572 px ? Q_psf ? **4.36**; hot pixel ? **~0**).

| Draft | Q_psf | DAO_ONLY median | Gaia-matched median |
|-------|-------|-----------------|---------------------|
| 501 | 4.36 | **4.85** | 4.26 |
| 435 | 5.39 | 4.14 | 3.06 |
| 500 | 6.05 | 4.66 | 3.50 |

**501: DAO_ONLY sits at the PSF expectation, not Q ? 0.** Neighbour profiles are **PSF-like**, not single-pixel defects. This **refutes** the hot-pixel / fixed-pattern-as-single-pixel hypothesis for the bulk of DAO_ONLY on 501.

Q lower-bound trade-off (501): no bound removes most DAO_ONLY without major Gaia loss — populations overlap near Q ? 4–5.

---

## 6. Persistence test (M2b)

**Method:** 12 non_calibrated native lights; one DAO pass per frame (cap 200); reference catalogue (x,y) recurrence within 1px.

| Population | n sampled | Mean recurrence | p95 recurrence |
|------------|-----------|-----------------|----------------|
| DAO_ONLY | 200 | **0.0004** | **0.0** |
| Gaia-matched | 120 | **0.0** | **0.0** |

**Result:** DAO_ONLY positions do **not** recur at fixed native pixels across frames. Combined with Q ? Q_psf, **hot pixels at fixed detector coordinates are ruled out** as the dominant origin of the ~471 positive-only excess.

*(Gaia recurrence ? 0 is expected: reference (x,y) is on the master grid; stars drift in native coords frame-to-frame.)*

---

## 7. Negative-flux DAO_ONLY rows (M2c)

| Draft | n_negative | Fraction of DAO_ONLY |
|-------|------------|----------------------|
| 501 | **142** | **20.4%** (census verified) |

These rows have **positive `peak_max_adu`** (median peak above median ? +42 ADU) but **negative aperture `flux`**. Interpretation: local background estimate exceeds aperture sum — **oversubtracted neighbourhood or mis-centred aperture on a weak gradient**, not hot pixels (which would give positive flux and Q ? 0). Negative-flux rows are **spatially dispersed** (similar spatial std to all DAO_ONLY), not a single gradient lobe.

---

## 8. SNR floor / sigma units (M3)

Configured floor: `masterstar_prematch_peak_sigma_floor = 1.8`  
Floor ADU = median + k·?_clip.

| Draft | Floor ADU (k=1.8) | Below floor in current CSV | Recorded merged ? after SNR |
|-------|-------------------|----------------------------|----------------------------|
| 501 | **33?493.9** | **0** | 1670 ? 1668 (?2) |
| 435 | **2?105.9** | **0** | 3777 ? 2951 (?826) |

Floor arithmetic verified independently (501: 33478.6 + 1.8×8.51 = 33?493.9).

### Sigma units: (peak_max ? median) / ?_clip

| Population | 501 median | 501 p05–p95 |
|------------|------------|-------------|
| DAO_ONLY | **4.94 ?** | 2.54 – 8.57 |
| Gaia-matched | **19.1 ?** | 5.2 – 438 |

**501 DAO_ONLY is not marginal** — median ~4.9? above pedestal. A 1.8? floor cannot reach it; confirmed (0 rows below floor in post-SNR CSV).

### k_floor trade-off (501)

| k_floor | DAO_ONLY removed | Gaia lost |
|---------|------------------|-----------|
| 1.8 | 0% | 0% |
| 4.0 | 34% | 1.4% |
| 5.0 | **51%** | **4.1%** |
| 6.0 | 66% | 7.9% |
| 8.0 | 90% | 18% |

**No k_floor removes most of the positive-only population (?471) with ?2% Gaia loss.** At k=5 (first ?50% DAO removal), Gaia loss is already 4.1%. **Sigma-floor line of enquiry closed** for draft_501 without unacceptable depth loss.

Predicted-? floor would be circular (brief section 0): inferred sky from ?_clip reproduces measured ?.

---

## 9. Synthesis — what is indicated and what is ruled out

### Ruled out

| Hypothesis | Evidence |
|------------|----------|
| Hot pixels / single-pixel defects dominate DAO_ONLY on 501 | Q ? Q_psf (~4.8); persistence ? 0 |
| SNR sigma floor (k=1.8 or moderate k) fixes 501 | DAO median 4.9?; k=5 costs 4% Gaia |
| Predicted-? prematch floor | Circular with pedestal inference |
| MASTERSTAR stack suppressing ? | Single frame; ? matches raw light |
| Newton RN resolver bug | bin2 scaling correct for C3-26000 |
| Sharpness “contradicts” hot-pixel on 501 (see errata) | DAO_ONLY sharper; weak separator only |

### Still consistent / indicated

| Reading | Evidence |
|---------|----------|
| **Additive calibration pedestal** on 501 | R_std = 0.082; white noise after source mask |
| **Quiet-frame fixed-pattern / residual structure** | Positive-only excess ~471; detections at ~5? above ~33.5k pedestal with PSF-like profiles |
| **No BPM on pre-cal 501** | 0 sidecars; importer BPM never applied |
| **Wide-rig R > 1 is correlated confusion** | Binning slope ? 0; not white noise mis-scale |
| **QHY294MM RN double-count in resolver** | DB values appear bin2-native; ×2 again |

### Physically indicated interventions (measurement only — no threshold chosen)

1. **Bad-pixel / fixed-pattern mask** for pre-calibrated imports (501 had none) — targets positive-only PSF-like residuals, not caught by sharpness alone.
2. **Frame-level pre-cal QC** (R_std gate on 501-like pedestals) — flags quiet externally calibrated frames where small residuals become formally significant.
3. **Do not rely on sigma-floor or sharpness alone** on 501 — insufficient separation without Gaia depth loss.
4. **Fix QHY294MM RN resolver** before calibrating any R gate on short exposures.

---

## Contradictions vs brief hypotheses

- **Hot-pixel / BPM hypothesis (section 2): refuted** for bulk 501 DAO_ONLY (Q and persistence).
- **Pedestal / quiet-frame reading (section 0): supported** (R_std, white slope on 501).
- **SNR floor as “failed lever” (section 0): confirmed closed** — not marginal population.

---

## Errors

None (full run ~6 min including 501 persistence on 12 frames).

## Files changed

- `dev/tools/dao_phys_measure.py` (extended)
- `dev/results/dao_phys_measure.json` (updated)
- `dev/results/CURSOR_RESULT_dao_phys2_measurement.md` (this file)
- `dev/results/CURSOR_RESULT_dao_phys_measurement.md` (errata section)

Re-run: `python dev/tools/dao_phys_measure.py --write-json dev/results/dao_phys_measure.json`
