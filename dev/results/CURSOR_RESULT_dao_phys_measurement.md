CURSOR RESULT - 2026-08-06 (DAO-PHYS-1)

What I did
Read-only measurement of DAO detection threshold physics on three drafts using stored MASTERSTAR frames and pipeline DAO helpers (import only). Script: `dev/tools/dao_phys_measure.py`. Machine-readable output: `dev/results/dao_phys_measure.json`. No changes to `src_py/`, `config.json`, or registry.

---

## 1. Premise check - reference frames and anchor reproduction

| Draft | Reference frame (from infolog / artefact) | MASTERSTAR path | Recorded pass-1 | Reproduced pass-1 | Anchor |
|-------|-------------------------------------------|-----------------|-----------------|-------------------|--------|
| draft_000501 | `TOI-1131.01.b_2025-04-22_23-59-57_V.fits` | `Archive/Drafts/draft_000501/platesolve/V_60_2/MASTERSTAR.fits` | 1654 | **1618** (97.8%) | **OK** (within 3%) |
| draft_000435_snapshot_skysurface_20260716 | DAO on platesolve ref (infolog) | `.../platesolve/NoFilter_60_2/MASTERSTAR.fits` | 2552 | **2534** (99.3%) | **OK** |
| draft_000500 | MASTERSTAR only (no infolog) | `.../platesolve/NoFilter_60_2/MASTERSTAR.fits` | - | **2879** | **Unanchored** |

**Preprocessing matched pipeline:** sky/median subtraction (`sigma_clipped_stats`), auto binning (`bfac=2` for 501, `1` for wide rigs), `_mean_bin2d_for_dao`, `_dao_convolved_background_rms_adu`, threshold `N_equiv x rms_conv` with `dao_detection_n_equiv=3.78`, `DAO_STAR_FINDER_NO_ROUNDNESS_FILTER`, FWHM from header `VY_FWHM` via `dao_detection_fwhm_pixels`.

**Catalogue counts (post-merge / SNR, from artefacts):**

| Draft | Merged | After SNR | DAO_ONLY | Gaia-matched |
|-------|--------|-----------|----------|--------------|
| 501 | 1670 | 1668 | 696 | 972 |
| 435 | 3777 | 2951 | 109 | 2842 |
| 500 | - | 4122 | 561 | 3561 |

501 anchor gap (?36 detections, 2.2%): likely pass-1 prefilter ordering or minor frame/header drift; within tolerance - measurements proceed.

---

## 2. Inverted-image false-alarm measurement (primary result)

Sky-subtracted reference frame negated; identical DAO configuration. At `k=3.78`:

| Draft | N_detected (pass-1) | N_inverted | DAO_ONLY observed | Positive-only residual* |
|-------|---------------------|------------|-------------------|-------------------------|
| 501 | 1618 | **225** | 696 | **471** |
| 435 | 2534 | **336** | 109 | 0 (clamped) |
| 500+ | 2879 | **999** | 561 | 0 (clamped) |

\* `max(0, DAO_ONLY ? N_inverted)` - hot pixels / cosmic rays / positive-only structure not present on inverted frame.  
+ Unanchored.

**Interpretation (501):** Pure symmetric noise false alarms (`N_inverted ? 225`) are well below observed DAO_ONLY (696). The bulk excess (~471) is **positive-only** - consistent with hot pixels, cosmic rays, residual calibration structure, or other non-symmetric artifacts. This **confirms** the brief's hypothesis that Gaussian noise alone does not explain DAO_ONLY on 501.

**435 / 500:** `N_inverted > DAO_ONLY` - inverted count exceeds the final DAO_ONLY census. Many inverted detections never enter the merged catalogue (pass-2 merge, Gaia match, SNR cap). The decomposition formula is most informative when `N_inverted ? DAO_ONLY`; here it only bounds noise false alarms from below.

### k curve (501 Newton V)

| k | N_detected | N_inverted | threshold (ADU) |
|---|------------|------------|-----------------|
| 3.0 | 4334 | 2327 | 13.7 |
| **3.78** | **1618** | **225** | **17.3** |
| 4.0 | 1446 | 143 | 18.3 |
| 4.5 | 1208 | 67 | 20.6 |
| 5.0 | 1098 | 35 | 22.9 |

### k curve (435 wide)

| k | N_detected | N_inverted |
|---|------------|------------|
| 3.78 | 2534 | 336 |
| 4.5 | 2087 | 49 |
| 5.0 | 1923 | 34 |

### k curve (500 wide, unanchored)

| k | N_detected | N_inverted |
|---|------------|------------|
| 3.78 | 2879 | 999 |
| 5.0 | 1713 | 452 |

---

## 3. Analytic comparison - N_res and N_FA(k)

Using header FWHM (not binned FWHM), `?_PSF = FWHM/2.3548`, `N_res = N_pix / (2? ?_PSF^2)`, `N_FA(k) = N_res x (1 ? ?(k))`.

| Draft | N_pix | FWHM (px) | N_res | N_FA(3.78) | N_inverted | Ratio N_inv/N_FA |
|-------|-------|-----------|-------|------------|------------|------------------|
| 501 | 6.53x10? | 2.572 | 8.71x10? | **68** | 225 | **3.3x** |
| 435 | 2.91x10? | 3.207 | 2.50x10? | **20** | 336 | **17x** |
| 500 | 2.91x10? | 3.830 | 1.75x10? | **14** | 999 | **73x** |

**vs brief predictions:** 501 `N_res ~ 9.4x10?`, `N_FA(3.78) ~ 74` - measured 8.7x10? / 68 (consistent). DAO_ONLY 696 ? N_FA - **confirmed**.

**Model vs measurement:** The independent-resolution-element model **underestimates** inverted false alarms by 3-73x depending on draft. Expected: DAOFIND convolves noise, correlating pixels. Section 1 is the measurement; section 2 is an upper-bound-ish model, not a calibrated predictor.

---

## 4. Noise consistency and pedestal

Gain and read noise from `resolve_gain` / `resolve_read_noise` (header + EQUIPMENTS DB).

| Draft | Camera | gain (e?/ADU) | RN (e?) | median (ADU) | ?_conv (ADU) | ?_pred (ped=0) | **R** | pedestal inverted (ADU) |
|-------|--------|---------------|---------|--------------|--------------|----------------|-------|-------------------------|
| 501 | C3-26000 | 3.12 (header) | 2.6 (DB) | 33479 | 4.57 | 104 | **0.044** | **~33416** |
| 435 | QHY294MM | 3.17 | 15.2 | 1955 | 63.4 | 25.3 | **2.51** | negative (unphysical) |
| 500 | QHY294MM | 3.17 | 15.2 | 2199 | 55.8 | 26.8 | **2.08** | negative (unphysical) |

**501:** R ? 1 - frame noise far below Poisson noise of the quoted median. Median is an **additive calibration pedestal**, not sky. Matches brief prediction (R ~ 0.08; difference from ? source: convolved RMS vs sigma-clipped std).

**435 / 500:** R > 1 - measured noise exceeds photon+RN prediction with pedestal=0 (VYVAR-calibrated control frames). Inverted pedestal negative ? formula mis-specified for these frames (true sky not zero-offset from median alone).

**Read-noise gap (Newton):** DB `READNOISE_E=1.3`; resolved runtime **2.6 e?** used (param_resolver path). QHY294MM: DB 7.6 e?, resolved **15.2 e?**.

### Photon transfer curve (spatial bins, 16x16 grid)

| Draft | Usable | Slope d(var)/d(median) | PTC gain | Pedestal x-intercept |
|-------|--------|------------------------|----------|----------------------|
| 501 | yes | **?10.4** (negative) | unusable | ~33486 ADU |
| 435 | yes | **?3.3** (negative) | unusable | ~2849 ADU |
| 500 | yes | +4.77 | ~0.21 e?/ADU (wrong) | ~1609 ADU |

501/435: **negative PTC slope** - pre-calibrated frames with flat large-scale structure; PTC fit has no leverage for gain/RN. 500 slope positive but gain implausible - insufficient sky-level variation across bins. **Honest gap:** PTC cannot independently verify pedestal on these MASTERSTAR stacks.

---

## 5. Shape statistics (sharpness / roundness)

Pass-1 detections joined to `masterstars_full_match.csv` within 5 px (binned?full coordinates for 501).

### Sharpness (median / p05 / p95)

| Population | 501 | 435 | 500 |
|------------|-----|-----|-----|
| DAO_ONLY | 0.928 / 0.81 / 1.08 | 0.518 / 0.29 / 1.26 | 0.577 / 0.22 / 0.95 |
| Gaia-matched | 0.882 / 0.80 / 0.94 | 0.646 / 0.39 / 0.82 | 0.671 / 0.31 / 0.94 |

**Contradiction vs hot-pixel hypothesis:** On 501, DAO_ONLY sharpness is **higher** (broader peaks) than Gaia-matched, not lower. A sharpness upper bound does **not** preferentially remove DAO_ONLY without heavy Gaia loss.

### Sharpness trade-off (501) - selected points

| sharpness ? | DAO_ONLY removed | Gaia lost |
|-------------|------------------|-----------|
| 0.90 | 65% | 35% |
| 0.94 | 43% | 4.3% |
| 0.99 | 23% | 0.4% |
| 1.03 | 12% | 0% |

**No knee** with ?50% DAO_ONLY removal and ?2% Gaia loss. Same conclusion on 435 and 500.

### Roundness trade-off

Tight roundness bounds remove Gaia and DAO_ONLY together (comatic / corner stars). **Confirms** disabling roundness filter was correct; restoring default (?1, 1) would cost real astrometric anchors.

---

## 6. Trivial physical filters (from catalogue CSV)

| Filter | 501 DAO_ONLY | 501 Gaia | 435 DAO_ONLY | 435 Gaia | 500 DAO_ONLY | 500 Gaia |
|--------|--------------|----------|--------------|----------|--------------|----------|
| negative flux | **20.4%** | 1.4% | 7.3% | 7.3% | 8.6% | 5.3% |
| fail snr50_ok | 98.7% | 60.7% | 85.3% | 64.8% | 44.2% | 66.9% |
| fail edge_safe_10px | 3.6% | 1.6% | 0.9% | 2.6% | 1.6% | 2.7% |
| **union removed** | **98.9%** | 61.1% | 85.3% | 66.0% | 44.9% | 68.1% |

**501 negative flux:** 142/696 = 20.4% - **independently verifies** census 142/696 = 20.4%.

**snr50_ok:** Already applied in pipeline SNR filter; high fail rate on DAO_ONLY reflects low-SNR spurious detections, but union with snr50 is not a free post-hoc filter (would also remove 61% of Gaia on 501).

---

## 7. FWHM predicted vs measured

Assumptions: ? = 550 nm, FWHM_atm = **2.5 arcsec** (not fitted), FWHM_diff = 1.03?/D, pixel floor 1.2 px.

| Draft | Telescope | scale (?/px) | FWHM_pred (px) | VY_FWHM (px) | ratio meas/pred |
|-------|-----------|--------------|----------------|--------------|-----------------|
| 501 | DDT 300/1200 | 1.30 | **1.95** | 2.57 | **1.32** |
| 435 | CZ 200/200 | 9.77 | **1.20** (floor) | 3.21 | **2.67** |
| 500 | CZ 200/200 | 9.77 | **1.20** (floor) | 3.83 | **3.19** |

**501:** Predicted ~1.95 px vs measured 2.57 px - excess ~32% from focus/tracking (matches brief).

**Wide rigs:** Predicted hits pixel floor; measured 3.2-3.8 px - **optics- and pixel-limited**, not seeing-limited. Seeing prior meaningless for threshold scaling on wide rig.

Ratio usable as focus/tracking diagnostic on Newton; not on wide rig where pixel floor dominates.

---

## 8. Decision table

| measure | draft | DAO_ONLY removed | Gaia-matched lost | would prevent 501 outcome? | risk |
|---------|-------|------------------|-------------------|---------------------------|------|
| restore sharpness upper bound | 501 | no knee (?50% DAO w/ ?2% Gaia) | n/a | **No** | comatic if roundness re-enabled |
| reject negative flux | 501 | 20% | 1.4% | partial only | low Gaia cost |
| noise-consistency R gate | 501 | frame reject | n/a | **Yes** (R=0.044) | flags pre-cal pedestal frames |
| union trivial filters | 501 | 99% | 61% | Yes but unusable | destroys Gaia depth |
| FAR k (N_FA=1) | 501 | k?4.73, N_det?1158 | depth ?460 vs 1618 | trade-off | purity vs catalogue depth |
| FAR k (N_FA=10) | 501 | k?4.23, N_det?1335 | depth ?283 | trade-off | milder |
| FAR k (N_FA=1) | 435 | k?4.46, N_det?2105 | - | n/a | 435 already clean |
| FAR k (N_FA=10) | 435 | k?3.94, N_det?2393 | - | n/a | still above current 2534? below |
| reject negative flux | 435 | 7% | 7% | n/a | symmetric loss |
| R gate | 435 | R=2.5 | n/a | No | misfires on calibrated wide |
| FAR k (N_FA=1) | 500+ | k?4.39, N_det?2175 | - | unanchored | high inverted noise |

**FAR depth cost (501):** At k=3.78 pass-1=1618. Raising k to 4.23 (N_FA=10) ? ~1335 (?17%). k=4.73 (N_FA=1) ? ~1158 (?28%). Inverted false alarms drop to ~107 and ~53 respectively - does not directly cap DAO_ONLY without knowing positive-only fraction at higher k.

---

## 9. Contradictions and honest gaps

**Confirmed**
- Fixed k does not fix false-detection count across rigs (N_res differs 5x; N_inverted differs more).
- 501 DAO_ONLY excess is not explained by Gaussian N_FA alone; positive-only residual dominates.
- 501 has massive calibration pedestal (R ? 1).
- Wide rig is pixel-limited; seeing prior irrelevant.
- Roundness filter should stay disabled.

**Refuted or weakened**
- Sharpness upper bound as DAO_ONLY discriminator on 501: **refuted** - DAO_ONLY sharper than Gaia, no useful knee.
- Analytic N_FA as quantitative false-alarm predictor: **refuted** by 3-73x underestimate vs inverted measurement.
- PTC independent pedestal check: **not usable** on these frames.

**Gaps**
- draft_500 unanchored (no infolog pass-1).
- Newton DB read noise (1.3 e?) vs resolved (2.6 e?).
- Shape stats match pass-1 detections to final catalogue (~92% of Gaia on 501); pass-2-only sources omitted.
- Decomposition breaks when N_inverted > DAO_ONLY (435, 500).

---

## Errors

None during measurement run.

## Files changed

- `dev/tools/dao_phys_measure.py` (new)
- `dev/results/dao_phys_measure.json` (new)
- `dev/results/CURSOR_RESULT_dao_phys_measurement.md` (this file)

Re-run: `python dev/tools/dao_phys_measure.py --write-json dev/results/dao_phys_measure.json`

---

## Errata - 2026-08-06 (DAO-PHYS-2 review)

### E1. Sharpness direction mis-stated (section 5 / section 9)

**Original text** described DAO-PHYS-1 draft_501 DAO_ONLY sharpness 0.928 vs Gaia 0.882 as 'higher (broader peaks)' and treated that as contradicting the hot-pixel hypothesis.

**Correction:** In photutils `DAOStarFinder`, sharpness is approximately the central pixel above neighbours divided by the Gaussian amplitude - **higher means sharper** (more concentrated). Default `sharphi = 1.0` rejects overly sharp sources (cosmic rays / hot pixels). On draft_501, DAO_ONLY is **sharper** than Gaia-matched (0.928 vs 0.882), which is the direction a hot-pixel hypothesis predicts.

**Restated verdict:** The effect is in the predicted direction but **weakly separating**. A cut at sharpness ? 0.94 removes ~43% of DAO_ONLY for ~4.3% Gaia loss - not 'refuted', but not a clean knee.

### E2. Rig-dependent sharpness sign (new finding)

On **435** and **500**, ordering **reverses**: DAO_ONLY sharpness **lower** (broader) than Gaia (435: 0.518 vs 0.646; 500: 0.577 vs 0.671). Artifact populations differ between rigs - a better reason a single sharpness rule cannot work than the original wording.

### E3. R comparison used mismatched noise scales

DAO-PHYS-1 `R_pedestal0` used **rms_conv** (convolved image) against per-pixel Poisson prediction. DAO-PHYS-2 recomputes with **std_clip** (see `phys2.noise_R` in updated JSON). Pedestal conclusion for 501 **survives** (R_std ? 0.082).

### E4. Newton read_noise_db_note withdrawn

C3-26000 gain x4 and RN x2 for header bin2=2 are **mutually consistent** with bin1 DB values (DAO-PHYS-2 M1c). The warning about DB READNOISE_E mismatch was a non-problem for Newton.

**Primary DAO-PHYS-1 result unchanged:** inverted-frame false alarms ~225 vs DAO_ONLY 696 on 501; ~471 positive-only excess.

