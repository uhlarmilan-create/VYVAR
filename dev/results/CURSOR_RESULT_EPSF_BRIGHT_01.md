CURSOR RESULT - 2026-08-23T06:00:00Z (EPSF-BRIGHT-01)

What I did
Phase 1 UI fix (separate commit): removed science-set table `.head(50)` cap, added row-count
caption, and per-star overlay caption when chi2 gate drops epochs. Phase 2 sandbox measurement
(M1-M3) on draft 516 science-set proc CSVs - chi2 gate mechanism, chi2 vs mag, ratio vs
brightness/linearity. **STOP before Phase 3** (FD-A/FD-B await architect + Milan).

---

## Phase 1 - UI fix

**Commit:** `086fb44` - `fix(epsf-ui): show full science table and report gated epoch drops`

| Change | Location |
|--------|----------|
| Full scrollable/sortable science-set table + caption `n shown / n science set` | `src_py/ui_epsf_dashboard.py` |
| Overlay caption `Epochs dropped by quality gate: N / M` when `psf_fit_ok` gaps exist | same |

**Gate:** `python dev/scripts/session_baseline_check.py --fast` @ `086fb44` - **OVERALL PASS**
(1508 passed, 32 skipped).

---

## Phase 2 - Measurement artifacts

**Session dir:** `dev/results/context/session_20260823_epsf_bright_01/`

| Artifact | Description |
|----------|-------------|
| `m1_bo_cvn_per_frame.csv` | BO CVn 134-row per-frame gate table |
| `m2_chi2_vs_mag.csv` | 192 full-coverage stars: median chi2, predicted inflation |
| `m2_chi2_vs_mag.png` | Scatter: median chi2 vs mag (+ threshold line) |
| `m3_ratio_vs_mag.csv` | 298 stars: PSF/DAO ratio vs mag, peak ADU |
| `m3_ratio_vs_mag_peak.png` | Ratio vs mag and vs peak/linearity |
| `m3_bright_cutout_residuals.csv` | 30 bright-star cutout residual probes (sample frames) |
| `summary.json` | Aggregated M1-M3 metrics |

**Harness:** `dev/sandbox/epsf_bright_01_measure.py` (sandbox only, not committed).

**Data:** draft 516 / `NoFilter_60_2`, science set n=333, chi2 threshold=50, gain=1.0.

---

## M1 - BO CVn per-frame gate table

**catalog_id:** `1498613634033133184` (BO CVn), **134 frames**

| Metric | Value |
|--------|------:|
| `psf_fit_ok` True | **54 / 134 (40.3%)** |
| `psf_chi2 >= 50` | **80 / 134 (59.7%)** |
| `psf_quality == bad` | **80** |
| `psf_quality == marginal` | **54** |
| chi2 median / p95 / min / max | **60.4 / 127.2 / 14.2 / 168.6** |

**Sample rows (first frames):**

| frame | psf_chi2 | psf_quality | psf_fit_ok | psf_snr |
|-------|---------:|-------------|:----------:|--------:|
| Light_001 | 48.9 | marginal | True | 253 |
| Light_003 | 54.3 | bad | False | 258 |
| Light_005 | 74.5 | bad | False | 251 |

**Verdict:** **CONFIRMED** - BO CVn sparse PSF overlay is explained by chi2 gate: 60% of epochs
fail `chi2 >= 50` ? `psf_quality=bad` ? `psf_fit_ok=False`. High SNR (~250) coexists with bad
chi2, consistent with statistic miscalibration rather than low-SNR fit failure.

---

## M2 - chi2 vs brightness (sky-only weights)

**Full-coverage stars (n=192, ?132/134 frames):**

| Metric | Value |
|--------|------:|
| Median obs chi2 (all stars) | 2.51 |
| Median predicted inflation (order proxy) | 1.0003 |
| Median obs / pred | 2.51 |
| Stars with median chi2 ? 50 | **43 / 192 (22%)** |
| Stars with pct_fit_ok = 0 | **29** |
| Stars with pct_fit_ok < 50% | **44** |
| corr(mag, median_chi2) | **?0.31** (brighter ? higher chi2) |
| mag < 10 with median chi2 ? 50 | **33 / 36 (92%)** |

**BO CVn (M2 row):** mag 9.72, median chi2 60.4, pred inflation 1.003, pct_fit_ok 40.3%.

**Bright-end examples (median chi2):**

| mag | median_chi2 | pred inflation | pct_fit_ok |
|----:|------------:|---------------:|-----------:|
| 5.94 | 35 931 | 1.021 | 0% |
| 7.12 | 897 | 1.001 | 0% |
| 9.72 (BO CVn) | 60.4 | 1.003 | 40% |

**Plot:** `m2_chi2_vs_mag.png` - clear monotonic rise toward bright end; dim stars cluster
chi2 ~ 0.8-3; threshold 50 cuts all stars brighter than ~mag 10.

**Mechanism verdict (M2):**

1. **Brightness-chi2 coupling CONFIRMED** - gate acts as de facto brightness cut on PSF branch.
2. **Sky-only miscalibration LIKELY PRIMARY** - dim stars show chi2 ~ 1-3 (healthy); bright-end
   inflation is orders of magnitude above threshold while SNR remains high. The sandbox order
   proxy `1 + F/(n_eff.?_sky^2)` (n_eff=9) stays ~1.0 for all stars and **underpredicts** absolute
   bright-end chi2; however the **observed pattern** (chi2 ~ 1 at faint, chi2 >> 50 at bright
   with good SNR) matches B3 sky-only weight defect, not random fit failure.
3. **Real shape misfit may add on top** for the very brightest (chi2 10^3-10?) - cannot fully
   separate without FD-A recalibrated statistic or controlled residual QA. Not required to explain
   BO CVn / mag < 10 gating.

---

## M3 - ratio vs mag + nonlinearity probe

**298 stars with PSF+DAO flux:**

| Metric | Value |
|--------|------:|
| BO CVn median psf_dao_ratio | **0.675** |
| Bright-10 ratio range | **0.09 - 0.70** |
| corr(mag, median_ratio) | **+0.51** (brighter ? lower ratio) |
| corr(median_ratio, peak/linearity) all stars | **?0.21** |
| corr(ratio, peak/linearity) bright-10 only | **+0.12** |

BO CVn: median peak/linearity ? 0.28 (well below saturation knee).

**Plot:** `m3_ratio_vs_mag_peak.png`

**Cutout residual probe:** 30 samples on 5 brightest comp stars x 6 frames. Sandbox rebuild of
model subtract used global ePSF + catalog x/y; several stars show huge core residuals
(O(10?) ADU) indicating position/flux-scale mismatch in the probe, not production fit failure.
Moderate-bright star `1496795041799526400` shows core_rms ~ 500 ADU - inconclusive for shape.

**Verdict (M3):**

1. **PSF/DAO ratio droop on bright stars CONFIRMED** (ratio vs mag ?=+0.51) - separate from
   chi2 gate; affects photometry comparison not overlay visibility.
2. **Sensor nonlinearity linkage WEAK** - peak ADU vs ratio correlation near zero; BO CVn and
   most gated stars are not near linearity knee. Ratio droop is **not primarily** a saturation
   artifact in this dataset; report as PSF-model / calibration domain, with D1-2 follow-up optional.
3. **Core shape mismatch at high SNR:** **INCONCLUSIVE** from sandbox cutouts; needs FD-A or
   dedicated residual QA with production fit coordinates.

---

## Fix recommendation (Phase 3 - NOT implemented)

| Option | When | Action |
|--------|------|--------|
| **FD-A (preferred)** | M2 confirms miscalibration | Add source Poisson term to fit error map: ?^2 = F_model/g + sky/g + (RN/g)^2; keep threshold 50 as genuine outlier gate; PSF columns only |
| **FD-B (stopgap)** | FD-A blocked | Brightness-normalized chi2 grading; document as interim |
| **Rejected** | - | Raising chi2 threshold alone (masks statistic defect) |

**Recommendation:** Proceed with **FD-A** after architect review. M1+M2 provide sufficient evidence
that sky-only weights inflate reduced chi2 on bright stars and drive dashboard dropout; aperture
fallback already protects science export.

**Phase 3 gates (when authorized):** `--fast` PASS + BO CVn near-full PSF coverage with honest
chi2 ~ 1-3 + `--full` recut confirming 9902d918 / 472bc9e4.

---

## Errors

None blocking. M3 cutout residual probe partially unreliable (sandbox coordinate/flux scaling);
marked inconclusive above.

---

## Files changed

| File | Change |
|------|--------|
| `src_py/ui_epsf_dashboard.py` | Phase 1 UI (committed `086fb44`) |
| `dev/results/CURSOR_RESULT_EPSF_BRIGHT_01.md` | This deliverable |
| `dev/results/context/session_20260823_epsf_bright_01/*` | M1-M3 measurement outputs (untracked) |

**STOP - awaiting architect review + Milan decision on Phase 3 (FD-A vs FD-B).**
