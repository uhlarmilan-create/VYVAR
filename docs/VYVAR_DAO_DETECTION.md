# VYVAR -- DAO detection and DAO_ONLY classification (closure reference)

**Date:** 2026-08-07
**Status:** Closed (DAO-CLOSE). Supersedes scattered result files from DAO-PHYS-1/2/2b, A-6, A-6b.
**Scope:** Detection-stage behaviour, informational census, and magnitude classification only.
No runtime gate; no row removal from catalogues.

---

## 1. How DAO detection works now

### 1.1 Two-pass MASTERSTAR DAO

Pass 1 runs `DAOStarFinder` on the resampled MASTERSTAR image with a convolved-RMS threshold.
Pass 2 seeds local DAO at Gaia positions that had no pass-1 neighbour within the astrometric
match radius.

- Pass-2 entry: `src_py/pipeline.py:7280-7406` (`_dao_targeted_pass2_unmatched_gaia`)
- Call sites: `src_py/pipeline.py:7651` (fast path), `src_py/pipeline.py:8402` (full cone path)

Pass 2 builds unmatched Gaia pixel positions, runs `DAOStarFinder` at each seed, and merges
new peaks into the pass-1 table when within separation limits.

### 1.2 Convolved-RMS threshold (Batch E / T4-1 Option B)

Detection threshold in ADU:

```
threshold = dao_detection_n_equiv * rms_conv
```

- Helper: `src_py/pipeline.py:7056-7075` (`_dao_detection_threshold_adu`)
- Applied at detection: `src_py/pipeline.py:7622-7624`, `8372-8374`
- Config default: `src_py/config.py:759` (`dao_detection_n_equiv = 3.78`)

`rms_conv` is the sigma of the image convolved with the DAO kernel (photon-limited noise model
on calibrated frames). `dao_detection_n_equiv` is dimensionless and was measured empirically
(Batch E Part 2b).

### 1.3 Pre-match peak sigma floor

Before Gaia matching, weak DAO peaks can be dropped using a median + k*sigma floor on peak ADU.

- Config default: `src_py/config.py:769` (`masterstar_prematch_peak_sigma_floor = 1.8`)
- Wired into platesolve: `src_py/pipeline.py:12249-12251`
- Pre-match SNR filter: `src_py/pipeline.py:8548-8574`
- Noise-floor helper: `src_py/photometry_core.py:1786-1801`

DAO-PHYS-2 showed no `k_floor` removes the draft_501 DAO_ONLY excess without unacceptable Gaia
depth loss (k=5 removes 51% DAO / 4.1% Gaia on draft_501).

### 1.4 DAO<->Gaia astrometric fit and retention

The astrometry optimizer re-matches detections to Gaia but **keeps all rows**, including
DAO-only detections:

- Retention comment: `src_py/astrometry_optimizer.py:1024-1042`
- `snr50_ok` computed but rows kept: `src_py/astrometry_optimizer.py:1044-1057`

Rows without a Gaia crossmatch are labelled `DAO_ONLY` in `source_type`.

### 1.5 Consumption gate (`snr50_ok`)

DAO_ONLY rows remain in `masterstars_full_match.csv` but are excluded from photometry admission
unless they pass `snr50_ok`:

- Comp selection admits DET stars only with `snr50_ok`: `src_py/comp_selection_per_target.py:514-523`
- Variability filter requires `snr50_ok=True`: `src_py/variability_detector.py:101`

On draft_501, 687/696 DAO_ONLY rows failed `snr50_ok` and never reached photometry.

---

## 2. What DAO_ONLY is and why rows are retained

`DAO_ONLY` means a DAO detection with no Gaia source within the astrometric match tolerance at
MASTERSTAR stage. Rows are retained for discovery QA and downstream diagnostics, not dropped at
detection (`src_py/astrometry_optimizer.py:1024`).

The old runtime gate `INV-MS-01` (WARN/FAIL on `dao_only_fraction`) was removed because thresholds
calibrated on one rig/calibration mode are not portable (see `docs/VYVAR_LIMITATIONS.md`,
INV-MS-01-REMOVED). VYVAR now reports an informational census instead of failing the run.

---

## 3. Magnitude classification (A-6 / A-6b)

Implemented in `src_py/dao_reconcile.py`. Wired before CSV write in `src_py/pipeline.py`
(~12470-12540). Additive columns only; no row removal.

### 3.1 Classes

| Class | Meaning |
|-------|---------|
| `artifact_negative` | Non-positive flux |
| `unmatched_in_range` | Implied G clearly below confirmable depth (purity signal) |
| `ambiguous_depth` | Implied G within +/- sigma_g of confirmable depth |
| `beyond_catalogue` | Implied G clearly above confirmable depth |
| `indeterminate` | Missing inputs, saturation, edge, or unresolvable depth |

### 3.2 Derived confirmable depth

`confirmable_depth_g = min(gaia_db_max_g_mag, effective_match_depth, cone_query_mag_limit)`
via `derive_confirmable_depth_g()` in `src_py/dao_reconcile.py`.

### 3.3 Per-row uncertainty (`sigma_g_row`)

```
sigma_g(row) = hypot(zp_residual_rms, 1.0857 / SNR(row))
```

Band classification uses `implied_g +/- sigma_g` against `confirmable_depth_g`. Large
`sigma_g` automatically widens bands into `ambiguous_depth`; **do not cap sigma_g into a
separate class** (self-protecting rule).

Population diagnostic `fleming_sigma_mag_population` is reported separately and is not used
for per-row margins.

### 3.4 Installation scope (critical)

**Class counts depend on the local Gaia DB build** (`max_g_mag`, row count, fingerprint) and
must **never** be compared across installations or wired to any gate. Comparable quantities:

- Per-row: `implied_g_mag`, `implied_g_minus_depth`, `sigma_g_row`
- Population: implied-G deciles, unmeasurable fraction, flux-to-G fit RMS

Reported in census log (`format_dao_only_census_log`), `pipeline_meta` flat keys
(`dao_only_class_meta_flat`), PDF/UI (`dao_only_report_lines`, `photometry_report.py`).

### 3.5 Unmeasurable fraction diagnostic

Fraction of DAO_ONLY rows with `sigma_g_row > 1.0 mag` is reported as
`sigma_g_unmeasurable_fraction` in census and `pipeline_meta`.

Because `sigma_g = hypot(zp_residual_rms, 1.0857 / SNR(row))`, every row has
`sigma_g >= zp_residual_rms` regardless of its photometry. With the measured ZP scatter
(501: 0.431; 435: 0.837; 500: 0.946 mag), a 1.0 mag threshold means different things per
draft:

- **draft_500** (`zp_rms` 0.946): the floor alone is almost 1.0 mag, so the reported 0.959
  is substantially a statement about the **flux-to-G calibration**, not individual detections.
- **draft_435** (0.837): 0.842 -- same effect, slightly weaker.
- **draft_501** (0.431): 0.388 requires `1.0857 / SNR` around 0.9 mag (SNR ~1.2), so here
  the figure does reflect genuinely faint rows.

Do not read 0.959 on wide rigs as "96% of detections are junk". On wide rigs the dominant
uncertainty in `implied_g_mag` is the global zero point fitted across a ~5.7 degree field at
~9.77 arcsec/px, not per-detection SNR. This is a known limitation of wide-rig classification;
the ZP model is not opened for rework here.

### 3.6 Flux-to-G fit quality

Implied G comes from a median zero-point fit of Gaia-matched rows (`fit_flux_to_g` in
`dao_reconcile.py`). Residual RMS per draft (verified DAO-CLOSE):

| Draft | ZP fit RMS (mag) |
|-------|------------------|
| 501 | 0.431 |
| 435 | 0.837 |
| 500 | 0.946 |

Nearly 1 mag scatter on wide rigs means `implied_g_mag` is a weak estimate there; class counts
must not be read as high-precision labels on drafts 435/500.

---

## 4. Measurement campaign (what was established)

### 4.1 Inverted-frame false alarms (DAO-PHYS-1)

On draft_501 at `N_equiv=3.78`:

- **225** symmetric noise false alarms on an inverted MASTERSTAR
- **696** observed DAO_ONLY
- **~471** positive-only excess (696 - 225); bulk is not Gaussian noise alone

### 4.2 Additive pedestal on pre-calibrated frames (DAO-PHYS-2)

draft_501 pre-calibrated median ~33487 ADU vs wide-rig VYVAR-calibrated ~2416 ADU.
Noise scale ratio `R_std = 0.082` (std_clip comparator; not convolved RMS vs per-pixel Poisson).
An additive pedestal makes a quiet frame; detection responds to structure, not rig geometry alone.

### 4.3 Binning noise structure

Log-log slope of noise vs binning N (DAO-PHYS-2):

| Draft | Slope | Interpretation |
|-------|-------|----------------|
| 501 | -0.91 | White (Poisson-like) |
| 435 | -0.05 | Correlated |
| 500 | -0.39 | Correlated |

Wide rigs are pixel-limited / correlated-noise dominated rather than seeing-limited at
~9.77 arcsec/px.

### 4.4 Reference class counts (local Gaia max G=17.5)

| Draft | artifact_neg | unmatched_in_range | ambiguous | beyond_cat | indeterminate |
|-------|-------------:|-------------------:|----------:|-----------:|--------------:|
| 501 | 142 | 26 | 310 | 203 | 15 |
| 435 | 8 | 81 | 17 | 0 | 3 |
| 500 | 48 | 455 | 49 | 0 | 9 |

`artifact_negative` on draft_501 = **142** (regression locked in `dev/tests/test_dao_reconcile.py`).

### 4.5 Confusion-blend test (DAO-CLOSE)

Hypothesis: wide-rig `unmatched_in_range` rows are blends of unresolved faint Gaia stars.

**Result: hypothesis not testable with the local Gaia build; closure verdict unchanged
(undecidable at detection stage).**

The tool (`dev/tools/dao_close_confusion_blend.py:56-78`, `_neighbor_stats`) counts every Gaia
source inside the search radius with no exclusion of the row's own counterpart. Control rows
are Gaia-matched by definition, so each control contains itself. That is why
`control_median_n_gaia_1x_fwhm = 1.0` while test rows read `0.0` -- the difference is the
self-match, not neighbour density. After subtracting it:

| draft | radius | test | control (raw) | control (self removed) |
|-------|--------|------|---------------|------------------------|
| 435 | 1x FWHM | 0 | 1 | 0 |
| 435 | 2x FWHM | 0 | 1 | 0 |
| 435 | 3x FWHM | 1 | 2 | 1 |
| 500 | 1x FWHM | 0 | 1 | 0 |
| 500 | 2x FWHM | 0 | 2 | 1 |
| 500 | 3x FWHM | 1 | 2 | 1 |

Corrected, the two populations are indistinguishable at 1x and 3x FWHM. There is no excess and
no meaningful deficit -- the measurement shows **no difference**, not a result in the direction
an earlier report described.

More fundamentally: the local Gaia database is right-censored at G = 17.5. A blend producing
implied G ~15 from components fainter than 17.5 is **invisible to the catalogue used to test for
it**. "No excess local Gaia density" is therefore the expected reading whether the hypothesis
is true or false.

| Draft | n in-range | verdict |
|-------|------------|---------|
| 501 | 26 | inconclusive (sample too small) |
| 435 | 81 | undecidable (catalogue-censored test) |
| 500 | 455 | undecidable (catalogue-censored test) |

Tool: `dev/tools/dao_close_confusion_blend.py`; results: `dev/results/dao_close_confusion_blend.json`.

A deeper all-sky catalogue would be required to test the blend hypothesis -- observation only,
not a queued task.

---

## 5. What was ruled out (and why)

| Filter / hypothesis | Outcome | Evidence |
|--------------------|---------|----------|
| Sigma-floor (`k_floor`) | No k removes ~471 excess without depth loss | DAO-PHYS-2 section 8 |
| Sharpness bounds | Weak; sign reverses between rigs | DAO-PHYS-2 |
| Split detections on bright stars | Median 10-16 px to nearest match; 5-17% within 5 px | A-6b split-detection tool |
| Hot pixels | DAO-PHYS-2 underpowered (cap 200); DAO-PHYS-2b corrected test is authoritative | DAO-PHYS-2b result |
| Confusion blends (faint unresolved Gaia sum) | Not testable (Gaia censored at G=17.5; control self-match artefact) | DAO-CLOSE confusion-blend test |
| Runtime `dao_only_fraction` gate | Removed (INV-MS-01) | DECISIONS INV-MS-01-REMOVAL |

### 5.1 DAO threshold parameters (closed question)

`masterstar_dao_threshold_sigma`, `sips_dao_threshold_sigma`, `qc_dao_detection_sigma` remain
group (a) rig-scoped at low confidence in the params registry, but **no detection-stage threshold
change was indicated** by the campaign. draft_501 high DAO_ONLY fraction was driven primarily by
**calibration mode** (additive pedestal on pre-calibrated frames) rather than Newton geometry vs
wide-rig threshold tuning.

**Reopen condition:** empirical two-rig sweep at matched calibration state (same master-flat
pedestal and noise model), not a single-draft census.

---

## 6. Residual risk

VYVAR no longer fails a run on catalogue inflation. Residual symptoms:

- LC quality (targets admitted via `snr50_ok` and comp selection)
- Informational `MASTERSTAR DAO_ONLY census` log line and PDF census row
- Per-class counts and unmeasurable fraction in `pipeline_meta`

Operators should treat wide-rig `unmatched_in_range` populations as **undecidable at detection
stage** on current evidence, not as confirmed uncatalogued sources or as pure artifacts.
The confusion-blend hypothesis is not testable with a catalogue censored at the depth it
concerns (`docs/VYVAR_DAO_DETECTION.md` section 4.5).

---

## 7. Key file index

| Topic | Location |
|-------|----------|
| Classification core | `src_py/dao_reconcile.py` |
| Pipeline wiring | `src_py/pipeline.py` (~12470-12540) |
| Census / PDF | `src_py/photometry_report.py`, `src_py/ui_masterstar_qa.py` |
| Unit tests | `dev/tests/test_dao_reconcile.py` |
| Offline tools | `dev/tools/a6_classify_offline.py`, `a6b_split_detection_measure.py`, `dao_close_confusion_blend.py` |
| Closure result | `dev/results/CURSOR_RESULT_dao_close.md` |
