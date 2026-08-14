CURSOR RESULT - U-XVAL-COMP-RMS localization (2026-08-14)

Task register ID: U-XVAL-COMP-RMS
Draft: 000510, target BO CVn (Gaia `1498613634033133184`)
Method: S0 definitional parity only; **stopped at R0** (pre-registered rule).

Diagnostic script (reproducible): `tmp/u_xval_comp_rms_s0_verify.py`

---

## 1. Section 1 -- recorded state verification

All values checked against stored artifacts on 2026-08-14. Sources in parentheses.

| Claim | Recorded | Confirmed value | Source | Status |
|-------|----------|-----------------|--------|--------|
| Frame count | 134 | **134** | `photometry/photometry_summary.csv` (`n_frames`) | CONFIRMED |
| Comparison stars | 5 | **5** (`n_clean=5`) | `lightcurves/trust_1498613634033133184.json` | CONFIRMED |
| `aperture_px` | 4.261 | **4.261** | `photometry/photometry_summary.csv` | CONFIRMED |
| `check_scatter` | 0.008638 | **0.00863848089884789** | `lightcurves/trust_1498613634033133184.json` | CONFIRMED |
| Trust flag | GREEN | **GREEN** | same trust JSON | CONFIRMED |
| Target xval agreement ~3 mmag | ~3 mmag | **3.151 mmag fleet median**; BO CVn **0.216 mmag** | `tmp/xval_out_wave7/xval_results.csv` | CONFIRMED (fleet); BO CVn much tighter |
| VYVAR comp RMS vs photutils | 0.0101 vs 0.0078 | **0.010098 vs 0.007848** (fleet medians) | same xval CSV | CONFIRMED |

**Corrections to section 1 wording (not failures):**

1. **`check_scatter` is not the xval "VYVAR comp RMS".** Production `check_scatter` = `np.nanstd(kmag, ddof=1)` on the check-star sidecar (`trust_flag_core.py:84-97`, file `lightcurves/check_kmag_{target}.csv`). The xval pairing uses **`comp_rms_dao`** (LOO differential scatter on exported `dao_flux`), which for BO CVn is **0.011133 mag**, not 0.008638.

2. **The "~2 mmag comp gap" is the difference of fleet medians**, not the median per-target gap. Fleet `median(comp_rms_dao - comp_rms_phot)` = **0.750 mmag**; difference of medians = **2.250 mmag**. For BO CVn alone: **0.659 mmag** (`comp_rms_dao` 0.011133 vs `comp_rms_phot` 0.010474).

3. **Target agreement statement applies fleet-wide.** Median |`target_rms_phot` - `vyvar_lc_rms`| across 12 xval targets = 3.151 mmag. BO CVn target gap = 0.216 mmag. The pairing "target agrees, comp disagrees" holds qualitatively on BO CVn (target 0.22 mmag vs comp 0.66 mmag) but the recorded "~3 mmag" is not BO CVn-specific.

No section-1 number failed to reproduce. Proceeding to S0 per task order.

---

## S0 -- Definitional parity

### S0.A -- Quantities compared (xval harness)

The register item compares two columns from `tmp/xval_out_wave7/xval_results.csv`:

- **VYVAR side:** `comp_rms_dao`
- **Photutils side:** `comp_rms_phot`

Both are computed in `xval_run.py` (lines 214-216) using shared helpers in `xval_harness_core.py`.

**Shared LOO differential series** (`xval_harness_core.py:69-82`, `diff_series`):

For star `sid`, comparison set `comps`, flux pivot `w` (frames x source_id):

```
f_i       = w[sid, frame_i]                         (flux of focus star)
F_j,i     = w[c_j, frame_i]  for c_j in comps       (flux of each other comp)
good_i    = all(F_j,i finite and > 0)
ES_i      = sum_j F_j,i  over comps used as pivot
valid_i   = good_i AND f_i finite AND f_i > 0 AND ES_i > 0
md_i      = -2.5 * log10(f_i / ES_i)   when valid_i
out_i     = md_i - median(md over valid frames)     (demeaned LOO diff mag)
```

**Shared scatter estimator** (`xval_harness_core.py:24-35`, `sclip_std`):

```
x <- finite values of out_i
repeat up to 5 times:
    m = median(x); s = std(x)
    keep points with |x - m| <= 3*s
    if all kept: break
comp_scatter(sid) = std(x)   (population std, ddof=0)
```

**Comparison-star RMS (both sides)** (`xval_harness_core.py:85-93`, `comp_loo_median`):

```
comp_rms = median over c in comps of comp_scatter(c)
```

where for each `c`, `diff_series(w, c, comps \ {c})` is used (leave-one-out within the 5-star ensemble).

**Stars entering the metric:** comparison-star IDs from `comparison_stars_per_target.csv` for each target (BO CVn: 5 IDs). Target itself is excluded from comp_rms; it enters only target RMS columns.

**Frames entering the metric:** intersection of frames present in the flux pivot. BO CVn dao pivot: **134 frames x 6 sources** (5 comps + target) from proc CSVs.

### S0.B -- VYVAR flux input (`comp_rms_dao`)

**Not recomputed.** `comp_rms_dao` reads pre-exported **`dao_flux`** from proc CSVs (`xval_run.py:71-84`, `load_dao`).

Production `dao_flux` is annulus sky-subtracted circular aperture flux (`photometry_core.py:12471-12473`):

```
flux = aperture_sum(r_ap) - sky_adu_per_px * pi * r_ap^2
dao_flux = flux
```

Per-star **`aperture_r_px`** from SNR sizing table (BO CVn comps: 3.461-4.261 px on frame 001). Sky annulus (`photometry_core.py:12443-12444`, config `annulus_inner_fwhm=4.75`, `annulus_outer_fwhm=9.0`, `fwhm_px_for_aperture=3.3014`):

```
r_in  = max(r_ap + 0.5, annulus_inner_fwhm * FWHM)
r_out = max(r_in + 0.5, annulus_outer_fwhm * FWHM)
```

BO CVn comp stars (frame 001): **r_in = 15.682 px, r_out = 29.713 px** (all comps share r_in/r_out given shared FWHM).

Centroid: VYVAR export columns `x`, `y` from the production DAO/aperture pipeline (stored in proc CSV).

### S0.C -- Photutils flux input (`comp_rms_phot`)

Re-extracted independently from aligned FITS (`xval_run.py:169-186`):

```
(xc, yc) = centroid_sources(data, x0, y0, box_size=9, centroid_func=centroid_com)
         fallback to Gaia seed (x0,y0) if non-finite
sky      = median annulus (sigma_clip=None)
phot_flux = aperture_sum(CircularAperture(pos, r=3.0)) - sky * pi * 3.0^2
```

Seed positions `(x0,y0)`: Gaia DR3 cone query projected through master WCS (`xval_run.py:134-137`, saved in `tmp/xval_out_wave7/xval_sources.csv`).

Sky annulus geometry from master-frame FWHM estimate (`xval_run.py:139-142`):

```
FWHM_master = median M2 estimate on 40 bright Gaia stars
r_ap        = max(2*FWHM, 2.5)        (used for annulus only)
r_small     = 3.0                     (fixed photometry aperture)
r_in        = r_ap + 3
r_out       = r_ap + 8
```

Measured on draft 510 master: **FWHM = 2.931 px, r_in = 8.863 px, r_out = 13.863 px**.

Gain: neither side applies explicit gain conversion in the scatter metric; both operate on ADU counts.

### S0.D -- Production `check_scatter` (NOT in the xval pairing)

`check_scatter` = `np.nanstd(kmag, ddof=1)` over check-star sidecar epochs (`trust_flag_core.py:84-97`). Different star (check star `1497313255374892800`), different definition (raw check kmag std, not LOO comp ensemble). **Not comparable to `comp_rms_dao` or `comp_rms_phot`.**

Production comp QA (`comp_qa_core.py:76-97`, `loo_diff_series`) uses **magnitude-space** LOO (`flux = sum 10^{-0.4 m}`) rather than flux-space LOO. The xval harness does not call this path.

### S0.E -- Input configuration parity table

| Input | VYVAR (`dao_flux` -> `comp_rms_dao`) | Photutils xval (`comp_rms_phot`) | Identical? |
|-------|--------------------------------------|----------------------------------|------------|
| Aperture radius | Per-star SNR table: 3.461-4.261 px (target 4.261) | Fixed **3.0 px** all stars | **NO** |
| Sky annulus r_in | **15.682 px** (4.75 * FWHM) | **8.863 px** (r_ap+3) | **NO** |
| Sky annulus r_out | **29.713 px** (9.0 * FWHM) | **13.863 px** (r_ap+8) | **NO** |
| Centroid | Production export x,y | `centroid_com` recentroid from Gaia seed | **NO** |
| Position seed | Catalog export | Gaia WCS on master (offset 0.26-0.94 px vs export on frame 001) | **NO** |
| Frame list | 134 proc light frames | Same 134 aligned FITS | YES |
| Comp ensemble | 5 IDs from `comparison_stars_per_target.csv` | Same 5 IDs | YES |
| LOO + demean formula | `diff_series` flux-space | Same function | YES |
| Scatter estimator | `sclip_std` (3-sigma iterative clip) | Same | YES |
| Sigma clip on annulus | None in production export path | `sigma_clip=None` (`xval_run.py:181`) | YES |

**Post-flux function identity:** Given identical flux pivots `w`, both sides compute the same `comp_rms`. The pivots are **not** built from identical photometry configuration.

### S0.F -- External definitions (section 8 preview)

- **photutils** documents aperture photometry and background estimation separately; it does not define a standard "comparison-star ensemble RMS". Practitioners typically take std of calibrated differential mags (`np.std(..., ddof=1)` is common in tutorials; see SNU AO differential photometry notes).
- **VaST** reports multiple scatter indices (weighted sigma, clipped sigma, MAD, IQR, RoMS) on **per-object lightcurves** after SExtractor photometry (Sokolovsky & Lebedev 2017, MNRAS 464, 274S). No LOO ensemble metric is implied.
- **Honeycutt (1992, PASP 104, 435)** ensemble differential photometry combines comparison stars to build a reference; scatter is not uniquely standardized across implementations.
- **Production VYVAR Phase-1 `comp_rms`** (`comp_frame_normalize.py:153-179`, `robust_comp_rms`) = `1.4826 * median(|f_i - median(f)|)` on detrended relative flux -- yet another definition, not used in the xval columns.

The xval pair shares one post-flux definition but feeds it **different photometry inputs**.

---

## S1 -- Absolute sanity gate

**Not executed.** Pre-registered rule **R0** fired at end of S0 (see below). Task section 6: stop if R0 fires.

**Partial check (informational only, not a task step outcome):** Independent numpy recomputation of `comp_rms_dao` from proc `dao_flux` pivots (script above, no VYVAR/xval imports) yields **0.011133383995211426**, matching xval `comp_rms_dao` for BO CVn to machine precision. This confirms the harness applies its stated formula to stored `dao_flux`; it does **not** establish comparability with the photutils side.

Photutils per-frame flux tables are **not persisted** by `xval_run.py` (only aggregate `xval_results.csv`). Independent S1 recomputation of `comp_rms_phot` would require re-extracting from FITS or re-running the harness.

---

## S2 -- Statistical distinguishability

**Not executed** (R0 stop).

---

## S3 -- Per-star decomposition

**Not executed** (R0 stop).

---

## S4 -- Stage localization

**Not executed** (R0 stop).

A stage walk would require matched photometry inputs. With aperture radius differing by up to **42%** (3.0 vs 4.261 px) and sky annulus area differing by an order of magnitude, the first stage (aperture geometry) is already known mismatched before pixel-level comparison.

---

## Pre-registered rule fired

**R0 (apparatus)** -- quoted from task section 5:

> If S0 shows that the two quantities are not the same function of the same stars over the same frames, or if S0 shows any input configuration value differing between the two sides, then the comparison is uncontrolled and the recorded number pair does not describe photometry. In that case the recorded finding is retracted, the difference between the two definitions is reported, and the task ends there pending a re-issued controlled comparison.

**Application:** Stars and frames match; the post-flux function matches. **Five photometry inputs differ** (aperture radius, annulus r_in, annulus r_out, centroid method, position seed). Therefore the recorded pair **`comp_rms_dao` median 0.0101 vs `comp_rms_phot` median 0.0078 does not describe a controlled photometry comparison.**

**Retraction:** Register entry U-XVAL-COMP-RMS should be amended from "open unexplained photometry gap" to **"RETRACTED -- uncontrolled comparison (R0); input mismatch documented."**

---

## Localization

**None under task rules.** R0 prevents attributing the median gap to any pipeline stage. The enumerated chain from pixels to RMS was not walked (S4 not authorized after R0).

If a controlled comparison were re-issued, minimum input parity would require at minimum:

1. Photutils extraction at each star's exported `aperture_r_px` from proc CSVs.
2. Matched sky annulus (`r_in`, `r_out`) per star or per frame as exported.
3. Matched centroids (export x,y or identical recentroid policy).
4. Explicit decision on `sclip_std` vs plain std (iterative 3-sigma clip violates production zero-clipping policy but is shared by both current xval columns).

---

## What could not be measured

| Item | Reason |
|------|--------|
| S1 photutils independent recomputation | No stored per-frame photutils flux table; R0 stop |
| S2-S4 | R0 stop |
| Pixel-stage disagreement magnitude | Requires controlled inputs first |
| Whether matched-aperture xval would close gap | Out of scope (would be new harness run + new task) |

---

## Sources

1. VYVAR code: `src_py/xval_harness_core.py`, `src_py/xval_run.py`, `src_py/trust_flag_core.py`, `src_py/photometry_core.py`, `src_py/comp_frame_normalize.py`, `src_py/comp_qa_core.py`
2. Artifacts: `Archive/Drafts/draft_000510/...`, `tmp/xval_out_wave7/xval_results.csv`, `tmp/xval_out_wave7/xval_sources.csv`
3. photutils background/aperture docs: https://photutils.readthedocs.io/en/stable/user_guide/background.html
4. Sokolovsky & Lebedev 2017, VaST variability indices: MNRAS 464, 274S (arXiv:1702.07715)
5. Honeycutt 1992, ensemble differential photometry: PASP 104, 435
6. SNU AO Python differential photometry tutorial (std of zero-point samples): https://ysbach.github.io/SNU_AOpython/chaps/06-diffphot.html

---

## Files changed

- `dev/results/CURSOR_RESULT_U_XVAL_COMP_RMS_localization.md` (this memo)
- `tmp/u_xval_comp_rms_s0_verify.py` (diagnostic script)

No production code modified.
