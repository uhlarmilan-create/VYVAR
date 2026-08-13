CURSOR RESULT - 2026-08-13

What I did
Reconciled two conflicting BO CVn peak measurements on draft 510 raw data; reproduced both methods; produced one authoritative table with per-frame self-check; restated saturation consequences using corrected numbers.

Artifact: `tmp/_sat_peak_reconcile.py` ? `tmp/_sat_peak_reconcile.json`

---

## Part A - Peak measurement reconciliation

### A.1 How each measurement located the star

| Report | Frames | Coordinate source | Transform | Box | Script / reference |
|--------|--------|-------------------|-----------|-----|-------------------|
| **Pre-push saturation** (`CURSOR_RESULT_prepush_5dd2a4d.md`) | **134** LC-matched | `ra_deg/dec_deg` ? raw FITS WCS (`astropy.wcs.WCS`) | Per-frame drift from BO CVn: `mag_guided(WCS_target) ? WCS_target` applied to every star | 7?7 peak after 45?45 mag-guided search (`half=22` search, `half=3` peak via `pipeline._box_peak_max_adu`) | `tmp/_prepush_sat_d_v3.py` (see prepush report ?D) |
| **Draft-510 saturation** (`CURSOR_RESULT_saturation_binning_draft510.md`) | **150** all raw | Fixed `(x,y)` from `variable_targets.csv` | **None** | 7?7 at fixed pixel (`half=3`) | Ad-hoc inline Python (session); reproduced exactly below |

Pipeline reference for box peak: `pipeline.py:5327-5341` (`_box_peak_max_adu`, default `half=3`).

### A.2 Which measurement lands on the star

**Pre-push / WCS method lands on the star. Fixed-xy method does not** (except occasionally when the wrong pixel crosses the PSF by chance).

Sample frames (draft 510 raw, BO CVn):

| Frame | Fixed xy (wrong) | Peak wrong | Centre ADU wrong | Auth xy (WCS+drift) | Peak auth | Self-check |
|-------|------------------|------------|------------------|---------------------|-----------|------------|
| Light_001 | (957.66, 822.44) | 6660 | **2680** (sky) | (963, 828) | **21360** | pass (ratio 8.1? ring) |
| Light_055 | (957.66, 822.44) | 2368 | **1836** (sky) | (963, 828) | **15908** | pass (ratio 8.8?) |
| Light_109 | (957.66, 822.44) | 17812 | 17812 | (959, 825) | **17812** | pass (coincidence) |
| Light_148 | (957.66, 822.44) | 18688 | **6644** (offset) | (958, 825) | **18688** | pass |

Wrong method centre ADU ? 1800-2700 (background). Auth method centre ADU ? 16000-21000 with 3?3 local maximum and ?1.8? ring contrast on **134/134** target frames.

**Reproduction of draft-510 wrong numbers** (fixed `variable_targets.csv` xy, 150 raw, `half=3`):

| n | min | median | max | ?16384 |
|---|-----|--------|-----|--------|
| 150 | 2056 | **3662** | 24168 | **61** |

Matches the contradictory report exactly. Max agrees because a few frames happen to align; median is sky level (~3600 ADU).

### A.3 The 16-frame difference

| Count | Set |
|------:|-----|
| 150 | All raw `BO_CVn_Light_*.fits` in draft 510 |
| 134 | Frames in draft 509 BO CVn LC (`lightcurve_1498613634033133184.csv` ? `proc_*` stems) |
| **16 excluded** | `002, 007, 009, 049, 056, 058, 066, 074, 111, 122, 131, 141, 142, 147, 149, 150` |

Same directory, same draft. Excluded frames failed alignment/photometry QC in draft 509 - not a different path or draft.

### A.4 Authoritative peak table (draft 510 raw, 134 LC frames)

**Method:** Pre-push v3 (WCS ? per-frame target drift ? mag-guided local peak ? `_box_peak_max_adu` `half=3`). Stars from draft **509** BO CVn photometry set (draft 510 has no photometry yet; same field, same raw data). Script: `tmp/_sat_peak_reconcile.py`.

**Self-check (every star, every frame):** argmax in 7?7 box; 3?3 local maximum; peak/ring(11-15 px) ? 1.8; peak ? 4000 ADU. **Would reject all 134 wrong-method target frames** (centre ADU ? sky, ratio ? 1.0-1.5).

| Star | Role | catalog_id | n | min | median | max | ?16384 | ?13926 | ?65535 | Self-check fail |
|------|------|------------|---|-----|--------|-----|--------|--------|--------|-----------------|
| BO CVn | target | 1498613634033133184 | 134 | 12352 | **17492** | 24168 | **94** | **126** | 0 | 0/134 |
| comp | | 1497771992240531712 | 134 | 13572 | 18544 | 24124 | 110 | 133 | 0 | 0/134 |
| comp | | 1499200223486564608 | 134 | 12524 | 16596 | 21008 | 74 | 126 | 0 | 0/134 |
| comp | | 1497974027502858240 | 134 | 4728 | 5766 | 7724 | 0 | 0 | 0 | 15/134 |
| comp | | 1499053747922698240 | 134 | 5376 | 6796 | 8052 | 0 | 0 | 0 | 0/134 |
| comp | | 1497368849430107904 | 134 | 5856 | 6924 | 8356 | 0 | 0 | 0 | 0/134 |
| check | | 1497313255374892800 | 134 | 8232 | 11132 | 13800 | 0 | 0 | 0 | 0/134 |

**Why this is correct:** Matches pre-push target row exactly (134 / 12352 / 17492 / 24168). Self-check catches fixed-xy mis-centreing. Uses the same 134 frames photometry will use.

### A.5 Restated threshold counts (authoritative table only)

BO CVn target: **94** frames ?16384 (not 61); **126** ?13926 (85%); **0** ?65535.

No star in the photometry set reaches raw saturation at any threshold.

---

## Part B - Limit convention (after Part A closed)

### B.1 Histogram - 4?14-bit sum model refuted

Measured draft 510 raw (150 frames):

| Claim | Number |
|-------|--------|
| Pile-up at **65535** | **13024** pixels |
| At **65532** (= 4?16383) | **1** pixel |
| 4096 multiples | 975 ? 147 ? 62 ? 22 (monotonic decay, not ladder) |
| Saturated core >60000 | **9/9 at exactly 65535** |

If 2?2 binning summed four 14-bit saturated wells, the plateau would sit near **65532**. It sits at **65535** with a single stray at 65532. **Option B2 as ?sum of four 14-bit wells? is not supported.**

### B.2 What the camera delivers in this mode

| Evidence | Reading |
|----------|---------|
| `BITPIX=16`, `BZERO=32768`, `BSCALE=1` | Unsigned 16-bit integer ADU in FITS |
| `XBINNING=2`, `YBINNING=2` | 2?2 hardware binning flagged |
| Ceiling at **65535** not 65532 | **16-bit container ceiling**, not a clean 4?16384 sum |
| `GAIN=0.0`, `OFFSET=0.0`, `READMODE=0.0` | Driver did not record gain/readmode - cannot infer sum-vs-mean from headers alone |

**Evidence does not fully decide** sum vs mean vs vendor rescaling, but it **does** decide the operative ceiling is **65535 ADU in FITS pixel units**. Any limit must be expressed in those units.

### B.3 Single scalar `SATURATE_ADU` vs mode/binning - break points

All compare FITS peak ADU directly with **no binning scale** (`pipeline.py:5285-5324`, `5491-5525`, `comp_selection_per_target.py:798-837`). Break when draft binning ? equipment assumption:

| Location | Break mode |
|----------|------------|
| `database.py:3127` | One scalar per equipment |
| `pipeline.py:5285-5324` | No `XBINNING?YBINNING` multiply |
| `comp_selection_per_target.py:798-837` | Static `peak_max_adu` vs limit |
| `photometry_core.py:6127-6304` | Per-frame peak vs limit |
| `psf_runner.py:453,972` | PSF comp skip |

Wrong binning in header ? limit wrong by factor of 4 with no warning.

### B.4 Options

| Option | Stored | Runtime | Existing drafts | Bad binning header |
|--------|--------|---------|-----------------|-------------------|
| **B1 - Binned FITS ceiling** | `SATURATE_ADU ? 65535` per equipment+mode | Direct compare | Retag zones; comps recover | Low risk if headers match data |
| **B2 - Unbinned + scale** | 16384 per sensor pixel | Multiply by binning at compare | Auto-scale if headers correct | **High** - data refute sum model; missing/wrong binning silent |
| **B3 - Derive per draft** | Optional override | Max ADU or BITPIX from frames | Self-describing per draft | Medium - raw vs calibrated BITPIX differ |
| **B4 - Per equipment+mode+binning row** | Table keyed by `(equipment, readmode, XBIN, YBIN)` | Lookup from headers | Migrate scalar | Needs complete header set |

### B.5 Supported option

**B1 or B3/B4.** Measurements support storing the ceiling in **FITS ADU units (~65535)**. B2 would require the sum-of-wells ladder the histogram does not show. B2 would only hold if binning were mean-not-sum **and** the ceiling were 16384?4 with pile-up at 65532 - observed pile-up is at **65535**.

---

## Part C - Consequences (Part A table only)

Admission gate = `peak_max_adu ? limit ? 0.70/0.85` on static MASTERSTAR catalog (`comparison_stars.csv`, 140 comps).

### C.1 Comparison-star survival (140 comps, draft 510 catalog)

| Limit | Admission threshold | Survive | Fail |
|-------|--------------------:|--------:|-----:|
| **16384** | 13493 | **78** | 62 |
| **13926** (85%) | 11469 | 71 | 69 |
| **65535** | 53970 | **124** | 16 |

### C.2 Draft 509 BO CVn comps excluded by static admission

| catalog_id | peak_max_adu (catalog) | Excluded @16384 | Excluded @65535 |
|------------|------------------------:|:---------------:|:-----------------:|
| 1497771992240531712 | 17150 | **yes** | no |
| 1499200223486564608 | 16639 | **yes** | no |
| 1497974027502858240 | 5817 | no | no |
| 1499053747922698240 | 6293 | no | no |
| 1497368849430107904 | 6236 | no | no |

At **16384**, **2 of 5** comps used in draft 509 would fail static admission. At **65535**, **0 of 5**.

### C.3 BO CVn target flagged?

| Limit | Zone / static flag | Per-frame raw (134 LC frames) |
|-------|-------------------|-------------------------------|
| 16384 | `likely_saturated` on many frames (94/134 peaks ? limit; 126/134 ? 85%) | Not saturated (max 24168 ? 65535) |
| 65535 | No | No |

With current code, 16384 mis-tags the target as hot on most frames even though it never approaches the true ceiling.

### C.4 Production run

**Do not start** before the limit decision. With limit stuck at 16384:

- Wrong saturation tagging on BO CVn (94 frames) and bright comps
- 2/5 known-good comps fail static admission; pool thinned 140?78
- Photometry can run, but saturation metadata, comp admission, and QC are systematically wrong

---

## Part D - Summary

**Part A closed:** The 3662/61-frame numbers came from **fixed master-grid xy on 150 raw frames without WCS** - blank sky most of the time. Authoritative: **17492 median, 94 frames ?16384** on **134 LC frames** with WCS+drift+local peak. Self-check rejects the wrong method on every frame.

**DECISION REQUIRED:** (B1) store ~65535 binned FITS ADU; (B2) store unbinned 16384 and scale by binning - **refuted by histogram**; (B3/B4) derive/store per draft or per mode+binning - **supported by 65535 pile-up and header evidence**.

**Production run:** **No** - output saturation flags and comp admission will be wrong until the limit convention is fixed.

## Files changed

None (read-only).
