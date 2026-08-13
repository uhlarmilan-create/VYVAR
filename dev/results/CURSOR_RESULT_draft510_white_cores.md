CURSOR RESULT - 2026-08-12

What I did
Whole-frame dip detector over all 135 `detrended_aligned` and 134 matched `calibrated` FITS in `NoFilter_60_2`; paired same-index and WCS-mapped comparisons; single-frame astroalign reproduction on `BO_CVn_Light_014.fits`; raw saturation census; runtime limit resolution; comp-star saturation counts at 16384 / 85%.

---

## Step 1 - Whole-frame detector

### Criterion (fixed before run)
A pixel is flagged when **all** hold:
1. `ring_median` = median of its 8-connected neighbours (3-3 excluding centre)
2. `ring_median > sky + 8 - bg_rms` (sky = 15th percentile; `bg_rms = MAD - 1.4826`)
3. `centre < ring_median - (1 ? 0.12)` (?12% dip)
4. `(ring_median ? centre) > max(80 ADU, 0.04 - ring_median)`

Connected components of flagged pixels = one affected source (?1 px). No brightness-rank filter.

### 1.1 Per-frame source count - `detrended_aligned` (135 frames, 846 sources total)

| Sources/frame | # frames |
|---------------|----------|
| 1 | 7 |
| 2 | 9 |
| 3 | 13 |
| 4 | 14 |
| 5 | 17 |
| 6 | 18 |
| 7 | 11 |
| 8 | 16 |
| 9 | 7 |
| 10 | 8 |
| 11 | 9 |
| 12 | 1 |
| 14 | 1 |
| 17 | 2 |
| 26 | 1 |

Median ? 5 sources/frame; range 1-26 (`BO_CVn_Light_014.fits` worst). `MASTERSTAR.fits`: 6.

### 1.2 Same detector on matched `calibrated` frames

**Not zero.** 519 sources over 134 matched frames:

| Sources/frame | # frames |
|---------------|----------|
| 1 | 11 |
| 2 | 27 |
| 3 | 31 |
| 4 | 25 |
| 5 | 20 |
| 6 | 10 |
| 7 | 5 |
| 8 | 1 |
| 9 | 1 |
| 11 | 1 |
| 14 | 1 |
| 24 | 1 |

Calibrated has fewer total sources (519 vs 846) and lower per-frame counts, but the detector is **not** clean on calibrated.

**Det-only worsening (same pixel index on the shared 1397-2082 grid):** of 846 det sources, **809** have no matching dip within 3 px in cal at the same index; **736** have det dip depth exceeding cal by **>80 ADU** at the same index. Milan-s visible effect aligns with dips that **appear or deepen after alignment**, not with cal being globally dip-free.

### 1.3 Five example 11-11 patches (same array index = post-alignment grid)

Patches centred on the flagged pixel; row 5 / col 5 = centre.

**Example A - `BO_CVn_Light_022.fits` pixel (row=321, col=1522)**

| | col 0-10 of row 5 |
|---|-------------------|
| **det** | 5831.7, 9827.9, **31580.8**, **62147.1**, **44678.1**, **7623.8**, 3813.7, 2593.6, 2121.5, 1901.5, 1789.3 |
| **cal** | 9481.4, 22241.6, **64552.8**, **64393.0**, **26784.8**, **6191.2**, 3257.5, 2387.5, 1988.4, 1955.4, 1774.0 |

**Example B - `BO_CVn_Light_004.fits` pixel (row=999, col=1377)**

| | row 5 |
|---|-------|
| **det** | 4738.8, 5670.4, 6587.9, 7872.7, 9031.6, **10874.2**, 14195.6, 11714.1, 7907.1, 6055.8, 4673.8 |
| **cal** | 2281.9, 2313.6, 2365.8, 2371.0, 2511.9, **2641.2**, 2878.4, 3004.9, 3201.3, 3444.8, 3722.5 |

**Example C - `BO_CVn_Light_014.fits` pixel (row=212, col=119)**

| | row 5 |
|---|-------|
| **det** | 2262.0, 3261.0, 7985.7, 11667.0, 7122.5, **4923.0**, 6842.4, 11614.0, 9361.4, 6175.0, 4607.0 |
| **cal** | 1777.8, 1836.2, 1754.8, 1780.9, 1858.0, **1777.2**, 1853.8, 1758.2, 1758.2, 1862.3, 1921.1 |

**Example D - `BO_CVn_Light_022.fits` pixel (row=805, col=1993)** - det centre 3259.1 ADU; cal same index 3259.1 neighbourhood smooth (~2200-4900), no ?12% dip in cal.

**Example E - `BO_CVn_Light_103.fits` pixel (row=1234, col=2007)** - det centre dip 1682 ADU on a ?60k plateau; cal same index smooth ~2100 ADU sky (no plateau).

(Full numeric arrays: `tmp/_draft510_fixed.json`.)

### 1.4 Anomalous value type

**All 846 det sources: `kind = finite`.** No NaN, Inf, exactly zero, or negative centres.

Representative centre values at flagged pixels:
- Ex. A: **7623.8 ADU** (ring ~10?159; neighbours up to **62?147**)
- Ex. B: **10?874.2 ADU** (ring ~12?859; local peak **63?348**)
- Ex. C: **4923.0 ADU** (ring ~6905)

The anomaly is a **finite positive ADU below a brighter immediate neighbourhood** on a saturated or near-saturated plateau - not an invalid pixel.

---

## Step 2 - Which stars, and why

### 2.1 Affected sources (sample; peaks = 7-7 max at reference-grid row/col)

| Frame | row | col | peak_cal | peak_det | catalog match |
|-------|-----|-----|----------|----------|---------------|
| Light_004 | 999 | 1377 | 2300 | 63?348 | Gaia 1497120565961994752 (5 px) |
| Light_004 | 1103 | 231 | 9140 | 62?686 | Gaia 1485619777414711680 |
| Light_004 | 947 | 1934 | 2229 | 32?228 | Gaia 1498906247391001088 |
| Light_022 | 321 | 1522 | 1772* | 68?969 | bright field star |
| Light_014 | 212 | 119 | 1777 | 4923 | field star |

\*WCS back-projection from det?cal mis lands off the plateau (cal WCS ? reference grid). Same-index cal peak at (321,1522) = **6191** with plateau to **64?553**.

Most affected sources sit on **?10? ADU plateaus** (often **>60?000 ADU** locally in det).

### 2.2 Unaffected bright sources (frame `BO_CVn_Light_001`, 5-5 local maxima above threshold, no dip within 3 px)

| row | col | peak_cal | peak_det |
|-----|-----|----------|----------|
| 0 | 443 | 2389 | 3298 |
| 3 | 527 | 2336 | 2764 |
| 4 | 237 | 2301 | 5675 |
| 6 | 443 | 2411 | 4354 |
| 7 | 326 | 2367 | 3141 |

Typical peaks **2300-5700 ADU** - bright but not saturated plateaus.

### 2.3 Separating property

**Peak ADU alone does not separate:** on `BO_CVn_Light_004`, comparison stars with a dip within 8 px have median peak **32?228 ADU** (7 stars) vs unaffected median **10?903 ADU**, but both groups reach **68?576 ADU** max.

**Operational separator:** a detector-flagged **?12% / ?80 ADU dip within ~8 px of the star-s 7-7 peak** on the **aligned (reference) grid**. These occur almost exclusively on **interpolated saturated plateaus** (local peak ? 15?000-68?000 ADU in det). Fainter bright stars (peak ? 6?000 ADU) appear in the unaffected set.

### 2.4 Frame-to-frame persistence

Clustering det dip positions in RA/Dec (4 px bins, ref WCS of Light_001):
- **~846 cluster positions**
- Frames per cluster: **min 1, max 94, median 1**
- **?100 frames:** 0 clusters; **5-99 frames:** ~occasional

**Same field locations recur, but most dips are frame-specific** (median 1 frame) ? property of the **alignment/interpolation operation per frame**, not a fixed bad catalog star.

---

## Step 3 - Which operation

### 3.1 Operations on the science array between `calibrated/` and `detrended_aligned/` (in order)

| # | Operation | Location |
|---|-----------|----------|
| 1 | Preprocess QC in-place: optional order-N sky-surface subtract | `pipeline.py:17528` (`preprocess_calibrated_to_processed`) ? `17343` (`_qc_enrich_calibrated_in_place`) ? `17255` (`_fit_subtract_preprocess_sky_surface`) |
| 2 | Write preprocessed data + headers in-place to `calibrated/lights/*.fits` | `pipeline.py:17276-17315` |
| 3 | Alignment job reads preprocessed `calibrated/lights` | `pipeline.py:1175` (`_archive_preprocess_lights_root`), `13972-13976` |
| 4 | Pick reference frame; load ref data | `pipeline.py:14035-14045` |
| 5 | Per frame: DAO star detect ? `astroalign.find_transform` | `vyvar_alignment_frame.py:330-334` |
| 6 | **`astroalign.apply_transform`** (bilinear resampling onto reference grid) | `vyvar_alignment_frame.py:335` |
| 7 | Optional WCS reproject path (not used for draft 510 science frames; alignment report = `astroalign`) | `vyvar_alignment_frame.py:517-528` |
| 8 | Black-frame / constant-frame guards | `vyvar_alignment_frame.py:564-593` |
| 9 | Copy reference WCS to aligned header; `_maybe_refine_aligned` | `vyvar_alignment_frame.py:767-772`; `pipeline.py:14525` |
| 10 | Write `detrended_aligned/lights/*.fits` | `pipeline.py:14541` (or `14910` RAM flush path) |

### 3.2 Which operation introduces the anomaly (evidence)

**Preprocess is excluded as primary cause** for the Milan-visible dips: at the **same pixel indices** where det flags a dip, calibrated at that index **does not** pass the detector (Examples A-C above).

**`astroalign.apply_transform` introduces/deepens the dips:**

Reproduction on `BO_CVn_Light_014.fits` (masterstars_full_match control points, same helper as pipeline):
- Cal input dips: **24**; after in-memory astroalign: **23**; on-disk det: **26**
- **20/26** on-disk dips are **>80 ADU deeper** than cal at the same index
- Worst case (row=212, col=119): det dip **1982 ADU**; cal same index **42 ADU** (not flagged)

Disk det matches reproduction: dips appear on the **aligned output grid** where cal on the same index is smooth or lacks the ?12% depression.

### 3.3 Exact write line

Anomalous values are **first created** at:

```335:335:src_py/vyvar_alignment_frame.py
        aligned_data, _ = astroalign.apply_transform(t, image_source, image_target)
```

Persisted to archive at:

```14541:14541:src_py/pipeline.py
            fits.writeto(out_fp, aligned_data, header=hdr_out, overwrite=True)
```

(RAM-batch path equivalent: `pipeline.py:14910-14914`.)

---

## Step 4 - Saturation limit (QHY294MM, draft 510)

### 4.1 Raw pixel ceiling (150 frames measured)

| Value (ADU) | Pixel count (top values) |
|-------------|--------------------------|
| **65535.0** | **13?024** |
| 64344.0 | 5 |
| 64780.0 | 4 |
| 65444.0 | 4 |
| 65268.0 | 4 |

**Maximum observed: 65535.0 ADU** (13?024 pixels at ceiling across the raw stack). Not inferred - measured.

### 4.2 Raw header keywords (first frame)

| Keyword | Value |
|---------|-------|
| BITPIX | 16 |
| BZERO | 32768 |
| BSCALE | 1 |
| XBINNING | 2 |
| YBINNING | 2 |
| ROWORDER | TOP-DOWN |
| GAIN | 0.0 |

No `XBAYBIN`/`YBAYBIN` in header; binning via `XBINNING`/`YBINNING` = 2.

### 4.3 Runtime saturation limit (draft 510)

| Quantity | Value |
|----------|-------|
| Effective limit | **16384.0 ADU** |
| Source | **`equipment_db`** |
| Resolution code | `pipeline.py:5285-5303` (`_effective_saturation_limit`) - header keywords absent ? `EQUIPMENTS.SATURATE_ADU` via `database.py:3127` |

### 4.4 Milan `EQUIPMENTS` change - current state

- `draft_manifest.json`: **`equipment_id`: 1** (QHY294MM)
- DB query: **`EQUIPMENTS.ID=1`, `SATURATE_ADU=16384.0`**
- **Not overridden** - reaches runtime resolution (previous -missing draft link- is **closed** for draft 510)

### 4.5 Saturation check at 16384 and 85% (13926.4 ADU) - `detrended_aligned`, 134 science frames

Using comparison-star / target coordinates on the aligned grid:

| Star | peak_max | peak_median | frames ? 16384 | frames ? 13926 |
|------|----------|-------------|----------------|----------------|
| COMP_0001 | 68?968 | 67?905 | **134/134** | **134/134** |
| COMP_0002 | 69?000 | 67?851 | **134/134** | **134/134** |
| COMP_0003 | 69?030 | 68?651 | **134/134** | **134/134** |
| COMP_0004 | 69?030 | 68?667 | **134/134** | **134/134** |
| COMP_0005 | 69?030 | 68?730 | **134/134** | **134/134** |
| ZTF J134621.53+390523.5 (target) | - | - | 0/134 | 0/134 |
| check (COMP_0001 coords) | 68?968 | 67?905 | **134/134** | **134/134** |

Bright comparison/check stars ** exceed both limits on every aligned frame** (peaks ~68k - consistent with uncorrected 16-bit/full-range scaling after calibration, while the configured limit is 14-bit binned). Target is faint at the listed coordinates on this frame set.

---

## Step 5 - Causal chain

```
calibrated/lights/*.fits  (preprocessed in-place: sky surface already applied)
    ? pipeline reads for alignment (pipeline.py:13972-14045)
astroalign.find_transform(source DAO ? reference catalog points)  (vyvar_alignment_frame.py:330-334)
    ?
astroalign.apply_transform(t, image_source, image_target)  (vyvar_alignment_frame.py:335)
    ? bilinear interpolation across saturated plateaus
    ? local centre pixel lands below ring neighbours (?12% / ?80 ADU dip)
    ?
fits.writeto(detrended_aligned/...)  (pipeline.py:14541)
    ? white/dark -cores- visible on SOME saturated stars in detrended_aligned only
```

**Chain closes** at `vyvar_alignment_frame.py:335` ? `pipeline.py:14541`. Preprocess (`pipeline.py:17255`) modifies calibrated before alignment but **same-index before/after comparison shows the anomalous dips are not present in calibrated at those pixels**; they are produced by the alignment resampling step.

---

## Errors (if any)

None fatal. WCS det?cal back-projection is unreliable for patch display (cal per-frame WCS vs reference-grid det); same-index comparison on the shared 1397-2082 array is the valid localisation for this draft.

## Files changed

None (read-only diagnostic).

## Artifacts

- `tmp/_draft510_white_cores.py`, `tmp/_draft510_white_cores.json`
- `tmp/_draft510_fixed.json`
- `tmp/_draft510_repro014.py`, `tmp/_draft510_repro014.json`
- `tmp/_draft510_sat_stars.py`, `tmp/_draft510_sat_stars.json`
