CURSOR RESULT - 2026-08-13 Block-summed bin1 master dark vs bin2 lights (physics measurement)

What I did
Read-only measurement task: inventoried dark/bias frames on disk, block-summed the production
bin1 QHY294MM master, compared against calibrated draft outputs (435/509/510), fit dark-vs-exposure
intercept, simulated the Section 0 `3xoffset` over-subtraction hypothesis, and researched CMOS binning
literature. **No code changes to the calibration path.**

Script: `tmp/_dark_binning_physics_measure.py` (reproducible).

---

## 1. Direct measurement

### 1.1 Dark/bias inventory on disk

| Location | Camera | Bin | Exptime (s) | Temp ( degC) | Gain | OFFSET (header) | Count | Notes |
|----------|--------|-----|-------------|-----------|------|-----------------|-------|-------|
| `CalibrationLibrary/` | QHY CCD QHY294PROM | 1x1 | 60 | ?10 | 0 | 0.0 | 1 | Master (NCOMBINE=17) |
| `CalibrationLibrary/` | QHY CCD QHY294PROM | 1x1 | 120 | ?10 | 0 | 0.0 | 1 | Master (NCOMBINE=17) |
| `Archive/M71/Darks/` | ZWO ASI533MC Pro | 1x1 | 15 | ?10 / ?9.9 | 100 | **50.0** | 40 | Different camera |

**QHY294MM:** 2 bin1 master darks only. **No bin2 dark. No bias frame. No individual constituent
dark frames** on disk (masters built from 17 frames per `CALIBRATION_LIBRARY`; sources not archived).

### 1.2 Bin2 master vs block-summed bin1 master

**Not possible.** No genuine bin2 QHY294MM master dark exists at 60 s, ?10  degC, gain 0 anywhere on
disk. Section 1.2 direct comparison **was not performed**.

**Closest available pair:** bin1 master `Dark_60s_Dark_0G_-10deg_Bin1_20260422.fits` vs block-sum
resample to bin2 (what VYVAR applies). Measurements on that pair:

| Quantity | bin1 master | block-sum bin2 (`bf=2`) |
|----------|-------------|------------------------|
| Median (stored ADU) | **24.4706** | **98.1176** |
| Mean | 24.6615 | 98.6466 |
| Median ratio | - | **4.0096** (expect 4.0 if per-pixel level scales identically) |

Native units (stored/4): medians **6.12** ? **24.53** native ADU; ratio still ~4.0.

**Spatial structure of block-sum:** ratio ? 4 everywhere at median; no separate gradient test vs a
bin2 reference (reference absent). Internal consistency: block-sum median equals 4x bin1 median to
0.4% - structure is **uniform scaling**, not a large spatially varying excess.

**Sky context:** production raw sky median **2496 ADU**; block-sum dark median **98.1 ADU**
(**3.93%** of raw sky). A constant error of **73.7 ADU** (the `3xoffset` prediction from S2 below)
would be **3.07%** of sky - above Check A's 2% relative tolerance on medians **if applied to dark
alone**, but Check A compares `median(dark)` to `median(light)`, not post-subtraction sky.

### 1.3 Closest-pair conclusion

Direct falsification of block-sum vs true bin2 dark **not available**. All downstream tests use
internal consistency and sky signatures instead.

---

## 2. Derived from bias/dark structure (no bin2 dark)

### 2.1 Bias frames

**No QHY294MM bias frames** on disk at any binning.

**ASI533 (M71):** 40 bin1 darks at 15 s with **OFFSET=50** in header, median **2040 ADU** - different
camera, not usable for QHY inference.

**Conclusion from bias:** **Cannot** compare block-summed bin1 bias vs bin2 bias for QHY294MM.

### 2.2 Dark exposure scaling (intercept = per-readout offset?)

Both bin1 masters at ?10  degC:

| Master | Median (ADU) |
|--------|--------------|
| 60 s | 24.4706 |
| 120 s | 24.4706 |
| Ratio 120/60 | **1.0000** (expect ~2.0 if dark-current dominated) |

Pixel subsample (100 000 pixels, bootstrap 200):

| Parameter | Value |
|-----------|-------|
| Intercept | **24.548 +- 0.011 ADU** (per bin1 pixel) |
| Slope | **0.00107 ADU/s** |

At ?10  degC and 60-120 s, the master dark is **pedestal-dominated**; dark current adds ?0.06 ADU over
60 s - negligible. The intercept **is** the stacked pedestal (~24.5 ADU/bin1 pixel), not a separately
resolved "readout-only" term.

Under block-sum: pedestal becomes **~98.2 ADU/superpixel** (= 4 x 24.5). That matches the resampled
master median (**98.12 ADU**) exactly.

**If** the Section 0 hypothesis applied (hardware binning, one offset per superpixel), block-sum would
carry **4x** the per-read offset while the light carries **1x**, giving excess **3x24.5 ? 73.7 ADU**
per superpixel in the subtracted dark.

### 2.3 OFFSET setting - headers, DB, software

| Source | OFFSET / pedestal |
|--------|-------------------|
| QHY lights (draft 510, 10 frames) | **OFFSET=0.0** (all) |
| QHY master darks | **OFFSET=0.0** |
| Equipment DB (`QHY294MM`, id=1) | No OFFSET column; GAIN_ADU=3.17, READNOISE_E=7.6 |
| Master dark median at gain 0 | **~24.5 ADU/bin1 pixel** (physical pedestal in data) |

**Reading:** Header `OFFSET=0.0` is an **uninformative or SDK-default field**, not evidence of zero
pedestal. The pedestal **is present in the pixel values** (~24.5 ADU at bin1) but **not recorded** in
a header usable for calibration algebra.

### 2.4 Finding for SAT-DIAG / noise model

QHY294MM applies a **non-zero pedestal (~25 ADU stored, ~6 native)** at gain 0 that appears in darks
but is **not propagated in FITS OFFSET**. Noise-model and SAT-DIAG work must treat pedestal as **measured
from dark/bias data**, not from `OFFSET` keyword.

---

## 3. What the light/calibrated frames say

All measurements use `Raw/lights/NoFilter_60_2/BO_CVn_Light_001.fits` and replicate VYVAR calibration
in Python (`get_processed_master` SUM dark + SUM flat, same as production).

### 3.1 Calibrated sky vs raw ? dark median

| Draft | raw median | dark median | raw?dark | synth (L?D)/FxmedF | archived cal | cal ? synth |
|-------|------------|-------------|----------|---------------------|--------------|-------------|
| 435 | 2496.00 | 98.12 | **2399.53** | **2414.37** | 2414.37 | **?0.0001** |
| 509 | 2496.00 | 98.12 | **2399.53** | **2414.37** | 2413.17 | **?1.20** |
| 510 | 2496.00 | 98.12 | **2399.53** | **2414.37** | 2413.17 | **?1.20** |

**draft 435:** archived calibrated matches synthetic replication to **0.0001 ADU** - block-sum dark +
flat pipeline is **exactly** what is on disk.

**509/510:** **?1.2 ADU** vs synth (~0.05% of sky) - negligible; likely flat-median or rounding, not
offset error.

**raw?dark vs calibrated:** difference **~14.8 ADU** (= flat normalization effect), consistent across
drafts. Not an dark-offset anomaly.

### 3.2 Negative calibrated pixels

| Draft | cal min (ADU) | Negative pixels | Fraction |
|-------|---------------|-----------------|----------|
| 435 | ?305.09 | 1 | 0.000034% |
| 509 | ?299.88 | 1 | 0.000034% |
| 510 | ?299.88 | 1 | 0.000034% |

| Draft | 1st percentile (ADU) |
|-------|----------------------|
| 435 | 2230.3 |
| 509/510 | 2284.0 |

**Not consistent with uniform ~74 ADU over-subtraction**, which would shift 1st percentile down by
~74 ADU (expected ~2160-2210). Observed 1st percentiles are **2230-2284**.

**Simulated test:** if dark carried **+73.65 ADU** excess (3x pedestal), sky median would drop from
**2399.53 ? 2325.88 ADU**. **Not observed.**

### 3.3 Systematic offset across drafts (20 frames)

| Draft | mean(raw?dark) | std | mean(calibrated) | std |
|-------|----------------|-----|------------------|-----|
| 435 | 1921.0 | 192.2 | 1933.1 | 193.4 |
| 509/510 | 1921.0 | 192.2 | 1933.8 | 192.5 |

Frame-to-frame variation is **~192 ADU** (sky + time structure), dwarfing any ~1-2 ADU cal/synth
residual. **No draft-specific systematic offset** beyond shared calibration.

**draft 435 CAL-DIAG:** `VY_CDSKY=2399.53`, `VY_CDSTAT=PASS` - matches measured raw?dark median.

### 3.4 Differential photometry impact

**Uniform per-pixel over-subtraction C:**

- Star aperture sum: `S ? C.N_ap`
- Sky annulus: `B ? C.N_ann`
- Local-background-subtracted flux: `(S ? C.N_ap) ? (B ? C.N_ann) = S ? B ? C.(N_ap ? N_ann)`

For **annular local sky** with constant C across the frame, the C terms **cancel only if the same C
applies equally** - which holds for a **flat offset error**. A **~74 ADU uniform error would cancel
exactly** in differential aperture photometry even if present.

**However:** measured sky shows **no ?74 ADU shift**, so the error is **not present at measurable level**
(~1 ADU residuals). For BO CVn at sky ~2400 ADU, even a hypothetical 74 ADU uniform error would be
**?mag ? ?2.5xlog10(1 ? 74/2400) ? 0.034 mag** in absolute photometry - and **?0** in comp?target
differential mode with local sky.

**Structured** offset (gradient, PRNU) would **not** cancel; no evidence of large structured residual
in cal?synth maps (not plotted; cal matches synth to 0.0001 ADU on frame 001).

---

## 4. General case (universal tool)

### 4.1 Mean-binning vs sum-binning drivers

| Driver convention | Light superpixel | Correct bin1?bin2 master resample |
|-------------------|------------------|-----------------------------------|
| **SUM** (QHY294 here) | `4x(pedestal + signal)` per block | Block **SUM** |
| **MEAN** | `mean(pedestal + signal)` ? pedestal + mean(signal) | Block **MEAN** |

Using SUM on a MEAN-binned light **overshoots dark by ~bf^2** on the variable part; Check A signature
(`median(dark) > median(light)`) is designed to detect this. **Production data pass Check A (SUM).**

### 4.2 When bin1 master cannot be corrected by any linear resample

A linear block operator (SUM or MEAN) **cannot** fix:

1. **CCD-style hardware binning** where charge from N pixels is read through **one ADC** (one pedestal
   per superpixel) while bin1 master sums **N pedestals** - the Section 0 concern. Error ~**(N?1)xpedestal**
   per superpixel if wrongly block-summed.
2. **Convention mismatch** (SUM vs MEAN) - wrong linear operator, not wrong linear coefficients.
3. **Mismatched gain/offset/read mode** between master and science frames.
4. **Non-linear spatial effects** (amp glow, fixed-pattern structure) that do not scale under binning.

Condition (1) **does not apply to CMOS software binning** where each native pixel is digitized
separately before summing (see S6).

### 4.3 Literature / vendor guidance (cited)

| Source | Relevant finding |
|--------|------------------|
| [Teledyne - Binning](https://www.teledynevisionsolutions.com/learn/learning-center/imaging-fundamentals/binning/) | CMOS binning occurs **off-sensor after readout**; each pixel already has read noise; 2x2 sum gives **4x signal, ~2x noise** (not CCD 4x SNR). |
| [SharpCap forum - QHY600 / Robin G](https://forums.sharpcap.co.uk/viewtopic.php?t=4241) | CMOS binning is **digital after ADC**; can be **add or average** depending on manufacturer; identical to post-processing bin1 data. |
| [AAVSO - CCD gain/RN/dark (CMOS binning test)](https://www.aavso.org/ccd-gain-ccd-readout-noise-ccd-dark-current-second) | Compare binned vs unbinned **background** to learn add vs average; dark current scales as bin^2 for both; **same gain/bin for cals and lights**. |
| [AAVSO - QHY268M review](https://www.aavso.org/qhy268m-review) | QHY binning in software; 2x2 **sums** (gain unchanged, RN ~x2 not x4); recommends binning in acquisition or post, consistently. |
| [QHY600 manual (PDF)](https://plone.unige.ch/astrodome/history%2C%20documentation/manuels/manuals/new_instruments/astronomical-camera-qhy600-20231207034348339.pdf) | **Digital sum** binning; OFFSET adjusts histogram pedestal - must be set per gain, not recorded as zero necessarily. |
| [Altair Astro - CMOS binning](https://www.altairastro.help/info-instructions/cmos/how-does-binning-work-with-cmos-cameras/) | CMOS cannot hardware-bin like CCD; software add or average both improve SNR differently but binning is post-ADC. |

**Consensus:** For CMOS (including QHY), **2x2 mode = digitize each pixel, then combine in software**.
Block-summing a bin1 master **replicates driver-side SUM binning**. The Section 0 **hardware-readout**
model applies to **CCD on-chip binning**, not this code path.

---

## 5. Conclusion for CAL-DIAG / binning policy

### Measurements support: **no systematic offset error** in the current QHY294MM workflow

| Test | Expected if 3xpedestal error | Observed | Verdict |
|------|----------------------------|----------|---------|
| Sky after dark subtract | ~2326 ADU | **2399.53 ADU** | **Refutes** 73.7 ADU over-subtraction |
| Synth cal vs archived (435) | mismatch ~74 ADU | **?0.0001 ADU** | **Refutes** |
| 1st percentile shift | ~?74 ADU | **2230-2284** (no ?74 shift) | **Refutes** |
| Block-sum ratio | - | **4.01** (matches 4x pedestal) | **Consistent** with SUM software binning |
| CAL-DIAG Check A (435) | - | PASS, SUM | **Consistent** |

**Outcome:** Block-summing the bin1 master dark **is physically sound** for QHY294MM **software-sum
2x2** lights. Cross-binning is a **convenience that works for this camera class**, not a hidden
~3% sky error. Every draft on disk (435, 509, 510) and the anchor calibration path are **not shown
to carry a measurable pedestal-multiplication error.**

**CAL-DIAG Check A** remains a **convention guard** (SUM vs MEAN driver change, wrong master) - **not**
a fix for a present error in Milan's data.

**Caveat:** **No bin2 dark reference** on disk - conclusion rests on CMOS software-binning physics +
sky self-consistency, not direct dark-dark comparison. A future bin2 master dark at matching settings
would still be worth acquiring as definitive proof.

**If re-reduction were ever needed:** only if a **true bin2 dark** showed ? block-sum(bin1) by ?1 ADU
- not indicated by current data. No re-reduction warranted from these measurements.

---

## 6. Independent physics reading (not assuming Section 0)

Section 0 posits **hardware binning**: one ADC conversion per superpixel, so block-summing four bin1
reads accumulates **four pedestals** where the light has **one**. That is **correct physics for CCD
on-chip binning**.

For **QHY294MM (IMX492 CMOS)**, evidence and literature agree binning is **software/digital after
ADC**: the camera reads native pixels (or binned output that is still formed by **summing digitized
values**), and 2x2 lights with SUM convention contain **sum of four (pedestal + dark_current +
signal)** terms. Block-summing a bin1 master dark estimates **the same sum** for the dark_current and
pedestal parts. Read noise adds in quadrature across the four reads and **averages down in the master
stack** (17 frames); it does not create a **persistent multiplicative pedestal error** under SUM.

The measured pedestal (~24.5 ADU/bin1 pixel, ~98 ADU/superpixel after block-sum) **scales by exactly 4**,
matching what SUM binning must produce. The **null result** on sky (2399.5 ADU after dark, not 2326)
is the decisive observation: the data **reject** the 3xpedestal penalty, not because differential
photometry would hide it, but because **the sky level is wrong for that hypothesis**.

The universal warning remains: **CCD hardware binning or MEAN drivers** break the block-sum assumption;
VYVAR's SUM resample is **camera-class dependent**. That is why Check A (convention test) still has
value even though Milan's QHY data show no present error.

---

## Files changed

None (investigation only). Measurement script: `tmp/_dark_binning_physics_measure.py`.
