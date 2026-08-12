CURSOR RESULT - 2026-08-12 (white cores v2)

What I did
Read-only pixel diagnostic of claimed white cores in draft 509 `detrend_aligned`, using draft 435 as differential control. No fixes.

## Output / findings

### Step 0 - Differential lever
Both 435 and 509 `detrend_aligned` Light_001: top bright stars are class **(a)** peaked cores (hollow=False). Anomaly as a low-value centre is **not present in either draft's data**.

### Step 1 - Data or display?
509 `detrend_aligned` Light_001:
- BITPIX=-32, dtype=float32, BZERO/BSCALE unset
- min=478.8 max=68566.8 med=2413.2
- NaN/Inf/neg: 0 / 0 / 0 / 0

Three brightest 11x11 patches: core = patch maximum (often a saturated plateau at 68566.8).  

**Classification: (a)** - highest in patch, peaked. **Display artifact / misread; data is fine. Investigation closes here.**

### Step 2 - Stage chain
Not required after (a). For the record: raw uint16 peaked at 65535; calibrated/aligned float32 peaked at ~68567. No stage introduces a hollow core.

### Step 3 - H1 / H2
Not pursued after (a). Notes that falsify the motivating hypotheses anyway:
- Sky median ? **2413**, not 2.2e8. Frame max/INT32_MAX ? 3e-5. **H2 overflow closed.**
- No negative cores. Saturated stars sit on a high plateau (~68567), which is the opposite of a punched low core.

### Step 4 - Blast radius
No anomalous (low) core pixels found. Brightest stars are at the ADU ceiling (raw 65535); target/comps may be saturated but cores are high, not hollow. MASTERSTAR float32, max peaked normally.

## Mechanism
The FITS arrays show bright, peaked (often saturated) cores. A white centre in an inverted viewer (white=LOW) does not match the pixel values. The reported white spot is a **viewer/stretch interpretation**, not a pipeline punch or integer wrap.

## Errors
None.

## Files changed
None.
