CURSOR RESULT - 2026-08-13

What I did
Supersedes draft-510 white-cores v1. Revised detector; inverse-transform mapping via astroalign matrix; Light_014 write-path step counts; installed astroalign resampling inspection. Read-only.

---

## Part 1 - Detector vs observation

### 1.1 Milan pixel confirmation

**Not available** - no frame/x,y supplied. Proceeded with 1.2-1.3.

### 1.2 Visual check on old-detector cal hits

Five `calibrated` dips (old criterion): inverted-display proxy = centre minus ring median. All five show **negative** visual score (centre not brighter than ring) - **no visible white centre** on cal at those pixels. Example cal patch row-5 values ~**1700-1800 ADU** (smooth sky/galaxy background).

### 1.3 Revised criterion

Tighten to require **local peak ? 15000 ADU** in addition to the original ring-dip rules (?12% dip, ?80 ADU depth, bright ring).

| Stage | Old detector | Revised (v3) |
|-------|-------------|--------------|
| `calibrated` (134 frames) | 696 sources | **268** |
| `detrended_aligned` (135 frames) | 852 sources | **454** |

v3 reduces cal false positives but **268 cal hits remain** where Milan sees nothing - criterion still does not match visual 'white core on det only.' The visible effect is **det-specific** at bright plateaus; cal at mapped positions stays smooth (~1.7 kADU).

---

## Part 2 - Same sky position (inverse transform)

Method: `astroalign.find_transform` on up to 200 masterstar matches (same family as pipeline); map each det dip back with `SimilarityTransform.inverse` - **not** index equality, **not** WCS.

### 2.1 Transform - `BO_CVn_Light_014.fits`

| Parameter | Value |
|-----------|------:|
| Translation tx, ty | **+27.94, ?11.01** px |
| \|shift\| | **30.0 px** |
| Rotation | **0.82 deg** |
| Scale | **0.977** |

**Not identity** - index matching was invalid for all frames (max shift ?30 px).

### 2.2 Dip correspondence (v3 detector, Light_014)

| Metric | Count |
|--------|------:|
| Flagged sources in det | 8 |
| Mapped cal position also has dip (old detector, within 4 px) | **0** |
| Det-only | **8** |

**All eight** v3 dips on disk have **no dip** in `calibrated` at the inverse-mapped position.

### 2.3 Example patches (numbers)

**Example - det (212, 119):** centre **4923 ADU**; ring ~6900; local peak ~63k nearby.  
**Cal at inverse map (227, 96):** centre **~1777 ADU**; patch row-5 all **1750-1860** - smooth, no plateau, no dip.

(Full 11x11 arrays: `tmp/_white_cores_v3.json` ? `part2.examples`.)

---

## Part 3 - In-memory vs on-disk (Light_014)

### 3.1 Steps after `apply_transform`

| Step | `file:line` | Notes |
|------|-------------|-------|
| `astroalign.apply_transform` | `vyvar_alignment_frame.py:335` | bicubic warp |
| `_as_fits_float32_image` | via alignment return | dtype cast |
| `_maybe_refine_aligned` | `pipeline.py:14276-14280` | **no-op** (immediate return) |
| `fits.writeto` | `pipeline.py:14541` or `14910-14914` | persist |

### 3.2 Dip counts after each step (reproduction vs disk)

| Step | Old detector | v3 detector |
|------|-------------|-------------|
| After `apply_transform` (repro) | 23 | 4 |
| After `_as_fits_float32_image` | 23 | 4 |
| **On-disk det** | **26** | **8** |

**Count changes between in-memory repro and disk:** old **+3**, v3 **+4**. Float32 cast does **not** change counts.

### 3.3 Which write path / why mismatch

Draft 510 alignment report shows per-frame `astroalign` (not RAM identity). `_maybe_refine_aligned` does nothing. The remaining gap is explained by **transform mismatch**: reproduction used masterstar WCS points; production uses **DAO-detected stars** (`vyvar_alignment_frame.py:330-334`, capped control points). **Different transform ? different warp ? different dip count on disk.**

Cannot attribute the +3/+4 disk-only dips to a post-`apply_transform` pipeline step - **no such step exists** (refine is no-op).

### 3.4 Write-time dtype

Aligned array ? `float32` via `_as_fits_float32_image`; written with `fits.writeto(..., overwrite=True)`. No BZERO/BSCALE re-application on aligned products (already linear float).

---

## Part 4 - What astroalign can do

### 4.1 Resampling method (installed source)

`astroalign.py:441-449`: `skimage.transform.warp(..., **order=3**)` - **bicubic**; `mode="constant"`, `cval=median(source)`.

### 4.2 Can output fall below input sample min?

**Yes** for order=3. Bicubic is not constrained to the convex hull of neighbouring pixels; **undershoot/ringing** between high samples is expected.

### 4.3 Implication for Parts 2-3

Dips are **not** present in cal at mapped sky positions (Part 2: 0/8). They are **consistent with** bicubic undershoot on bright plateaus during warp. Part 3 does **not** close to disk pixel-for-pixel because production transform ? reproduction transform.

---

## Part 5 - Conclusion

**Previous chain (`apply_transform` ? white cores) is not fully closed.**

| Claim | Status |
|-------|--------|
| Dips appear in det at bright plateaus | **Yes** (v3: 454 sources) |
| Same sky position clean in cal | **Yes** (0/8 mapped cal dips on Light_014) |
| Bicubic can create dips | **Yes** (implementation) |
| Reproduced disk file dip count | **No** (23/4 in memory vs 26/8 on disk) |
| Post-warp pipeline step adds dips | **No evidence** (refine no-op) |

**Unaccounted step:** production **DAO-based transform** vs this task's **masterstar-based repro**. Closing the chain requires re-running alignment with the **exact** production control points for Light_014, then re-counting dips.

**No fix applied** (read-only).

## Files changed

None.

## Artifacts

- `tmp/_white_cores_v3.json`
- `tmp/_white_cores_v3.py`
