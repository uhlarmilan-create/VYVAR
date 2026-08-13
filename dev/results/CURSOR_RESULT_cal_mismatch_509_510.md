CURSOR RESULT - 2026-08-13

What I did
Read-only investigation of P2 failure: why drafts 509/510 recalibrate 150/150 mismatched
(~123 ADU max on 510) while draft 435 is pixel-identical. Measured diff form,
reproducibility, provenance, and science impact. Root cause isolated by applying
the archived processing chain to fresh calibration.

---

## 1. Name the difference

### 1.1 Draft 510 frame 001 - diff character

| Quantity | Archive `calibrated/` | Fresh recal (current code) |
|----------|----------------------|----------------------------|
| Pixel `nanmedian` | **2413.169** | **2414.373** |
| `VY_QCBG` / manifest `qc.background` | **2414.373** | **2414.373** |
| Median diff (fresh ? arch) | - | **+1.204 ADU** |
| Max \|diff\| | - | **123.4 ADU** |
| Ratio med (fresh/arch) | - | **0.999454** (~0.05% scale) |
| `mad_diff` | - | **~25 ADU** |
| corr(diff, arch level) | - | **0.057** (weak) |

**Form:** Mostly a **slight global shift (~1.2 ADU)** plus **structured residual**
(max 123 ADU, ~5% of sky) consistent with a **low-order multiplicative/additive
sky model**, not a constant offset alone and not a pure flat-ratio pattern.

**Spatial confirmation:** Archive pixels equal fresh calibration **after**
order-2 sky-surface subtract (`_fit_subtract_preprocess_sky_surface`, p2p
**177.30 ADU**); max \|diff\| **0.0** on 150/150 frames.

### 1.2 Per-frame variation

150 frames: per-frame median diff (fresh ? arch) has **148 unique values** (not a
single constant offset). After applying sky surface to fresh cal: **150/150
pixel-identical**, max diff **0.0**.

### 1.3 Draft 509

509 and 510 archive pixels are **byte-identical** (max diff **0.0**); only QC
headers differ (`VY_QCHFR`, `VY_QCNS`). Same mismatch vs fresh recal as 510.
Same root cause.

### 1.4 Origin (physics form)

The residual is **not** dark-subtract, flat-divide, flat-normalization, or
calibration order/precision. It is **post-calibration order-2 sky-surface
subtraction** applied in-place to `calibrated/lights/` (`VY_SKYSF=True`,
`VYSKYORD=2`, `VYVARPR=True`). Headers `VY_QCBG` still record **pre-sky**
calibrate-time QC.

Pure calibration `(L?D)/F` with current masters matches draft **435** and fresh
code to **<0.001 ADU**; 509/510 archive differs only after the sky step.

---

## 2. What produced the archived frames

### 2.1 Provenance

| Draft | Cal FITS mtime (UTC) | Manifest `updated_utc` | Masters (manifest) |
|-------|----------------------|------------------------|---------------------|
| 435 | 2026-07-16 09:15 | 2026-08-11 10:47 | Same Dark_60s Bin1 + Flat_0.15s NoFilter Bin1 |
| 509 | 2026-08-12 10:18-10:19 | 2026-08-12 10:32 | Same |
| 510 | 2026-08-12 13:11-13:11 | 2026-08-13 15:49 | Same |

CalibrationLibrary masters rebuilt **2026-08-12 ~10:10 UTC** (before 509 cal).
Headers: `VY_MDP`, `VY_MFP`, `VY_FLATM=32975`, `VY_CFLAG=DF`.

**435 vs 509/510 on disk:**

| | 435 | 509/510 |
|---|-----|---------|
| `VY_SKYSF` | absent | **True** |
| `VYVARPR` | absent | **True** |
| Pixel vs pure cal | **match** | **match after sky subtract** |
| `VY_QCBG` vs pixels | stale (~1.05 ADU low vs pixels) | stale (~1.20 ADU high vs pixels) |

435 `VY_QCBG` (2413.32) vs pixel median (2414.37) is the normal sigma-clipped
sky QC vs frame median gap, not a second processing stage.

### 2.2 Code path (not a cal-formula regression)

Fresh `_calibrate_one_light_*` at HEAD, at `967f835`, and at `0ab686f` (Aug 12
10:15, 3 min before 509 disk write) all produce **2414.373** - not 2413.169.

The archive was **not** produced by a different `(L?D)/F` implementation. It was
produced by **`_qc_enrich_calibrated_in_place`** inside
`preprocess_calibrated_to_processed` (called from `night_run.py`), which:

1. Opens each file in `calibrated/lights/`
2. Applies order-2 sky surface when `preprocess_sky_surface_order=2`
3. Writes modified pixels + `VY_SKYSF` / `VYVARPR` back **in place**
4. Does **not** update `VY_QCBG`

Relevant commits (already on tree before Aug 12 session):

- `ff08002` - restore order-2 sky-surface on mono preprocess path
- `84174ae` - `VY_SKYSF` guard headers for in-place sky subtract

Placed-aperture work (2026-08-13) updated proc CSVs; **did not rewrite**
calibrated FITS (confirmed: cal mtimes unchanged, `VY_SKYSF` present from Aug 12).

### 2.3 What changed vs 435

Not calibration math. **435 never received in-place preprocess on `calibrated/`.**
509/510 went through night-run preprocess QC enrichment after import/calibrate.

---

## 3. Reproducibility

### 3.1 Double calibrate 510 (current tree, two processes)

Two full `calibrate_lights_to_calibrated` runs ? **150/150 pixel-identical**,
frame 001 SHA match. **Bit-reproducible** for pure calibration today.

### 3.2 Run-to-run variation sources

| Source | Measured |
|--------|----------|
| RAM vs disk cal path | **Identical** (max diff 0) in prior session |
| Double sequential cal | **Identical** (above) |
| MP cal (`VYVAR_CALIBRATE_MP`) | Not enabled in test; default sequential |
| RNG in cal path | None identified |
| Post-cal sky in-place | **Deterministic** given same input cal (150/150 match) |

### 3.3 Field practice (bit reproducibility)

Reduction pipelines generally **do not guarantee byte-identical products across
machines or library versions**; they guarantee **documented processing steps and
versioned recipes**:

- **DRAGONS** (Gemini): recipe/primitive sequences, AstroData tags, configurable
  but automated steps - reproducibility via pinned version + recipe, not
  cross-platform bitwise identity ([IOP 2024](https://iopscience.iop.org/article/10.3847/2515-5172/ad0044)).
- **ccdproc / NumPy**: float promotion and platform FP semantics can change
  low bits; practitioners cast to `float32` before write when stable file size
  and near-reproducibility matter ([astropy#9056](https://github.com/astropy/astropy/issues/9056)).
- **General FP**: IEEE-754 non-associativity makes bitwise identity require same
  hardware, compiler, and operation order ([Stack Overflow
  discussion](https://stackoverflow.com/questions/21212326/floating-point-arithmetic-and-reproducibility);
  [Stan reproducibility
  note](https://mc-stan.org/docs/reference-manual/reproducibility.html)).

**Tolerance:** For 16-bit ADU data, sub-ADU drift from FP is common; **~1 ADU
systematic from mismatched processing stages** (as here) is a **provenance bug**,
not acceptable 'FP noise.' Structured 100+ ADU residuals are **decidedly not**
rounding.

---

## 4. Science impact

### 4.1 Full photometry on copies

Full Phase 2A re-run on copied trees was **not executed** (scope: read-only,
avoid overwriting archive; full re-platesolve/align would be a separate campaign).

**Proxy measurements:**

- `detrended_aligned` frame 001 median **2413.169** - **identical** to
  sky-subtracted `calibrated/` (alignment did not remove sky signature).
- Fresh pure cal + sky ? **pixel-identical** to archive on all 150 frames.

### 4.2 Expected photometric direction

Sky-surface subtract removes a smooth bias (median **?1.20 ADU** on frame 001,
p2p **177 ADU** across field). **Local-background aperture photometry** partially
cancels uniform components; **low-order polynomial** components do **not** fully
cancel for comps at different field positions. A full LC re-run on pure-cal inputs
could shift check scatter at the **fewx10?? mag** level or below - not assessed
numerically here.

### 4.3 Photometric inertness?

**Not guaranteed inert** - sky surface is spatially structured. **Likely small**
for differential photometry with local sky annuli at BO CVn field scale, but
this was not re-quantified end-to-end.

### 4.4 Validated draft 510 result (0.008629)

**Produced from archived sky-subtracted `calibrated/` ? `detrended_aligned/` chain**,
not from freshly recalibrated pure `(L?D)/F` products.

Evidence:

- `pipeline_meta.json` stages start **2026-08-12T13:14** (after cal+preprocess)
- Archive cal carries `VY_SKYSF=True` / `VYVARPR=True`
- `dev/results/CURSOR_RESULT_placed_aperture.md`: check scatter **0.008629**,
  5 comps, 134 pts - photometry on existing aligned products

---

## 5. What should be true

### 5.1 Reproducibility guarantee

**Yes:** a draft's **stated product** should be reproducible from raw + recorded
config + **explicit processing stage**.

Today missing:

- **Stage stamp on calibrated products** (`VY_SKYSF` exists but P2 ignores it)
- **Content hash** of pixel data vs manifest QC fields
- **Separation of 'calibrated' vs 'preprocessed-calibrated'** trees (in-place
  mutation blurs the contract)

Field practice: versioned recipes (DRAGONS), pinned dependencies, header
provenance (LSST/IVOA obs core), product-type keywords.

### 5.2 Check that would have caught this (design only)

**`INV-CAL-02` (proposal):** After any write to `calibrated/lights/`:

1. Record `VY_CALSTAGE` ? `{PURE, SKYSF_N}` and SHA-256 of pixel array in
   `draft_manifest.json` / sidecar.
2. P2 gate: if archive has `VY_SKYSF`, recal must apply the same sky order
   before compare **OR** compare only files with matching `VY_CALSTAGE`.
3. Alert when `VY_QCBG` disagrees with `nanmedian(data)` by >2 ADU.

### 5.3 Which set is correct?

| Product | Status |
|---------|--------|
| **Pure `(L?D)/F`** (435, fresh recal) | **Correct calibration** per current pipeline spec |
| **509/510 `calibrated/` pixels** | **Correct preprocess-modified product** for the Aug 12 run, **mislabeled** as if QC header described current pixels |
| **Manifest QC 2414.373** | **Correct for pure cal**; **stale relative to sky-subtracted pixels** on 509/510 |

Not a 'wrong physics' fork - an **apples-to-oranges P2 comparison**.

---

## Summary chain (cause)

```
night_run / preprocess_calibrated_to_processed
  ? _qc_enrich_calibrated_in_place (order=2 sky)
  ? modifies calibrated/lights pixels in place (VY_SKYSF=True)
  ? leaves VY_QCBG from earlier calibrate_lights_to_calibrated
INV-CAL-01 P2 recalibrates pure (L?D)/F only
  ? compares to sky-subtracted archive
  ? 150/150 mismatch (max ~123 ADU); manifest QC matches fresh, pixels don't
435 never got in-place sky on calibrated/
  ? P2 PASS
```

---

## Decisions for Milan

| Question | Answer |
|----------|--------|
| **INV-CAL-01 affected?** | **P2 predicate is wrong for preprocess-modified cal products**, not the CAL-DIAG v2 gate itself. **Do not push** until P2 compares like-with-like (apply sky or exclude `VY_SKYSF` frames). P1/P3-P8 remain meaningful. |
| **Draft 510 validated result stands?** | **Yes** - it used the actual archived preprocess chain. P2 does not invalidate 0.008629. |
| **Archived 509/510 calibrated FITS** | **Keep for science reproducibility** of validated photometry. Optionally **add** pure-cal copies under a distinct stage path (`calibrated_pure/`) or restore pure cal from raw for future gates - **do not silently overwrite** existing files. |

---

## Files / artifacts

- `tmp/_repro510_a`, `tmp/_repro510_b` - double-cal outputs (bit-identical)
- `tmp/_cal_mismatch_investigate.py` - comparison harness (prior session)
- Measurements: this document

## Errors

None.

## Files changed

None (read-only investigation).
