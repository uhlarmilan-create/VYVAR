# VYVAR ù CAL-STAGE / INV-CAL-02: Calibrated product stage integrity (spec)

Status: **DESIGN ù investigation complete 2026-08-13; awaiting Milan authorization.**
Date: 2026-08-13.
Grounding: INV-CAL-01 P2 investigation (`dev/results/CURSOR_RESULT_cal_mismatch_509_510.md`),
`dev/results/specs/VYVAR_CAL_DIAG_V2_SPEC.md` ù12.2,
`dev/results/CURSOR_RESULT_skysf_double.md`,
P-10 sky-surface sign fix (`docs/VYVAR_DECISIONS.md` ùP-10-SKYSURF-SIGN).

**Governing principle:** a directory name must not be the only record of what pixels
mean. Any gate, harness, or investigator that compares or re-derives calibrated frames
must know the **processing stage** before interpreting pixels or QC headers.

**What this spec fixes (class of defect):**

- **P-10 (2026-07):** in-place sky-surface sign error on data presenting as calibrated;
  reader could not tell stage from path alone.
- **P2 / INV-CAL-01 (2026-08):** fresh pure `(L?D)/F` compared to sky-subtracted archive;
  150/150 ùmismatchù was apples-to-oranges, not a calibration regression.

INV-CAL-01 stage-aware P2 (`cal_diag.apply_calibrated_stage_for_compare`) is a **compare
helper**, not a product-integrity invariant. INV-CAL-02 closes the provenance gap at write
time and at read/verify time.

---

## 1. Purpose

`Archive/Drafts/<draft>/calibrated/lights/` currently holds a **mutable, multi-stage**
product:

| Stage token | Pixel content | Typical markers today |
|-------------|---------------|------------------------|
| `PURE` | `(L?D)/F` only | no `VY_SKYSF`; `VY_QCBG` from calibrate QC |
| `SKYSF_N` | `(L?D)/F` minus order-N sky surface | `VY_SKYSF=True`, `VYSKYORD=N`, `VYVARPR=True` |

`VY_QCBG` and manifest `qc.background` describe **calibrate-time** edge-sampled sky QC.
After in-place preprocess they can disagree with the frame median by ~1.2 ADU (509/510) or
more when calibrate QC used a different estimator domain (435: 148/150 frames >1 ADU delta,
**without** sky subtract ù not a stage bug, but confounds naive header-vs-median checks).

INV-CAL-02 registers:

1. **What stage** is on disk (header + manifest, written by the step that mutates pixels).
2. **How to verify** stage against pixels (content hash + optional QC coherence).
3. **What gates do** when stage is unknown, inconsistent, or stale.

Registered as **`INV-CAL-02`** (invariant, not config). Zero new configuration keys.

---

## 2. Scope / non-goals

### In scope

- Stage stamp and verification for **`calibrated/lights/**/*.fits`** (mono and OSC channel dirs).
- Draft manifest `files[]` fields synced in the **same operation** as the pixel write.
- Read-path resolver used by diagnostic gates and documented for investigators.
- Legacy draft inference rules with honest `INDETERMINATE_*` outcomes.
- QC coherence alert when calibrate-time `VY_QCBG` disagrees with post-stage sky metric
  beyond a fixed tolerance **and** stage is `SKYSF_*`.

### Out of scope

- Re-splitting historical `processed/lights/` copy trees (draft 435 family); read-only legacy.
- Changing alignment, photometry, or MASTERSTAR algorithms.
- Replacing `VY_SKYSF` / `VYSKYORD` (retained; `VY_CALSTAGE` supersedes them as authoritative).
- Implementation code, UI, PARAMS registry (follow-on task after Milan authorization).
- Immutable `calibrated/` directory layout (Option B) ù documented as alternative in ù8.

---

## 3. Stage model (normative)

### 3.1 Stage tokens

`VY_CALSTAGE` is a single string keyword on the primary HDU:

| Value | Meaning |
|-------|---------|
| `PURE` | Calibrate (or passthrough) complete; no preprocess sky surface on these pixels |
| `SKYSF_0` | Reserved ù treat as `PURE` (order 0 = no subtract) |
| `SKYSF_1` ù `SKYSF_9` | Calibrate + in-place order-N sky surface subtract |
| `PASSTHROUGH` | Copied from import without `(L?D)/F`; mutually exclusive with `SKYSF_*` |

Stages compose in fixed order only: **`PURE|PASSTHROUGH ? SKYSF_N`**. No second mutable
stage on `calibrated/` (alignment/detrending writes **new paths** under `detrended_aligned/`).

### 3.2 Who writes what (must match pixel mutation)

| Step | Function | Sets `VY_CALSTAGE` | Sets content hash |
|------|----------|-------------------|-------------------|
| Calibrate / passthrough | `_calibrate_one_light_disk`, `_passthrough_lights_to_calibrated` | `PURE` or `PASSTHROUGH` | yes |
| Preprocess in-place QC | `_qc_enrich_one_frame` (sky apply branch) | `SKYSF_N` | yes |
| Preprocess skip (guard) | `_qc_enrich_one_frame` (skip branch) | unchanged | no pixel change |

**Constraint 4.5:** hash and stage keywords are updated in the **same FITS flush** as
`hdul[0].data` assignment. Manifest sync runs in the same function before return (existing
calibrate DB sync pattern).

### 3.3 Content hash

Use FITS **DATASUM** convention (32-bit 1's-complement checksum of data records;
[FITS Checksum Keyword Convention](https://fits.gsfc.nasa.gov/registry/checksum.html)):

- Keyword: **`VY_CALDATASUM`** (string form per FITS standard).
- Computed with `astropy.io.fits` `checksum="datasum"` on write/update.
- Stored in manifest `files[]` as `cal_datasum` (string).

Rationale: registered FITS practice for detecting silent pixel mutation; no custom hash.

### 3.4 Post-stage sky QC (SKYSF stages only)

When preprocess applies sky surface, write **`VY_PSTBG`** = full-frame sigma-clipped median
after subtract (same estimator family as `VY_QCBG` at calibrate time).

- **`VY_QCBG`** remains the calibrate-time QC value (do not rewrite ù preserves audit trail).
- Coherence check compares `VY_PSTBG` to live median, not `VY_QCBG` to live median.

---

## 4. Manifest contract

For each light FITS under `calibrated/lights/`, manifest `files[]` (keyed by raw light path)
gains or updates on every stage-changing write:

| Field | Type | Writer |
|-------|------|--------|
| `cal_stage` | string | calibrate / preprocess |
| `cal_datasum` | string | calibrate / preprocess |
| `cal_stage_ut` | ISO UTC timestamp | same operation |
| `cal_pstbg` | float, optional | preprocess when `SKYSF_*` |

Draft-level summary block in `cal_stage.json` (new, beside `cal_diag.json`):

```json
{
  "schema": "vyvar_cal_stage_v1",
  "draft_id": 510,
  "frames_total": 150,
  "stages": {"SKYSF_2": 150},
  "legacy_inferred": 0,
  "verify_last": {"ut": "...", "pass": 150, "fail": 0, "indeterminate": 0}
}
```

---

## 5. Read resolver (normative)

All production compare/gate code must obtain stage via:

```python
resolve_calibrated_stage(hdr, manifest_row) -> CalStageResolution
```

Resolution order:

1. **`VY_CALSTAGE` present** ? authoritative.
2. **Else `manifest_row.cal_stage`** if present and `cal_datasum` verifies against pixels.
3. **Else legacy inference:**
   - `VY_SKYSF=True` + `VYSKYORD=N` ? `SKYSF_N`, confidence `LEGACY_INFERRED`.
   - `VY_SKYSF` absent + `VY_CALIB=PASSTHROUGH` ? `PASSTHROUGH`, `LEGACY_INFERRED`.
   - `VY_SKYSF` absent + no passthrough ? `PURE`, `LEGACY_INFERRED`.
   - `VYSKYP2P` without `VY_SKYSF` (435 `processed/` era) ? `INDETERMINATE_LEGACY` ù do not assume pure.
   - Otherwise ? `INDETERMINATE_UNKNOWN`.

**Never assume `PURE` when confidence is not `AUTHORITATIVE` or `LEGACY_INFERRED` with verified hash.**

Existing helper `calibrated_stage_from_header` remains; resolver wraps it with manifest +
hash + indeterminate classes.

---

## 6. Verification gate (`INV-CAL-02`)

### 6.1 When it runs

- **On write:** self-check after flush (stage + DATASUM re-read; hard error in worker if mismatch).
- **On demand:** `verify_calibrated_stage(draft_id)` ù used by session baseline / pre-push harness.
- **Before pixel compare gates:** P2 and any future calibrated A/B must call resolver first.

### 6.2 Per-frame outcomes

| Outcome | Condition | Gate action |
|---------|-----------|-------------|
| `PASS` | Stage authoritative; `VY_CALDATASUM` matches pixels; manifest agrees | continue |
| `WARN_COHERENCE` | `SKYSF_*` and `|median - VY_PSTBG| > 2 ADU` | log + milestone; do not abort science path |
| `FAIL_STAMP` | Stage/hash manifest disagreement | **ABORT** obs_group (fail-closed) |
| `FAIL_CORRUPT` | DATASUM mismatch | **ABORT** obs_group |
| `INDETERMINATE_LEGACY` | Old draft; inference ambiguous | **WARN**; compare gates must not assume pure |
| `INDETERMINATE_UNKNOWN` | No stage evidence | **WARN**; compare gates **refuse** pure-vs-archive compare |

Constants (spec-only, not config): coherence tolerance **2.0 ADU** (matches P2 sky-subtract delta order).

### 6.3 What the gate does **not** cover

- Correctness of sky-surface **sign** or polynomial fit (P-10 class ù separate algorithm tests).
- Whether alignment or photometry should consume `PURE` vs `SKYSF_*` (downstream policy).
- Legacy `processed/lights/proc_*.fits` trees (outside `calibrated/` contract).
- Re-derivation of `(L?D)/F` (INV-CAL-01).

---

## 7. Legacy drafts (435, 509, 510)

| Draft | On-disk today | INV-CAL-02 read behavior | Migration |
|-------|---------------|--------------------------|-----------|
| **435** | `calibrated/` pure (150/150 no `VY_SKYSF`); separate `processed/` copy-tree with sky | `PURE` via `LEGACY_INFERRED`; `processed/` out of scope | **None** ù no overwrite |
| **509** | `calibrated/` in-place `SKYSF_2` (150/150) | `SKYSF_2` via `LEGACY_INFERRED` from `VY_SKYSF` | **None** |
| **510** | same as 509 | same | **None** |

Optional **read-only backfill** (separate authorized task): add `cal_stage.json` summarizing
inferred stages without rewriting FITS. Not required for validated science to remain usable.

---

## 8. Design options (for Milan)

### Option A ù Stamp and verify, keep in-place mutation **(recommended)**

- Add `VY_CALSTAGE`, `VY_CALDATASUM`, `VY_PSTBG`; manifest sync; verify gate.
- **Storage:** ~0 extra (one DATASUM string + stage keyword per file).
- **I/O:** one extra DATASUM compute per write (~negligible vs existing FITS rewrite); verify pass reads 150 headers + optional datasum check (~1.7 GB read per full draft verify).
- **Migration:** none for 435/509/510; legacy inference.
- **Would have caught:** P2 immediately (stage mismatch before pixel loop); P-10 investigation faster (explicit stage + hash proves what changed).
- **Would not catch:** wrong sky **formula** if stage stamp still says `SKYSF_2` and hash matches wrong pixels.

### Option B ù Separate products; immutable `calibrated/`

- `calibrated/lights/` = pure only; new `preprocessed/lights/` or restore `processed/lights/` copy tree.
- **Storage:** +**~1.75 GB** per 150-frame draft (11.6 MB/frame ù 150, measured draft 510).
- **I/O:** full extra write at preprocess (+1.75 GB write); alignment reads second tree.
- **Migration:** split 509/510 in place or re-preprocess; high touch on paths, manifest, UI.
- **Would have caught:** P2 structurally (compare like stages by path).
- **Would not catch:** accidental overwrite within a tree; still needs stamps.

### Option C ù Resolver + manifest sidecar only (no new FITS keywords)

- Like LSST JSON sidecar pattern ([DMTN-229](https://github.com/lsst-dm/dmtn-229)): stage/hash only in manifest/`cal_stage.json`.
- **Risk:** violates constraint 4.5 if manifest drifts from FITS; sidecar can desync on manual copy.
- **Rejected** as primary design; acceptable only as **supplement** to Option A.

---

## 9. Invariant registration

Add to `docs/VYVAR_INVARIANTS.md`:

| ID | Rule | Enforcement | On fail |
|----|------|-------------|---------|
| **INV-CAL-02** | Every `calibrated/lights` FITS must carry authoritative `VY_CALSTAGE` + `VY_CALDATASUM` after implementation; legacy frames resolved honestly; pixel compare gates forbidden from assuming `PURE` without resolution | `invariants_runtime.check_cal_stage` + write-path self-check | FAIL closed per obs_group on stamp/hash mismatch; INDETERMINATE blocks naive compare |

`INV-GATE-REMOVAL` applies: byte-identity alone cannot prove the gate remains.

---

## 10. Implementation map (follow-on, not this task)

| Location | Change |
|----------|--------|
| `pipeline.py` `_calibrate_one_light_disk` ~16231 | stamp `PURE`, DATASUM, manifest |
| `pipeline.py` `_passthrough_lights_to_calibrated` ~16461 | stamp `PASSTHROUGH` |
| `pipeline.py` `_qc_enrich_one_frame` ~17668 | stamp `SKYSF_N`, `VY_PSTBG`, DATASUM |
| `cal_diag.py` | extend resolver; wire P2 to `resolve_calibrated_stage` |
| `invariants_runtime.py` | `check_cal_stage` |
| `dev/tools/inv_cal01_validate.py` | P2a stage verify |
| `dev/tests/test_cal_stage_gate.py` | new |

---

## 11. Plain language ù what changes for Milan

**What you gain**

- Opening a calibrated frame or running a validation harness tells you **immediately** whether
  pixels are pure calibration or sky-subtracted ù not by guessing from the folder name.
- A repeat of the July ù150/150 mismatchù investigation stops at **stage mismatch**, not
  hours of dark-binning forensics.
- If something mutates pixels without updating the stamp, the verify step fails loudly.

**What it costs you**

- **Option A (recommended):** essentially nothing day-to-day ù a few new FITS keywords and
  a verify line in the pre-push harness. No re-run of drafts 435/509/510; no extra disk.
- **Option B:** ~**1.7 GB per draft** duplicate and a migration project; only worth it if
  you want folder names to be physically truthful, not just stamped.

**What it would have caught**

- **P2 (2026-08):** yes ù before comparing pixels.
- **P-10 (2026-07):** partially ù would not fix the sign bug, but would show *which* stage
  was applied and a hash delta proving pixels changed in preprocess.

**What it would not catch**

- A wrong formula that still runs to completion and stamps the expected stage name.
- QC header staleness on **pure** frames where edge QC ? full-frame median (435) ù that is
  why SKYSF coherence uses `VY_PSTBG`, not `VY_QCBG`.

---

## 12. References

- FITS DATASUM: [FITS Checksum Keyword Convention](https://fits.gsfc.nasa.gov/registry/checksum.html)
- DRAGONS staged filenames (`_flatCorrected`, `_skyCorrected`): [GSAOI tutorial ù2](https://dragons.readthedocs.io/projects/gsaoiimg-drtutorial/en/v3.0.4/02_data_reduction.html)
- LSST Butler / provenance sidecars: [DMTN-229](https://github.com/lsst-dm/dmtn-229), [Recording Provenance](https://pipelines.lsst.io/v/weekly/modules/lsst.pipe.base/recording-provenance.html)
- IVOA ObsCore `calib_level`: [ObsCore v1.1 WD ù3.3.2](http://www.ivoa.net/documents/ObsCore/20150609/WD-ObsCore-v1.1-20150605.pdf)
- ccdproc: new output file per step + `HISTORY` cards ([Ohio State ccdproc test notes](https://www.astronomy.ohio-state.edu/pogge.1/Software/CCDProc/testccd.html))
- BANZAI: separate extensions/products per reduction stage ([LCO BANZAI docs](https://lco.global/documentation/data/BANZAIpipeline/))
- VYVAR prior art: `dev/results/CURSOR_RESULT_skysf_double.md`, CAL-DIAG v2 ù12.2
