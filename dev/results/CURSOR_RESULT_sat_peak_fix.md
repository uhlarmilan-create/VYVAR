CURSOR RESULT - 2026-08-13

What I did
Fixed raw peak search (anchored WCS-residual + brightness plausibility + tightened
pass gates). Saturation gates use **aligned** peak when raw is not `RAW_VERIFIED`,
with explicit `ALIGNED_INTERIM` / `MIXED` provenance (`sat_peak_source`, `VY_SATPS`).
Re-measured on draft 510 BO CVn (134 frames). Committed locally; **not pushed**
(awaiting Milan go-ahead).

Validation script: `tmp/_sat_peak_fix_validate.py`

---

## 1 - Search fix (not downstream filter)

### 1.1 Expected brightness (plausibility)

**Chosen:** ratio test `raw_peak / aligned_peak` in **[1/3, 3]** (`PEAK_RAW_ALIGNED_MAX_RATIO`).

**Why not catalog magnitude alone:** proc frames already carry aligned `peak_max_adu`
from DAO photometry on the same star. The ratio is per-star, per-frame, uses the
frame's actual flux scale (ZP, extinction, airmass already in aligned peak), and
needs no field-specific threshold tuning.

**When aligned >= 85% of `sat_adu`:** ratio skipped (resampling can push aligned
above raw container; ratio is meaningless). Anchor distance still required.

**When prediction is poor:** missing/NaN aligned peak -> plausibility skipped
(anchor + self-check only). Variable star, blend, or bad aligned photometry may
fail plausibility or anchor -> `peak_loc_fail`, saturation falls back to aligned.

### 1.2 Aligned centroid anchor

**Chosen:** `expected_raw_from_aligned_centroid` -- raw WCS sky position plus aligned
DAO offset `(x - wcs_x, y - wcs_y)`. Fallback: plate-offset from bright reference star;
last resort WCS + ref-star drift.

**Verification:** disk search within **12 px** (`PEAK_ALIGNED_MAX_DIST_PX`) for pixels
passing self-check **and** plausibility. Best ADU among passes wins. Removed
`mag_guided_centroid` on target stars (kept for ref-star drift only).

**Pass meaning:** `peak_loc_ok=true` => same star as aligned photometry (anchor +
plausibility + self-check), not merely "bright pixel nearby".

### 1.3 Combined with self-check

All three must pass for `RAW_VERIFIED`. Failure -> `peak_loc_fail`, authoritative
`peak_max_adu` = aligned peak, `sat_peak_source=ALIGNED_INTERIM`.

---

## 2 - ALIGNED_INTERIM saturation (temporary, recorded)

| Item | Implementation |
|------|----------------|
| Per-row provenance | proc CSV `sat_peak_source` |
| Draft aggregate | `sat_diag.json` `sat_peak_source` (`RAW_VERIFIED` / `ALIGNED_INTERIM` / `MIXED`) |
| FITS header | `VY_SATPS` via `stamp_sat_fits_headers` |
| Authoritative peak | `peak_max_adu = peak_max_adu_raw` if verified else `peak_max_adu_aligned` |

**Conservative interim caveat (spec 8.4):** aligned peaks can exceed raw ceiling
(draft 510 up to ~69000 vs 65535). **Over-admission risk:** comps that are raw-saturated
but aligned-sub-threshold may pass admission. **Under-exclusion risk:** none vs raw-only
path for aligned-bright cases. Cannot detect raw-container saturation when interim
peak is used.

**P1 check (sat gate simulation, 134 frames reprocessed):** all five 509 comps pass
admission gate including `1497974027502858240` (0/134 over threshold vs 59/90 before).

---

## 3 - Measurements (draft 510 BO CVn, 134 frames)

Script reprocessed every `proc_BO_CVn_Light_*.csv` with raw FITS + aligned FITS headers.

### 3.1 Drop comp `1497974027502858240` vs aligned position

| | Before (mag-guided) | After (anchored) |
|--|-------------------|------------------|
| `peak_loc_ok` | 90 / 134 | **42 / 134 (31%)** |
| `peak_loc_fail` | 44 | **92** |
| Sat gate (auth peak) | **reject** (59/90 over) | **pass** (0/134 over) |

Raw search locates the star on a minority of frames; ALIGNED_INTERIM restores correct
saturation admission.

### 3.2 Hijack pattern stars

| Metric | Before | After |
|--------|-------:|------:|
| Stars with raw/aligned > 3x on >10% pass frames (validation script) | 85 | **0** |
| Pre-push memo definition (faint comp hijack subset) | 19 | **0** |
| Of those failing sat gate | 7 | **0** |

### 3.3 Peak self-check failure counts (selected)

| Star | Before ok/fail | After ok/fail |
|------|----------------|---------------|
| `1497974027502858240` (drop comp) | 90 / 44 | **42 / 92** |
| `1498613634033133184` (BO CVn target) | 134 / 0* | **12 / 122** |
| `1500347838748255360` (bright ref) | - | **5 / 129** |

\*Before target showed 90 ok in reprocessed baseline because old proc mixed hijack
semantics; pre-push memo reported 134/0 for target.

Draft-level peak sources after full pass: **MIXED** (11187 RAW_VERIFIED, 78679 ALIGNED_INTERIM).

### 3.4 Genuinely saturated star

Bright ref `1500347838748255360` frame 001: aligned peak **68567 ADU** ->
`likely_saturated=True`, `is_saturated=True` via **ALIGNED_INTERIM** (raw not verified).
Saturation decision preserved; star not made invisible.

### 3.5 Draft 510 fresh photometry / check-star scatter

**Not re-run** in this session (full catalog re-export + photometry pipeline required).
Existing 510 photometry still reflects 4-comp ensemble (pre-fix export).

509 reference (same check star `1497313255374892800`, BO CVn):
- `check_scatter` = **0.008629** (trust_flag_core)
- `ac_scatter` = separate metric (do not compare to check_scatter; see prepush report)

---

## 4 - Pre-registered predictions

| ID | Stated before run | Result | Measured |
|----|-------------------|--------|----------|
| **P1** | 5 comps, same IDs as 509 with aligned peaks | **PASS** | Sat gate sim: all 5 pass; 509 list matches |
| **P2** | check_scatter within **0.0005** of 0.008629 | **NOT RUN** | Photometry not re-exported; 5-comp restoration expected to match 509 |
| **P3** | Drop star located near aligned on **>=90%** of 134 frames | **FAIL** | **42/134 (31%)** RAW_VERIFIED |
| **P4** | `--fast` OVERALL PASS | **PASS** | 1304 passed, 27 skipped; pytest OK |
| **P5** | Hijack-pattern stars (>10% frames) -> 0 | **PASS** | 19 -> **0** (pre-push definition); script ratio test 85 -> 0 |

---

## 5 - Push status

Two local commits on `main` (ahead of origin). **`--fast` PASS.** Raw output captured.
**Not pushed** -- awaiting Milan go-ahead per task 5.2.

---

## 6 - Is raw search trustworthy for saturation?

**Not yet for all stars.** The hijack failure mode is eliminated (P5 PASS), and the
drop comp is correctly located when verified (raw ~7k ADU, not 49k). But P3 FAIL:
only **31%** of frames achieve `RAW_VERIFIED` for the faint comp; the bright target
is **12/134**. Saturation currently depends on ALIGNED_INTERIM for the majority of
star-frames (draft 510: ~88% interim).

**Evidence to settle:** (a) raise anchor recovery without re-opening hijack -- tune
WCS residual vs plate offset, or relax self-check for faint cores within anchor disk;
(b) `RAW_VERIFIED` fraction **>90%** on BO CVn comps and target; (c) re-run draft 510
photometry with P2 PASS; (d) side-by-side raw vs aligned saturation on a draft with
known raw-saturated comps (not just aligned-resampled).

---

## Files changed

- `src_py/sat_diag.py` -- anchored search, plausibility, ALIGNED_INTERIM gates
- `src_py/pipeline.py` -- pass `aligned_hdr`, stamp `meta["sat_peak_source"]`
- `dev/tests/test_sat_diag.py` -- plausibility + anchored faint-star tests
- `dev/results/specs/VYVAR_SAT_DIAG_SPEC.md` -- sections 8.2-8.4
- `dev/results/CURSOR_RESULT_sat_diag_implement.md` -- ASCII fix
- `dev/results/CURSOR_RESULT_sat_peak_fix.md` -- this file
