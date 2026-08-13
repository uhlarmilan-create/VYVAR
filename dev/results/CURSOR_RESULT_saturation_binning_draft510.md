CURSOR RESULT - 2026-08-13

What I did
Measured raw ADU histograms, headers, and saturated-star plateaus for draft 510; audited saturation-limit code paths; resolved runtime limits for drafts 435/509/510; computed per-star peak statistics on **correct** BO CVn target and check star (Gaia `1498613634033133184`, check `COMP_0021` / `1498428263244809344`) for drafts 509 and 510.

---

## Part A - Facts (measured)

### A.1 Raw draft 510 - top of histogram (150 frames, all pixels)

| Rank | ADU value | Pixel count (cumulative over stack) |
|------|-----------|-----------------------------------|
| 1 | **65535** | **13024** |
| 2 | 65532 | 1 |
| 3 | 65524 | 1 |
| 4 | 65520 | 1 |
| 5 | 65516 | 2 |
| 6 | 65512 | 3 |
| 7 | 65508 | 2 |
| 8 | 65504 | 2 |
| 9 | 65500 | 3 |
| 10 | 65496 | 1 |

**Pile-up at ceiling:** 13?024 pixels at exactly **65535**. The next value down (65532) has count **1** - the ceiling is a hard plate at 65535, not a gradual rolloff.

**Highest pile-up below 65535:** value **5184**, count **523** (background mode; not a saturation shoulder).

### A.2 Values near 16384 (14-bit unbinned hypothesis)

| Metric | Count |
|--------|------:|
| Exactly **16384** | 22 |
| Range **16380-16390** | 77 |

No concentration pattern consistent with ?four summed 14-bit wells? (4096 multiples show small counts: 4096?975, 8192?147, 12288?62, 16384?22, then falling - not a ladder of summed plateaus).

### A.3 Raw header keywords (verbatim, first frame `BO_CVn_Light_001.fits`)

| Keyword | Value |
|---------|-------|
| BITPIX | 16 |
| BZERO | 32768 |
| BSCALE | 1 |
| XBINNING | 2 |
| YBINNING | 2 |
| GAIN | 0.0 |
| OFFSET | 0.0 |
| READMODE | 0.0 |
| CCD-TEMP | -10.0 |
| EXPTIME | 60.0 |
| ROWORDER | TOP-DOWN |
| XPIXSZ | 9.26 |
| YPIXSZ | 9.26 |

No `XBAYBIN`/`YBAYBIN`/`READOUTM` in this header. Full keyword dump available in measurement script output.

### A.4 Saturated star core plateau (raw)

Frame: `BO_CVn_Light_010.fits`, bright comp region near **(y=349, x=1710)** (local peak hunt near COMP_0001 position).

| Observation | Value |
|-------------|-------|
| Peak | **65535** |
| 5?5 core value counts | **65535 ? 7**; shoulder pixels **10056, 12168, 11504, 9024, 6428, 16452, 22308, 25328, 19104** (1 each) |
| Pixels > 60000 in 11?11 | **9**, all **65535** |

Flat top at the **65535** ceiling with steep shoulders - not a comb of exact multiples of 16384.

### A.5 Code sites - saturation limit read/compared/stored

**Resolution (all treat limit as same units as FITS pixel values - binned image ADU; no `XBINNING` scaling anywhere):**

| Location | Role | Units |
|----------|------|-------|
| `database.py:3127` | Read `EQUIPMENTS.SATURATE_ADU` | Stored DB value ? compared directly |
| `database.py:3253-3350` | `get_combined_metadata` exposes `saturate_adu` | No binning multiply |
| `pipeline.py:5285-5324` | `_effective_saturation_limit` | Header ? equipment DB ? DATAMAX ? BITPIX(16)?65535 |
| `pipeline.py:5491-5525` | `_star_saturation_flags` | `peak_max_adu` vs `sat_limit * sat_frac` |
| `pipeline.py:6178-6303` | `_annotate_masterstars_flux_zones` | `is_saturated = peak > peak_sat_lim` |
| `pipeline.py:7777-8158` | DAO detection + catalog match | passes equipment sat to flags |
| `pipeline.py:14013-14020` | Alignment job | loads equipment sat |
| `comp_selection_per_target.py:383-386, 798-837` | Comp admission | `peak_max_adu` vs `saturate_limit_adu_85pct * (0.70/0.85)` |
| `photometry_core.py:6127-6304` | Target `skip_photometry` / per-frame sat gate | `likely_saturated`, `peak_max_adu` |
| `psf_runner.py:453, 972` | PSF comp skip | peak vs `0.85 * saturate_limit_adu` |

**Mixed-units risk:** DB stores **16384** (interpreted as 14-bit **unbinned** well depth) but FITS pixels after 2?2 hardware binning reach **65535**. Code never multiplies limit by `XBINNING * YBINNING`. **This is the unit mismatch.**

### A.6 Historical `SATURATE_ADU` and runtime resolution

| State | Value | Source |
|-------|-------|--------|
| **Current DB** `EQUIPMENTS.ID=1` | **16384.0** | `database.py:3127` (live query) |
| **Draft 435** runtime | **16384.0** | `equipment_db` (`MASTERSTAR.fits` header) |
| **Draft 509** runtime | **16384.0** | `equipment_db` |
| **Draft 510** runtime | **16384.0** | `equipment_db` |

**Before Milan's change:** `SATURATE_ADU` column is not versioned in git (SQLite data). `docs/VYVAR_JOURNAL.md` (2026-07 era) documents zone tagging against **65535 ADU** (?85% of equipment ceiling (55.7 k from **65535 ADU**)?). When `SATURATE_ADU` was NULL, `_effective_saturation_limit` fell through to `_infer_sat_limit_from_bitpix` ? **65535** for raw 16-bit headers (`pipeline.py:5257-5260`). **No git commit sets 16384 in code** - the change is DB-only and now active for all three drafts.

---

## Part B - Decision options (no implementation)

**Evidence summary:** Raw data ceiling = **65535 ADU** in binned FITS units. Runtime limit = **16384 ADU** with no binning factor applied. 13?024 pixels sit at 65535 while the code treats saturation above 16384.

| Option | DB | Code | Existing drafts | Rig binning mismatch risk |
|--------|----|------|-----------------|---------------------------|
| **B1 - Store limit in binned FITS units** | Set `SATURATE_ADU ? 65535` (or measured ceiling) | No scaling logic | Re-tag MASTERSTAR zones; many comps regain ?linear? | Low if header binning matches |
| **B2 - Store unbinned, scale at compare** | Keep 16384 | Multiply by `XBINNING*YBINNING` in `_effective_saturation_limit` / `_star_saturation_flags` | Limits auto-scale from headers | **High** if header missing/wrong binning or sum-vs-mean binning ambiguity |
| **B3 - Derive from data/BITPIX** | Optional override only | Use measured frame max or BITPIX ceiling; drop stored limit | Headers differ draft-to-draft | Medium; raw vs calibrated BITPIX differs (-32 float) |

**Measurement supports B1 or B2.** The hardware path is 2?2 binned sum to **65535 ? 4 ? 16384**. B2 matches physical sensor semantics if scaling is **sum binning** (factor 4). B1 is simpler and matches what the FITS files actually contain. B3 alone does not fix the 16384-vs-65535 contradiction without knowing binning mode.

---

## Part C - Consequence check at active limit 16384

Limit **16384**; 85% = **13926.4**; admission gate = **70%/85% ? limit ? 13495 ADU** (`admission_sat_peak_frac=0.70`, `saturate_limit_fraction=0.85`).

Stars identified by **Gaia catalog_id** on MASTERSTAR grid `(x,y)` from `variable_targets.csv` / `comparison_stars.csv`.

### C.1 Per-star peaks (7?7 box), drafts 509 & 510 (identical raw/det for BO CVn night)

| Star | catalog_id | Stage | n | min | median | max | frames ?16384 | frames ?13926 |
|------|------------|-------|---|-----|--------|-----|---------------|---------------|
| **BO CVn (target)** | 1498613634033133184 | raw | 150 | 2056 | 3662 | **24168** | **61** | **65** |
| BO CVn | 1498613634033133184 | det | 134 | 10629 | 14733 | 18487 | **20** | **79** |
| **Check (COMP_0021)** | 1498428263244809344 | raw | 150 | 1948 | **31994** | **55836** | **77** | **77** |
| Check COMP_0021 | 1498428263244809344 | det | 134 | 35160 | 42326 | 55049 | **134** | **134** |

Target BO CVn is **not** saturated on most frames (median 3662 raw) but crosses 16384 on **61/150** raw frames at peak. Check star is **saturated on most frames** even in raw.

**Comparison ensemble (MASTERSTAR `peak_max_adu`):** 140 comps; **61** have peak ? 13926; **62** fail admission gate (>13495); **78 survive** admission.

At a **65535** limit, only **16** comps would fail admission - **124 survive**.

### C.2 Pipeline actions when saturated (file:line)

| Stage | Action | Location |
|-------|--------|----------|
| MASTERSTAR zone tag | `is_saturated`, `likely_saturated`, `photometry_ok=False` | `pipeline.py:5491-5525`, `6296-6302` |
| Comp pool filter | Exclude `is_saturated`, `likely_saturated` | `comp_selection_per_target.py:383-386` |
| Comp admission (per-frame) | Reject if peak > `saturate_limit_adu_85pct ? 0.70/0.85` | `comp_selection_per_target.py:798-837` |
| Target photometry | `skip_photometry` if saturated / per-frame gate | `photometry_core.py:6127-6304`, `8334+` |
| LC export flag | `flag=saturated` in per-frame CSV | `photometry_core.py:2460` |
| PSF comps | Skip if peak > 0.85 ? limit | `psf_runner.py:453, 972` |
| Aperture correction | Uses non-saturated ref stars (`ac_skip_reason`) | via `photometry_summary.csv` columns |

### C.3 BO CVn comp survival at 16384

**78 of 140** comparison stars pass static MASTERSTAR admission at the current limit.

Draft 509 photometry already ran BO CVn with **5 clean comps** (`photometry_summary.csv`, catalog_id `1498613634033133184`). That run used the same catalog geometry; the 16384 limit thins the comp pool but does not zero it.

---

## Part D - Report summary

**Contradiction confirmed:** FITS pixels reach **65535**; runtime limit is **16384** in the **same units** (no binning scale). Milan's DB value **is active** for drafts 435/509/510 via `equipment_db`.

**DECISION REQUIRED:** (B1) store ~65535 binned ADU in DB, or (B2) store 16384 unbinned and multiply by binning at compare, or (B3) derive ceiling from BITPIX/measurement - **measurement supports B1 or B2** (65535 ? 4?16384 for 2?2 sum binning).

**Production run:** **Do not treat saturation as resolved.** A run can proceed for BO CVn with ~78 comps, but photometry/saturation tagging will be wrong until the limit units are fixed - 61 raw target frames and 61 bright comps are misclassified relative to the true **65535** ceiling.

## Files changed

None (read-only + decision memo).

## Artifacts

- `tmp/_white_cores_v3.json`
- `tmp/_sat_binning_white_v3.py` (partial)
