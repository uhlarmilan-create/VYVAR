CURSOR RESULT - 2026-08-04 15:45 UTC+2

WIDE-ERR STEP 0: check-star validity gate (read-only).
Target: draft_000435_snapshot_skysurface_20260716, NoFilter_60_2, equipment_id 1.
Harness: dev/tools/wide_err_step0_checkstar.py
Machine JSON: tmp/wide_err_step0_checkstar.json

## STEP 1 -- check_catalog_id census

Read all 248 platesolve/NoFilter_60_2/photometry/lightcurves/check_kmag_*.csv;
id column is check_catalog_id (no check_cid alias present).

| metric | value |
|--------|-------|
| n_files | 248 |
| n_distinct check ids | 2 |
| 1497265907653703680 | 240 |
| 1499064399440851968 | 8 |

**Cross-check vs select_check_star (check_star_kmag.py:241-305):**

comparison_stars_per_target.csv has **no is_check_star column** and **no p2p_rms
column** (only comp_rms). select_check_star therefore falls back to comp_rms on
this export.

| metric | value |
|--------|-------|
| n_distinct select_check_star ids | 17 (222 targets return NONE) |
| sidecar vs select_check_star mismatch | **248 / 248** |

Sample mismatches:

| target_cid | sidecar check_catalog_id | select_check_star id |
|------------|--------------------------|----------------------|
| 1485516006708222208 | 1497265907653703680 | NONE |
| 1485534187306501376 | 1497265907653703680 | NONE |
| 1485540612577549568 | 1497265907653703680 | NONE |

**Finding (worse than n=1 convergence):** the sidecar writer hardcodes a single
check_catalog_id on 240/248 files regardless of target. This is a metadata-constant
bug in the PROC-MAG-NAMING / K2-ledger class: the id does **not** reflect
select_check_star convergence.

**Relation to WIDE-ERR T1 note:** tmp/wide_error_budget_diag.json (2026-08-04 run)
used **different** sidecar metadata: 160/162 fields carried
1499906247391001088 (G~8.74), 2 carried 1497528072458898432. Current on-disk
sidecars have since been overwritten with 1497265907653703680 (G~11.9). The T1
"same star" claim applied to the **prior** sidecar epoch, not the present files.

## STEP 2 -- VSX cone search

Coordinates from masterstars_full_match.csv / local Gaia DB.

**WIDE-ERR dominant star 1499906247391001088 (G=8.74, RA=206.313 deg, Dec=+40.421 deg):**

| item | value |
|------|-------|
| VSX match within 10 arcsec | **NO** |
| nearest VSX separation | **166.6 arcsec** |
| nearest name | Gaia DR3 1499904632482792320 |
| type / period / mag range | not assigned (Gaia placeholder entry) |

**Current sidecar dominant 1497265907653703680 (G=11.93):**

| item | value |
|------|-------|
| VSX match within 10 arcsec | **NO** |
| nearest VSX separation | **30.1 arcsec** |
| nearest name | Gaia DR3 1497265873295605120 |

Expected and near-tautological: comp selection vetoes VSX-known stars
(pipeline.py:6068-6070). A clean VSX result proves little for low-amplitude
candidates.

## STEP 3 -- Gaia DB schema and var_flag

Table: gaia_dr3 (vyvar_gaia_dr3.db).

| column | present in schema |
|--------|-------------------|
| var_flag | **YES** (alias for phot_variable_flag per build_gaia_catalog.py:384,404) |
| phot_variable_flag (literal) | **NO** |
| non_single_star | **YES** |
| g_mag | **YES** |
| bp_rp | **YES** |
| teff_gspphot | **YES** |
| logg_gspphot | **YES** |

**Row 1499906247391001088:**

| field | value |
|-------|-------|
| var_flag | NOT_AVAILABLE |
| non_single_star | 0 |
| g_mag | 8.743 |
| bp_rp | 0.792 |
| teff_gspphot | null |
| logg_gspphot | null |

**Full-table var_flag distribution:**

| var_flag | count |
|----------|------:|
| NOT_AVAILABLE | 205,902,247 |
| VARIABLE | 5,810,353 |

**428-fixbatch claim (CURSOR_RESULT_428_fixbatch.md:19):** column-name confusion,
not real absence. phot_variable_flag is stored as **var_flag**. The local DB
does carry variability data for ~5.8M sources; this check star is NOT_AVAILABLE.

## STEP 4 -- did the Gaia variability gate fire?

**gaia_variable_df population (pipeline.py):**

- Initialized None at pipeline.py:10220.
- _prefetch_export_shared_catalog_for_process_pool (9333-9411): if g_df is None,
  sets g_df = pd.DataFrame() with **no query**. VSX is loaded; Gaia variable
  catalog is not.
- gvar_hit (8514-8520): positional match against gaia_variable_df rows
  (ra_deg, dec_deg), **not** var_flag on the matched catalogue row.
- catalog_known_variable = vsx_hit | gvar_hit (8523).

**On draft_435 masterstars_full_match.csv (4122 rows):**

| flag | count True |
|------|----------:|
| vsx_known_variable | 269 |
| gaia_dr3_variable_catalog | **0** |

**Finding:** Gaia arm of catalog_known_variable is a **silent no-op** on this
data. n_gaia_variable_in_field = 0 always. Docstring at pipeline.py:6049 and
FLOW prose at dev/tools/docs_pdf/build_flow_doc.py:563 overstate what runs
(they imply phot_variable_flag / gaia_variable_df gate; only VSX proximity fires).

## STEP 5 -- empirical scatter discriminator

Pre-registered reading: sigma_p2p/err ~ 1 with sigma_total/err ~ 2 -> RED
(smooth); both ~ 2 -> WHITE (error model); else report numbers.

### A. Prior WIDE-ERR sample (1499906247391001088, n=160 from JSON)

From tmp/wide_error_budget_diag.json (production-path LCs at run time):

| metric | value |
|--------|-------|
| median sigma_total / err | **1.96** |
| median scatter | 20.3 mmag |
| median quoted err | 9.5 mmag |
| p95 ratio | 3.88 |

**sigma_p2p / err:** **NOT MEASURABLE.** Cached diag LCs at
photometry/diag_check_lc/*/lightcurve_1499906247391001088.csv are header-only
(0 data rows). Re-photometry on 2026-08-04 returns n=0 valid epochs for this
star on representative targets.

**Lomb-Scargle on 1499906247391001088:** not run (no LC data).

### B. Current sidecar star 1497265907653703680 (n=248 cached diag LCs)

| metric | value |
|--------|-------|
| median sigma_total / err | **4.36** |
| median sigma_p2p / err | **4.22** |
| representative target | 1485516006708222208 |
| n_epochs | 149 |
| baseline | 0.209 days (~5.0 h) |
| LS frequency range | 4.79 - 357 c/d |
| LS best period | 0.142 h |
| LS FAP | 0.984 |
| folded peak-to-peak amplitude | 0.96 mag |

Both ratios ~4.2 (similar, not split) -> **WHITE pattern** on current wrong
sidecar star, but ratio magnitude differs from prior ~2x WIDE-ERR verdict.

## STEP 6 -- rank 1 / 2 / 3 degeneracy break

select_check_star ranks on p2p_rms, but comparison_stars_per_target.csv on disk
**lacks p2p_rms**. STEP 6 used **comp_rms ascending** (select_check_star fallback
path) on 20 fields with three usable ranks.

| rank | n fields | median sigma_total / err |
|------|----------|-------------------------|
| 1 (best comp_rms) | 20 | **4.35** |
| 2 | 20 | **3.75** |
| 3 | 19 | **5.55** |

Ratio does not collapse toward ~1 at rank 2-3; rank 3 is higher. On this export,
the excess is not isolated to a single p2p-selected star (but see STEP 1: current
sidecar id is not the selector output anyway).

## Verdict

**UNDECIDED** -- sigma_p2p/err for the WIDE-ERR dominant check star
(1499906247391001088, 160/162 prior fields) is missing: cached production LCs are
empty and re-photometry yields n=0, so the pre-registered RED vs WHITE
discriminator cannot be applied to the sample that motivated H1-global.

Secondary findings (do not change verdict line but block naive proceed):

1. check_kmag sidecar metadata is broken (248/248 mismatch with select_check_star;
   240/248 share one hardcoded id) -- invalidates any id census from sidecars alone.
2. Gaia variability gate (gaia_variable_df) never loads on draft_435.
3. Current sidecar epoch gives ~4x underquote (both estimators), not the prior ~2x.

Do not proceed to ensemble-SEM on the prior WIDE-ERR verdict until sigma_p2p/err
is recovered for 1499906247391001088 on the 160-field sample (or the sample is
rebuilt with valid check-star selection metadata).

## Files created

| path | role |
|------|------|
| dev/tools/wide_err_step0_checkstar.py | read-only diagnostic harness |
| dev/results/CURSOR_RESULT_wide_err_step0_checkstar.md | this report |
| tmp/wide_err_step0_checkstar.json | machine output |

No src_py/, config, or anchor changes.

## STEP 0b -- provenance (2026-08-04)

### Retraction (STEP 1 sidecar writer claim)

STEP 1's finding "the sidecar writer hardcodes a single check_catalog_id regardless of
target" is withdrawn. `photometry_core.py:9251-9263` selects the check star from
`field_check_star_candidate_pool(state.comp_df, target_comps=...)`
(`check_star_kmag.py:210-238`), a field-wide pool deduped to one row per catalog_id.
The same star winning for most targets is designed behaviour; only the excluded
ensemble differs per target. The reported "248/248 mismatch" came from re-running
`select_check_star` on `comparison_stars_per_target.csv`, a per-target export that
lacks both `p2p_rms` and `is_check_star` -- a harness-input error, not a pipeline
defect. Raw id census stands: 248 files, 2 distinct ids, split 240 / 8.

### Q1 -- artifact mtimes

Harness glob path (`wide_error_budget_diag.py:123-126,161`):
`Archive/Drafts/draft_000435_snapshot_skysurface_20260716/platesolve/NoFilter_60_2/photometry/lightcurves/check_kmag_*.csv`
-- same directory as listed below.

| artifact | count | oldest mtime (local) | newest mtime (local) |
|----------|------:|----------------------|----------------------|
| check_kmag_*.csv | 248 | 2026-08-04 11:25:52 | 2026-08-04 11:33:10 |
| diag_check_lc/ subdirs | 248 | 2026-08-04 15:23:10 | 2026-08-04 15:50:47 |
| comparison_stars_per_target.csv | 1 file | -- | 2026-08-04 11:24:59 |
| masterstars_full_match.csv | 1 file | -- | 2026-08-04 10:20:09 |
| pipeline_meta.json | 1 file | -- | 2026-08-04 11:47:29 |
| tmp/wide_error_budget_diag.json | 1 file | -- | 2026-08-04 09:17:57 |
| tmp/wide_err_step0_checkstar.json | 1 file | -- | 2026-08-04 15:46:31 |

lightcurve_1499906247391001088.csv under diag_check_lc/ (4 files):

| path suffix | mtime (local) | size bytes |
|-------------|---------------|----------:|
| .../1485540612577549568/lightcurve_1499906247391001088.csv | 2026-08-04 15:47:36 | 533 |
| .../1485552329248338816/lightcurve_1499906247391001088.csv | 2026-08-04 15:47:37 | 533 |
| .../1485574899299782528/lightcurve_1499906247391001088.csv | 2026-08-04 15:47:39 | 533 |
| .../1485609538212672000/lightcurve_1499906247391001088.csv | 2026-08-04 15:47:40 | 533 |

162 vs 248: WIDE-ERR JSON (09:17:57) reports n_check_fields=162 on the same glob path.
All 248 current check_kmag_*.csv mtimes postdate that JSON (earliest 11:25:52). Current
248-file sidecar set was written BETWEEN the 09:10 WIDE-ERR run and the 15:45 STEP 0 run
(~11:25-11:33). CANNOT DETERMINE the check_kmag file count at 09:10 -- prior sidecars
overwritten, no backup on disk.

### Q2 -- pre- or post-batch-E (pipeline_meta.json)

Last-writer-wins; no older copy or backup of pipeline_meta.json found under the draft tree.

| field | value |
|-------|-------|
| provenance.git_hash | 20dde2bcbacae25b14d532d3ef78524dd4e24d29 |
| provenance.git_dirty | true |
| provenance.entry_point | run_phase2a |
| config_snapshot.admission_sat_peak_frac | present, 0.7 |
| config_snapshot.dao_detection_n_equiv | present, 3.78 |
| config_snapshot.enable_lacosmic | present, true |

Key present -> artifacts are POST batch E physical re-cut (per reading rule).

### Q3 -- 1499906247391001088 admissibility (current on-disk)

**masterstars_full_match.csv:** row present.

Admission columns that exist on file:
is_saturated, photometry_ok, is_usable, peak_dao, peak_max_adu, saturate_limit_adu,
likely_saturated.
Absent from file: peak, sat_peak_frac, saturate_limit_fraction, status, skip_photometry.

| column | value |
|--------|-------|
| is_saturated | false |
| photometry_ok | true |
| is_usable | true |
| likely_saturated | false |
| peak_dao | 33097.54296875 |
| peak_max_adu | 35296.69140625 |
| saturate_limit_adu | 65535 |

**comparison_stars_per_target.csv:** 0 rows with catalog_id=1499906247391001088.
Distinct status values: (none).

**Peak-to-full-well fraction:** peak_max_adu / saturate_limit_adu = 35296.69140625 /
65535 = **0.5386**.
Compared to admission_sat_peak_frac = 0.70: below threshold.
Compared to is_saturated flag (0.85 threshold per task): is_saturated = false.

