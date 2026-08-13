CURSOR RESULT - 2026-08-05 (WIDE-ERR-CROSSRIG DRAFT-501 zero-LC diagnosis)

What I did
Read-only inspection of draft_000501 platesolve/photometry artifacts. Joined
active_targets.csv catalog_id against masterstars_full_match.csv. Compared zone
column presence and distribution against draft_435 anchor. Located zone-assignment
writer in pipeline.py. Diagnostic outputs under tmp/draft501_diag/.

## 1 -- Draft location

| field | value |
|-------|-------|
| draft path | Archive/Drafts/draft_000501 |
| setup subfolder | V_60_2 |
| git_hash | 2c964cb660e8e0ecd3b9dfe29063e30fb1e2b54c |
| entry_point | run_phase2a |

pipeline_meta.json: platesolve/V_60_2/photometry/pipeline_meta.json
(provenance.git_hash, provenance.entry_point).

## 2 -- masterstars_full_match.csv

Path: Archive/Drafts/draft_000501/platesolve/V_60_2/masterstars_full_match.csv

| check | result |
|-------|--------|
| file exists | yes |
| row count | 1668 |
| file size | 413764 bytes |
| columns (29) | name, ra_deg, dec_deg, mag, b_v, catalog, catalog_id, x, y, flux, vsx_known_variable, gaia_dr3_variable_catalog, exo_host_obj_id, exo_host_name, exo_cat_source, exo_disposition, exo_match_sep_arcsec, peak_dao, peak_max_adu, saturate_limit_adu, likely_saturated, photometry_ok, bp_rp, phot_g_mean_mag, catalog_mag, match_sep_arcsec, edge_safe_10px, snr50_ok, coord_source |

NOT present (present in draft_435): zone, is_saturated, is_noisy, is_usable,
noise_floor_adu, saturate_limit_adu_85pct, source_type.

Not failure mode A.

## 3 -- Target presence in masterstars

active_targets.csv: 22 rows, all catalog_id populated.

Join active_targets.catalog_id vs masterstars_full_match.catalog_id:

| metric | count |
|--------|-------|
| active targets present in masterstars | 22 |
| active targets absent from masterstars | 0 |
| absent catalog_ids | (none) |

Not failure mode B.

## 4 -- Zone status for targets in masterstars

All 22 active target catalog_ids appear in masterstars. Extracted columns below.
Note: masterstars has no `zone` or `is_saturated` column; `likely_saturated` is
the nearest saturation flag. `phot_g_mean_mag` used as brightness.

| catalog_id | zone | is_saturated | phot_g_mean_mag | peak_max_adu | x | y |
|------------|------|--------------|-----------------|--------------|---|---|
| 1625385430634374272 | (col missing) | (col missing; likely_sat=false) | 17.0599 | 33571.46 | 818.38 | 774.36 |
| 1625382892308863488 | (col missing) | (col missing; likely_sat=false) | 13.9515 | 34844.23 | 1043.40 | 393.02 |
| 1625358840491946496 | (col missing) | (col missing; likely_sat=false) | 14.4852 | 34300.57 | 1826.59 | 1137.82 |
| 1625409074429526912 | (col missing) | (col missing; likely_sat=false) | 13.7630 | 35322.51 | 538.03 | 1575.82 |
| 1625573829375431296 | (col missing) | (col missing; likely_sat=false) | 10.9998 | 49710.05 | 2676.40 | 1965.63 |
| 1625373404725030528 | (col missing) | (col missing; likely_sat=true) | 9.7793 | 91320.63 | 2093.69 | 1332.43 |
| 1625465733638160000 | (col missing) | (col missing; likely_sat=true) | 8.9794 | 86359.19 | 1496.49 | 1707.87 |
| 1624628764771224960 | (col missing) | (col missing; likely_sat=false) | 14.3881 | 34559.67 | 665.55 | 163.31 |
| 1625361348752915968 | (col missing) | (col missing; likely_sat=false) | 16.4591 | 33612.95 | 2596.19 | 690.62 |
| 1625398732148334592 | (col missing) | (col missing; likely_sat=false) | 11.6290 | 48238.60 | 765.62 | 1041.83 |
| 1625400858159986432 | (col missing) | (col missing; likely_sat=false) | 17.2152 | 33528.70 | 1255.41 | 1063.99 |
| 1625369354571886336 | (col missing) | (col missing; likely_sat=false) | 14.9303 | 33891.43 | 2048.37 | 1154.27 |
| 1625379250176675968 | (col missing) | (col missing; likely_sat=false) | 15.2344 | 33867.02 | 910.78 | 305.59 |
| 1625379593774131840 | (col missing) | (col missing; likely_sat=false) | 15.2603 | 33687.43 | 636.35 | 276.47 |
| 1625336025625730816 | (col missing) | (col missing; likely_sat=false) | 13.4117 | 34965.48 | 2871.80 | 268.58 |
| 1625378906579210240 | (col missing) | (col missing; likely_sat=false) | 14.7392 | 34303.43 | 886.92 | 215.21 |
| 1625559295205753344 | (col missing) | (col missing; likely_sat=false) | 14.6009 | 34122.09 | 2952.23 | 1472.83 |
| 1625368358139478656 | (col missing) | (col missing; likely_sat=false) | 17.2382 | 33566.79 | 2359.87 | 1368.21 |
| 1625371858536837376 | (col missing) | (col missing; likely_sat=false) | 17.2064 | 33524.79 | 1899.43 | 1379.75 |
| 1625370591522474752 | (col missing) | (col missing; likely_sat=false) | 15.3200 | 33684.70 | 2004.60 | 1366.51 |
| 1625467932661420928 | (col missing) | (col missing; likely_sat=false) | 13.5907 | 34929.87 | 1222.81 | 1962.84 |
| 1625564139928877568 | (col missing) | (col missing; likely_sat=false) | 15.6107 | 33702.34 | 2170.69 | 1928.44 |

Aggregate (22 targets):

| field | counts |
|-------|--------|
| zone column | missing on all 22 rows (100%) |
| likely_saturated | false: 20, true: 2 |

Aggregate (all 1668 masterstars rows):

| field | counts |
|-------|--------|
| zone column | missing on all 1668 rows (100%) |
| likely_saturated | false: 1657, true: 11 |

Classifier did-not-run signal: `zone` column absent from the entire
masterstars_full_match.csv (not merely blank values). Draft_435 anchor CSV
carries populated zone/is_saturated/is_noisy/is_usable columns from
_annotate_masterstars_flux_zones; draft_501 CSV does not.

active_targets.csv confirms downstream effect: zone_flag=neznama_zona on all 22,
skip_photometry=True, skip_reason=zone_flag. pipeline_meta lc_quality_summary:
no_data=22, good=0.

## 5 -- draft_435 anchor comparison

Path: Archive/Drafts/draft_000435_snapshot_skysurface_20260716/platesolve/NoFilter_60_2/masterstars_full_match.csv

| zone value | count (draft_435, n=2951) |
|------------|---------------------------|
| linear | 1799 |
| noisy3 | 732 |
| noisy1 | 210 |
| noisy2 | 182 |
| saturated | 28 |

draft_501 (n=1668): zone column absent; no zone values to count.

Difference: draft_435 masterstars has a fully populated zone classifier output
(linear/noisy1-3/saturated); draft_501 masterstars has no zone column at all.

## 6 -- Zone assignment site in code

Function: `_annotate_masterstars_flux_zones`
File: src_py/pipeline.py, definition at line 6192; called at line 12459 before
`_vyvar_df_to_csv(df_final, csv_path)`.

Assignment excerpt (pipeline.py:6266-6298):

    out["zone"] = "linear"

    if sat_lim is not None:
        out.loc[peak_s > float(sat_lim), "zone"] = "saturated"

    if nf is not None:
        ...
        out.loc[noisy1_mask, "zone"] = "noisy1"
        out.loc[noisy2_mask, "zone"] = "noisy2"
        out.loc[noisy3_mask, "zone"] = "noisy3"

    if sat_lim is not None:
        out["is_saturated"] = (peak_s > float(sat_lim)).fillna(False)
    ...
    out["is_usable"] = out["zone"].eq("linear") & flux_s.notna()

Reader mapping (photometry_core.py:12677-12690): empty zone -> neznama_zona
(unless is_saturated=True).

## Files changed

None (read-only). Diagnostic artifacts: tmp/draft501_diag/diag_results.json,
tmp/draft501_diag/target_zone_table.csv.

DRAFT501-CAUSE-C -- masterstars present, targets present, zone column empty
