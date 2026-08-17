# CURSOR RESULT - EXPORT-HDR-01

Date: 2026-08-17
Compared with: EXPORT-HDR-01 spec (parent Part C) vs re-export of BO CVn
from Part 3 product SHA **36a53b0**. 8f107cf is quarantined; not used.
Push: NOT authorized.

Premise: header fields on AAVSO + VarAstro BO CVn exports must match the
sidecar/ensemble chain of the 48-LC product, with PFS ON via the Part 2.4
run override (not persisted config.json).

JSON: `dev/results/EXPORT_HDR_01_summary.json`.
Tool: `dev/tools/export_hdr_01_bo_cvn.py`.

Outputs:
- `Archive/Drafts/draft_000515/platesolve/NoFilter_60_2/photometry/lightcurves_reports/aavso/BO_CVn_20260423.txt`
- `.../varastro/BO_CVn_20260423.txt`

## C1 NOTES + length-guard

NOTES: `meth=aperture|n_comp=4 GaiaDR3 ensemble|trust=GREEN|4 clean comps`

Truncated Gaia IDs on any output path: **[]** (length-guard
`find_truncated_gaia_ids`). Test: `dev/tests/test_export_hdr_01.py`.

## C2 COMP TABLE

VarAstro prints `w_pre` and `w_post` (pytics-updated weights, both
labeled). Guard: test_varastro_comp_table_prints_pre_and_post_weight_columns.

## C3 CHK KNAME/KMAG

From the check sidecar, not `na`. 0 `na` of 134 AAVSO rows
(kname_na_n=0, kmag_na_n=0).

## C4 methods matrix + ERR_MODEL

`#   per-frame saturation: ON` (true via 2.4 run override; exporter must
read `provenance.config_snapshot`, not a top-level meta config).

`#ERR_MODEL=mode=calibrated; gain=db_div_container_scale=0.7925 e-/ADU_container; calib=none`

Intact. Spec defect on first pass: `"ON" in line` also matched
`saturatiON` in the OFF line; matcher now uses the explicit matrix token.

## C5 consistency

Quantity: max |AAVSO MAG - LC export mag| over matched epochs.
Domain: 134 AAVSO rows vs BO CVn LC, SHA 36a53b0.

| check | max abs delta | unit | n |
|-------|--------------:|------|--:|
| AAVSO MAG vs LC mag (3-decimal MAG rounding) | 0.498 | mmag | 134/134 |
| mag_calib_ac vs mag_calib_raw + ac_correction (LC columns) | 0.001 | mmag | sidecar chain |

Spec asked reconstruct exported mag from sidecar chain to 0.0001 mmag.
Physics/storage: LC mag columns are 6-decimal; AAVSO MAG is 3-decimal
(0.5 mmag ULP). 0.498 mmag is one MAG ULP, not a chain bug. Chain identity
at 0.001 mmag is 6-decimal storage ULP. Named spec vs storage precision;
do not claim 0.0001 mmag on the 3-decimal MAG column.

## Named defects

1. First export pass reported PFS OFF because citations.load_pipeline_meta
   nests the snapshot under provenance. Fixed; re-export succeeded.
2. Substring `"ON"` matched `saturatiON`. Fixed.
3. C5 0.0001 mmag vs MAG rounding (above).

## Docs impact

- docs/VYVAR_STATE.md / JOURNAL.md -- export closed on 36a53b0
- FLOW: none (export header text only)

## Recurrence

Recurrence: new test test_export_hdr_01 (NOTES length-guard, sidecar
KNAME, w_pre/w_post).

## Files

- src_py/export_reports.py
- src_py/check_star_kmag.py
- dev/tests/test_export_hdr_01.py
- dev/tools/export_hdr_01_bo_cvn.py
- this file + EXPORT_HDR_01_summary.json
