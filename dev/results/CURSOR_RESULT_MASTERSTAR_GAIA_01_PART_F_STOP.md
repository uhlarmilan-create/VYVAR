# MASTERSTAR-GAIA-01 Part F - STOP REPORT (E4)

Date: 2026-08-19  
Decision: Option 1 proceed (seed-pool gate OFF). Options 3/4 rejected. Option 2 deferred.

## F-pre: GREEN

Crossed `e3b_comp_survival.csv` with anchor `comp_qa_*.json` membership_ids for all **48** LC targets.

| Result | Value |
|--------|-------|
| Targets checked | 48 |
| All selected comps DETECTED_P1/P2 | **48/48** |
| Any SEED_REJECTED / FORCED_SEED in selected ensemble | **0** |

Raw: `session_20260819_msgaia01/f_pre_report.json`, `f_pre_selected_ensemble.csv`

## Rebuild attempt

Harness: `tmp/masterstar_gaia_01_part_f_rebuild.py`  
Runtime: ~2676 s (Phase 0 0.4 s, Phase 1 1999 s, Phase 2A 407 s) + MS rebuild ~57 s

Steps executed:
1. `generate_masterstar_and_catalog(skip_build=True, skip_solve=True)` - **MS catalog redetected with tightened pass2**
2. Zone re-annotation, comp pool regen, photometry wipe, Phase 0+1+2A

### Critical failure: MS catalog shrink

| Metric | Anchor (477dc8cf) | After MS rebuild |
|--------|-------------------|------------------|
| masterstars rows | 3621 | **2113-2114** |
| comparison_stars | 2356 | **1907** |
| LC targets | 48 | **45** |
| gaia_source_state_census.csv | n/a | **missing** (enrich did not write) |

Tightened pass2 (?=5, tol=2 px) removed ~1500 pass-2 detections from the MS build. F-pre validated **selected comp positions** still detect at Gaia seeds; it did not authorize replacing the full 3621-row MS catalog with a fresh redetection pass.

## E4: STOP (binding)

Compared 45 rebuilt LCs vs `draft_000516_snapshot_cleanrebuild_20260818` (477dc8cf):

| Gate | Result |
|------|--------|
| MAG byte-identical 48/48 | **FAIL** (0/45 identical; all show float deltas) |
| Missing LC vs snapshot | 4 targets (e.g. 1496733984545821696, 1497421866507840896) |

Sample MAG max_abs deltas (mmag-scale floats, not bitwise):

| Target | max_abs_mag |
|--------|-------------|
| HIP 67011 | 0.259 |
| HAT-188-0001788 | 0.257 |
| FU CVn | 0.207 |
| FY CVn | 0.170 |
| BO CVn | 0.071 |

Full delta list: `part_f_rebuild.json` ? `E4.mag_deltas`

**Per task spec: STOP. No anchor recut, no exports, no push.**

## E3a

Not evaluable - `gaia_source_state_census.csv` was not produced (enrich hook did not run on MS rebuild path).

## E5 (informational)

| Meter | Snapshot | Rebuild |
|-------|----------|---------|
| BO check MAD | 7.151 mmag | 6.608 mmag |
| FW check MAD | 8.201 mmag | 6.608 mmag |

## Recovery action taken

Live `draft_000516` **restored from** `draft_000516_snapshot_cleanrebuild_20260818` (477dc8cf product) after STOP to honor 'anchor untouched until F13 green'.

Verify: `compute_photometry_sha(draft_000516)` should return prefix `477dc8cf`, n=97.

## Architect decision needed

Option 1 + F-pre green implies **photometry inputs for the 48 LC ensembles are unchanged**. That is incompatible with **full MS redetection** under tightened pass2.

Recommended path for a retry:

1. **Keep anchor MS detection table** (3621 rows / lock-existing identities).
2. Apply **gaia accounting overlay only** (census CSV, source_state columns, no detection-row loss).
3. Re-annotate zones + regen pool + Phase 0+1+2A **without** `generate_masterstar_and_catalog` redetection.
4. Re-run E4 gate.

Alternative: accept MS membership change as a **new anchor product** (explicit Milan authorization) - E4 would not apply; new golden SHAs required.

## Not executed

- Anchor + P1 recut (516-04)
- `--fast` / `--full` session baseline
- Exports + SUBMIT-01
- Docs ledger update
- Push
