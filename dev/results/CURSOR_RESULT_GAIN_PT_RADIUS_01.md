# CURSOR RESULT - GAIN-PT-RADIUS-01 + SUBMIT-01

Date: 2026-08-17
Compared with: GAIN-AUTH-VERIFY-01 (36a53b0 fallback at r=2.499,
authority db_div_container_scale=0.7925) vs pinned r_pt=4.0 rebuild
(authority g_pt). MAG columns must stay byte-identical; err may rise.
Code tip before commits: 6b23633. Push: NOT authorized.

Premise (0.1): leftover `dynamic_params.aperture_r_px` is a previous-run
photometry radius, not a PT empty-aperture radius. Pinning r_pt=4.0
ignores that leftover. WIDE-ERR-04 g_pt=0.63707 at r~4 is the comparable
authority; the 2.499 fit is not.

JSON: `dev/results/GAIN_PT_RADIUS_01_summary.json`.
Tool: `dev/tools/gain_pt_radius_01.py`.
Test: `dev/tests/test_gain_pt_radius_01.py`.

## Part A - GAIN-PT-RADIUS-01

### A1. Mechanism (chosen: pin 4.0)

`resolve_photon_transfer_aperture_r_px()` in
`src_py/gain_photon_transfer.py` always returns
`(4.0, "pinned_sky_dominated_4px")`. Leftover meta is ignored.
`force_aperture_px` no longer overrides PT (star photometry only).
Phase 2A calls the resolver (`photometry_core.py` gain block).
Sidecar fields: `aperture_r_px` + `aperture_r_px_source`.
No new config key (registry stays 291).

### A2. Fire proof

| test | result |
|------|--------|
| (a) `legacy_pt_aperture_from_leftover_dynamic_params({2.499})` -> 2.499 | PASS |
| (b) `resolve_photon_transfer_aperture_r_px({2.499})` -> 4.0 | PASS |
| CV-like leftover `{1.999}` ignored by resolver | PASS |

`pytest dev/tests/test_gain_pt_radius_01.py` + `test_wide_err_03_gain.py`:
11 passed.

### A3. PT on draft 515

| quantity | value | units | domain |
|----------|------:|-------|--------|
| r_pt | 4.0 | px | pinned_sky_dominated_4px |
| g_pt | 0.6370667331227862 | e-/ADU_container | 134 proc frames |
| CI | [0.44338, 1.09419] | e-/ADU_container | same |
| ci_width_factor | 2.4678353116926237 | hi/lo | < 3.0 gate |
| authority | g_pt | - | sidecar |
| abs delta vs WIDE-ERR-04 g_pt | 0.0 | e-/ADU_container | byte-identical |

CI gate PASS. No STOP. No force.

## Part B - ERR-only re-export

Phase 2A elapsed: 789.8 s. PFS ON (per-run override). 48 LCs.

| check | result |
|-------|--------|
| MAG byte-identity vs pre-rebuild backup | **48/48** PASS |
| BO mag_calib / delta_mag / bjd byte-equal | PASS |
| `#ERR_MODEL` | `gain=g_pt=0.6371 e-/ADU_container` |
| `test_export_hdr_01` | 5 passed |

| quantity | before (36a53b0) | after | units |
|----------|-----------------:|------:|-------|
| BO LC median err | 8.365 | 8.945 | mmag |
| BO delta | - | **+0.580** | mmag |
| FW LC median err | 7.010 | 7.420 | mmag |
| AAVSO MAGERR median | 0.008 | **0.009** | mag |
| check MAD (mag_calib-kmag) BO | 194.284 | 194.284 | mmag (delta 0) |
| check MAD FW | 14.014 | 14.014 | mmag (delta 0) |

Spec expected ~+0.18 mmag on the GAIN-AUTH recombined estimate; measured
LC median err +0.580 mmag. Physics/measurement outranks the estimate
(named defect). Direction is conservative (house rule).

### B2. Photometry SHA

| | value |
|--|------|
| previous product SHA prefix | 36a53b0 |
| new core SHA | **de6f7c8155d141376cf6df895144873f470555c5bb2de426ddad5b46cd981301** |
| prefix | **de6f7c8** |
| n core files | 97 |

SHA changed (err columns in LC core fileset). Mag-based meters are
byte-unaffected (asserted). METER-DRAFT-DEP-01: any later err-sensitive
meter must cite **de6f7c8**, not 36a53b0.

## Part C - SUBMIT-01 checklist

All lines PASS on
`.../aavso/BO_CVn_20260423.txt` and
`.../varastro/BO_CVn_20260423.txt`.

| id | check | result |
|----|-------|--------|
| C1 | OBSCODE=UMIA | PASS |
| C1 | TYPE + SOFTWARE VYVAR/1.0 | PASS |
| C1 | band CV | PASS |
| C1 | KNAME/KMAG from check 1497613731286514432, 0 na | PASS |
| C1 | NOTES n_comp=4 GaiaDR3 ensemble, no truncated IDs | PASS |
| C1 | DATE=BJD (U-09 mid-exposure) | PASS |
| C1 | MAGERR 3-decimal; 134 rows; ERR_MODEL g_pt | PASS |
| C2 | VarAstro w_pre/w_post; 134 epochs | PASS |
| C3 | mag depth 0.470; quiet median 9.462; no mandatory na | PASS |
| C5 | Do not submit (Milan only) | noted |

### C4. Submit note (Slovak, for Milan)

Nahraj `BO_CVn_20260423.txt` (AAVSO Extended) do AAVSO WebObs pod kodom
**UMIA**, filter **CV**, 134 bodov, BJD. VarAstro subor rovnaky night do
var.astro.cz. Dataset: BO CVn 2026-04-23, 60 s NoFilter, VYVAR aperture
ensemble, g_pt=0.637 e-/ADU, mid-exposure BJD. Manualne - agent
nesubmituje.

## Named defects

1. Spec B1 ~+0.18 mmag recombined estimate vs measured LC median err
   +0.580 mmag (AAVSO MAGERR 0.008 -> 0.009). Report the measured value.
2. GAIN-AUTH-VERIFY left the pin unimplemented; this task closes it.

## Docs impact

- docs/VYVAR_DECISIONS.md -- GAIN-PT-RADIUS-01; GAIN-AUTH CLOSED
- docs/VYVAR_ROADMAP.md / STATE.md / JOURNAL.md -- product SHA de6f7c8;
  SUBMIT-01 ready
- FLOW: none (no new param)

## Recurrence

Recurrence: new test test_gain_pt_radius_01 (legacy hole + pin);
existing WIDE-ERR-03B B3 class.

## Files changed

- `src_py/gain_photon_transfer.py`
- `src_py/photometry_core.py`
- `dev/tests/test_gain_pt_radius_01.py`
- `dev/tools/gain_pt_radius_01.py`
- `dev/results/CURSOR_RESULT_GAIN_PT_RADIUS_01.md`
- `dev/results/GAIN_PT_RADIUS_01_summary.json`
- docs listed above
- on-disk draft 515 photometry (err + sidecar + BO exports; SHA de6f7c8)

## --fast

`python dev/scripts/session_baseline_check.py --fast` on tip **6b23633**
(pre-commit): **OVERALL PASS**. pytest 1447 passed, 28 skipped
(+4 from test_gain_pt_radius_01). P1 env unset. No push.
