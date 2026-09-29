CURSOR RESULT - 2026-09-29 ERA-520-RECUT-01

Date: 2026-09-28/29. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: tip after LEDGER-EPSF-DOD-06.
Class: Stage 3/4 - numbers move by design; era05 cut.
Milan GO 2026-09-28. Push: origin consolidate-01 only.

Convention cited: docs/VYVAR_PROCESS.md (Anchor re-cut /
SNAPSHOT_NAME) + naming draft_000516_snapshot_eraNN_YYYYMMDD
(era04 precedent draft_000516_snapshot_era04_20260826).

## What I did

Landed closed-list items 1,2,3,5,6,4 (one commit each + G2 after
each), cut era05 from G2 work-copy, regenerated aperture/ext/epsf01
hashes beside era04 history, re-checked VAL-02 (c1) and AC-02 (c2),
enacted D-EPSF-XVAL-CLOSE-TEXT-02. Hardened GAIN null-g_pt after
first --full-epsf STOP. Pushed consolidate-01 only.

## Per-item table (G2 vs era04 aperture d55fcc9d / ext cc8b532e)

| item | commit | expected | observed | work-copy |
|---|---|---|---|---|
| 1 FIXPOS-NOOP-01 | 9e8dc91 | byte-identical | PASS core=d55fcc9d ext=cc8b532e | 20260928T140412Z |
| 2 FIT-OK-ADMISSION-01 | 4d365e7 | aperture identical (PSF-LC only) | PASS aperture identical | 20260928T144418Z |
| 3 GAIN-FALSY-01 | 1efc08c | aperture identical | PASS aperture identical | 20260928T152632Z |
| 5 skip_reason | 05da165 (+3a55065 test) | LC hashes move; values identical after drop | FAIL hashes (expected); value-ident 53/53 after drop skip_reason | 20260928T160546Z core=b48ce82b ext=b987ace5 |
| 6 lc_flux_method | cede01d | LC hashes move; values identical after drop | FAIL hashes (expected); value-ident after drop | 20260928T163750Z core=d9968b47 ext=e723fd6c |
| 4 RED-TARGET-T4 | d79945f | only the two T4 targets move | HAT-188 LC only moved (52/53 byte-ident vs item6); CV CVn has no science LC (skip class=other n_rows=0) | 20260928T171821Z core=87197716 ext=dd92e99d |

Follow-up: 0708abe GAIN null authority.g_pt harden (era freeze
sidecars store g_pt=null; float(None) TypeError on --full-epsf #1).
dd76070 points SNAPSHOT_NAME at era05 aperture anchors.

## T4 micro-measurement (RMS-first vs colour-first)

Targets: CV CVn 1497007144465726080, HAT-188 1497683996951418880.
Both fire color_rms_t4_fallback under T4; both used color_rms_wide
under RMS-first (item6).

| target | metric | RMS-first | colour-first | helps? |
|---|---|---:|---:|---|
| HAT-188 | med \|dBP-RP\| of 8 comps | 1.489 | 1.063 | YES |
| HAT-188 | med comp_rms | 0.0167 | 0.0190 | tradeoff (by design) |
| HAT-188 | LC mag_calib_final RMS (variable) | 2270.8 mmag | 2271.4 mmag | n/a (EB amplitude) |
| CV CVn | med \|dBP-RP\| | 2.031 | 1.605 | YES |
| CV CVn | science LC | absent | absent | n/a |

Reading: colour-first HELPS colour match on both; does not merely
permute. D-RED-TARGET-T4-01 reland COMPLETE with measurement.

## era05 cut and delta report

Path: Archive/Drafts/draft_000516_snapshot_era05_20260928
(from tmp/session_baseline/20260928T171821Z aperture; PSF LCs from
20260928T183918Z --full-epsf). era04 freeze kept on disk. Live 516
G4 still bfa24039 / 13e77cf8 / 172f9540 (read-only).

| anchor | era04 (history) | era05 (new) | n |
|---|---|---|---:|
| core_aperture | d55fcc9d... | 87197716... | 53 |
| ext_aperture | cc8b532e... | dd92e99d... | 157 |
| core_psf (epsf01) | c743b8ba... | 552ace75... | 53 |

Aperture delta era04 -> era05 (53 science LCs):
- 52/53 schema-only (skip_reason + lc_flux_method); value-identical
  after column drop. Attributed: items 5+6.
- 1/53 HAT-188 also value-moved (mag_calib* / delta_mag / err;
  max |delta| mag_calib_final=0.944 mag). Attributed: item 4.
- Items 1-3: no aperture movement (as EXPECTED).

## Criterion 3 unit tests

| defect | test | pre-fix fail / post-fix pass |
|---|---|---|
| FIXPOS-NOOP-01 | test_fixpos_noop_01.py | yes (landed with 9e8dc91) |
| FIT-OK-ADMISSION-01 | test_psf_internal_lc.py::test_fit_ok_admission_01_for_zp_honours_fit_ok_false | yes (4d365e7) |
| GAIN-FALSY-01 | test_gain_falsy_01.py (+ null g_pt case in 0708abe) | yes |

## Re-check (not re-registration)

VAL-02 on era05 products (tmp/_era520_val02_recheck.py;
session_20260928_era520_val02_recheck/summary.json):
- R-W1 PASS: domain D median r=1.049 max 1.415 (n=37)
- R-W2 PASS: n_method_psf=0; lc_flux_method on 53/53 (VERIFIABLE)
- R-W3 informational FAIL under DOD-03 wording (superseded by
  DOD-05 2a/2b; not judged for closure)
- criterion 1 PASS

AC-02 (same thresholds DOD-05/06; VAL-03 products):
- R-AC1 PASS |b|=2.678; R-AC2 PASS odd/even 5.594 / chrono 6.604
- criterion 2 PASS

## --full-epsf (20260928T183918Z)

- full-pipeline PASS 1734s; full-epsf-stage PASS n_stars=64 wrote 53
  PSF LCs in 49877s
- aperture SHA PASS era05 87197716 / dd92e99d
- core_psf run 552ace75 (recorded as era05 epsf01)
- full-g3-residual FAIL: BO n_full=1 FW n_full=0
  Cause: FIT-OK-ADMISSION-01 + INV-PSF-LC-PIN-01. Pinned ensemble
  comps with fit_ok=False (e.g. 1499200223486564608 sample rate 0/20)
  NaN every epoch under full-pin membership. era04 G3 refs
  12.505/4.629 n_full=134 kept as history. G3 meter rewrite is
  Milan's (not invented here). OVERALL FAIL on G3 only.

## Readings

- R-E1: items 1-6 landed with EXPECTED movement; criterion-3 tests
  pass -> criterion 3 met.
- R-E2: re-check c1+c2 PASS -> ENACT D-EPSF-XVAL-CLOSE-TEXT-02;
  T4 micro HELPS; D-RED-TARGET-T4-01 reland COMPLETE.
- G3 FAIL recorded (not a criterion 1-3 FAIL; sequels Milan's).

## Errors on the record

1. First --full-epsf FAIL INV-PSF-FRAME-01: float(None) on
   authority.g_pt=null in era freeze sidecars. Fixed 0708abe.
2. G3 n_full collapse post FIT-OK (see above).

## Files changed (high level)

- src_py: psf_photometry, psf_internal_lc, photometry_lightcurve,
  photometry_comp, phase2a_target, comp_selection_per_target, ...
- dev/tests: test_fixpos_noop_01, test_gain_falsy_01,
  test_psf_internal_lc FIT-OK, skip_reason / T4 tests
- dev/scripts/session_baseline_check.py (era05 SNAPSHOT_NAME + hashes)
- dev/validation/VYVAR_VALIDATION_LEDGER.json
- docs: STATE, ROADMAP, JOURNAL, DECISIONS (CLOSE-TEXT-02 enacted)
- Archive (not git): draft_000516_snapshot_era05_20260928

## Docs impact

STATE / ROADMAP / JOURNAL / DECISIONS updated. PROCESS convention
cited (no text change required).

## Recurrence

new test test_gain_falsy_01_null_g_pt_falls_to_container_scale |
existing test_fixpos_noop_01 / test_fit_ok_admission_01 /
test_gain_falsy_01_uses_g_pt_not_config_default

## STOP

After R-E2 enact. Main push, ens4 pin, PIN-RMS-ABORT-01, upload
release remain Milan's. G3 meter rewrite remains Milan's.
