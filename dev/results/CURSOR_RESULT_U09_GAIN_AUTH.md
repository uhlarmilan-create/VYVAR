# CURSOR RESULT - U-09 DATE-OBS + GAIN-AUTH-VERIFY-01

Date: 2026-08-17
Compared with: FITS DATE-OBS / EXPTIME on draft 515 raw BO CVn frames vs VYVAR
jd_mid / exported BJD; and WIDE-ERR-04 g_pt=0.637 e-/ADU_container vs
36a53b0 export ERR_MODEL gain=db_div_container_scale=0.7925.
Code SHA 6b23633. Product SHA 36a53b0
(36a53b0cacd58e9fdb922726023e806dd6fbf5a42fb2caeb70f37187a822799d).
Push: NOT authorized. No code change (none authorized).

Premise (0.1): Part A compares capture DATE-OBS (UTC ISOT on real frames)
with the LC `jd` / AAVSO DATE columns. They differ by construction:
DATE-OBS is a header stamp; export is BJD_TDB of mid-exposure. Part B
compares the closed WIDE-ERR-04 photon-transfer gain (r=3.999 px, g_pt=
0.63707 e-/ADU_container, SHA da9cce4-era) with the authority actually
used in the 36a53b0 Phase 2A sidecar. Those two g_pt estimates are not
the same measurement: this rebuild's PT used r=2.499 px and was rejected
by the CI gate.

JSON: `dev/results/U09_GAIN_AUTH_summary.json`.

## Part A - U-09 DATE-OBS

### A1. Real frames (draft 515, 60 s)

Raw lights: 150 files
`Archive/Drafts/draft_000515/Raw/lights/NoFilter_60_2/BO_CVn_Light_*.fits`.
Calibrated: 150, same DATE-OBS. Aligned: 134 (LC sample).

| quantity | value | units | domain | SHA |
|----------|------:|-------|--------|-----|
| EXPTIME | 60.0 | s | raw BO CVn, all 150 | 36a53b0 product / frames 2026-04-23 |
| DATE-OBS[i+1]-DATE-OBS[i] median | 121.000 | s | 149 consecutive raw pairs | 36a53b0 |
| gap min / max | 120.943 / 121.280 | s | same | 36a53b0 |
| overhead = gap - EXPTIME median | 61.000 | s | same | 36a53b0 |
| n_gap < EXPTIME | 0 | count | 149 gaps | 36a53b0 |
| DATE-END / EXPMID / DATE-AVG | absent | - | all sampled headers | 36a53b0 |
| DATE-OBS FITS comment | UTC start date of observation | - | on-frame INDI comment | 36a53b0 |

Gaps table (first 6 of 149; full head in JSON):

| i | DATE-OBS[i] | DATE-OBS[i+1] | gap_s | EXPTIME_s | overhead_s |
|---|-------------|---------------|------:|----------:|-----------:|
| 0 | 2026-04-23T19:35:20.355 | 2026-04-23T19:37:21.355 | 121.000 | 60.0 | 61.000 |
| 1 | 2026-04-23T19:37:21.355 | 2026-04-23T19:39:22.355 | 121.000 | 60.0 | 61.000 |
| 2 | 2026-04-23T19:39:22.355 | 2026-04-23T19:41:23.365 | 121.010 | 60.0 | 61.010 |
| 3 | 2026-04-23T19:41:23.365 | 2026-04-23T19:43:24.355 | 120.990 | 60.0 | 60.990 |
| 4 | 2026-04-23T19:43:24.355 | 2026-04-23T19:45:25.355 | 121.000 | 60.0 | 61.000 |
| 5 | 2026-04-23T19:45:25.355 | 2026-04-23T19:47:26.333 | 120.978 | 60.0 | 60.978 |

Cadence is a metronome at EXPTIME + 61 s dead time (likely Ekos Capture
delay and/or plate-solve/save), not EXPTIME + USB readout only. All 149
gaps are >= EXPTIME: DATE-OBS timestamps do not overlap 60 s exposures.
Gap-only cannot separate start vs mid vs end (all three give the same
inter-stamp interval). Discriminator on these frames: the DATE-OBS card
comment names it as UTC start; DATE-END/EXPMID are absent.

Ekos/INDI capture logs: **not on this Windows PC**. StellarMate is a
separate Linux box; nothing under the draft 515 tree or a local kstars
log matched this night. Do not treat KStars documentation as A1. If a
log excerpt is supplied, the exact need is:

- `~/.local/share/kstars/kstars.log` (or Capture module log) around
  2026-04-23 19:35:20 UTC
- lines "Starting capture" / "Exposure complete" for
  `BO_CVn_Light_001.fits` vs DATE-OBS `2026-04-23T19:35:20.355`
- INDI `CCD_EXPOSURE` start timestamp for the same frame
- Ekos Capture **Delay** (s) that would explain the 61 s dead time

### A2. JD chain (file:line)

1. FITS DATE-OBS parsed as UTC ISOT start: `src_py/time_utils.py:62-85`
   (`mid_exposure_jd`).
2. EXPTIME/2 added: `time_utils.py:107-128`
   `t_mid = t_start + TimeDelta(exptime / 2.0 * u.s)`.
3. `compute_time_columns` (`time_utils.py:271-294`) writes `jd_mid`,
   then `compute_hjd_bjd` (`time_utils.py:141-168`) adds TDB + barycentric
   light-travel to produce `bjd_tdb_mid`.
4. Phase 2A LC: `photometry_core.py:3307-3310`
   `bjd` <- `bjd_tdb_mid`, `jd` <- `jd_mid`.
5. AAVSO/VarAstro DATE: `export_reports.py:1283` `#DATE=BJD` and
   `export_reports.py:1342-1349` writes `row["bjd"]` at 6 decimals.

EXPTIME/2 is added once, in `mid_exposure_jd`, not again at export.

Measured on 134 BO CVn LC rows matched
`proc_BO_CVn_Light_NNN.csv` -> raw `BO_CVn_Light_NNN.fits`, SHA 36a53b0:

| quantity | median | min | max | units |
|----------|-------:|----:|----:|-------|
| jd_mid - jd(DATE-OBS) | 30.000 | 30.000 | 30.000 | s |
| abs(LC jd - header mid) max | 0.00008 | - | - | s |
| bjd - jd_mid (TDB + LTT) | 368.840 | 368.594 | 369.087 | s |
| jd_export - jd(DATE-OBS) | 398.847 | 398.582 | 399.111 | s |

jd_export - jd(DATE-OBS) = EXPTIME/2 (30.000 s) + barycentric/TDB
(~368.84 s). The ~399 s figure is not a missing half-exposure. AAVSO
DATE is 6-decimal JD (ULP ~0.086 s), which accounts for the 0.007 s
median offset vs LC bjd.

### A3. Verdict: (a)

DATE-OBS on these frames is start-of-exposure (on-frame comment +
stable gap >= EXPTIME, 0 overlaps). VYVAR adds EXPTIME/2 in
`mid_exposure_jd`; export ships mid-exposure BJD_TDB.

U-09 CLOSED for this rig (NoFilter_60_2 / QHY294PROM / draft 515).
Other rigs remain unverified (LIMITATIONS).

Times-of-minimum: a missing EXPTIME/2 would be 30 s = 0.000347 d early
on 60 s frames, comparable to good amateur CCD ToM (often 10-60 s).
That bias is **not** present on this chain. Residual convention risk
without an Ekos log is start vs end; the on-frame comment names start.
Mid-exposure DATE-OBS is incompatible with that comment.

No code change. No export JD patch.

## Part B - GAIN-AUTH-VERIFY-01

### B1. Mechanism

Sidecar name: `gain_photon_transfer.json`
path: `.../photometry/gain_photon_transfer.json`
**exists** on the 36a53b0 tree.

Written by `apply_photometric_gain_authority`
(`src_py/gain_photon_transfer.py:326-388`) from Phase 2A
(`photometry_core.py:9655-9696`). PT estimate is
`estimate_photon_transfer_gain_from_proc_dir`
(`gain_photon_transfer.py:81-133`): Theil-Sen on **existing proc CSV**
rows with `aperture_r_px` within 0.05 of the requested radius (not a
live empty-aperture photometry of FITS at a pinned 4 px).

Authority (`resolve_photometric_gain`, `gain_photon_transfer.py:194-268`):
use g_pt if `ok` and `ci_width_factor = g_pt_ci_hi/g_pt_ci_lo <= 3.0`;
else `db_div_container_scale`.

Headless harness `dev/tools/draft_515_headless_phase012a.py:200-211`
calls `run_full_photometry_pipeline`. PT **did run** on this path.
Not a missing-sidecar skip.

36a53b0 sidecar (pipeline_meta git_hash `1a516c75`, stage phase2a):

| quantity | value | units |
|----------|------:|-------|
| g_pt (this run) | 1.4027 | e-/ADU_container |
| g_pt CI | [0.780, 4.853] | e-/ADU_container |
| ci_width_factor | 6.220 | hi/lo |
| n_frames | 134 | proc CSVs |
| PT aperture_r_px | 2.499 | px |
| authority | db_div_container_scale = 0.7925 | e-/ADU_container |
| native | 3.17 / scale 4.0 | e-/ADU native / container |

WIDE-ERR-03/04 (same draft, r=3.999 px): g_pt=0.63707, CI [0.443, 1.094],
ci_width_factor=2.468 < 3, authority=g_pt.

Why 2.499: Phase 2A requests PT **before** this-run `dynamic_params` is
written (`photometry_core.py:9687` vs `_build_phase2a_dynamic_params` at
`12506`). After 2A, meta `dynamic_params.aperture_r_px` is 3.999; the
sidecar still records 2.499 (leftover meta and/or faint-star scatter
rows). Proc frame 001 census: median aperture 1.999 px, 324 rows near
2.499, only 82 near 3.999. PT therefore fitted the small-aperture empty
sky (WIDE-ERR-03B B3 warned this widens CI). Gate fired; fallback is
the designed behaviour.

### B2. Photon-term impact

Empirical/Howell photon piece: `var = F/g + sigma_bkg_ap^2`
(`photometry_core.py:1272-1275`). Relative photon-only
`sigma = sqrt(F/g)/F`. Mag: `(2.5/ln(10)) * sigma`.

**Physics vs spec:** sigma_photon ? 1/sqrt(g). Larger g (0.7925 vs
0.6371) **under-quotes** the photon term. The task expected
sqrt(0.7925/0.6371)=1.115 over-quote; measured ratio
sigma(0.7925)/sigma(0.63707) = 0.8966 = sqrt(0.63707/0.7925).
Physics outranks the spec.

BO CVn, 134 rows, median dao_flux=1.1433e5 ADU, SHA 36a53b0:

| quantity | g_pt=0.63707 | g_fb=0.7925 | delta fb-pt | units |
|----------|-------------:|------------:|------------:|-------|
| photon-only median | 4.023 | 3.607 | -0.416 | mmag |
| empirical F/g+bkg median | 5.984 | 5.723 | -0.277 | mmag |
| LC err_photon (rel->mmag) | - | 5.723 | - | mmag |
| LC err median | - | 8.365 | - | mmag |
| AAVSO MAGERR median | - | 0.008 | - | mag |

LC err_photon matches the empirical term at g=0.7925 (5.723 mmag).
AAVSO MAGERR ULP is 1 mmag (3 decimals).

### B3. Recommendation: (c) other

Not (a): the harness did run PT.
Not (b): fallback is the CI gate working, but the radius was not the
WIDE-ERR-03B sky-dominated ~4 px; leftover `dynamic_params` / small
proc-CSV apertures caused a wide CI.

Proposed fix (no implementation without go):

1. Pin PT radius at 4.0 px (or the post-2A production median ~3.999),
   do not read leftover `dynamic_params.aperture_r_px` before apertures
   exist.
2. Optionally move PT to after `state.apertures_px` is known.
3. Re-export ERR only (mag byte-identity expected).

AAVSO submit: **MAG and times are submit-worthy as-is** (U-09 closed;
mid-exposure BJD). MAGERR is slightly anti-conservative on the photon
piece vs closed g_pt (0.42 mmag photon-only median, 0.28 mmag with
bkg), below the 1 mmag AAVSO MAGERR ULP, and the ERR_MODEL header
honestly names `db_div_container_scale=0.7925`. Waiting for a g_pt
re-export is hygiene for the authority story, not a MAG/time blocker.
If the submit rule is "WIDE-ERR-04 g_pt on the header", wait.

## Named defects (this spec / this product)

1. Spec B2 direction: expected over-quote ~sqrt(g_fb/g_pt); production
   var=F/g under-quotes when g_fb > g_pt.
2. Consecutive BO CVn DATE-OBS gap is 121 s, not EXPTIME+readout, because
   of 61 s dead time. Still the >= EXPTIME signature.
3. 36a53b0 PT radius 2.499 px vs WIDE-ERR-03B ~4 px (leftover meta /
   small-aperture proc rows). CI gate then correctly fell back.
4. LIMITATIONS previously said U-09 "verified" while audit_stage2 still
   said QHY shutter-open vs readout UNVERIFIED. This measurement closes
   the home-rig submit prerequisite; other rigs stay unverified.

No production code changed.

## Docs impact

- docs/VYVAR_DECISIONS.md -- U-09 CLOSED (a); GAIN-AUTH-VERIFY-01 (c)
- docs/VYVAR_LIMITATIONS.md -- U-09 515 numbers
- docs/VYVAR_ROADMAP.md -- U-09 no longer an export prerequisite
- docs/VYVAR_STATE.md / JOURNAL.md -- this result
- docs/VYVAR_AUDIT_CLOSURE_REGISTER.md -- U-09 CLOSED
- FLOW: none (no code)

## Recurrence

Recurrence: existing WIDE-ERR-03B B3 (PT must use sky-dominated ~4 px);
U-09 n/a (verification, not a bug-class).

## Files changed

- `dev/results/CURSOR_RESULT_U09_GAIN_AUTH.md`
- `dev/results/U09_GAIN_AUTH_summary.json`
- docs listed above
- scratch only: `tmp/u09_gain_auth_measure.py`, `tmp/u09_a2b2_measure.py`

## --fast

`python dev/scripts/session_baseline_check.py --fast` on SHA **6b23633**:
**OVERALL PASS**. pytest 1443 passed, 28 skipped. P1 env unset.
Elapsed 368.5 s. No production code changed.
