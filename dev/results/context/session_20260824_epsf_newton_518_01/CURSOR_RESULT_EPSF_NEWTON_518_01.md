CURSOR RESULT - 2026-08-24T18:45:00Z (EPSF-NEWTON-518-01)

What I did
Cross-rig ePSF validation on Newton draft 518. STOP at N2: Part C gated
pool is 26 stars, below `epsf_min_stars=30`. No ePSF write, no 518 PSF
merge, no 516 touch. EPSF-ZP-OK-01-WIRE stays parked. Chi2 and flux
reported separately (none measured on 518 PSF - there is no PSF). Not
pushed.

Premise (0.1): SHAPE-01-F / AC-01 / PIN-CENSUS-01 predictions P-A..P-E
were written for a well-sampled Newton (~0.65 arcsec/px unbinned). Draft
518 is the Newton TOI-1131 night that exists on disk. It is bin2
(~1.30 arcsec/px). Those two sampling states are not the same. This
task compares the predictions to what 518 can actually host. The
gated-pool floor is production `_epsf_prepare_stars` / `epsf_min_stars=30`,
not a Godden table rewrite.

## Gates

| Gate | Status | Detail |
|------|--------|--------|
| G0 census commit | PASS | `2926a9523de814a8272e624bdfd1e0156fceac63` PIN-CENSUS-01 harness/result/artifacts. Rig-tag correction: 516 = WIDE Carl-Zeiss 200 mm / QHY294MM, not Newton. STATE/ROADMAP/JOURNAL census one-liners left uncommitted (mixed with dirty AC-02 docs). AC-02 code left unstaged. |
| G1 tip | PASS | `2926a95` is a descendant of `876053a`. |
| G2 516 untouched | PASS (science path) | Production ePSF SHA `172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20` unchanged. 188 hashed 516 products before N2: 0 mismatches. After `--fast`, 1 file differed: `lightcurve_1498613634033133184_psf.csv` (T1 live-516 writer; same watch as AC-02). Aperture LCs, AAVSO, VarAstro, ePSF FITS+meta unchanged. This task did not write 516. |
| G3 `--fast` | see close | Run at task end. |

## N0 - draft 518 inventory (prerequisites present; no STOP here)

Rig from headers + manifest (no TELESCOP/INSTRUME cards):

| item | value |
|------|-------|
| draft | 518, TOI-1131 field, RA 248.83 Dec +61.61 |
| calibration | `pre_calibrated`, status PROCESSED, is_calibrated=0 |
| filter / exp | V / 60 s (`V_60_2`) |
| binning | XBINNING=YBINNING=2 |
| detector size | 3126 x 2088 (C3-26000 bin2 envelope; unbinned 26 MP is not this night) |
| plate scale | 1.301 arcsec/px (raw CDELT/CD) |
| gain authority | header 3.12 e-/ADU (`gain_photon_transfer.json`; g_pt failed n_frames=0). Not 1.0, not wide g_pt. |
| CCD-TEMP | -9.91 C (science light) |
| pixel size card | XPIXSIZE=10.0 um (header; not used as saturate/gain) |
| location/equip | manifest location_id=1, equipment_id=2, telescope_id=2 |

FWHM (this drives sampling claims):

| source | FWHM px |
|--------|---------|
| draft_manifest inspection median / p16 / p84 (n=78) | 4.004 / 3.896 / 4.142 |
| science-light `VY_FWHM` | 4.027 |
| MASTERSTAR `VY_FWHM` | 4.019 |
| MASTERSTAR `VY_FWHM_GAUSS` (production `header_core_fwhm_px` first pick) | **2.2919** |

Seeing from inspection FWHM x scale: 4.004 * 1.301 = **5.21 arcsec**.
osamp=2 gridpoints per FWHM: 8.01 at inspection 4.00 px (passes >=4);
4.58 at Gauss 2.29 px (barely >=4). Production cutout would be
`_to_odd_cutout(int(2.2919*5))` = **15** (wide production is 17).

Counts: 78 raw lights, 71 science aligned FITS + MASTERSTAR, 71 proc
CSV, 10 aperture LCs, 0 PSF LCs, 0 ePSF FITS. `aperture_snr_table`
absent; `aperture_snr_table_REJECTED.json` present (ERR-518 global_fixed
path). INV-WCS-01 WARN in pipeline_meta: matched world2pix p95=44 px.

Saturate: sat_diag header/equipment 65535, source CONFLICT_DERIVED,
warning "no raw light frames found". Sidecar `saturate_limit_adu=65535`.
`pipeline_meta` has `saturate_limit_fraction=0.8` (0.8*65535=52428, the
same number as the QHY 85% product but **derived from this 16-bit
container**, not copied as a QHY constant). Comp `peak_max_adu` p99 =
88300, above 65535, because the frames are float pre-calibrated.
N1 did not re-derive saturate (aperture branch already present).

Field/targets: 20 active_targets (10 skip_photometry, mostly VSX ROT
out of scope). 10 aperture LCs including ZTF J163831.02+614701.9
(XVAL target), TIC 198213332 / TOI-1131.01, GH Dra, CzeV4348. Gaia
catalog_rows=26504, n_gaia_detected=923 (completeness 3.48%). VSX
present in active_targets.

Ensembles: `pinned_ensembles_enabled=true` but the pin file is the WIDE
ERA-03 48-target list (`dev/validation/pinned_ensembles.csv`). No 518
Gaia IDs in that file. Internal-LC resolver would use
`comparison_stars_per_target` (65 rows). Per-target comps exist; they
are not the 516 pin set.

ePSF science set: n_total=47 (20 targets + 27 per-target comps).

Aperture branch is present and photometry-ready. N0 does not STOP.

Elapsed N0: 1.5 s inventory + 3.0 s prepare funnel.

## N1 - aperture baseline (already present; not re-run)

Skipped full canonical rebuild: Phase 2A LCs exist (10). Recorded
product SHAs in `n1_aperture_lc_hashes.json`. pipeline_meta already
has `export_err_mode=calibrated`, `err_background_mode=empirical`.
PFS: platesolve bundle present (`PLTSOLVD=true`). No 518 aperture
rewrite. No 516 rewrite.

## N2 - gated ePSF build: STOP

Same Part C gates as production (`_epsf_apply_build_selection_gates` +
`_epsf_prepare_stars`). Funnel:

| gate | n |
|------|---|
| csv input | 1215 |
| variable excluded | 1194 |
| not likely_saturated / not is_saturated | 790 |
| photometry_ok / not noisy / usable | 790 |
| zone linear | 790 |
| clean source_state | 492 |
| **science scope** | **26** |
| edge-safe cutout | 26 |
| interim top-N (200) | 26 |
| final | 26 |

`epsf_min_stars=30`. Found 26. `build_epsf_model` raises before
extract_stars. No `masterstar_epsf.fits` written.

Choke is science_scope: 47 science-set IDs minus VSX/Gaia variables
leaves 26 non-variable comps in scope. Isolation never ran.

Godden & Blundell o=2 M=6 95% ~48 stars: 26 is below that too. This is
a pool-size STOP, not a census failure.

## N3 / N4 - not run

No model => no F6 merge, no internal PSF LCs, no chi2-vs-mag, no
PSF/DAO, no ePSF norm audit, no pin-census, no LC meters.

## N5 - predictions and STOP

| id | prediction | measured | verdict |
|----|------------|----------|---------|
| P-A | bright chi2 median ~1-3, not ~68 | not measured (no ePSF) | **FAIL to test** |
| P-B | PSF/DAO vs mag flatter than wide 1.27->2.21 | not measured | **FAIL to test** |
| P-C | signed negative fraction smaller than wide -0.302 | not measured | **FAIL to test** |
| P-D | strict `psf_fit_ok` coverage high; interim ZP not needed | not measured | **FAIL to test** |
| P-E | FWHM ratio closer to 1.0 than wide 0.716 | not measured (no model). Context Gauss/inspection = 2.292/4.019 = 0.570 on MASTERSTAR header, which is **not** an ePSF native/data ratio | **FAIL to test** |

Sampling mismatch vs the SHAPE-01-F Newton paragraph: that paragraph
assumed ~0.65 arcsec/px unbinned, FWHM ~4.6 px at 3 arcsec seeing.
Draft 518 is **bin2 1.30 arcsec/px**, inspection FWHM ~4.0 px, seeing
~5.2 arcsec. Even if the pool had passed, this night is not the
unbinned well-sampled case those sentences named.

### Is interim `psf_fit_ok_for_zp` needed on Newton?

Unknown. P-D was the deciding meter and it was not taken.

Recommendation for parked EPSF-ZP-OK-01-WIRE: **(ii) revise**. Do not
GO the interim as "wide-rig mitigation with Newton evidence that strict
suffices there". 518 provided no such evidence. Combined with
CENSUS-01 (WIDE 516: 100% of pin drops are stored chi2>=50 and
admitting them holds quality): the wide-rig measurement still stands on
its own, but Newton 518 is not a cross-rig confirmation. Keep ZP-OK
parked until either (a) a Newton draft builds a gated ePSF and P-D is
measured, or (b) Milan GOs the interim as **wide-rig-only** without
claiming Newton.

Do not lower `epsf_min_stars` to 26 to force a 518 build without a
separate GO. That would not be the production Part C gate.

### SHAPE-01-F "well-sampled rig" paragraph

Updated in `CURSOR_RESULT_EPSF_SHAPE_01_F.md` from prediction-only to:
518 is the on-disk Newton draft; it is bin2 1.30"/px; gated pool STOP;
P-A..P-E unmeasured. Pointer to this result.

### EPSF-CORE-01

Acceptance coverage (WIDE strict predicate) unchanged: BO 23/134, FW
0/134. Newton 518 is **not** yet a cross-rig ePSF baseline. Add: gated
pool 26<30 on this night; science_scope choke; bin2 1.30"/px.

## Files touched

This task:
`dev/scripts/epsf_newton_518_n0_inventory.py`,
`dev/results/CURSOR_RESULT_EPSF_NEWTON_518_01.md`,
`dev/results/CURSOR_RESULT_EPSF_SHAPE_01_F.md` (Newton paragraph),
`docs/VYVAR_STATE.md`, `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_JOURNAL.md`.

G0 already committed: `dev/scripts/epsf_pin_census_01.py`,
`dev/results/CURSOR_RESULT_EPSF_PIN_CENSUS_01.md`,
session/context `session_20260824_epsf_pin_census_01/`.

No production code. No 516 bytes. No 518 ePSF/proc/LC rewrite.

Artifacts: `dev/results/session_20260824_epsf_newton_518_01/` and
context copy `dev/results/context/session_20260824_epsf_newton_518_01/`.

## Errors (if any)

N2 `ValueError`: EPSF build needs at least 30 clean stars; found 26.
Expected STOP, not a crash.

## `--fast` close

PASS: 1524 passed, 32 skipped, OVERALL PASS on HEAD `2926a95` (declared-dirty
AC-02 tree). Watch `test_comp` did not appear as a failure. git-origin-main
WARN: local series ahead of origin (do not pull/push). db-quick-check WARN
via committed waiver. Elapsed 600 s. T1 rewrote BO CVn PSF LC on live 516
(expected; not this task's writer).
