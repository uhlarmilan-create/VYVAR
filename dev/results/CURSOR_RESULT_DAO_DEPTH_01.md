CURSOR RESULT - DAO-DEPTH-01 - 2026-08-14

Register ID: DAO-DEPTH-01
Scope: report only. No production code or parameter changes.
Artifacts: `dev/results/DAO_DEPTH_01_ledger.csv` (and draft_512 / draft_435 splits),
`dev/results/DAO_DEPTH_01_snr_replay_512.json`.

================================================================
1. Architecture as implemented (Section 2)
================================================================

Milan's two-pass sketch is close but incomplete. Corrected chain:

### A. MASTERSTAR catalog build (`detect_stars_and_match_catalog` on MASTERSTAR.fits)

1. **Catalogue query.** Local Gaia SQLite cone/box via `_query_gaia_local`, written as
   `field_catalog_cone.csv`. Cap = max(max_catalog_rows, 50000, catalog_query_max_rows)
   typically **100000 brightest**. Draft 512 meta: cone radius **13.635 deg**, center near
   RA 209.48 Dec +41.16. Mag limit for MASTERSTAR path floored at **18.0** (DB max G=17.5
   applied). This is NOT a forced-photometry input list of "all Gaia in the chip"; it is a
   wide cone brightest-N table.

2. **Pass 1.** `photutils.detection.DAOStarFinder` on the MASTERSTAR image (optional mean
   binning). Threshold from convolved background RMS x `masterstar_dao_threshold_sigma`
   (and `dao_detection_n_equiv`). FWHM from `dao_detection_fwhm_pixels(header)` (VY_FWHM /
   config fallback).

3. **Pass 2 input list.** Gaia rows from that cone table projected into the frame that have
   **no pass-1 DAO neighbor** within the catalog match radius. Not "all Gaia", not
   masterstars from pass 1. Size in draft 512 log: **3681** unmatched Gaia positions.

4. **Pass 2 "signal" test.** Not forced aperture flux. For each unmatched Gaia position:
   take a ~21x21 cutout; local annulus std; run DAOStarFinder with
   `threshold = masterstar_dao_pass2_sigma * local_std` (default **1.9**); keep if a
   detection falls within **5 px** of the Gaia pixel. Empty cutouts counted (512: 25/3681).

5. **Merge + spatial flux cap** (max_catalog_rows; not binding at 100000 here).

6. **Prematch SNR peak filter (critical).** Before catalogue matching, drop detections whose
   box peak is below `median + k * std` with
   `k = masterstar_prematch_peak_sigma_floor` (**1.8** on both drafts) and `std` from
   `plain_mean_med_std` on the **full frame**. Draft 512 log:
   **735/3621** kept, noise_floor~**2427** ADU.

7. **Match to Gaia**, assign `source_type = GAIA_MATCHED` if `catalog_id` else `DAO_ONLY`.
   Astrometry optimizer may rematch; DAO_ONLY kept in `masterstars_full_match.csv`.

### B. Per-frame proc CSV (MASTERSTAR lock)

Both drafts logged: `Per-frame catalog: MASTERSTAR lock enabled (... masterstars_full_match.csv)`.
Fast path calls `detect_stars_match_master_reference`: DAO on the frame, then match to the
**locked masterstar list**. Rows without a master match do not become GAIA_MATCHED proc
rows. The masterstar list therefore **upper-bounds** catalogue depth for every frame.

### C. Labels

| Label | Meaning in code | Why UI shows 0 |
|-------|-----------------|----------------|
| `GAIA_MATCHED` | Non-empty `catalog_id` after match | Present (704/2590) |
| `DAO_ONLY` | Detection with empty `catalog_id` | Present on MASTERSTAR (12/109); **dropped** for photometry by `_proc_drop_unmatched_dao_rows` / keep-matched (TODO-13). Proc CSV therefore has 0. |
| `FORCED_APERTURE` | Checked in UI/plot helpers | **Never assigned** by current writers. Zero means the label is not produced, not that forced photometry ran and found none. |

Pass 2 is **targeted re-detection**, not classical forced photometry at fixed coordinates.

================================================================
2. Comparability drafts 435 vs 512 (Section 4)
================================================================

| Quantity | draft 435 | draft 512 | Notes |
|----------|-----------|-----------|-------|
| Calibrated lights | 150 files, same names | 150, same names | Same DATE-OBS per paired file |
| Grid | 2082 x 1397 | same | XBINNING=YBINNING=2 |
| EXPTIME | 60 s | 60 s | |
| GAIN header | 0.0 | 0.0 | |
| Calibrated pixels | -- | -- | **NOT byte-identical** (`allclose` False on Light_001/004) |
| MASTERSTAR DATE-OBS | 2026-04-23T19:49:27 | 2026-04-23T23:13:08 | **Different source frame** |
| MASTERSTAR VY_FWHM | 3.207 px | 5.195 px | |
| MASTERSTAR VY_FWHM_GAUSS | 2.395 | 3.3014 | |
| `masterstar_dao_threshold_sigma` (snapshot) | **2.1** | **3.8** | Joint-effect contributor |
| `masterstar_prematch_peak_sigma_floor` | 1.8 | 1.8 | Same k |

D-R0: every 435-vs-512 number below is **descriptive** (months of code + config + calibration).

Controlled comparison: draft 512 products vs independent detection on
`draft_000512/.../BO_CVn_Light_004.fits` (same pixels SIPS used).

================================================================
3. M1 ledger -- draft 512 (reconciling)
================================================================

ALL-mag (from draft 512 infolog + artifacts; SNR mag bins from MASTERSTAR replay):

```
stage                              in      out     lost   reason
catalog_cone_query                   -   100000      0   brightest cap
gaia_local_db_footprint_G<=17.5      -    12579      0   true in-chip DR3
gaia_cone_in_footprint_G<=17.5  100000     4130  95870   off-chip / cone incompleteness
dao_pass1                              -      307      0   VY_NDAO=307
dao_pass2_input (unmatched Gaia)       -     3681      0
dao_pass2_completions               3681     3314    367   no local DAO / center_tol
dao_merged (+spatial)                  -     3621      0   cap not binding
prematch_snr_peak_filter            3621      735   2886   peak < med+1.8*full_std
masterstars_full_match               735      735      0   GAIA_MATCHED=723 DAO_ONLY=12
drop_DAO_ONLY (TODO-13)              735      723     12   policy
proc_BO_CVn_Light_001                723      704     19   per-frame master-lock
vsx_out_of_scope ROT (VT only)         -      633      -   POLICY column; not a detection loss
```

Reconciliation: 3621 = 735 + 2886 (SNR). 735 = 723 + 12 (DAO_ONLY). 723 - 19 = 704 (proc).

**By G (SNR filter replay; Gaia NN mag within 8"):**

```
G    SNR_drop   SNR_keep   proc_001   localDB_footprint
5-11      0        ~431       ~460         ~483
12      270         220        224          542
13      768           6         10          943
14      888           3          4         1589
15      177           4          4         2592
16-17     0           0          0      3887+2543
```

G12-G15 collapse of O1 sits in **prematch_snr_peak_filter**. Downstream stages only
carry that list. Cone incompleteness matters for G15+ vs full DR3 but not for the G12-G14
cliff (cone footprint still has hundreds/thousands there).

VSX ROT: config `vsx_out_of_scope_types=["ROT"]`; 633 ROT rows in variable_targets.
Excluded targets: `no_dao_detection=2` only. **Policy exclusion is small and separate.**

================================================================
4. M2 ledger -- draft 435 (DESCRIPTIVE)
================================================================

From historical infolog (not reproducible under current code on the same MASTERSTAR.fits):

```
pass1=2552, pass2_in=1332, pass2_out=1225, merged=3777
SNR: 2951/3777 kept (noise_floor~2105.9 ADU, k=1.8)
masterstars=2951 (GAIA_MATCHED=2842, DAO_ONLY=109)
proc_001=2590 all GAIA_MATCHED
```

Replay under today's code on 435 MASTERSTAR yields pass1~492 and SNR keep~755 -- **does not
match the July log**. Treat 435 as a historical reference only (D-R0). Likely joint effects:
`masterstar_dao_threshold_sigma` 2.1 vs 3.8, different MASTERSTAR source/time, calibration
pixels, and possibly an older noise estimator (logged floor~2106 matches MAD-like sigma on
the current file; full-frame std floor is ~2979).

================================================================
5. M3 -- independent count on draft 512 Light_004
================================================================

SIPS (Milan): **1828** UCAC4-matched stars on
`draft_000512/detrended_aligned/lights/NoFilter_60_2/BO_CVn_Light_004.fits`.

Independent photutils DAOStarFinder on the same file (this task):

```
FWHM_px   thresh_sigma   n_detections
3.0       2.5            7494
3.77      3.5            2816
3.77      5.0            1894
5.19      3.5            2162
5.19      5.0            1436
```

SIPS ~1828 sits in the same ballpark as independent DAO (~1.4k-2.8k) on these pixels.
VYVAR proc for neighbouring Light_001 holds **704** rows because the locked masterstar
list after SNR filtering has only ~735 entries.

Where SIPS-like positions leave VYVAR: they never enter the masterstar list because
**prematch_snr_peak_filter** discarded the faint DAO/pass2 peaks on MASTERSTAR. Per-frame
MASTERSTAR lock cannot resurrect them.

No SIPS star table was available on disk; independent DAO substitutes for the count
comparison. Exact per-star crossmatch to SIPS was not measured (gap named).

================================================================
6. M4 -- width anomaly (O4)
================================================================

`fwhm_estimate_px` = per-star **image moment FWHM** at (x,y) on the photometry frame
(`compute_fwhm_gaussian_for_aperture_catalog` / `_fwhm_moment_at`). Same definition in both
drafts. Same binning (2x2) and grid (2082x1397).

| Sample | median fwhm_estimate_px |
|--------|-------------------------|
| 435 all stars | 6.418 |
| 512 all stars | 3.771 |
| 435 G5-11 only | 3.280 |
| 512 G5-11 only | 3.280 |
| Paired common catalog_ids (n=693) | 3.922 vs 3.746 (ratio ~1.008) |

**Resolved:** the global-median gap is a **population effect**. Draft 435's median is
pulled by G12-G15 stars (median FWHM ~6.77 there) that draft 512 largely lacks. On the
shared bright set the widths agree. No binning conversion required.

Separate note (not O4): MASTERSTAR header VY_FWHM is 3.21 (435) vs 5.19 (512) -- different
quantity/stage (stacked/selected-frame DAO FWHM), opposite ranking from proc medians.

================================================================
7. M5 -- cross-tool grounding
================================================================

| Package | Sub-threshold catalogue sources | Measurability | Non-detection |
|---------|----------------------------------|---------------|---------------|
| DAOPHOT / IRAF daofind | FIND detects above threshold; PHOT/ALLSTAR can measure at supplied coordinates (Stetson 1987; IRAF daophot) | Peak vs local sky*threshold | Absent from FIND list; PHOT may still return mag/limit |
| IRAF apphot | Photometry at coordinate list independent of detection | Aperture SNR | Magnitude limit / INDEF |
| SExtractor | DETECT_THRESH; ASSOC can inject catalogue | Flux/SNR flags | FLAGS / missing assoc |
| photutils DAOStarFinder | Detection only | Peak vs threshold | NoDetectionsWarning / empty table |
| sep | extract threshold | Flux/SNR | Not in catalog |
| AstroImageJ | Multi-aperture at user/catalogue apertures (forced-style) | SNR in table | Blank/NaN photometry |
| VaST | Detection + optional catalogue cross-ID | SNR cuts | Unmatched / rejected |
| LSST forced photometry | Explicit measure-at-coordinate without detection (DP1 tutorial) | PSF flux + errors | Measured flux with large error / non-detection flag |

VYVAR difference: pass 2 is **local re-detection**, then a **global peak gate** can discard
those recoveries, and per-frame photometry is bounded by the surviving masterstar list.
Classical forced photometry would measure at Gaia (x,y) even when pass 2 / prematch fail,
and report a non-detection rather than omit the row.

Citations: Stetson PASP 99, 191 (1987); IRAF `daofind` docs; LSST DP1 forced-photometry
tutorial; photutils DAOStarFinder API.

================================================================
8. Localization
================================================================

**D-R1 fires.** The G12-G15 population is lost at

  `prematch_snr_peak_filter` (MASTERSTAR): 3621 -> 735
  (`peak < median + 1.8 * plain_mean_med_std(full_frame)`).

Upstream: pass 1 is shallow (307) under draft 512 settings (threshold_sigma=3.8, FWHM=5.19,
thr~2402 ADU), so most faint stars arrive only via pass 2; pass 2 then feeds the same
global peak gate that rejects them. Downstream (match, DAO_ONLY drop, proc) do not
independently remove the G12-G15 cliff.

**D-R3 (partial, G16-G17 only):** the cone table under-represents G15+ in the footprint
versus local DR3 (567 vs 2592 at G15; 0 vs thousands at G16-17). That does **not** explain
the G12-G14 loss (cone still contains those stars; SNR drops them after pass 2).

**D-R2 does not apply:** the enumerated chain contains the loss; residual to SIPS/independent
DAO is explained by the post-SNR masterstar bound.

**D-R0** applies to all 435-vs-512 contrasts.

**D-R4:** VSX ROT counted separately; not absorbing residual.

Quoted rule: **"D-R1. The step at which the M1 ledger loses the G12 to G15 population is
the localization. Steps downstream of it carry no independent information."**

================================================================
9. Implied fix (not executed)
================================================================

Authorized separately, with measured delta. Candidates consistent with this localization:

- Do not apply the global full-frame peak gate to pass-2 recoveries, or use sky/MAD sigma
  for the noise floor instead of full-frame `plain_mean_med_std`; and/or
- Measure forced aperture / forced PSF at Gaia positions that fail detection and retain
  rows with an explicit non-detection flag; and/or
- Revisit `masterstar_dao_threshold_sigma` (2.1 historical vs 3.8 on 512) as a joint lever.

================================================================
10. Could not measure
================================================================

- Exact SIPS star list crossmatch (no export on disk); used independent DAO counts instead.
- Historical 435 intermediate mag-binned SNR losses (log has ALL-mag only; replay under
  current code is not that run).
- Which single commit changed the prematch sigma estimator (out of scope; adjacent
  DAO-SNR-SIGMA-01 deferred).

================================================================
11. Confirmations
================================================================

- O1 G-bin table for proc_001: re-verified (435 2590 / 512 704).
- O2: stars present on draft 512 pixels (independent DAO and SIPS).
- O3: all proc rows GAIA_MATCHED; DAO_ONLY=0 on proc; FORCED_APERTURE label unused.
- O4: population effect; common-star FWHM agrees.
- O5: zone schema differs (descriptive).
- Nothing committed or pushed. Science pixels not modified.
