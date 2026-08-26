# VYVAR -- Development State

**Status:** **CLOSE-OUT 2026-08-26 C8 STOP.** Push incident recorded;
do not force-push `origin/main`. P0 guard in. C8-1 R1' blocked
(iter4 one-file copy cannot import iter2). C8-2 frame 29 QC admitted
(n_stars 263). DEPTH-AUTH-01 recorded. C4 ZP-OK v2 next. C6 waits
Milan GO after C4 STOP. Live 516/520 SHA unchanged.
**CLOSE-OUT 2026-08-25 C3/C7 STOP.** C3 COMP-RMS-DEF-01-B
wired (k=5, LOO mag, ZONE-SAT-01). C7: R1 harness was contaminated;
lost VSX `1500387696044768384` is depth+DAO, freeze LC already
no_data; C6 not asked. C4 ZP-OK v2 task is on disk (run next, Push:
NO). origin/main stays `b1f5b8c`. Live 516/520 SHA unchanged.
**CLOSE-OUT 2026-08-25 C1/C2 STOP.** C0 pushed to
`origin/sel-ghost-01` at `78b349598b6fc2fc56ee9c8380fbd0728e614831`
(`--fast --clean` OVERALL PASS). origin/main stays `b1f5b8c`. C1
ANCHOR-DRIFT-01: R1-vs-R0 is freeze lag; census 4+1 unexplained; no
fix commit; C6 waits Milan GO. C2 COMP-RMS-DEF-01-A measured; C3
waits GO. C4 ZP-OK v2 locate-fail. Live 516/520 SHA unchanged.
**SEL-GHOST-01 B-STOP-3** (2026-08-25): T1/T2 on `main`
(`e410130`/`6e0fd5c`); INV-CAL-01 honors `pre_calibrated` (`6950495`).
Production-path 516 three-way (R0 frozen / R1 `c592ecf` / R2 HEAD)
measured; no re-cut, no push. Live 516/520 SHA unchanged. **SEL-GHOST-01 B-STOP-2** (2026-08-25): D1-D4 wired;
`--fast --clean` PASS at `936512f`; 520 `g_60_4` re-solves at 0.566 "/px
(gate-out 61, D2 refit rejected). 516 catalog_id 3583 vs c592ecf 3581;
sandbox LCs not byte-identical to live; `--full` FAIL (D3 raises on
frozen MS missing `vy_identity_gate`; 14 empty-comp). No re-cut. No
push. **SEL-GHOST-01 B-STOP-1b** (2026-08-25): clean-tree `--fast`
holes closed (`0684ba9`/`b39982c`/`6dad937`); B1 changes 516 by 10 catalog_ids,
0 in LC ensembles, `--full` byte-identical YES; H-LABEL TRUE and
INV-SOURCE-STATE-01 wired (`58a2187`); 520 has no pre-optimizer WCS
on disk. B3-B5 still wait D1-D3. **SEL-GHOST-01 B-STOP-1** (2026-08-25):
B1+B2 on `main` (`d8c18a7`/`e2a0a84`/`01f6f77`); 520 optimizer entry
7==gate-out (not 347); 1/8 ghost IDs still injected post-gate at Gaia
xy; 516 fail=0 and widen not fired; catalog_id set not byte-identical;
B1e lock rejects=10 not 0. STOP for Milan D1-D3 before B3-B5. **REG-520-01 STOP**
(2026-08-24, measure): June 0.06 vs today
0.39 on the same non-cal button is S2 selection (rms ceiling 0.1 +
59-px false locks), not S1 pairing starvation and not H-CAL-MISCLASS.
CAL-520-01 library facts remain; causal story superseded.
**DAO-GAIA-XFER-01 CLOSED** (2026-08-24, Milan GO): STAGE-01 sandbox
gate pinned to hand_validated params; 520 g/i/r calibration certificate unblocked
(z_90_4 remains solve-rejected). EPSF-VALID-02 **CLOSED** (2026-08-22 S6). **EPSF-BRIGHT-01 CLOSED** (2026-08-23 FD-A).
**EPSF-AC-01 CLOSED** (measure) and **EPSF-AC-02 CLOSED** (Milan GO wire, 2026-08-24): production
F6 AC = P4 (`p4_none`, uncorrected stamp); internal PSF LC = P4 + INV-PSF-LC-PIN-01.
**EPSF-PIN-CENSUS-01 CLOSED** (measure, 2026-08-24): 100% of pin drops are stored chi2>=50;
STOP for Milan on interim `psf_fit_ok_for_zp`. **EPSF-NEWTON-518-01 CLOSED** (STOP N2):
gated pool 26<30; ZP-OK remains parked.
Draft 516 production ePSF is **gated 67-star** model with edge-star build guard; pre-gated 1475-star
model archived. Draft 517 has first gated production ePSF (66 stars). FD-A full-CCD variance model
live (`psf_weight_mode=full_ccd`); BO CVn PSF overlay **133/134** fit_ok post F6 re-merge.
**EPSF-SHAPE-01** remains **OPEN HIGH** for the **narrow ePSF core** (FWHM 2.36 vs 3.30 px;
bright chi2 ~68), now routed to **EPSF-CORE-01**. **EXPORT-PARITY-01** remains **OPEN HIGH**.
Anchor era unchanged: core **9902d918** n=121; extended **472bc9e4** n=179; P1 golden **6af4539c** n=115.

**vyvar.sqlite3:** MASTER_SOURCES btrees still corrupt in file; table **retired from code**
(MS-SOURCES-RETIRE). **DB swap dropped** (Milan 2026-08-22 files-only direction);
`db-quick-check` **WARN/waived** via committed marker (`dev/validation/db_quick_check_waiver.json`).
**DB-RETIRE-01** (FUTURE): migrate remaining DB stores to files; retire vyvar.sqlite3 entirely.

**Band letter:** CV affirmed (D10-1/D10-1b); first AAVSO/VarAstro uploads (BO -> FW) pending.

**ePSF:** production-ready on 516/517 (Part C gated pool, INV-PSF-ADDITIVE-01 merge path, FD-A)
for relative photometry under P4 (uncorrected fit flux). Absolute PSF flux scale on bright
stars is untrusted until EPSF-CORE-01 rebuilds the core. Canonical AC, when wanted, is
DAOGROW/DOLPHOT growth-curve totals, not chi2-gated DAO ratio.

Last updated: **2026-08-26** (CLOSE-OUT C8 STOP; C4 ZP-OK next; C6 waits GO; C3/C7 STOP; C0-C2 STOP; SEL-GHOST-01 B-STOP-3; B-STOP-2; B-STOP-1b; B-STOP-1; REG-520-01 STOP `92361a3`; CAL-520-01 H-CAL-MISCLASS superseded as cause; DAO-GAIA-XFER-01 CLOSED `e5a6149`/`505fa13`; MULTIFILTER-WCS-01 carry).

## 2026-08-25 -- CLOSE-OUT C3 COMP-RMS-DEF-01-B STOP

k=5 from C3-0 p90=3.672. Selector `comp_rms` is LOO mag MAD; ceiling
min(0.1, 5 x photon). 520 G=7.63 zone=saturated. P-C3-1 HIT.
P-C3-2 one miss r=5.10 (`1496315070616056064`). Evidence:
`CURSOR_RESULT_COMP_RMS_DEF_01_B.md`.

## 2026-08-25 -- CLOSE-OUT C7 pre-C6 verification STOP

R1 T3 expand used HEAD STAGE-01 (harness-fixed). `--full` copy list
in PROCESS. Headline no_data from snapshot vs live proc_029.
Lost VSX G=15.56 is depth 15.0 + DAO miss; freeze LC already
no_data. C6 not asked. Evidence: `CURSOR_RESULT_CLOSEOUT_C7.md`.

## 2026-08-25 -- CLOSE-OUT C0 (hygiene; branch `sel-ghost-01`)

C0a: one haversine for `_dist_deg`, persist 9 decimal degrees
(`c929c0b`). C0b: D3 gates on `snr_ap_pixscaled` (`78b3495`).
`--fast --clean` OVERALL PASS. Pushed `main:sel-ghost-01` remote SHA
`78b349598b6fc2fc56ee9c8380fbd0728e614831`. origin/main stays
`b1f5b8c`.

## 2026-08-25 -- CLOSE-OUT C1 ANCHOR-DRIFT-01 STOP

R1 vs R0 is freeze lag: C1c no MS-moving commit in `ad19e14..c592ecf`
on the pre-expand table. Census 4+1 IDs unexplained. No fix commit.
C6 waits Milan GO. Evidence: `CURSOR_RESULT_ANCHOR_DRIFT_01.md`.

## 2026-08-25 -- CLOSE-OUT C2 COMP-RMS-DEF-01-A STOP

Selector `comp_rms` is mag-bin relative flux, not LOO differential mag.
ZONE-SAT-01: 85pct NaN skips the peak test (G=7.63 linear). C3 waits
GO. Evidence: `CURSOR_RESULT_COMP_RMS_DEF_01_A.md`.

## 2026-08-25 -- CLOSE-OUT C4 EPSF-ZP-OK-01-WIRE v2 locate-fail

v2 task file not found. No wiring. Evidence:
`CURSOR_RESULT_EPSF_ZP_OK_01_WIRE_v2.md`.

## 2026-08-25 -- SEL-GHOST-01 B-STOP-3 (production-path evidence; no re-cut)

T1 D1 radius = max(12", 3xFWHM x scale); T2 MASTERSTAR `snr` =
aperture SNR. 516 R0/R1/R2 production-path three-way; 520 V0612 rms
ceiling measured (no wiring). Honest match rate redefined on DETECTED
rows. Candidate snapshot SHA from R2 is in
`CURSOR_RESULT_SEL_GHOST_01_B3.md`. No re-cut. No push.

## 2026-08-25 -- SEL-GHOST-01 B-STOP-2 (D1-D4 wired; 516/520 sandbox)

D1 one-pass radius, D2 refit guard+backup, D3 candidacy, D4 lock
reject=3xFWHM. 520 g_60_4 solves when DB 0.566 "/px is not overwritten
by 15.511 UI scale. `--full` FAIL on frozen CSV missing D3 columns.
Evidence: `CURSOR_RESULT_SEL_GHOST_01_B2.md`. Not pushed.

## 2026-08-25 -- SEL-GHOST-01 B-STOP-1b (measure + label honesty)

Clean-tree `--fast` at `01f6f77` FAIL (STAGE-01 not in tree); `0684ba9`
4-tuple callers; still FAIL on untracked empty-sky CSV; `b39982c`
tracks it. 516 B1 effect is 10 pass1 edge IDs vs `c592ecf` control
(harness is 4+1 vs live); K=0 in 60 aperture LC ensembles; `--full`
YES. H-LABEL TRUE; leftover ghost and G<12 injects now
`catalog_membership`. No pre-optimizer 520 WCS; M4b not run.
Evidence: `CURSOR_RESULT_SEL_GHOST_01_B1b.md`. Not pushed.

## 2026-08-25 -- SEL-GHOST-01 B-STOP-1 (B1+B2 wired; STOP for D1-D3)

Sandbox 520 skip-solve: optimizer 7/685 == gate-out 7 (not 347). P-B2
MISS 1/8 (`1111922300852743808` locked at Gaia xy, gate empty). P-B3
optimizer skipped. P-B4 11/11 vs final WCS is mostly membership at
d_px=0. 516: fail=0, match_sep_effective=12", cid set 3571 vs 3584,
`n_lock_geometry_reject=10`. Live SHAs unchanged. Evidence:
`CURSOR_RESULT_SEL_GHOST_01_B.md`. Not pushed.

## 2026-08-25 -- SEL-GHOST-01 A measured: H-MATCH-WIDEN stopped (P-A5 false; name-export restore; no refine rematch)

Evidence: `CURSOR_RESULT_SEL_GHOST_01_A.md`. Architect writes Part B.

## 2026-08-24 -- REG-520-01 (measure, STOP)

Same SS Cam 2026-06-08, same `RUN VYVAR (non-cal)` button: June
`lc_rms` 0.0622 vs today 0.3949. S1 matching A/B/C does not change
G<12 / G<14 DETECTED counts (already 11/11 and 57/57). Selected 8
comps are DETECTED_P1 but 59 px median from Gaia (aperture != star).
June-band stars fail `max_comp_rms=0.1` (fieldwide 0.16-0.30). G<14
complete ensemble from today's `dao_flux` recovers lc_rms 0.068.
S3 `"time (unknown)"` is `_LC_OVERVIEW_COLS` omitting `time_base`.
STOP menu: S2 selection-input first; (a) rms-derived floor hygiene;
(b) `non_cal_declared` banner; (c) time_base; (d) PRECAL informative.
Evidence: `CURSOR_RESULT_REG_520_01.md`.

## 2026-08-24 -- CAL-520-01 (measure, STOP; cause superseded by REG-520-01)

Library facts stand: no eq=4/tel=6 masters; INV-PREP-01 0.02x
informational; donuts real. H-CAL-MISCLASS is **not** the cause of
0.39 vs 0.06 (Milan: same non-cal button in June). Evidence:
`CURSOR_RESULT_CAL_520_01.md`.

## 2026-08-24 -- DAO-GAIA-XFER-01 (Milan GO wired)

STAGE-01 iter4 sandbox gate pinned to `hand_validated()` params. Draft-derived
centroid tols stay production-scope on the certificate with identity stamps
(catalog fp, sandbox SHAs, hand CSV, 516 plate scale/FWHM). Cause was H-GATE-XFER
(REGRESS-01). z_90_4 solve reject stands (g-WCS-on-z gate 2.7%, contrast
9.06 ADU); carry MULTIFILTER-WCS-01. Evidence:
`CURSOR_RESULT_DAO_GAIA_XFER_01.md`.

## 2026-08-24 -- EPSF-NEWTON-518-01 (STOP N2)

Newton draft 518 is photometry-ready (bin2 1.30 arcsec/px, V 60 s, 71
science lights, 10 aperture LCs) but the Part C gated ePSF pool is 26
stars (science_scope choke) vs `epsf_min_stars=30`. No ePSF, no P-A..P-E.
EPSF-ZP-OK-01-WIRE stays parked. 516 hash-identical. Evidence:
`CURSOR_RESULT_EPSF_NEWTON_518_01.md`.

## 2026-08-24 -- EPSF-PIN-CENSUS-01 (measure; STOP for Milan)

100% of 7453 pin drops on draft 516 are stored `psf_chi2 >= 50`. Inferred
non-converged / quality-fallback class is empty. Admitting chi2>=50 holds
PSF-vs-aperture quality (BO 38.8 -> 37.3 mmag on 134/134; FW 0 -> 134 at
48.5 mmag RMS, offset not scatter). Interim proposal `psf_fit_ok_for_zp`
(convergence + finite only) is not wired. Evidence:
`CURSOR_RESULT_EPSF_PIN_CENSUS_01.md`.

## 2026-08-24 -- EPSF-AC-02 (Milan GO wired)

Production F6 merge and internal PSF LC use `psf_ac_policy=p4_none`. INV-PSF-LC-PIN-01
drops any epoch that cannot use the full pinned ensemble (BO CVn 23/134 survive;
PSF-vs-aperture RMS 614 -> 39 mmag on those epochs, RMS~=|median| = +40 mmag level
offset). SHAPE-01 narrow-core root remains OPEN, routed to EPSF-CORE-01. Evidence:
`CURSOR_RESULT_EPSF_AC_02_WIRE.md`.

## 2026-08-24 -- EPSF-AC-01 (measure; closed by AC-02 GO)

SHAPE-01-F: the 0.671 bright-star droop is F6 `psf_ac_factor` trained on chi2<5, not a
fitter-class split. EPSF-AC-01 measured the full-range uncorrected PSF/DAO (not flat in mag)
and the AC ensemble census (0/30 brightest ever admitted). Evidence: `CURSOR_RESULT_EPSF_AC_01.md`.
Root narrow-core remains EPSF-SHAPE-01 / **EPSF-CORE-01** (FUTURE).

## 2026-08-22/23 -- SESSION-CLOSE ePSF arc

**EPSF-VALID-02 CLOSED:** F1-F6 wired invariants (INV-PSF-FRAME-01, INV-PSF-ADDITIVE-01),
science-set dashboard scope, gated 67-star build + S6 swap on draft 516, first gated build on 517,
F6 PSF-only merge 134/134. Evidence: `CURSOR_RESULT_EPSF_VALID_02_*` (F1-S6, R1R4, S1S4, S5B,
ACCEPT).

**EPSF-BRIGHT-01 CLOSED:** UI table cap removed + gated-epoch caption; M1-M3 confirmed sky-only
chi2 brightness gate; **FD-A GO** - full CCD variance model; F6 re-merge (BO CVn 133/134 fit_ok).
**EPSF-SHAPE-01 OPEN HIGH:** PSF/DAO ratio droop on bright stars persists post-FD-A. Reports:
`CURSOR_RESULT_EPSF_BRIGHT_01.md`, `CURSOR_RESULT_EPSF_BRIGHT_01_P3.md`.

**Open HIGH:** EXPORT-PARITY-01; EPSF-SHAPE-01 (routed to EPSF-CORE-01).

## Canonical run mode (516 freeze)

Headless `--full` / P1 / rebuild: PFS ON, `export_err_mode=calibrated`,
`err_background_mode=empirical` (INV-ERR-MODE-01 fail-loud),
`saturate_limit_fraction=0.80` (52428 ADU; one authority including
MASTERSTAR zone writer). Snapshot inputs are copied into
`tmp/session_baseline/<ts>/` so the gate cannot mutate the freeze or
follow live-draft drift.

## 2026-08-18 -- ANCHOR-516-04 CLOSED (pending Milan push)

Clean 516 rebuild from Phase 0 after MASTERSTAR zone re-annotation at
0.80. E1-E5 PASS. MAG 48/48 vs de6f7c8. ct_n_comp 2346->2345, ct_c1
unchanged (-0.373). 48 LCs; CV CVn `per_frame_saturation`. BO median
err 8.532 mmag; 01B MAD identical to 515 (BO 7.151, FW 8.201 mmag).
AAVSO MAG 134/134 identity; MAGERR 82/134 rows change at 3-decimal.
SUBMIT-01 PASS (Milan submits; nothing previously uploaded).
`--fast` OVERALL PASS (1460 passed, 21 skipped, 1664 s). `--full`
OVERALL PASS (1461 passed, 21 skipped; pipeline 6135 s; SHA 477dc8cf
n=97 / f71e0722 n=145; plan-regen 873; 48 LCs). Report:
`dev/results/CURSOR_RESULT_ANCHOR_516_04.md`.

## 2026-08-17 -- XVAL-AIJ-02 CLOSED


EXTERNAL-XVAL extended to the production clean 4-comp ensemble and two
frame states. AIJ 6.0.10 vs VYVAR de6f7c8 on BO CVn 2026-04-23: 4.86
mmag RMS / 134 epochs (4c); 3.27 mmag remains the 5c XVAL-AIJ-01 row.
Frame-set (detrended_aligned vs calibrated) 3.80 mmag RMS / 131 epochs,
no airmass/time correlation. QC: 13 VYVAR-rejected calibrated epochs
are ~2.4x worse in AIJ vs a local curve; 049 and 111 photometrically
fine (alignment-reason). Evidence under `dev/results/XVAL_AIJ_02_*`.
Report: `CURSOR_RESULT_XVAL_AIJ_02.md`.

## 2026-08-17 -- GAIN-PT-RADIUS-01 + SUBMIT-01

PT radius pinned at 4.0 px (`pinned_sky_dominated_4px`); leftover
dynamic_params ignored. A3: g_pt=0.63707, CI factor 2.468, authority
g_pt. ERR-only Phase 2A: MAG 48/48 identical; BO median err
8.365 -> 8.945 mmag; AAVSO MAGERR 0.008 -> 0.009; ERR_MODEL
`gain=g_pt=0.6371`. Product SHA **de6f7c8** (supersedes 36a53b0 for
err-sensitive meters). SUBMIT-01 all PASS; Milan submits manually.
Report: `dev/results/CURSOR_RESULT_GAIN_PT_RADIUS_01.md`.

## 2026-08-17 -- U-09 CLOSED + GAIN-AUTH-VERIFY-01

DATE-OBS on draft 515 60 s frames is start-of-exposure; VYVAR adds
EXPTIME/2; export is mid-exposure BJD. jd_mid - DATE-OBS = 30.000 s.
36a53b0 ERR_MODEL gain=0.7925 because PT at r=2.499 px had CI width
6.22 > 3; not a skipped sidecar. Photon term under-quotes vs g_pt=0.637
by 0.42 mmag median (BO CVn). MAG/times submit-worthy; g_pt re-export
optional. Report: `dev/results/CURSOR_RESULT_U09_GAIN_AUTH.md`.

## 2026-08-17 -- PFS-SEMANTICS-01 + SAT-RERANK-01B + EXPORT-HDR-01

PFS rescue keyed on recorded skip_reason. TARGET-DEPTH-02 outranks PFS.
One peak-test authority: 52428 ADU (INV-SAT-LIMIT). Third 515 rebuild:
97 -> 48 LCs (da9cce4 49 minus CV CVn `1497007144465726080`,
sat_clean_frac=0.448 < 0.5). B2: 0/24 saturated IDs in ensembles.
01B meters: BO check 8.5798 mmag (membership changed; supersede da9cce4
7.0498), FW check 10.6836 mmag (unchanged). BIN-8-9 OPEN at 11.9885 mmag
n=15 (byte-identical to D515-ACCEPT-01; proc-CSV LOO). EXPORT-HDR-01
re-export of BO CVn AAVSO+VarAstro from 36a53b0 with PFS ON via per-run
override. SAT-LIMIT-01 CLOSED. D1-2 OPEN.

Reports: `CURSOR_RESULT_PFS_SEMANTICS_01.md`,
`CURSOR_RESULT_SAT_RERANK_01B.md`, `CURSOR_RESULT_EXPORT_HDR_01.md`.

XVAL-AIJ-01 remains CLOSED (AIJ Table.tbl at
`dev/results/XVAL_AIJ_01_Table.tbl`).

P1 A/B on this tip: ERROR. `test_invariants_p1_golden` 1 passed, 4 ERROR
in 443.4 s. Cause: INV-CAL-01 `cal_diag block missing after dark
calibration` on `draft_000435_p1mini` (photometry-ready mini has no
cal_diag.json; pipeline stamps a dark-applied calibration_mode). Headless
vs UI byte-identity pair was not obtained. P1-RECUT remains OPEN. Not a
PFS-SEMANTICS defect.

`session_baseline_check.py --fast` OVERALL PASS: pytest 1442 passed, 28
skipped (P1 env unset; flow_doc green via restored config.json default
false, not edited facts).

## 2026-08-17 -- SAT-RERANK-01 blocked then superseded (8f107cf)

Defective 96-LC product SHA **8f107cf** quarantined: PFS rescued 45
zone_noise + 2 below_target_depth. Ledger VL-PFS-8F107CF. Report:
`dev/results/CURSOR_RESULT_SAT_RERANK_01.md`. On-disk state overwritten
by the 36a53b0 rebuild.

## 2026-08-17 -- XVAL-AIJ-01 + SAT-LIMIT-01

EXTERNAL-XVAL independent-tool **CLOSED**: AIJ vs VYVAR 3.3 mmag RMS / 134
epochs / BO CVn. SAT-LIMIT-01 **CLOSED** in code (INV-SAT-LIMIT); 515 catalog
reclassified (24 saturated including C2). D1-2 remains OPEN. Production BO
ensemble CSV not rewritten; check MAD without C2 rose 7.05 -> 8.58 mmag on
the product meter. Reports: `CURSOR_RESULT_XVAL_AIJ_01.md`,
`CURSOR_RESULT_SAT_LIMIT_01.md`.

## 2026-08-16 -- WIDE-ERR-04 (physical model)

WIDE-ERR + SEM **CLOSED** at identity calibration (s=1, sigma_r=0) with
container-domain g_pt and weighted SEM. Draft 515 LC err re-exported; mag
byte-identity 49/49. CORR-ERR-01 remains OPEN as LOW research note.
WIDE-ERR-CROSSRIG stays OPEN. Report: `dev/results/CURSOR_RESULT_WIDE_ERR_04.md`.

## 2026-08-16 -- Policy snapshot (post IMPL-05 C)

**Aperture.** Per-magnitude scatter table (IMPL-05 B): bright ~6.5 px -> faint
~2.0 px; ladder persisted as `aperture_flux_ladder.parquet`. Exact overlap
masking (IMPL-04); no sawtooth.

**Selection.** RMS -> |delta(BP-RP)| -> distance (COMP-ASSIGN-03); single-source
isolation at `snr_cog_isolation_fwhm` x FWHM; `phase01_comparison_max_comp_rms`
ceiling authoritative; `n_comp_max` is a ceiling not a target; 3-8 honoured
end-to-end. No ensemble size cut in v1.0 (IMPL-01). Colour LEVEL corrected at
export; shape null on this rig.

**QA.** Stability / Comp QA are **post-LC verdicts**, never selectors
(COMP-ASSIGN-01; IMPL-05 D pool guard). Fixed-meter acceptance (10-target
subset): BO 8.6 mmag, FW 9.8 mmag.

**Next:** P1 A/B on new tip; COMP-RMS-DEF-01; WIDE-ERR-CROSSRIG when other
rigs exist. See ROADMAP.

## 2026-08-14 -- CLOSE-AND-PUSH (unpushed until Milan)

Iron gates wired (IRON-GATES-01 PARTIAL: INV-PIXELS-01 still open). SKY-CLIP-01 plain median. PP-KWARG-01. INV-CAL-02 / INV-SAT-01 disk gates. COG-A1-01: seeing systematic not established (C-R2). A-1 cause: `VY_FWHM_GAUSS` override 3.3014 px vs night FWHM ~5.19 px. Successor A-1-OVERRIDE authorized in principle.

**Commit B invalidates `--full` anchor and P1 golden SHA.** Re-cut is follow-up.

Reports: `dev/results/CURSOR_RESULT_CLOSE_AND_PUSH.md`.

## 2026-08-14 -- Session close (audit Wave 7 + A-1 push)

**A-1.** SNR sizing authority: per-draft median frame DAO moment FWHM (not `VY_FWHM_GAUSS`). Code + tests pushed. Draft 510 re-export + Phase 2A: `aperture_px` **4.261**, check_scatter **0.008638**, GREEN trust. New checksum manifest: `dev/validation/anchor_510_checksums_a1_dao_fwhm_20260814.json` (237 files changed vs placed-aperture manifest; diff JSON alongside).

**Audit.** Seven waves closed; referee document `docs/VYVAR_AUDIT_2026_CLOSURE.md`. Register corrections: **U-P5-PRED** (P5 could not test aperture-dependent saturation), **U-XVAL-COMP-RMS** (2 mmag comp gap open).

**Physics findings (independent of code):** QHY294MM 14-bit-in-16-bit container (65535 = clip); pedestal ~24.5 ADU/bin1 vs header `OFFSET=0`; dark at -10  degC pedestal-dominated (60 s = 120 s median); CMOS block-sum dark resample correct, CCD on-chip binning would not be (INV-CAL-01 derives convention).

**Handoff:** `docs/VYVAR_HANDOFF_2026-08-14.md`.

---

## 2026-08-13 -- INV-CAL-02 / calibrated stage integrity (implemented, pushed)

Option A: `VY_CALSTAGE` + FITS `VY_CALDATASUM` stamped in same flush as pixel change;
legacy resolver honest (`INDETERMINATE_*`); compare gates refuse unknown stage; force
reapply -> `SKYSF_N_R{pass}`. Anchors 435/509/510 **unchanged** (sha256). Rename:
`qc_enrich_calibrated_lights_in_place`. Spec: `dev/results/specs/VYVAR_CAL_STAGE_SPEC.md`.
Report: `dev/results/CURSOR_RESULT_inv_cal_02_impl.md`.

## 2026-08-13 -- INV-CAL-01 / CAL-DIAG v2 (pushed)

CAL-DIAG v2 as **INV-CAL-01** (zero config keys). P1: draft 435 **150/150 pixel-identical**.
P2 (corrected): apply archived `VY_SKYSF` sky order before compare -- 509/510 **150/150
identical, max diff 0.0**. Hazard **FIXED** by INV-CAL-02 (DECISIONS 11.3). Reports: `CURSOR_RESULT_inv_cal_01_impl.md`, `CURSOR_RESULT_cal_mismatch_509_510.md`,
`CURSOR_RESULT_inv_cal_01_p2_push.md`.

## 2026-08-13 -- SAT-DIAG placed aperture (draft 510 BO CVn)

Peak search removed; raw saturation uses aligned DAO grid lock + 11 px COM
refinement on raw frames (`PLACED_APERTURE`). Mag-guided retained on variable
target for drift diagnostic only. Draft 510 BO CVn: **5 comps** (incl.
`1497974027502858240`), check scatter **0.008629** (=509), GREEN trust, 134 pts.
See `dev/results/CURSOR_RESULT_placed_aperture.md`. Post A-1 re-export (2026-08-14): check scatter **0.008638**, `aperture_px` **4.261**.

---

## 2026-08-12 -- ZP-CLIP-REMOVAL (draft 509 BO CVn)

Same raw as draft 435; HEAD run admitted 5 comps and armed a dormant per-frame
3xMAD zeropoint clip in `ensemble_normalize` (`len(z) >= 4`). Intermittent rejection
of a good TIER1 comp produced ~50 mmag two-state ZP (check scatter 0.025).

**Fix:** remove the clip; keep Broeg weights. Post-fix 509: check `...4892800`
scatter **0.00863**, target residuals unimodal, n_points 134, instrumental unchanged
(~0.009). Trust GREEN. Decision: `docs/VYVAR_DECISIONS.md` ZP-CLIP-REMOVAL.
Invariant: INV-COMP-MEMBERSHIP. `phase01_comparison_max_mag_diff` left at 2.0.

Note: `c9e1f8f` claimed to remove all science-path sigma-clip but left this clip;
token search failed. Sweeps must be by behaviour.

## Session close 2026-08-07 (DAO detection closure + doc sync)

DAO detection workstream **closed**. Reference: `docs/VYVAR_DAO_DETECTION.md` (amended 2026-08-07:
confusion-blend untestable not refuted; unmeasurable-fraction wide-rig caveat). A-6 magnitude
classification and INV-MS-01 removal complete. Commits `8ed215f..44e8656` on main.

**Closed this arc:** INV-MS-01 runtime gate; A-6/A-6b DAO_ONLY classification (report, not
filter); DAO-THRESHOLD-PARAMS question; D1 inactive unit-normalisation plumbing; params scope
triage (`scope_key`, groups a/b/c); calibration-path audit (read-only); DAO-PHYS through
DAO-CLOSE.

**Open (explicit carry-forward):**

0. **INV-DAG-01 re-stamp friction (2026-08-12).** Forward stage order in
   `invariants_runtime.py:22-32` is real; `stamp_pipeline_stage` rejects
   `idx < max_seq` at `invariants_runtime.py:494-507` when re-running photometry
   on a draft that already has `postprocess` stamped (`photometry_core.py:10648`).
   Supported re-entry path blocked; workaround was trimming `pipeline_meta.json`
   stages. No fix yet.
1. **SAT-DIAG (IMPLEMENTED 2026-08-13).** Wired gate: raw peaks, pile-up derivation,
   INV-SAT-01. Spec: `dev/results/specs/VYVAR_SAT_DIAG_SPEC.md`. Does not close
   too-high ceiling gap (exposure ramp only).
2. **A-1 FWHM estimator disagreement (carry-forward).** Moment FWHM on single
   frames vs fitted Gaussian FWHM on MASTERSTAR stack differ by factor ~1.6-2.1;
   aperture follows the smaller value. Characterised in ZP-clip close-out;
   open finding, not a regression from ZP-CLIP-REMOVAL.
3. **P1-RECUT** -- golden ledger stale since `a9d7eb0` (2026-07-28); science commits
   `9b74548`, `683fba1`, `f0b310e`, `5cd6ae9` moved outputs. Three P1 ledger tests fail.
   Interim standard: local A/B (two headless runs at one HEAD, no `VYVAR_P1_REUSE_FROZEN`).
4. **Task A regression remediation** -- `test_masterstars_csv_write_survives_bp_rp_failure`
   reimplements fixed control flow; does not call `generate_masterstar_and_catalog`.
5. **D1b** -- conversion table exists; normalised companion defaults await Milan review (all
   companions still `None`; behaviour unchanged).
6. **D2** -- design complete; blocked on storage choice (nested `config.json` dict vs DB table
   keyed on `(ID_EQUIPMENTS`, `ID_TELESCOPE)`; Cursor recommended DB table).
7. **F-B01 / F-B02** -- PASSTHROUGH runs record `CALIBRATION_MODE=vyvar_calibrated`; fix order
   in `dev/results/CURSOR_RESULT_calpath_audit.md` sections 13-14.
8. **QHY294MM read-noise double-count** -- DB 7.6 e- may be bin2 value re-scaled to 15.2 e-.
9. **BPM sidecars** -- no `*_dark_bpm.json` found for any draft; path status unresolved.

Report: `dev/results/CURSOR_RESULT_session_close_20260807.md`.

## Session close 2026-08-04c (WIDE-ERR investigation record)

WIDE-ERR investigation state written to `docs/VYVAR_LIMITATIONS.md` (WIDE-ERR section).
Decision on sigma_sys_mag **DEFERRED** pending WIDE-ERR-CROSSRIG. Three ROADMAP items
added (HONEYCUTT-PDF, CROSSRIG, DB-DEFECT-DIAMETER). E4 per-comp excess measurement
committed. Report: `dev/results/CURSOR_RESULT_session_close_20260804c.md`.

## WIDE-ERR

Investigated 2026-08-04. Per-comp excess ~20 mmag measured directly on rig.
Decision on sigma_sys_mag DEFERRED pending WIDE-ERR-CROSSRIG. See LIMITATIONS:47
for full state including 5 retracted hypotheses.

## Session close 2026-08-04b (evidence chain; WIDE-ERR harness fix)

Evidence chain committed; WIDE-ERR diag LC output redirected to `tmp/` (15:47 writes
accounted; 10:20-11:47 writes still unattributed). Manifest tripwire:
`dev/results/anchor_restore/manifest_restored_20260804.txt` +
`dev/tools/anchor_manifest_check.py`. ROADMAP **WIDE-ERR-POP-DELTA** added.

## Session close 2026-08-04 (ANCHOR-RESTORE-1)

Restored `draft_000435_snapshot_skysurface_20260716` from July offline zip; post-11:47
mutated tree quarantined. Gate seeding (`5bccd85a` / `7fdcdca4`) **not** updated and
**does not** match on-disk July generation (`3d26f469` / `6420f1da`). Anchor **not**
verified via `--full` gate. Report: `dev/results/CURSOR_RESULT_anchor_restore1.md`.

## Session close 2026-08-04 (BATCH-E-PARAMS-REGISTRY)

Docs/test hygiene only; **no science path touched**; anchor SHA gate **not re-run** (nothing
under it changed). Registry **271 -> 277** entries; `VYVAR_PARAMS.md` regenerated. FLOW threshold
**2.1 -> 3.8** synced in `flow_doc_facts.py`, builder prose, and PDF. ASCII policy clean
(**34** tracked files; **215** U+FFFD of which **43** hand-repaired). `aperture_snr_sizing`
reclassified from "dead" to **partially wired** in `docs/VYVAR_LIMITATIONS.md`. Hygiene subset
and full suite on main tree: **1235 passed / 0 failed / 26 skipped**. Commits `8094af8..33ec2dc`.
Report: `dev/results/CURSOR_RESULT_batch_e_params_hygiene.md`.

## Session close 2026-08-04 (audit closed)

Batch E **GATE 2 authorized**. Physical re-cut fingerprints pushed (core `5bccd85a...` n=497;
extended `7fdcdca...` n=744). Register item 29 **FIXED**. Science audit **closed**. Future
threads: **WIDE-ERR**, **MASTERSTAR stacking** (not audit-open; logged in ROADMAP). Reports:
`dev/results/CURSOR_RESULT_final_closure.md`, `CURSOR_RESULT_batch_E_physical_recut.md`.
Referee deliverable: `docs/VYVAR_AUDIT_CLOSURE.md`. **Superseded by ANCHOR-RESTORE-1:**
on-disk snapshot reverted to July generation; gate fingerprints no longer describe live tree.

## Session close 2026-08-03 (batch D GATE 1; batch E blocked)

Batch D **GATE 1 authorized**: fingerprints pushed (core `b9c9489a...`, extended
`65bc826c...`; superseded `b7f980c0...` / `2c43bbbf...`). Code base `683fba1`.
Wide-rig `sigma_sys_mag` floor fit for equipment_id 1: **anomaly** -- fitted
~15 mmag (outside Everett & Howell 2-5 mmag sanity); **not applied**. Batch E
**not started** (task rule: stop if floor fails sanity). Reports:
`dev/results/CURSOR_RESULT_batch_D.md`, `dev/results/CURSOR_RESULT_batch_E.md`.

## Session close 2026-08-03 (batch D; GATE 1 pending)

Batch D **implemented and pushed** (`683fba1`): I-04, I-11, P-02 scintillation wired. Re-cut #1
`--full`: science compare **PASS** (flux unchanged); err SHA changed (expected). **Fingerprints
await Milan GATE 1.** Batch E **not started**. Report: `dev/results/CURSOR_RESULT_batch_D.md`.

## Session close 2026-08-02 (audit closure B-revised)

D5-2 **CONFIRMED**: saturation/non-linearity G 8-9; fix = C-1/C-2 gate. Decisions 5-9 recorded.
`docs/VYVAR_AUDIT_CLOSURE.md` created. **Re-issue batch D**, then batch E.

## Session close 2026-08-02 (batches D and E blocked)

Batch D and E **not started**. Missing: Milan choices in `VYVAR_DECISIONS.md` (I-11, I-04,
P-02/A-6; T4-1 for batch E). Batch B precondition satisfied (B-open). Reports:
`dev/results/CURSOR_RESULT_batch_{D,E}.md`.

## Session close 2026-08-02 (closure batch C)

Decision brief for Milan: I-11, I-04, P-02/A-6, T4-1. `docs/VYVAR_DECISION_BRIEF.md`.
Awaiting Milan choices in `VYVAR_DECISIONS.md` before batch D.

## Session close 2026-08-02 (closure batches A and B)

**Batch A:** A-1, A-9, D1-1, U-09 documented; `docs/VYVAR_LIMITATIONS.md` created.
**Batch B:** B-open -- D5-2 mechanism DEFERRED; D1-2 DEFERRED. Reports:
`dev/results/CURSOR_RESULT_batch_{A,B}.md`.

## Session close 2026-08-01 (closure Step 1n)

Step 1n (N-none): C-sat and C-sky excluded; D5-2 localised to G 8-9 bin; flux_large at fixed
radius confirms compression is not aperture coupling. Report: `dev/results/CURSOR_RESULT_closure_step1n.md`.

## Session close 2026-08-01 (closure Step 1m)

Step 1m (M-x): COG-normalised flux slope -0.280 (H-ap and H-2 rejected); measured EE(r_ap) vs G
slope -0.016. D5-2 confirmed; mechanism open. Step 1l L-a withdrawn. fwhm_estimate_px varies
every frame (35/35 stars). Report: `dev/results/CURSOR_RESULT_closure_step1m.md`.

## Session close 2026-08-01 (closure Step 1l)

Step 1l (L-a): D5-2 mechanism is magnitude-dependent aperture radius (D5-1); Step 1k K-a
(non-linearity) withdrawn. pearson(residual, r50) = -0.024; peak trend vanishes after aperture
control. Fixture annulus r_in = 14.0 px. Report: `dev/results/CURSOR_RESULT_closure_step1l.md`.

## Session close 2026-08-01 (closure Step 1k)

Step 1k (K-a): F(12) has two defects -- additive per-star sky (~56% of G 11.52/11.53 pair) and
multiplicative flux-vs-G compression (harness slope -0.285, production -0.296). Residual tracks
peak ADU (r=+0.36); bright-half slope -0.185 vs faint -0.404. D5-2 opened. Fixture annulus
aligned to production geometry. Report: `dev/results/CURSOR_RESULT_closure_step1k.md`.

## Session close 2026-08-01 (closure Step 1j)

Step 1j (J-a): F(12) inconsistent with catalogue G (slope -0.285 vs -0.4); G 11.52/11.53 F(12)
ratio 2.6x at dG = 0.006 mag from per-star sky offset. Step 1i production annulus claim
withdrawn. Step 1k: sky model + amp/peak gate. Report: `dev/results/CURSOR_RESULT_closure_step1j.md`.

## Session close 2026-08-01 (closure Step 1i)

Step 1i located the EE(1.916) failure mechanism (I3): 12 px normalisation collapse on faint
stars (dominant) plus rare placement errors (low-EE tail). E5 WCS-position control excludes
placement as scatter fix. Step 1j: cause-based admissibility, not EE-band filtering.
Report: `dev/results/CURSOR_RESULT_closure_step1i.md`.

## Session close 2026-08-01 (closure Step 1h)

Step 1h diagnosed G6 magnitude dependence (H1): fainter-proxy COG numerator instability at
r=1.916 px; denominator and catalogue contamination excluded. No consolidated number. Step 1i:
numerator QC gate then re-measurement. Report: `dev/results/CURSOR_RESULT_closure_step1h.md`.

## Session close 2026-07-31 (science audit closure)

Twelve-domain science audit synthesised in `docs/VYVAR_AUDIT_FINAL.md`. Thirty-item closure
register in `docs/VYVAR_AUDIT_CLOSURE_REGISTER.md`. Stage 3 forensics (Parts 0c--0e) committed.

**Audit complete.** Remediation is tracked in the closure register, not open discovery.

**Next work item:** **Batch D** (I-04, I-11, P-02/A-6) -- preconditions met. Then batch E after D
re-cut authorization.

**Anchor re-cut:** still **BLOCKED** on T4-1 detection-noise decision (Milan), DAO threshold
recalibration post-stack, Part 0c `source_file` pairing fix, and Part 0e DAO-centroid stability.

Key Stage 3 findings:
- Part 0c delta table **invalid** (positional CSV pairing); correct pairing median p95 |delta| **0.104 mag**.
- Part 0e: focus-target tail from **DAO centroid shift** on same `catalog_id`, not ensemble/neighbour swap.
- Part 1c: Part 1b chi2=649 was **total chi2** mis-index; true median chi2_red ~ **4.7**.
- Part 2b: correct-path threshold sweep slope **-1.58**; **no N selected** (R5).

## Session close 2026-07-26 .. 2026-07-28 (BO CVn arc)

Eight production defects closed or guarded; entry-point equivalence demonstrated (454 UI vs 455
headless byte-identical on 198 LCs). See JOURNAL entry and `dev/results/CURSOR_RESULT_session_close.md`.

**New wiring this close:**
- **Infolog:** durable session log is the authoritative operator artefact; ring-buffer export labeled partial.
- **CATALOG-PROVENANCE:** Gaia + VSX DB identity (path, size, mtime, head/tail SHA, row count) in
  `pipeline_meta.json`; anchor `--full` compares fingerprints and reports *input catalogue changed*.
- **INV-PREP-01:** measured healthy **0.03x** (454) vs regression **20-60x**; threshold **10x** kept
  (margin documented in `VYVAR_INVARIANTS.md`).

**Disk cleanup (operator):** evidence for `DRAFT451-CAL-FRAME001` extracted to
`dev/results/context/frame001_evidence/`; deletion plan + manifest in session-close result -- agent did
not delete under `Archive/`.

## POST-451 remediation (2026-07-27)

- **Part A DONE** (`926a94c`): exoplanet promotion path resolved against `data_root`; fail-loud DB;
  VT schema preserves exo columns when frame empty.
- **Part B DONE** (`63b902d`): `VSX-GAIA XM:` and `FAZA 0 funnel:` routed through `log_event`;
  Phase 2A `skip_reason` propagated for `n_frames=0`.
- **Part C.1 DONE** (`ff08002`): order-2 sky-surface subtract restored on mono preprocess path.
- **Part C.3 DONE** (`1191579`): INV-PREP-01 + INV-MS-01 guards wired.
- **Part C.4 PENDING**: full raw-to-photometry BO CVn acceptance (no calibrated FITS on disk in
  draft_451 tree for local re-run).

Registered params: **270** (removed ``phase01_match_radius_arcsec``; was 271).

## Cython release readiness (RELEASE-1 + RELEASE-2) -- preview live

RELEASE-1 @ `b4c372a`: **86** science modules compiled (84 @ RELEASE-1; +``vyvar_runtime``
@ RELEASE-2; +``run_preflight_log`` @ BUNDLE-FIELD-FIXES-2); anchor identity gates PASS (see
`CURSOR_RESULT_cython_release1.md`). RELEASE-2: bundle builder, embedded Python 3.12,
data-dir separation, install docs, public repo staging, runbook.

**Preview `preview-20260723`:** live on `VYVAR-release` with **both platforms** (win64 +
linux-x64). Private tip through bug #10 (`10608bb` BUNDLE-DB-THREADING). Field-run findings
**#11-#13** fixed in repo (next preview refresh; no bundle rebuild this session).

**3-way verification protocol established:** Cursor smoke (`--fast` / registry sweep) +
Claude sandbox round-trips v3-v6 + Milan Linux field box.

**First real-sky astrometry through the public bundle:** FI Boo 147 frames (draft_000001,
NoFilter_60_2) -- astrometry + MASTERSTAR + RAM handoff **complete** on Milan Linux box;
photometry resumed after VSX path fix (see JOURNAL).

**Field bugs #1-#10** from Milan Linux preview install / Claude RTv4-v6 -- fixed and bundles
refreshed (final SHAs `7d8e0d30` / `abc8580e0` @ bug #10).

**Pending:** M71 E2E acceptance; v1.0.0 declaration (joint decision, open).

## Invariants program -- honest scope

Pinned at P4 closeout (2026-07-20). Claims verified against the tree
(`flow_doc_facts` counts, `WIRED_INV_IDS`, ledger IDs, `sigma_sys_mag` keys).
Full program P1-P4: DONE. Result: `dev/results/CURSOR_RESULT_invariants_P3P4.md`.

### GUARANTEED today

- Anchor byte-identity of the science file set via `--full` (`VL-ANCHOR-WCSINV`,
  draft_435).
- Golden-mini E2E equivalence incl. UI-order vs headless byte identity on that
  night (`VL-P1-GOLD`).
- Wired runtime gates: FLUX-01/02 FAIL, FLAT-01 WARN, WCS-01 WARN, DAG-01 FAIL,
  QC-01 FAIL (skip-processed alignment allowlist), PROV-01 + CFG-01 end-of-run FAIL;
  RNG AST guard.
- Docs<->code sync on 43 config facts + 18 function names + docs layout + FLOW
  PDF presence (`test_docs_sync_guard`).
- ASCII-only tracked text + LF-normalized repo (guards in `--fast`).

### NOT guaranteed (honest gaps)

- Behavior on rigs/configs outside the anchor night (Newton dense field, Brno)
  until their own validation data exist.
- Gated features under ON: `per_frame_saturation`, k2 NIGHT_FIT, COG, PSF branch
  -- implemented and synthetically tested where stated, NOT real-data validated.
- UI widget-level interactions beyond the encoded UI call sequence of the P1
  test.
- Multi-night behavior (single night is the canonical unit).
- Registry-only invariants (no runtime wiring yet): the non-`[wired]` IDs in
  `docs/VYVAR_INVARIANTS.md`.
- `sigma_sys` floors exist for band `"4"` only (other bands lack the systematic
  term -- GAPS A2).
- **MASTERSTAR detection and stack rebuild** sit outside the anchor `--full` gate
  (frozen `MASTERSTAR.fits` + CSV); preprocess/detrend changes can alter pass-1 DAO
  counts without failing byte-identity photometry SHA checks.
- **Anchor `--full` begins at `detrended_aligned/lights`** (copies frozen MASTERSTAR +
  CSV, runs photometry only). **Not covered:** calibration, preprocess (sky-surface in-place QC),
  and alignment. Together with plan-time VSX export regeneration and MASTERSTAR/detection
  (above), these are the three anchor coverage gaps found in the 2026-07 arc -- the boundary of
  what the anchor byte-SHA gate actually guarantees.
- **Anchor snapshot photometry portability:** `draft_000435_snapshot_skysurface_20260716`
  lives under gitignored `Archive/`; photometry was patched locally from an agreeing run
  (2026-07-28 schema re-cut). SHA verification on another machine requires the same snapshot
  patch or a fresh `--full` run -- not guaranteed from git alone.
- **Raw-to-photometry rebuild validated (2026-07-27):** draft_452 acceptance run after
  SKY-SURFACE restore (`ff08002`) matches anchor MASTERSTAR catalogue metrics (pass-1 **2552**,
  2951 rows, DAO_ONLY 3.7%, bg_std 83.8 ADU). Anchor `--full` photometry
  SHA gate remains the regression lock on frozen inputs; raw-path equivalence is now demonstrated
  on BO CVn (`dev/results/CURSOR_RESULT_post451_remediation.md` PRE-PUSH CLOSEOUT).
  **photutils 3.0 is not implicated:** pass-1 2552 under photutils 3.0 with production FWHM
  (~3.23 px from header); the earlier +1.1% replay-harness count used config default 2.5 px.
- **sigma_pp convention (MASTERSTAR):** report **46.90 ADU** = unmasked full-frame MAD on
  byte-identical MASTERSTAR; **~45.03 ADU** with star mask + 40 px margin; legacy **46.13 ADU**
  used a different margin -- not an image change.

## Repository layout (REPO-REORG -- DONE, gate PASS)

Root tidied into a stable layout: production code in `src_py/`, dev material in
`dev/` (`dev/tests|tools|validation|scripts|sandbox|orchestrator`), all Cursor
result/task docs in `dev/results/`, scratch in `tmp/` (gitignored). Root `app.py`
is a thin Streamlit shim. Commits `c611353` (dev/ move) + `8f4d7b4` (src_py/ move).
Anchor #3 `--full` gate **PASS byte-identical** (core `1c48d9fc...`
n=325; extended `744bce94...` n=487; PHASE0-IDENTITY-GATE re-cut 2026-07-27, active 165).
Milan UI smoke confirmed (app launched via root shim, Settings + Parameters tab render,
modified-counter=10) plus e2e draft_000436 anchor run. See CLAUDE.md /
VYVAR_PROCESS.md for the map; result `CURSOR_RESULT_repo_reorg.md`.

Parameter provenance mapped in PARAM-SOURCE-AUDIT (`dev/results/PARAM_SOURCE_AUDIT.md`
+ `param_source_audit.csv`, 304 keys): one config.json + DB reference tables +
FITS-resolved values; DB `SETTINGS` table found vestigial. Wave-A ownership arc
adds the `owner` axis and fixes the config-write render side-effect.

**WAVE-B-PARAM-REDUCTION -- DONE (pending final gate/push).** Registered params
reduced 304 -> 269 (config.json persists 249), per the PARAM-BUDGET-AUDIT dispositions
(Milan-approved): DELETE-DEAD 4, MERGE 14 -> 3 structured keys (`comp_color_tiers`,
`phase01_tiers`, `aperture_snr_sizing`), DELETE-DB-DUP 9 (DB/FITS resolver now sole
source; vestigial DB `SETTINGS` table dropped), HARDCODE 20 solver internals, INTERNALIZE
2 frame dims, and WIRE-IN `calibration_master_ccd_temp_tolerance_c` (bug fix). The 80-key
"never-touched expert" pool was explicitly REJECTED (stays KEEP; universality). Commits
`617b76f`/`08e5684`/`03d640c`/`c828c9c`/`d6c0d55`/`715e754` + STEP 7 docs. Result:
`CURSOR_RESULT_wave_b_reduction.md`.

**CONFIG-HUMAN-EDIT -- DONE (pending final gate/push).** config.json is now a
generated, grouped, commented JSONC-lite file editable by hand without the UI: the loader
tolerates `//` line comments and warns on unknown keys (difflib suggestion), the writer
emits a file header + pipeline-ordered sections (basic->advanced->expert) with a comment
for every group and key. Per-key help was ported from CONFIG_GUIDE_EN into the registry
`help` field (single source of truth); section comments come from `__meta__.phase_help`. A
standalone `dev/scripts/validate_config.py` reports syntax/unknown-key/range/type problems.
Live config.json migrated value-identical (249 keys). Commits `1307b73` (help port),
`742fee5` (loader), `86c6748` (writer + migration), `0b75a69` (validator). Result:
`CURSOR_RESULT_config_human_edit.md`.

## Test-hygiene backlog

- `dev/tests/test_g7_f003c_report_cfg_snapshot.py` (2 tests) is order-dependent: its
  `_factory` monkeypatch self-recurses through `config.AppConfig` when the qc-metrics
  path constructs `AppConfig()`. Passes in the full suite, fails in isolation. Confirmed
  pre-existing (reproduces on committed code with no wave-A edits). Repro:
  `python -m pytest dev/tests/test_g7_f003c_report_cfg_snapshot.py -q`. Fix later:
  make `_factory` bind the real `AppConfig` (e.g. capture it before patching) instead of
  re-importing the patched name.
- **NOQA-TRUNCATED-EXCEPT-BULK** (ROADMAP): 15 malformed `# noqa` directives across 10
  `src_py` files from truncated EXCEPT-BULK 2026-07-08 census comments; see
  `dev/results/CURSOR_RESULT_batch_e_params_hygiene.md` STEP 5C. Not started.

(BATCH-E-PARAMS-REGISTRY and the six `--fast` hygiene failures: **CLOSED 2026-08-04**,
commits `8094af8..33ec2dc`.)

## UI block

**Wave 1 (parameters): DONE.** Machine-readable registry (270 entries post PIPELINE-SIMPLIFY-1; was 272) with
parity / freshness / hidden-tier guard tests, generated `VYVAR_PARAMS.md`, tiered
Parameters dashboard, PDF Configuration page in the SUMMARY MEASURE REPORT. min_comps=3
closed as intentional (DECISIONS). Results:
`CURSOR_RESULT_params_registry_ui.md`, `CURSOR_RESULT_params_closeout.md`.

**Next: data dashboards wave** - identity QA series, trust/census fingerprint, AC status,
excluded_targets, VSX stamp, sky-surface stats, export skips. Spec by Claude next session.

## Data items (top)

1. **Fresh darks** - expiry ~**2026-07-21** (do first when scheduling next night).
2. **Anchor offline zips** (`C:\ASTRO\backups\`):
   - `draft_000435_snapshot_skysurface_20260716.zip` SHA256
     `a35d22354666e359ce1bdd9a6eb207d5d768466a67fcdb77c22425eabb3f84a0` (~4.82 GB)
   - `draft_000435_anchor_live_20260716.zip` SHA256
     `a4bb42d255e542b4a516197d5efe1a6304602b331680ac554caf41a244070faf` (~4.82 GB)
   - Historical: `vyvar_anchor_424_sigma_floor_20260713_core-bf3743a1.zip` (untouched)
3. **Archive freed** ~**48.7 GB** (drafts 428-434 + pass1 photometry backup). In-Archive
   remain: `draft_000435` + `draft_000435_snapshot_skysurface_20260716`.
4. F-428 arc evidence: `validation/f428_arc_evidence/` (moved from `tmp/`).

## Current snapshot (Anchor #3 on disk -- July generation restored 2026-08-04)

**draft_000435_snapshot_skysurface_20260716** -- restored from offline zip
(`C:\ASTRO\backups\draft_000435_snapshot_skysurface_20260716.zip`, SHA256
`a35d22354666e359ce1bdd9a6eb207d5d768466a67fcdb77c22425eabb3f84a0`) on 2026-08-04.
Post-11:47 mutated tree quarantined at
`Archive/Drafts/_quarantine/draft_000435_snapshot_MUTATED_20260804_1147`.

**DOCUMENTED FACT (not a gate value):** draft_435 snapshot, July generation, restored
2026-08-04 from `draft_000435_snapshot_skysurface_20260716.zip`; produced at `10d610c`
with `git_dirty=true`; photometry SHA core `3d26f4692ac81fc52db6ef9f70b148f9f7c56a5bb5e84e637339c4883ba47a96` n=333 /
extended `6420f1daa53a0d5d0a92bfd1ab30eba68e2ab88be8fe5f4c68048a5463054ac8` n=499.
166 `check_kmag_*.csv` (164 x `1499906247391001088`, 2 x `1497528072458898432`).
`pipeline_meta`: git `10d610c0`, `git_dirty=true`, `admission_sat_peak_frac` absent.
Integrity tripwire: `dev/results/anchor_restore/manifest_restored_20260804.txt`;
check with `python dev/tools/anchor_manifest_check.py` (see JOURNAL 2026-08-04b).

**Gate values do NOT describe this tree.** `session_baseline_check.py` and
`test_invariants_p1_seed.py` expect batch-E draft_000500 fingerprints
(`5bccd85a...` n=**497** / `7fdcdca4...` n=**744**). Neither the count nor the hash
describes any draft_435 generation: draft_435 has never produced 497 lightcurves (July
core n=333; mutated August core n=1121). Seeding error; see ROADMAP **ANCHOR-GATE-SEED**.
Do **not** treat `--full` SHA gate as verified until reseeding is resolved.

**Prior anchor history (reference only):** LABBE-DET pass used core `3d26f469...` /
extended `6420f1da...` (2026-07-16). Batch E physical re-cut on scratch `draft_000500`
produced `5bccd85a...` / `7fdcdca4...` -- fingerprints pushed to gate files but live
snapshot was never faithfully that generation before restore.

**Root cause of prior STOP:** ensemble SEM join / dict-order nondeterminism in phase2a `err`
(not Labbe placements). Labbe hardened anyway (canonical stars + SeedSequence + dump).

**INVARIANTS P1:** **DONE (2026-07-19).** Golden mini `draft_000435_p1mini` +
`VL-P1-GOLD` active (core `074ae881...` n=333; extended `66285d3f...` n=497). Opt-in
`VYVAR_INVARIANTS_P1=1`. Result: `dev/results/CURSOR_RESULT_invariants_p1.md`.

**APCORR-MIXEDFRAME:** **DONE (2026-07-19)** - night-level all-or-nothing COG gate
(`cog_night_fallback` provenance); COG still default OFF.

**INVARIANTS P2:** **DONE (2026-07-19).** Contract registry `docs/VYVAR_INVARIANTS.md` +
runtime gates in `invariants_runtime.py` (meta-only; science SHA untouched). FLOW 4.5
sky-surface pedestal wording corrected (was wrongly "flux-conserving").

---

## Prior: Anchor #3 STOP (superseded)

**draft_435 HEALTHY** but protocol-v2 SHA gate FAILED on `err` only (166/166 LCs). See
`CURSOR_RESULT_anchor435_closeout.md`. Fixed in LABBE-DET (`CURSOR_RESULT_labbe_det.md`).

---

## Prior: T3-RESTORE - FIX A+B committed; next gates

**T3 FIX A+B** on `89842ff` (+ docs `3db0879`). Sky-surface headers + `qc_metrics.csv` confirmed
on draft_435.

**T3 FIX A (done):** order-2 sky-surface subtract in shared ``preprocess_calibrated_to_processed``
(``preprocess_sky_surface_order`` default **2**). Empirical vs draft_429 Light_008: DAO pass-1
sim **2579** (2500-3000 band), smooth residual p99 **173 ADU**; ~237 below 429 logged 2816
(cal-only mask/clip gap documented).

**T3 FIX B (done):** ``git_dirty_code`` / scratch vs import-relevant root ``*.py`` classifier;
anchor / FAIL-CLOSED gates trip on ``git_dirty_code`` only.

**T2 verdict (434 baseline):** draft_434 on ``1e2e8d6`` = **UI-SICK** (6699-class). Milan re-run
after push expected: cal!=proc, ~2875 matched, sky ~1478 meta, identity p95 < 1 px,
``git_dirty_code=false``.

**T4 (prior):** VSX stamp after VT write; Labbe content-hash seed; NightRun match sep 2.0.

**429:** unprovenanced quality target; deterministic pre-T3 path = 6699-class.

**Next:** Milan UI RUN VYVAR -> HEALTHY? -> anchor #3 protocol-v2.

## Prior snapshot (F-431 root closure - partial)

**Anchor (was in-Archive ACCEPTED):** `draft_000424_snapshot_sigma_floor_20260713`; core `bf3743a1...`.
Now offline only (see above).

## 0714 arc detail (code unchanged)

**Err model:** empirical empty-aperture background (`sigma_bkg_ap`, F-BINGAIN-1) +
hybrid Howell fallback; **c4** small-sample SEM; mag/flux **unit fix** at combine; per-rig
`sigma_sys` in LC `err` (`sigma_floor_core.py`, `sigma_sys_mag` column). Newton eq4 floor
**18.0 mmag** [15.6, 20.2]; wide eq1 **un-floored** (bootstrap ~4.8 mmag, below SIGMA-A3 band --
unstable). Spec: `docs/VYVAR_SIGMA_FLOOR_SPEC.md`.

**SPARSE-TRUST (live):** check-star ensemble at n>=2, Howell 1988 triangulation, CI trust bands,
sidecar columns (`check_sparse`, `trust_R`, ...). External K sourcing on sparse branch. SS Cam
r_60_4 **YELLOW confirmed** (R=2.008 [1.224, 3.886], Milan 2026-07-14). Spec:
`docs/VYVAR_SPARSE_TRUST_SPEC.md`. Arc closed `7886157`.

**k'' per-rig record:** wide eq1 **LOW PRIORITY subdominance** (K2-STATS-FIX: T1 rho=-0.013,
bootstrap colour bound B=0.076 vs b_X scatter 0.094; plausible k'' not excluded but not dominant).
Newton eq4 **OPEN suggestive** (T1 rho=-0.325, n=19, power 0.40; T2 rho=+0.470 raw p=0.043;
underpowered for pre-registered DOWN). Results: `CURSOR_RESULT_k2_stats_fix.md`,
`CURSOR_RESULT_k2_cohort_correct.md`.

**CAL-DIAG:** gate default ON; v1.1 closeout re-verified on `237dd34` (150/150 VY_DKRSMP=SUM;
core SHA unchanged). Section-10 ledger items closed in CAL-LEDGER-BUNDLE (`5817c9b..b268a6c`).
Spec: `docs/VYVAR_CAL_DIAG_SPEC.md`.

**WIDE-SLOPE-NOISE: PARKED** (verdict **UNIFIED_PHENOMENON_PARK**, `114c423`). Faint tertile
sigma_slope_pt **5.17 mmag** aligns with PZQ sigma_r **5.5 mmag** and rig constant **4.5 mmag**
(one unidentified ~5 mmag driver; mechanism unknown). Neighbor contamination **untestable-here**
(pre-test gate). Bounds table: `CURSOR_RESULT_wsn2.md`. Spec **APPROVED**:
`docs/VYVAR_WIDE_SLOPE_NOISE_SPEC.md`. Prior EXCESS_UNATTRIBUTED retractions documented in
`CURSOR_RESULT_wsn_fix.md`, `CURSOR_RESULT_wide_slope_noise.md`.

## Session history (0713-0715, compressed)

| Date | Arc | Outcome | Ref |
|------|-----|---------|-----|
| 2026-07-15 | ARCHIVE-CLEANUP | Archive wiped; anchor offline; --full SUSPENDED | `CURSOR_RESULT_archive_cleanup.md` |
| 2026-07-14 | WSN-2 | UNIFIED_PHENOMENON_PARK; P4 excess integration | `114c423`, `CURSOR_RESULT_wsn2.md` |
| 2026-07-14 | WSN-FIX | SE/tertile/drift fixes; honest P2/P4 | `928a300`, `CURSOR_RESULT_wsn_fix.md` |
| 2026-07-14 | WSN initial | EXCESS_UNATTRIBUTED **SUPERSEDED** | `59478bb`, `CURSOR_RESULT_wide_slope_noise.md` |
| 2026-07-14 | CAL-LEDGER-BUNDLE | Section-10 closed; RN header fix | `b268a6c`, `CURSOR_RESULT_cal_ledger_bundle.md` |
| 2026-07-14 | CAL-DIAG closeout | Gate verified; pushed `237dd34` | `CURSOR_RESULT_cal_diag_impl.md` |
| 2026-07-14 | K2-STATS-FIX | Bootstrap CIs authoritative; wide subdominance | `13341b3`, `CURSOR_RESULT_k2_stats_fix.md` |
| 2026-07-14 | K2-COHORT | Full cohort; DOWN retracted | `CURSOR_RESULT_k2_cohort_correct.md` |
| 2026-07-14 | SPARSE-TRUST arc | CLOSED; SS Cam YELLOW | `7886157`, `CURSOR_RESULT_arc_close.md` |
| 2026-07-13 | ANCHOR-CHAIN-ACCEPT | Anchor accepted | `7ed7459`, `CURSOR_RESULT_anchor_chain.md` |
| 2026-07-13 | PROD-SIGMA-FLOOR | c4 + sigma_sys shipped | `8fb21b3`, `CURSOR_RESULT_sigma_floor.md` |
| 2026-07-13 | SESSION-CLOSE-0713 | Color-WB arc CLOSED | `601689d`, `CURSOR_RESULT_close_0713.md` |

## Earlier session history

**2026-07-10 snapshot (SESSION-CLOSE-0710):** Two workstreams **DONE** that day:
(1) **TODO-12 HRD arc** 12/12b/12c/12d/12e/12f - session-aware extreme-object table, enrichment,
identification tiers, PDF/UI details, summary.json freshness stamps (`generated_at_utc`, `git_head`).
(2) **F-BINGAIN-1 RESOLVED** - empirical empty-aperture `sigma_bkg_ap` in production `err` (IRAF/SExtractor/
photutils-aligned); hybrid `howell_scaled` fallback for crowded fields; regate PASS (decomposition-driven
gates). Result files: `CURSOR_RESULT_todo12_hrd.md` ... `_todo12f_hrd.md`, `CURSOR_RESULT_bingain_fix.md`,
`CURSOR_RESULT_bingain_acceptance.md`, `CURSOR_RESULT_bingain_regate.md`.

**Byte-identity baseline (F-BINGAIN-1):** Re-anchored for documented **`err` column divergence**
(empirical bkg term); **non-err proc-CSV science columns verified byte-identical** on patch-only
acceptance (draft_426 g, draft_424/425 all setups). LC `err` is the authoritative production uncertainty.

**2026-07-10 snapshot (TODO-12f):** `_make_row` extended (dist, parallax, raw SpT/otype, DSC p,
teff_source); PDF follow-on page per object; UI expander with full column set; validate
summary.json stamped with `generated_at_utc` + `git_head`. PDF +1 page (390), overflow 0.

**2026-07-10 snapshot (TODO-12e):** Enrichment cache v2 (+ SIMBAD sp_type, Gaia DSC WD prob);
`ident` tier column; RS Per (`458407464445792384`) RSG confirmed via SIMBAD lum class (was Very cool);
WD row confirmed via otype WD*; `hrd_dsc_confirm_prob=0.90`. draft_425 B: 5 confirmed / 2 candidate.

**2026-07-10 snapshot (TODO-12d):** `hrd_nss_category_enabled=False` (default) drops Gaia NSS binary
rows from Stage-1/2; draft_425 table 7 rows/setup (was 10 with 3 Binary). Annotated field image uses
MASTERSTAR FITS 1:1 PNG + pixel-scale guard; PDF HRD page embeds annotated field (overflow 0).
RSG alignment check draft_425 B: peak/bg 6.4 (was 1.9 on field_map before FITS background fix).

**2026-07-10 snapshot (TODO-12c):** Stage-2 priority RSG>RG before Very cool; `hrd_min_per_net=4`
per-net Stage-1 reservations; draft_425 RSG count 3/setup (was 2 + mislabeled s*r); luminous-net
stars reach enrichment (teff mostly NaN, none >=25k - acceptable for reddened OB). pytest **696 passed**.

**2026-07-10 snapshot (TODO-12b):** Parallax gate promoted to config (`hrd_parallax_min_mas=0.15`,
`hrd_parallax_snr_min=5`); draft_425 reliable count B/V/R 7989/6651/8011 (was ~1015/795/1021); chi Per
M supergiants in table; binaries capped (<=3 per category, NSS fills budget last). pytest **693 passed**,
15 skipped; PDF overflow 0.

**2026-07-10 snapshot:** TODO-12-HRD: absolute extreme-object selection (Stage 1 net + Stage 2 classify);
`hrd_enrich.py` Gaia TAP + SIMBAD fail-open cache per setup; PDF/UI titles `Field HRD -- <obs_group>`;
lite Gaia DB teff/logg confirmed NULL (211712600 rows, 0 non-null). draft_425 B/V/R HRD membership
differs (19/20/19 candidates); PDF overflow verify 0 violations (draft_425 B_20_2). pytest **691 passed**,
15 skipped.

**2026-07-09 snapshot:** SESSION-CLOSE-0709: Sigma budget Phase A **DONE (wide rig)**; sparse-comp
diagnostics + proposed gate redesign recorded; draft_426 equipment verified eq4 (no DB change).

**2026-07-09 snapshot:** SESSION-CLOSE-0709: draft_426 FITS headers (INSTRUME=C5A-150M, 3552x2664 @
bin4) verify OBS_DRAFT eq4 - GAIN=12.48 anomaly is F-BINGAIN-1 not wrong equipment;
`scripts/fix_draft_equipment.py` + tests. Sigma Phase A wide-rig DONE (6.5 mmag floor, k2 attribution
zero, ~4.5 mmag rig constant). Sparse-comp: ~95% field-wide offset cancels; temporal 8-12 mmag healthy.
pytest **681 passed**, 15 skipped. Pushed to origin/main (Milan-authorized 2026-07-09).

**2026-07-09 snapshot (prior):** SIGMA-A4: wide-rig floor attribution (k2 pooled R^2~0, floor_after k2=6.5 mmag
unchanged); Newton bin4 forensics (header gain 12.48, sigma_ratio~1.13, chi^2_pred~0.78); hypothesis
gain/RN correction moves chi^2 away from 1. pytest **679 passed**, 15 skipped.

**2026-07-09 snapshot (prior):** SIGMA-A3: variant (e) `howell_scint_fresid_floor_ensemble` (+ Honeycutt ensemble SEM);
dual SEM paths (LC decomposition + production `ensemble_normalize`); draft_424 joint refit (d) unchanged
f_resid=0.74 sigma_floor=10.5 mmag, joint (e) f_resid=0.0 sigma_floor=6.5 mmag - prediction
**floor_did_not_collapse**. pytest **674 passed**, 15 skipped.

**2026-07-09 snapshot (prior):** SIGMA-A2: rig fixes (TELESCOPE.DIAMETER 72->200 mm, alt<=0 guard),
`sigma_floor` variant + joint (f_resid, sigma_floor) fit with bootstrap CIs; draft_424 rerun
D=0.2 m alt=275 m, joint fit f_resid=0.74 sigma_floor=10.5 mmag. G9.3 calibrator not saturation-flagged
(fill_max=0.53). pytest **669 passed**, 15 skipped. Pushed e2c9466 (A) + 0b901aa (A2).

**2026-07-09 snapshot (prior):** SIGMA-BUDGET-A + SPARSE-COMP-DIAG: committed `sigma_budget.py` (Howell wrap +
Osborn scintillation), `scripts/chi2_sigma_gate.py`, `scripts/select_constant_calibrators.py`,
`scripts/sparse_comp_diag.py`, `tests/test_sigma_budget.py`. Archive runs:
`tmp/sigma_budget/calibrator_chi2_summary.json`, `tmp/sigma_budget/sparse_comp_diag.json`.
No production wiring; `delta_mag` flux-sum canonical. pytest **666 passed**, 15 skipped.

**2026-07-09 snapshot (prior):** DAO-RECONCILE-CLOSE: flat-curve no-crossing censoring fix (draft_426
G_lim was spurious 13.0, now `>=17.5 no crossing`); 2-pass DAO recovery CLOSED
(not-worth-complexity); PUB-QC-MISSRESIDUAL parked. Completeness 89.7-98.3% across rigs;
miss@G90 health signals live in QA dashboard. pytest **661 passed**, 15 skipped. Chain pushed
to origin/main (Milan-authorized 2026-07-09).

**2026-07-09 snapshot (prior):** DAO-RECONCILE-2b (`78febea`): right-censor G_lim vs reference depth;
`missed_below_g90` / `missed_fadezone` 2-pass metric; `match_depth` forensics; missed-G
histogram in diag. draft_424: compl_50=89.7%, missed=353 but **miss@G90=15** (fadezone=338).
draft_425 B/R: G_lim_50 censored at 17.5 (was spurious 19.25). All-drafts diag:
`tmp/dao_reconcile/cross_draft_summary.json`. pytest **659 passed**, 15 skipped. Chain pushed
to origin/main (Milan-authorized 2026-07-09).

**2026-07-09 snapshot (prior):** DAO-RECONCILE-2 (`bd6244a`...`b7df7c6`): footprint Gaia reference +
Fleming (1995) completeness curve; `completeness_50` headline. draft_424 R-2: G_lim_50~14.97,
completeness_50~89.7%, genuinely-missed=353 (R-1: 96k @ 3.4% - population bug). All-drafts
diag: `tmp/dao_reconcile/cross_draft_summary.json`. pytest **653 passed**, 15 skipped. Anchor
unchanged (`92939fab` / `76642318`). Unpushed local chain.

**2026-07-08 snapshot:** draft_424 coherent anchor `draft_000424_snapshot_20260708_full`
(`run_full_photometry_pipeline`; core SHA `92939fab...` n=357). Hybrid snapshot retired.
`session_baseline_check.py --full` OVERALL PASS (run-2 verification). Ledger VL-ANCHOR-424 +
VL-COUNTERS-ZERO **passing**. Full pytest **631 passed**, 15 skipped. Local chain unpushed until
PROC-STORE-TRUST-FIX push (Milan-authorized 2026-07-08).

**2026-07-08 snapshot (prior):** EXCEPT batch **CLOSED** (BULK-2 applied 98/98 drift rows, `97affe3`).
Validation ledger + session baseline check live (`bfe710e`, `00dd0cd`). Full pytest **625 passed**,
15 skipped. **Next:** `--full` draft_424 anchor verification (session baseline); **PUBLICATION**
venue decision pending (Milan).

**Done 2026-07-08 (DEV-PROCESS group A):** JSON validation ledger + guard test; `session_baseline_check.py`
(`--fast` / `--full` with draft_424 anchor + counters zero-check); EXCEPT-BULK-2 (98 rows);
roadmap closeouts (RECUT-HARNESS-FIDELITY superseded; exoplanet row stale).

**Done 2026-07-08 (EXCEPT batch):** EXCEPT-BATCH-S0 through EXCEPT-RETRIAGE-4 + FIX-1..4 + BULK +
BULK-2; batch **CLOSED**. Census: `docs/VYVAR_EXCEPT_CENSUS.md`.

**Done 2026-07-08 (INPUT-GUARDS-0708):** `resolve_site` null-island guard (`166cbf4`); PDF cfg from
`provenance.config_snapshot` (`80aab21`, synergy with PROV-FIX `e7ce7ea`). `604 passed` gate.

**Done 2026-07-08 (CAL-AGE-CLOCK):** `resolve_master_age` unified import scan + library UI.

**Done 2026-07-08 (PROV-FIX):** `provenance` block in `pipeline_meta.json` via
`merge_photometry_pipeline_meta` when `cfg` passed - `git_hash`, `git_dirty`, full
`AppConfig.to_dict()` snapshot, `stamped_at_utc`, `entry_point`; last-writer-wins at
`run_phase2a` and `generate_masterstar_and_catalog`. Archaeology: never-wired (not regression).
No secrets in `AppConfig` (post-`c26e351` credentials reset).

**Done 2026-07-07 (K2 v1):** Literature k'' path via `k2_extinction.py`; `band_classify` wired to
`resolve_apply_color_term` (CV/CR flip live); LC columns `k2_source`/`k2_value`/`k2_colour_ref`;
NIGHT_FIT deferred (`k2_fit_enabled` OFF). Spec: `docs/VYVAR_K2_DESIGN_SPEC.md`. Validation:
424/425/427 matrix **PASS** (`tmp/k2_land/validation_report.json`, 2026-07-07).

**Done 2026-07-07 (CAL-DIAG-IMPL):** Calibration-time radiometry gate per `VYVAR_CAL_DIAG_SPEC.md`
v1.1 - Check A (SUM/MEAN convention) + Check B (post-dark sky sanity); parent pre-gate for MP
variant (a); provenance headers `VY_DKRSMP`/`VY_CDSKY`/`VY_CDSTAT` + `archive/<draft>/cal_diag.json`.
14 gate unit tests; `549 passed` full suite; draft_424: **150/150** frames `VY_DKRSMP=SUM`, 0
WARN/FAIL, calibrated arrays and photometry science byte-identical to baseline. **RN-HEADER-NONE**
and **CAL-PASSTHRU-DEAD** closed 2026-07-08 (`1830527`, `21c20e3`); **CAL-AGE-CLOCK** closed 2026-07-08 (header-age unified). CAL-AGE-CLOCK was the last open CAL-DIAG ledger item.

**Done 2026-07-07 (session close):** F-BINGAIN-1 Stage A -> **LATENT** (not live on wide rig);
CAL-DIAG workstream registered (calibration radiometry gate; spec pending). F-AIRMASS-CITE fixed;
GAIA-ID guard closed. Commits: `4f18f02` (Fable B+C), `d594b27` (session close).

**Done 2026-07-07 (Fable audit B+C):** Kasten & Young (1989) airmass attribution corrected;
`kastenyoung1989` in `CITATIONS.bib`. GAIA-ID-FLOAT-GUARD closed (live-tree parity check).
F-BINGAIN-1 Stage A diagnostic only - no exponent change yet.

**Done 2026-06-25 (F-BJD-1 Stage D):** per-target LC column `time_base` labels BJD recompute path;
`_recompute_bjd_hjd_with_status` reports cause (`BJD_TDB` vs `JD_FALLBACK`). Purely additive -
`bjd`/`hjd`/`jd` byte-identical. Closes the 2026-06-25 citation/error-model audit.

**Done 2026-06-25 (F-HOWELL-3 Stage C):** explicit annulus-sky column for Howell err; `_photometric_error`
reads `sky_adu_per_px_annulus` with legacy `noise_floor_adu` fallback. Verified on real draft_424
(`run_full_photometry_pipeline`): 178/178 LCs science-identical; sky-dominated err inflation measured
**~12-14%** (detection vs annulus) on faint targets.

**Done 2026-06-25 (citation audit Stage A):** F-RIELLO-1 - B-V/Riello report citation removed
(BP-RP is raw Gaia); F-HOWELL-1 units comment; F-CITE-HONEYCUTT Honeycutt in CORE (`5a1bae0`).

**In-flight / gated:** NIGHT_FIT k'' pre-gate (v2; K2-DATA-BLOCKER). See ROADMAP ACTIVATED v1.

**Done this session (prior):** band classifier (`fe9b375`) - now wired with k'' v1.

**In-flight / PARKED - band-aware k'' (second-order extinction):** **ACTIVATED v1** - see STATE
2026-07-07 K2 paragraph and `VYVAR_K2_DESIGN_SPEC.md`. NIGHT_FIT = v2.

Prior: **2026-06-22** - Forced-aperture / catalog_only removed; DAO+Gaia photometry only. Variable
targets measured **only on direct DAO `catalog_id` hit** (miss -> nondetection/NaN; no XY fallback).
Unmatched VSX excluded in Faza 0. Validated do-no-harm vs draft 419. See DECISIONS.

Prior: **2026-06-19** - Stage B held pending validation (forced-aperture removal draft).

Prior: **2026-06-18** - **Fix C / Phase C1: dense-field alignment DIAGNOSED -> root = PSF/FWHM
bloat; recovery NOT APPLICABLE.** The 14 late-night (post-flip, back-half) frames Fix B drops are **not**
"good data that only failed alignment" - they are **PSF-degraded**: median **FWHM 8.60 px = 1.85x the
good baseline 4.64 px**, concentration flux_large/flux **13.1 vs 1.65**, **corr(FWHM,
alignment-residual)=0.95** (161 frames; `tmp/phaseC1/fixC_root_cause.png`). The bloated-donut centroid
noise (~2.4 px) is the single root - it breaks astroalign (misalignment is the *symptom*) and is what
B.2 (concentration) + Fix-B (residual) measure. Likely **late-night focus drift on the defocused rig**
(a transparency/flux drop alone would not bloat FWHM); post-flip-half-not-refocused is an observer
question. **Not recoverable to sub-px** (centroid floor ~2.4 px > 1.37 px gate; cap50->3/14, WCS absent
0/162, translation-refine inapplicable). **Fix B + B.2 are the correct PERMANENT quality gate** - not a
stop-gap awaiting Fix C. Logged a SEPARATE control-point-cap perf ticket (astroalign mcp~200 -> ~654
s/frame on dense fields; cap ~50 -> ~3-10 s; ROADMAP). **A.B.: Fix A `005716d` + Fix B `fa03410` pushed
to origin/main this session (Milan-authorized).** `CURSOR_RESULT_fixC_diag.md`. See DECISIONS/JOURNAL.
Prior: 2026-06-18 - **Fix B: reject-on-alignment-residual frame gate** (default-OFF;
`frame_align_residual_gate_enabled`). Two additive pieces: (1) **always-on QC** - a per-frame
**alignment residual** (median deviation of bright matched sources from their across-night median
position) is computed at the Phase-2A frame-selection point and recorded as `align_residual_px` in
`alignment_report.csv` (additive metadata -> photometry byte-identical); it reproduces the run-414
diagnostic separation (astroalign med **0.358**/max **1.648** px vs phase_corr min **1.450**/med
**2.130** px). (2) **gate (default-OFF)** - rejects frames whose residual exceeds
`frame_align_residual_max_frac x science-aperture-radius-px` (**rig-agnostic** fraction, default
**0.25** -> 1.37 px, in the 1.206->1.450 px good/bad gap; safety floor `min_keep_frames`). Verified on
run-414 g: **OFF byte-identical** (70 targets, V0454 `mag_calib`/`delta_mag`/`err` max|diff|=0); **ON
drops 14 frames = all 13 phase_correlation + 1 mis-aligned astroalign** (dr=1.648, itself an LC
outlier) - V0454 outliers 22->10, the catastrophic +3.7 mag/NaN points gone (clean SIPS-grade egress;
`tmp/fixB_v0454.png`). **B.2 cross-check:** residual gate supset B.2 (overlap 13, residual-only the 1
astroalign, B.2-only 0) - cause-correct (alignment) superset of B.2's aperture-integrity symptom; both
kept distinct. **[C1 correction: PERMANENT gate, not "self-deactivating once Fix C fixes alignment" -
the frames are PSF/FWHM-bloated and unrecoverable.]** See DECISIONS/JOURNAL.
Prior: 2026-06-18 - **Fix A: per-point error model bug fixed** (default; no flag). The LC
`err` term-3 was `np.std(comp instrumental mags)/sqrtn` (`photometry_core.py:2567`) - for a sparse/
brightness-spread ensemble this is the comps' brightness *spread* (a fixed ~0.58 mag floor on V0454,
23x the empirical 0.025), not a per-point uncertainty. Replaced with the per-frame **ensemble-ZP
standard error from comp residuals** (each comp vs its own across-night median -> brightness/colour
cancels; Honeycutt 1992); the redundant `comp_rms/sqrtn` term-2 was dropped (no double-count); photon
term-1 (incl. SNR-blowup on bad frames) kept. Verified on run-414 g: centres `mag_calib`/`delta_mag`
**byte-identical**, V0454 err 0.581->0.013 (~empirical), faint targets photon-dominated, the 13
mis-aligned frames still flagged (Fix B). `err` does NOT feed trust/lc_rms/production-Broeg-combine;
it does feed SysRem IVW weights (default-OFF) - improved, not broken. See DECISIONS/JOURNAL.
Prior: 2026-06-17 (end-of-day) - clean committed **+ pushed** baseline at `955b850`
(8 commits: `1eea2d2` masterstar recovery, `e042bc1` A-durable, `d222eb7` B-cap, `2cc2b76`
completeness gate, `63e57c0` log-flood, `a126980` B.2 gate, `15c699e`/`955b850` docs). `draft_413` =
Boyden V454 CrA non-cal sandbox (g+r; **g fully validated** this session). Validated this session:
non-cal ingest, headless run, meridian-flip handling, Brno gate, B-cap, B.2 (default-OFF). **V0454 CrA
flip diagnostic:** the 0.45 mag rise = real eclipse egress (~0.37 mag, comp-invariant, SIPS-corroborated)
dominating ~4:1 over a ~+0.1 mag position-dependent meridian-flip step (explains the 0.45-vs-SIPS-0.548
gap as comp choice, not pixels; see DECISIONS + `docs/round2_figs/v0454_flip_diag.png`). **Pending:
UI-VYVAR live test of A-durable** (ROADMAP).
Prior: 2026-06-17 (Part A clean baseline committed [6 commits, push gated]; Round 2:
B.1 aperture-skirt **refuted** by COG/scatter diagnostic [not implemented]; B.2 transparency
**frame-quality gate** implemented behind default-OFF `frame_quality_gate_enabled` -- isolated
measurement on draft_413 g cuts bright-target LC scatter by median -257 mmag, trust still RED
[structural check-star/comp]. See `CURSOR_RESULT_round2.md`).
Prior: 2026-06-17 (Round 1 four known fixes verified on draft_413 g: A-durable MP-reload
robustness, B-cap spatial-first variable_targets [+comp-purity coupling, Milan-accepted], measurable
completeness gate, NoDetections log-flood summary; simple-differential PRODUCTION).
Prior: 2026-06-16 (Phase-1 graceful comp degradation committed + matrix `164157` validated;
known-issue (b) closed).

This is the **entry point**: a snapshot of what is true *now* + an index. It deliberately holds
no history and no open-task detail -- those live in the linked files.

| File | Holds |
|------|-------|
| `docs/VYVAR_ROADMAP.md` | Open work (the only place to look for "what's next"). |
| `docs/VYVAR_DECISIONS.md` | Durable design decisions + *why* they hold. |
| `docs/VYVAR_JOURNAL.md` | Chronological session log (history, append-only). |
| `docs/VYVAR_PROCESS.md` | How we work: Definition of Done, validation discipline, config<->UI parity, tests. |
| `docs/VYVAR_CLAUDE_OPERATING_PRINCIPLES.md` | Claude operating charter (session-init required read; governs reasoning and answers). |
| `docs/VYVAR_PARAMS.md` | Config-key <-> default <-> clamp <-> UI-location registry. |
| `docs/VYVAR_DECISION_GROUNDING_RULE.md` | Adopted rule: cite physics/literature/practice before design forks. |
| `docs/VYVAR_REPORTING_COLUMN_GROUNDED_DECISION.md` | Workstream B reporting column (supersedes B1/B2). |
| `docs/VYVAR_CANONICAL_COMBINATION_LOGIC.md` | Flux-sum vs Broeg IVW -- conditional hold until sigma budget. |
| `docs/VYVAR_SIGMA_BUDGET_SPEC.md` | PARKED sigma-budget work item (Howell + scintillation + chi-squared gate). |
| `docs/VYVAR_VALIDATION.md` | Inject-and-recover synthetic validation harness (matrix, FAIL policy). |
| `docs/VYVAR_PIPELINE_CZ.md` | Czech pipeline manual for the paper (ASCII, rev. 2026-06-09). |
| `docs/VYVAR_CALIBRATION.md` | Magnitude calibration data-flow (`mag_calib_final`, CT/AC, consumers). |
| `docs/VYVAR_GAIA_DR3_AUDIT.md` | Gaia DR3 ingest audit (build schema, match, ref mag; 2026-06-10). |
| `docs/VYVAR_LC_QUALITY_SHORT_BASELINE_SPEC.md` | `short_baseline` LC-quality spec #3 (rev b, ready; 2026-06-10). |
| `docs/VYVAR_RUNBOOK.md` | Chi_and_H zaloha-only night-run procedure (alias -> baseline runbook). |
| `docs/VYVAR_CHIANDH_BASELINE_RUNBOOK.md` | Chi_and_H baseline re-cut procedure (byte-identity anchor; 2026-06-11). |
| `docs/VYVAR_TRUST_CHECKSTAR_HARDENING_SPEC.md` | Trust Findings A/B + CS-1 hardening (2026-06-11). |
| `docs/VYVAR_CHECKSTAR_SELECTION_SPEC.md` | Check-star selection CS-2..4 (2026-06-11). |
| `docs/VYVAR_COMP_FLOOR_POLICY_SPEC.md` | Comp trust floor policy; Option B adopted. |
| `docs/VYVAR_MATH_PHYS_AUDIT.md` | Math/physics audit (first pass; citation scoping landed). |
| `docs/VYVAR_COMP_DEGRADATION_SPEC.md` | Phase-1 graceful comp degradation spec (committed 2026-06-16). |

---

## Mission

A high-automation differential-photometry pipeline that lets amateur astronomers contribute
science with confidence: **trust in the numbers** (comp-stability QA via comp_qa + per-target
trust gate) and **guardrails for non-experts**. Independent extraction cross-validation lives
in the offline `xval_run.py` harness (not in-pipeline). **Aperture photometry remains the
validated workhorse** on the wide rig; **PSF is validated publication-grade at fine scale on
synthetic truth** (draft-367-like, mismatch ~0) but **gated OFF** in production LC until a
real Newton / dense-field draft passes the characterization gate.

## Pipeline (current)

```
Raw -> Calibrate -> QC (in-place) -> Align -> MASTERSTAR + Gaia DR3 catalog
   -> Phase 0+1: tier-ladder comp selection (colour window + RMS rank; bounds 3/8)
   -> Phase 2A: simple differential photometry
        (flux-sum ensemble; SNR-opt per-star aperture; NO temporal comp binning)
        -> reporting postprocess (ensemble ZP mag; NO per-target airmass detrend;
           mask-first outlier guard for known variables)
        -> comp stability QA (per-frame ensemble residual p2p)
   -> comp_qa (Sokolovsky LOO QA, read-only)
   -> trust gate (GREEN/YELLOW/RED; comp-health + check-star + lc_quality + stability)
   -> reports/exports (PDF SUMMARY MEASURE REPORT, AAVSO, VarAstro)
```

Plate scale is **WCS-derived** (~9.77 arcsec/px on the wide rig). Ensemble combine is **flux-sum**
(`delta_mag` canonical; AIJ/SIPS validated). Broeg inverse-variance ensemble combine is **PARKED**
until sigma budget validates (`docs/VYVAR_SIGMA_BUDGET_SPEC.md`). Comp selection ranks colour tier
first, stability second (both gated), proximity as a distance gate only. (See DECISIONS.)

## Production defaults (feature flags)

| Area | Flag / behaviour | Default |
|------|------------------|---------|
| Comp temporal binning | `temporal_binning_enabled` | **OFF** (ALG-3 breaks common-mode) |
| Color term | `apply_color_term` | **OFF** (colour-matched comps) |
| Comp selection | tier ladder 0.15/0.30/0.55, cap 0.79; bounds 3/8 | `_select_comps_by_color_then_rms` |
| Comp RMS floor | `comp_select_rms_floor` | **1e-6** (drop isolated-bin artefact) |
| Reporting | `apply_reporting_postprocess` | ensemble ZP `mag_calib`; no target airmass detrend; mask-first outliers |
| Comp stability | `check_comparison_stability` | p2p on **per-frame ensemble residual** (not raw `mag_inst`) |
| LC precision display | `lc_rms_ooe` on card | brightest-tertile scatter for variables |
| Aperture on card | `aperture_px` | **measured** proc `aperture_r_px` (not Phase-2A replan) |
| Comp QA | `comp_qa_enabled` | **ON** |
| Trust gate | `trust_flag_enabled` | **ON** (GREEN observed on draft_409 V0612) |
| Proximity tie-break | `phase01_comparison_proximity_tiebreak` | OFF |
| PSF (all) | `psf_photometry_enabled`, `psf_adaptive_enabled`, `psf_grouper_enabled`, `psf_spatial_enabled` | OFF |
| NEIGHBOR-SUB | `psf_neighbor_sub_enabled` | OFF |
| COG aperture corr. | `cog_aperture_correction_enabled` | OFF |
| Crowding classifier | `crowding_classifier_enabled` | OFF (wide rig) |
| Sparse comp fallback | `comp_sparse_fallback_enabled` | **ON** (per-target; inert on rich anchor) |
| Detrend | `sysrem_enabled`, `savgol_detrend_enabled` | OFF |
PSF flags stay **OFF** on the wide rig (correct). The PSF path is now **validated-but-gated**:
enable only on characterized fine-scale data after the Brno / Newton characterization gate
(see DECISIONS + ROADMAP).

Comp bounds (user-configurable): Phase-1 selection `phase01_comparison_n_comp_min/max` = **3 / 8**
(unchanged). Trust-only floor `comp_trust_min_comps` = **3** (Phase-1; GREEN requires >=3 clean T1/T2
comps + check); `check_star_min_epochs` = **5**; CS-2 artefact floor `check_select_rms_floor` =
**1e-4**; CS-4 uses `aperture_correction_max_contamination` = **0.15** when
`contamination_idx` is present. `max_comp_rms` = 0.1; colour cut <= 0.79.

### Phase-1 graceful comp degradation (2026-06-16)

**Status:** committed + structurally validated (matrix `164157`; check-star preselect active).

Graded routing keeps 1-N good comps on default path (sparse only at 0); honest `comp_rms` /
`comp_rms_fieldwide` split; `comp_path` on summary + PDF; sigma scales with N; SS Cam fold
(pool attach, check-star field preselect).

### Phase-1b: per-target comp_rms gate authoritative for N_good (2026-06-16)

**Status:** committed (gate-authority part of known-issue (b) **CLOSED**). The per-target
`max_comp_rms`=0.1 gate is now the hard quality bar for N_good. RMS fallback no longer relaxes above the
gate (the `0.15` step is gone); auto-routing counts gate-passers (`_count_gate_passing_comps`), not raw
`len(result)`. Matrix re-run `185831`: SS Cam flips **default -> sparse_fallback** (its 0.134 comp fails
the gate, no longer a good default comp); V0612 + BO CVn + V0842 Her unchanged.

**OPEN -- SS Cam trust band (RED vs YELLOW) is UNRESOLVED, not closed.** SS Cam came out **YELLOW**, not
the predicted RED, but whether YELLOW is the grounded-correct band is **not yet decided**. The tension:
the sparse comp_rms (~0.35 mag) is a **field-wide-scale** quantity (different definition from the 0.1
per-target gate), and the check-star scatter (0.043 < 0.05 hard line) is **ensemble-dependent** -- comps
look bad, check looks OK, and neither has been verified. Resolve **diagnostic-first** in Phase-2 (does
field-wide sparse comp_rms cancel in the differential? is check-0.043 reliable given N points /
baseline?) **before** setting any sanity-ceiling threshold. Do NOT reverse-engineer RED. No threshold
re-tuning was done here.

## Rigs (known sets)

| ID | Telescope | Camera | Scale | Site |
|----|-----------|--------|-------|------|
| 1 | Carl-Zeiss 200 mm | QHY294MM | ~9.77 arcsec/px (wide) | Jirny |
| 2 | Newton 300/1200 | C3-26000 | ~0.65 arcsec/px (fine) | Dablice |
| 3 | Noctutec 206/560 f/2.72 | C3-26000 | TBD | TBD |

Per-set config architecture is still pending (ROADMAP: TODO-MULTISET).

## Status snapshot

### Gaia DR3 catalog integration

PM (`pmra`/`pmdec`) and `ruwe` are **NOT** in the DR3 catalog; **deferred to the DR4 build**
(~Dec 2026). Platesolver PM propagation is present but a no-op against DR3. Fine-scale dense
fields carry the GAIA-1 mis-association caveat until DR4 (DECISIONS).

### Brno AZ800 / C5A-150M (production solver - 2026-06-14)

**Brno AZ800 / C5A-150M onboarded.** Production solver uses **catalog-recovery verification**
(Gaia-in-frame / DAO at 2.5 px) as the MASTERSTAR accept gate; detection match% is informational.
Stale FITS pointing (`VY_TARG`) -> **`hint_sep_warn`** when VERIFIED (Lang et al. 2010 prior), not
hard reject. Cone recenter at solved center when hint offset **>= 0.05 deg** unchanged.
`generate_masterstar_and_catalog` passes `app_config` + scoped flags.

**Brno `r_60_4`:** catalog recovery tight **~84%** -> **VERIFIED** under new gate (was rejected on
`hint_sep` + detection-denominated metrics). **`z_90_4`:** recovery **~34%** -> stays rejected.
**Open:** Milan overlay sign-off on `tmp/diag_overlay_r.png`; anchor + home-rig regression re-run.

### Comp sparse-only fallback (2026-06-11 lock)

**Sparse-only fallback live (default ON).** Historical byte-identity anchor `3f7c9e7a` / `d5b72d08`
retired by simple-differential algorithm change; regression now uses empirical SIPS/AIJ cross-validation
on V0612 plus `compare_photometry_science_meaningful` for archaeology vs the zaloha cut.

### Reference draft and validation (not byte-identity)

The simple-differential algorithm change **retired** the old photometry SHA byte-identity anchors
(by design). Validation is now **empirical cross-validation** vs AIJ/SIPS on V0612:

- Out-of-eclipse RMS ~0.011 mag; eclipse shape correlation ~0.95+.
- draft_409 (2026-06-16): eclipse + single shared bright outlier at ~JD 2461200.385 matches SIPS
  -> frame-level artifact (cosmic-ray-like on target), not a VYVAR reduction bug.

Historical SHA anchors (`3f7c9e7a` / `d5b72d08`, Chi_and_H chi Per zaloha cut) remain documented for
regression archaeology; current code does not byte-reproduce them. Optional fresh anchor cut after
Milan sign-off (see ROADMAP / JOURNAL).

**Regeneration recipe (historical zaloha anchor)** (`docs/VYVAR_CHIANDH_BASELINE_RUNBOOK.md`):

1. **Source data (must retain):** `Archive/Chi_and_H` - pre-calibrated FITS (only non-regenerable input).
2. **Catalog + blind index:** `GAIA_DR3/zaloha/vyvar_gaia_dr3.db` (G<=16), zaloha blind PKLs
   (`gaia_triangles_fine.pkl`, `gaia_triangles_wide.pkl`). **Do not read** in-progress
   `GAIA_DR3/vyvar_gaia_dr3.db`.
3. **Run:** `python scripts/chiandh_night_run_bvr.py` (#3 code; Newton bin2 ~1.30"/px).
4. **Verify:** `compute_photometry_sha(draft_root)` core + full vs recorded SHAs (`3f7c9e7a...` /
   `d5b72d08...`). For regression vs the historical cut, use
   `compare_photometry_science_meaningful` (PROCESS) - excludes provenance/`err` QC drift.

**Setups (filter-wheel labels):** `B_20_2`, `V_20_2`, `R_20_2`, `L_20_2` - **B/V/R/L** are wheel
positions. **V** = visual/green (`G/` folder); **L** = clear/broadband (`L_20_2` in anchor).

**Provenance at anchor cut (2026-06-11, zaloha):**

| Item | Value |
|------|-------|
| `git_commit` | `7317ece87944b749461a7b6abca6615f1a30dc72` (re-baseline lock) |
| Catalog | `GAIA_DR3/zaloha/vyvar_gaia_dr3.db` (G<=16) + zaloha blind PKLs |
| Rig | Newton 300/1200 + C3-26000, bin2 ~1.30"/px |
| Ephemeral draft | `draft_000386` (~1401 LCs; deletable) |
| Completeness gate | `night_run.audit_photometry_completeness` - >=90% summary/active per setup |

Scoped trust at cut (`comp_trust_min_comps=5`, floor-5 baseline): **1382 YELLOW / 106 RED**
(1488 summary rows; re-trust on draft_387). Pre-floor-5 counts were 1400/88 - superseded.
Anchor photometry SHA is numeric and trust-independent.
- **Last science-validated wide draft:** `draft_000365` (V842 Her, 127 frames, 143 targets).
- **CT science locked:** h & chi Per `draft_000380` (Johnson-Cousins B/V/Rc); details in JOURNAL.

### PSF photometry (gated OFF in production LC)

Infrastructure complete. Validated **publication-grade on synthetic fine-scale** truth
(draft-367-like, ePSF-vs-star mismatch ~0):

| pillar | result (V3d harness, mag 12-17) |
|--------|----------------------------------|
| ACCURACY | mid-mag bias **<~2%** via brightness-independent **sky-only fit weights** (`psf_weight_mode=sky_only`; Astier 2013 / Lacroix 2025) |
| PRECISION | PSF scatter wins from ~mag 13 |
| UNCERTAINTY | P3 ~1 via **sandwich** variance (`psf_err_mode=sandwich_skyonly`) |
| ePSF FWHM QC | robust azimuthal-profile estimator (EPSF-1); warning band [0.80, 1.25] (diagnostic only) |

Sky estimate: aperture-consistent **annulus** / **residual_annulus** (`psf_sky_method` column).
Real-field enablement **blocked on a Newton / dense-field draft** (incoming Brno data will
unblock after the characterization gate).

### NEIGHBOR-SUB

**VALIDATED_FINE_SCALE_IDLE** -- works at fine scale (A9 HV ~83%, FAIL-SILENT 0), fail-safe
guards + full provenance, gated OFF; no current real use case (draft 367 sparse crowding).
Coarse / under-sampled fields fall back to SAFE_LOW_YIELD (correct REFUSE, not silent deblend).

### Fail-safety #4 (2026-06-08)

- MASTERSTAR WCS persist: **fail-closed** (draft solve fails; Phase 2A blocked for that draft).
- Edge-ok check: **fail-open + loud flag** (`edge_filter_failed` on `variability_candidates.csv`).
- Dead UI modules removed (`ui_photometry_results`, `ui_suspected_lightcurves`).

### Citations (PSF arc)

Astier et al. 2013, Lacroix et al. 2025, Guy et al. 2010, Stetson 1987, Mighell 1999 wired in
`CITATIONS.bib` where the methods run.

### Cross-validation, trust, tests, reporting

- **Cross-validation:** CLOSED for the aperture path (offline `xval_run.py`: sep reproduces
  VYVAR to 0.2 %/frame); in-pipeline `sep_xval` retired 2026-06-03; PSF cross-val deferred.
- **Trust distribution (draft_000365 baseline):** GREEN 69 / YELLOW 59 / RED 15.
- **Tests:** **963 passed / 19 skipped** (full `dev/tests/` run; 2026-07-18, DOCS-FIX-ARC1).
- **Lint:** `ruff check . --select BLE001,E722` clean (`pyproject.toml` + pre-commit + pytest).
- **Reporting:** R1 overflow guarantee holds (0 violations); R3 (aperture-vs-PSF overlay) pending.

---

## Top of mind

**Next when resuming:** Data-gated backlog only (ROADMAP). Startup: `git pull` -> STATE ->
ROADMAP -> `session_baseline_check.py --fast` -> await Milan data (fresh darks ~2026-07-21 first).

**Simple differential photometry is PRODUCTION** (SIPS/AIJ cross-validated). Canonical column:
`delta_mag` flux-sum; reporting `mag_calib` via `apply_reporting_postprocess`. Broeg IVW **PARKED**
until sigma budget validates (`docs/VYVAR_SIGMA_BUDGET_SPEC.md`).

**Band-aware k'' v1 ACTIVATED** (literature path); NIGHT_FIT v2 gated on filtered draft + dX>=0.3.

**Deferred findings (none blocking -- see ROADMAP parked rows):** GAIA-ID-FLOAT-GUARD; G7-F003c;
EQUIP-BINNING-ASYM; dense-field astroalign cap (cross-rig regression pending).
