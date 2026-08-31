# VYVAR - Roadmap (open work)

Single source of truth for **open** tasks. Closed work lives in `VYVAR_JOURNAL.md`;
durable rationale in `VYVAR_DECISIONS.md`; current architecture in `VYVAR_STATE.md`.

Reconciled against the full development log on **2026-07-14** (SESSION-CLOSE-0714) -- standing
open-items table reduced to data-gated backlog + parked rows with revisit triggers.

Priority legend: **HIGH / MEDIUM / LOW / FUTURE**. Each item is a short status, not a history.

---

## NEXT SESSION - entry point (2026-08-27 CLOSE-OUT APERTURE-01d LOCK; era04 locked)

Local `main` tracks `origin/sel-ghost-01`. Push incident: do not
force-push `origin/main` (now `7c086e8` lineage). Pushes name the
ref. APERTURE-01d: annulus 2.7/5.2, independent AIJ gate PASS
(1.9503 mmag). Ledger v6: 0 UNNAMED. era04 is the `--full` gate
(core 9367f998 n=160 / ext d3cefff3 n=210). config f=1.35,
annulus 2.7/5.2. era03 freeze kept on disk. SEL-GHOST-01 not
closed. origin/main stays `7c086e8` until Milan writes PUSH_AUTH.

| Step | Action |
|------|--------|
| **1** | EDGE-ANNULUS-01 (record only): aperture on-chip + annulus >=50% on-chip (masked) vs current NAXIS-r_out dropping FR CVn |
| **2** | First AAVSO/VarAstro uploads **BO -> FW** (band **CV**) -- ledger locked |
| **3** | MULTIFILTER-WCS-01 sibling-seed for z_90_4 |
| **4** | INPUT-PATH-ARCH-01 discussion (non-cal stays; raw-without-masters split) |
| **5** | **EXPORT-PARITY-01** (HIGH) |

| Pri | Carry list |
|-----|------------|
| **HIGH** | **EPSF-SHAPE-01** - root narrow ePSF core OPEN (FWHM 2.36 vs 3.30); routed to EPSF-CORE-01 |
| **HIGH** | **EXPORT-PARITY-01** - standing two-path defect (R5 audit); PSF merge path fixed |
| **MED** | **EPSF-XVAL-01** - external ePSF gate: same ensemble, same frames, independent PSF photometry reference (e.g. DAOPHOT/ALLSTAR or photutils forced). Method to be specced with literature when scheduled. |
| **MED** | **EPSF-ZP-OK-XRIG-01** - extend `fit_ok_for_zp` past wide `1:1`. Requires (1) master dark+flat in CalibrationLibrary for that rig and (2) CENSUS-01 replay of pin-drop vs quality. Newton 518 gated pool 26 does not qualify. |
| **MED** | **MULTIFILTER-WCS-01** - sibling-seed for z_90_4 (see prior entry) |
| **MED** | **FRAME-QC-PARITY phase 2** (Layer A log honesty + QC provenance stamp). C8-2: draft 516 frame 29 admitted (`status=ok`) with `n_stars_detected=263` vs ~100 on 028/030; FWHM/elongation normal. Do not wire in C8. |
| **LOW** | **DEPTH-AUTH-01** - derive `masterstar_gaia_census_target_depth_g` from MASTERSTAR completeness vs Gaia (DAO-GAIA certificate bins). Not wired. G=15.56 VSX stays absent at re-cut. |
| **FUTURE** | **EPSF-CORE-01** - literature-parameter ePSF rebuild |
| **FUTURE** | **EPSF-PERF-01** - forced linear refit path; deferred by Milan. |
| **OPEN** | **INPUT-PATH-ARCH-01** |

## NEXT SESSION - prior (2026-08-26 CLOSE-OUT APERTURE-01c STOP; era04 not locked)

Local `main` tracks `origin/sel-ghost-01`. Push incident: do not
force-push `origin/main` (now `7c086e8` lineage). Pushes name the
ref. APERTURE-01c: f=1.35 set, independent AIJ gate PASS
(2.7833 mmag). Ledger v5 vs era03: 6 UNNAMED, so era04 is not
locked. SNAPSHOT_NAME stays era03. config f stays 1.35.
SEL-GHOST-01 not closed.

| Step | Action |
|------|--------|
| **1** | Name the 6 UNNAMED (lost LC at r=7 / Phase1 n_survivors<3 / dmag undefined) or drop them from the union before lock |
| **2** | Then C6-2 `--full` twice on era04 and lock (C6-4) |
| **3** | EDGE-ANNULUS-01 (record only): aperture on-chip + annulus >=50% on-chip (masked) vs current NAXIS-r_out dropping FR CVn |
| **4** | MULTIFILTER-WCS-01 sibling-seed for z_90_4 |
| **5** | INPUT-PATH-ARCH-01 discussion (non-cal stays; raw-without-masters split) |
| **6** | First AAVSO/VarAstro uploads **BO -> FW** (band **CV**) -- hold until a locked ledger exists |
| **7** | **EXPORT-PARITY-01** (HIGH) |

| Pri | Carry list |
|-----|------------|
| **HIGH** | **EPSF-SHAPE-01** - root narrow ePSF core OPEN (FWHM 2.36 vs 3.30); routed to EPSF-CORE-01 |
| **HIGH** | **EXPORT-PARITY-01** - standing two-path defect (R5 audit); PSF merge path fixed |
| **MED** | **EPSF-ZP-OK-XRIG-01** - extend `fit_ok_for_zp` past wide `1:1`. Requires (1) master dark+flat in CalibrationLibrary for that rig and (2) CENSUS-01 replay of pin-drop vs quality. Newton 518 gated pool 26 does not qualify. |
| **MED** | **MULTIFILTER-WCS-01** - sibling-seed for z_90_4 (see prior entry) |
| **MED** | **FRAME-QC-PARITY phase 2** (Layer A log honesty + QC provenance stamp). C8-2: draft 516 frame 29 admitted (`status=ok`) with `n_stars_detected=263` vs ~100 on 028/030; FWHM/elongation normal. Do not wire in C8. |
| **LOW** | **DEPTH-AUTH-01** - derive `masterstar_gaia_census_target_depth_g` from MASTERSTAR completeness vs Gaia (DAO-GAIA certificate bins). Not wired. G=15.56 VSX stays absent at re-cut. |
| **FUTURE** | **EPSF-CORE-01** - literature-parameter ePSF rebuild |
| **OPEN** | **INPUT-PATH-ARCH-01** |

## NEXT SESSION - prior (2026-08-26 CLOSE-OUT APERTURE-01b STOP; no f*)

APERTURE-01b accuracy grid: no f*. config was 0.385228 then.
era03 remains the `--full` gate. Evidence: `CURSOR_RESULT_APERTURE_01B.md`.

## NEXT SESSION - prior (2026-08-26 CLOSE-OUT APERTURE-01 STOP; era04 not locked)

APERTURE-01 wired (option i) but P-A1/P-A3 FAIL at ladder
f=0.385228. PUSH_AUTH SHA `a23ee3d` on origin/sel-ghost-01.

## NEXT SESSION - prior (2026-08-26 CLOSE-OUT C6-3d STOP; era04 not locked)

C6-0 A-only (6 files) imports. Chain FAIL: iter4 4-tuple vs
c592ecf `lock_existing` 3-tuple. M not copied. era04 not locked.

## NEXT SESSION - prior (2026-08-26 CLOSE-OUT C8+C4 STOP; C6 waits GO)

Local `main` tracks `origin/sel-ghost-01`. Push incident: do not
force-push `origin/main` (now `7c086e8` lineage). Pushes name the
ref. C8 STOP. C4 ZP-OK v2 wired wide-rig only (`1:1`). C6 full-chain
era04 waits Milan GO in chat (named-cause ledger; never overwrite
era03). C6 GO then STOPPED at C6-0 (missing `dao_gaia_stage_01`).

## NEXT SESSION - prior (2026-08-26 CLOSE-OUT C8 STOP; C4 next; C6 waits)

Local `main` tracks `origin/sel-ghost-01`. 2026-08-25 push incident:
`origin/main` is at `7c086e8` (same tip); do not force-push back to
`b1f5b8c`. Pushes name the ref. C8: R1' one-file copy cannot import
(iter4 needs iter2/iter3, all first-tracked at `0684ba9`). Frame 29
QC admitted (n_stars 263). DEPTH-AUTH-01 recorded, not wired. C4
ZP-OK v2 is the next wire. C6 waits Milan GO after C4 STOP.

| Step | Action |
|------|--------|
| **1** | C4 EPSF-ZP-OK-01-WIRE v2 (W1-W4; push `sel-ghost-01` by name) |
| **2** | Milan: C6 full-chain era04 GO (after C8 + C4 STOPs; named-cause ledger) |
| **3** | MULTIFILTER-WCS-01 sibling-seed for z_90_4 |
| **4** | INPUT-PATH-ARCH-01 discussion (non-cal stays; raw-without-masters split) |
| **5** | First AAVSO/VarAstro uploads **BO -> FW** (band **CV**) |
| **6** | **EXPORT-PARITY-01** (HIGH) |

| Pri | Carry list |
|-----|------------|
| **HIGH** | **EPSF-SHAPE-01** - root narrow ePSF core OPEN (FWHM 2.36 vs 3.30); routed to EPSF-CORE-01 |
| **HIGH** | **EXPORT-PARITY-01** - standing two-path defect (R5 audit); PSF merge path fixed |
| **MED** | **MULTIFILTER-WCS-01** - sibling-seed for z_90_4 (see prior entry) |
| **MED** | **FRAME-QC-PARITY phase 2** (Layer A log honesty + QC provenance stamp). C8-2: draft 516 frame 29 admitted (`status=ok`) with `n_stars_detected=263` vs ~100 on 028/030; FWHM/elongation normal. Do not wire in C8. |
| **LOW** | **DEPTH-AUTH-01** - derive `masterstar_gaia_census_target_depth_g` from MASTERSTAR completeness vs Gaia (DAO-GAIA certificate bins). Not wired. G=15.56 VSX stays absent at re-cut. |
| **FUTURE** | **EPSF-CORE-01** - literature-parameter ePSF rebuild |
| **OPEN** | **INPUT-PATH-ARCH-01** |

## NEXT SESSION - prior (2026-08-25 CLOSE-OUT C3/C7 STOP)

Local `main` tracks `origin/sel-ghost-01`. `origin/main` stays
`b1f5b8c` until C6. C3 wired (k=5). C7: R1 not a pre-B1 control;
C6 not asked (VSX row lost from MS). C4 ZP-OK v2 is on disk
(`CURSOR_TASK_EPSF_ZP_OK_01_WIRE_v2.md`); Push: NO.

| Step | Action |
|------|--------|
| **1** | C4 EPSF-ZP-OK-01-WIRE v2 (W1-W4; Push: NO) |
| **2** | Milan: C6 full-chain re-cut GO (C7 lost-MS-row + contaminated R1 are facts) |
| **3** | MULTIFILTER-WCS-01 sibling-seed for z_90_4 |
| **4** | INPUT-PATH-ARCH-01 discussion (non-cal stays; raw-without-masters split) |
| **5** | First AAVSO/VarAstro uploads **BO -> FW** (band **CV**) |
| **6** | **EXPORT-PARITY-01** (HIGH) |

## NEXT SESSION - prior (2026-08-25 CLOSE-OUT C1/C2 STOP)

Local `main` `78b3495` tracks `origin/sel-ghost-01` (same SHA).
`origin/main` stays `b1f5b8c` until C6. C0 hygiene pushed. C1: R1-vs-R0
is freeze lag; census 4+1 IDs unexplained; no fix commit. C2: comp_rms
is mag-bin relative flux; C3 waits Milan GO. C4 ZP-OK v2 locate-fail
(re-issue v2). C6 re-cut waits Milan GO in chat after C1 STOP (full
chain, not photometry-only). Live 516/520 SHA unchanged.

| Step | Action |
|------|--------|
| **1** | Milan: GO C3 (COMP-RMS-DEF-01-B + ZONE-SAT-01 wiring) or hold |
| **2** | Milan: C6 full-chain re-cut GO (or wait on census 4+1 unexplained) |
| **3** | Architect re-issues EPSF-ZP-OK-01-WIRE v2 (locate failed) |
| **4** | MULTIFILTER-WCS-01 sibling-seed for z_90_4 (15.511 UI overwrite of DB scale fixed at `1ec035f`; not a new DAO-SCALE-AUTH-01) |
| **5** | INPUT-PATH-ARCH-01 discussion (non-cal stays; raw-without-masters split) |
| **6** | First AAVSO/VarAstro uploads **BO -> FW** (band **CV**) |
| **7** | **EXPORT-PARITY-01** (HIGH) |

## NEXT SESSION - prior (2026-08-25 SEL-GHOST-01 B-STOP-3)

Local `main` ahead of `origin/main` `b1f5b8c`. Push: NO until Milan
reviews B-STOP-3. Production-path 516 three-way measured. Re-cut of
the era03 frozen snapshot is a Milan GO, not done in this task.
V0612 still one-comp at ceiling 0.1 (forced-7 script lc_rms 0.053).

| Step | Action |
|------|--------|
| **1** | Milan: review B-STOP-3 re-cut proposal (R2 candidate SHA vs anchor) |
| **2** | `--full` recut of 9902d918 / 472bc9e4 only after Milan GO |
| **3** | COMP-RMS-DEF-01: 0.1 mag vs photon (T4 table; no wiring yet) |
| **4** | MULTIFILTER-WCS-01 sibling-seed for z_90_4 |
| **5** | INPUT-PATH-ARCH-01 discussion (non-cal stays; raw-without-masters split) |
| **6** | ZONE-SAT-01 (G=7.63 peak 88781 zone=linear) |
| **7** | First AAVSO/VarAstro uploads **BO -> FW** (band **CV**) |
| **8** | **EXPORT-PARITY-01** (HIGH) |

## NEXT SESSION - prior (2026-08-24 REG-520-01 STOP)

Local tip **`92361a3`** plus SESSION-CLOSE-20260824 stack (AC-02 +
REG-520-01; not pushed until this close). Production 516 ePSF SHA
`172f95403beae36d...` unchanged. Non-cal is first-class (Milan).
0.39 vs 0.06 is S2 selection, not the button. z_90_4 remains
solve-rejected. AC-02 wiring is in the close stack.


| Step | Action |
|------|--------|
| **1** | Milan: GO S2 selection-input fix (rms ceiling 0.1 + no Gaia-DAO residual / visibility gate; comps must sit on the Gaia star) -- REG-520-01 menu 1 |
| **1a** | Optional later: production pass2/seed floor = f(solve rms, FWHM); cite M2 curve; i_70_4 is the worst live gap -- menu (a) |
| **1b** | Milan: GO `calibration_mode=non_cal_declared` banner + cautious LC class + submit lock -- menu (b) |
| **1c** | One-line UI: add `time_base` to `_LC_OVERVIEW_COLS` -- menu (c) |
| **1d** | PRECAL metric stays informative, does not block -- menu (d); PRECAL-INPUT-CONTRACT-01 as a blocking gate is the wrong direction |
| **2** | Milan: Newton 518 ePSF - hold for a night with gated pool >=30, or GO a wide-only ZP-OK without claiming Newton |
| **3** | Milan: GO or hold interim `psf_fit_ok_for_zp` (wide-rig CENSUS-01 still stands; no Newton P-D) |
| **4** | SESSION-CLOSE-20260824: AC-02 wiring + REG-520-01 results (this close) |
| **5** | Milan: dashboard curve review + push the local commit series (LC-LOG through XFER) |
| **6** | **`--full` recut** **9902d918** / **472bc9e4** (post-swap anchor guard on frozen snapshot) |
| **7** | First AAVSO/VarAstro uploads **BO -> FW** (band **CV**) |
| **8** | **EXPORT-PARITY-01** (HIGH): full export vs pipeline photometry divergence |

| Pri | Carry list |
|-----|------------|
| **HIGH** | **EPSF-SHAPE-01** - root narrow ePSF core OPEN (FWHM 2.36 vs 3.30); routed to EPSF-CORE-01 |
| **HIGH** | **EXPORT-PARITY-01** - standing two-path defect (R5 audit); PSF merge path fixed |
| **MED** | **MULTIFILTER-WCS-01** - one field shot through multiple filters in one observation group shares pointing; the solver should SEED, never trust, a sibling set's VERIFIED WCS. Design: within a draft/observation group, when a set's blind solve fails, retry with the WCS of a VERIFIED sibling (closest DATE-OBS) as the initial hypothesis; the catalog-recovery verification gate runs unchanged and unrelaxed; provenance stamps `wcs_source=sibling_seed:<set>`; guard on pointing drift between the sets. Decisive first measurement (520, 2026-08-24, read-only): project g_60_4 VERIFIED WCS onto z_90_4 MASTERSTAR -> catalog_recovery_gate **2.7%** (n_tight=3 / n_det=113 / n_cat=169); in-memory bulk-shift not confirmed. i_70_4 (closest DATE-OBS) -> **0%**. z p99-median contrast **9.06 ADU** vs g **40.0 ADU**. Blind-solve reject stands (z is physically shallow); do not lower `masterstar_catalog_recovery_min`. **SEL-GHOST-01 S6 (2026-08-25):** g_60_4 re-solves when Equipment+Telescope DB scale (0.566 "/px at bin4) is not overwritten by the FITS/config/UI 15.511 "/px Zeiss-wide default; first attempt with the wrong triangle-filter scale failed (`n_cat_tri=113`, cap 33 detections). **`1ec035f` keeps DB plate scale over FITS/UI** (governing-value defect closed; do not open DAO-SCALE-AUTH-01). z_90_4 still needs sibling-seed. Wiring of sibling-seed is still the general design for other drafts. Hand-CSV re-lock with full stamps at the next natural STAGE-01 iteration (not now). |
| **FUTURE** | **EPSF-CORE-01** - literature-parameter ePSF rebuild (multi-frame samples, osamp vs FWHM, smoothing) |
| **FUTURE** | **DB-RETIRE-01**, **MS-POOL-POLICY-01** |
| **MED** | **WIDE-ERR/CORR-ERR-01** |
| **MED** | **PRECAL-INPUT-CONTRACT-01** (informational stamp, not a blocking gate; REG-520-01 menu d), **DAO-TOL-FLOOR-01** (pass2/seed floor = f(rms, FWHM); cite M2 curve) |
| **CLOSED** | **COMP-RMS-DEF-01-B** / **ZONE-SAT-01** (C3 2026-08-25; k=5; LOO mag; peak test never skipped) |
| **OPEN** | **INPUT-PATH-ARCH-01** (discussion pending - Milan 2026-08-25: non-cal route stays; externally-calibrated vs raw-without-masters split still to be discussed) |
| **MED** | **FRAME-QC-PARITY phase 2** (Layer A log honesty + QC provenance stamp) |
| **FUTURE** | ePSF spatial / OSC validation (Part B/D full scope deferred) |
| **LOW** | **SIGMA-BKG-VAR-01**, gaussian_fwhm_px_override provenance, exposure ramp |

---

## NEXT SESSION - entry point (2026-08-24 EPSF-NEWTON-518-01 STOP)

Local tip **`2926a95`** (census committed; AC-02 wiring still dirty; not
pushed). Production 516 ePSF SHA `172f95403beae36d...` unchanged. Newton
518 ePSF STOP: gated pool 26 < 30. EPSF-ZP-OK-01-WIRE stays parked.
SHAPE-01 root remains OPEN, routed to **EPSF-CORE-01**.

| Step | Action |
|------|--------|
| **1** | Milan: Newton 518 ePSF - hold for a night with gated pool >=30, or GO a wide-only ZP-OK without claiming Newton |
| **2** | Milan: GO or hold interim `psf_fit_ok_for_zp` (wide-rig CENSUS-01 still stands; no Newton P-D) |
| **3** | Milan: authorize commit/push of AC-02 wiring + remaining docs |
| **4** | Milan: dashboard curve review + push the local commit series (LC-LOG through census) |
| **5** | **`--full` recut** **9902d918** / **472bc9e4** (post-swap anchor guard on frozen snapshot) |
| **6** | First AAVSO/VarAstro uploads **BO -> FW** (band **CV**) |
| **7** | **EXPORT-PARITY-01** (HIGH): full export vs pipeline photometry divergence |

| Pri | Carry list |
|-----|------------|
| **HIGH** | **EPSF-SHAPE-01** - root narrow ePSF core OPEN (FWHM 2.36 vs 3.30); routed to EPSF-CORE-01 |
| **HIGH** | **EXPORT-PARITY-01** - standing two-path defect (R5 audit); PSF merge path fixed |
| **FUTURE** | **EPSF-CORE-01** - literature-parameter ePSF rebuild (multi-frame samples, osamp vs FWHM, smoothing) |
| **FUTURE** | **DB-RETIRE-01**, **MS-POOL-POLICY-01** |
| **MED** | **PRECAL-INPUT-CONTRACT-01**, **COMP-RMS-DEF-01**, **WIDE-ERR/CORR-ERR-01** |
| **MED** | **FRAME-QC-PARITY phase 2** (Layer A log honesty + QC provenance stamp) |
| **FUTURE** | ePSF spatial / OSC validation (Part B/D full scope deferred) |
| **LOW** | **SIGMA-BKG-VAR-01**, gaussian_fwhm_px_override provenance, exposure ramp |

---

## NEXT SESSION - entry point (2026-08-22 EPSF-VALID-02 S6 close) [superseded]


**EPSF-VALID-02 CLOSED** (2026-08-22): gated 67-star production ePSF on draft 516; 517 first
gated build; edge-star guard in production path; S5b aligned certificates; STOP-B signed.

| Step | Action |
|------|--------|
| **1** | Milan: dashboard curve review + push today's commit series |
| **2** | **`--full` recut** **9902d918** / **472bc9e4** (post-swap anchor guard on frozen snapshot) |
| **3** | First AAVSO/VarAstro uploads **BO -> FW** (band **CV**) |
| **4** | **EXPORT-PARITY-01** (HIGH): full export vs pipeline photometry divergence |

| Pri | Carry list |
|-----|------------|
| **HIGH** | **EXPORT-PARITY-01** - standing two-path defect (R5 audit); PSF merge path fixed |
| **FUTURE** | **DB-RETIRE-01**, **MS-POOL-POLICY-01** |
| **MED** | **PRECAL-INPUT-CONTRACT-01**, **COMP-RMS-DEF-01**, **WIDE-ERR/CORR-ERR-01** |
| **MED** | **FRAME-QC-PARITY phase 2** (Layer A log honesty + QC provenance stamp) |
| **FUTURE** | ePSF spatial / OSC validation (Part B/D full scope deferred) |
| **LOW** | **SIGMA-BKG-VAR-01**, gaussian_fwhm_px_override provenance, exposure ramp |

---

## NEXT SESSION - entry point (2026-08-22 EPSF-VALID-02) [superseded]

## NEXT SESSION - entry point (2026-08-21 SESSION-CLOSE) [superseded]

## NEXT SESSION - entry point (2026-08-21 DOCS-SYNC-517 + FRAME-QC-PARITY-01) [superseded]

## NEXT SESSION - entry point (2026-08-20 ERA-03 CLOSE)

Local tip: **`8dea595`** (ERA-03 close pushed). Canonical product SHA
**9902d918** n=121 / extended **472bc9e4** n=179 on draft 516. P1 golden
**6af4539c** n=115. `--fast` and `--full` OVERALL PASS on
`draft_000516_snapshot_era03_20260820`.

| Pri | Next item |
|-----|-----------|
| **HIGH** | **MS-POOL-POLICY-01** -- new-field comp pool refill policy (48 pinned targets with color re-validation drops; deferred at ERA-03 close). **Superseded by rescope above (2026-08-21).** |
| **MED** | **ePSF-VALID** -- next-session candidate: internal ePSF-vs-aperture identity + independent reference (see V1-VALIDATION-PROTOCOL). |
| **MED** | **COMP-RMS-DEF-01** -- unify CSV `comp_rms` vs LOO per-star noise. |
| **MED** | **RUN-WORKER-01** -- subprocess worker so UI reruns cannot kill photometry. |

**CLOSED this arc (DAO-GAIA ERA-03):** pinned ensembles (48 targets) +
check-star pinning (KNAME continuity); INV-PIN-01..04; core scope 97->121
(+14 additive LCs/comp_quality; -2 honest zone_noise); anchor freeze
`draft_000516_snapshot_era03_20260820`; panel GREEN; BO/FW MAD 7.151/8.201 mmag;
SUBMIT-01 PASS on BO export; GH color question merged into **D10-1** and **resolved**
with CV affirmed (2026-08-21; see DECISIONS D10-1-CLOSE).

---

## NEXT SESSION - entry point (2026-08-16 SESSION-CLOSE)

Local tip at close science stamp: **`4fe84b4`** (IMPL-05 C). origin/main after
SESSION-CLOSE stamp: **`1b8fcab`**. Acceptance so far is a **10-target**
Phase 2A subset on draft 514, not the full 97-target field.

| Pri | Next item |
|-----|-----------|
| **HIGH** | (1) **COMP-RMS-DEF-01** -- unify CSV `comp_rms` vs LOO per-star noise (BO pred 14.9 vs measured 8.6 mmag); same class as U-SCATTER-DEF. Untouched this arc. |
| **MED** | **BIN-8-9-REGRESSION-01** OPEN: full-field LOO 11.9885 mmag n=15, byte-identical to D515-ACCEPT-01 gate. SAT-LIMIT/PFS/re-rank did not change proc-CSV LOO. |
| **MED** | **RUN-WORKER-01** (open decision) -- subprocess/job worker so UI reruns cannot kill photometry; B+C mitigations already in RUN-HARDEN-01. |
| **MED** | **WIDE-ERR-CROSSRIG** -- per-rig physical-model / calib when Newton/Boyden drafts exist. |
| **HIGH** | (3) **NET-TEST-01** **DONE** (2026-08-17): `--fast` icon list vendored from local streamlit 1.55.0 (`dev/tests/data/streamlit_material_icon_names.txt`); no network in that test. `check_pytest` prints FAILED/ERROR node ids. |
| **LOW** | **CORR-ERR-01** research note -- covariance-aware err budget (out of v1.0). |

**CLOSED this arc:** PFS-SEMANTICS-01 (rescue on skip_reason; one peak-test
authority). SAT-RERANK-01 DONE on SHA 36a53b0 (48 LCs; pointer
`CURSOR_RESULT_SAT_RERANK_01B.md`). SAT-LIMIT-01 CLOSED. EXPORT-HDR-01
closed on the same product. NET-TEST-01 DONE (vendored icon list).
WIDE-ERR + SEM (WIDE-ERR-04). EXPORT-READY
interim MAGERR item **OBSOLETE**. U-09 **CLOSED** on the home wide rig
(verdict a, draft 515; `CURSOR_RESULT_U09_GAIN_AUTH.md`). Other rigs
still need a per-rig DATE-OBS check. GAIN-AUTH-VERIFY-01 **CLOSED** by
GAIN-PT-RADIUS-01 (pin r=4.0; product SHA **de6f7c8**). SUBMIT-01 ready
(BO CVn 2026-04-23; Milan manual). 8f107cf quarantined (VL-PFS-8F107CF).
XVAL-AIJ-02 **DONE** (production 4c + two frame states; 4.9 mmag RMS).

Carry (unchanged): INV-PIXELS-01 awaiting Milan;
D1b; D2; WIDE-ERR-CROSSRIG; D1-2 OPEN; C-EXPORT-GAP; W6-PROP; D11-1;
U-SKY-FALLBACK-01; LOCATION_OLD; zone rename; DET-vs-MEAS-01; A-1-OVERRIDE;
blended-target merging (deferred past v1.0); drafts 512/513/510 INGESTED repair.

---

## DONE - CYTHON-RELEASE (closed-source bundle; preview 2026-07-23)

**Status:** **DONE (2026-07-23).** RELEASE-1: 85-module MODULE_LIST compile (84 @ `b4c372a`
+ `vyvar_runtime` @ `3369832`), compiled==interpreted anchor identity. RELEASE-2: bundles,
embedded runtime, data-dir separation (B2 `--full` PASS). `VYVAR-release` populated;
`preview-20260723` pre-release (both platforms). Results:
`dev/results/CURSOR_RESULT_release_close.md`.

**v1.0.0 semantic versioning:** open -- no `v1.0.0` until Milan + Claude declare it.

| Phase | Pri | Deliverable |
|-------|-----|-------------|
| **Spike** | HIGH | **DONE (2026-07-21).** |
| **RELEASE-1** | HIGH | **DONE (2026-07-23).** |
| **RELEASE-2** | HIGH | **DONE (2026-07-23).** Win + Linux preview bundles; smoke PASS. |
| **Public upload** | HIGH | **DONE (2026-07-23).** `preview-20260723` on `VYVAR-release`. |

Frozen deps for release track: numpy 2.4.4, astropy 8.0.1, photutils 3.0.0.
**DEPS note (2026-07-24):** photutils 3.0 runtime deprecation warnings observed
(`npixels` -> `n_pixels`, `semimajor_sigma` removed in 4.0) -- next DEPS cycle item.

---

## OPEN - Post-preview validation and docs (2026-07-23)

| ID | Pri | Status / deliverable |
|----|-----|----------------------|
| **CONFIG-MATERIALIZE-CHECK** | -- | **DONE** (2026-07-24, BUNDLE-BOOTSTRAP-WIRING): `bootstrap_release_data_dir()` is the single entrypoint for skeleton + canonical `config.json` materialization; wired from bundled app startup (`ensure_release_data_dir` in `app.py`) and `--selftest` (truthful per-item report on the resolved `VYVAR_DATA_DIR`). |
| **INSTALL-GAIA-DEC-CUTOUT** | MED | Docs: Gaia builder supports `--dec-min` / `--dec-max`; INSTALL should lead with a declination cutout decision (observer southernmost usable DEC), size/time table, and note ESA archive login is **optional**. |
| **SYNTH-SKY-GENERATOR** | MED | Validation-protocol tool: Claude WCS-true synthetic field generator (sub-pixel geometry debug pending); then known-truth photometry validation against injected variables. |
| **V1-VALIDATION-PROTOCOL** | HIGH | Enrich validation strategy: detrended_aligned validation packs (small frame-level pack across many fields; large LC pack for 1-2 fields via split release assets; one raw-FITS E2E mini field). ePSF validation: internal ePSF-vs-aperture identity; DAOPHOT/allstar independent PSF reference; independent plain-photutils EPSFBuilder script. Comparison: Munipack/SExtractor/IRAF headless in Claude sandbox; AIJ hybrid (Milan GUI, Claude evaluates exports). Scripted PDF QA (overflow/clipping/encoding + report-vs-CSV consistency). Fallback: Claude-authored harness on Milan full data. Sandbox capable: headless installs, P1-golden E2E, bundle verification. NOT capable: visual UI clicking, ESA archive downloads. |
| **M71 E2E acceptance** | HIGH | OSC arc code-complete; M71 validation dataset (eq id=5). |
| **Milan field Linux testing** | HIGH | First real-sky E2E astrometry DONE (FI Boo 147 frames); photometry after VSX fix; findings #11-#13 in repo for next bundle. |
| **CITATIONS.bib ASCII disposition** | LOW | Open encoding/normalization decision for bundled citations file. |
| **KNOWN_REMOVED_KEYS builder-prose sweep** | LOW | Docs-guard sweep for removed config keys still mentioned in builder prose. |
| **preprocess-QC summary durability** | MED | Persist QC summary into `pipeline_meta.json` (survive reruns). |
| **PHASE0-BORDER-MARGIN-GEOMETRY** | MED | Derive Phase 0 `out_of_frame` border margin from aperture + sky annulus outer radius (FWHM-driven), not a fixed 50 px constant. Draft 451 `out_of_frame=78` is consistent with a 50 px strip (~12% frame area) -- count is not anomalous; only the constant is arbitrary. **No implementation in POST-451 closeout.** |
| **D1** | MED | **DONE (2026-08-05).** Inactive unit-normalisation plumbing: `unit_resolver.py`, 10 arcsec/FWHM companion fields (`None` defaults), registry + call-site wiring. Result: `dev/results/CURSOR_RESULT_d1_unit_normalisation.md`. |
| **D1b** | MED | **OPEN.** Conversion table and proposed normalised defaults in D1 result; companions remain `None` -- behaviour unchanged. Awaits Milan review before activating defaults. |
| **D2** | HIGH | **OPEN (design done 2026-08-05).** Per-rig storage for group (a) parameters. Implementation blocked on storage choice: nested dict in `config.json` (`sigma_sys_mag` pattern) vs DB table keyed on `(ID_EQUIPMENTS`, `ID_TELESCOPE)`. Cursor recommended DB table. Result: `dev/results/CURSOR_RESULT_d2_per_rig_storage.md`. |
| **DRAFT451-CAL-FRAME001** | MED | Draft 451 frame-001 calibrated product differs from draft 452 by **659.6 ADU** max while siblings differ ~121 ADU; raw 451 vs 452 inputs are byte-identical (0.0 ADU). Old-vs-new preprocess on the same night: **149/150** calibrated frames match exactly (452 vs 454); only frame 001 differs (**533.5 ADU**). **Evidence preserved (2026-07-29):** `dev/results/context/frame001_evidence/draft451_BO_CVn_Light_001.fits` SHA256 `15DED344DBC1CA4504FD419E38CC5A19DA16A6BBF001E5529B0005A0C2CFB041`; `draft452_BO_CVn_Light_001.fits` SHA256 `E5B6B3E261842E833CE341ADAD669E5A686136734475D674AFA12387C5D3D774`. Root cause still needs draft 451 calibration run logs. |
| **CATALOG-PROVENANCE** | HIGH | **DONE (2026-07-29).** `catalog_provenance.py` stamps Gaia + VSX identity into run provenance; anchor `--full` gate `full-catalog-provenance` compares to snapshot (or live DB when snapshot lacks block) and names *input catalogue changed* before anonymous SHA fail. Fingerprint: `sha256(size + first_1MiB + last_1MiB)` -- full SHA over ~53 GB Gaia impractical. |
| **SKY-SURFACE-BLAST-RADIUS** | MED | Drafts 438, 439, 444, 448, 449, 450, 451 carry inflated catalogues (40-75% DAO_ONLY). Draft 451 analysis: shared-target photometry unaffected (Group A RMS ratio 1.007); damage expected confined to spurious faint targets. **Open:** confirm whether any export from those drafts reached AAVSO or VarAstro. |
| **GAIA-PM-COLUMNS** | MED | Gaia DB lacks `pmra`/`pmdec`; matcher runs `broadened` mode. **Deliberate deferral** to DR4 build (~Dec 2026), not an oversight -- five-month horizon. See existing GAIA-1/2 rows below. |
| **R-CVN-EMPTY-COMP** | LOW | Empty-comp drop now reports `no_comps`; confirm nothing further needed (POST-453 ledger). |
| **mono gaia_johnson adoption** | MED | Gated mono path adoption of `gaia_johnson` catalog helper. |
| **v1.0.0 declaration** | HIGH | Joint Milan + Claude decision; open. |

---

## OPEN - Post-DAO session carry-forward (2026-08-07)

| ID | Pri | Status / deliverable |
|----|-----|----------------------|
| **FULL-ANCHOR-RECUT** | -- | **CLOSED 2026-08-27 (ERA-04).** Frozen `draft_000516_snapshot_era04_20260826`; core `9367f998` n=160; ext `d3cefff3` n=210. APERTURE-01d f=1.35 annulus 2.7/5.2. Supersedes ERA-03 / `9902d918`. era03 freeze kept on disk. `--fast` + `--full` twice OVERALL PASS. |
| **P1-RECUT** | -- | **CLOSED 2026-08-20 (ERA-03).** Golden `draft_000516_p1mini` from era03 snapshot; core `6af4539c` n=115. |
| **A-1-435-RECUT** | -- | **CLOSED 2026-08-18.** 435 golden retired by design (ROT out-of-scope is canonical policy; 435 predates it). Not repaired. Recut onto 516. |
| **FULL-ANCHOR-RECUT** | -- | **CLOSED 2026-08-27 (ERA-04).** Frozen `draft_000516_snapshot_era04_20260826`; core `9367f998` n=160. era03 freeze `9902d918` n=121 kept on disk. `--full` OVERALL PASS twice. |
| **TASK-A-REGRESSION** | MED | `test_masterstars_csv_write_survives_bp_rp_failure` (`dev/tests/test_invariants_p2.py`) reimplements fixed A2 control flow in the test body; never calls `generate_masterstar_and_catalog`. A2 CSV-write mechanism untested against pre-fix pipeline. |
| **F-B01-F-B02** | HIGH | Calibration-path audit: PASSTHROUGH runs record `CALIBRATION_MODE=vyvar_calibrated` and PDF may claim VYVAR calibration when none applied. Fix order: `dev/results/CURSOR_RESULT_calpath_audit.md` section 14; decisions section 13. |
| **QHY294MM-RN-DOUBLE** | LOW/MED | DB read-noise 7.6 e- appears to be bin2 measurement (`draft_303`) that `param_resolver` scales by binning again to 15.2 e-. Minor at current sky levels; material for short/dark exposures. |
| **BPM-SIDECAR-PATH** | MED | No `*_dark_bpm.json` found for any draft including VYVAR-calibrated runs. Whether path is dead, disabled, or writes outside draft tree is unresolved. |

---

## OPEN - 2026 audit closure carry-forward (2026-08-14)

| ID | Pri | Status / deliverable |
|----|-----|----------------------|
| **A-1-435-RECUT** | -- | **CLOSED 2026-08-18 (ANCHOR-516-04).** 435 retired by design (ROT out-of-scope policy); recut onto frozen 516. |
| **A-1-OVERRIDE** | HIGH | Remove `VY_FWHM_GAUSS` as `gaussian_fwhm_px_override`. Authorized in principle; own measured delta. |
| **A-1-DECISION-4** | MED | Advanced: r90 5.0-5.8 px, target 5.31 px, current EE ~84.6%. Not implemented. |
| **FULL-ANCHOR-RECUT** | -- | **CLOSED 2026-08-18 (ANCHOR-516-04).** Frozen `draft_000516_snapshot_cleanrebuild_20260818`; core `477dc8cf`. `--full` OVERALL PASS (pipeline 6135 s). 0.85-gate hole closed (MASTERSTAR writer + proc columns at 52428). |
| **U-XVAL-COMP-RMS** | MED | RETRACTED (see register). |
| **D1-2-LINEARITY-RAMP** | HIGH | Exposure ramp at telescope; nothing else substitutes. |
| **WIDE-ERR** | -- | **CLOSED** (WIDE-ERR-04): physical model g_pt + weighted SEM; identity s=1,sr=0. |
| **CORR-ERR-01** | LOW | Research note: ZP common-mode vs diagonal budget; out of v1.0. |
| **WIDE-ERR-CROSSRIG** | MED | Per-rig when Newton/Boyden drafts exist. |
| **W6-PROP-*** | MED | Authorized 2026-08-13, not implemented: `VY_QCBG_PRE`, detect_outliers Option A, library delete guards, preprocess rename. |
| **C-EXPORT-GAP** | LOW | Headless `night_run` omits AAVSO/VarAstro export. |

Full audit closure: `docs/VYVAR_AUDIT_2026_CLOSURE.md`. Handoff: `docs/VYVAR_HANDOFF_2026-08-14.md`.

---

## QUEUED - Audit closure Steps 1-10 (2026-07-31)

**Status:** Science audit **closed** (2026-08-04). Next work: **WIDE-ERR** and **MASTERSTAR
stacking** (independent future threads, not audit-open).
Wide-rig H1 diagnostic complete (no floor). D5-2 **CONFIRMED**. MASTERSTAR architecture = separate enhancement thread.
See `docs/VYVAR_AUDIT_CLOSURE.md`.
Source: `docs/VYVAR_AUDIT_CLOSURE_REGISTER.md` (30 items; Steps 1-10 active).
Audit synthesis: `docs/VYVAR_AUDIT_FINAL.md`. Operational TODO index:
`docs/VYVAR_TODO_MASTERSTAR_REFERENCE.md`.

| Step | ID | Pri | Status / deliverable |
|------|-----|-----|----------------------|
| 1 | **A-1** | HIGH | Frame selection metric `I_j = F_j^2/(sigma_j^2.FWHM_j^2)` for MASTERSTAR stack ranking |
| 2 | **A-2** | HIGH | Selection rule N_min=10, N_max=20, quality gate 0.5xmax(I_j) |
| 3 | **A-3** | HIGH | Median/sigma-clip stack (replace single-frame copy) |
| 4 | **A-4** | HIGH | Stack provenance in header + `pipeline_meta.json` |
| 5 | **A-5** | HIGH | Recalibrate DAO threshold against stack noise/PSF |
| **DAO-THRESHOLD-PARAMS** | -- | **CLOSED (2026-08-07).** `masterstar_dao_threshold_sigma`, `sips_dao_threshold_sigma`, `qc_dao_detection_sigma`: no detection-stage change indicated; draft_501 inflation driven by pre-calibrated additive pedestal, not rig geometry alone. Reopen: two-rig empirical sweep at matched calibration state. See `docs/VYVAR_DAO_DETECTION.md` section 5.1. |
| 6 | **A-6** | DONE | Split `DAO_ONLY` health metric by magnitude vs Gaia cap (per-class census + CSV columns; no runtime gate). **DAO detection workstream closed 2026-08-07** (`docs/VYVAR_DAO_DETECTION.md`). |
| 7 | **C-1** | HIGH | Admission gate: predicted per-epoch SNR (`g_lim_*` + Labbe sigma_bkg_ap) |
| 8 | **C-2** | HIGH | CONTEXT-ONLY vs PHOTOMETRY-CANDIDATE catalogue flags |
| 9 | **CR-1** | MED | Cosmic-ray rejection (L.A.Cosmic or equivalent) |
| 10 | **T4-1** | HIGH | **DECISION:** detection noise on resampled frames (options A/B/C/D) |

Items 11-30 in closure register (P-10 FIXED, anchor re-cut BLOCKED, etc.) tracked but not in
this execution wave.

---

## OPEN - MASTERSTAR reference architecture (2026-07-30)

**Status:** NOT STARTED. Does **not** block closure Step 1 start. Full spec:
`docs/VYVAR_MASTERSTAR_REFERENCE_ARCHITECTURE.md`; checklist:
`docs/VYVAR_TODO_MASTERSTAR_REFERENCE.md`.

| ID | Pri | Status / deliverable |
|----|-----|----------------------|
| **TODO-C** | HIGH | Separate **admission gate** (predicted per-epoch SNR via `g_lim_*` + Labbe `sigma_bkg_ap`) from **detection threshold**; flag catalogue rows CONTEXT-ONLY vs PHOTOMETRY-CANDIDATE. Independent of stack reference. **Closure Steps 7-8.** |
| **CR-REJECTION** | MED | Cosmic-ray rejection (L.A.Cosmic or equivalent). Standalone hygiene gap; prerequisite for TODO-B; improves TODO-A median stacks. Verified: no CR step in `src_py` today (`cosmic` = gain/RN only). **Closure Step 9.** |
| **TODO-A** | HIGH | Replace single-frame MASTERSTAR (`build_masterstar_from_detrended` lowest-`VY_FWHM` copy) with **median/sigma-clip stack** of best N frames ranked by `I_j = F_j^2/(sigma_j^2.FWHM_j^2)`; mandatory provenance; recalibrate DAO threshold; split `DAO_ONLY` health metric by magnitude vs Gaia cap. **Closure Steps 1-6.** |
| **TODO-B** | MED | Zackay & Ofek (2017) **proper coaddition** - optimal version of A; blocked on CR rejection, uncorrelated inputs, per-frame PSF, PSF-based `F_j`. Register item 30. |

**Suggested order:** TODO-C -> CR-REJECTION -> TODO-A -> TODO-B (see closure register Steps 1-10).

---

## QUEUED - VYVAR-INVARIANTS (starts after Anchor #3)

**Status:** **UNBLOCKED** (2026-07-16). Anchor #3 ACTIVE (`VL-ANCHOR-WCSINV` /
`draft_000435_snapshot_skysurface_20260716`). Begin **P1** now.

| Phase | Pri | Deliverable |
|-------|-----|-------------|
| **P1** | HIGH | **DONE (2026-07-19).** Golden mini `draft_000435_p1mini` (16-frame stride) + `test_invariants_p1_golden.py` (headless SHA, UI-order identity, census bands, physics). Ledger `VL-P1-GOLD`. Result: `dev/results/CURSOR_RESULT_invariants_p1.md`. |
| **P2** | HIGH | **DONE (2026-07-19).** `docs/VYVAR_INVARIANTS.md` registry + `invariants_runtime.py` wired gates (flux, flatness, WCS WARN, stage DAG, RNG AST, provenance schema, config<->behavior). Result: `dev/results/CURSOR_RESULT_invariants_P2.md`. |
| **P3** | MED | **DONE (2026-07-20).** PROCESS recurrence + forensic promotion rubric + weekly `invariants_report.py`; pilot PROMOTE+ARCHIVE of `forensic_disc_ui_match2.py`. Result: `dev/results/CURSOR_RESULT_invariants_P3P4.md`. |
| **P4** | MED | **DONE (2026-07-20).** STATE honest-scope statement (GUARANTEED / NOT guaranteed); GAPS D2 void addendum; P1-P4 program closed. Result: `dev/results/CURSOR_RESULT_invariants_P3P4.md`. |

Separate commits per phase; pytest green each. Result files: `CURSOR_RESULT_invariants_P<n>.md`.

---

## IN-FLIGHT / ACTIVATED - band-aware k'' (second-order extinction) v1

**Status:** **ACTIVATED v1 (2026-07-07).** Spec: `dev/results/specs/VYVAR_K2_DESIGN_SPEC.md`. Code: `k2_extinction.py`,
`k2_mode=literature` default, band_classify CT wiring + CV/CR flip, LC provenance columns.
**NIGHT_FIT v2 fit implemented + synthetically validated (gated OFF)** (`k2_fit_enabled=false`;
`fit_k2_night` / `k2_feasibility_pregate` in `k2_extinction.py`; recovery suite
`dev/tests/test_k2_night_fit_recovery.py`). **Activation blocked ONLY on data night (B2 /
K2-DATA-BLOCKER).**

**Rejected (do not port):** comp-select **grow-redesign** - population validation showed ~**45%**
regressions; sandbox-only; **nothing ported to production**.

**Policy (shipped):** airmass = **CORRECTION not selection**; literature k'' for STANDARD_FILTER
(excluding OSC RGB tokens); CLEAR/CV/CR/L/UNKNOWN -> k2 none; CV/CR CT flip live.

**Shipped additive precursor:** `band_classify.py` + tests (`fe9b375`); now wired to
`resolve_apply_color_term`.

### K2 ledger (2026-07-07)

| ID | Sev | Item |
|----|-----|------|
| **PROV-HEADLESS** | - | **FIXED** (`e7ce7ea`): `pipeline_meta.json` `provenance` block (`git_hash`, `git_dirty`, full `config_snapshot`, `stamped_at_utc`, `entry_point`); last-writer-wins at `run_phase2a` + `generate_masterstar_and_catalog`. |
| **PROC-MAG-NAMING** | - | **FIXED** (`0913665`, documented): proc CSV `mag` = Gaia catalog G (constant per star at match time); science uses `dao_flux`. Docstring + `VYVAR_PIPELINE_CZ.md` schema note; PROCESS dao_flux rule verified. |
| **K2-DATA-BLOCKER** | - | Activation only: calibrated filtered draft, dX >= ~0.3, comp residual floor << 15 mmag. v2 fitter + synthetic recovery DONE (2026-07-20); Boyden blocked on flat/flip systematics. |
| **K2-SLOPE-TRACE** | - | **FIXED** (`8c44b71`): GR slope 0.859->1.054 (Jordi 2010 Table 6 inverse, FGK g-r=0.48); k2_g~-0.0169, k2_r~-0.0042. UG retained 1.091 as documented exception -> **K2-SLOPE-UG** (FUTURE). |
| **K2-SLOPE-UG** | FUTURE | No Jordi u-g row; spec-anchored UG slope 1.091 retained pending citable source. |

---

## IN-FLIGHT / PARKED - band-aware k'' (second-order extinction) [SUPERSEDED - see ACTIVATED v1 above]

**Status:** design parked; first code piece shipped additive (`fe9b375`); production CT rewiring **PENDING**.

**Rejected (do not port):** comp-select **grow-redesign** - population validation showed ~**45%**
regressions; sandbox-only (`sandbox/comp_grow*.py`, `tmp/` metrics); **nothing ported to production**.

**Correct direction (when unblocked):** treat **airmass as CORRECTION, not selection** - derive k''
from **constant comps** (signal-safe); **band-aware** policy via `band_classify.py`:
- **STANDARD_FILTER** (B/V/R/etc.): k'' is reliable; apply second-order extinction correction.
- **CLEAR_UNFILTERED** / unfiltered: tight colour-match is primary; **do not** rely on k''.
- **LUMINANCE (L):** own class; distinct from clear for future policy.
- CV/CR -> CLEAR_UNFILTERED (physically clear-transformed); flip activates **with** CT rewiring, not alone.

**Shipped additive (not wired):** `band_classify.py` + 52 tests (`fe9b375`); consolidates legacy
`_is_nofilter_obs_group` / broadband CT auto. Ledger: `VYVAR_AUDIT_LEDGER.md` BAND-DETECT (`5d6801c`).

### REAL BLOCKERS - resume band-aware k'' (Milan-side data, not code) [SATISFIED 2026-07-07 for v1 literature path]

1. **Filtered draft (V/B/R) for validation.** k'' only does real work on filtered data. Locally
   **NoFilter** draft **424** exists; **filtered drafts now exist on disk (Milan, 2026-07-09) -
   data availability to be re-verified when NIGHT_FIT v2 is scheduled** (do not unblock/schedule here).
2. **Newton/Brno literal FITS FILTER strings.** Needed so those rigs get k'' routing instead of
   fail-safe clear (UNKNOWN -> CLEAR). Capture on dev PC / rigs: `FILTER`, `FILT`, `FILTER1`, `INSTFILT`
   on Chi_and_H **B/V/R/L** frames and Brno **`r_60_4`**; SQL on dev DB:
   `SELECT DISTINCT FILTER, ID_EQUIPMENTS FROM FITS_HEADER_CACHE`.

**Activate together (code, when data unblocked):** wire `band_classify` into production CT gating +
CV/CR->clear behavioral flip + band-aware k'' correction path.

---

## Deferred findings (carry forward - none blocking)

| ID | Sev | Notes |
|----|-----|-------|
| **F-BINGAIN-1** | - | **RESOLVED (2026-07-10).** Empirical empty-aperture `sigma_bkg_ap` + hybrid `howell_scaled` fallback. Regate PASS (decomposition-driven gates G1-G4). Result chain: `CURSOR_RESULT_bingain_fix.md`, `CURSOR_RESULT_bingain_acceptance.md`, `CURSOR_RESULT_bingain_regate.md`. Harness aligned 2026-07-13 (**SIGMA-BUDGET-EMPIRICAL FIXED**). |
| **SIGMA-BUDGET-EMPIRICAL** | - | **FIXED (2026-07-13).** ``chi2_sigma_gate.sigma_arrays_from_lc_and_proc`` uses proc ``sigma_bkg_ap`` / ``err_bkg_source``; ``production_lc_err`` variant added. Tests in ``tests/test_sigma_budget.py``. Result: ``CURSOR_RESULT_sigma_newton.md``. |
| **SIGMA-PROV-FORENSIC** | HIGH | **DONE (2026-07-13).** draft_426 archive LC **stale pre-Fix-A err** (semantic 7.46x i); fresh i_70_4 rerun: V0611 chi2=**2.13**, err=**0.0175 mag**. P2/P3 quantitative predictions FAIL. Part C unit fix landed. Result: ``CURSOR_RESULT_sigma_prov_forensic.md``. |
| **426-REGEN** | - | **DONE (2026-07-13); baseline SUPERSEDED by MASTERSTAR-EPOCH-FIX.** Stale at ``Archive/evidence/draft_000426_stale_20260626``. |
| **MASTERSTAR-EPOCH** | HIGH | **CLOSED (2026-07-13).** ``proc_MASTERSTAR.csv`` excluded from epoch collection; draft_426 regen 25 epochs. Result: ``CURSOR_RESULT_masterstar_epoch.md``. |
| **EXPORT-PARITY-01** | HIGH | **OPEN (2026-08-22).** ``export_per_frame_catalogs`` full recompute diverges from full-pipeline photometry on live draft 516: 99/134 frames row shrink, 51k flux numeric diffs, 41k mag losses (R5 audit). PSF RUN job fixed via INV-PSF-ADDITIVE-01 / F6 merge path; standing two-path defect for full export vs pipeline photometry remains. Evidence: ``CURSOR_RESULT_EPSF_VALID_02_R5F5.md``. |
| **EPSF-SHAPE-01** | HIGH | **OPEN (core), routed to EPSF-CORE-01.** Narrow production ePSF (FWHM 2.36 vs input 3.30 px; bright chi2 ~68). SHAPE-01-M/F: H5 fitter-scale superseded (iter==single with AC off); osamp=4 builder closed; F2b pedestal retracted. Flux-scale **policy** closed by **EPSF-AC-02** (P4). Evidence: `CURSOR_RESULT_EPSF_SHAPE_01_M.md`, `CURSOR_RESULT_EPSF_SHAPE_01_F.md`. |
| **EPSF-AC-01** | HIGH | **CLOSED (measure, 2026-08-24).** chi2<5 AC gate admits 0/30 brightest; uncorrected PSF/DAO mag-sloped (bins 1.27 -> 2.21). Evidence: `CURSOR_RESULT_EPSF_AC_01.md`. |
| **EPSF-AC-02** | HIGH | **CLOSED (wire, 2026-08-24, Milan GO).** Production F6 AC = P4 (`p4_none`); internal PSF LC = P4; INV-PSF-LC-PIN-01 same-membership-or-NaN. BO CVn PSF-vs-aperture RMS 614 -> 39 mmag on 23/134 full-membership epochs. Evidence: `CURSOR_RESULT_EPSF_AC_02_WIRE.md`. |
| **EPSF-PIN-CENSUS-01** | HIGH | **CLOSED (measure, 2026-08-24).** 100% of pin drops are stored chi2>=50; inferred/nonfinite/missing = 0. Admitting chi2>=50 holds PSF-vs-aperture quality (BO 38.8->37.3 mmag on 134/134; FW 0->134 at 48.5 mmag RMS). STOP for Milan on `psf_fit_ok_for_zp`. Evidence: `CURSOR_RESULT_EPSF_PIN_CENSUS_01.md`. |
| **EPSF-ZP-OK-01-WIRE v2** | HIGH | **CLOSED (wire, 2026-08-26, Milan GO wide-only).** `fit_ok_for_zp` on rig `1:1` only. W2 HIT: BO 134/134 demeaned 8.495 mmag; FW 134/134 demeaned 5.218 mmag. T1 no longer rewrites live 516. Evidence: `CURSOR_RESULT_EPSF_ZP_OK_01_WIRE_v2.md`. |
| **EPSF-ZP-OK-XRIG-01** | MED | Extend `fit_ok_for_zp` past wide `1:1`. Two conditions: master dark+flat in CalibrationLibrary for that rig; CENSUS-01 replay of pin-drop vs quality. Newton 518 pool 26 does not qualify. |
| **EPSF-NEWTON-518-01** | HIGH | **CLOSED (STOP N2, 2026-08-24).** Newton draft 518 (bin2 1.30 arcsec/px) Part C gated pool 26 < 30 (science_scope). ePSF not built; P-A..P-E unmeasured. ZP-OK stays parked. Evidence: `CURSOR_RESULT_EPSF_NEWTON_518_01.md`. |
| **EPSF-CORE-01** | FUTURE | Custom / literature-parameter ePSF to move `epsf_fwhm_native_px` toward ~3.3. **Acceptance coverage (PIN-CENSUS-01):** pinned-comp `psf_fit_ok` fraction on BO CVn / FW CVn (current 23/134 and 0/134 full-membership epochs under chi2<50). **Newton 518 (EPSF-NEWTON-518-01):** not yet a cross-rig ePSF baseline - Part C gated pool 26 < 30 (science_scope choke); night is bin2 1.30 arcsec/px, not unbinned 0.65. **Samples:** multi-frame ePSF samples (production 67 stars x 134 lights ~ 9000 samples; star x frame, not unique stars). Godden & Blundell 2025 GJW precedent 99 x 31. **Oversampling:** >=4 gridpoints per FWHM (osamp=2 at FWHM 3.3 px already gives 6.6; raising osamp needs the sample table: ~240 stars-samples per the paper's Table 2 for osamp=4 M=6 at 95 percent). **Smoothing:** minimal-or-no smoothing at low oversampling (5x5 quadratic at osamp=2 spans 2.5 native px = documented over-smoothing/ringing regime, matches F2b). **Gridpoints:** 2D polynomial surface over sigma-clipped median (photutils default underestimates the peak gridpoint). **Upstream:** watch photutils for the Godden & Blundell 2025 fixes (initial-ePSF centre indexing, flux+position degeneracy constraint, normalization). Cross-rig validation on Newton remains desired but is no longer the sole unlock. |
| **COMP-POOL-R** | MED | **RESOLVED (2026-07-14):** r baseline filled; sparse trust validated. SS Cam
   **RESOLVED: YELLOW** (Milan confirmed 2026-07-14, evidence-based): R=2.008 [1.224, 3.886],
   p_stab=0.0, x2_pair=2.96e-4 mag^2 (=17.2 mmag pair excess, 26% below X2_RED cap of 20 mmag),
   production_lc_err chi2=21.38, n=2, N=25, external K ``1112110935816253440``. Near-boundary note:
   regular spec outcome, not an artifact; X2_RED was NOT adjusted post-hoc to flip the verdict
   (recorded to preempt future re-litigation). Practical effect: AAVSO submissions from this target
   carry the caution flag. **r_60_4 field:** all 6 sparse targets YELLOW with R~2.0-2.9; two
   comp-pair groups (one clean: p~1, x2=0; one carrying the 17 mmag excess); coherent with Newton
   overdispersion that the white floor covers at per-point level only. Result:
   ``CURSOR_RESULT_sparse_check_pool.md``, ``CURSOR_RESULT_arc_close.md``. |
| **SPARSE-TRUST** | - | **CLOSED (2026-07-14).** Spec + Amendment 1 implemented; S1-S4 validated; external K
   sourcing live; r baseline row filled; SS Cam resolved YELLOW. Spec: ``docs/VYVAR_SPARSE_TRUST_SPEC.md``.
   Result: ``CURSOR_RESULT_arc_close.md``. |
| **ANCHOR-CHAIN-ACCEPT** | HIGH | **DONE (2026-07-13):** Intermediate baseline ``b5364e6`` cut; exact c4 validation 23542/23542 PASS; anchor **ACCEPTED** (core ``bf3743a1``). Pushed `7ed7459`. Result: ``CURSOR_RESULT_anchor_chain.md``. |
| **ANCHOR-ERR-VERIFY** | HIGH | **DONE (2026-07-13):** Floor leak **not confirmed**; err rise = bingain + unit fix vs stale anchor. Result: ``CURSOR_RESULT_anchor_err_verify.md``. |
| **PROD-SIGMA-FLOOR** | HIGH | **DONE (2026-07-13):** c4 SEM + Newton floor 18 mmag; wide un-floored. Anchor accepted via ANCHOR-CHAIN-ACCEPT. Spec: ``docs/VYVAR_SIGMA_FLOOR_SPEC.md``. |
| **PROVENANCE-GUARD** | - | **DONE (2026-07-13).** ``provenance_guard.py``; harness refuse + VL-PROVENANCE ledger. |
| **STALE-LC-SWEEP** | - | **CLOSED via regen (2026-07-13).** Evidence preserved; draft_426 on HEAD. |
| **SIGMA-SEM-CAUSE** | - | **DONE (2026-07-13); SUPERSEDED.** Trends confirmed; dominant-cause attribution retracted -- see SIGMA-PROV-FORENSIC. Result: ``CURSOR_RESULT_sigma_sem_cause.md``. |
| **F-EXCEPT-TIER1** | - | **CLOSED (2026-07-08)** - 40 fixes + EXCEPT-BULK; census `docs/VYVAR_EXCEPT_CENSUS.md` all **625 EVIDENCE**. |
| **GAIA-ID-FLOAT-GUARD** | MED | **CLOSED** (verified 2x clone + live tree, 2026-07-07). |
| **F-HOWELL-3** | MED/HIGH | **FIXED (Stage C)** | `sky_adu_per_px_annulus`; draft_424 science byte-identical |
| **F-BJD-1** | LOW | **FIXED (Stage D)** | `time_base` LC column; numeric times unchanged |
| **F-AIRMASS-CITE** | LOW | **FIXED** (2026-07-07) | Kasten & Young (1989) attribution |
| **G7-F003c** | - | **FIXED** (`80aab21`): PDF reads `provenance.config_snapshot` from `pipeline_meta.json`; live `AppConfig` fallback footer-annotated. |
| **EQUIP-BINNING-ASYM** | LOW | Asymmetric binning (`XBINNING != YBINNING`) warns but does not scale gain/RN; all current rigs symmetric. |
| **TIER1-OBSLOC-ZERO** | - | **FIXED** (`166cbf4`): `resolve_site` null-island guard (|lat|,|lon| < 0.01 deg -> UNRESOLVED); airmass refuses, BJD `JD_FALLBACK`. |
| **TIER1-UI-DEBT** | LOW | 38 SAFE UI/plotly broad-except `pass` sites - cosmetic; subset of T3-UI in `docs/VYVAR_EXCEPT_CENSUS.md`. |
| **HRD-PLOT-TUPLE** | - | **FIXED** (EXCEPT-BATCH-S0): `sqlite3.Row` iterates values not keys (`hrd_analysis.py:113`); PDF now emits HRD page or explicit unavailable placeholder (`photometry_report.py`). |
| **299-defensive cluster** | - | **CLOSED (2026-07-08)** - EXCEPT batch complete; ledger `docs/VYVAR_EXCEPT_CENSUS.md`. |
| **PROV-HEADLESS** | - | **FIXED** (`e7ce7ea`) - see K2 ledger row. |
| **PROC-MAG-NAMING** | - | **FIXED** (`0913665`) - documented, not renamed. |
| **K2-DATA-BLOCKER** | - | NIGHT_FIT needs dX >= ~0.3, comp floor << 15 mmag; Boyden blocked on flats/flip. |

---

## PUBLICATION - VYVAR methods paper + software DOI (workstream, opened 2026-07-08)

**Scheduling: LAST** - after functional work completes (Milan, 2026-07-09). Do not delete PUB items.

**Status:** venue research **DONE** (2026-07-08, Claude web research); venue decision **PENDING**
(Milan). Two-track strategy recommended: **JAAVSO** methods paper + **JOSS** software DOI;
**OEJV** (Masaryk Univ. Brno - SQUADRA home venue) for subsequent science papers citing the
pipeline paper; **Astronomy & Computing** as a possible expanded v2 paper later.

**Venue matrix (researched 2026-07-08):**

| Venue | Fit | Cost | Notes |
|-------|-----|------|-------|
| JAAVSO | methods paper, observer audience, ADS-indexed | free for AAVSO members ($100/page non-members) | 2 issues/yr, window closes 6 wks before Jun 15 / Dec 15; Word or LaTeX; journal@aavso.org |
| JOSS | software itself gets citable DOI; review = code+docs+tests via public GitHub issues | free | REQUIRES: OSI license, public repo, English README/docs; paper ~1000 words, must NOT contain science results; 2026 criteria value sustained commit history (VYVAR: months of history, 590+ tests) |
| OEJV | science-results papers using VYVAR; SQUADRA/MUNI home venue | free | PDF to oejv@physics.muni.cz; referee report <=1 month; arXiv-mirrored, Crossref-indexed |
| Astronomy & Computing | professional pipeline-paper league (PHOTOMETRYPIPELINE, Photometry+) | OA $2220 or subscription route free | candidate for expanded v2 paper (post NIGHT-FIT v2) |

**Sub-items:**

- [ ] **PUB-VENUE** (Milan decision): confirm JAAVSO+JOSS two-track; optionally ask OEJV
      editors informally whether a methods paper is in scope (home-field option).
- [ ] **PUB-OUTLINE**: Claude drafts paper outline (Intro / Architecture / Photometric methods /
      QA & trust / Validation / Comparison with existing tools / Summary) - first writing task.
- [ ] **PUB-FIGS**: figure set from real drafts (RMS-vs-mag, example LC (BO CVn), comp-QA panel,
      architecture diagram, CAL-DIAG/k2 provenance example) - Cursor generates, Milan approves.
- [ ] **PUB-JOSS-PREREQS** (only if JOSS confirmed): OSI license file in repo, repo public,
      English README + install/user docs, tagged release + Zenodo DOI at acceptance.
- [ ] **PUB-POLICY**: check chosen venue's AI-assisted-writing policy before submission;
      acknowledge per policy (Milan authorial decision).
- [ ] **PUB-VALIDATION-SECTION**: existing threads feed here - **TODO-SEP-XVAL** longitudinal
      record ("mission + paper Validation section") and the single-night canonical unit
      (**TODO-GS8** / DECISIONS product-scope boundary) become the Validation section's backbone.
      **JAAVSO methods (XVAL-AIJ-01 + XVAL-AIJ-02):** VYVAR and AstroImageJ
      agree to 3.3 mmag RMS (matched 5-star ensembles) and 4.9 mmag RMS on
      the production clean 4-comp ensemble over 134 epochs of a 0.47-mag
      eclipser; processing-chain (alignment+detrending) effect bounded at
      3.8 mmag RMS with no airmass/time correlation.

**Depends on / feeds:** nothing blocks writing the outline now; PUB-FIGS wants a
representative filtered draft (425 BVR qualifies today); JOSS track blocked only on repo
licensing decision.

---

## DONE - XVAL-AIJ-02 (production 4-comp + two frame states)

**Status:** **DONE** 2026-08-17 (docs+evidence; no science code). AIJ 6.0.10
run2/run3 vs VYVAR de6f7c8 on BO CVn. Tools RMS 4.86 mmag on the production
4c ensemble (134 epochs); 3.27 mmag 5c row unchanged. Frame-set 3.80 mmag
RMS / 131 epochs. QC: 13 rejected calibrated epochs ~2.4x worse in AIJ.
DECISIONS XVAL-AIJ-02. Report: `dev/results/CURSOR_RESULT_XVAL_AIJ_02.md`.

---

## DONE - SAT-DIAG (saturation and linearity limit gate)

**Status:** **IMPLEMENTED** 2026-08-13 (Milan authorized). Placed-aperture raw
peaks committed 2026-08-13 (replaces mag-guided search). `src_py/sat_diag.py`,
INV-SAT-01 wired. Reports: `dev/results/CURSOR_RESULT_sat_diag_implement.md`,
`dev/results/CURSOR_RESULT_placed_aperture.md`.

Draft 510 checksum: `dev/validation/anchor_510_checksums_placed_aperture_20260813.json`.

Remaining (not blocking): exposure-ramp measured linearity; placement-residual
WARN flag (spec 8.6 proposal); generic `anchor_integrity_check.py` tool.

---

## CLOSED - INV-CAL-01 / CAL-DIAG v2 (2026-08-13)

**Implemented and pushed** per `dev/results/specs/VYVAR_CAL_DIAG_V2_SPEC.md` (Milan authorized).
Replaces v1 and restores verification removed in `967f835`. Zero config keys;
`INDETERMINATE_NEGLIGIBLE` / `INDETERMINATE_UNMEASURED` split. P2 corrected for stage-aware
`calibrated/` compare (2026-08-13). Reports: `dev/results/CURSOR_RESULT_inv_cal_01_impl.md`,
`dev/results/CURSOR_RESULT_cal_mismatch_509_510.md`,
`dev/results/CURSOR_RESULT_inv_cal_01_p2_push.md`.

## DONE - INV-CAL-02 / calibrated product stage integrity (2026-08-13)

**Status:** IMPLEMENTED (Option A; Milan authorized 2026-08-13). Spec:
`dev/results/specs/VYVAR_CAL_STAGE_SPEC.md`. Report:
`dev/results/CURSOR_RESULT_inv_cal_02_impl.md`.

**Delivered:** `VY_CALSTAGE` + FITS `VY_CALDATASUM` stamped in same flush as pixel
mutation; `resolve_calibrated_stage` with honest `INDETERMINATE_*`; compare gates refuse
when stage unknown; force reapply stamps `SKYSF_N_R{pass}`; `qc_enrich_calibrated_lights_in_place`
rename (replaces misnamed `preprocess_calibrated_to_processed`).

**Deliberate departure:** field convention is immutable per-stage files; VYVAR keeps
in-place mutation at ~0 extra storage vs ~1.75 GB/draft duplicate tree (Option B), because
stamps close the reader-ambiguity gap (see DECISIONS INV-CAL-02-OPTION-A).

### Legacy stage census (measured 2026-08-13, pre-stamping archive)

Resolver run over all `calibrated/lights/` and `processed/` FITS under `Archive/Drafts/`:

| Resolution | Frames | Meaning |
|------------|--------|---------|
| `PURE` + LEGACY_INFERRED | **316** | Positive calibrate-only markers; no preprocess sky markers |
| `SKYSF_2` + LEGACY_INFERRED | **300** | `VY_SKYSF` + `VYSKYORD=2` present (509/510 era) |
| INDETERMINATE_LEGACY | **278** | e.g. `VYSKYP2P` without `VY_SKYSF` (435 `processed/` copy-tree era) |
| INDETERMINATE_UNKNOWN | **0** | No stage evidence at all |

**This is designed behaviour, not unfinished work.** Frames in the **278** bucket predate
`VY_CALSTAGE` stamping. Compare gates **correctly refuse** rather than assume `PURE` on
those frames. Validated science on drafts 435/509/510 does not require backfill.

Optional **read-only** `cal_stage.json` backfill (separate task) is a **convenience**
for investigators summarizing inferred stages -- not a migration requirement and not a
gate to closing INV-CAL-02. Re-run census after backfill or new reductions to detect drift.

## OPEN - INV-CAL-02 / calibrated product stage integrity (2026-08-13) [SUPERSEDED - see DONE above]

**Priority:** HIGH (provenance; prevents repeat of P2/P-10 investigation cost).

**Proposal (design only; not implemented):**

1. **`VY_CALSTAGE`** header + manifest field: `PURE` vs `SKYSF_N`.
2. **Pixel-array SHA-256** in manifest sidecar at each stage transition.
3. **Alert** when `VY_QCBG` disagrees with frame `nanmedian` beyond stated tolerance (e.g. 2 ADU).
4. **Architecture question:** keep mutating `calibrated/`, or write sky-subtracted product to a
   separate tree (e.g. `preprocessed/lights/`)?

**Related open items (relative priority in push report):** exposure-ramp linearity (SAT-DIAG Tier
2/3); generic checksum tool; INV-DAG-01 re-stamp friction.

## SUPERSEDED - CAL-DIAG (calibration-time radiometry gate) - **REMOVED 2026-08-11**

**Removed in commit `967f835`** ("Remove CAL-DIAG gate and its five config
parameters"). Spec retained for reference: `dev/results/specs/VYVAR_CAL_DIAG_SPEC.md`.
Historical validation: draft_424 150/150 `VY_DKRSMP=SUM`. Draft 435 carries legacy
headers; drafts 509+ do not.

## IN-FLIGHT - CAL-DIAG (calibration-time radiometry gate) - **IMPLEMENTED (pending Milan review)** [historical -- see SUPERSEDED above]

**Agreed 2026-07-07; spec APPROVED 2026-07-14 (Milan).** Spec: `dev/results/specs/VYVAR_CAL_DIAG_SPEC.md` v1.1.
Grounding: `CURSOR_RESULT_caldiag_flow.md`. Implementation: `CURSOR_RESULT_cal_diag_impl.md`.

Camera-agnostic calibration-time diagnostic gate (shipped):

1. **Post-dark-subtraction sky-median sanity** - median > 0 and plausible (reject negative/absurd
   calibrated sky).
2. **Pre-subtraction convention cross-check** - `median(light)` vs `median(resampled dark)`; detect
   SUM vs MEAN dark-resample mismatch with loud auto-correction (SUM -> MEAN retry) or fail-closed abort.
3. **Provenance** - `VY_DKRSMP` = `SUM` | `MEAN_AUTOCORRECTED` | `PASSTHROUGH`; `archive/<draft>/cal_diag.json`.

**Validation (2026-07-14 re-verify on HEAD 13341b3; pushed `237dd34`):** 14 pytest gate tests; full `pytest tests/` green;
draft_424 regression PASS (150/150 `VY_DKRSMP=SUM`, 0 WARN/FAIL; photometry core SHA
`bf3743a1...` unchanged).

**Related ledger (spec section 10 -- RESOLVED 2026-07-14 CAL-LEDGER-BUNDLE):**

| ID | Sev | Summary |
|----|-----|---------|
| **CAL-AGE-CLOCK** | MED | **RESOLVED** (`5143485`, `ee89de8`; warn centralized in bundle). `resolve_master_age` unifies import scan + library UI on header `VY_CDATE`/`DATE-OBS` with mtime fallback + one-time WARN. |
| **RN-HEADER-NONE** | LOW/MED | **RESOLVED** (`1830527`; bundle test). SNR precompute passes MASTERSTAR header to `resolve_read_noise` (bin-scaled RN). |
| **CAL-PASSTHRU-DEAD** | LOW | **RESOLVED** (`21c20e3`). Dead `allow_passthrough` branch removed; no production or test callers. |

Flat-norm Stage 2 and other ROADMAP calibration items unchanged (D1/D2/D3 codify current behavior).

---

## CLOSED - F-428 fix batch (draft_428 forensics, 2026-07-15)

Deferred findings from BO CVn UI run (`draft_000428`, `NoFilter_60_2`). Evidence: `infolog_20260715_130441.txt`.

| ID | Severity | Status | Notes |
|----|----------|--------|-------|
| F-428-VSXFLAG | HIGH | **FIXED** | `variable_targets.csv` path probe (platesolve setup dir vs `photometry/`) |
| F-428-REPAIR-FLOOD | MED | **FIXED** | Skip `DET_*` placeholders; aggregated `REPAIR summary:` |
| F-428-TAP | MED | **FIXED** | TAP retry + `hrd_enrich_tap_timeout_s`; summary.json + PDF note |
| F-428-EPSF-LOGNOISE | LOW | **FIXED** | ePSF skip logged once per run |
| F-428-LOGTZ | LOW | **FIXED** | Infolog timestamps unified UTC |
| F-428-AC-VISIBILITY | MED | **FIXED** | `ac_applied` / `ac_skip_reason` in summary CSV |
| F-428-EXCLUDED-VISIBILITY | MED | **FIXED** | `excluded_targets.csv` + INFO name list |
| F-428-MS-STAMP | MED | **FIXED** | `vsx_known_variable` via `catalog_id` join (`stamp_vsx_known_variable_on_masterstars`); draft_428 dry-run: 46->**197** id_join (**207** total) |
| F-428-COORD | MED | **CLOSED** (WCS-INV) | v5 RECLASSIFY-PROJECTION + fix batch: `wcs_invertibility.py`, SIP Gaia-sky refit, coord finalization, identity gate. draft_429 validated. |
| F-428-A3-RADIUS | - | **OPEN** (Milan) | Deeper/wider field DB (GAIA-DR4 ~Dec 2026); v5 T4 Poisson vs control reconciled (non-uniform cone export). |
| F-428-PASS2-CONTAM | - | **CLOSED** | 428 pass-2 census contamination CONFIRMED; see DECISIONS + `CURSOR_RESULT_429_validate_anchor.md`. |

---

## CLOSED - F-429 validate + regressions (2026-07-16)

| ID | Sev | Status | Notes |
|----|-----|--------|-------|
| F-429-STAMP-WIRE | MED | **FIXED** (`fc177be`) | VSX stamp after `finalize_masterstar_sky_coords`; tests added |
| F-429-AC-SUMMARY | LOW | **FIXED** (`fc177be`) | `[AC] run summary` via `log_event` |
| F-429-TAP-RETRY | LOW | **VERIFIED** | 429 `_hrd_cache` `enrich_attempts=3`; INFO per retry added |
| VALIDATE-429 | HIGH | **DONE** | draft_429 checklist; A1 FAILED-AS-WRITTEN (`git_dirty`); superseded by anchor pair |
| VL-ANCHOR-430 | HIGH | **DISQUALIFIED / RESTART** | draft_000430 non-anchor (mixed commits, dirty tree); fresh pair via `anchor_pair_run.py` |
| **F-431-HEADLESS-DIVERGENCE** | **HIGH** | **T1 FIXED (2026-07-27)** | Mono sky-surface restored (`ff08002`); guards INV-PREP-01/INV-MS-01 (`1191579`). C.4 BO CVn acceptance re-run on draft_452 in closeout. T2: await Milan clean UI. T3: **gated**. T4 partial landed. Blast radius: drafts 438-450 (historical). |

---

## OPEN - low priority hygiene

| ID | Sev | Item |
|----|-----|------|
| **BATCH-E-PARAMS-REGISTRY** | MED | **DONE (2026-08-04).** Commits `8094af8..33ec2dc`. Six batch E fields registered; VYVAR_PARAMS.md 277 entries; dashboard owner assert 249 config_runtime; FLOW threshold 3.8; BLE001 + ASCII hygiene. Result: `dev/results/CURSOR_RESULT_batch_e_params_hygiene.md`. |
| **DEPTH-AUTH-01** | LOW | Derive MASTERSTAR census/target depth from detection completeness vs Gaia (DAO-GAIA certificate per-mag recovery). Config constant G=15.0 today (`masterstar_gaia_census_target_depth_g`). C8-3: G=15.56 VSX stays absent at re-cut. Not wired. Evidence: `CURSOR_RESULT_CLOSEOUT_C8.md`. |
| **NOQA-TRUNCATED-EXCEPT-BULK** | LOW | 15 malformed `# noqa` directives across 10 `src_py` files, from truncated EXCEPT-BULK 2026-07-08 census comments; ruff emits Invalid-directive warnings; risk is that a genuinely malformed suppression hides in the noise. Sites listed in `CURSOR_RESULT_batch_e_params_hygiene.md` STEP 5C. Not started. |
| **ANCHOR-GATE-SEED** | -- | **CLOSED 2026-08-18 (ANCHOR-516-04).** Gates recut onto frozen 516: `session_baseline_check.py` DRAFT_ID=516 / snapshot `draft_000516_snapshot_cleanrebuild_20260818` / SHA `477dc8cf`; `test_invariants_p1_seed.py` same. 435 `5bccd85a` retired as superseded-with-pointer. |
| **ANCHOR-CLEAN-BUILD** | -- | **CLOSED 2026-08-18 (ANCHOR-516-04).** Canonical product is a clean Phase 0+MASTERSTAR-reannotate+P1+P2A rebuild of 516 on tip `4a65675` (provenance n_stale=0). `--full` copies frozen snapshot inputs into tmp; git-staged is FAIL. 435 never had a clean-tree generation and is retired. |
| **WIDE-ERR-POP-DELTA** | MED | Before any old-vs-new WIDE-ERR comparison, explain why the July generation has 333 core LCs / 166 check sidecars while the mutated August generation had 1121 / 248. Ratios 3.4x and 1.5x do not scale together, so the two populations are not directly comparable. Evidence: `dev/results/CURSOR_RESULT_anchor_prov.md`, `dev/results/CURSOR_RESULT_anchor_restore1.md`. Not started. |
| **WIDE-ERR-HONEYCUTT-PDF** | HIGH | Retrieve primary Honeycutt (1992) PASP 104:435 error section (eq. 3-5). Rengstorf 2004 eq. 7 & 13 are Honeycutt-attributed but not primary. Options: SAO archive, JADAAV, direct request to author estate. Without it AUDIT 1 row 3 stays DIFFERS-unclosed. Not started. |
| **WIDE-ERR-CROSSRIG** | HIGH | Measure per-comp excess on Newton/Dablice and Boyden data using `dev/tools/wide_err_e4.py` against a check-star draft on either rig. Deciding measurement for whether 20 mmag is wide-rig or pipeline. Not started. |
| **DB-DEFECT-DIAMETER** | HIGH | FITS APTDIA=70 mm contradicts DB TELESCOPE DIAMETER=200 mm. `fix_telescope_diameter.py` (JOURNAL:1116, SIGMA-A2) wrote focal length (200 mm) into the diameter column. Effect on scintillation: ~9% of WIDE-ERR gap. Independent of WIDE-ERR verdict. Not started. |
| ~~BORDER-PREALIGN~~ | LOW | **DONE (2026-07-24).** RAM-handoff / pre-alignment `[BORDER] Glob 0` -> explicit defer log; post-flush rewrite no longer silent (`FIELD-RUN-FINDINGS #11`). |
| A-DURABLE-UI | LOW | Save watched `.py` during alignment - pending Milan confirmation on 429 run |
| ~~ENCODING-POLICY~~ | MED | **DONE (2026-07-20).** ASCII-only tracked text + `test_ascii_policy` guard + `ascii_migrate.py`; EOL normalize (`* text=auto` + `.editorconfig`) in follow-up commit. Incident: cp1252/U+FFFD editor corruption class. |

---

## IN-FLIGHT - EXCEPT batch (silent broad-except census) - **CLOSED (2026-07-08)**

**Ledger:** `docs/VYVAR_EXCEPT_CENSUS.md` - **625 EVIDENCE** (all tranches); **40 fix-batch** sites
surfaced; remainder **EXCEPT-BULK** processed. Scanner: `scripts/_except_census_scan.py`.

| Tranche | Scope | Sites | Status |
|---------|-------|------:|--------|
| 1 | science kernel | 144 | EVIDENCE + FIX-1 + EXC-0626 |
| 2 | pipeline.py | 170 | EVIDENCE + FIX-2 |
| 3+3b | astrometry/import/database | 98 | EVIDENCE + FIX-3 |
| 4+4b | report/export/UI + remaining | 213 | EVIDENCE + FIX-4 + BULK |
| bulk | delete-dead / log / comment / narrow | remainder | **DONE** |

**Fixes landed:** 40 (FIX-1..4 + companions/closeouts). **Gate: 623 passed**, 15 skipped.
**Empirical check:** run `except_fix_counters` snapshot on next healthy draft - expect all zeros.

**Next:** none (batch closed). See JOURNAL for history pointer.

---

### Phase-1 graceful comp degradation - DONE (2026-06-16)

Validated matrix `matrix_20260616_164157.json`. Spec: `docs/VYVAR_COMP_DEGRADATION_SPEC.md`.
Known-issue **(b)** deferred to immediate next item (comp_rms gate authoritative for N_good).

---

### Simple differential + reporting + trust cleanup - DONE (2026-06-16)

- **Workstream A** (defaults + Phase-1 tier-ladder selector): DoD-A PASS 2026-06-15; commit `2a8355b`.
- **Workstream B** (reporting column grounded decision): DoD-B PASS 2026-06-15; `apply_reporting_postprocess`.
- **draft_409 Fixes 1-3** (trust/consistency): comp stability on per-frame ensemble residual;
  measured aperture + observed-band SNR sizing; `lc_rms (OOE)` for variables. Cross-validated vs
  SIPS on V0612 (eclipse + shared frame anomaly at ~JD 2461200.385).

Grounded specs: `VYVAR_DECISION_GROUNDING_RULE.md`, `VYVAR_REPORTING_COLUMN_GROUNDED_DECISION.md`,
`VYVAR_CANONICAL_COMBINATION_LOGIC.md`, `VYVAR_SIGMA_BUDGET_SPEC.md` (PARKED).

---

### Workstream B - DONE (DoD-B PASS 2026-06-15)

Tier-2 comp-ensemble extinction k **parked** (wide-field delta-airmass; ROADMAP).

Evidence: ``tmp/phase11/dod_b_workstream_b.json``; ``apply_reporting_postprocess`` in
``photometry_core.py``.

---

### Round 1 - four known fixes - DONE (2026-06-17)

Verified g-only on draft_413. A-durable (MP reload robustness), B-cap (spatial-first
variable_targets; comp-purity coupling Milan-accepted), measurable completeness gate, NoDetections
log-flood summary. Writeup: `CURSOR_RESULT_round1.md`; decisions in `VYVAR_DECISIONS.md`.

### Round 2 - frame-quality gate (B.2) DONE; aperture-skirt (B.1) REFUTED (2026-06-17)

Diagnosed and measured in isolation on draft_413 g; writeup `CURSOR_RESULT_round2.md`. Push gated on Milan.
- **B.1 aperture-skirt / COG - REFUTED, not implemented.** COG confirms 5 px aperture captures only
  EE=0.65, but widening to the plateau does not reduce differential scatter (flat 24->27 mmag) and
  adaptive sizing is worse; the skirt swing is common-mode (cancelled differentially). Decision in
  `VYVAR_DECISIONS.md`.
- **B.2 transparency frame-quality gate - DONE (default-OFF).** `frame_quality_gate_enabled` (+
  `ratio_k`, `fwhm_factor`, `min_keep_frames`) rejects PSF-collapsed frames via `flux_large/flux`
  robust-outlier + FWHM guard; cuts bright-target LC scatter median -257 mmag on draft_413 g.

**Round 2 open follow-ups (HIGH):**
- **OUTSTANDING - UI-VYVAR live test of A-durable** (the one validation still pending from this
  session). In the Streamlit UI, reach the alignment MP stage, save a watched `.py` to trigger a
  mid-run reload, and confirm **no PicklingError** (MP continues via fresh-attr dispatch, or the
  single-process fallback engages). Headless never reproduces it - must be exercised in the UI.
- **Structural comp / check-star yield = next lever for GREEN on bright/sparse fields** - post-B.2,
  trust RED is **structural** (check-star gaps, thin comps, colour-term-off), **not** LC scatter
  (the gate already fixed scatter, median -257 mmag). `green_min=3` is literature-grounded -> improve
  comp/check-star **yield**; do NOT relax the gate.
- **Flip-aware comp selection (candidate)** - when a meridian flip is detected, prefer **near-target**
  comps (per-side normalization as an alternative). Root cause = uncorrected flat-field on non-cal
  data under the 180 deg p->-p mapping; evidence = V0454 flip diagnostic (+-0.1 mag position-dependent
  post-flip step; `docs/round2_figs/v0454_flip_diag.png`, DECISIONS). Connects to the comp follow-up.
- **B.2 frame-quality gate Phase-0+1 extension (future)** - gate collapsed frames at
  **comp-selection** time too, not only Phase 2A; may recover comps and feed the structural follow-up.

### run-414 fixes - Fix A DONE; Fix B DONE; Fix C DIAGNOSED -> N/A; control-point-cap perf ticket OPEN (2026-06-18)

From the run-414 V0454 diagnostic (`CURSOR_RESULT_414_diag.md`) + the C1 diagnostic
(`CURSOR_RESULT_fixC_diag.md`). Fix A + Fix B pushed to origin/main 2026-06-18 (Milan-authorized).
- **Fix A - per-point `err` model - DONE (default).** `err` term-3 was std of comp instrumental mags
  (brightness spread, ~0.58 mag floor); now per-frame ensemble-ZP residual SEM. Decisions/JOURNAL.
- **Fix B - reject-on-alignment-residual gate - DONE (default-OFF), PERMANENT quality gate.**
  `frame_align_residual_gate_enabled` (+ `max_frac`=0.25 of aperture radius, `min_keep_frames`).
  Per-frame `align_residual_px` recorded always-on in `alignment_report.csv`. On run-414 g, ON drops the
  13 phase_correlation + 1 mis-aligned astroalign frame; OFF byte-identical. C1 confirmed Fix B + B.2 are
  the **correct permanent handling** of these frames (not a stop-gap awaiting Fix C).
- **Fix C - dense-field alignment recovery - DIAGNOSED -> NOT APPLICABLE (2026-06-18, C1).** The premise
  ("good data, only alignment failed -> recoverable") is **refuted**: the 14 frames are **PSF/FWHM-bloated**
  (median FWHM 8.60 px = 1.85x the good 4.64 px; concentration 13.1 vs 1.65; **corr(FWHM,residual)=0.95**).
  Bloated-donut centroid noise (~2.4 px) breaks astroalign (misalignment = symptom) and is what Fix-B/B.2
  measure. **Not recoverable to sub-px** (centroid floor ~2.4 px > 1.37 px gate; cap50->3/14, WCS absent
  0/162, translation-refine inapplicable). Likely **late-night focus drift on the defocused rig** (FWHM
  bloat, not a flux drop); post-flip-half-not-refocused is an observer question. **Closed** - recovery
  would be useless and risky. `CURSOR_RESULT_fixC_diag.md`, `tmp/phaseC1/fixC_root_cause.png`.
- **NEW - dense-field astroalign control-point cap (perf/robustness, OPEN, MED).** *Separate from Fix-C
  recovery.* astroalign at `mcp~200` on dense fields is ~654 s/frame (and still fails); cap to ~50
  (astroalign's design point) -> ~3-10 s. Two shapes: (i) **additive recovery rung** - cap tried only
  after the current attempts fail, before `phase_correlation` (hook: `vyvar_alignment_frame.`
  `_alignment_compute_one_frame`, between the `_attempts` loop end and the phase_correlation block);
  keeps the 147 byte-identical, does **not** fix slowness. (ii) **primary cap** - fixes slowness but
  changes the 147 transforms -> requires a **cross-rig regression (home + narrow rigs)** before adoption.
  **Defer until cross-rig data is available.** **Watch-item:** mildly-bloated near-threshold frames kept
  by the gate (e.g. g_0231, FWHM 1.10x) - likely benign (differential photometry cancels common-mode
  FWHM), watch the LC near the good->bad transition.

---

## NEXT SESSION - entry point (2026-07-15 ARCHIVE-CLEANUP)

**Start here:** `Archive/` cleared; anchor offline at `C:\ASTRO\backups\` (see VL-ANCHOR-424).
`--fast` daily; `--full` SUSPENDED until new anchor. First functional step: import new measurement
dataset into `Archive/Drafts/draft_NNNNNN`.

**Re-anchor row (OPEN -- pending new data):**

| Item | Status | Action |
|------|--------|--------|
| **VL-ANCHOR-424 / --full** | **SUPERSEDED offline** | Historical zip only; live anchor = VL-ANCHOR-WCSINV / draft_435 sky-surface |
| **VL-ANCHOR-WCSINV / --full** | **ACTIVE** | `draft_000516_snapshot_cleanrebuild_20260818`; core `477dc8cf` n=97 |
| eq4 bin4 bias/darks | HIGH (Milan) | >=6 frames, GAIN=12.48, T~-15 C |
| Fresh darks | HIGH (Milan) | Before ~2026-07-21 expiry |
| BVR night dX>=0.3 | HIGH (Milan) | Home rig; k'' NIGHT_FIT v2 |
| bin2 flats | MED (Milan) | Calibration inventory |

**0714 arcs (code unchanged):** SPARSE-TRUST CLOSED; k'' wide LOW / Newton OPEN; CAL-DIAG ON;
WSN PARKED. See session history in STATE.

## NEXT SESSION - entry point (2026-07-14 SESSION-CLOSE, superseded)

**0714 arcs closed (commit refs):**

| Arc | Final status | Ref |
|-----|--------------|-----|
| SPARSE-TRUST | CLOSED; SS Cam YELLOW confirmed | `7886157`, `CURSOR_RESULT_arc_close.md` |
| K2-COHORT / K2-STATS-FIX | wide LOW PRIORITY; Newton OPEN suggestive; DOWN retracted | `13341b3`, `CURSOR_RESULT_k2_stats_fix.md` |
| CAL-DIAG + CAL-LEDGER-BUNDLE | Gate ON; section-10 resolved | `237dd34`, `b268a6c` |
| WIDE-SLOPE-NOISE | PARKED unified ~5 mmag phenomenon | `114c423`, `CURSOR_RESULT_wsn2.md` |
| PROD-SIGMA-FLOOR + anchor | ACCEPTED `8fb21b3` / core `bf3743a1` | `CURSOR_RESULT_anchor_chain.md` |

**Standing open items - data-gated backlog (exact list):**

| Item | Priority | Action / trigger |
|------|----------|------------------|
| eq4 **bin4 bias/darks** | HIGH (Milan) | >=6 frames, GAIN=12.48, T~-15 C; empirical RN (Stage B blocked at 0 pairs) |
| **Fresh darks** | HIGH (Milan) | Before ~**2026-07-21** expiry |
| **BVR night** dX >= 0.3 | HIGH (Milan) | Home rig; k'' NIGHT_FIT v2 gate |
| **bin2 flats** | MED (Milan) | Calibration inventory |
| Newton nights for k'' n>=46 | MED (accumulates) | Re-test OPEN suggestive eq4 signature when constant-star pool reaches pre-registered power |
| PSF dense-field enablement | MED (gated) | Production PSF OFF; enable on Newton dense-field draft when available |

**Parked rows (revisit triggers):**

| Item | Status | Revisit when |
|------|--------|--------------|
| k'' wide eq1 | **LOW PRIORITY subdominance** (`13341b3`) | Filtered wide dataset or dominant slope-noise removed |
| **WIDE-SLOPE-NOISE** | **PARKED** (`114c423`) | New flats, defocus study, EPD decorrelation; bounds in `CURSOR_RESULT_wsn2.md` |
| k'' Newton eq4 | **OPEN suggestive** (`036a570`) | >=46 Newton constant stars (see data-gated row) |
| DAO-RECONCILE | Parked CLOSED 2026-07-09 | missed@G90 material on new rig/config |
| Colorfield PDF wiring | Parked | Milan reopens 12g-12g6 arc |
| Broeg IVW / sigma budget IVW flip | Parked | Sigma budget validates on Newton gate |
| rig-aware X2_RED (design note) | NOT implemented | Multi-night sigma_r on eq4 stabilizes rig noise estimate |
| dense-field astroalign cap | OPEN MED | Cross-rig regression data (home + narrow rigs) |
| EXTERNAL-XVAL / independent-tool | **CLOSED** 2026-08-17 | AIJ 6.0.10 vs VYVAR: 3.3 mmag RMS (5c, XVAL-AIJ-01) and 4.9 mmag RMS (production 4c, XVAL-AIJ-02) / 134 frames / BO CVn; frame-set 3.8 mmag; `CURSOR_RESULT_XVAL_AIJ_02.md` |
| EXTERNAL-XVAL / FWHM claim | MED | Confirm/deny "true ~7.7-8.6 px" before any factor retune |
| Frame-level CR rejection | MED | Shared V0612 anomaly class |
| TODO-MULTISET | FUTURE | Per-rig config for multi-rig production |
| Publication workstream | LAST | After functional backlog; see PUBLICATION section |

---

## NEXT SESSION - open items (historical backlog)

**Resume here when Milan data unblocks k'':** filtered V/B/R draft + Newton/Brno FITS FILTER capture
(see IN-FLIGHT section above). Until then, other HIGH items below remain valid but k'' is the designed
next calibration lever.

0. **Phase-1b - comp_rms gate authoritative for N_good - DONE (2026-06-16).** Gate-authority shipped;
   SS Cam sparse_fallback routing unchanged. SS Cam trust band **RESOLVED: YELLOW** (2026-07-14, Milan
   confirmed) - see COMP-POOL-R / SPARSE-TRUST ledger rows.

1. **Phase-2 comp degradation - sparse-comp sanity - CLOSED (2026-07-14 SPARSE-TRUST arc).** Sparse
   trust CI bands implemented + validated (S1-S4 PASS). SS Cam r_60_4 **RESOLVED: YELLOW**
   (R=2.008 [1.224, 3.886], external K, chi2_prod=21.38, n=2, N=25). Field-wide comp_rms headline
   remains a diagnostic only (offset structure ~95% cancels in differential). k'' per-rig:
   wide **LOW PRIORITY (subdominance argument)**; Newton **OPEN suggestive**
   (underpowered T1+T2 probes). Overall priority **UNCHANGED** (verbatim pre-registered rule).
   Results: CURSOR_RESULT_k2_cohort_correct.md, CURSOR_RESULT_k2_cohort.md,
   CURSOR_RESULT_sparse_check_pool.md, CURSOR_RESULT_arc_close.md. Diagnostic:
   (`tmp/sigma_budget/sparse_comp_diag.json`).

2. ~~**Sigma budget Phase A (wide rig)**~~ **DONE (2026-07-09, wide rig; Newton bin4 open).**
   Validated model: photon (+) Honeycutt ensemble SEM (+) **6.5 mmag floor**; scintillation ~2 mmag
   negligible on D=0.2 m; f_resid -> 0 (variant e). Attribution (draft_424): k2-signature pooled R^2~0 -
   **k'' would recover 0.0 mmag** (colour matching works; NIGHT_FIT v2 will not reduce this floor);
   phase/PRNU strongest candidate (6.5 -> 4.5 mmag); **~4.5 mmag rig constant** remains. Committed:
   `sigma_budget.py`, `scripts/chi2_sigma_gate.py`, `scripts/select_constant_calibrators.py`,
   `scripts/sigma_floor_attribution.py`, `scripts/bin4_sigma_forensics.py`, `scripts/fix_telescope_diameter.py`,
   `scripts/fix_draft_equipment.py`. **F-BINGAIN-1 fix landed (2026-07-10):** empirical background err term;
   Newton bin4 chi2 gate re-validation pending Milan review on reprocessed drafts.
   **Open decisions (Milan):**
   - **PROD-SIGMA-FLOOR:** add per-rig `sigma_sys` floor to production err (changes outputs -> re-anchor;
     bright-star AAVSO error bars currently underestimated by floor) - separate session.
   - **SIGMA-NEWTON:** archive draft_426 baseline **INVALIDATED** (SIGMA-PROV-FORENSIC). Fresh
     i_70_4 baseline: V0611 chi2=**2.13**, err=**0.0175 mag**. Unit fix landed; re-anchor with
     PROD-SIGMA-FLOOR. Result: ``CURSOR_RESULT_sigma_prov_forensic.md``.
   Blocks TODO-GS8 + TODO-MULTISET for IVW flip. **`delta_mag` flux-sum canonical until Newton gate passes.**

3. **EXTERNAL-XVAL - external validation campaign (MEDIUM).** Independent-tool
   cross-check **CLOSED** (XVAL-AIJ-01 + XVAL-AIJ-02): VYVAR and AstroImageJ
   agree to 3.3 mmag RMS (matched 5-star ensembles) and 4.9 mmag RMS on the
   production clean 4-comp ensemble over 134 epochs of a 0.47-mag eclipser;
   processing-chain (alignment+detrending) effect bounded at 3.8 mmag RMS
   with no airmass/time correlation. Remaining: FWHM external-validation
   claim (confirm/deny "true ~7.7-8.6 px" before any factor retune). Feeds
   **TODO-SEP-XVAL** ledger and the future publication Validation section.

4. **Frame-level CR / bad-pixel rejection (MEDIUM)** - shared V0612 anomaly (~JD 2461200.385) is a
   single-frame artifact matching SIPS; consider pixel-level CR rejection **before** photometry
   (not LC sigma-clip, which must preserve real eclipse features).

5. **Source_id exact-match audit (MEDIUM)** - verify loose `contains()` matching is confined to
   throwaway harnesses, NOT production (19-digit exact match; draft_409 audit neighbor contamination
   tied to Decimal-precision discipline in `build_gaia_catalog`).

6. **Optional fresh byte-identity anchor (LOW)** - cut new SHA + recipe on committed simple-diff path
   for regression on top of SIPS/AIJ empirical validation (Milan call).

7. **Brno / Newton characterization gate (finish)** - `g_60_4` solves on **production path**
   (draft_400: 75.5% brightest-N, WCS persists). **Per-set fault isolation shipped** (2026-06-14) -
   one bad filter no longer aborts RUN VYVAR; surviving sets reach photometry. **Open:** Milan
   **draft_401 UI sign-off + overlay**; Brno **r/i/z** end-to-end (draft_402: `r` hint_sep reject,
   `z` blocked by whole-run abort - now unblocked for survivors); gate `_brno_check` tail fixed
   (draft_400 / skip). **Plate-solver TASK 2 shipped (2026-06-14):** catalog-recovery gate +
   hint-as-prior; pending Milan `r` overlay + anchor/home-rig regression re-run.
8. **TODO-MULTISET** - per-telescope-set config (wide vs fine optics); blocks clean multi-rig
   production and crowding gating per rig.
9. ~~**Short-baseline LC quality (#3)**~~ **DONE (2026-06-10)** - `short_baseline` terminal class,
   config keys, trust YELLOW non-escalating, exportable. ~~Finding E~~ **re-checked (2026-06-11)**.
10. **Phase C catalog rebuild** - DR3 full-sky build completes on existing schema (G<=17.5).
   GAIA-1/GAIA-2 columns deferred to DR4 (~Dec 2026) -- see DECISIONS.
11. ~~**Chi_and_H baseline re-cut**~~ **DONE (2026-06-11)** - full zaloha anchor
   (core `3f7c9e7a...`, full `d5b72d08...`; draft_000387 re-cut x2); completeness
   gate in `night_run.py`. See STATE + RUNBOOK.
12. ~~**Trust Findings A/B + CS-1..4 + comp trust floor (Option B)**~~ **DONE (2026-06-11)** -
   specs under `docs/`; trust baseline 1382/106 on draft_387 at `comp_trust_min_comps=5`.
13. ~~**Broad-except hygiene (BLE001/E722 regression guard)**~~ **DONE (2026-06-11)** - see
   DECISIONS + `pyproject.toml` / pre-commit / `tests/test_ble001_regression.py`.
14. ~~**DEV-PROCESS-A - JSON pass/fail validation ledger**~~ **DONE (2026-07-08)** -
    `validation/VYVAR_VALIDATION_LEDGER.json` + `tests/test_validation_ledger.py` (frozen
    required-ID guard). Rules: agents edit only `passes` / `last_verified` / `commit` / `notes`.
15. ~~**DEV-PROCESS-B - session-start baseline-check script**~~ **DONE (2026-07-08)** -
    `dev/scripts/session_baseline_check.py` (`--fast` / `--full`); `--full` re-verifies the draft_435
    science anchor + `except_fix_counters` zero-check; documented in RUNBOOK + CLAUDE_OPERATING_PRINCIPLES.
16. **INSTALL-MANUAL** - user install + catalog build guide; Lenovo T460 + TODO-LIB.
13. ~~**Per-frame proc export perf (DAO pre-filter + Moffat gate)**~~ **DONE (2026-06-12)** -
   `_proc_drop_unmatched_dao_rows` before aperture/PSF; Moffat gated on `_run_epsf` only; ~4.8x on
   `draft_000389` B_60_1 (171 -> 36 s/frame). See JOURNAL + DECISIONS.
14. ~~**Sparse-only comp fallback**~~ **DONE (2026-06-11)** - default **ON**; anchor
    `3f7c9e7a` / `d5b72d08`; science-meaningful comparator for regression vs prior cut.
15. ~~**Comp-slope stability (B2+B1)**~~ **DONE (2026-06-11)** - common-mode detrend BJD sort +
    significance gate (`comp_slope_significance_k`); Honeycutt 1992 conditional. Anchor footprint
    LC-neutral on `draft_000387`.

---

## HIGH

- ~~**Sigma budget Phase A - validated per-measurement uncertainty (wide rig)**~~ **DONE (2026-07-09).**
  Howell + Honeycutt ensemble SEM + 6.5 mmag floor; scintillation negligible on wide rig; f_resid->0.
  Newton bin4 chi2 gate still open (F-BINGAIN-1). **Do not** flip ensemble combine to Broeg IVW until
  Newton gate passes. See `VYVAR_SIGMA_BUDGET_SPEC.md`, `scripts/chi2_sigma_gate.py`.

- **TODO-MULTISET - per-telescope-set config architecture.** One config per rig (wide
  Carl-Zeiss 200 mm + QHY294MM ~ 9.77 arcsec/px vs Newton 300/1200 + C3-26000 ~ 0.65 arcsec/px).
  Underpins per-set plate scale, aperture, and crowding gating; blocks clean multi-rig
  production.
- **TODO-GS8 - Multi-Night Global Matching + global ZP solver - FUTURE / nice-to-have
  (descoped from HIGH 2026-06-25).** Cross-night comp matching + inter-night zeropoint.
  NOT a priority: the canonical publishable unit is a single night (see DECISIONS,
  product-scope boundary). Build only if a long-baseline science case demands it.
  Dep: AAVSO validation (GS6b [OK]). ~2-4 days. -> **PUBLICATION** workstream.
  **Schema note:** ``FIELD_REGISTRY`` and ``COMP_STAR_LIBRARY`` are retained as idle
  FUTURE multi-night infrastructure (not dead schema); see TODO-GS8.
- ~~**APCORR-MIXEDFRAME**~~ **DONE (2026-07-19)** - all-or-nothing COG per night
  (`APCORR-MIXEDFRAME-ALLORNOTHING` in DECISIONS). After per-frame COG AC, if any science
  frame lacks usable `cog_ok`, Phase 2A disables COG for the entire night (standard Metoda B
  AC path for every row); provenance `cog_night_fallback` in `pipeline_meta.json`. COG remains
  default OFF pending validation. **FUTURE refinement:** per-frame `fallback_ee` wiring (not
  in this closure).
- **PSF on fine-scale (Newton ~ 0.65 arcsec/px) data.** Infrastructure is DONE and default **OFF**
  (wiring `psf_flux`->Phase 2A, adaptive selector, per-star quality + auto-fallback, spatial
  grid, grouper - all lose to aperture at 9.77 arcsec/px, correctly kept off). OPEN:
  - ~~**Crowding rule-2 bug**~~ **CLOSED (2026-06-09):** resolvable-blend->PSF (rule 2) was
    **removed** from adaptive routing (`photometry_core.py` ~5484); `is_blended` at 1.5 FWHM
    remains for crowding metrics only. Adaptive PSF uses faint-isolated rule only. NEIGHBOR-SUB
    is the blend path (not grouped PSF).
  - **TODO-PSF-NEIGHBOR-SUB** - subtract neighbour + aperture residual (gated OFF).
    **Steps 1-2a + pre-2b DONE**: A9 envelope, `psf_neighbor_sub.py`, fail-safe guards +
    `bright_close_regime` edge guard. Fine-scale draft 367: mismatch **~1.0**, HV **~83%**,
    FAIL-SILENT **0**; real crowding **sparse** (9 blended) -> **VALIDATED_FINE_SCALE_IDLE**;
    **2b deferred** until blended fine-scale field. Coarse bin2 remains **SAFE_LOW_YIELD**.
    Design: `docs/VYVAR_NEIGHBOR_SUB_DESIGN.md`; crowding: `docs/VYVAR_DRAFT367_CROWDING.md`.
  - **TODO-PSF-V3d-FINE-SCALE** - **DONE (2026-06-08)** harness `tests/validation/v3d_fine_scale.py`.
    Inject-and-recover at draft-367-like scale; PASS on accuracy/precision/calibration pillars.
    Report: `tier_v3d/v3d_fine_scale.md`. Production PSF still OFF; real-field enablement separate.
  - ~~**TODO-PSF-V3d-MIDMAG-BIAS**~~ **DONE (2026-06-09)** sky-only PSF fit weights
    (`psf_weight_mode=sky_only`; Astier 2013 / Lacroix 2025). V3d mid-mag <2%, drift sub-%.
    Report: `tier_v3d/v3d_weight_proof.md`. PSF arc ready to batch (pre-2b + V3d + sky + weights).
  - **TODO-PSF-MULTIFRAME** - multi-frame ePSF stacking (isolation part done).
  - **TODO-PSF-ASYMMETRY** - tracking-smear diagnostics (BO CVn right-tail PSF).
  - ~~**TODO-FWHM-CONSISTENCY**~~ **DONE (2026-06-09)** - `header_core_fwhm_px` in
    `masterstar_context.py`; `crowding_index._load_wcs_meta` + `psf_photometry.get_epsf_fwhm_from_context`
    prefer `VY_FWHM_GAUSS` -> `VY_FWHM_GAUSSIAN` -> `VY_FWHM`, matching aperture path
    (`pipeline.py:9206`). h & chi Per L: is_blended 77/87 -> 58/53; numeric SHA 770966c3 unchanged.
    See `docs/VYVAR_HCHIPER_CROWDING_RECOMPUTE.md`.
  - ~~**TODO-EPSF-1-FWHM-QC**~~ **DONE (2026-06-08)** - `_epsf_fwhm_native_from_profile`
    (azimuthally-binned radial profile); QC warning band [0.80, 1.25]. V3e PASS (NEW ratios
    1.038-1.049 on synthetic Moffat). Diagnostic only; numeric SHA 770966c3 unchanged. See
    `docs/VYVAR_EPSF_AUDIT.md`, `tier_v3e/v3e_epsf_fwhm.md`.
  - ~~**Realistic per-star PSF uncertainties**~~ **DONE (2026-06-09)** via sandwich variance
    (`psf_err_mode=sandwich_skyonly`; V3d P3 ~1 mag<=17). **2026-07-09 audit (HEAD f38e924):**
    four latent findings **FIXED** - PSF-ERR-DECOUPLED, PSF-AC-FALLBACK, PSF-FLAG-VESTIGIAL,
    PSF-LEGACY-SELECTOR. Enablement checklist: real-field Newton draft + ERR/AC gates DONE;
    production PSF still OFF (`psf_photometry_enabled=False`).
- **Per-frame saturation (not whole-star skip).** Implemented gated OFF
  (`per_frame_saturation_enabled`, default false) + synthetic validation
  (PER-FRAME-SAT-GATED, 2026-07-19). Master zone remains the default whole-star
  skip; when ON, targets use clean-frame fraction vs
  `per_frame_sat_min_clean_frac`. **Validation pending** the next photometric
  night containing saturated bright stars (M67 astrophoto evidence re-graded
  HIGH->MED - extreme long-exposure case, dataset gone). Keep open until that
  revisit.

## MEDIUM

- ~~**TODO-COMP-P2P-RESIDUAL**~~ **DONE (already implemented; found stale 2026-07-19).**
  `check_comparison_stability` (`photometry_core.py`) computes p2p on
  `_comp_lc_frame_ensemble_residual` output with `_common_mode_detrend_comp_lc` applied
  (`common_mode_detrend=True` default). Docstring: "Abbeho point-to-point scatter on
  common-mode-detrended comp residuals"; run flag `common_mode_detrend_applied` logged.
  Caller audit: production call sites (`photometry_core`, `check_star_kmag`, `method_lc_output`)
  and tracked scripts all pass `common_mode_detrend=True`; no caller disables it.
- **Expose comp_qa / trust as Settings toggles** + write their defaults to
  `config.json`. These are **user-facing** QA features (the trust badge is observer-facing),
  currently config-only and UI-hidden. Add a "Data quality & validation" Settings section for
  the two. Keep the experimental `phase01_comparison_proximity_tiebreak` / `rms_bin_mag` and
  the PSF/COG/crowding flags hidden.
- **TODO-SEP-XVAL - Persist offline cross-val results to a tracked ledger.** Append one summary row per
  `xval_run.py` harness run to `validation/xval_ledger.csv` (in git):
  `date, draft_id, engine, n_targets, median_ratio, n_confirmed, n_review, n_indep_failed,
  commit, notes`. The offline `xval_out/` stays scratch; per-draft photometry tree is unchanged
  (no in-pipeline `sep_xval_*` artifacts). Gives a longitudinal validation record (mission +
  paper Validation section). *(Re-scoped 2026-06-03 after in-pipeline sep_xval retirement.)*
  -> **PUBLICATION** workstream.
- **TODO-APCORR-COLOR - Extrapolation guard warn->block: DONE** (2026-06-03). Target BP-RP
  outside comp range -> CT skipped, target kept uncorrected (`phase01_ct_extrapolation_tol`,
  default strict 0). **NoFilter CT enable: PARKED** - prototype on draft_000366 showed modest
  effect; **filtered drafts now exist (Milan, 2026-07-09) - revisit when scheduled; do not
  unblock here.** Revisit when filtered / Newton fine-scale data is re-verified.
- **Verify wide-field WCS distortion model (from SIPS comparison).** Confirm whether the
  plate solve fits higher-order distortion (SIP-style) or only a linear CD matrix. On the wide
  rig (~5.6 degx3.8 deg, with a corrector) residual field curvature can reach several pixels at the
  edges -> edge astrometry/photometry offsets. SIPS models this with a 3rd-order 2D polynomial
  (Monomial/Legendre). **Verify first:** if SIP is already fitted, close this; if linear-only,
  add higher-order distortion terms. (See DECISIONS: *What VYVAR deliberately does NOT adopt from SIPS* - WCS distortion scoped here.)
- **TODO-GS10 - AAVSO Direct API upload** ("Submit to AAVSO" button; WebObs API after the
  GS6b validation). Dep GS6b [OK].
- **TODO-45 - RGB camera support** (IMX533 RGGB). **OSC arc phase 1 DONE (2026-07-22):**
  CFA calibrate + plane-split extraction to `oneRGGB`/R/G/B obs-groups (`osc_extract.py`,
  `EQUIPMENTS.BAYERMASK`, OSC-01). **OSC-2 DONE (2026-07-22):** WCS solve-once on oneRGGB,
  registration handoff + per-channel photometry path (`osc_align.py`, OSC-02). **OSC-3 DONE
  (2026-07-22):** TR/TG/TB AAVSO/VarAstro exports, Gaia->Johnson comp mags (`gaia_johnson.py`,
  OSC-03). Arc code-complete; M71 E2E acceptance pending Milan. Gap inventory
  `dev/results/OSC_GAP_INVENTORY.md`; M71 validation dataset (eq id=5).
- **TODO-8-BOO - Bootes globular-cluster validation** (ePSF vs aperture on a dense ~2 h
  field). Pairs with the PSF/Newton work.
- ~~**TODO-FORCED-COMP - forced-aperture `catalog_only` without Phase-1 tier selection.**~~ **SUPERSEDED (`7f0dc86`)** - forced-aperture / catalog_only path removed; DAO+Gaia matched only.
- **TODO-LC-TREND - differential extinction + ALG audit** (partial; re-validate on a
  moonless night).
- **TODO-LIB - Cython `.pyd` compilation** (hide source, enable C translation).
- **TODO-CONFIG-CHURN - dedicated day.** The app rewrites session/UI state into the tracked
  `config.json` each run -> perpetual git diff. **Zero functional effect** (the resolver
  ignores the config site; UI uses `LOCATION.IS_DEFAULT`). Fix = split a session-state store
  from the static overrides. **Do NOT gitignore `config.json`** - it holds real overrides.
- ~~**TODO-BROAD-EXCEPT-HYGIENE**~~ **DONE (2026-06-11).** Phase G critical-path logging
  (2026-06-08) + **BLE001/E722 regression guard** (pyproject, pre-commit, pytest); 168 sites
  grandfathered `# noqa: BLE001`; 4 bare excepts fixed; 8 `photometry_core` narrowings. ~1200
  pre-existing noqa sites remain opportunistic. MASTERSTAR writeto fail-closed + edge-ok
  fail-open (#4) unchanged.
- **Phase H cosmetic lint (2026-06-08, DONE):** value-filtered subset applied (SIM118 x11,
  RUF022 x2, RUF007 x2, RUF034 x3 dead-ternary); ProcFrameStore SIM118 x2 kept; ~89 style
  findings accepted per PROCESS. **Clean-code campaign Phases A-H COMPLETE.**
- ~~**TODO-GEO**~~ **CLOSED (2026-06-09)** - superseded by PARAM-PROVENANCE (`param_resolver.py`:
  per-draft `ID_LOCATION` -> header -> config; BJD/airmass config-independent). See DECISIONS.
- ~~**Comparison-star floor policy**~~ **DONE (2026-06-11, Option B).** Trust-only
  `comp_trust_min_comps=5`; Phase-1 selection min stays 3. Spec:
  `VYVAR_COMP_FLOOR_POLICY_SPEC.md`. Option A (selection floor 5) parked - moves anchor.
- **`classify_lc_quality` short-baseline (#3).** **DONE (2026-06-10)** - see
  `VYVAR_LC_QUALITY_SHORT_BASELINE_SPEC.md`. Follow-up: vsx_type-aware thresholds.
- ~~**Night-run false success (draft_383)**~~ **DONE (2026-06-11)** - completeness gate in
  `night_run.py` (>=90% summary/active per setup).
- ~~**Check-star trust audit CS-1..4**~~ **DONE (2026-06-11)** - specs +
  `VYVAR_CHECKSTAR_SELECTION_SPEC.md`; CS-3 ensemble exclusion via `ensemble_ids`.
- **Reserved check-star (hold-one-out).** PARKED - moves photometry anchor; see DECISIONS.
- **AAVSO-standard output (#4).** PARKED - G->standard B/V/Rc (Broeg band/colour point).

## LOW

- **Gaia local DB: add pmra/pmdec columns (MATCHER-FIX-3 follow-up).** Current
  ``vyvar_gaia_dr3.db`` build returns ``pm_columns_present=false``, ``n_pm_finite=0`` on the dev
  machine; positions cannot propagate from Gaia epoch 2016.0 to VSX J2000 and the proper-motion tail
  stays in the separation residual. Add PM columns at the next catalog rebuild (benefits astrometry
  across VYVAR, not only the VSX matcher). Not a blocker for the contamination-budget acceptance fix.
- **GAIA-1 / GAIA-2 (pmra/pmdec, ruwe)** -- **DEFERRED to Gaia DR4 build** (~Dec 2026). Not
  restarting the DR3 rebuild. See DECISIONS + `VYVAR_GAIA_DR3_AUDIT.md`. DR4 migration hooks
  (epoch J2017.5, build columns, lite-table check) recorded in DECISIONS.
- ~~**comp_qa fix-once magnitude locus (CQ-C)**~~ **DONE (2026-06-09)** - fix-once pass-1 locus;
  order-independent flagging; bounded diff 1 flag / 1 n_clean / 0 trust on draft_000366; SHA
  `edbd97e7...` (426 files incl. comp_qa). Sibling ddof+threshold co-calibration remains open.
- **FITS-side proc glob consistency.** `pipeline.py` uses inline `aligned_dir.glob("proc_*.fits")`
  (~5578, 12604, 12683) rather than a shared helper; functionally correct (`proc_*` matches both
  naming styles). Optional consistency cleanup only.
- **Spatial term in calibration (from SIPS comparison) - only for a future whole-field absolute
  mode.** SIPS's ensemble adds x,y polynomial terms (`x1.X + y1.Y + x2.X^2 + y2.Y^2 + xy.XY`) for
  field gradient / vignetting. **Not a gap in VYVAR's current per-target differential path** -
  the local-comp ensemble already cancels most spatial systematics. Relevant only if/when VYVAR
  does whole-field absolute photometry (all targets against one frame-wide solution). Sibling of
  APCORR-COLOR (the colour term). (See DECISIONS: *What VYVAR deliberately does NOT adopt from SIPS* - spatial term scoped here.)
- ~~**TODO-WIDE-RIG-REPROCESS - clean re-run of 361/362.**~~ **CLOSED (OBE, 2026-07-09):** drafts
  361/362 deleted; item no longer actionable.
- ~~**B-V legacy removal - Stages 2-4.**~~ **Closed 2026-06-03** (scope A+B; commits in
  JOURNAL). Regenerate `vyvar_vsx_local.db` on the catalog machine after pulling.
- **TODO-10 - Settings-tab refactor + `CONFIG_GUIDE.md`.** Ties to `VYVAR_PARAMS.md` /
  config<->UI parity.
- **DAO-RECONCILE - Gaia<->DAO field accounting (reframes TODO-13).** **CLOSED
  (2026-07-09).** Arc: R-1 (`becc274`/`e9daec9`) population bug fix -> R-2 (`bd6244a`/
  `4279a52`/`b7df7c6`) footprint+Fleming fit -> R-2b (`78febea`) censoring+miss@G90 -> close
  (`DAO-RECONCILE-CLOSE`) flat-curve no-crossing fix. **Outcome:** completeness 89.7-98.3%
  across rigs; G_lim characterized (wide ~15.0, Newton V ~16.7, B/R >=17.5 censored,
  narrow-band >=17.5 no-crossing); `completeness_50` + `missed_below_g90` live in QA dashboard
  as ongoing health signals; original TODO-13 fully superseded. Diagnostic:
  `scripts/dao_reconcile_diag.py --all-drafts`.
  **2-pass DAO recovery: CLOSED (not-worth-complexity, 2026-07-09).** Decision basis: after
  the reference-population fix, true anomalies (missed@G90) are 15 (424 wide), 6/18 (427 g/r),
  10 (425 V); 425 B/R misses (314/212 under censored depth) are the practical ceiling of
  catalog cross-match (Gaia near-bright-star spurious sources, sub-blend-radius pairs, proper
  motion) - 2-pass would harvest artifacts, not signal. **REOPEN CONDITION:** missed@G90
  becomes material (hundreds) on a new rig/config in the QA dashboard.
  **PUB-QC-MISSRESIDUAL (PARKED, LOW):** positional classification of the 425 B/R missed
  sample (near-bright-star distance, pair separation, PM) - explains the residual 2-3%, strong
  QC passage for future publication Validation section. Not scheduled.
  **Artificial-star injection tests** (photutils) = future gold-standard validation; candidate
  for EXTERNAL-XVAL / publication Validation section.
- **TODO-LC-QUALITY - LC classification filter.** `lc_quality_flag` exists and is consumed by
  the trust gate; verify the saturated/noisy export policy is complete.
- **TODO-14 - PDF size optimization** (29 MB -> < 10 MB).
- ~~**TODO-MASTERSTAR-QA - FORCED_APERTURE cyan overlay** in the QA UI.~~ **SUPERSEDED (`7f0dc86`)** - `FORCED_APERTURE` proc rows no longer produced.
- **Misc LOW:** TODO-7 plate-solver refactor . TODO-11 auto-trigger watchdog (`night_run`
  foundation exists) . ~~TODO-12 HRD classification (after new DB)~~ **DONE (2026-07-10):** session-aware
  HRD extreme-object table; online Gaia TAP + SIMBAD enrichment for candidates only (lite DB teff/logg
  NULL); absolute thresholds (Pecaut & Mamajek 2013; Andrae et al. 2023). **TODO-12b (2026-07-10):**
  parallax gate 0.15 mas + SNR 5 (config-driven); per-category cap 3; NSS deprioritized in enrich budget;
  apparent-G legend/caption fix (chi Per / draft_425 cluster admitted to M_G plane). **TODO-12c (2026-07-10):**
  luminosity-first Stage-2 label priority (RSG before Very cool); Stage-1 per-net reserved slots
  (`hrd_min_per_net=4`). **TODO-12d (2026-07-10):** `hrd_nss_category_enabled=False` drops NSS/binary
  from table; annotated MASTERSTAR field image (scale guard, short labels, legend) wired to PDF HRD page.
  **TODO-12e (2026-07-10):** identification tiers (confirmed/likely/candidate); enrichment cache v2
  (+ SIMBAD sp_type, Gaia DSC); SIMBAD lum-class logg substitute (RS Per RSG fix).
  **TODO-12f (2026-07-10):** PDF/UI `Extreme objects -- details` per-object blocks; extended row
  payload (dist, parallax, raw SpT/otype, DSC p); validate summary stamps.
  Arc **DONE** - see `CURSOR_RESULT_todo12_hrd.md` ... `_todo12f_hrd.md`.
  **TODO-12g (2026-07-10):** catalog-color field rendering (mono luminance x Gaia BP-RP chrominance;
  Wyman et al. 2013 CMFs; Pecaut & Mamajek BP-RP->Teff; UI expander + validate PNGs; no photometry/PDF).
  **TODO-12g2 (2026-07-10):** polish -- hue-preserving highlight scale, chroma SNR gate (background
  speckle), field-median white point (relative colors; reddened chi/h Per B stars correctly near
  neutral under absolute D65); saturation default 0.85.
  **TODO-12g4 (2026-07-11):** chroma boost (distance-from-white expansion, default 1.6, caption-disclosed)
  - validate draft_424 + draft_425 V at boost 1.0/1.6/2.2; draft_424 blotchy at 2.2.
  **TODO-12g5 (2026-07-11):** local-background SNR gate (`hrd_color_bg_box_px`), tapered splat stamp,
  hardened G2 (8x6 worst-patch grid + heatmap); fixes draft_424 sky blotches at high boost.
  **TODO-12g6 (2026-07-11):** caption provenance stamps (UTC + git short hash); boost default 2.2;
  archive overlapping `tmp/todo12_hrd/` runs; canonical `tmp/colorfield_final/` + manifest.
  for visible Teff contrast when stars sit near the white point; display enhancement only.
  . TODO-20 mean-stack
  MASTERSTAR (improves WCS/FWHM only, not LC SNR) . TODO-CACHE-CENTRAL centralize `csv_cache`
  . TODO-PIXEL-XCHECK-BINNING binning-aware pixel cross-check (cosmetic log) .
  **TODO-RECUT-HARNESS-FIDELITY** - **CLOSED (superseded, 2026-07-08):** draft_387 zaloha no longer
  exists (Milan confirmed); frozen-anchor premise dead. Current regression references: the draft_435
  skysurface anchor + `dev/scripts/session_baseline_check.py --full`; rebaseline cut1 SHAs where
  fixture present (`dev/tests/test_photometry_sha_baseline.py`). Historical SHA constants in
  `dev/tests/photometry_sha.py` retained (self-skipping).
  **TODO-INSTALL-MANUAL - instalacny manual + instalator pre noveho uzivatela (vratane
  katalogov)** . TODO-PLATESCALE-PERSET focalxpixel per-set plate-scale fallback.

### TODO-INSTALL-MANUAL (naviazane: **TODO-9**, **TODO-LIB**)

**Ciel:** novy uzivatel (bez znalosti vnutra) nainstaluje cely VYVAR + sprevadzkuje katalogy
podla manualu; na referencnom stroji **LENOVO T460** (Linux - potvrdit presnu distro) to spravi
**jednym instalatorom**. Instalator = realizacia; manual = dokumentacia toho isteho ciela.
Prepojit s Cython balikom (**TODO-LIB**). Nahradza/rozsiruje stary one-liner **TODO-9**.

**Manual (dokumentacia):**
1. Prostredie - Python, venv, `requirements.txt`, spustenie Streamlit dashboardu.
2. Aplikacia - ziskanie repa/balika, prve spustenie.
3. **Katalogy (klucove)** - 3 gitignored subory: `vyvar_gaia_dr3.db` +
   `gaia_triangles_fine.pkl` + `gaia_triangles_wide.pkl`. Dve cesty: **(a) build**
   (`build_gaia_catalog.py` -> ESA TAP, hodiny; potom `build_blind_index.py`) alebo **(b) stiahnut
   hotove** (ak hostovane). Na T460 je full-sky build @ G<=16.5 neprakticky -> uprednostnit hotove
   (prip. slim podmnozina).
4. Konfiguracia - Settings: `GAIA_DB_PATH`, `BLIND_INDEX_FINE_PATH`, `BLIND_INDEX_WIDE_PATH`,
   `archive_root`, `calibration_library_root`, `database_path`; Test connection / Skontrolovat index.
5. Prvy beh - verifikacny checklist (test solve / kratky pipeline, trust gate GREEN).

**Instalator (T460):** skript/balik pripravi prostredie + app + nasmeruje na katalogy (download
hotovych alebo build). Zohladnit RAM (fine PKL ~1.3 GB + in-memory verify katalog), disk (~10 GB DB),
CPU. Jedna zdrojova pravda s manualom.

**Otvorene (rozhodnut pri implementacii):** distribucia katalogov (build vs download vs slim);
presna OS/distribucia T460; forma instalatora (shell / conda / pip / Cython wheel); min. RAM/disk v
manuali.

**Definition of Done:** manual -> cista instalacia od nuly po GREEN trust na testovacom poli;
instalator -> rovnaky vysledok na T460; manual a instalator konzistentne.

### Plate-solver robustness backlog (future, lower priority)

**Future (lower priority):**
1. **Odds-based verification** - replace flat brightest-N % with confidence/odds vs chance (cone
   density aware; astrometry.net lesson).
2. **RANSAC / sigma-clipped SIP on inlier pairs** - fit distortion on inliers when ~35% spurious
   pairs prevent SIP adoption (`lin <= sip` guard).
3. **Geometry/position verification** - score full matched-set position residuals (band-agnostic)
   instead of brightness-ranked brightest-N alone.

## FUTURE

- **Blind index - 3rd rig tier (Noctutec 206/560).** When a validated draft exists, add a
  third PKL tier + config path; architecture is ready (`blind_index_fine_path` /
  `blind_index_wide_path` + `blind_index_select_mode=auto`).
- **TODO-GS7 - paper draft** (PASP / AN). Working title locked: *VYVAR: An Automated
  Differential Photometry Pipeline for Amateur Variable Star Observers*.
- **Comet photometry mode** - major parallel phase **after** the variable-star pipeline is
  finished (shared front-end calibrate->platesolve->star-stack->Gaia ZP; forked back-end:
  comet-rate stacking + extended coma photometry + ICQ/COBS export). Analysis only - do NOT
  start yet. (DECISIONS.)
- **TODO-SCENE-FORWARD-MODEL** - conditional on crowded-faint science (Brno / globular
  clusters); priority lowered after the grouper-negative result.

---

## Parked (next - Milan chooses, 2026-06-14)

| Item | Notes |
|------|-------|
| **CM-detrend differential** | ~10x lever; opt-in; needs transit injection-recovery test before opt-in |
| **Exoplanet / TOI catalog** | **DONE** - NASA Exoplanet Archive TAP (14,185 rows), TOI annotation live; see JOURNAL 2026-07-08 EXO entry |
| **Newton-V colour-term** | Per-rig c1 from field BP-RP |
| **Meridian-flip handling** | Qatar-8 class |
| **`[TODO-RECUT-HARNESS-FIDELITY]`** | **CLOSED (superseded 2026-07-08)** - draft_387 zaloha gone; use draft_435 + DEV-PROCESS-B --full |
| **Pre-filled camera catalog** | New-user onboarding - sensor-keyed research, camera-keyed rows; **PARKED** (see below) |

**Backlog (unchanged):** broad-except Tier-1 (~25); B-V legacy removal Stages 2-4; TODO-46 (skip
airmass detrend for known VSX variables); TODO-LC-QUALITY; TODO-LC-TREND; TODO-GEO; GS8-GS11.

### TODO (PARKED) - Pre-filled camera catalog for new-user onboarding

**Idea:** Ship VYVAR with a curated catalog of monochrome astronomy cameras (the ones amateurs /
professionals actually use). A new user picks their camera from the list to pre-populate their
EQUIPMENTS row and set it as default, instead of hand-entering specs. Lowers the onboarding barrier
(fits the non-expert / "trust in the numbers" mission).

**Status:** PARKED - discuss scope + format with Milan before any implementation.

**Design decisions already reached (do not relitigate without discussion):**

- **Two-tier table.**
  - *Static / authoritative (safe to ship as fixed):* sensor model, pixel size, sensor dimensions,
    bit depth, nominal saturation / full-well. These are genuinely fixed per camera/sensor.
  - *Setting-dependent (ship as "base defaults", NOT authority):* gain (e-/ADU) and read noise.
    These are **not fixed** - they vary with the gain setting and binning. Store them **per gain
    mode** (e.g. gain-0/LCG, unity, HCG/high), clearly marked "nominal, verify against your FITS
    header / measurement." A single value per camera would give a false sense of correctness.
- **Integrates with the header-first gain logic (already implemented).** Catalog = base/fallback;
  the FITS header overrides gain per-session when it carries e-/ADU (Moravian). Read noise has no
  header source in tested cameras, so the catalog/DB per-mode RN is what's used -> per-mode RN
  matters most.
- **"Trust in the numbers":** recommend the user measure their own gain/RN (photon-transfer) or rely
  on the header value; the catalog is a starting point, not a substitute.
- **Research sensor-keyed, deliver camera-keyed.** Many cameras share one Sony BSI sensor
  (IMX571 = ASI2600/QHY268/C3-26000; IMX455 = ASI6200/QHY600; etc.). Research gain/RN at the sensor
  level, deliver rows that map to EQUIPMENTS columns (CAMERANAME, SENSORTYPE, SENSORSIZE, PIXELSIZE,
  SATURATE_ADU, GAIN_ADU, READNOISE_E; FOCAL stays null = per telescope).
- **Suggested first scope:** dominant Sony BSI mono family + the cameras using each - IMX571, IMX455,
  IMX411, IMX492/294, IMX533, IMX585, IMX461, plus legacy IMX174/178/290 and Panasonic MN34230
  (ASI1600). Covers ~90% of real usage. Not an exhaustive "every camera" list (scope + accuracy).
- **Sources:** manufacturer spec pages + peer-reviewed characterizations (e.g. Alarcon et al. for
  IMX455/IMX411), not memory.

**Open questions to settle before implementing:**

1. Scope breadth (start with the Sony BSI family above? wider/narrower?).
2. Format: seed data the app offers for selection (CSV/SQL) vs a reference document for review first.
3. Schema: does EQUIPMENTS need a gain-mode dimension (per-mode gain/RN rows), or one base row per
   camera + reliance on header-first gain for the per-session value?

---

## Parked (round 2 - refinements, not blocking)

- **Magnitude-aware check-star threshold for the trust gate.** The flat `0.02` / `0.05`
  cutoffs carry the same magnitude-dependence the comp_qa locus fixed; extend the
  Sokolovsky/locus treatment to the check-star axis.
- **PSF cross-validation** - needs a PSF-heavy/faint draft + per-frame ePSF (the aperture
  cross-val is CLOSED).

## Dropped / resolved (do not re-open)

- ~~**GAPS A1 / AUTO-VSX-LIMIT**~~ **DONE (report layer, 2026-07-19).** PDF compares
  `vsx_variable_targets_mag_limit` to measured field depth (`G_lim_90`, SNR5) and
  warns when the limit is deeper by >0.3 mag. No selection-behavior change; no new
  config keys. **FUTURE:** opt-in automation of the limit from depth.
- **TODO-GS9 - Ground-LC period analysis in the PDF** - **closed: descoped 2026-06-09.**
  Lomb-Scargle/BLS + folded diagram on VYVAR's own Phase-2A LC as a PDF science product is
  out of scope; period finding/classification is downstream (Peranso, VStar, Period04). See
  DECISIONS (product scope boundary). LS/BLS citations remain for `tess_verify` TESS cross-check.
- **Blind solver in dense fields + index series + rig-prior** - **RESOLVED 2026-06-04** (Newton):
  mag14 tiers, `vyvar_blind_series`, solve-rate harness, scale/FOV hard gates (`blind_use_rig_prior`,
  `blind_scale_tol_frac`). ~~**Wide-rig blind HIT**~~ **CLOSED (2026-07-09):** believed resolved by
  later solver work (Milan); reopen on the next wide-rig blind-solve failure.
- **V/R re-run (draft_375)** - **RESOLVED 2026-06-04** via draft_380 clean full run (all filters).
- **Trust `n_clean=0` diagnosis** - **RESOLVED 2026-06-04:** root cause = pre-cal proc-CSV glob in
  `load_proc_pivot` (draft-specific, not a cleaning regression); folded into canonical pre-cal
  proc-CSV resolution (HIGH).
- **Canonical pre-cal proc-CSV resolution** - **CLOSED 2026-06-08:** `load_proc_pivot` uses
  `list_proc_csvs` / `PROC_CSV_GLOB="proc_*.csv"`; verified `tests/test_proc_csv_glob.py` +
  calibrated draft_000366 n_clean populated.
- **ProcFrameStore pre-cal naming** - folded into canonical pre-cal proc-CSV resolution (HIGH,
  2026-06-04); do not fix per-consumer.
- **MASTERSTAR-EPSF-ALL** - dropped 2026-06-02: plate scale is WCS-derived; affected drafts
  311/321/358 are deleted; 361/362 ePSF already ~ 9.77. No recurrence risk.
- **GS6b** - DONE 2026-05-20 (`scripts/validate_aavso_export.py`). Residual delta only: add a
  headroom check so the new `trust=` AAVSO NOTES field stays within Extended-Format limits.
- **IRAF / PyRAF cross-val** (and TODO-32 EPADU) - closed as unnecessary; two independent
  engines (sep matches VYVAR to 0.2 %) already validate extraction; not feasible on
  Py3.12/Ubuntu24.
- **TODO-WEIGHTED-LC, TODO-SKY-PLANE** - tested negative, closed.
- ~~**TODO-DEV-PROCESS**~~ **DONE (2026-07-08)** - **DEV-PROCESS-A** (validation ledger) +
  **DEV-PROCESS-B** (`session_baseline_check.py`); Definition-of-Done discipline in
  `VYVAR_PROCESS.md`. Charter: `VYVAR_CLAUDE_OPERATING_PRINCIPLES.md`.
- The full **TODO-1...45 / PERF-1...10 / ALG-1...5 / CQ-1...7 / GS1-GS5** series - closed; see
  `VYVAR_JOURNAL.md`.

---

## Parked for next session (2026-06-11)

| Item | Notes |
|------|-------|
| Reserved check-star | Hold-one-out by design - **moves photometry anchor** |
| AAVSO-standard output #4 | G->B/V/Rc (Broeg band/colour) |
| TODO-MULTISET | Per-rig config (wide vs fine) |
| TODO-GS8 | Phase-3 global ZP |
| DR4 build | ~Dec 2026; J2017.5 epoch hook `vyvar_platesolver.py:63` |
| PSF / NEIGHBOR-SUB | Needs bin1 ~0.65"/px data (Brno gate) |
| `build_gaia_catalog.py` adaptive-split | Next full-sky build only (not this commit) |
| **D1-combination (Broeg-weighted vs flux-sum)** | Re-test weighted `ens_med` after **sigma budget + chi^2 gate** - **moves anchor**; blocked on sigma completeness |
| **D3 extinction/colour physics** | Second-order extinction + standard system -> ties **AAVSO #4** |
| **C second pass (Howell + aperture corr.)** | CCD error budget, curve-of-growth / APCORR; citation-integrity follow-up |
