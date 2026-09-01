# VYVAR - Roadmap (open work)

Single source of truth for **open** tasks. Closed work lives in `VYVAR_JOURNAL.md`;
durable rationale in `VYVAR_DECISIONS.md`; current architecture in `VYVAR_STATE.md`.

Rebuilt **2026-08-31 CONSOLIDATE-01D D3**: stacked dated NEXT SESSION sections
collapsed to one OPEN table, one CLOSED-this-arc list, and RETIRED lines.
No task id was dropped. Historical prose of the stacked sections is in git:
`git log -- docs/VYVAR_ROADMAP.md` (parent of the CONSOLIDATE-01D D3 commit).

Cross-check: **EDGE-ANNULUS-01** is CLOSED-DECIDED in `VYVAR_DECISIONS.md`
(Milan 2026-08-31).

---

## OPEN

| id | one-line state | owner | blocked-on |
|----|-----------------|-------|------------|
| **A-1** | Frame selection metric I_j for MASTERSTAR stack ranking | Cursor | TODO-A |
| **A-1-DECISION-4** | Advanced r90 5.0-5.8 px target 5.31; not implemented | Cursor | schedule |
| **A-1-OVERRIDE** | Remove VY_FWHM_GAUSS as gaussian_fwhm_px_override; authorized in principle; own measured delta | Cursor | measured delta |
| **A-2** | Selection rule N_min=10 N_max=20 quality gate 0.5 x max(I_j) | Cursor | TODO-A |
| **A-3** | Median/sigma-clip stack replacing single-frame copy | Cursor | TODO-A |
| **A-4** | Stack provenance in header + pipeline_meta.json | Cursor | TODO-A |
| **A-5** | Recalibrate DAO threshold against stack noise/PSF | Cursor | TODO-A |
| **ANCHOR-CHAIN-ACCEPT** | Anchor chain accept leftover | Cursor | LOW |
| **ANCHOR-CLEAN-BUILD** | Anchor clean-build leftover | Cursor | LOW |
| **ANCHOR-ERR-VERIFY** | Anchor err verify leftover | Cursor | LOW |
| **ANCHOR-GATE-SEED** | Anchor gate seed leftover | Cursor | LOW |
| **BATCH-E-PARAMS-REGISTRY** | Batch-E params registry leftover if any | Cursor | LOW |
| **BIN-8-9-REGRESSION-01** | Bin 8/9 regression leftover | Cursor | LOW |
| **BPM-SIDECAR-PATH** | No *_dark_bpm.json found; path dead/disabled/outside tree unresolved | Cursor | forensics |
| **C-1** | Admission gate: predicted per-epoch SNR (g_lim + Labbe sigma_bkg_ap) | Cursor | TODO-C |
| **C-2** | CONTEXT-ONLY vs PHOTOMETRY-CANDIDATE catalogue flags | Cursor | TODO-C |
| **C-EXPORT-GAP** | Headless night_run omits AAVSO/VarAstro export | Cursor | schedule |
| **CAL-AGE-CLOCK** | Calibration master age clock | Cursor | LOW |
| **CAL-PASSTHRU-DEAD** | Passthrough calibration honesty; related F-B01-F-B02 | Cursor | F-B01-F-B02 |
| **COMP-POOL-R** | Comp pool R follow-up | Cursor | parked |
| **CORR-ERR-01** | ZP common-mode vs diagonal budget; out of v1.0 | Milan | research |
| **CR-1** | Same as CR-REJECTION (closure Step 9) | Cursor | TODO-A |
| **CR-REJECTION** | Cosmic-ray rejection (L.A.Cosmic or equivalent); no CR step in src_py today | Cursor | TODO-A |
| **D1-2-LINEARITY-RAMP** | Exposure ramp at telescope; nothing else substitutes | Milan | telescope night |
| **D1B-UNITS-01** | Decide preferred-unit defaults for the three px/unit pairs (qc_max_hfr[_fwhm_ratio], hrd_color_bg_box_px[_arcsec], masterstar_centre_rms_max_px[_arcsec]); behaviour change, must be measured per rig before flipping any default | Milan | per-rig measure |
| **D10-1** | D10-1 leftover from audit register | Cursor | LOW |
| **DAO-TOL-FLOOR-01** | DAO tolerance floor leftover | Cursor | LOW |
| **DB-DEFECT-DIAMETER** | DB defect diameter | Cursor | LOW |
| **DB-RETIRE-01** | Retire stale DB paths | Cursor | FUTURE |
| **DEPTH-AUTH-01** | Derive masterstar_gaia_census_target_depth_g from MASTERSTAR completeness vs Gaia; G=15.56 VSX absent | Cursor | not wired |
| **DRAFT451-CAL-FRAME001** | Draft 451 frame-001 calibrated product differs 659.6 ADU; root cause needs 451 cal logs | Cursor | logs |
| **EPSF-CORE-01** | Literature-parameter ePSF rebuild (multi-frame samples, osamp vs FWHM, smoothing) | Milan+Cursor | FUTURE |
| **EPSF-NEWTON-518-01** | Newton 518 ePSF STOP: gated pool 26 < 30 | Milan | night with pool>=30 |
| **EPSF-PERF-01** | Forced linear refit path; deferred by Milan | Milan | FUTURE |
| **EPSF-PIN-CENSUS-01** | ePSF pin census leftover / Newton 518 | Cursor | EPSF-ZP-OK-XRIG-01 |
| **EPSF-SHAPE-01** | Root narrow ePSF core OPEN (FWHM 2.36 vs 3.30); routed to EPSF-CORE-01 | Milan+Cursor | EPSF-CORE-01 |
| **EPSF-XVAL-01** | External ePSF gate: same ensemble/frames, independent PSF photometry reference; method unspecced | Milan | literature spec |
| **EPSF-ZP-OK-XRIG-01** | Extend fit_ok_for_zp past wide 1:1; needs master dark+flat + CENSUS-01; Newton 518 pool 26 does not qualify | Milan | CalibrationLibrary + night with gated pool >=30 |
| **EQUIP-BINNING-ASYM** | Equipment binning asymmetry | Cursor | LOW |
| **F-AIRMASS-CITE** | Airmass citation hygiene | Cursor | LOW |
| **F-B01-F-B02** | PASSTHROUGH runs may claim VYVAR calibration; PDF honesty | Cursor | calpath audit s14 |
| **F-BINGAIN-1** | Newton bin4 chi2 gate still open; do not flip ensemble to Broeg IVW until it passes | Cursor | Newton gate |
| **F-BJD-1** | BJD time-base follow-up | Cursor | LOW |
| **F-EXCEPT-TIER1** | Remaining tier-1 except hygiene | Cursor | LOW |
| **F-HOWELL-3** | Howell citation/path follow-up after F-BINGAIN-1 | Cursor | F-BINGAIN-1 |
| **FRAME-QC-PARITY** | Remaining: Layer A log honesty + n_stars outlier gate (frame 29, 263 vs ~100). Landed INV-FRAME-QC-01: `_dqf` None raises; provenance stamp | Cursor | not C8 |
| **GAIA-ID-FLOAT-GUARD** | Gaia id float guard follow-up if any residual | Cursor | LOW |
| **GAIA-PM-COLUMNS** | Gaia DB lacks pmra/pmdec; defer to DR4 ~Dec 2026 | Milan | DR4 |
| **HRD-PLOT-TUPLE** | HRD plot tuple hygiene | Cursor | LOW |
| **INPUT-PATH-ARCH-01** | Discussion: non-cal stays; raw-without-masters split | Milan | discussion |
| **INSTALL-GAIA-DEC-CUTOUT** | INSTALL should lead with declination cutout decision for Gaia builder | Cursor | docs |
| **INSTALL-MANUAL** | New-user install manual + T460 installer including catalogs | Milan | TODO-LIB |
| **K2-DATA-BLOCKER** | K2 data blocker | Milan | data |
| **K2-SLOPE-TRACE** | K2 slope trace | Cursor | K2-DATA-BLOCKER |
| **K2-SLOPE-UG** | K2 slope UG | Cursor | K2-DATA-BLOCKER |
| **MASTERSTAR-EPOCH** | MASTERSTAR epoch / PM | Cursor | GAIA-PM-COLUMNS |
| **MS-POOL-POLICY-01** | MASTERSTAR pool policy | Cursor | FUTURE |
| **MULTIFILTER-WCS-01** | Sibling-seed VERIFIED WCS for z_90_4; catalog-recovery gate unrelaxed; 520 measurement 2.7%/0% | Cursor | Milan GO |
| **NET-TEST-01** | Network/test harness item still listed open | Cursor | LOW |
| **NOQA-TRUNCATED-EXCEPT-BULK** | noqa truncated except bulk leftover | Cursor | LOW |
| **PHASE0-BORDER-MARGIN-GEOMETRY** | Phase 0 50 px margin is not EDGE r_out; not merged into EDGE-ANNULUS-01 | Cursor | not EDGE |
| **PRECAL-INPUT-CONTRACT-01** | Pre-cal input contract | Cursor | MED |
| **PROC-MAG-NAMING** | Proc mag naming | Cursor | LOW |
| **PROD-SIGMA-FLOOR** | Production sigma floor | Cursor | LOW |
| **PROV-HEADLESS** | Headless provenance | Cursor | LOW |
| **PROVENANCE-GUARD** | Provenance guard follow-up | Cursor | LOW |
| **PUB-FIGS** | Methods paper figures | Milan+Claude | PUBLICATION |
| **PUB-JOSS-PREREQS** | JOSS prerequisites | Milan+Claude | PUBLICATION |
| **PUB-OUTLINE** | Paper outline | Milan+Claude | PUBLICATION |
| **PUB-POLICY** | Publication policy | Milan+Claude | PUBLICATION |
| **PUB-VALIDATION-SECTION** | Paper validation section | Milan+Claude | PUBLICATION |
| **PUB-VENUE** | Venue choice | Milan | PUBLICATION |
| **QHY294MM-RN-DOUBLE** | DB RN 7.6 e- may be bin2 then scaled again to 15.2 e- | Cursor | low priority |
| **R-CVN-EMPTY-COMP** | Empty-comp drop reports no_comps; confirm nothing further | Cursor | POST-453 |
| **RELEASE-1** | Release-1 checklist | Milan | v1.0 |
| **RELEASE-2** | Release-2 checklist | Milan | v1.0 |
| **RN-HEADER-NONE** | Read-noise has no FITS header source | Cursor | LOW |
| **RUN-WORKER-01** | Run-worker follow-up | Cursor | LOW |
| **SIGMA-BKG-VAR-01** | Sigma background variance follow-up | Cursor | LOW |
| **SIGMA-BUDGET-EMPIRICAL** | Empirical sigma budget remaining Newton gate | Cursor | F-BINGAIN-1 |
| **SIGMA-PROV-FORENSIC** | Sigma provenance forensic leftover | Cursor | LOW |
| **SIGMA-SEM-CAUSE** | SEM cause leftover | Cursor | LOW |
| **SKY-SURFACE-BLAST-RADIUS** | Drafts 438-451 inflated catalogues; confirm no AAVSO/VarAstro export from those drafts | Milan | export check |
| **SPARSE-TRUST** | Sparse-field trust gate follow-up | Cursor | parked |
| **STALE-LC-SWEEP** | Stale LC sweep | Cursor | LOW |
| **SYNTH-SKY-GENERATOR** | WCS-true synthetic field generator for known-truth photometry | Claude | sub-pixel debug |
| **T4-1** | DECISION: detection noise on resampled frames (options A/B/C/D) | Milan | decision |
| **TASK-A-REGRESSION** | A2 CSV-write test never calls generate_masterstar_and_catalog | Cursor | test rewrite |
| **TIER1-OBSLOC-ZERO** | Observer location zero hygiene | Cursor | LOW |
| **TIER1-UI-DEBT** | Tier-1 UI debt | Cursor | LOW |
| **TODO-9** | Superseded/extended by INSTALL-MANUAL | Milan | INSTALL-MANUAL |
| **TODO-A** | Median/sigma-clip MASTERSTAR stack of best N frames ranked by I_j; provenance; DAO recailbration | Cursor | audit Steps 1-6 |
| **TODO-B** | Zackay & Ofek proper coaddition; blocked on CR, uncorrelated inputs, per-frame PSF | Cursor | CR-REJECTION |
| **TODO-BROAD-EXCEPT-HYGIENE** | Broad-except tier-1 leftover (~25) | Cursor | LOW |
| **TODO-C** | Admission gate vs detection threshold; CONTEXT-ONLY vs PHOTOMETRY-CANDIDATE flags | Cursor | audit Steps 7-8 |
| **TODO-GEO** | Backlog geography/site item | Milan | parked |
| **TODO-GS8** | Multi-night global matching + global ZP; descoped from HIGH; canonical unit is one night | Milan | FUTURE science case |
| **TODO-LIB** | Cython closed-source bundle; ties to installer | Milan | CYTHON-RELEASE follow-up |
| **TODO-MULTISET** | Per-telescope-set config architecture (wide vs Newton) | Milan+Cursor | design |
| **TODO-PSF-ASYMMETRY** | Tracking-smear diagnostics (BO CVn right-tail PSF) | Cursor | FUTURE |
| **TODO-PSF-MULTIFRAME** | Multi-frame ePSF stacking (isolation part done) | Cursor | FUTURE |
| **TODO-PSF-NEIGHBOR-SUB** | Neighbour subtract + aperture residual; 2b deferred until blended fine-scale field | Cursor | blended fine-scale draft |
| **TODO-SCENE-FORWARD-MODEL** | Conditional on crowded-faint science; priority lowered after grouper-negative | Milan | FUTURE |
| **TODO-SEP-XVAL** | SEP independent witness; aperture xval CLOSED | Cursor | parked |
| **V1-VALIDATION-PROTOCOL** | Enrich validation packs: ePSF identity, DAOPHOT ref, PDF QA, E2E mini field | Milan+Claude | protocol |
| **WIDE-ERR-CROSSRIG** | Per-rig when Newton/Boyden drafts exist | Cursor | other-rig drafts |
| **WIDE-ERR-HONEYCUTT-PDF** | Honeycutt SEM PDF honesty | Cursor | LOW |
| **WIDE-ERR-POP-DELTA** | Wide-err population delta | Cursor | LOW |
| **WIDE-SLOPE-NOISE** | Wide slope vs noise | Cursor | LOW |

Standing operator items without a hyphenated id (kept as prose, not an id row):
first AAVSO/VarAstro uploads BO -> FW (band CV) once a locked ledger exists;
`origin/main` is `5b1068d` (MERGE-MAIN-01; SEL-GHOST-01 MERGED 2026-09-01).

---

## CLOSED this arc

Closed, locked, or superseded during the 2026-06..2026-08 stacked-session era
(APERTURE / SEL-GHOST / ePSF / ERA-04 / audit closure). One line each.

- **A-1-435-RECUT** -- CLOSED 2026-08-18; 435 retired by ROT policy; recut onto 516.
- **A-6** -- DONE 2026-08-07 DAO detection workstream closed.
- **APCORR-MIXEDFRAME** -- DONE 2026-07-19 all-or-nothing COG per night.
- **APERTURE-01** -- Wired option i; later locked as APERTURE-01d.
- **APERTURE-01b** -- STOP 2026-08-26; no f* on accuracy grid.
- **APERTURE-01c** -- STOP 2026-08-26; AIJ PASS 2.7833 mmag; era04 not yet locked.
- **APERTURE-01d** -- LOCK 2026-08-27; annulus 2.7/5.2; AIJ 1.9503 mmag; era04 --full gate.
- **ARCHIVE-CLEANUP** -- NEXT SESSION 2026-07-15; historical.
- **CAL-DIAG** -- CLOSED 2026-08-13; SUPERSEDED heading removed 2026-08-11 then implemented.
- **CATALOG-PROVENANCE** -- DONE 2026-07-29.
- **COMP-RMS-DEF-01** -- Wired C3 2026-08-25 (k=5 LOO mag).
- **COMP-RMS-DEF-01-B** -- Wired C3 2026-08-25.
- **CONFIG-MATERIALIZE-CHECK** -- DONE 2026-07-24 BUNDLE-BOOTSTRAP-WIRING.
- **CONFIG-PREREZ** -- DONE 2026-09-01 CONSOLIDATE-01D P2 except remaining OPEN **D1B-UNITS-01**.
- **CYTHON-RELEASE** -- DONE closed-source bundle preview 2026-07-23.
- **DAO-THRESHOLD-PARAMS** -- CLOSED 2026-08-07; reopen only on two-rig empirical sweep.
- **DEV-PROCESS-A** -- DONE 2026-07-08 validation ledger.
- **DEV-PROCESS-B** -- DONE 2026-07-08 session_baseline_check.py --full.
- **DOCS-SYNC-517** -- Superseded NEXT SESSION 2026-08-21.
- **EDGE-ANNULUS-01** -- CLOSED-DECIDED Milan 2026-08-31: edge stars not used; full on-chip aperture+annulus.
- **EPSF-AC-01** -- Closed in ePSF AC measurement arc 2026-08-24.
- **EPSF-AC-02** -- Closed/wired in ePSF AC arc; Newton ZP-OK still open as EPSF-ZP-OK-XRIG-01.
- **EPSF-VALID-02** -- CLOSED 2026-08-22 gated 67-star production ePSF on 516.
- **ERA-03** -- era03 freeze kept on disk; superseded as --full gate by era04.
- **EXCEPT-BULK** -- CLOSED 2026-07-08 silent broad-except census.
- **EXPORT-PARITY-01** -- CLOSED this arc (v2, d6c84e0: one production entry, G7 --parity permanent).
- **F-428** -- CLOSED 2026-07-15 draft_428 forensics.
- **F-429** -- CLOSED 2026-07-16 validate + regressions.
- **F-431-HEADLESS-DIVERGENCE** -- CLOSED 2026-07-16 / T3 (DECISIONS).
- **FRAME-QC-PARITY-01** -- Phase 1 heading superseded 2026-08-21; phase 2 remains FRAME-QC-PARITY.
- **FULL-ANCHOR-RECUT** -- CLOSED 2026-08-27 ERA-04 lock.
- **INV-CAL-01** -- CLOSED 2026-08-13 CAL-DIAG v2.
- **INV-CAL-02** -- DONE 2026-08-13 calibrated product stage integrity.
- **P1-RECUT** -- CLOSED 2026-08-20 ERA-03 golden mini.
- **REG-520-01** -- STOP 2026-08-24 measure; ghost/WCS notes carried in SEL-GHOST-01.
- **SAT-DIAG** -- DONE saturation and linearity limit gate.
- **SEL-GHOST-01** -- MERGED 2026-09-01; origin/main `5b1068d` (fast-forward from 7c086e8 via consolidate-01).
- **TODO-COMP-P2P-RESIDUAL** -- DONE already implemented; found stale 2026-07-19.
- **TODO-DEV-PROCESS** -- DONE 2026-07-08 as DEV-PROCESS-A + DEV-PROCESS-B.
- **TODO-EPSF-1-FWHM-QC** -- DONE 2026-06-08.
- **TODO-FWHM-CONSISTENCY** -- DONE 2026-06-09.
- **VYVAR-INVARIANTS** -- P1/P2 DONE 2026-07-19; remaining phases in git history.
- **WIDE-ERR** -- CLOSED WIDE-ERR-04 physical model g_pt + weighted SEM.
- **XVAL-AIJ-02** -- DONE production 4-comp + two frame states.
- **ZONE-SAT-01** -- Wired with COMP-RMS-DEF-01-B.

---

## RETIRED

Dropped, retracted, status-token, or explicitly superseded-do-not-reopen.

- **CLOSE-OUT** -- Not a task id; heading token from stacked NEXT SESSION titles.
- **CLOSED-DECIDED** -- Not a task id; status token from EDGE-ANNULUS-01 row.
- **MASTERSTAR-EPSF-ALL** -- Dropped 2026-06-02; plate scale is WCS-derived.
- **SESSION-CLOSE** -- Not a task id; heading token from stacked NEXT SESSION titles.
- **TODO-RECUT-HARNESS-FIDELITY** -- CLOSED superseded 2026-07-08; draft_387 zaloha gone.
- **U-XVAL-COMP-RMS** -- RETRACTED (audit register).

---

## Parked (not blocking; ids already in OPEN if they had one)

- CM-detrend differential (~10x lever; opt-in; needs transit injection-recovery).
- Newton-V colour-term (per-rig c1 from field BP-RP).
- Meridian-flip handling (Qatar-8 class).
- Pre-filled camera catalog for new-user onboarding (PARKED; design notes in git history).
- Magnitude-aware check-star threshold for the trust gate.
- Reserved check-star (hold-one-out; moves photometry anchor).
- AAVSO-standard output #4 G->B/V/Rc.
- Blind index 3rd rig tier (Noctutec 206/560) when a validated draft exists.
- Comet photometry mode (after variable-star pipeline; analysis only).
- TODO-GS7 paper draft (see PUBLICATION ids in OPEN).

