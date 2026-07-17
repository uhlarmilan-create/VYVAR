# VYVAR -- Development State

Last updated: **2026-07-17** (REPO-REORG + PARAM-SOURCE-AUDIT + wave-A ownership; layout stabilized).

## Repository layout (REPO-REORG -- DONE, gate PASS)

Root tidied into a stable layout: production code in `src_py/`, dev material in
`dev/` (`dev/tests|tools|validation|scripts|sandbox|orchestrator`), all Cursor
result/task docs in `dev/results/`, scratch in `tmp/` (gitignored). Root `app.py`
is a thin Streamlit shim. Commits `c611353` (dev/ move) + `8f4d7b4` (src_py/ move).
Anchor #3 `--full` gate **PASS byte-identical** on `8f4d7b4` (core `3d26f469...`
n=333; extended `6420f1da...` n=499; git_dirty_code=false). Milan UI smoke
confirmed (app launched via root shim, Settings + Parameters tab render,
modified-counter=10) plus e2e draft_000436 anchor run. See CLAUDE.md /
VYVAR_PROCESS.md for the map; result `CURSOR_RESULT_repo_reorg.md`.

Parameter provenance mapped in PARAM-SOURCE-AUDIT (`dev/results/PARAM_SOURCE_AUDIT.md`
+ `param_source_audit.csv`, 304 keys): one config.json + DB reference tables +
FITS-resolved values; DB `SETTINGS` table found vestigial. Wave-A ownership arc
adds the `owner` axis and fixes the config-write render side-effect.

## Test-hygiene backlog

- `dev/tests/test_g7_f003c_report_cfg_snapshot.py` (2 tests) is order-dependent: its
  `_factory` monkeypatch self-recurses through `config.AppConfig` when the qc-metrics
  path constructs `AppConfig()`. Passes in the full suite, fails in isolation. Confirmed
  pre-existing (reproduces on committed code with no wave-A edits). Repro:
  `python -m pytest dev/tests/test_g7_f003c_report_cfg_snapshot.py -q`. Fix later:
  make `_factory` bind the real `AppConfig` (e.g. capture it before patching) instead of
  re-importing the patched name.

## UI block

**Wave 1 (parameters): DONE.** Machine-readable registry (304 entries) with parity /
freshness / hidden-tier guard tests, generated `VYVAR_PARAMS.md`, tiered Parameters
dashboard (basic 12 / advanced 75 / expert 217), PDF Configuration page in the SUMMARY
MEASURE REPORT. min_comps=3 closed as intentional (DECISIONS). Results:
`CURSOR_RESULT_params_registry_ui.md`, `CURSOR_RESULT_params_closeout.md`.

**Next: data dashboards wave** - identity QA series, trust/census fingerprint, AC status,
excluded_targets, VSX stamp, sky-surface stats, export skips. Spec by Claude next session.

## Data items (top)

1. **Fresh darks** — expiry ~**2026-07-21** (do first when scheduling next night).
2. **Anchor offline zips** (`C:\ASTRO\backups\`):
   - `draft_000435_snapshot_skysurface_20260716.zip` SHA256
     `a35d22354666e359ce1bdd9a6eb207d5d768466a67fcdb77c22425eabb3f84a0` (~4.82 GB)
   - `draft_000435_anchor_live_20260716.zip` SHA256
     `a4bb42d255e542b4a516197d5efe1a6304602b331680ac554caf41a244070faf` (~4.82 GB)
   - Historical: `vyvar_anchor_424_sigma_floor_20260713_core-bf3743a1.zip` (untouched)
3. **Archive freed** ~**48.7 GB** (drafts 428–434 + pass1 photometry backup). In-Archive
   remain: `draft_000435` + `draft_000435_snapshot_skysurface_20260716`.
4. F-428 arc evidence: `validation/f428_arc_evidence/` (moved from `tmp/`).

## Current snapshot (Anchor #3 ACTIVE)

**draft_000435_snapshot_skysurface_20260716** — Anchor #3. Dual full-photometry SHA gate
**PASS** on HEAD `10d610c` (LABBE-DET). Core `3d26f469…` n=333; extended `6420f1da…` n=499.
Census 2552/2842/169; identity p95 baseline **1.54 px**. Ledger `VL-ANCHOR-WCSINV` ACTIVE;
`VL-ANCHOR-424` superseded (offline zip historical). `--full` re-enabled against 435.

**Root cause of prior STOP:** ensemble SEM join / dict-order nondeterminism in phase2a `err`
(not Labbe placements). Labbe hardened anyway (canonical stars + SeedSequence + dump).

**INVARIANTS P1:** unblocked (seed started).

---

## Prior: Anchor #3 STOP (superseded)

**draft_435 HEALTHY** but protocol-v2 SHA gate FAILED on `err` only (166/166 LCs). See
`CURSOR_RESULT_anchor435_closeout.md`. Fixed in LABBE-DET (`CURSOR_RESULT_labbe_det.md`).

---

## Prior: T3-RESTORE — FIX A+B committed; next gates

**T3 FIX A+B** on `89842ff` (+ docs `3db0879`). Sky-surface headers + `qc_metrics.csv` confirmed
on draft_435.

**T3 FIX A (done):** order-2 sky-surface subtract in shared ``preprocess_calibrated_to_processed``
(``preprocess_sky_surface_order`` default **2**). Empirical vs draft_429 Light_008: DAO pass-1
sim **2579** (2500–3000 band), smooth residual p99 **173 ADU**; ~237 below 429 logged 2816
(cal-only mask/clip gap documented).

**T3 FIX B (done):** ``git_dirty_code`` / scratch vs import-relevant root ``*.py`` classifier;
anchor / FAIL-CLOSED gates trip on ``git_dirty_code`` only.

**T2 verdict (434 baseline):** draft_434 on ``1e2e8d6`` = **UI-SICK** (6699-class). Milan re-run
after push expected: cal≠proc, ~2875 matched, sky ~1478 meta, identity p95 < 1 px,
``git_dirty_code=false``.

**T4 (prior):** VSX stamp after VT write; Labbe content-hash seed; NightRun match sep 2.0.

**429:** unprovenanced quality target; deterministic pre-T3 path = 6699-class.

**Next:** Milan UI RUN VYVAR → HEALTHY? → anchor #3 protocol-v2.

## Prior snapshot (F-431 root closure — partial)

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
(1) **TODO-12 HRD arc** 12/12b/12c/12d/12e/12f — session-aware extreme-object table, enrichment,
identification tiers, PDF/UI details, summary.json freshness stamps (`generated_at_utc`, `git_head`).
(2) **F-BINGAIN-1 RESOLVED** — empirical empty-aperture `sigma_bkg_ap` in production `err` (IRAF/SExtractor/
photutils-aligned); hybrid `howell_scaled` fallback for crowded fields; regate PASS (decomposition-driven
gates). Result files: `CURSOR_RESULT_todo12_hrd.md` … `_todo12f_hrd.md`, `CURSOR_RESULT_bingain_fix.md`,
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
stars reach enrichment (teff mostly NaN, none >=25k — acceptable for reddened OB). pytest **696 passed**.

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

**2026-07-09 snapshot:** SESSION-CLOSE-0709: draft_426 FITS headers (INSTRUME=C5A-150M, 3552×2664 @
bin4) verify OBS_DRAFT eq4 — GAIN=12.48 anomaly is F-BINGAIN-1 not wrong equipment;
`scripts/fix_draft_equipment.py` + tests. Sigma Phase A wide-rig DONE (6.5 mmag floor, k2 attribution
zero, ~4.5 mmag rig constant). Sparse-comp: ~95% field-wide offset cancels; temporal 8–12 mmag healthy.
pytest **681 passed**, 15 skipped. Pushed to origin/main (Milan-authorized 2026-07-09).

**2026-07-09 snapshot (prior):** SIGMA-A4: wide-rig floor attribution (k2 pooled R²≈0, floor_after k2=6.5 mmag
unchanged); Newton bin4 forensics (header gain 12.48, σ_ratio≈1.13, χ²_pred≈0.78); hypothesis
gain/RN correction moves χ² away from 1. pytest **679 passed**, 15 skipped.

**2026-07-09 snapshot (prior):** SIGMA-A3: variant (e) `howell_scint_fresid_floor_ensemble` (+ Honeycutt ensemble SEM);
dual SEM paths (LC decomposition + production `ensemble_normalize`); draft_424 joint refit (d) unchanged
f_resid=0.74 sigma_floor=10.5 mmag, joint (e) f_resid=0.0 sigma_floor=6.5 mmag — prediction
**floor_did_not_collapse**. pytest **674 passed**, 15 skipped.

**2026-07-09 snapshot (prior):** SIGMA-A2: rig fixes (TELESCOPE.DIAMETER 72→200 mm, alt<=0 guard),
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
(not-worth-complexity); PUB-QC-MISSRESIDUAL parked. Completeness 89.7–98.3% across rigs;
miss@G90 health signals live in QA dashboard. pytest **661 passed**, 15 skipped. Chain pushed
to origin/main (Milan-authorized 2026-07-09).

**2026-07-09 snapshot (prior):** DAO-RECONCILE-2b (`78febea`): right-censor G_lim vs reference depth;
`missed_below_g90` / `missed_fadezone` 2-pass metric; `match_depth` forensics; missed-G
histogram in diag. draft_424: compl_50=89.7%, missed=353 but **miss@G90=15** (fadezone=338).
draft_425 B/R: G_lim_50 censored at 17.5 (was spurious 19.25). All-drafts diag:
`tmp/dao_reconcile/cross_draft_summary.json`. pytest **659 passed**, 15 skipped. Chain pushed
to origin/main (Milan-authorized 2026-07-09).

**2026-07-09 snapshot (prior):** DAO-RECONCILE-2 (`bd6244a`…`b7df7c6`): footprint Gaia reference +
Fleming (1995) completeness curve; `completeness_50` headline. draft_424 R-2: G_lim_50≈14.97,
completeness_50≈89.7%, genuinely-missed=353 (R-1: 96k @ 3.4% — population bug). All-drafts
diag: `tmp/dao_reconcile/cross_draft_summary.json`. pytest **653 passed**, 15 skipped. Anchor
unchanged (`92939fab` / `76642318`). Unpushed local chain.

**2026-07-08 snapshot:** draft_424 coherent anchor `draft_000424_snapshot_20260708_full`
(`run_full_photometry_pipeline`; core SHA `92939fab…` n=357). Hybrid snapshot retired.
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
`merge_photometry_pipeline_meta` when `cfg` passed — `git_hash`, `git_dirty`, full
`AppConfig.to_dict()` snapshot, `stamped_at_utc`, `entry_point`; last-writer-wins at
`run_phase2a` and `generate_masterstar_and_catalog`. Archaeology: never-wired (not regression).
No secrets in `AppConfig` (post-`c26e351` credentials reset).

**Done 2026-07-07 (K2 v1):** Literature k'' path via `k2_extinction.py`; `band_classify` wired to
`resolve_apply_color_term` (CV/CR flip live); LC columns `k2_source`/`k2_value`/`k2_colour_ref`;
NIGHT_FIT deferred (`k2_fit_enabled` OFF). Spec: `docs/VYVAR_K2_DESIGN_SPEC.md`. Validation:
424/425/427 matrix **PASS** (`tmp/k2_land/validation_report.json`, 2026-07-07).

**Done 2026-07-07 (CAL-DIAG-IMPL):** Calibration-time radiometry gate per `VYVAR_CAL_DIAG_SPEC.md`
v1.1 — Check A (SUM/MEAN convention) + Check B (post-dark sky sanity); parent pre-gate for MP
variant (a); provenance headers `VY_DKRSMP`/`VY_CDSKY`/`VY_CDSTAT` + `archive/<draft>/cal_diag.json`.
14 gate unit tests; `549 passed` full suite; draft_424: **150/150** frames `VY_DKRSMP=SUM`, 0
WARN/FAIL, calibrated arrays and photometry science byte-identical to baseline. **RN-HEADER-NONE**
and **CAL-PASSTHRU-DEAD** closed 2026-07-08 (`1830527`, `21c20e3`); **CAL-AGE-CLOCK** closed 2026-07-08 (header-age unified). CAL-AGE-CLOCK was the last open CAL-DIAG ledger item.

**Done 2026-07-07 (session close):** F-BINGAIN-1 Stage A -> **LATENT** (not live on wide rig);
CAL-DIAG workstream registered (calibration radiometry gate; spec pending). F-AIRMASS-CITE fixed;
GAIA-ID guard closed. Commits: `4f18f02` (Fable B+C), `d594b27` (session close).

**Done 2026-07-07 (Fable audit B+C):** Kasten & Young (1989) airmass attribution corrected;
`kastenyoung1989` in `CITATIONS.bib`. GAIA-ID-FLOAT-GUARD closed (live-tree parity check).
F-BINGAIN-1 Stage A diagnostic only — no exponent change yet.

**Done 2026-06-25 (F-BJD-1 Stage D):** per-target LC column `time_base` labels BJD recompute path;
`_recompute_bjd_hjd_with_status` reports cause (`BJD_TDB` vs `JD_FALLBACK`). Purely additive —
`bjd`/`hjd`/`jd` byte-identical. Closes the 2026-06-25 citation/error-model audit.

**Done 2026-06-25 (F-HOWELL-3 Stage C):** explicit annulus-sky column for Howell err; `_photometric_error`
reads `sky_adu_per_px_annulus` with legacy `noise_floor_adu` fallback. Verified on real draft_424
(`run_full_photometry_pipeline`): 178/178 LCs science-identical; sky-dominated err inflation measured
**~12–14%** (detection vs annulus) on faint targets.

**Done 2026-06-25 (citation audit Stage A):** F-RIELLO-1 — B-V/Riello report citation removed
(BP-RP is raw Gaia); F-HOWELL-1 units comment; F-CITE-HONEYCUTT Honeycutt in CORE (`5a1bae0`).

**In-flight / gated:** NIGHT_FIT k'' pre-gate (v2; K2-DATA-BLOCKER). See ROADMAP ACTIVATED v1.

**Done this session (prior):** band classifier (`fe9b375`) — now wired with k'' v1.

**In-flight / PARKED — band-aware k'' (second-order extinction):** **ACTIVATED v1** — see STATE
2026-07-07 K2 paragraph and `VYVAR_K2_DESIGN_SPEC.md`. NIGHT_FIT = v2.

Prior: **2026-06-22** — Forced-aperture / catalog_only removed; DAO+Gaia photometry only. Variable
targets measured **only on direct DAO `catalog_id` hit** (miss → nondetection/NaN; no XY fallback).
Unmatched VSX excluded in Fáza 0. Validated do-no-harm vs draft 419. See DECISIONS.

Prior: **2026-06-19** — Stage B held pending validation (forced-aperture removal draft).

Prior: **2026-06-18** — **Fix C / Phase C1: dense-field alignment DIAGNOSED → root = PSF/FWHM
bloat; recovery NOT APPLICABLE.** The 14 late-night (post-flip, back-half) frames Fix B drops are **not**
"good data that only failed alignment" — they are **PSF-degraded**: median **FWHM 8.60 px = 1.85× the
good baseline 4.64 px**, concentration flux_large/flux **13.1 vs 1.65**, **corr(FWHM,
alignment-residual)=0.95** (161 frames; `tmp/phaseC1/fixC_root_cause.png`). The bloated-donut centroid
noise (~2.4 px) is the single root — it breaks astroalign (misalignment is the *symptom*) and is what
B.2 (concentration) + Fix-B (residual) measure. Likely **late-night focus drift on the defocused rig**
(a transparency/flux drop alone would not bloat FWHM); post-flip-half-not-refocused is an observer
question. **Not recoverable to sub-px** (centroid floor ~2.4 px > 1.37 px gate; cap50→3/14, WCS absent
0/162, translation-refine inapplicable). **Fix B + B.2 are the correct PERMANENT quality gate** — not a
stop-gap awaiting Fix C. Logged a SEPARATE control-point-cap perf ticket (astroalign mcp≈200 → ~654
s/frame on dense fields; cap ~50 → ~3–10 s; ROADMAP). **A.B.: Fix A `005716d` + Fix B `fa03410` pushed
to origin/main this session (Milan-authorized).** `CURSOR_RESULT_fixC_diag.md`. See DECISIONS/JOURNAL.
Prior: 2026-06-18 — **Fix B: reject-on-alignment-residual frame gate** (default-OFF;
`frame_align_residual_gate_enabled`). Two additive pieces: (1) **always-on QC** — a per-frame
**alignment residual** (median deviation of bright matched sources from their across-night median
position) is computed at the Phase-2A frame-selection point and recorded as `align_residual_px` in
`alignment_report.csv` (additive metadata → photometry byte-identical); it reproduces the run-414
diagnostic separation (astroalign med **0.358**/max **1.648** px vs phase_corr min **1.450**/med
**2.130** px). (2) **gate (default-OFF)** — rejects frames whose residual exceeds
`frame_align_residual_max_frac × science-aperture-radius-px` (**rig-agnostic** fraction, default
**0.25** → 1.37 px, in the 1.206→1.450 px good/bad gap; safety floor `min_keep_frames`). Verified on
run-414 g: **OFF byte-identical** (70 targets, V0454 `mag_calib`/`delta_mag`/`err` max|diff|=0); **ON
drops 14 frames = all 13 phase_correlation + 1 mis-aligned astroalign** (dr=1.648, itself an LC
outlier) — V0454 outliers 22→10, the catastrophic +3.7 mag/NaN points gone (clean SIPS-grade egress;
`tmp/fixB_v0454.png`). **B.2 cross-check:** residual gate ⊇ B.2 (overlap 13, residual-only the 1
astroalign, B.2-only 0) — cause-correct (alignment) superset of B.2's aperture-integrity symptom; both
kept distinct. **[C1 correction: PERMANENT gate, not "self-deactivating once Fix C fixes alignment" —
the frames are PSF/FWHM-bloated and unrecoverable.]** See DECISIONS/JOURNAL.
Prior: 2026-06-18 — **Fix A: per-point error model bug fixed** (default; no flag). The LC
`err` term-3 was `np.std(comp instrumental mags)/√n` (`photometry_core.py:2567`) — for a sparse/
brightness-spread ensemble this is the comps' brightness *spread* (a fixed ~0.58 mag floor on V0454,
23× the empirical 0.025), not a per-point uncertainty. Replaced with the per-frame **ensemble-ZP
standard error from comp residuals** (each comp vs its own across-night median → brightness/colour
cancels; Honeycutt 1992); the redundant `comp_rms/√n` term-2 was dropped (no double-count); photon
term-1 (incl. SNR-blowup on bad frames) kept. Verified on run-414 g: centres `mag_calib`/`delta_mag`
**byte-identical**, V0454 err 0.581→0.013 (≈empirical), faint targets photon-dominated, the 13
mis-aligned frames still flagged (Fix B). `err` does NOT feed trust/lc_rms/production-Broeg-combine;
it does feed SysRem IVW weights (default-OFF) — improved, not broken. See DECISIONS/JOURNAL.
Prior: 2026-06-17 (end-of-day) — clean committed **+ pushed** baseline at `955b850`
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
| `docs/VYVAR_RUNBOOK.md` | Chi_and_H zaloha-only night-run procedure (alias → baseline runbook). |
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
| Skip processed/ | `skip_processed_directory` | OFF |

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

### Brno AZ800 / C5A-150M (production solver — 2026-06-14)

**Brno AZ800 / C5A-150M onboarded.** Production solver uses **catalog-recovery verification**
(Gaia-in-frame / DAO at 2.5 px) as the MASTERSTAR accept gate; detection match% is informational.
Stale FITS pointing (`VY_TARG`) → **`hint_sep_warn`** when VERIFIED (Lang et al. 2010 prior), not
hard reject. Cone recenter at solved center when hint offset **≥ 0.05°** unchanged.
`generate_masterstar_and_catalog` passes `app_config` + scoped flags.

**Brno `r_60_4`:** catalog recovery tight **~84%** → **VERIFIED** under new gate (was rejected on
`hint_sep` + detection-denominated metrics). **`z_90_4`:** recovery **~34%** → stays rejected.
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

1. **Source data (must retain):** `Archive/Chi_and_H` — pre-calibrated FITS (only non-regenerable input).
2. **Catalog + blind index:** `GAIA_DR3/zaloha/vyvar_gaia_dr3.db` (G<=16), zaloha blind PKLs
   (`gaia_triangles_fine.pkl`, `gaia_triangles_wide.pkl`). **Do not read** in-progress
   `GAIA_DR3/vyvar_gaia_dr3.db`.
3. **Run:** `python scripts/chiandh_night_run_bvr.py` (#3 code; Newton bin2 ~1.30"/px).
4. **Verify:** `compute_photometry_sha(draft_root)` core + full vs recorded SHAs (`3f7c9e7a…` /
   `d5b72d08…`). For regression vs the historical cut, use
   `compare_photometry_science_meaningful` (PROCESS) — excludes provenance/`err` QC drift.

**Setups (filter-wheel labels):** `B_20_2`, `V_20_2`, `R_20_2`, `L_20_2` — **B/V/R/L** are wheel
positions. **V** = visual/green (`G/` folder); **L** = clear/broadband (`L_20_2` in anchor).

**Provenance at anchor cut (2026-06-11, zaloha):**

| Item | Value |
|------|-------|
| `git_commit` | `7317ece87944b749461a7b6abca6615f1a30dc72` (re-baseline lock) |
| Catalog | `GAIA_DR3/zaloha/vyvar_gaia_dr3.db` (G<=16) + zaloha blind PKLs |
| Rig | Newton 300/1200 + C3-26000, bin2 ~1.30"/px |
| Ephemeral draft | `draft_000386` (~1401 LCs; deletable) |
| Completeness gate | `night_run.audit_photometry_completeness` — >=90% summary/active per setup |

Scoped trust at cut (`comp_trust_min_comps=5`, floor-5 baseline): **1382 YELLOW / 106 RED**
(1488 summary rows; re-trust on draft_387). Pre-floor-5 counts were 1400/88 — superseded.
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
- **Tests:** **852 passed / 15 skipped** (last full `tests/` run incl. `-m slow`; 2026-07-14 close).
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
