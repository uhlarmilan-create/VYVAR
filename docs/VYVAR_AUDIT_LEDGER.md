# VYVAR — Audit ledger

**Started:** 2026-06-08 · **Workflow:** tooling-first triage → targeted fixes → manual critical-path read

Stav modulov a disposícia nálezov. Obnoviteľný register — nie čierna diera.

**Stavy:** `todo` · `auditing` · `done` · `deferred`

| Modul | Stav | Nálezy (Fáza 0→1) | Disposícia |
|---|---|---|---|
| `photometry_report.py` | done | F821×27 → 0 (TYPE_CHECKING); F841×11 → 1; F811 cm/mm | fix batch 1 — kozmetika + dead locals |
| `photometry_core.py` | done | batch 3: removed `ra_ms`/`de_ms`, `gaia_teff` dead reads | F841 clear |
| `comp_selection_per_target.py` | done | batch 3: removed `g_teff` dead reads | F841 clear |
| `comp_pool_rms.py` | done | F841 `avail_cols` | removed dead assignment |
| `comp_qa.py` | done | batch 3: removed `lc_map` | dead local, no logic loss |
| `comp_qa_core.py` | done | Phase F CQ-A..E (see AUDIT_FINDINGS) | CQ-A verify pre-cal then close ROADMAP HIGH |
| `trust_flag_core.py` | done | Phase E: Findings A+B implemented; C1 ddof=0; D guard doc; E deferred | 9 unit tests |
| `calibration.py` | done | Phase F CAL-A..D (see AUDIT_FINDINGS) | clean; CAL-A caller logging verify |
| `database.py` | done | Phase F DB-A..D (see AUDIT_FINDINGS) | mostly sound; DB-B threading verify |
| `vyvar_platesolver.py` | done | Phase F PS-A..C + batch 3 `center` | PS-B = Phase G priority |
| `pipeline.py` | done | batch 3: removed `n0`×4, `cfg` in `extract_fits_metadata` | F841 clear |
| `psf_photometry.py` | done | batch 3: `fit_shape`; **EPSF-1** (2026-06-08) FWHM QC estimator bias | EPSF-1 diagnostic only; ROADMAP TODO-EPSF-1-FWHM-QC + harness V3e |
| `psf_neighbor_sub.py` | done | step 2/2a validation core + guards (gated OFF) | A9 scored; 2b blocked SAFE_LOW_YIELD |
| `tests/validation` (A9) | done | blend grid envelope + mismatch diagnostic | `a9_core.py`; tier_a9 reports gitignored |
| `export_reports.py` | done | F401×2 | ruff --fix batch 1 |
| `config.py` | done | — | no F841 in scope |
| `ui_*` (aggregate) | done | Phase F: `n_rms_candidates` wired to m2 | UI-only; reverses prior help-text |
| `tess_verify.py` | done | batch 3: removed `center_col_tpf`/`center_row_tpf` | TESS side path |
| `xval_run.py` | done | batch 3: removed `sf`, `PS` | offline harness |
| `psf_runner.py` | done | batch 3: removed `mn_cid` | dev CLI |
| `orchestrator/` | done | F401/F541 | ruff --fix batch 1 |

## Dávka 1 (2026-06-08) — hotovo

| Check | Výsledok |
|---|---|
| F821 | **0** (bolo 27) |
| F811 | **0** (bolo 7) |
| F401/F541 | **0** (auto-fix + review) |
| F841 | **22** (bolo 44) |
| `pytest tests` | **174 passed, 6 skipped** |
| Byte-identita fotometrie | neoverená v tejto dávke (len refaktor mŕtvych premenných / importy) |

## Dávka 2 (2026-06-08) — session close

F841 batch 2 cleanup: removed `dist_score`, `rms_f2`, redundant `c1_stderr@7141`;
`lc_df@7786` read-guard preserved (`pd.read_csv` without binding); `g_teff`/`gaia_teff`
deferred (benign).

| Check | Výsledok |
|---|---|
| `pytest tests` | **174 passed, 6 skipped** |
| Photometry byte-identity (`draft_000366`, 284 artifacts) | **OK** — SHA-256 `ad12325d262e913dc57fa0e805e07c2115aec5005268c704177d7fb72856aa69` unchanged |
| PDF overflow (`draft_000366`) | **0** violations (160 pages) |
| commit | `10b81fa` |

## Dávka 3 (2026-06-08) — F841 batch 3 + audit findings re-encode

F841 production scope 18 -> 1 pending (`ui_variability.py:1480` Cat 3, awaiting Milan).
Removed 17 dead locals / dead read blocks (Cat 1 + Cat 2). Phase A1: re-encoded
`VYVAR_AUDIT_FINDINGS.md` to UTF-8 ASCII; `tmp/_gen_audit_findings.py` emits ASCII.

| Check | Výsledok |
|---|---|
| F841 production | **1** (`ui_variability.py:1480` pending) |
| `pytest tests` | **174 passed, 6 skipped** |
| Photometry byte-identity (`draft_000366`, 284 artifacts) | **OK** — SHA-256 `ad12325d262e913dc57fa0e805e07c2115aec5005268c704177d7fb72856aa69` unchanged |
| PDF overflow (`draft_000366`) | **0** violations (160 pages) |
| commit | pending (git identity not configured locally; includes Phase B + C staged) |

## Phase C (2026-06-08) -- ruff safe auto-fix (production scope)

SIM114 17 / RUF010 8 / B009 4 / B010 4 / SIM300 3 / SIM910 1 = **37 fixed** across 16 modules.
Parenthesized dense SIM114 and/or merges in `lunar_context`, `photometry_core`, `pipeline`,
`catalog_crossmatch`, `importer` (cosmetic readability only).

| Check | Vysledok |
|---|---|
| SIM114/RUF010/B009/B010/SIM300/SIM910 production | **0** remaining |
| `pytest tests` | **174 passed, 6 skipped** |
| Photometry byte-identity (`draft_000366`, 284 artifacts) | **OK** -- SHA-256 `ad12325d262e913dc57fa0e805e07c2115aec5005268c704177d7fb72856aa69` unchanged |
| PDF overflow (`draft_000366`) | **0** violations (160 pages) |
| commit | pending (git identity not configured locally) |

## Phase D (2026-06-08) -- bug-class lint sweep (production scope)

B023 23 / B905 22 / B904 1 / RUF012 1 / B007 4 = **51 fixed** (ruff instance count).
B023: default-arg binding (all benign in-iteration closures). B905: `strict=True` where
equal-length by construction; `strict=False` for ragged `.get(col, Series())` zips and
untested UI. B904 `from exc`; RUF012 `ClassVar`; B007 `_` renames. Note: `_norm_med_for_bin`
duplicated in `comp_pool_rms` + `comp_selection_per_target` -- Phase F manual audit.

| Check | Vysledok |
|---|---|
| B023/B905/B904/RUF012/B007 production | **0** remaining |
| `pytest tests` | **174 passed, 6 skipped** |
| Photometry byte-identity (`draft_000366`, 284 artifacts) | **OK** -- SHA-256 `ad12325d262e913dc57fa0e805e07c2115aec5005268c704177d7fb72856aa69` unchanged |
| PDF overflow (`draft_000366`) | **0** violations (160 pages) |
| commit | pending (git identity not configured locally) |

## Phase E (2026-06-08) -- trust_flag_core Findings A+B

Finding A: un-evaluated summary rows default RED + reason (was GREEN). Finding B: nan
check-star scatter adds soft note. C1: keep ddof=0 (documented). D: forward guard kept.

| Check | Vysledok |
|---|---|
| `pytest tests` | **183 passed, 6 skipped** (+9 trust tests) |
| Numeric photometry (`draft_000366`, LC + comp_quality + comparison_stars) | **unchanged** (283 files; trust does not touch numbers) |
| Trust baseline diff (`draft_000366` re-run) | **10** GREEN->YELLOW (no-check soft); **0** GREEN->RED; **0** other level flips; 8 YELLOW JSON reason-only updates |
| PDF overflow (`draft_000366`) | **0** violations (160 pages) |
| commit | pending (git identity not configured locally) |

## Phase F follow-ups (2026-06-08) -- consolidated

CQ-B dead ternary; CQ-E `norm_id_or_empty` in `gaia_catalog_id.py`; shared `norm_med_for_bin`;
DB-A editable-table allowlist; m2 shows RMS-only count. Part 3 verified: CQ-A (proc glob tests),
CAL-A (no `allow_passthrough=True` callers), DB-B (pipeline per-rerun, not in session_state).

| Check | Vysledok |
|---|---|
| `pytest tests` | **183 passed, 6 skipped** |
| RUF034 production | **3** remaining (CQ-B cleared: was 4) |
| On-disk photometry SHA (`draft_000366`, 283 LC+comp+comparison) | **unchanged** across Part 1-2 diff (`770966c3...`); historical baseline `ad12325d...` drifted pre-Phase-F (on-disk tree) |
| PDF overflow (`draft_000366`) | **0** violations (160 pages) |
| commit | pending (git identity not configured locally) |

## Phase G batch 1 (2026-06-08) -- broad-except hygiene (result paths)

G1.0: `photometry_core` COMP_QA/TRUST wrappers already `logging.warning` -- no change, ROADMAP
sub-point closed. G1.1: 8 solve-result-path excepts in `vyvar_platesolver.py` now `LOGGER.debug`
(logging-only; no type narrowing, no control-flow change).

| Check | Vysledok |
|---|---|
| `pytest tests` | **183 passed, 6 skipped** |
| Numeric photometry SHA (`draft_000366`, 283 files) | **unchanged** (`770966c3...`) |
| PDF overflow (`draft_000366`) | **0** violations (160 pages) |
| commit | pending (git identity not configured locally) |

## Phase G batch 2 (2026-06-08) -- platesolver pass-style except triage

6 pass-excepts now log (1 warning MASTERSTAR WCS persist, 5 debug refinements/headers); ~25
confirmed skip-OK. OPEN QUESTION: fatal MASTERSTAR WCS write? (ROADMAP)

| Check | Vysledok |
|---|---|
| `pytest tests` | **183 passed, 6 skipped** |
| Numeric photometry SHA (`draft_000366`, 283 files) | **unchanged** (`770966c3...`) |
| PDF overflow (`draft_000366`) | **0** violations (160 pages) |
| commit | pending (git identity not configured locally) |

## Phase G batch 3 (2026-06-08) -- photometry_core pass/return-path triage

7 sites logged (3 warning, 4 debug); high-risk return/write subset of 230 excepts reviewed;
graceful-fallback/loop-skip/already-logged remainder skip-OK. OPEN QUESTION: edge-ok fail-open.

| Check | Vysledok |
|---|---|
| `pytest tests` | **183 passed, 6 skipped** |
| Numeric photometry SHA (`draft_000366`, 283 files) | **unchanged** (`770966c3...`) |
| PDF overflow (`draft_000366`) | **0** violations (160 pages) |
| commit | pending (git identity not configured locally) |

## Phase G batch 4 (2026-06-08) -- pipeline.py triage; critical path COMPLETE

3 sites logged (comparison-star sync warning; cone/variables + prefetch CSV debug); worker
status-dict error surfacing + graceful fallbacks reviewed. Phase G critical path done.

| Check | Vysledok |
|---|---|
| `pytest tests` | **183 passed, 6 skipped** |
| Numeric photometry SHA (`draft_000366`, 283 files) | **unchanged** (`770966c3...`) |
| PDF overflow (`draft_000366`) | **0** violations (160 pages) |
| commit | pending (git identity not configured locally) |

## Phase H (2026-06-08) -- cosmetic lint; clean-code campaign COMPLETE

SIM118 x11 (ProcFrameStore x2 kept `.keys()`), RUF022 x2, RUF007 x2, RUF034 x3 dead-ternary;
~89 style findings accepted (PROCESS). Campaign A-H done.

| Check | Vysledok |
|---|---|
| `pytest tests` | **183 passed, 6 skipped** |
| Ruff SIM118/RUF022/RUF007/RUF034 | **2** SIM118 (ProcFrameStore intentional); rest **0** |
| Numeric photometry SHA (`draft_000366`, 283 files) | **unchanged** (`770966c3...`) |
| PDF overflow (`draft_000366`) | **0** violations (160 pages) |
| commit | pending (git identity not configured locally) |

## FWHM-CONSISTENCY (2026-06-09) -- crowding + ePSF context prefer VY_FWHM_GAUSS

Shared `header_core_fwhm_px(hdr)` in `masterstar_context.py`. Two read sites only:
`crowding_index._load_wcs_meta` (line ~62) and `psf_photometry.get_epsf_fwhm_from_context` (line ~181).
Aperture path untouched.

| Check | Vysledok |
|---|---|
| `pytest tests` | **183 passed, 6 skipped** |
| Numeric photometry SHA (`draft_000366`, 283 files) | **unchanged** (`770966c3...`) |
| Crowding 375 L (live) | is_blended **58**, hard **39**, blend@1 **0.0268**, blend@2 **0.0737** |
| Crowding 380 L (live) | is_blended **53**, hard **34**, blend@1 **0.0269**, blend@2 **0.0751** |
| ePSF ctx FWHM 375/380 L | **2.744 / 2.730** px; QC ratio ~**0.78 / 0.81** |

## NEIGHBOR-SUB step 2 + 2a (2026-06-08) -- A9 envelope, joint-fit core, fail-safe guards

Validation-scoped (`psf_neighbor_sub_enabled` default OFF; production measurement sites unwired).
Joint-fit target+neighbour, subtract neighbour only, aperture residual via
`_catalog_only_fixed_aperture_flux`. `BlendMapEntry` / `_load_blend_worklist` in `photometry_core.py`.
A9 harness: `tests/validation/a9_core.py`, `gen_a9.py`, `run_a9.py`; recover `--a9`.

Step **2a** guards: inclusive sep floor (`nn_dist_fwhm <= 0.8`); catalog-anchored
`neighbor_overfit`, `target_undershoot`, `subtract_harmed`, sky-noise SNR floor.

| Check | Vysledok |
|---|---|
| `pytest tests` | **203 passed, 6 skipped** |
| Numeric photometry SHA (`draft_000366`, 283 files) | **unchanged** (`770966c3...`) |
| A9 ideal pass (coarse) | **85.7%** |
| A9 realistic mismatch post-2a | FAIL-SILENT **0**; HV PASS-RECOVER **17.6%**; verdict **SAFE_LOW_YIELD** |
| Step 2b (pipeline wire) | **blocked** -- low yield at coarse bin2; fine-scale A9 / ePSF first |
| commit | `055595d` feat(validation) |

Mismatch diagnostic: `python -m tests.validation.run_a9_mismatch_diagnostic` ->
`tests/validation/data/tier_a9/a9_mismatch_diagnostic.md` (gitignored; regenerate locally).
Design: `docs/VYVAR_NEIGHBOR_SUB_DESIGN.md` (joint fit §3b; guards §6).

## NEIGHBOR-SUB pre-2b guard + 367 crowding (2026-06-08)

`bright_close_regime` guard (dM>=2.5 + sep<=1.1 FWHM). A9 re-run: 367 FAIL-SILENT **0**, HV **83.3%**;
coarse realistic FAIL-SILENT **0**. 367 crowding Red_180_2: is_blended **9**, hard **4** -> **SPARSE**;
combined **VALIDATED_FINE_SCALE_IDLE** (defer 2b). Brno gate in `VYVAR_DECISIONS.md`.

| Check | Vysledok |
|---|---|
| Numeric photometry SHA | **unchanged** (`770966c3...`) |
| `psf_neighbor_sub_enabled` | **OFF** |

## NEIGHBOR-SUB fine-scale test draft 367 (2026-06-08) -- read-only diagnostic

Part 1 ePSF-vs-star Moffat audit on draft 367 Red_180_2 (0.3889 arcsec/px): **mismatch ratio
0.9994** (vs h & chi Per 375 L **1.112**). Part 2 A9 draft367-calibrated: HV PASS-RECOVER **83.3%**
(coarse realistic 17.6%), FAIL-SILENT **1** (sep1.0/dM-3 edge), REFUSE **100%**. Verdict:
**FINE_SCALE_HOME_EDGE_FAIL_SILENT** -- sampling rescued mismatch; 2b wire blocked on one edge cell.
`psf_neighbor_sub_enabled` OFF; no production wiring.

| Check | Vysledok |
|---|---|
| Numeric photometry SHA (`draft_000366`, 283 files) | **unchanged** (`770966c3...`) |
| ePSF audit JSON | `tmp/epsf_fwhm_367.json` |
| A9 report | `tier_a9/a9_draft367_diagnostic.md` (regenerate: `run_a9_draft367`) |

## Dalsie kroky (Faza 1 pokracovanie)

1. **Split track** -- `pipeline.py` / `photometry_core.py` (samostatny spec, byte-identita).
2. **comp_qa fix-once locus (CQ-C)** + **ddof co-calibration** -- methodology passes with bounded diff.

## Nástroje

```bash
python tmp/_gen_audit_findings.py   # regeneruje docs/VYVAR_AUDIT_FINDINGS.md
python -m ruff check . --select F821,F811,F841 --statistics
python -m pytest tests -q
```
