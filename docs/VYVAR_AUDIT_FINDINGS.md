# VYVAR - Audit findings (Faza 0, lokalny snapshot)

**Generovane:** 2026-06-08 - **Prikaz:** `python tmp/_gen_audit_findings.py`

Automaticka previerka pred rucnym auditom. Priorita oprav: **F841 -> tichy except -> F821/F811 -> F401/F541**.

## Suhrn (ruff `--select F,B,SIM,RUF --statistics`)

```
[1m1294[0m	[1;31mRUF100[0m	[[36m*[0m] unused-noqa
[1m 308[0m	[1;31mRUF046[0m	[[36m-[0m] unnecessary-cast-to-int
[1m 151[0m	[1;31mRUF001[0m	[ ] ambiguous-unicode-character-string
[1m  93[0m	[1;31mRUF002[0m	[ ] ambiguous-unicode-character-docstring
[1m  88[0m	[1;31mRUF003[0m	[ ] ambiguous-unicode-character-comment
[1m  79[0m	[1;31mSIM105[0m	[ ] suppressible-exception
[1m  36[0m	[1;31mSIM102[0m	[ ] collapsible-if
[1m  24[0m	[1;31mB905  [0m	[ ] zip-without-explicit-strict
[1m  24[0m	[1;31mF841  [0m	[ ] unused-variable
[1m  23[0m	[1;31mB023  [0m	[ ] function-uses-loop-variable
[1m  22[0m	[1;31mRUF059[0m	[ ] unused-unpacked-variable
[1m  17[0m	[1;31mSIM114[0m	[[36m*[0m] if-with-same-arms
[1m  13[0m	[1;31mSIM118[0m	[[36m-[0m] in-dict-keys
[1m  12[0m	[1;31mRUF005[0m	[ ] collection-literal-concatenation
[1m  11[0m	[1;31mF401  [0m	[[36m*[0m] unused-import
[1m  10[0m	[1;31mSIM108[0m	[ ] if-else-block-instead-of-if-exp
[1m   8[0m	[1;31mRUF010[0m	[[36m*[0m] explicit-f-string-type-conversion
[1m   6[0m	[1;31mSIM103[0m	[ ] needless-bool
[1m   6[0m	[1;31mSIM113[0m	[ ] enumerate-for-loop
[1m   4[0m	[1;31mB007  [0m	[ ] unused-loop-control-variable
[1m   4[0m	[1;31mB009  [0m	[[36m*[0m] get-attr-with-constant
[1m   4[0m	[1;31mB010  [0m	[[36m*[0m] set-attr-with-constant
[1m   4[0m	[1;31mRUF034[0m	[ ] useless-if-else
[1m   3[0m	[1;31mSIM300[0m	[[36m*[0m] yoda-conditions
[1m   2[0m	[1;31mRUF007[0m	[ ] zip-instead-of-pairwise
[1m   2[0m	[1;31mRUF022[0m	[[36m-[0m] unsorted-dunder-all
[1m   1[0m	[1;31mB904  [0m	[ ] raise-without-from-inside-except
[1m   1[0m	[1;31mRUF012[0m	[ ] mutable-class-default
[1m   1[0m	[1;31mSIM910[0m	[[36m*[0m] dict-get-with-none-default
Found 2251 errors.
[[36m*[0m] 1571 fixable with the `--fix` option (222 hidden fixes can be enabled with the `--unsafe-fixes` option).
```

### Klucove kody (production scope kde uvedene)

| Kod | Celkom (repo) | Production core* | Poznamka |
|---|---:|---:|---|
| F841 unused-variable | 24 | 24 | **najvyssi bug-yield** |
| F821 undefined-name | 0 | 0 | vacsinou `photometry_report` string anotacie |
| F811 redefined-while-unused | 0 | 0 | redundantne re-importy |
| F401 unused-import | 11 | - | auto-fix + review |
| F541 f-string bez placeholders | 0 | - | auto-fix + review |
| `except: pass` | 3 | 0 (kriticka cesta) | audit po jednom |
| `except Exception` | 1426 | 683 (kriticka cesta) | **nehromadne** |
| `print(` | 1310 | 164 (root `.py`) | triage debug vs zamer |
| `import *` | 7 | - | vacsinou re-export shims |
| TODO/FIXME | 96 | - | ledger / ROADMAP |

\* Production core = root moduly + `orchestrator/` + `GAIA_DR3/` (bez sandbox/tmp/archive scripts).

## F821 / F811 / F841 - detail (production)

| subor:riadok | trieda | kratky popis |
|---|---|---|
| `comp_qa.py:53` | F841 | Local variable `lc_map` is assigned to but never used |
| `comp_selection_per_target.py:243` | F841 | Local variable `g_teff` is assigned to but never used |
| `comp_selection_per_target.py:1538` | F841 | Local variable `dist_score` is assigned to but never used |
| `comp_selection_per_target.py:1952` | F841 | Local variable `rms_f2` is assigned to but never used |
| `photometry_core.py:7141` | F841 | Local variable `c1_stderr` is assigned to but never used |
| `photometry_core.py:7786` | F841 | Local variable `lc_df` is assigned to but never used |
| `photometry_core.py:10351` | F841 | Local variable `ra_ms` is assigned to but never used |
| `photometry_core.py:10352` | F841 | Local variable `de_ms` is assigned to but never used |
| `photometry_core.py:10405` | F841 | Local variable `gaia_teff` is assigned to but never used |
| `photometry_report.py:4092` | F841 | Local variable `cols` is assigned to but never used |
| `pipeline.py:6319` | F841 | Local variable `n0` is assigned to but never used |
| `pipeline.py:15805` | F841 | Local variable `cfg` is assigned to but never used |
| `psf_photometry.py:1643` | F841 | Local variable `fit_shape` is assigned to but never used |
| `psf_runner.py:1382` | F841 | Local variable `mn_cid` is assigned to but never used |
| `tess_verify.py:912` | F841 | Local variable `center_col_tpf` is assigned to but never used |
| `tess_verify.py:913` | F841 | Local variable `center_row_tpf` is assigned to but never used |
| `tests/test_alg_functions.py:341` | F841 | Local variable `target_cid` is assigned to but never used |
| `tests/test_comp_determinism.py:116` | F841 | Local variable `cs345` is assigned to but never used |
| `ui_finalization.py:436` | F841 | Local variable `field` is assigned to but never used |
| `ui_variability.py:1480` | F841 | Local variable `n_rms_candidates` is assigned to but never used |
| `vyvar_alignment_frame.py:351` | F841 | Local variable `max_detected_stars` is assigned to but never used |
| `vyvar_platesolver.py:2739` | F841 | Local variable `center` is assigned to but never used |
| `xval_run.py:79` | F841 | Local variable `sf` is assigned to but never used |
| `xval_run.py:100` | F841 | Local variable `PS` is assigned to but never used |

## F841 - mimo production (scripts/tests/sandbox)

| subor:riadok | trieda | kratky popis |
|---|---|---|

## `except: pass` - top subory (cely repo)

| subor | pocet |
|---|---:|
| `tmp/_gen_audit_findings.py` | 3 |

## `except Exception` - top subory (cely repo)

| subor | pocet |
|---|---:|
| `pipeline.py` | 317 |
| `photometry_core.py` | 230 |
| `photometry_report.py` | 84 |
| `vyvar_platesolver.py` | 72 |
| `ui_variability.py` | 45 |
| `comp_selection_per_target.py` | 37 |
| `psf_photometry.py` | 35 |
| `ui_aperture_photometry.py` | 33 |
| `tess_verify.py` | 31 |
| `variability_detector.py` | 24 |
| `database.py` | 22 |
| `importer.py` | 22 |
| `app.py` | 21 |
| `psf_runner.py` | 20 |
| `astrometry_optimizer.py` | 19 |

## F401 + F541 (auto-fix kandidati)

| subor:riadok | trieda | kratky popis |
|---|---|---|
| `tests/test_alg_functions.py:10` | F401 | `math` imported but unused |
| `tests/test_blind_knn_construction.py:9` | F401 | `pytest` imported but unused |
| `tests/test_blind_knn_construction.py:116` | F401 | `vyvar_blind_solver.BlindCandidate` imported but unused |
| `tests/test_blind_rig_prior.py:5` | F401 | `math` imported but unused |
| `tests/test_blind_verify.py:9` | F401 | `pytest` imported but unused |
| `tests/test_geo.py:6` | F401 | `types.SimpleNamespace` imported but unused |
| `tests/test_geo.py:7` | F401 | `unittest.mock.MagicMock` imported but unused |
| `tests/test_gs11_pipeline.py:8` | F401 | `pytest` imported but unused |
| `tests/test_lc_quality.py:5` | F401 | `math` imported but unused |
| `tests/test_masterstar_obs_group.py:6` | F401 | `pytest` imported but unused |
| `tests/test_pre_calibrated_run.py:5` | F401 | `json` imported but unused |

## TODO/FIXME - top subory

| subor | pocet |
|---|---:|
| `sandbox/_merge_vyvar_state.py` | 22 |
| `photometry_core.py` | 11 |
| `pipeline.py` | 8 |
| `tmp/_gen_audit_findings.py` | 6 |
| `sandbox/scripts/_integration_test_todos.py` | 6 |
| `config.py` | 5 |
| `ui_masterstar_qa.py` | 4 |
| `sandbox/scripts/lc_trend_diagnostic.py` | 3 |
| `ui_settings.py` | 2 |
| `sandbox/scripts/sky_gradient_sky_plane_361_362.py` | 2 |
| `sandbox/scripts/test_airmass_order.py` | 2 |
| `sandbox/scripts/archive/verify/_todo44_verify_draft321.py` | 2 |
| `app.py` | 1 |
| `comp_selection_per_target.py` | 1 |
| `dilution.py` | 1 |

## Recalibration (Claude audit) + batch 1 status

- **F821:** was 27, now **0** - TYPE_CHECKING import in `photometry_report.py` (string annotations only).
- **F811:** was 7, now **0** - removed duplicate merge-block imports in `photometry_core.py`; dead locals in `photometry_report`.
- **F841:** was 44, now **22** - removed dead `run_phase2a` state unpacks, `comp_pool_rms.avail_cols`; 22 remain (triage next).
- **F401/F541:** auto-fixed in batch 1 (production modules); re-run after `tests/` restore may show ~11 F401 in tests.
- **`except: pass`:** bare `except: pass` rare locally (3 repo-wide); batch 1 logged Gaia lookup in `comp_selection_per_target.py`.
- **`except Exception`:** 1426 repo / 683 critical path - do not bulk-edit.

**Verify:** `python -m pytest tests -q` - 174 passed, 6 skipped (2026-06-08 batch 1).

## trust_flag_core.py - Phase 2 manual logic audit (2026-06-08)

A. **[MEDIUM-HIGH, mission] Un-evaluated target defaults to GREEN.**
   `write_trust_artifacts` (lines ~296-298, `trust_map.get(norm_id(x), "GREEN")`),
   `format_export_trust_note` (~340) and `format_varastro_trust_comment` (~352)
   default a missing trust to GREEN. `compute_trust_for_photometry_dir` skips rows
   with empty `catalog_id` (~219), so an un-evaluated target is written to
   `photometry_summary.csv` and AAVSO/VarAstro notes as GREEN (highest trust).
   Fix: default to RED (or UNKNOWN) + reason "not evaluated"; `LOGGER.warning` when
   a summary id is absent from `trust_map`.

B. **[MEDIUM, mission] Missing check-star treated as pass.**
   `check_star_scatter` returns nan when the check file is missing or has <2 points
   (~69-82); `classify_warnings` only penalizes when `math.isfinite(check_scatter)`
   (~109). A target with no usable check star gets no penalty - can be GREEN.
   Fix: add a soft note ("no check-star verification available") when nan.

C. **[LOW-MEDIUM, methodology] `np.nanstd` uses population std (ddof=0) at line ~80.**
   For a check-star scatter gated at 0.02/0.05 mag, sample std (ddof=1) is the
   conventional, less-biased estimator (min N=2 - ddof=0 under-reports by ~sqrt(2)).
   Decide consciously; record in DECISIONS.

D. **[LOW, latent] `trust_level` "len(soft) >= 3 -> RED" (~124) is currently
   unreachable:** `classify_warnings` produces at most 2 soft warnings (thin comp +
   check-soft). Document as a guard for future soft sources, or drop.

E. **[LOW] Empty/missing `lc_quality` treated as OK:** `evaluate_target` maps empty ->
   "?" (~174) and `classify_warnings` skips "?" (~106). If comp_qa fails to emit
   the flag, the target isn't penalized on that axis. Consider a soft note.

F. **[info, not a bug] Redundancies:** `trust_level`'s explicit `n_clean < min_comps`
   (~124) duplicates the hard warning from `classify_warnings`; the
   `elif th.min_comps <= nc` (~100) is also redundant. Harmless.

Regression note: this module is read-only w.r.t. photometry numbers, so fixes do
NOT change photometry byte-identity, but DO change trust artifact contents (by
design). Guard with unit tests on trust logic + a trust-output baseline, NOT the
photometry SHA.

## EPSF-1 -- ePSF FWHM QC estimator bias (2026-06-08, read-only audit)

**File:** `psf_photometry.py:500-516` (`_epsf_build_imagepsf_from_stars`).

**Finding:** `epsf_fwhm_native` half-max uses first pixel below 0.5 after `v/v.max()` sort --
systematically underestimates FWHM => `epsf_vs_input_fwhm_ratio` biased < 1.

**Evidence:** h & chi Per probe ratio 0.59-0.67. **Resolved 2026-06-09** (`docs/VYVAR_EPSF_FWHM_TEST.md`):
dominant cause = **EXPLANATION 3** (OBS_FILES seeing ~3.84 px vs ePSF/stellar core ~2.0 px).
Secondary = **EXPLANATION 1** (buggy half-max pinned at 2.236 = sqrt(5)). EXPLANATION 2 rejected
(ePSF Moffat matches stellar Moffat).

**Blast radius:** QC/diagnostic only -- NOT used by `assess_psf_quality` or aperture correction.

**Disposition:** ROADMAP `TODO-EPSF-1-FWHM-QC` (estimator fix still worthwhile); seeing denominator
alignment is separate. Do not block PSF on ratio alone.

## F841 batch 3 audit notes (2026-06-08)

- **`pipeline.py:15805` `cfg`:** dead local in `extract_fits_metadata`; `AppConfig()` had no side effect.
- **`vyvar_alignment_frame.py:351` `max_detected_stars`:** read from ctx but never used; cap enforced via `align_star_cap`.
- **`psf_photometry.py:1643` `fit_shape`:** vestigial ePSF copy-paste in Moffat path; PSF OFF in production.
- **`ui_variability.py:1480` `n_rms_candidates`:** Cat 3 pending Milan decision (wire m2 vs remove dead local).

## Phase F manual audit -- comp_qa_core.py (2026-06-08, DONE)

Read-only logic audit at HEAD fe8201c. calibration / database / vyvar_platesolver pending.

### CQ-A [doc drift; ROADMAP HIGH likely RESOLVED] -- canonical proc-CSV resolution

ROADMAP HIGH + STATE 2026-06-04 (`load_proc_pivot` hardcodes `proc_*_Light_*.csv`) is **stale**.
At HEAD: `load_proc_pivot` (227) -> `list_proc_csvs` (proc_frame_store.py:84) globs
`PROC_CSV_GLOB = "proc_*.csv"` (79), matching calibrated and pre-cal native basenames.
Same constant/helper used by comp_qa_core, ProcFrameStore.build, photometry_core, xval_run.
Pre-cal zero-glob -> n_clean NaN -> trust RED defect fixed on the CSV path.
**Disposition:** VERIFY on a pre-cal draft (re-run comp QA, confirm n_clean populated), then
CLOSE ROADMAP HIGH and update STATE. Residual LOW: pipeline FITS selection uses inline
`aligned_dir.glob("proc_*.fits")` (~5578, 12608, 12687) -- correct but not one shared resolver.

### CQ-B [LOW-MED, latent/clarity] -- useless ternary in final QA pass

`compute_comp_qa` line 435: `peers = surv_final if cid in surv_final else surv_final`
(RUF034). Both arms identical; dropped comps scored vs clean surviving set via loo_diff_series.
**Disposition:** if intent is identical peers, simplify to `peers = surv_final` (byte-identical).
If dropped comps were meant different peers, science change + bounded-diff validation.

### CQ-C [MED, methodology] -- field locus mutates with target processing order

Drop loop (348-404) rebuilds magnitude-sigma locus (357-368) from `dropped_global` that
accumulates drops across prior targets. Locus is order-coupled but deterministic (groupby sort).
**Disposition:** (a) compute locus once over full pass-1 pool, or (b) document coupling as
intended. Record in DECISIONS. Changing locus source breaks byte-identity -> bounded n_clean/trust
diff on reference draft.

### CQ-D [perf, non-blocking] -- O(N_targets^2 * N_comps^2 * drop_iters) locus rebuild

Locus rebuild inside each target's while-drop loop. Acceptable per overnight-batch decision;
LOW if rich-field runtime hurts; ties to CQ-C option (a) or caching.

### CQ-E [LOW, duplication] -- norm-id helper duplicated

`comp_qa_core._normalize_id` (256-268) duplicates `trust_flag_core.norm_id` (54-66).
**Disposition:** one canonical `norm_id` in gaia_catalog_id.py; byte-identical import swap.

### Cross-ref: duplicated `_norm_med_for_bin`

Near-identical in comp_pool_rms.py and comp_selection_per_target.py (Phase D flag).
**Disposition:** extract shared magnitude-bin normalizer; validate byte-identity on draft_000366.

## Phase F manual audit -- calibration.py (2026-06-08, DONE)

Overall: clean; no real bugs. Binning math correct (dark = block sum, flat = block mean);
trailing-pixel trim documented; shape mismatch raises `MasterResamplingError`; Bayer flat
per-tile norm uses in-place views; DB connection closed in `finally`. Only 2 broad-excepts,
both narrow.

### CAL-A [LOW] -- silent passthrough on a missing master

`get_processed_master` (440-453): missing master + `allow_passthrough=True` returns no-op base
(zeros dark / ones flat) with `is_passthrough=True`. Calibration becomes a silent no-op.
**Disposition:** verify callers log/surface passthrough so misconfigured masters cannot quietly
skip calibration. No change to calibration.py itself. LOW.

### CAL-B [LOW / future] -- RGGB assumed for OSC without BAYERPAT

`normalize_flat_master` (342-345): no `BAYERPAT` + OSC EQUIPMENTS hint -> assume RGGB
(`assumed_pat=True`, logged). Wrong for BGGR/GRBG/GBRG. Mono wide rig today (QHY294MM).
**Disposition:** when TODO-45 RGB lands, read true pattern from sensor metadata. FUTURE.

### CAL-C [trivial] -- `align_resampled_master_to_light_shape` unused `kind`

Line 195: `kind` accepted but unused (`_ = kind`); API symmetry with
`resample_master_to_light_binning`. Intentional; no action.

### CAL-D [LOW, design note] -- Bayer path also runs final global rescale

After per-tile Bayer norm, global `abs(gfin - 1.0) > 0.02` rescale (405-408) still applies.
Negligible (gfin ~= 1) and defensive (bad-pixel->1.0 guard); Bayer branch could skip it.
**Disposition:** document intent or gate rescale to global branch. LOW.

## Phase F manual audit -- database.py (2026-06-08, DONE)

Overall: targeted audit; mostly sound. Values parameterized via `?`; transactions use
BEGIN/commit/rollback; gaia ids via canonical `normalize_gaia_source_id`. f-string SQL only on
identifiers SQLite cannot bind. Findings defensive / latent.

### DB-A [LOW, defensive] -- f-string SQL on identifiers trusts `table` argument

14 `execute(f"...")` sites interpolate table/column/type names (necessary for DDL). Sources
today trusted: schema constants, pragma-derived columns, known editable set (`apply_main_table_editor_save`
line 1197 checks EQUIPMENTS/TELESCOPE etc.). Values always `?`-bound. Not exploitable today.
**Disposition:** harden editable-table paths with explicit allowlist assert (`table in EDITABLE_TABLES`).
LOW.

### DB-B [MEDIUM-LOW, latent threading] -- default check_same_thread=True

`VyvarDatabase.__init__` (759): `sqlite3.connect` without `check_same_thread=False`. Streamlit
reruns across threads could raise if instance retained in session_state. app.py does not cache
VyvarDatabase via `@st.cache_resource` (suggests per-operation instances, unproven).
**Disposition:** verify instances never reused across Streamlit threads; else
`check_same_thread=False` + lock or re-open per operation. MEDIUM-LOW latent.

### DB-C [LOW, perf] -- full schema create + migrations on every instantiation

`__init__` (760-766) runs idempotent CREATE/ALTER on every `VyvarDatabase(...)` (31 call sites).
**Disposition:** optional one-time schema-init guard or shared connection. LOW.

### DB-D [good] -- transaction handling correct

`apply_main_table_editor_save` (1286-1364): BEGIN/commit/rollback/raise. No action.

## Phase F manual audit -- vyvar_platesolver.py (2026-06-08, DONE)

Overall: targeted audit; sophisticated, well-gated blind solver. Multi-gated acceptance: WCS
scale vs rig prior, min matches/fraction, FOV cone, gnomonic sides, cluster-RANSAC. Tuning /
hygiene findings, not correctness bugs.

### PS-A [LOW-MED, tuning] -- wide-rig verify relaxation

`_verify_blind_candidates` (1678-1679): plate scale >= 5"/px reduces `min_matches` by 4
(12 -> 8, floor 6). Loosens verify on sparse wide fields; scale gate + 0.30 fraction still apply.
OPEN wide-rig vote starvation is pre-verify. **Disposition:** when working wide-rig ROADMAP item,
decide if 8 matches + 0.30 fraction + scale gate is sufficient or fraction/scale_tol should tighten.
Document rationale.

### PS-B [MED, hygiene; Phase G priority] -- high silent-except density

72 `except Exception`; ~36 silent pass/continue vs ~6 log. Inner-loop sample skips OK; solve-result
path silent except -> None/degraded WCS can mask failure. **Disposition:** Phase G
(TODO-BROAD-EXCEPT-HYGIENE) priority site -- log on solve-result path; narrow exception types.

### PS-C [good] -- rig-prior gates sound

`_wcs_scale_gate` (1460), FOV cone, gnomonic sides per 2026-06-04 decision. No action.

## Phase F -- COMPLETE (2026-06-08)

Modules audited: comp_qa_core (CQ-A..E), calibration (CAL-A..D), database (DB-A..D),
vyvar_platesolver (PS-A..C). No new correctness bugs beyond comp_qa_core CQ-B (dead ternary).

Highest-value outcome: CQ-A -- ROADMAP HIGH "canonical pre-cal proc-CSV resolution" likely
already resolved; verify on pre-cal draft and close.

Actionable (consolidated; Milan decisions where noted):
- CQ-A: verify pre-cal draft -> close HIGH + update STATE
- CQ-B: simplify dead ternary to `peers = surv_final` if intent identical
- CQ-C: locus order-coupling -- fix-once vs document
- CQ-E + cross-ref: extract canonical norm_id + shared `_norm_med_for_bin` (byte-identity check)
- CAL-A: ensure callers log `is_passthrough`
- DB-A: allowlist assert on editable `table`
- DB-B: verify VyvarDatabase not retained across Streamlit threads
- PS-B: platesolver = Phase G priority site
- Carried-over: trust ddof C2, ui_variability `n_rms_candidates`
