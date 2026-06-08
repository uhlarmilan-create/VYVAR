# VYVAR ? Audit findings (Fáza 0, lokálny snapshot)

**Generované:** 2026-06-08 ? **Príkaz:** `python tmp/_gen_audit_findings.py`

Automatická previerka pred ruèným auditom. Priorita opráv: **F841 ? tichý except ? F821/F811 ? F401/F541**.

## Súhrn (ruff `--select F,B,SIM,RUF --statistics`)

```
[1m1294[0m	[1;31mRUF100[0m	[[36m*[0m] unused-noqa
[1m 308[0m	[1;31mRUF046[0m	[[36m-[0m] unnecessary-cast-to-int
[1m 151[0m	[1;31mRUF001[0m	[ ] ambiguous-unicode-character-string
[1m  93[0m	[1;31mRUF002[0m	[ ] ambiguous-unicode-character-docstring
[1m  88[0m	[1;31mRUF003[0m	[ ] ambiguous-unicode-character-comment
[1m  79[0m	[1;31mSIM105[0m	[ ] suppressible-exception
[1m  36[0m	[1;31mSIM102[0m	[ ] collapsible-if
[1m  24[0m	[1;31mB905  [0m	[ ] zip-without-explicit-strict
[1m  24[0m	[1;31mF841  [0m	[ ] unused-variable
[1m  23[0m	[1;31mB023  [0m	[ ] function-uses-loop-variable
[1m  22[0m	[1;31mRUF059[0m	[ ] unused-unpacked-variable
[1m  17[0m	[1;31mSIM114[0m	[[36m*[0m] if-with-same-arms
[1m  13[0m	[1;31mSIM118[0m	[[36m-[0m] in-dict-keys
[1m  12[0m	[1;31mRUF005[0m	[ ] collection-literal-concatenation
[1m  11[0m	[1;31mF401  [0m	[[36m*[0m] unused-import
[1m  10[0m	[1;31mSIM108[0m	[ ] if-else-block-instead-of-if-exp
[1m   8[0m	[1;31mRUF010[0m	[[36m*[0m] explicit-f-string-type-conversion
[1m   6[0m	[1;31mSIM103[0m	[ ] needless-bool
[1m   6[0m	[1;31mSIM113[0m	[ ] enumerate-for-loop
[1m   4[0m	[1;31mB007  [0m	[ ] unused-loop-control-variable
[1m   4[0m	[1;31mB009  [0m	[[36m*[0m] get-attr-with-constant
[1m   4[0m	[1;31mB010  [0m	[[36m*[0m] set-attr-with-constant
[1m   4[0m	[1;31mRUF034[0m	[ ] useless-if-else
[1m   3[0m	[1;31mSIM300[0m	[[36m*[0m] yoda-conditions
[1m   2[0m	[1;31mRUF007[0m	[ ] zip-instead-of-pairwise
[1m   2[0m	[1;31mRUF022[0m	[[36m-[0m] unsorted-dunder-all
[1m   1[0m	[1;31mB904  [0m	[ ] raise-without-from-inside-except
[1m   1[0m	[1;31mRUF012[0m	[ ] mutable-class-default
[1m   1[0m	[1;31mSIM910[0m	[[36m*[0m] dict-get-with-none-default
Found 2251 errors.
[[36m*[0m] 1571 fixable with the `--fix` option (222 hidden fixes can be enabled with the `--unsafe-fixes` option).
```

### Kµúèové kódy (production scope kde uvedené)

| Kód | Celkom (repo) | Production core* | Poznámka |
|---|---:|---:|---|
| F841 unused-variable | 24 | 24 | **najvy¹¹í bug-yield** |
| F821 undefined-name | 0 | 0 | väè¹inou `photometry_report` string anotácie |
| F811 redefined-while-unused | 0 | 0 | redundantné re-importy |
| F401 unused-import | 11 | ? | auto-fix + review |
| F541 f-string bez placeholders | 0 | ? | auto-fix + review |
| `except: pass` | 3 | 0 (kritická cesta) | audit po jednom |
| `except Exception` | 1426 | 683 (kritická cesta) | **nehromadne** |
| `print(` | 1310 | 164 (root `.py`) | triage debug vs zámer |
| `import *` | 7 | ? | väè¹inou re-export shims |
| TODO/FIXME | 96 | ? | ledger / ROADMAP |

\* Production core = root moduly + `orchestrator/` + `GAIA_DR3/` (bez sandbox/tmp/archive scripts).

## F821 / F811 / F841 ? detail (production)

| súbor:riadok | trieda | krátky popis |
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

## F841 ? mimo production (scripts/tests/sandbox)

| súbor:riadok | trieda | krátky popis |
|---|---|---|

## `except: pass` ? top súbory (celý repo)

| súbor | poèet |
|---|---:|
| `tmp/_gen_audit_findings.py` | 3 |

## `except Exception` ? top súbory (celý repo)

| súbor | poèet |
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

## F401 + F541 (auto-fix kandidáti)

| súbor:riadok | trieda | krátky popis |
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

## TODO/FIXME ? top súbory

| súbor | poèet |
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

- **F821:** was 27, now **0** ? TYPE_CHECKING import in `photometry_report.py` (string annotations only).
- **F811:** was 7, now **0** ? removed duplicate merge-block imports in `photometry_core.py`; dead locals in `photometry_report`.
- **F841:** was 44, now **22** ? removed dead `run_phase2a` state unpacks, `comp_pool_rms.avail_cols`; 22 remain (triage next).
- **F401/F541:** auto-fixed in batch 1 (production modules); re-run after `tests/` restore may show ~11 F401 in tests.
- **`except: pass`:** bare `except: pass` rare locally (3 repo-wide); batch 1 logged Gaia lookup in `comp_selection_per_target.py`.
- **`except Exception`:** 1426 repo / 683 critical path ? do not bulk-edit.

**Verify:** `python -m pytest tests -q` ? 174 passed, 6 skipped (2026-06-08 batch 1).

## trust_flag_core.py ? Phase 2 manual logic audit (2026-06-08)

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
   (~109). A target with no usable check star gets no penalty ? can be GREEN.
   Fix: add a soft note ("no check-star verification available") when nan.

C. **[LOW-MEDIUM, methodology] `np.nanstd` uses population std (ddof=0) at line ~80.**
   For a check-star scatter gated at 0.02/0.05 mag, sample std (ddof=1) is the
   conventional, less-biased estimator (min N=2 ? ddof=0 under-reports by ~?2).
   Decide consciously; record in DECISIONS.

D. **[LOW, latent] `trust_level` "len(soft) >= 3 ? RED" (~124) is currently
   unreachable:** `classify_warnings` produces at most 2 soft warnings (thin comp +
   check-soft). Document as a guard for future soft sources, or drop.

E. **[LOW] Empty/missing `lc_quality` treated as OK:** `evaluate_target` maps empty ?
   "?" (~174) and `classify_warnings` skips "?" (~106). If comp_qa fails to emit
   the flag, the target isn't penalized on that axis. Consider a soft note.

F. **[info, not a bug] Redundancies:** `trust_level`'s explicit `n_clean < min_comps`
   (~124) duplicates the hard warning from `classify_warnings`; the
   `elif th.min_comps <= nc` (~100) is also redundant. Harmless.

Regression note: this module is read-only w.r.t. photometry numbers, so fixes do
NOT change photometry byte-identity, but DO change trust artifact contents (by
design). Guard with unit tests on trust logic + a trust-output baseline, NOT the
photometry SHA.
