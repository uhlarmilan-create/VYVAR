CURSOR RESULT - CYTHON-RELEASE-1 (2026-07-23)

What I did
Promoted spike Cython tooling to `dev/tools/cython_release/`, compiled the full
RELEASE-1 MODULE_LIST (84 science modules) on Windows MSVC, ran latent-bug sweep,
import smoke, MP spawn verification, compiled pytest, P1 golden, and `--full` anchor
gate under compiled build; verified interpreted `--fast` after clean. STOP before push.

## MODULE_LIST (S2)

Rule: all `src_py/*.py` except `app.py`, `ui_*.py`, plus `EXPLICIT_EXCLUDE`.
Source: `dev/tools/cython_release/module_list.py`.

| Module | Status | Reason |
|--------|--------|--------|
| app | EXCLUDED | UI layer (S1): Streamlit entry; stays interpreted for reload/tracebacks/no compute benefit |
| ui_aperture_photometry | EXCLUDED | UI layer (S1): ui_* stays interpreted for Streamlit reload/tracebacks |
| ui_calibration | EXCLUDED | UI layer (S1): ui_* stays interpreted for Streamlit reload/tracebacks |
| ui_calibration_library | EXCLUDED | UI layer (S1): ui_* stays interpreted for Streamlit reload/tracebacks |
| ui_components | EXCLUDED | UI layer (S1): ui_* stays interpreted for Streamlit reload/tracebacks |
| ui_dao_stars | EXCLUDED | UI layer (S1): ui_* stays interpreted for Streamlit reload/tracebacks |
| ui_database_explorer | EXCLUDED | UI layer (S1): ui_* stays interpreted for Streamlit reload/tracebacks |
| ui_epsf_dashboard | EXCLUDED | UI layer (S1): ui_* stays interpreted for Streamlit reload/tracebacks |
| ui_finalization | EXCLUDED | UI layer (S1): ui_* stays interpreted for Streamlit reload/tracebacks |
| ui_hrd | EXCLUDED | UI layer (S1): ui_* stays interpreted for Streamlit reload/tracebacks |
| ui_masterstar_qa | EXCLUDED | UI layer (S1): ui_* stays interpreted for Streamlit reload/tracebacks |
| ui_params_dashboard | EXCLUDED | UI layer (S1): ui_* stays interpreted for Streamlit reload/tracebacks |
| ui_photometry | EXCLUDED | UI layer (S1): ui_* stays interpreted for Streamlit reload/tracebacks |
| ui_photometry_quality | EXCLUDED | UI layer (S1): ui_* stays interpreted for Streamlit reload/tracebacks |
| ui_quality_dashboard | EXCLUDED | UI layer (S1): ui_* stays interpreted for Streamlit reload/tracebacks |
| ui_settings | EXCLUDED | UI layer (S1): ui_* stays interpreted for Streamlit reload/tracebacks |
| ui_variability | EXCLUDED | UI layer (S1): ui_* stays interpreted for Streamlit reload/tracebacks |
| (84 others) | INCLUDED | See module_list.py (includes osc_extract, osc_align, gaia_johnson) |

EXPLICIT_EXCLUDE: empty (no compile blockers found).

## Latent-bug sweep

| Tool | Result |
|------|--------|
| ruff F821 (84 modules) | 0 issues |
| pyflakes (84 modules) | 20 style/unused warnings only; no undefined-name bugs |

No `fix(latent)` commit required.

## Compile (Windows MSVC, Cython 3.2.8)

| Item | Value |
|------|-------|
| Modules built | 84 |
| .pyd in src_py | 84 |
| Build wall | ~640 s |
| Pinned flags | annotation_typing=False, Options.docstrings=False, language_level=3 |
| Log | tmp/cython_release/build.log, compile_stdout.log |

## Per-module import smoke

84/84 PASS (all loaded from `.pyd`). Log: tmp/cython_release/smoke_imports.log

## MP spawn verification (Windows spawn)

PASS: worker imported compiled `comp_selection_per_target` and `photometry_core`.
Log: tmp/cython_release/mp_spawn.log

Evidence:
- comp_file: src_py/comp_selection_per_target.cp312-win_amd64.pyd
- pc_file: src_py/photometry_core.cp312-win_amd64.pyd

## pytest delta (compiled vs interpreted)

| Mode | Passed | Skipped | Failed |
|------|--------|---------|--------|
| Interpreted (--fast) | 1126 | 18 | 0 |
| Compiled (PYTHONPATH=src_py) | 1115 | 29 | 0 |

Conditional skips under compiled (+5 vs interpreted baseline skips):

| Test | Module | Reason |
|------|--------|--------|
| test_no_fixed_3_in_dilution_resolver_source | photometry_core | inspect.getsource |
| test_comp_exclusion_does_not_use_exoplanet_columns | comp_selection_per_target | inspect.getsource |
| test_obscode_warning_only_when_unset | export_reports | inspect.getsource |
| test_catalog_known_variable_not_includes_exoplanet | pipeline | source text scan |
| test_dataclass_defaults_match_post_init | config | source text scan |

`test_cython_mp_spawn` skips on interpreted path (no .pyd); runs under compiled build.

Log: tmp/cython_release/pytest_compiled2.log

## Science identity gates

### P1 golden compiled (VYVAR_INVARIANTS_P1=1)

7/7 PASS (byte-identity tests included). Wall ~370 s.
Log: tmp/cython_release/p1_golden_compiled.log

### --full compiled (draft_435 anchor)

| Check | Result |
|-------|--------|
| full-snapshot-sha-core | PASS 03d8fb6491bc3c22... n=333 |
| full-photometry-sha-extended | PASS bbfcc92e7ac5c4c5... n=499 |
| full-science-compare | PASS n_lc=166 failures=0 |
| OVERALL | PASS |

Pipeline wall (full-pipeline line): 2331 s
Total session wall (incl. pytest): 2973 s
Log: tmp/cython_release/full_compiled.log

### --fast interpreted (after `setup_cython.py clean`)

OVERALL: PASS (1126 passed, 18 skipped)
Log: tmp/cython_release/fast_interpreted2.log

## Performance

| Run | Wall (s) | Notes |
|-----|----------|-------|
| --full compiled (total session) | 2973 | this run |
| --full compiled (pipeline only) | 2331 | session_baseline full-pipeline line |
| Interpreted baseline range | 2113-2549 | prior runs per task brief |
| Spike P1 headless (3 modules) | 480 -> 404 (+15.8%) | rootcause result |

Full-night compiled pipeline (2331 s) falls within the interpreted baseline range;
total session overhead higher due to compiled pytest pass at start. No hard threshold
applied (report only).

## Protection artifacts

- `test_cython_annotation_typing.py`: full MODULE_LIST coverage + flag-drift guard
- `build_release._assert_pinned_flags()`: refuses annotation_typing/docstrings drift

## Gates summary

| Gate | Status |
|------|--------|
| ruff F821 latent sweep | PASS |
| Full compile + smoke 84/84 | PASS |
| MP spawn compiled | PASS |
| pytest compiled (zero new fails) | PASS |
| P1 golden compiled 7/7 | PASS |
| --full compiled anchor identity | PASS |
| --fast interpreted | PASS |

## STOP before push

Not pushed. Commits local only.

## Files changed

- dev/tools/cython_release/ (build_release.py, module_list.py, latent_sweep.py, smoke_imports.py, verify_mp.py, README.md, cython_build_release.bat)
- build/setup_cython.py (shim)
- build/README.md
- dev/tests/cython_compat.py, test_cython_mp_spawn.py, test_cython_annotation_typing.py
- dev/tests/test_gs11_dilution_aperture_g2_f003.py, test_exoplanet_local_match.py, test_export_citations.py, test_master_validity_days_g6_f002.py
- docs/VYVAR_DECISIONS.md, VYVAR_STATE.md, VYVAR_ROADMAP.md
- INSTALL.md
- dev/validation/VYVAR_VALIDATION_LEDGER.json

## Errors

None (all gates PASS).
