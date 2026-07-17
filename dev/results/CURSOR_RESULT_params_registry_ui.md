# CURSOR RESULT - PARAMS-REGISTRY-UI

Date: 2026-07-17
Arc: UI block, wave 1 (parameters)
Baseline: origin/main `e782786` (Anchor #3 ACTIVE, draft_000435 sky-surface snapshot)

## What I did

Turned config <-> UI parity into a tested property, then built the tiered
Parameters dashboard and the PDF Configuration page on top of it. Four steps,
one commit each, full pytest green after every commit. No science-path changes
(all diffs are registry data, tests, doc generator, Streamlit UI, PDF layout).

## Commits (one per step)

| Step | Hash | Summary |
|------|------|---------|
| 1 | `548e7ae` | machine-readable params registry + parity guard test |
| 2 | `addb05a` | generated VYVAR_PARAMS.md + freshness test |
| 3 | `547008a` | tiered Parameters dashboard generated from registry |
| 4 | `e37826d` | PDF Configuration page in SUMMARY MEASURE REPORT |

Not pushed (Milan authorizes pushes).

### git diff --stat per commit

STEP 1 (`548e7ae`):
```
 params_registry.py              |  163 ++
 tests/test_params_registry.py   |  102 ++
 validation/params_registry.json | 3348 ++++++++++++++++++++++++++++++++++++++
 3 files changed, 3613 insertions(+)
```

STEP 2 (`addb05a`):
```
 docs/VYVAR_PARAMS.md          | 930 +++++++++++++++++-----------------
 tests/test_params_registry.py |  40 ++
 tools/gen_params_md.py        | 173 ++++++++
 3 files changed, 598 insertions(+), 545 deletions(-)
```

STEP 3 (`547008a`):
```
 tests/test_ui_params_dashboard.py |  53 +++++
 ui_params_dashboard.py            | 265 +++++++++++++++++++++++++++++
 ui_settings.py                    |  45 +++-
 3 files changed, 362 insertions(+), 1 deletion(-)
```

STEP 4 (`e37826d`):
```
 photometry_report.py             | 179 +++++++++++++++++++++++++
 tests/test_report_config_page.py |  74 ++++++++++
 2 files changed, 253 insertions(+)
```

No science module (calibration, detection, alignment, photometry, comp
selection, trust, extinction numerics, export numerics) appears in any diff.
`config.py` is unchanged (`git diff --stat config.py` empty).

## Registry summary (validation/params_registry.json)

304 entries (one per public AppConfig field; underscore-prefixed and
non-field class attributes skipped).

By tier:
- basic: 12
- advanced: 81
- expert: 211

By kind:
- static: 284
- resolved: 20
- derived: 0

By widget:
- auto: 128
- custom: 162
- hidden: 14

By phase:
- detection 90, photometry 75, comp_selection 45, qc 19, reports 17,
  calibration 11, paths 11, trust 11, observer 9, extinction 8,
  alignment 3, system 3, export 2

All tier/kind assignments are PROPOSED and reviewable via the regenerated
`docs/VYVAR_PARAMS.md`.

## resolved-key list with justifications

20 keys seeded `kind=resolved` (runtime value can be auto-derived/overridden
by the pipeline; rendered read-only "computed" in the dashboard). Override
sites grepped:

Density / crowding overrides (`config.py:2443 DENSITY_OVERRIDES`,
`CROWDING_LOOSEN_OVERRIDES`, `CROWDING_TIGHTEN_OVERRIDES`, applied by
`apply_density_overrides()` @ `config.py:2501`):
- `phase01_comparison_max_mag_diff` - density/crowding delta target.
- `phase01_comparison_n_comp_min` - density/crowding delta target.
- `phase01_comparison_max_dist_deg` - density delta (added to FOV-derived max_dist_deg at runtime).
- `phase01_comparison_min_dist_arcsec` - dense/tighten delta target.
- `phase01_comparison_max_comp_rms` - dense/tighten delta target.
- `comp_max_delta_bprp` - density + crowding delta target (sums additively).
- `annulus_inner_fwhm` - dense/tighten delta target.

Aperture sizing (SNR-optimal aperture table; configured factors are fallback
only, effective aperture resolved per-star at runtime):
- `aperture_fwhm_factor`
- `aperture_fwhm_factor_small`
- `aperture_fwhm_factor_medium`
- `aperture_fwhm_factor_large`
- `annulus_outer_fwhm`

Plate scale / FOV (resolved from the solved WCS at platesolve time):
- `plate_scale_arcsec_per_px`
- `phase01_plate_scale_arcsec_per_px`
- `plate_solve_fov_deg`

Optics / detector autodetect (resolved from FITS headers / equipment config):
- `gain`
- `read_noise`
- `frame_width_px`
- `frame_height_px`

Parallelism:
- `qc_preprocess_workers` - recomputed in `AppConfig.__post_init__`
  (`config.py:1037`, env override `VYVAR_PARALLEL_WORKERS`); config default not honored verbatim.

No `derived` keys were needed (no field is never user-honored / always
computed); count is 0 as the task anticipated.

## Dispositions for the 6 stale registry-only keys

None of the six exist as AppConfig fields today (`grep` in `config.py` -> no
match). They were per-star pipeline OUTPUT columns (photometry_summary.csv)
or a private module constant that never belonged in a config registry. All
six are dropped (no registry entry):

| Stale key | Disposition |
|-----------|-------------|
| `aperture_px` | Removed. Per-star measured aperture, an output column consumed by `photometry_report.py:508` (`summary_df["aperture_px"]`). Never an AppConfig field. |
| `aperture_px_planned` | Removed. Legacy planned-aperture diagnostic output column; no longer produced and no live code reference. |
| `lc_rms` | Removed. Per-star light-curve RMS output column (photometry_summary.csv), read across report/export/QC (e.g. `photometry_report.py:492`, `night_run.py:496`). Not config. |
| `lc_rms_ooe` | Removed. Out-of-eclipse RMS output-column variant; no longer produced and no live reference. |
| `n_stability_good` | Removed. Per-star stability-count output metric; no longer produced and no live reference. |
| `_APERTURE_SIZING_MAG_COLS` | Removed. Underscore-prefixed private module constant (SNR aperture-sizing mag-column list), excluded by the "skip underscore-prefixed / non-field" rule. |

## Guard-test failure demonstration (STEP 1 acceptance)

Temporarily added `fake_registry_probe: int = 0` to `AppConfig` and ran the
guard test:

```
E   AssertionError: AppConfig fields with no registry entry: ['fake_registry_probe']
E   assert not ['fake_registry_probe']
tests\test_params_registry.py:38: AssertionError
FAILED tests/test_params_registry.py::test_registry_covers_every_public_field_exactly_once
1 failed, 6 passed in 0.19s
```

The failure names the offending key, as required. Probe reverted; `config.py`
diff is now empty and the guard suite is green again (`7 passed`).

## STEP 2 freshness

`tools/gen_params_md.py` regenerates `docs/VYVAR_PARAMS.md` from
`params_registry.json` + `dataclasses.fields(AppConfig)` (defaults/types come
from code, not the registry). Header carries generation timestamp + git HEAD
(marked volatile). `test_generated_params_md_is_fresh` regenerates in a temp
dir and asserts byte-identity with the committed file after stripping the
volatile header lines. Green. Old historical narrative sections dropped
(history lives in JOURNAL/DECISIONS).

## STEP 3 dashboard

`ui_params_dashboard.py` mounted as the FIRST tab of the Settings dashboard
in `ui_settings.py` (existing hand-built tabs untouched):
- BASIC: flat list of `tier=basic` + `widget=auto`.
- ADVANCED: one collapsed `st.expander` per phase, `tier=advanced` + `widget=auto`.
- EXPERT: single collapsed section with a science-impact warning, `tier=expert` + `widget=auto`, grouped by phase.
- `widget=custom` never auto-rendered (existing widgets remain authoritative); `widget=hidden` never rendered.
- `kind=resolved` rendered read-only "computed" (configured value + resolved runtime value from latest run provenance when available).
- Widget type inferred from field type + range (checkbox / number_input with clamps / selectbox / text_input).
- Deviation markers per parameter, "Reset to default" per parameter and per section, and a global "N parameters modified" counter (computed over the FULL config vs dataclass defaults, including custom/hidden keys) at the top of the dashboard.

Smoke test `tests/test_ui_params_dashboard.py` exercises the pure
registry->widget-kind resolution for every `widget=auto` key (total coverage,
no exceptions) without a Streamlit runtime. Green.

## STEP 4 PDF Configuration page

`photometry_report.py`: added `config_deviation_model()` + a Configuration
page wired into `build_pdf()`. Deviations only (snapshot value != current
dataclass default), columns key / run value / default. Header block:
git_hash, git_dirty_code, stamped_at_utc, entry_point, N modified, and a
fingerprint line (preprocess_sky_surface_order, seed policy, density profile).
Unknown/legacy snapshot keys listed separately (never crash the page). Footer
points to pipeline_meta.json. Fallback to live config with the visible warning
"provenance snapshot missing - showing live config, not run config", plus the
required caveat line about defaults being evaluated against current code.

Report-only verification on draft_000435 / NoFilter_60_2 (no photometry rerun):
- Configuration page rendered on page 186 of 187, 0 overflow violations.
- Source: run snapshot (fallback=False), git_hash 10d610c..., entry_point run_phase2a.
- 10 deviations from defaults, 0 unknown keys. Deviations: observer_name,
  observer_code, observer_location_name; blind_index_fine_path,
  blind_index_wide_path, gaia_db_path, vsx_local_db_path (path config);
  comp_iterative_clip_enabled=True, comp_trust_min_comps=3, sigma_sys_mag={'4':0.018}.

`tests/test_report_config_page.py`: deviation table from a synthetic snapshot
(2 modified keys + 1 unknown legacy key `aperture_px` -> listed under unknown
keys without crashing) and the missing/empty-snapshot fallback path. Green.

## Full-suite status

`python -m pytest -q` -> 902 passed, 19 skipped after the STEP 4 commit.
Each step commit was made with the full suite green.

## Deviations from the task

- STEP 1 seeding produced `custom=162` (larger than one might expect) because
  the k2 / masterstar / psf / alignment / HRD / catalog composite families are
  broad; these keep their authoritative hand-built widgets and are simply not
  auto-rendered in the new dashboard. No behavior change, purely a rendering
  routing decision consistent with the `custom` rule.
- No other deviations. All hard constraints honored: no science-path changes,
  anchor/`--full` machinery untouched, ASCII-only English, separate commit per
  step with pytest green, and no push.
