CURSOR RESULT — 2026-07-08 (EXCEPT-FIX-1-CLOSEOUT)

What I did
Confirmed RN-HEADER-NONE footprint for aperture_px_planned drift; fixed EXC-0626 silent
empty-comp drop; harness schema guard; committed postmortem docs; validated draft_424; pushed.

## 1. aperture_px_planned attribution — CONFIRMED

Star `1496795041799526400` (R CVn): catalog mag lookup **7.12** ? SNR bin **7.0**.

| RN (e?) | mag 7.0 planned (px) |
|--------:|---------------------:|
| 7.6 (pre Item 4) | 3.918 |
| 15.2 (header-scaled) | 3.868 |

? = **?0.05 px** — matches `tmp/quickwins_0708/item4_report.json` mag-7.0 bin and baseline
`ecc4a2ea…` ? restored `3628f626…` single-cell drift. **Verdict (d): deterministic
RN-HEADER-NONE footprint (Item 4 `1830527`), not rerun noise.** Science columns identical;
no anchor ambiguity.

## 2. EXC-0626 fix

- `_phase2a_skip_empty_comps_target` — ERROR + summary stub + `phase2a_empty_comp_drop` counter
- `_require_comparison_stars_per_target_schema` — fail loudly on pool CSV (missing `target_catalog_id`)
- Harness: `sandbox/_except_fix1_validate_424.py` — per-target CSV only + schema check

## Validation

- `tests/test_except_fix_top10.py`: **7 passed** (+2 closeout tests)
- Full pytest: **595 passed**; ruff green
- draft_424 closeout rerun: **180 rows**, `byte_identical: true` (`3628f626…`), all
  `except_fix_summary` zeros including `phase2a_empty_comp_drop: 0`
- Artifact: `tmp/except_fix1_closeout_validate.json`

## Commits

- `91b421b` fix(photometry): EXC-0626 empty-comp drop + pool CSV schema guard
- `a992f85` docs: closeout postmortem + EXC-0626 census + CURSOR_RESULT

## Errors (if any)

None.

## Files changed

- `photometry_core.py`
- `except_fix_counters.py`
- `tests/test_except_fix_top10.py`
- `docs/VYVAR_EXCEPT_CENSUS.md`
- `docs/VYVAR_JOURNAL.md`
- `CURSOR_RESULT_except_fix1_closeout.md`
