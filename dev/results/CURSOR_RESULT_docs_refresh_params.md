CURSOR RESULT - 2026-07-22 16:55 UTC+2

What I did
Regenerated parameter handbook and FLOW PDF after PIPELINE-SIMPLIFY-1 param removals
(`skip_processed_directory`, `vsx_variable_targets_mag_limit`). Purged stale mentions from
both builders; bundled dangling push-record append in `CURSOR_RESULT_pipeline_simplify_1.md`
(ASCII fix for en-dash bytes). **STOP before push.**

## Builder edits

### `dev/tools/docs_pdf/build_parameter_handbook.py`
- Removed `D['vsx_variable_targets_mag_limit']` detail block.
- Verified `skip_processed_directory`: no entry (grep clean).
- State line: `Stav: 2026-07-22, VYVAR HEAD 2c520c6. Registrovanych parametru: 270.`

### `dev/tools/docs_pdf/build_flow_doc.py` (5 mentions)
| Line | Change |
|------|--------|
| 270 | Dropped `vsx_variable_targets_mag_limit=14.5` from param dump paragraph |
| 308 | VSX path (1): detection-limited (DAO+Gaia on MASTERSTAR) |
| 321 | Dropped mag-limit token; kept `exoplanet_match_max_sep_arcsec=3.0` |
| 651 | G_lim_90: depth diagnostic only; VSX scope automatic |
| 733 | Table row: `VSX scope: automaticky (DAO+Gaia detekce); parametr odstranen 2026-07` |

## PDF regeneration

```
python dev/tools/docs_pdf/build_parameter_handbook.py  -> built: params=251, detailed=82, deep boxes=13
python dev/tools/docs_pdf/build_flow_doc.py            -> ok
```

Outputs:
- `docs/VYVAR_PARAMETER_HANDBOOK_CZ.pdf` (84222 bytes)
- `docs/VYVAR_FLOW_CZ.pdf` (132469 bytes)

Double-build: both scripts exit 0 twice; PDF SHA256 differs run-to-run (ReportLab embeds
creation timestamp in metadata - content rebuild is stable, bytes are not byte-identical).

## Residual sweep (grep)

### Builders + gen output - CLEAN
- `dev/tools/docs_pdf/build_parameter_handbook.py` - no hits
- `dev/tools/docs_pdf/build_flow_doc.py` - no hits
- `docs/VYVAR_PARAMS.md` (gen_params_md output) - no hits

### `docs/` - reported hits (no edit; out of builder scope or intentional history)

| File:line | Token | Note |
|-----------|-------|------|
| `docs/VYVAR_DECISIONS.md:1135` | `skip_processed_directory` | Decision record (removal) - OK |
| `docs/VYVAR_DECISIONS.md:1153` | `vsx_variable_targets_mag_limit` | Decision record (removal) - OK |
| `docs/VYVAR_DECISIONS.md:1136` | `processed/lights` | Retired copy-tree note - OK |
| `docs/VYVAR_ROADMAP.md:882` | `vsx_variable_targets_mag_limit` | Stale "DONE" dropped item; superseded by VSX-AUTO-MAGLIM (a0e3431) - ROADMAP cleanup deferred |
| `docs/VYVAR_JOURNAL.md:3052-3084` | `skip_processed_directory`, `processed/` | Historical journal (2026-07) - OK |

## Gates

### `--fast`
```
OVERALL: PASS
pytest: 1069 passed, 24 skipped
```
(Initial run FAIL on ASCII policy due to 0x97 en-dash bytes in uncommitted push record;
fixed before commit.)

## Files changed (this commit)

- `dev/tools/docs_pdf/build_parameter_handbook.py`
- `dev/tools/docs_pdf/build_flow_doc.py`
- `docs/VYVAR_PARAMETER_HANDBOOK_CZ.pdf`
- `docs/VYVAR_FLOW_CZ.pdf`
- `dev/results/CURSOR_RESULT_pipeline_simplify_1.md` (bundled push record, ASCII)
- `dev/results/CURSOR_RESULT_docs_refresh_params.md` (this file)

## STOP before push

Single commit ready locally; not pushed.
