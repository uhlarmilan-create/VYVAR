CURSOR RESULT — 2026-07-08 (CLOSE-0708)

What I did
Reproduced draft_424 PDF HRD failure (read-only); logged HRD-PLOT-TUPLE in ROADMAP; day-close
JOURNAL + STATE; commit + push.

## HRD reproduce (draft_424 NoFilter_60_2)

**Traceback:**
```
IndexError: tuple index out of range
  File "hrd_analysis.py", line 241, in build_hrd_dataframe
    gdf = _fetch_gaia_columns_by_source_id(...)
  File "hrd_analysis.py", line 113, in _fetch_gaia_columns_by_source_id
    d = {k: row[k] for k in row}
            ~~~^^^
```

**Swallow site:** `photometry_report.py:4365` — `logging.warning("PDF HRD: build/plot failed (%s)")` then `return`.

**Panel status:** **missing** — HRD page never emitted (`c.showPage()` not reached); no
`hrd_field_summary.png`; PDF bytes lack `Field astrophysics` / Hertzsprung strings
(`VYVAR_report_NoFilter_60_2_inputguards_test.pdf`).

## Docs
- ROADMAP: **HRD-PLOT-TUPLE** (LOW) ? 299-defensive-cluster batch
- JOURNAL: 2026-07-08 day-close entry
- STATE: one-liner

## Commit
(pending push)
