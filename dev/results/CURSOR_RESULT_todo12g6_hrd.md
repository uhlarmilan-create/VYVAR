CURSOR RESULT — 2026-07-13

What I did
TODO-12g6: caption provenance stamps (UTC + git short hash), boost default 2.2, archived
overlapping `tmp/todo12_hrd/` runs, canonical re-render set in `tmp/colorfield_final/`.

## Output / findings

**Part 1 — caption stamps:** every PNG footer now ends with
`rendered YYYY-MM-DD HH:MM UTC @ {git_short}.` (`nogit` fallback). Tests cover suffix,
parseable timestamp, and hash format.

**Part 2 — default:** `hrd_color_chroma_boost` 1.6 ? **2.2** (PARAMS, schema, JOURNAL, ROADMAP).

**Part 3 — archive:** moved `tmp/todo12_hrd/` ? `tmp/todo12_hrd_archive_0711/` (nothing deleted).
Top-level inventory: `pre12b/`…`pre12g5/`, `run_0711/`, `run_0711_boost/`, draft424/425 PNGs,
`summary.json`, `summary_12g2.json`, A/B strips, `pre12g5_part0.json`, plus loose PNGs.

**Part 4 — canonical set** (`tmp/colorfield_final/manifest.json`):

| File | WP | G2 worst | pct_colored |
|------|-----|----------|-------------|
| `d424_NoFilter60_2_fm_b2.2_color.png` | field_median | **0.013** | 99.7% |
| `d424_NoFilter60_2_d65_b2.2_color.png` | d65 | 0.027 | 99.7% |
| `d425_V20_2_fm_b2.2_color.png` | field_median | 0.005 | 99.3% |
| `d425_V20_2_d65_b2.2_color.png` | d65 | 0.019 | 99.3% |

All G2 < 0.03. fm renders use **default config only** (boost 2.2 unoverridden). draft_424
bright-center region: no squarish blotches (G2 0.013 vs pre-fix ~0.072). Crops + G2 heatmaps
in subdirs.

**Tests:** 759 passed. `session_baseline_check --fast` PASS.

## Errors (if any)
None.

## Files changed
- `hrd_colorfield.py` — caption stamp, `_repo_short_git_hash`, default boost helper 2.2
- `config.py` — default boost 2.2
- `tests/test_hrd_colorfield.py` — caption stamp tests
- `scripts/todo12g6_hrd_canonical.py` — new
- `docs/VYVAR_PARAMS.md`, `docs/config_schema.md`, `docs/VYVAR_JOURNAL.md`, `docs/VYVAR_ROADMAP.md`
