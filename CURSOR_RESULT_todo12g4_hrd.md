CURSOR RESULT — 2026-07-13 10:30 UTC+2

What I did
Implemented TODO-12g4 chroma boost for catalog-color field: `apply_chroma_boost()` after white-point
and desaturation, before SNR gate; config `hrd_color_chroma_boost` (default 1.6, clamp 1..3);
caption suffix when boost > 1. Validation A/B at boosts 1.0/1.6/2.2 + d65@1.6.

## Implementation

- `rgb_boosted = 1 - (1 - rgb) * boost; clip; hue-preserving unit-max renorm`
- boost=1.0 returns input unchanged (12g2 regression guard)
- Caption: ` chroma enhanced x{N}.` appended when boost > 1 (both WP modes)

## Metric table (field_median WP)

### draft_425 V_20_2

| boost | % colored | mean R/B reddest | cluster R/B std | G2 bg neutrality | render (s) |
|-------|-----------|------------------|-----------------|------------------|------------|
| 1.0 | 99.31% | 1.255 | 0.204 | 0.0040 | 3.27 |
| 1.6 | 99.31% | 1.583 | 0.449 | 0.0039 | 2.36 |
| 2.2 | 99.31% | 2.466 | 2.143 | 0.0052 | 2.39 |

### draft_424 NoFilter_60_2

| boost | % colored | mean R/B reddest | cluster R/B std | G2 bg neutrality | render (s) |
|-------|-----------|------------------|-----------------|------------------|------------|
| 1.0 | 99.70% | 1.860 | 0.041 | 0.0068 | 1.23 |
| 1.6 | 99.70% | 3.880 | 0.052 | 0.0063 | 1.25 |
| 2.2 | 99.70% | 7.689 | 0.051 | 0.0088 | 1.25 |

**G2: PASS at all boosts** (all < 0.03). Mean R/B and cluster std rise monotonically with boost.

## Decision strips (`tmp/todo12_hrd/run_0711_boost/`)

- `draft425_V_strip_reddest.png`, `_strip_cluster_core.png`, `_strip_mid_field.png`
- `draft424_strip_reddest.png`, `_strip_cluster_core.png`, `_strip_mid_field.png`
- Full renders: `*_fm_boost{1.0,1.6,2.2}_field_color.png`, `*_d65_boost1.6_field_color.png`

## Tests / gates

- `tests/test_hrd_colorfield.py`: **15 passed** (+6 new boost tests)
- Full pytest: **752 passed**, 15 skipped
- `session_baseline_check --fast`: **PASS**

## Files changed

- `hrd_colorfield.py` (apply_chroma_boost, caption, wiring)
- `config.py` (+hrd_color_chroma_boost)
- `tests/test_hrd_colorfield.py`
- `scripts/todo12g4_hrd_validate.py` (new)
- `docs/config_schema.md`, `docs/VYVAR_PARAMS.md`, `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_JOURNAL.md`
- `CURSOR_RESULT_todo12g4_hrd.md`

Default 1.6 left open for Milan after strip review (one-line config change if different).

## Errors (if any)

None.
