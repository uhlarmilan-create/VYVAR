CURSOR RESULT - 2026-07-13

What I did
Implemented TODO-12g5: local-background chroma SNR gate (`hrd_color_bg_box_px`), tapered
Gaussian splat stamp, star-masked bg-map construction, hardened 8x6 G2 grid + heatmaps,
tests, validation script, and docs. Confirmed Part 0 blind spot on pre-fix draft_424 @2.2.

## Output / findings

**Part 0 (pre-fix `pre12g5/draft424_fm_boost2.2`):** worst-patch |R-B|/L = **0.072** at
**(130, 1280)** (frame center); corner patch ? **0.015** - single-corner G2 blind spot confirmed.
? `tmp/todo12_hrd/pre12g5_part0.json`

**Post-fix validation (`summary_12g5.json`):**

| Render | G2 worst | Cluster R/B std | Pre std | Pass |
|--------|----------|-----------------|---------|------|
| draft424 boost 1.6 | 0.0090 | - | - | G2 ? |
| draft424 boost 2.2 | 0.0130 | - | - | G2 ? (was ~0.072) |
| draft425_V boost 1.6 | 0.0034 | 0.400 | 0.449 | G2 ? |
| draft425_V boost 2.2 | 0.0051 | **1.946** | 2.143 | G2 ?, cluster **90.8%** ? |

All four G2 worst-patch < 0.03. draft_425 V @ boost 2.2 cluster std within ?10%.
A/B strips + G2 heatmaps: `tmp/todo12_hrd/run_0711_boost/`. Pre-fix archive: `pre12g5/`.

**Tests:** 758 passed (21 `test_hrd_colorfield`). `session_baseline_check --fast` PASS.

## Errors (if any)
None.

## Files changed
- `hrd_colorfield.py` - local bg maps, tapered stamp, hardened G2, star-masked bg build
- `config.py` - `hrd_color_bg_box_px`
- `tests/test_hrd_colorfield.py` - 6 new tests
- `scripts/todo12g5_hrd_validate.py` - new
- `docs/VYVAR_PARAMS.md`, `docs/config_schema.md`, `docs/VYVAR_JOURNAL.md`, `docs/VYVAR_ROADMAP.md`
