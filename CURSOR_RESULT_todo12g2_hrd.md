CURSOR RESULT  2026-07-10 21:20 UTC+2

What I did
Implemented TODO-12g2 catalog-color field polish: hue-preserving highlights (soft Reinhard +
scale-by-max), chroma SNR gate, field-median von Kries white point, saturation default 0.85.
Validation A/B vs 12g pre12g2 archive with hard gates G1-G4.

## Output / findings

### Defect fixes
1. **Highlight:** `compose_catalog_color_rgb` -- soft L/(1+L) then hue-preserving scale when max>1
   (no per-channel clip). Config `hrd_color_highlight_mode` (soft|scale, default soft).
2. **Chroma SNR gate:** w = s/(s+snr), s = L/sigma_bg; blend chroma to white. Config
   `hrd_color_chroma_snr` (default 3.0; 0=12g off).
3. **White point:** `field_median` von Kries divide before desaturation; caption extended with
   median Teff. Config `hrd_color_white_point` (field_median|d65).
4. **Saturation default:** 0.7 -> 0.85.

### Hard gates (draft425_B + draft424)

| Gate | Metric | Value | Pass |
|------|--------|-------|------|
| G1 RS Per mean R/B | draft425_B | **1.330** | PASS (>1.3) |
| G2 background \|R-B\|/L | draft425_B | **0.0042** | PASS (<0.03) |
| G2 background \|R-B\|/L | draft424 | **0.0066** | PASS |
| G3 cluster R/B std | draft425_B | **0.167** | PASS (>0.05) |
| G4 background R/B spot-check | both | **1.0, 1.0, 1.0** | PASS |

**overall_pass: true**

### Render stats (12g2 defaults)

| Setup | % colored | render (s) | BP-RP range |
|-------|-----------|------------|-------------|
| draft424 | 99.7% | 1.19 | -0.14 .. 5.68 |
| draft425_B | 98.9% | 2.55 | -0.31 .. 3.65 |

### A/B artifacts
- Archive: `tmp/todo12_hrd/pre12g2/` (12g PNGs)
- Side-by-side: `draft425_B_ab_rs_per.png`, `_ab_cluster_core.png`, `_ab_background.png`;
  `draft424_ab_red_giant.png`, `_ab_background.png`
- Summary: `tmp/todo12_hrd/summary_12g2.json`

### Tests / gates
- `tests/test_hrd_colorfield.py`: 9 passed (+3 new: hue scale, SNR gate, field_median WP)
- Full pytest: **746 passed**, 15 skipped
- `session_baseline_check --fast`: **PASS**

## Errors (if any)
None.

## Files changed
- `hrd_colorfield.py` (polish pipeline)
- `config.py` (+3 keys, saturation default)
- `tests/test_hrd_colorfield.py`
- `scripts/todo12g2_hrd_validate.py` (new)
- `docs/config_schema.md`, `docs/VYVAR_PARAMS.md`, `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_JOURNAL.md`
- `CURSOR_RESULT_todo12g2_hrd.md`
