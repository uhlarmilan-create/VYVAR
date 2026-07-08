CURSOR RESULT ù 2026-07-08 (QUICKWINS-0708)

What I did
Executed five-step batch: determinism evidence (item 0), four ledger fixes (items 1ù4,
science-affecting last), pytest+ruff gates, draft_424 validation, docs close, push.

## Output / findings

### Item 0 ù draft_425 B determinism (evidence only, no commit)
- Reran `run_phase2a` on `draft_000425` `B_20_2` at HEAD `21c20e3`
- Science-column compare (provenance excluded): **PASS** ù 0 diffs / 363 LCs
- Evidence: `tmp/quickwins_0708/item0_determinism.json`, `item0_pre_science.json`

### Item 1 ù K2-SLOPE-TRACE (`8c44b71`)
- `SLOPE_GR_PER_BPRP`: 0.859 ? **1.054** (Jordi 2010 Table 6 inverse, FGK g-r=0.48)
- k2_g ? ?0.0169, k2_r ? ?0.0042; spec ù3 illustrative values updated
- UG: **1.091 retained** ù explicit exception comment; ledger **K2-SLOPE-UG** FUTURE
- Analytic 427 (no rerun, extreme-colour targets from comparison pools):
  - g_60_2 max |?mag_calib| ? **12.3 mmag**
  - r_60_2 max |?mag_calib| ? **3.2 mmag**
- Tests: `tests/test_k2_extinction.py` 14 passed

### Item 2 ù PROC-MAG-NAMING (`0913665`, documented not renamed)
- `pipeline.py` `_vyvar_df_to_csv` docstring: `mag` = Gaia catalog G
- `docs/VYVAR_PIPELINE_CZ.md` proc CSV schema subsection (Czech)
- `docs/VYVAR_PROCESS.md` dao_flux rule verified present

### Item 3 ù CAL-PASSTHRU-DEAD (`21c20e3`)
- Caller audit: **no** production or test callers of `allow_passthrough=True`
- Removed parameter + synthetic zero/one master branch from `get_processed_master`

### Gate (items 3?4)
- `567 passed` pytest + ruff BLE001/E722 ù **PASS**

### Item 4 ù RN-HEADER-NONE (`1830527`)
**Fix:** `precompute_and_save_snr_aperture_table_for_draft` passes `_snr_header` to
`resolve_read_noise` (parity with Phase 2A `:6661`).

1. **Unit test:** `tests/test_snr_table_rn_header.py` ù bin2 header ? RN=2.6 in JSON ù **PASS**
2. **SNR table (draft_424 bin2, equipment 1):**
   - Old (bug, RN=7.6): vs new (fix, RN=15.2)
   - Max aperture shift: **2.2%** (mag 13.0: 2.268?2.218 px, ?0.05 px)
   - Bright bins unchanged; faint bins shrink ù **correct physics**
   - STOP rules: not triggered
3. **Snapshot:** `Archive/Drafts/draft_000424_snapshot_20260708` + manifest + 5 raw-light SHA-256 samples
4. **Phase 2A rerun:** 178 LCs compared vs snapshot
   - median/max |?mag_calib|: **0.0** (byte-identical ù Phase 2A already used header RN)
   - median lc_rms before/after: **0.0863** / **0.0863** (no degradation)
   - Raw checksum drift: **none**
5. **New 424 baseline anchor** ù provenance block hash (excl. `stamped_at`):
   `e1a7a311b02c81a5bf602080b345ac95d8ba351327c2f63edd5ca185ff29e80f`
- Evidence: `tmp/quickwins_0708/item4_report.json`, `snr_aperture_delta.json`

### Final gate
- `568 passed`, 15 skipped; ruff BLE001/E722 ù **PASS**

## Errors (if any)
None. Batch **PASS** ù no STOP rules triggered.

## Files changed
| Item | Commit | Files |
|------|--------|-------|
| 0 | ù | evidence only (`tmp/quickwins_0708/`) |
| 1 | `8c44b71` | `k2_extinction.py`, `docs/VYVAR_K2_DESIGN_SPEC.md` |
| 2 | `0913665` | `pipeline.py`, `docs/VYVAR_PIPELINE_CZ.md` |
| 3 | `21c20e3` | `calibration.py` |
| 4 | `1830527` | `photometry_core.py`, `tests/test_snr_table_rn_header.py` |
| close | `d2c2c4f` | `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_STATE.md`, `docs/VYVAR_JOURNAL.md`, `CURSOR_RESULT_quickwins_0708.md` |
