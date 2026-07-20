CURSOR RESULT - 2026-07-19

What I did
Three-part small-fixes batch on `main`: APCORR-MIXEDFRAME all-or-nothing COG night gate;
AUTO-VSX-LIMIT report-layer depth check; close stale TODO-COMP-P2P-RESIDUAL. Three commits,
not pushed. FLOW builder wording (ch 11.5 + 13.5) updated once and PDF regenerated.

## Output / findings

### Part 1 - APCORR-MIXEDFRAME
- Night gate: `evaluate_cog_night_apcorr_gate` in `photometry_core.py`; if any science frame
  lacks usable `cog_ok`, `use_apcorr_flux=False` for the whole night.
- Provenance: `cog_night_fallback` (+ counts) in `photometry/pipeline_meta.json`.
- Log: `[APCORR] COG night fallback: N/M frames without cog_ok -> whole night uses standard AC`.
- `cog_aperture_correction_enabled` remains default **False**.
- Unit tests: `dev/tests/test_apcorr_mixedframe_night_gate.py` (5 passed).
- **`--full` BYTE-IDENTICAL vs VL-ANCHOR-WCSINV: OVERALL PASS**
  - `full-science-compare` n_lc=166 failures=0
  - `full-photometry-sha-core` `3d26f4692ac81fc5...` n=333
  - `full-photometry-sha-extended` `6420f1daa53a0d5d...` n=499
  - pipeline 2205 s; pytest 982 passed / 24 skipped
- Commit 1: `b0575c9` `fix(apcorr): all-or-nothing COG per night - no mixed-frame LC step`

#### Docs impact (Part 1)
- **DECISIONS:** new entry `APCORR-MIXEDFRAME-ALLORNOTHING` (2026-07-19); COG section updated.
- **ROADMAP:** APCORR-MIXEDFRAME ? DONE (+ FUTURE per-frame `fallback_ee` note).
- **STATE:** one-line closure under Current snapshot.
- **FLOW:** ch 11.5 / quick-ref COG wording updated in builder (regenerated with Part 2).
- **facts:** no `flow_doc_facts.py` key changes.

### Part 2 - AUTO-VSX-LIMIT
- Pure compare: `vsx_limit_vs_depth_status(limit, g_lim_90, snr5)` - WARN if
  `limit > min(depths) + 0.3`.
- Depth load: `load_field_depth_metrics` (pipeline_meta `g_lim_90`;
  `crowding_index.json` SNR5; optional in-memory compute; never writes science files).
- Report: resolved-facts + config page line
  `VSX limit: - | field depth: G_lim_90=-, SNR5=-`; title-page WARN badge when warranted.
- No new config keys; Phase 0 selection untouched.
- Unit/smoke: `dev/tests/test_auto_vsx_limit_report.py` (+ wave_a row-count update).
- **P1 golden (`VYVAR_INVARIANTS_P1=1`): 5 passed** in 482 s (SHAs match VL-P1-GOLD).
- Commit 2: `80e0e66` `feat(report): VSX limit vs measured field depth check (AUTO-VSX-LIMIT, report layer)`

#### Docs impact (Part 2)
- **ROADMAP:** GAPS A1 / AUTO-VSX-LIMIT ? DONE (report layer; config automation FUTURE).
- **FLOW_DOC_V3_GAPS.md:** A1 marked DONE (report layer).
- **FLOW:** ch 13.5 sentence updated (report compares limit vs depth; full automation FUTURE).
- **facts:** none.
- **STATE / DECISIONS:** not required for this report-only part.

### Part 3 - TODO-COMP-P2P-RESIDUAL (docs-only)
- Caller audit: **no** caller passes `common_mode_detrend=False`.
  Production: `photometry_core.py` (~8989), `check_star_kmag.py` (~1036),
  `method_lc_output.py` (~118) all `True`. Tracked `dev/scripts/*` same.
- Evidence: docstring at `check_comparison_stability` (~2900-2903); residual via
  `_comp_lc_frame_ensemble_residual`; CM via `_common_mode_detrend_comp_lc` when
  `comp_bjd` present; flag `common_mode_detrend_applied` (~2917).
- ROADMAP MEDIUM item ? **DONE (already implemented; found stale 2026-07-19)**.
- No code changes.

#### Docs impact (Part 3)
- **ROADMAP:** TODO-COMP-P2P-RESIDUAL closed with evidence pointers.
- **STATE / DECISIONS / FLOW:** none (bookkeeping correction; FLOW never claimed raw-LC p2p).

### FLOW PDF (once, Parts 1+2 wording)
- Builder: `dev/tools/docs_pdf/build_flow_doc.py` (ch 11.5 COG; quick-ref COG; ch 13.5 VSX depth).
- Regenerated: `docs/VYVAR_FLOW_CZ.pdf`.
- `test_docs_sync_guard` run as part of wrap-up.

## Errors (if any)
None blocking. (Benign EXC-0030 Gaia BP-RP Row.get noise during `--full` Phase 2A; pre-existing.)

## Files changed
- Commit 1 `b0575c9`: `src_py/photometry_core.py`,
  `dev/tests/test_apcorr_mixedframe_night_gate.py`,
  `docs/VYVAR_{ROADMAP,DECISIONS,STATE}.md`,
  `dev/validation/VYVAR_VALIDATION_LEDGER.json` (auto-stamp from `--full`)
- Commit 2 `80e0e66`: `src_py/photometry_report.py`,
  `dev/tests/test_auto_vsx_limit_report.py`,
  `dev/tests/test_wave_a_report_config.py`,
  `docs/VYVAR_ROADMAP.md`, `dev/results/FLOW_DOC_V3_GAPS.md`
- Commit 3: `docs(roadmap): close TODO-COMP-P2P-RESIDUAL as stale - CM-detrended p2p already in production`
  (`docs/VYVAR_ROADMAP.md`, `dev/tools/docs_pdf/build_flow_doc.py`,
  `docs/VYVAR_FLOW_CZ.pdf`, this RESULT)

Not pushed (Milan authorizes pushes). Stack tip: `git log -3 --oneline`.
