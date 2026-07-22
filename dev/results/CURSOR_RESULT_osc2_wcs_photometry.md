CURSOR RESULT - 2026-07-22 20:00 UTC+2

What I did
Implemented OSC-2 (phase 2/3): unified QC verdict replication from oneRGGB, WCS
solve-once + registration handoff to R/G/B, per-channel MASTERSTAR photometry path,
band tokens (TR/TG/TB/CLEAR), OSC-02 wired gate, tests, and docs. **STOP before push.**

## Implementation summary

### New module `src_py/osc_align.py`
- QC D1: `replicate_qc_verdict_from_one_rggb` (status from oneRGGB -> R/G/B, `qc_source=oneRGGB`)
- `merge_osc_qc_metrics_at_lights_root` for alignment QC-01 join
- Job orchestration: `partition_jobs_for_osc_alignment` (oneRGGB first; fail-closed without sibling)
- Artifacts: `osc_registration_handoff.json` (D2), `osc_wcs_propagation.json` (D3 meta + match_rate)
- `propagate_wcs_between_fits`, `apply_registration_handoff_to_frame`
- Band tokens: `obs_group_band_token` (R->TR, G->TG, B->TB, oneRGGB->CLEAR)
- Match-rate verification helper (60% WARN threshold; FAIL only if oneRGGB failed)

### Pipeline integration (`pipeline.py`)
- Extraction hook: QC replicate + root `qc_metrics.csv` merge after per-channel QC
- `_run_osc_multi_group_alignment`: oneRGGB full path + handoff write; R/G/B WCS propagate + handoff align
- `_astrometry_align_impl_body`: `osc_registration_handoff` apply path; handoff capture on oneRGGB
- `astrometry_align_and_build_masterstar`: OSC bundle partition + OSC-02 before orchestration

### Alignment frame (`vyvar_alignment_frame.py`)
- Registration handoff capture on successful align (astroalign matrix, wcs_reproject, phase/wcs shift)

### Policy / invariants
- `invariants_runtime.py`: **OSC-02 [wired]** `check_osc02_unified_frame_sets`
- `k2_extinction.filter_token_from_obs_group` -> OSC band tokens via `obs_group_band_token`

## Propagation artifacts

| Artifact | Location | Content |
|----------|----------|---------|
| `osc_registration_handoff.json` | `platesolve/<base>_oneRGGB/` | `reference_file`, per-frame `method` + transform params |
| `osc_wcs_propagation.json` | `platesolve/<base>_R|G|B/` | donor dir, channel, post-propagate match_rate |

## Per-channel match rates

- **Synthetic:** handoff + WCS tests in `dev/tests/test_osc2_wcs_photometry.py` (no end-to-end platesolve harness)
- **M71:** not run on this machine (data-limited; deferred to Milan eq id=5)

## Anchor protection enumeration

| Shared function | OSC guard | Mono path |
|-----------------|-----------|-----------|
| `_qc_enrich_calibrated_in_place` | Unchanged; per-channel sky-surface when `VY_CHANNEL` | Unchanged |
| `run_osc_channel_extraction_for_archive` | + QC replicate + root merge (OSC only) | Skipped when mono equipment |
| `astrometry_align_and_build_masterstar` | OSC bundle partition + `_run_osc_multi_group_alignment` when channel jobs detected | Existing single/multi job loops unchanged |
| `_astrometry_align_impl_body` | Handoff apply when `osc_registration_handoff` set; handoff write when `osc_write_registration_handoff` | Default MP alignment path unchanged |
| `_alignment_run_astroalign_points` | + optional handoff matrix in return (additive) | Same call signature (+3rd return ignored by mono callers) |
| `generate_masterstar_and_catalog` | Channel: build local MS + skip-solve after WCS copy | Unchanged |
| `filter_token_from_obs_group` | OSC suffix -> TR/TG/TB/CLEAR | `obs_group_first_token` fallback |
| `check_osc02_*` / `check_osc01_*` | Wired at alignment orchestration | N/A when no OSC bundles |

## Gates

### pytest + ruff
- `1097 passed, 24 skipped`
- `ruff check` on touched files - clean

### `--fast`
```
OVERALL: PASS (1097 passed, 24 skipped)
```

### `--full` draft_435 (mono anchor)
```
full-snapshot-sha-core       PASS   03d8fb6491bc3c22... n=333
full-photometry-sha-core     PASS   03d8fb6491bc3c22... n=333
full-photometry-sha-extended PASS   bbfcc92e7ac5c4c5... n=499
full-science-compare         PASS   n_lc=166 failures=0
OVERALL: PASS (~2169s)
```
Mono anchor byte-identical preserved.

## STOP before push

Uncommitted working tree. Await Milan authorization + M71 E2E on eq id=5.

## Pre-push verification (2026-07-22 21:35 UTC+2)

### Task item 4 - production-path test

**Name:** `test_production_path_howell_variance_uses_vy_egain_rdnois`

**Path:** Phase-2A helpers inside `run_full_photometry_pipeline`:
`precompute_and_save_snr_aperture_table_for_draft` + `read_flux_from_csv` ->
`_photometric_error_with_bkg_mode` / `_howell_variance_adu2`.

**Assertion targets:**
- OSC channel MASTERSTAR: `resolve_gain` -> `VY_EGAIN` effective value (`g_eff`);
  header-only `resolve_read_noise` -> `VY_RDNOIS` effective value (`rn_eff`);
  SNR table gain matches `g_eff`; `read_flux_from_csv` `err` matches Howell model at
  `(gain=g_eff, read_noise=rn_eff)`.
- Mono MASTERSTAR: SNR table gain from header `EGAIN`; read_noise from EQUIPMENTS DB
  (equipment_id=2, ~1.3 e-); channel variance > mono (effective gain differs).

### Task item 7 - per-channel comp selection smoke

**Name:** `test_osc_comp_selection_smoke_per_channel` (parametrized: oneRGGB, R, G, B)

Synthetic dense field (80 stars, 20 frames); `select_comparison_stars_per_target` returns
non-empty comp set (>=3) with expected band token per channel obs-group suffix.

### Match-rate threshold source

`vyvar_platesolver.MASTERSTAR_PLATESOLVE_MIN_MATCH_RATE` (0.60) - same gate as
`generate_masterstar_and_catalog` platesolve reject in `pipeline.py`. OSC-2 match-rate
logging imports this constant (no new config parameter). See DECISIONS OSC-WCS-SOLVE-ONCE.

### Gates (pre-push re-run)

```
pytest dev/tests: 1100 passed, 24 skipped
```

## Push (2026-07-22, Milan authorized)

### Pre-push checks

| Check | Result |
|-------|--------|
| `git fetch origin`; `origin/main` before push | `9cfcefc` |
| Stack `git log origin/main..HEAD --oneline` | 2 commits - exact match (see below) |
| `git status --short` | Clean; allowlisted untracked only (`dy_peg_night_run_bvr.py`, `qatar8_night_run_v.py`) |
| `session_baseline_check.py --fast` (final HEAD `d02d512`) | **OVERALL PASS** - 1100 passed, 24 skipped |

### Commit inventory (`origin/main..HEAD`, newest first)

```
d02d512 docs(osc): OSC-2 invariants, decisions, state, gate record
224c442 feat(osc): WCS solve-once, registration handoff, unified QC, OSC-02 (phase 2)
```

Base: `9cfcefc` -> stack tip: `d02d512` (`git push origin main` succeeded).

### Final origin/main tip

Local HEAD matches `origin/main` at `d02d512` after push. For current tip: `git rev-parse origin/main`.
