CURSOR RESULT - 2026-07-22 18:10 UTC+2

What I did
Implemented OSC-1 (phase 1/3): BAYERMASK DB authority, CFA-domain calibration path,
plane-split channel extraction (oneRGGB/R/G/B), effective gain/RN model, per-channel
QC+sky-surface, import cross-check guards, OSC-01 alignment gate, tests, and docs.
**STOP before push.** `--full` anchor gate started in background (see Gates).

## Implementation summary

### New module `src_py/osc_extract.py`
- Plane split for RGGB/BGGR/GBRG/GRBG (no interpolation).
- Channels: `oneRGGB` = (R+G1+G2+B)/4, `G` = (G1+G2)/2, `R`/`B` direct planes.
- `osc_channel_binning` NxN AVERAGE post-extraction.
- Effective noise: `gain_eff = gain_raw * plane_count * bin^2`, `rn_eff = rn_raw / sqrt(...)`.
- Headers: `VY_CHANNEL`, `VY_BAYERMASK`, `VY_OSC_BIN`, `VY_EGAIN`, `VY_RDNOIS`, pixel scale x (2*bin).

### DB + UI
- `EQUIPMENTS.BAYERMASK` migration (empty=mono); id=5 -> RGGB when row exists.
- DB Explorer: editable + validation (RGGB/BGGR/GBRG/GRBG/mono/empty).

### Pipeline integration
- `extract_fits_metadata`: surfaces `bayerpat`.
- Import: `validate_bayer_crosscheck` (FAIL if BAYERPAT + mono equipment; WARN on mismatch).
- After calibration (`quick_calibrate_last_import`): `run_osc_channel_extraction_for_archive`.
- Mosaic: no sky-surface, no PERF-10/post-cal QC during calibration.
- Per-channel: `_qc_enrich_calibrated_in_place(..., apply_sky_surface=True)` + qc_metrics.csv.
- Alignment: `check_osc01_channel_extraction_required` (OSC-01 **[wired]**).
- `param_resolver`: prefers `VY_EGAIN` / `VY_RDNOIS` on channel FITS.

### Config
- `osc_channel_binning` (int 1-4, default 2); registry **271** params.

## Effective gain/RN derivation

For `n_avg = plane_count * bin_n^2` pixels averaged with AVERAGE semantics:
- Poisson: variance per output ADU scales as `1/gain_eff` with `gain_eff = gain_raw * n_avg`.
- Read noise: `rn_eff = rn_raw / sqrt(n_avg)`.
- Validated by `test_monte_carlo_noise_model` (400 trials, 25% tolerance).

## Per-mask test evidence

`dev/tests/test_osc1_extraction.py`: parametrized RGGB/BGGR/GBRG/GRBG plane means;
extract+checkerboard (cb_before ~50 ADU, cb_after ~0); OSC-01 invariant; sky-surface
regression (no call on mosaic, call on channel).

## M71 smoke

**Partial / data-limited on this machine.** `Archive/M71/` contains only 4 dark FITS
(no light subset present for headless E2E). Full M71 smoke deferred to Milan (equipment id=5,
BAYERMASK=RGGB via migration). Synthetic extraction tests confirm checkerboard removal and
header contract.

## Anchor protection enumeration (mono path guards)

| Shared function | OSC guard | Mono path |
|-----------------|-----------|-----------|
| `calibrate_lights_to_calibrated` / `_calibrate_one_light_disk` | Unchanged math; skips post-cal QC + PERF-10 when `BAYERPAT` mosaic | Unchanged |
| `normalize_flat_master` | Uses `EQUIPMENTS.BAYERMASK` when header missing; only when DB mask set | Unchanged when mono/empty mask |
| `_qc_enrich_calibrated_in_place` | Sky-surface only if `VY_CHANNEL` or `apply_sky_surface=True`; never on BAYERPAT mosaic | Default: no sky-surface (unchanged vs pre-OSC production) |
| `quick_calibrate_last_import` | Calls extraction only when `get_equipment_bayermask(id) != None` | No extraction call |
| `astrometry_align_and_build_masterstar` | OSC-01 only when equipment BAYERMASK set | No OSC check |
| `extract_fits_metadata` | Adds optional `bayerpat` key | Additive field only |
| `smart_import_session` | Cross-check fail/warn only when BAYERPAT present | No-op when mono |

## Gates

### pytest + ruff
- `1079 passed, 24 skipped`
- `ruff check src_py dev/tests` - clean

### `--fast`
```
OVERALL: PASS
pytest: 1079 passed, 24 skipped
```

### `--full` draft_435 (mono anchor)
Started in background at task end (`tmp/osc1_full_gate.log`). **Await result before push.**
Expected: byte-identical (OSC code gated on BAYERMASK; mono draft_435 unchanged).

## Docs impact

- `VYVAR_INVARIANTS.md`: OSC-01 [wired]
- `VYVAR_DECISIONS.md`: OSC-CHANNEL-EXTRACTION
- `VYVAR_STATE.md`, params registry 271, `VYVAR_PARAMS.md` regen
- Handbook + FLOW PDF regenerated (OSC branch note in ch. 4.5)

## STOP before push

Await `--full` OVERALL PASS confirmation; then Milan push authorization.
