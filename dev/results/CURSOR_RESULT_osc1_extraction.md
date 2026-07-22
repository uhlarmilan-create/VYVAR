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
```
full-snapshot-sha-core       PASS   03d8fb6491bc3c22... n=333
full-photometry-sha-core     PASS   03d8fb6491bc3c22... n=333
full-photometry-sha-extended PASS   bbfcc92e7ac5c4c5... n=499
full-science-compare         PASS   n_lc=166 failures=0
OVERALL: PASS
```
Run on working tree with OSC-1 changes (~2131s pipeline). Mono byte-identical anchor preserved.

## STOP before push

Local commit `0f1c07f`. Await Milan push authorization.

---

## PRE-PUSH VERIFICATION - noise model pairing (2026-07-22)

### (a) `rn_eff` in `osc_extract.py` (verbatim, after fix)

Module docstring + `effective_gain_rn()`:

```
n_avg = plane_count * bin_n**2
gain_eff = gain_raw * n_avg
rn_eff = rn_raw * sqrt(n_avg)
```

```172:174:src_py/osc_extract.py
    gain_eff = g * n_avg
    rn_eff = rn * math.sqrt(n_avg)
    return gain_eff, rn_eff
```

### (b) Per-pixel variance in photometry error model (verbatim)

Howell (1989) eq. 2, per-pixel (`area=1`, `sky_pp=0`):

```1011:1013:src_py/photometry_core.py
    g = float(gain) if math.isfinite(gain) and gain > 0 else 1.0
    rn = float(read_noise) if math.isfinite(read_noise) and read_noise >= 0 else 10.0
    return flux / g + max(0.0, sky_pp) / g * area + (rn / g) ** 2 * area
```

PSF sky-only per-pixel sigma uses the same RN term:

```2174:2177:src_py/psf_photometry.py
    g = max(1e-6, float(gain))
    rn = max(0.0, float(read_noise_e))
    var = sky / g + (rn / g) ** 2
    return float(math.sqrt(max(var, 1e-12)))
```

Headers consumed via `param_resolver`: `VY_EGAIN`, `VY_RDNOIS` (preferred over `EGAIN`/`RDNOISE`).

### Pairing proof (AVERAGE semantics)

For `n = plane_count * bin^2` pixels averaged (variance scales as `1/n`):

- `gain_eff = n * gain_raw` => Poisson term `S_ADU / gain_eff = S_ADU / (n * g_raw)` OK
- RN term must be `(rn_eff / gain_eff)^2 = (rn_raw / g_raw)^2 / n`
- Therefore `rn_eff = rn_raw * sqrt(n)` (not `rn_raw / sqrt(n)`)

**Bug found and fixed:** pre-verification code had `rn_eff = rn_raw / sqrt(n_avg)`, inconsistent with `(rn/g)^2` consumers. Signal-dominated Monte Carlo masked this (~3% RN bias at 25% tolerance).

### Test strengthening

- `test_monte_carlo_noise_model`: prediction now uses production `_howell_variance_adu2` with `(rn_eff/gain_eff)^2`.
- Added `test_monte_carlo_noise_model_rn_dominated` parametrized over `channel in {R, G, oneRGGB}`, `osc_bin in {1, 2}`; S ~ 1.5-2.9 ADU (~3-5 e- at g=1.8), RN=8 e-, 1200 trials, **5%** tolerance.

### `VY_RDNOIS` header values (tests)

`test_extract_writes_headers_and_reduces_checkerboard` does **not** assert numeric `VY_RDNOIS` (only `VY_EGAIN > 1.5`). Example stamped values for `gain=1.5`, `rn=3.0`, G channel, `osc_bin=2` (`n_avg=8`):

| | Before fix | After fix |
|---|------------|-----------|
| `VY_RDNOIS` | 1.061 e- | 8.485 e- |

Any downstream FITS written before this fix understates read noise by factor `n_avg`.

### Gates (post-fix)

```
pytest: 1085 passed, 24 skipped
session_baseline_check --fast: OVERALL PASS
```

Committed as `7ffcc06`; see Push section below.

## Push (2026-07-22, Milan authorized)

### Pre-push checks

| Check | Result |
|-------|--------|
| `git fetch origin`; `origin/main` before push | `2c520c6` (unchanged since PIPELINE-SIMPLIFY push; DOCS-REFRESH not yet on remote) |
| Stack `git log origin/main..HEAD --oneline` | 4 commits - exact match (see below) |
| `git status --short` | Clean; allowlisted untracked only (`dy_peg_night_run_bvr.py`, `qatar8_night_run_v.py`) |
| `session_baseline_check.py --fast` (final HEAD `7ffcc06`) | **OVERALL PASS** - 1085 passed, 24 skipped |

### Commit inventory (`origin/main..HEAD`, newest first)

```
7ffcc06 fix(osc): correct rn_eff pairing (rn*sqrt(n)) + RN-dominated Monte Carlo test
4634bd1 docs(results): OSC-1 gate record (--full PASS on mono anchor)
0f1c07f feat(osc)!: CFA calibration, Bayer channel extraction, BAYERMASK, and OSC-01 gate (phase 1)
c055ac3 docs(pdf): regenerate parameter handbook + FLOW after param removals (DOCS-REFRESH; was local-only until this push)
```

Base: `2c520c6` -> stack tip: `7ffcc06` (`git push origin main` succeeded).

### Bookkeeping

One docs commit (ledger stamp, this result file push record) pushed with the stack.

### Final origin/main tip

Local HEAD matches `origin/main` at `7ffcc06` after push. For current tip: `git rev-parse origin/main`.
