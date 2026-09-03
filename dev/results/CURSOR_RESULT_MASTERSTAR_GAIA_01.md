# CURSOR RESULT - MASTERSTAR-GAIA-01 (2026-08-19)

Parts A-E implemented on tip; Part F (full 516 rebuild + anchor recut) **not run** - requires ~5000 s headless pipeline and Milan authorization before push/recut.

## What I did

### Part A - False-acceptance audit (E1/E2)
- Built verified-empty position set: **2200** main + **200** corner on MASTERSTAR 516 (`session_20260819_msgaia01/`).
- Measured production pass-2 acceptance **verbatim** at baseline params (?=1.9, center_tol=5 px):
  - **E1 DEVIATE**: main false-accept **55.5%** (1220/2200); corners **60%** - confirms Milan's empty-circle observation.
  - **E2 DEVIATE**: FORCED_SEED false-accept **1.32%** (29/2200) at SNR?3, centroid?2 px.
- Parameter sweep ? minimal tightening meeting ?1%:
  - pass2: **?=5.0**, **center_tol=2.0 px** ? 0.091% false-accept
  - seed: **SNR?4.0** ? 0.36% false-accept
- Applied tightening as production defaults in `config.py`; refactored `_dao_targeted_pass2_unmatched_gaia` to call shared `dao_pass2_try_at_position`.
- Fire-proof tests: `dev/tests/test_masterstar_gaia_01.py` (INV-DET-FALSEFILL-01, INV-SEED-FALSEFILL-01) - **4/4 PASS**.

### Part B - Lock-existing + leftover assignment
- New module `src_py/masterstar_gaia_accounting.py`: `lock_existing_and_leftover_assign()` with `vy_match_mode` (`locked` / `leftover_promotion`).
- INV-MS-IDENTITY-01 test on anchor `masterstars_full_match.csv` pairings - **PASS** (zero remaps at 0.01 px tol).

### Part C - FORCED_SEED admission + census
- `enrich_masterstar_gaia_complete()` adds FORCED_SEED rows (centroid?2 px, SNR?4), `SEED_REJECTED` accounting, writes `gaia_source_state_census.csv`.
- Wired into MS build path after `source_type` assignment (`pipeline.py`).
- INV-MS-CENSUS-01 checked at write time (FAIL policy).
- Config: `masterstar_forced_seed_comp_pool_enabled=False` (default; comp pool expansion gated off).

### Part D - BLENDED accounting
- `annotate_blended_groups()` - FWHM pairs, group id + Gaia G flux ratio on non-owner; census rows get `BLENDED` state (no photometry change).

### Part E - Display truth
- `ui_masterstar_qa.py` counts `source_state` (DETECTED_P1/P2, FORCED_SEED, DAO_ONLY); falls back to `forced_photometry` column - kills FORCED_APERTURE label bug.

### Part F - Not executed
- Full clean rebuild 516 ? E3/E4/E5 evaluation, anchor/P1 golden recut, BO/FW exports **pending**.
- **Expected coupling**: pass2 fill drops from baseline **3314** (?=1.9) - tightening will reduce pass-2 detections on real Gaia; hole recovery shifts toward FORCED_SEED path.
- Anchor **477dc8cf** and goldens **untouched** per task constraint.

## Output / findings

| Expectation | Status | Notes |
|-------------|--------|-------|
| E1 pass2 empty-sky | **PASS** (after tighten) | Baseline DEVIATE 55.5%; tightened 0.09% |
| E2 seed empty-sky | **PASS** (after tighten) | Baseline 1.32%; tightened 0.36% |
| E3 membership census | **PENDING** | Needs Part F rebuild |
| E4 MAG 48 LC | **PENDING** | Sacred; STOP if any bit moves |
| E5 BO/FW meters | **PENDING** | If E4 green |

Raw data: `dev/results/context/session_20260819_msgaia01/`
- `empty_positions_main.csv`, `empty_positions_corner.csv`
- `part_a_false_accept.json`, `part_a_param_sweep.csv`

Harness (tmp, not tracked): `tmp/masterstar_gaia_01_part_a.py`, `tmp/masterstar_gaia_01_sweep.py`

## Config changes (new keys)

| Key | Default | Purpose |
|-----|---------|---------|
| `masterstar_dao_pass2_sigma` | **5.0** (was 1.9) | Pass-2 threshold |
| `masterstar_dao_pass2_center_tol_px` | **2.0** (was 5) | Pass-2 centroid gate |
| `masterstar_forced_seed_centroid_max_px` | 2.0 | FORCED_SEED admission |
| `masterstar_forced_seed_snr_min` | **4.0** (was 3) | FORCED_SEED admission |
| `masterstar_lock_leftover_radius_px` | 3.0 | Leftover greedy radius |
| `masterstar_lock_pair_tol_px` | 3.0 | Lock pair tolerance |
| `masterstar_gaia_census_edge_margin_px` | 10.0 | EDGE state band |
| `masterstar_forced_seed_comp_pool_enabled` | false | Comp pool gate |

## Errors (if any)

None in implemented parts. Part F not attempted (runtime + authorization).

## Files changed

- `src_py/masterstar_gaia_accounting.py` (new)
- `src_py/pipeline.py` (pass2 refactor + gaia enrich hook)
- `src_py/config.py` (new keys + tightened defaults)
- `src_py/ui_masterstar_qa.py` (source_state overlay counts)
- `dev/tests/test_masterstar_gaia_01.py` (new)
- `dev/results/CURSOR_TASK_MASTERSTAR_GAIA_01.md` (new)
- `dev/results/context/session_20260819_msgaia01/*` (measurement artifacts)

## Runtime (Rule 0.3)

| Part | Wall time |
|------|-----------|
| A audit (2200+200 positions) | ~17 s |
| A param sweep (28 combos) | ~144 s |
| Invariant tests | ~10 s |
| B-E code | (implementation) |
| F rebuild | **not run** (~5000 s est.) |

## Next steps (Part F)

1. Full clean rebuild draft 516 on fixed tip (Phase 0 ? re-annotation ? 1 ? 2A, PFS ON).
2. Evaluate E3: G?13 DETECTED+FORCED = 96+-1%; on-chip G?17.5 census sum; zero remaps; promotions/seeds counts.
3. **E4 gate**: MAG 48 LC byte-identical to **477dc8cf** - if red, STOP with per-target delta report.
4. If E4 green: recut `--full` anchor + P1 golden; `--fast`/`--full` OVERALL PASS; docs (STATE, ROADMAP, DECISIONS, JOURNAL, PARAMS, ledger, CHANGELOG).
5. Push only on Milan authorization.
