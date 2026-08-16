# CURSOR RESULT - COMP-ASSIGN-02

Date: 2026-08-16
Baseline: 0f5f7b0 (COMP-ASSIGN-01, stamp 635404f), on IMPL-04 5cfb285
Tip: (pending commit)
Push: NO

## What I did

Restored the authoritative per-target `phase01_comparison_max_comp_rms`
ceiling inside `_select_comps_by_color_then_rms` (before colour ladder and
before `head(n_comp_max)`), unified check-star selection on the backfill path,
remeasured the acceptance subset at r=9.5 with a fixed meter.

## Item 1 - RMS ceiling

- `_select_comps_by_color_then_rms` now filters `comp_rms <=
  phase01_comparison_max_comp_rms` (densified value passed via
  `max_comp_rms=` from `select_comparison_stars_per_target`) before widen /
  `head(n_comp_max)`.
- `n_comp_max` is a ceiling, not a pad target (DECISIONS entry).
- Existing relax path unchanged: colour ladder -> full **under-ceiling** set
  (log) -> thin-set graceful degradation if still `< n_comp_min`; never
  re-admits above-ceiling comps; empty / zero gate-passers still hit
  `COMP_ASSIGNMENT_RELAX_ORDER` sparse_fallback.

### Fire proof

| Test | Result |
|---|---|
| `test_fire_comp_assign_01_snapshot_breached_ceiling` | PASS (395/748 above 0.1 on COMP-ASSIGN-01 snapshot) |
| `test_select_comps_max_comp_rms_ceiling_before_head` | PASS (noisy colour bin dropped; ladder fills with clean) |
| `test_fire_rebuilt_comparison_csv_under_ceiling` | PASS (0 above dense ceiling 0.08) |

Rebuilt membership: 97 targets, 734 pairs, min/max/median 3/8/8,
`comp_rms` max 0.0796, **0 above ceiling**.

FW CVn after: rms
`[0.019, 0.024, 0.025, 0.026, 0.049, 0.062, 0.069, 0.078]` (was up to 0.460).
BO lost the 0.150 member. Both stayed at colour ladder step 1 (T1).

## Item 2 - One check-star path

- Deleted parallel lowest-RMS pick + `ens_ids=set()` in
  `dev/scripts/backfill_check_kmag_sidecars.py`.
- Authority: `select_check_star` (field pool) for non-sparse;
  `select_external_check_star` only for sparse (`n_ens <= 2`).
- Grep: no third production pick path in `src_py/` / backfill.

### Star `1497145751650265600`

| Field | Value |
|---|---|
| Gaia DR3 | 1497145751650265600 |
| G / BP-RP | 9.863 / 1.096 |
| VSX / Gaia variable catalog | false / false |
| peak_dao / saturated | 8570 ADU / false |
| contamination_idx | 0.094 |
| pool `comp_rms` (as comp) | 0.0107 |
| Sidecars using it (draft 514) | 45 (all acceptance) |

Same twin-222 pattern: one field-pool pick shared across many targets.
Pool RMS is excellent; measured KMAG scatter against cleaned ensembles is
still ~15-19 mmag (quiet set median ~17). Not a VSX/Gaia-variable /
saturation reject - residual variability or ensemble-relative systematics
below catalog flags.

## Item 3 - Fixed-meter remeasure (r=9.5)

Meter held identical: `1497145751650265600` on COMP-ASSIGN-01 snapshot and
COMP-ASSIGN-02 after (same_meter=True).

| Target | check before | check after | delta_mag before | delta_mag after | mag_calib before | mag_calib after | pred sigma_ens |
|---|---:|---:|---:|---:|---:|---:|---:|
| BO CVn | 17.52 | 16.39 | 146.57 | 146.19 | 145.64 | 145.50 | 14.47 |
| FW CVn | 20.32 | 18.70 | **80.02** | **23.63** | 21.16 | 20.27 | 17.42 |

IMPL-04 reference (different meters): BO 9.06, FW 8.58.

### Verdict (stated in advance)

Check scatter does **not** return to the IMPL-04 ~9 mmag class with this
meter. Ensemble repair is real (FW `delta_mag_std` 80 -> 24 mmag; predicted
`sigma_ens` now matches check order). The residual ~16-19 mmag is a finding
about the shared check star / selection key, not about the RMS ceiling.

Quiet acceptance (same meter): check scatter ~15-19 mmag typical; two
outliers 29 and 57 mmag (`1498783199341798016`, `1498842882207281152`).

Artifacts: `dev/results/COMP_ASSIGN_02_measure.json`,
`dev/results/COMP_ASSIGN_01_comparison_stars_per_target.csv` (fire-proof
fail side), `docs/VYVAR_DECISIONS.md` entry.

## Tests

- Fire proofs above
- `--fast` (running / see tip SHA)

## Files changed

- `src_py/photometry_core.py` - ceiling in `_select_comps_by_color_then_rms`
- `src_py/comp_selection_per_target.py` - pass densified `max_comp_rms`
- `dev/scripts/backfill_check_kmag_sidecars.py` - delete parallel pick
- `dev/tests/test_forced_phot_and_weights.py` - fire proofs
- `docs/VYVAR_DECISIONS.md`
- `dev/tools/comp_assign_02_measure.py`
- `dev/results/COMP_ASSIGN_02_measure.json`, this result
