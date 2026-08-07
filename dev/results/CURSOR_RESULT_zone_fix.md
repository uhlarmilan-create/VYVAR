CURSOR RESULT - 2026-08-07 (zone_fix)

What I did
Implemented scale-invariant MASTERSTAR flux-zone classification behind
``masterstar_zone_mode`` (default ``legacy``), threaded ``bg_sigma_adu`` from
DAO detection metadata, added unit tests and an offline measurement tool, ran
local P1 A/B at legacy default, updated registry/docs.

## 1. Premise check (draft_452 fixture, re-derived)

Path: ``dev/results/context/session_20260727/draft_452_masterstars_full_match.csv``
(n=2951; zone distribution matches draft_435).

| Check | Value |
|-------|------:|
| ``noise_floor_adu`` | 2105.92 |
| sky median (task reference) | 1955.0 |
| k | 1.8 |
| ``sigma_px = (nf - sky) / k`` | 83.85 |
| Matched ``peak_max_adu`` median | 2398.2 |
| ``peak_max_adu - sky`` | 443.2 |
| Matched ``peak_dao`` median | 430.6 |
| DAO_ONLY ``peak_dao/sigma_px`` median | 2.19 |
| Matched ``peak_dao/sigma_px`` median | 5.14 |
| Legacy zone ``peak_sig`` medians | linear 9.08, noisy1 2.61, noisy2 2.34, noisy3 2.38 |
| linear p05 ``peak_sig`` | 2.38 (overlaps all noisy sub-classes) |
| ``peak_sig >= 3.5`` count | 1859 (63.0%) vs legacy linear 1799 (61.0%) |
| ``peak_sig >= 4.0`` count | 1718 (58.2%) |

All four premise numbers from the task reproduce. Legacy sub-classes are not
monotonic in peak significance.

## 2. ``sigma_px`` source

Primary: ``det_meta["bg_sigma_adu"]`` written in ``detect_stars_and_match_catalog``
(``pipeline.py`` ~8553) as ``max(std, 1.0)`` from the same ``sigma_clipped_stats``
call that sets ``noise_floor = med + k * bg_sigma``. Also stored:
``det_meta["sky_median_adu"]``.

Passed into ``_annotate_masterstars_flux_zones`` at the MASTERSTAR write call
(``pipeline.py`` ~12461). Inversion ``(noise_floor - sky) / k`` is fallback only
(logged once) when ``bg_sigma_adu`` is missing (offline CSV reclassification).

## 3. Threshold sweep (offline reclassification on existing CSVs)

T2 = T1 - 1.0, T3 = T1 - 2.0. ``is_usable`` = linear count.

### draft_452 fixture (wide rig reference)

| T1 | is_usable (was 1799) | GAIA_MATCHED linear frac | DAO_ONLY linear frac |
|----|---------------------:|-------------------------:|---------------------:|
| 3.0 | 1999 | 0.695 | 0.220 |
| 3.5 | 1831 | 0.637 | 0.202 |
| 4.0 | 1690 | 0.587 | 0.193 |
| 4.5 | 1560 | 0.542 | 0.183 |
| 5.0 | 1443 | 0.501 | 0.174 |

### draft_435 (Archive NoFilter_60_2)

| T1 | is_usable (was 1799) | GAIA linear | DAO linear |
|----|---------------------:|------------:|-----------:|
| 3.0 | 804 | 0.278 | 0.119 |
| 3.5 | 719 | 0.248 | 0.119 |
| 4.0 | 644 | 0.223 | 0.083 |
| 4.5 | 583 | 0.203 | 0.064 |
| 5.0 | 540 | 0.188 | 0.064 |

### draft_500 (Archive NoFilter_60_2)

| T1 | is_usable (was 1713) | GAIA linear | DAO linear |
|----|---------------------:|------------:|-----------:|
| 3.0 | 438 | 0.121 | 0.012 |
| 3.5 | 374 | 0.104 | 0.004 |
| 4.0 | 335 | 0.094 | 0.004 |
| 4.5 | 289 | 0.081 | 0.002 |
| 5.0 | 269 | 0.075 | 0.002 |

### draft_502 (Newton pre-cal V_60_2)

| T1 | is_usable (was 4) | GAIA linear | DAO linear | targets >= n_comp_min=2 |
|----|------------------:|------------:|-----------:|--------------------------:|
| 3.0 | 1280 | 0.976 | 0.476 | 10 / 10 |
| 3.5 | 1123 | 0.959 | 0.274 | 10 / 10 |
| 4.0 | 1036 | 0.941 | 0.174 | 10 / 10 |
| 4.5 | 977 | 0.922 | 0.116 | 10 / 10 |
| 5.0 | 917 | 0.890 | 0.075 | 10 / 10 |

Full JSON: ``dev/results/zone_fix_measurement.json``.

## 4. Monotonicity (new sub-classes)

Under ``peak_significance``, subclass medians of ``peak_dao/sigma_px`` are
**monotonic** (noisy3 < noisy2 < noisy1 < linear) for all threshold sets on all
four drafts (``subclass_monotonic: true`` throughout). Legacy classes were not.

Example medians (draft_452, T1=3.5): noisy3 0.75, noisy2 1.72, noisy1 2.42,
linear 8.08.

## 5. Local P1 A/B (legacy default, same HEAD)

P1 mini uses frozen ``masterstars_full_match.csv``; zone code runs at MASTERSTAR
build, not in this photometry path. Two fresh subprocess runs (pre-fix reverted
vs zone-fix at ``masterstar_zone_mode=legacy``):

| run | core SHA | core n |
|-----|----------|-------:|
| pre_zone_fix_reverted | ``9b39d899be0853311d7acf0ced956f4ff9226871df23aeebb5f00c916fc7b479`` | 81 |
| with_zone_fix_legacy_default | ``9b39d899be0853311d7acf0ced956f4ff9226871df23aeebb5f00c916fc7b479`` | 81 |

**Identical** (``dev/tools/zone_fix_p1_ab.py``). Golden ledger still stale; not used
as gate.

## 6. Recommendation (**awaiting Milan's approval**)

Recommend **T1 = 4.0** (T2=3.0, T3=2.0):

- On the wide-rig fixture, 1690 ``linear`` vs 1799 legacy (94%) -- closest balance
  among thresholds that materially reject DAO_ONLY (19% vs 22% at T1=3.5).
- On draft_502, restores photometry viability (1036 usable, 94% Gaia linear,
  10/10 phase-1 targets clear ``n_comp_min=2`` in offline sim vs 4 usable today).
- T1=3.5 tracks legacy wide-rig count slightly tighter (1831 vs 1799) but admits
  more DAO_ONLY as linear (22% vs 17% at T1=4.0).
- T1=5.0 is safer on DAO rejection but drops wide-rig usable count further.

**Do not flip default** until approved. Draft_502 can re-run today with
``"masterstar_zone_mode": "peak_significance"`` in ``config.json`` (and chosen
T1/T2/T3 if not using code defaults 3.5/2.5/1.5).

## 7. Premise anomalies

None. All section-0 fixture numbers reproduced.

## Tests

- ``dev/tests/test_masterstar_zone_classifier.py``: 5 passed
- ``test_generated_params_md_is_fresh``: passed
- Full ``dev/tests`` except ``test_invariants_p1_golden.py``: 1266 passed; expected
  P1 ledger failures in ``test_invariants_p1_seed.py`` (2); pre-existing
  ``test_flow_doc_config_facts`` fails on local ``config.json`` vs
  ``flow_doc_facts.py`` (``vsx_out_of_scope_types``), unrelated to this change.

## Files changed

- ``src_py/pipeline.py`` -- classifier, ``bg_sigma_adu`` in det_meta
- ``src_py/config.py`` -- 4 new keys, default ``legacy``
- ``src_py/params_registry.py`` -- literal options
- ``dev/validation/params_registry.json`` -- 4 entries
- ``docs/VYVAR_PARAMS.md`` -- regenerated
- ``dev/tests/test_masterstar_zone_classifier.py``
- ``dev/tests/test_ui_params_dashboard.py`` -- owner count 259->263
- ``dev/tools/zone_fix_measurement.py``, ``zone_fix_p1_ab.py``, ``zone_fix_p1_run_once.py``
- ``dev/results/zone_fix_measurement.json``
- ``dev/results/CURSOR_RESULT_zone_fix.md``

No new ``WIRED_INV_IDS`` entry.
