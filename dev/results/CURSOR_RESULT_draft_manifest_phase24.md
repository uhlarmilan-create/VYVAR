CURSOR RESULT - 2026-08-11 Phase 2.4 manifest-first OBS_DRAFT core fields

What I did
Extended fetch_obs_draft_by_id with apply_manifest_core_to_draft_row (paths, status,
center, JD, is_calibrated, final_observation_id). Flipped time_utils center,
sigma_budget rig lookup, masterstar path getters. Includes uncommitted 2.2/2.3
prerequisites in shared modules.

## Gates
- --fast: OVERALL PASS (1289 passed)
- P1 A/B (with full 2.4/2.5 vs without): core SHA 24820ee2... n=325 BOTH -- byte-identical

## Tests
dev/tests/test_manifest_core_overlay.py, manifest rig suite, test_obsloc_null_island, test_sigma_a2
