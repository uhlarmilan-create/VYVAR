TASK_ID: GATE-OWNERSHIP-01 + GATE-REGIME-01

CURSOR RESULT - 2026-08-15

Both tasks completed locally. Push: NO. Waiting for Milan.

## GATE-REGIME-01 -- commit 18c770e

- Explicit CompPoolRegime (DERIVED|LEGACY|FAILED); mutually exclusive by construction
- INV-NO-SILENT-EMPTY wired at derived admission; FAILED no silent legacy downgrade
- Persists photometry/comp_pool_admission.json (+ pipeline_meta) with reject_reason_counts
- Happy path draft 512 pool ID SHA-256 identical: cccdda39bd74cfbd58f141bbe602e2eb58e45eea7780f51d54055d5b6598d77b
- Tests: 4/4 pass in test_gate_regime_01.py
- Detail: dev/results/CURSOR_RESULT_GATE_REGIME_01.md

## GATE-OWNERSHIP-01 -- commit 5612f42

- 59 gates in dev/validation/gates_inventory.json; validator OK
- R1-R5 in dev/results/CURSOR_RESULT_GATE_OWNERSHIP_01.md
- Fourth conflict class: multi-floor detection significance
- Rank cuts: literature does not defend p84 as comparison-admission owner (Broeg absolute; Sokolovsky 2017 for variable search)

## --fast

OVERALL PASS. 1364 passed, 27 skipped (on working tree including new tests).
