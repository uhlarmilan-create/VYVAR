CURSOR RESULT - 2026-07-19

What I did
VYVAR-INVARIANTS P2: contract registry + runtime gate library + minimal wired set
+ parity/RNG tests + FLOW 4.5 pedestal correction. One commit, not pushed.

## SHA-scope finding

`pipeline_meta.json` does **NOT** participate in:
- core photometry SHA (`lightcurve_*.csv`, `comp_quality_*.json`, `comparison_stars_per_target.csv`)
- extended SHA (+ `comp_qa_*.json`)
- `--full` science comparator (LC science columns + comparison_stars)

Source: `dev/tests/photometry_sha.py`. Therefore invariants/stages blocks are written
into `pipeline_meta.json` (no sibling file). Science byte-identity contract unchanged.

## Output / findings

### Deliverables
- D1: `docs/VYVAR_INVARIANTS.md` - full registry (wired + registry-only IDs).
- D2: `src_py/invariants_runtime.py` - `inv_check`, stage stamps, schema/CFG validators.
- D3: wiring - calibration FLUX-01/02; preprocess residual flatness stats; WCS-01 WARN;
  DAG stamps (masterstar / phase01 / phase2a / postprocess); end-of-run PROV+CFG;
  COG meta keys only when COG enabled (CFG-01).
- D4: `dev/tests/test_invariants_p2.py` - registry parity, unit FAIL/WARN, RNG AST.

### INV-RNG-01 hit list
**Empty.** No naked `np.random.<fn>(` calls in `src_py` (Generator / SeedSequence /
`default_rng` only). Allowlist unused.

### Gates
- Unit: 12 passed (`test_invariants_p2.py`)
- docs-sync guard: green (after FLOW PDF regen)
- `--fast`: OVERALL PASS
- P1 golden (`VYVAR_INVARIANTS_P1=1`): 5 passed (SHAs unchanged)
- `--full`: OVERALL PASS - BYTE-IDENTICAL
  - `full-science-compare` n_lc=166 failures=0
  - core `3d26f4692ac81fc5...` n=333; extended `6420f1daa53a0d5d...` n=499

## Docs impact

- **FLOW builder ch 4.5:** removed false "flux-conserving / mean-preserving" claim;
  replaced with full-surface pedestal convention (T3; ~-96 ADU). Priloha E adds
  `VYVAR_INVARIANTS.md`. PDF regenerated.
- **ROADMAP:** P2 ? DONE.
- **STATE:** invariants P2 one-liner.
- **DECISIONS:** `INVARIANTS-P2-REGISTRY` (wired set, policies, meta/SHA scope, FLOW
  doc-drift evidence).
- **facts:** no `flow_doc_facts.py` key changes.

## Errors (if any)
(none expected; fill if --full STOPs)

## Files changed
(see git commit)
