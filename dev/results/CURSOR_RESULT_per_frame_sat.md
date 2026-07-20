CURSOR RESULT - 2026-07-19 / 2026-07-20

What I did
Implemented PER-FRAME-SAT-GATED: per-frame target saturation decisions behind
`per_frame_saturation_enabled` (default OFF) + `per_frame_sat_min_clean_frac`
(default 0.5). Synthetic validation, params parity, INV-CFG-01 extension, docs.
ONE commit; not pushed.

## Output / findings

### Design (DECISIONS: PER-FRAME-SAT-GATED)
- OFF: no-op on `skip_photometry` (zone-based legacy); no new markers (INV-CFG-01).
- ON (targets only): master zone sat/likely_sat advisory; clean_frac from
  per-frame `is_saturated` / `peak_max_adu`; ? threshold ? measure; < threshold
  ? `skip_reason=per_frame_saturation`; missing peak data ? zone fallback +
  `per_frame_sat_fallback=true`.
- Wired in Phase 2A prepare before aperture star-set selection; target loop no
  longer re-forces zone skip when flag ON.

### Synthetic outcomes (S1/S2/S3)

| Star | Sat frames | clean_frac | OFF | ON |
|------|------------|------------|-----|-----|
| S1 | 6/20 | 0.70 | skip (zone/legacy) | **measured** (rescued); 14 clean |
| S2 | 15/20 | 0.25 | skip (`zone_flag`) | skip (`per_frame_saturation`) |
| S3 | 0/20 | 1.00 | measure | measure (identical) |

Also covered: flag-OFF apply no-op equivalence; fail-safe fallback when peak/sat
columns missing. Tests drive `decide_target_saturation_policy` +
`apply_per_frame_saturation_to_active_targets` (narrowest production decision
path; full Archive night fixture not required).

### Summary-column comparator scope
`photometry_summary.csv` is **outside** photometry SHA / science-compare file
set (`dev/tests/photometry_sha.py` patterns: lightcurves, comp_quality,
comparison_stars_per_target only). `sat_clean_frac` / `skip_reason` go on
summary (additive) when ON - no SHA impact.

### Gates
- ruff: clean on touched modules
- `--fast`: **OVERALL PASS** (1022 passed, 17 skipped)
- P1 golden: **5/5**
- `--full`: **OVERALL PASS** - BYTE-IDENTICAL vs VL-ANCHOR-WCSINV
  - science-compare n_lc=166 failures=0
  - core SHA `3d26f4692ac81fc5...` n=333
  - extended SHA `6420f1daa53a0d5d...` n=499

### Docs impact
- `docs/VYVAR_DECISIONS.md` - PER-FRAME-SAT-GATED (design + M67 HIGH?MED + deferred validation)
- `docs/VYVAR_ROADMAP.md` - item kept open: implemented gated OFF; validation pending next saturated dataset
- `docs/VYVAR_STATE.md` - one-liner
- `docs/VYVAR_INVARIANTS.md` - INV-CFG-01 amendment (per_frame markers)
- `docs/VYVAR_PARAMS.md` - regenerated (271 entries)
- `docs/VYVAR_PARAMETER_HANDBOOK_CZ.pdf` - regenerated
- `docs/VYVAR_FLOW_CZ.pdf` + `flow_doc_facts.py` + ch.9 sentence - regenerated; docs-sync green
- `CLAUDE.md` - registry count 271

## Errors (if any)
None blocking. Pre-existing EXC-0030 sqlite3.Row `.get` noise during --full
pipeline (unchanged; SHA still byte-identical).

## Files changed
- `src_py/photometry_core.py` - decision helpers + Phase 2A wiring
- `src_py/config.py` - two AppConfig fields + load/to_dict
- `src_py/invariants_runtime.py` - INV-CFG-01 per-frame markers
- `config.json`, `dev/validation/params_registry.json`
- `dev/tests/test_per_frame_saturation.py` (new)
- `dev/tests/test_invariants_p2.py`, `dev/tests/test_ui_params_dashboard.py`
- docs / FLOW / handbook / PARAMS as above
- Commit: `69432ee` (not pushed)
