CURSOR RESULT — 2026-06-16 (Phase-1 graceful comp degradation commit)

What I did
Verified Phase-1 production files match validated matrix `164157` state. Fixed stale
`test_alg_functions.py` assertion (`temporal_binning_enabled` now False). Updated STATE / ROADMAP /
DECISIONS / JOURNAL. Committed Phase-1 + spec + tests + force-added validation harness. Pushed to
origin/main.

## Output / findings

### Pre-commit verification
- Phase-1 files in working tree: `comp_selection_per_target.py`, `photometry_core.py`,
  `trust_flag_core.py`, `check_star_kmag.py`, `comp_pool_rms.py`, `comp_qa_core.py`, `config.json`,
  `photometry_report.py` (+ trust tests).
- **pytest:** 322 passed, 15 skipped (was 321 pass / 1 fail before `test_alg_functions` sync).
- **ruff:** all checks passed.

### Matrix `164157` (reference)
| Target | trust | Notes |
|--------|-------|-------|
| BO CVn | GREEN | 4 T1, check 0.007 |
| V0842 Her | YELLOW | 8 T1, soft check 0.023 |
| V0612 / SS Cam | RED | degraded proc, honest fail-safe |
| V0611 / degenerate | YELLOW | sparse + check cap |

### Known issue (b) — NOT fixed in this commit
RMS fallback + `len(result)>=1` routing; next = Phase-1b per ROADMAP.

## Errors (if any)
None (suite green after test default sync).

## Files changed
Production: comp_selection*, photometry_core, trust_flag_core, check_star_kmag, comp_pool_rms,
comp_qa_core, config.json, photometry_report.py
Tests: test_trust_flag.py, test_trust_checkstar_hardening.py, test_alg_functions.py
Docs: VYVAR_COMP_DEGRADATION_SPEC.md, STATE, ROADMAP, DECISIONS, JOURNAL
Harness: sandbox/comp_degradation_validate.py (force-added; `/sandbox/` gitignored)

**HEAD:** (see git log after push)
