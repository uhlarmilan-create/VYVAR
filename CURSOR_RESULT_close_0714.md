CURSOR RESULT -- 2026-07-14 (SESSION-CLOSE-0714)

What I did
Full sync verification, documentation refresh for post-0714 world, integrity gate on HEAD,
clean tree push.

## Part 0 -- Sync state

**git fetch + status:** local `main` == `origin/main` at code baseline **`114c423`**
(0 ahead / 0 behind). Restored tracked `CURSOR_RESULT*.md` files that had been locally deleted.

**Untracked inventory (decisions):**

| Path | Decision |
|------|----------|
| `.worktrees/` | Leave (local git worktrees) |
| `CURSOR_RESULT_except_retriage3.md` | Leave (scratch result) |
| `docs/VYVAR_CODE_AUDIT.md` | Leave (audit scratch; not session deliverable) |
| `docs/round2_figs/v0454_lc_vyvar.png` | Leave (local figure) |
| `scripts/dy_peg_night_run_bvr.py` | Leave (one-off night-run; hardcoded paths) |
| `scripts/qatar8_night_run_v.py` | Leave (one-off night-run; hardcoded paths) |

No production code changes in this close.

## Part 1 -- Documentation refresh

| File | Changes |
|------|---------|
| `docs/VYVAR_STATE.md` | Single post-0714 current snapshot (anchor, err model, SPARSE-TRUST, k'', CAL-DIAG, WSN park, guards, 852/15); 0713-0714 history compressed to table; stale NOT-PUSHED lines removed; test count updated |
| `docs/VYVAR_ROADMAP.md` | NEXT SESSION consolidated; standing open items = data-gated backlog (6 rows) + parked rows with revisit triggers; 0714 arc commit refs; ANCHOR-CHAIN marked pushed |
| `docs/VYVAR_JOURNAL.md` | SESSION-CLOSE-0714 entry; CAL-LEDGER + SPARSE-TRUST journal lines updated (pushed) |
| `CHANGELOG.md` | User-facing 0713/0714 entries: err re-anchor, sparse trust, CAL-DIAG, k2 cohort, WSN study |
| `README.md` | Full test suite command + 852/15 count |
| `validation/VYVAR_VALIDATION_LEDGER.json` | `--full` run stamped commit `114c423` |

**Cross-doc links verified:** `VYVAR_SPARSE_TRUST_SPEC.md`, `VYVAR_SIGMA_FLOOR_SPEC.md`,
`VYVAR_CAL_DIAG_SPEC.md`, `VYVAR_WIDE_SLOPE_NOISE_SPEC.md` exist at stated paths.

**CITATIONS.bib:** all four entries present (`pzq2006`, `merlinehowell1995`,
`everetthowell2001`, `howellwarnockmitchell1988`); `howellwarnockmitchell1988` referenced in
`sparse_trust_core.py`; others in `VYVAR_SIGMA_FLOOR_SPEC.md`. No dangling refs found.

## Part 2 -- Integrity gate

**session_baseline_check.py --full** on HEAD `114c423`: **OVERALL: PASS**

```
pytest                       PASS   852 passed, 15 skipped
full-science-compare         PASS   n_lc=178 failures=0
full-snapshot-sha-core       PASS   bf3743a150d78828... n=357
full-photometry-sha-core     PASS   bf3743a150d78828... n=357
full-counters-runtime        PASS   {}
full-counters-meta           PASS   {}
full-provenance              PASS   anchor git_hash=8fb21b32bd0b...
```

Runtime: ~50 min (2778s pipeline step).

**pytest tests/ (full, incl. slow markers):** **852 passed**, 15 skipped.

**pytest -m slow only:** 3 passed, 4 skipped (slow subset; full suite above is authoritative).

## Part 3 -- Close

**Final origin/main:** `5d4bce0` (close commits `0f1c941`, `db2e386`, `5d4bce0`; code baseline `114c423`).

## Errors (if any)

None.

## Files changed

- `docs/VYVAR_STATE.md`, `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_JOURNAL.md`
- `CHANGELOG.md`, `README.md`
- `validation/VYVAR_VALIDATION_LEDGER.json`
- `CURSOR_RESULT_close_0714.md`

---

**Next-session entry point:** Data-gated backlog only; startup ritual: git pull -> STATE ->
ROADMAP -> `session_baseline_check.py --fast` -> await Milan data (darks ~2026-07-21 first).
