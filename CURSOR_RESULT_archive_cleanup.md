CURSOR RESULT -- 2026-07-15 (ARCHIVE-CLEANUP)

What I did
Inventoried Archive/, verified offline anchor backup, wiped draft/evidence trees,
implemented ledger-driven --full SUSPENDED, updated docs/ledger/tests.

## Part 0 -- Pre-check

**session_baseline_check --fast:** OVERALL PASS (852 passed, 15 skipped; HEAD dcd88bf).

## Part 1 -- Inventory (pre-deletion)

**Archive top-level:** Drafts/ (44.974 GB), evidence/ (5.636 GB). Total **50.611 GB**.
No other top-level entries. No stray files at Archive root.

| Entry | Size (GB) | Kind | Repo references |
|-------|-----------|------|-----------------|
| draft_000424 | 6.105 | draft tree (pre-cal + Raw + processed) | session_baseline --full working draft; utils.resolve_draft_dir |
| draft_000424_snapshot_sigma_floor_20260713 | 0.250 | snapshot (ACCEPTED anchor) | VL-ANCHOR-424; session_baseline SNAPSHOT_NAME |
| draft_000424_snapshot_intermediate_b5364e6_20260713 | 0.250 | snapshot (chain validation) | CURSOR_RESULT_anchor_chain.md; ledger notes |
| draft_000424_snapshot_20260708_full | 0.250 | snapshot (superseded) | JOURNAL/STATE history |
| draft_000424_snapshot_20260708_hybrid_deprecated | 5.235 | snapshot (deprecated hybrid) | JOURNAL; retired per anchor chain |
| draft_000425 | 3.037 | draft tree (filtered BVR) | K2 matrix docs; CAL-DIAG historical |
| draft_000425_snapshot_20260707 | 2.512 | snapshot | JOURNAL K2/determinism refs |
| draft_000426 | 5.638 | draft tree (Newton) | MASTERSTAR-EPOCH; sigma forensic docs |
| draft_000427 | 12.808 | draft tree (Boyden) | K2 cohort; frame-gate docs |
| draft_000427_snapshot_20260707 | 8.889 | snapshot | JOURNAL rerun refs |
| draft_000426_stale_20260626 | 5.636 | evidence tree | ROADMAP 426-REGEN; scripts/draft_426_regen.py |

**Part 1.2 -- Unknown / non-draft entries:** NONE. All entries are draft, snapshot, or evidence trees.

**Part 1.3 -- Sole-copy raw FITS verdict:**

| Location | Files | Size (GB) | Notes |
|----------|-------|-----------|-------|
| draft_000424/Raw | 150 FITS | 0.814 | Pre-cal pipeline raw staging |
| draft_000425/non_calibrated | 37 | 0.438 | Raw lights import |
| draft_000426/non_calibrated | 101 | 1.763 | Raw lights import |
| draft_000427/non_calibrated | 323 | 3.917 | Raw lights import |

Within `C:\ASTRO\python\VYVAR\`, these are the **only in-repo copies** of raw/original FITS
(no parallel `Archive/DY Peg/` or other raw roots present). Observatory-side originals were
**not audited** in this task. Milan decision 2026-07-15 (ARCHIVE-CLEANUP) explicitly authorizes
clearing the Archive for new measurements after offline anchor backup. **Proceed authorized.**

Snapshots contain platesolve/photometry outputs only (no raw FITS).

## Part 2 -- Anchor offline backup

**Source:** `Archive/Drafts/draft_000424_snapshot_sigma_floor_20260713`

**Backup path:** `C:\ASTRO\backups\vyvar_anchor_424_sigma_floor_20260713_core-bf3743a1.zip`
(7z unavailable; zip used per task fallback)

| Field | Value |
|-------|-------|
| Size | 257667235 bytes (0.240 GB compressed) |
| SHA256 | 8706c0d6412ac2dd3e318f7f74eba5baaf275d07e9c775fa062985640e2652d48 |
| Extract core SHA | bf3743a150d788283eab2ab51db7b31f59e6d1c481159208bbe3f573092ec975 |
| n (core) | 357 |
| Match expected | **YES** |

Temp extraction deleted after verification.

## Part 3 -- Wipe

| Metric | Value |
|--------|-------|
| Before | 50.611 GB |
| After | 0.000 GB |
| Freed | **50.611 GB** |

**Skeleton state:** `Archive/Drafts/` and `Archive/evidence/` recreated empty.
`config.AppConfig` creates `archive_root` on init; `night_run.py` writes
`Archive/Drafts/draft_{id:06d}/` on import -- skeleton sufficient for next run.

## Part 4 -- SUSPENDED --full + --fast

**session_baseline_check --full:**
```
full-baseline                SUSPENDED full baseline SUSPENDED pending new anchor (Archive cleared 2026-07-15; golden reference offline at C:\ASTRO\backups\vyvar_anchor_424_sigma_floor_20260713_core-bf3743a1.zip)
OVERALL: SUSPENDED
```
Exit code: 0

**session_baseline_check --fast:** OVERALL PASS (852 passed, 15 skipped)

**Ledger:** VL-ANCHOR-424 `status: suspended_offline`, `passes: false`, `offline_backup` fields set.
VL-COUNTERS-ZERO `passes: false` (suspended with anchor).

## Errors (if any)

None.

## Files changed

- `scripts/session_baseline_check.py` (SUSPENDED --full via ledger)
- `validation/VYVAR_VALIDATION_LEDGER.json` (offline_backup + suspend)
- `tests/test_session_baseline_check.py` (new)
- `tests/test_validation_ledger.py` (optional offline_backup schema)
- `docs/VYVAR_STATE.md`, `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_JOURNAL.md`
- `CURSOR_RESULT_archive_cleanup.md`

## pytest

Full suite green: **855 passed**, 16 skipped (+4 new session_baseline tests; draft_426 integration test skipped post-wipe).

## Push status

**NOT PUSHED** -- awaiting Milan review of this result (deletion is irreversible).

---

**Next-session entry point:** Import first new measurement into empty `Archive/Drafts/`;
`git pull` -> STATE -> ROADMAP -> `session_baseline_check.py --fast`.
