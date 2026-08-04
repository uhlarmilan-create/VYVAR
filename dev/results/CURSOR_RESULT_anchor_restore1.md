CURSOR RESULT - 2026-08-04 (ANCHOR-RESTORE-1)

What I did
Restored draft_000435_snapshot_skysurface_20260716 from the July offline zip after
quarantining the post-11:47 mutated tree. Verified zip integrity, extraction fidelity,
and ANCHOR-PROV characterisation. Documented restored photometry SHA. Updated JOURNAL,
ROADMAP, and STATE. Gate seeding files left untouched per FORBIDDEN list.

## A -- Preconditions

| check | result |
|-------|--------|
| Zip SHA256 | PASS -- a35d22354666e359ce1bdd9a6eb207d5d768466a67fcdb77c22425eabb3f84a0 |
| Free disk space (Archive volume) | PASS -- ~144.85 GB free (>= 12 GB required) |
| Process lock on snapshot path | PASS -- no python/streamlit holding files under snapshot |
| Scheduled task on Archive | none found |
| git rev-parse HEAD | e092218 |
| git status --porcelain | 3 untracked (non-blocking): CURSOR_RESULT_anchor_prov.md, CURSOR_RESULT_wide_err_step0_checkstar.md, wide_err_step0_checkstar.py |

All preconditions passed; proceed authorized.

## B -- Quarantine (preserve mutated generation)

**Quarantine path:**
Archive/Drafts/_quarantine/draft_000435_snapshot_MUTATED_20260804_1147

**Manifest:**
dev/results/anchor_restore/manifest_mutated_20260804.txt

| metric | value |
|--------|-------|
| photometry file count | 2999 |
| photometry total bytes | 846,001,768 |

Live tree moved (not copied, not deleted) before zip extraction.

## C -- Restore

**Source:** C:\ASTRO\backups\draft_000435_snapshot_skysurface_20260716.zip
**Target:** Archive/Drafts/draft_000435_snapshot_skysurface_20260716

Full fresh extraction (no merge into existing tree).

| verify metric | result |
|---------------|--------|
| photometry zip entries | 1196 |
| size mismatches | 0 |
| hash mismatches (50-file random sample) | 0 |

Zero mismatches -- restore faithful to zip archive.

## D -- Characterisation (restored tree vs ANCHOR-PROV expected)

| item | expected | actual | match |
|------|----------|--------|-------|
| check_kmag_*.csv count | 166 | 166 | YES |
| distinct check_catalog_id | 2 | 2 | YES |
| 1499906247391001088 count | 164 | 164 | YES |
| 1497528072458898432 count | 2 | 2 | YES |
| pipeline_meta git_hash | 10d610c0e79ddbd67f91b6c01b1073ca2d3099dd | 10d610c0e79ddbd67f91b6c01b1073ca2d3099dd | YES |
| pipeline_meta git_dirty | true | true | YES |
| admission_sat_peak_frac in config_snapshot | absent | absent | YES |

**New documented photometry SHA** (NOT a gate value):

Label: draft_435 snapshot, July generation, restored 2026-08-04, produced at 10d610c
with git_dirty=true.

| tier | SHA256 | n |
|------|--------|---|
| core | 3d26f4692ac81fc52db6ef9f70b148f9f7c56a5bb5e84e637339c4883ba47a96 | 333 |
| extended | 6420f1daa53a0d5d0a92bfd1ab30eba68e2ab88be8fe5f4c68048a5463054ac8 | 499 |

## E -- Docs

Updated:
- docs/VYVAR_JOURNAL.md -- ANCHOR-RESTORE-1 entry
- docs/VYVAR_ROADMAP.md -- ANCHOR-GATE-SEED, ANCHOR-CLEAN-BUILD (Not started)
- docs/VYVAR_STATE.md -- anchor section: July generation on disk; gate values do not describe it

FORBIDDEN items respected:
- session_baseline_check.py -- NOT edited
- test_invariants_p1_seed.py -- NOT edited
- VYVAR_VALIDATION_LEDGER.json -- NOT edited
- anchor gate -- NOT run
- quarantine tree -- NOT deleted

## What remains broken

1. **Gate seeding (ANCHOR-GATE-SEED):** session_baseline_check.py and test_invariants_p1_seed.py
   expect draft_000500 fingerprints (5bccd85a / 7fdcdca4) on the draft_435 snapshot path.
   Evidence: dev/results/CURSOR_RESULT_anchor_prov.md C3.

2. **No clean-build anchor (ANCHOR-CLEAN-BUILD):** no draft_435 generation has ever been
   produced from a clean git tree (July: 10d610c dirty; August mutated: 20dde2b dirty).
   session_baseline_check.py has no git_dirty guard.

3. **Unlogged writer (10:20-11:47):** identity of process that wrote into the snapshot path
   on 2026-08-04 remains unresolved. Evidence preserved in quarantine tree and manifest.

## Files changed

- docs/VYVAR_JOURNAL.md
- docs/VYVAR_ROADMAP.md
- docs/VYVAR_STATE.md
- dev/results/CURSOR_RESULT_anchor_restore1.md
- dev/results/anchor_restore/manifest_mutated_20260804.txt

Commit: f067304

## Errors

None.
