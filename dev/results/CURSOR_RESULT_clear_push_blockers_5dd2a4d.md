# Draft 435 restore plan (FINAL -- awaiting Milan approval)

Generated: 2026-08-13 (B.1 resolved; B.2 pre-restore checksum written)

## Reference

- Whole-tree zip: `C:\ASTRO\backups\draft_000435_anchor_live_20260716.zip` (2026-07-16)
- Live tree: `Archive/Drafts/draft_000435` (2778 files at pre-restore checksum)
- Pre-restore checksum: `dev/validation/anchor_435_checksums_pre_restore_20260813.json`
- Snapshot sibling: `Archive/Drafts/draft_000435_snapshot_skysurface_20260716` (do not touch)

---

## B.1 Resolved: two undecided items

### `draft_manifest.json` -- **KEEP CURRENT**

| field | live (2026-08-11) | zip (2026-07-16) |
|-------|-------------------|------------------|
| `schema_version` | 3 | absent (pre-v3) |
| `updated_utc` | 2026-08-11T10:47:26+00:00 | 2026-07-16T09:15:28+00:00 |
| `rig`, `paths`, `status`, `center`, `files[]` | populated (150 files) | absent |

**What changed it:** manifest migration Phase 1a-3 (commits `d775af6` through `1c6ca81`,
Aug 2026), via `backfill_draft_manifests.py` / manifest-direct writes -- **not** the
Aug-12 killed photometry re-run. The Aug-11 timestamp is when v3 backfill ran on this
machine. Restoring the zip copy would revert to a pre-v3 stub and break manifest-first
UI/parity paths that `--fast` manifest-db-parity now depends on.

### `platesolve/NoFilter_60_2/_hrd_cache/summary.json` -- **KEEP CURRENT**

| field | live | zip |
|-------|------|-----|
| `generated_at_utc` | 2026-07-17T16:04:51+00:00 | 2026-07-16T10:31:52+00:00 |
| `git_head` | d4c7953 | 89842ff |
| `enrich_attempts` | identical | identical |

**Reasoning:** Jul-17 intentional regen (newer than zip anchor). Content matches;
only generation metadata differs. Not an Aug-12 photometry artifact.

---

## B.2 Pre-restore checksum (DONE)

Written: `dev/validation/anchor_435_checksums_pre_restore_20260813.json`

- 2778 files, sha256, size, mtime_utc
- git_head at write: `d758c83`
- Reversible baseline before any zip extraction

---

## B.3 Final restore plan (NOT EXECUTED -- approve to proceed)

### RESTORE from zip (549 files, Aug-12 mtime photometry outputs)

Extract from `draft_000435_anchor_live_20260716.zip` into live tree, **only** paths
under `platesolve/NoFilter_60_2/photometry/` that differ from zip:

| group | count | action |
|-------|------:|--------|
| lightcurve_sidecar_json | 238 | restore |
| export_reports (AAVSO/VarAstro) | 184 | restore |
| lightcurve_csv | 79 | restore (except BO CVn -- see KEEP) |
| photometry_png | 41 | restore |
| photometry_csv | 6 | restore |
| pipeline_meta.json | 1 | restore |

**Method:** for each changed member in zip under `draft_000435/platesolve/NoFilter_60_2/photometry/`,
overwrite live file if sha256 differs from zip (skip if already matches).

**Reason:** Aug-12 mistaken photometry re-run (killed). Only BO CVn LC was manually
restored afterward.

### KEEP CURRENT (no zip overwrite)

| item | reason |
|------|--------|
| `lightcurve_1498613634033133184.csv` (BO CVn) | already matches zip + `tmp/_435_lc_before_zpclip_rm.csv` |
| `draft_manifest.json` | v3 manifest backfill (B.1) |
| `_hrd_cache/summary.json` | Jul-17 regen (B.1) |
| All calibrated / raw / detrended / MASTERSTAR / inputs | hash-match zip already |
| `photometry/_report_cache/**` (838 JPG) | post-July UI cache; no zip member |
| `photometry/pdf_embed/**` | post-July embed cache |
| `draft_000435_snapshot_skysurface_20260716` | separate tree |

### Optional cleanup (defer)

No required deletes. Report caches can be pruned later if redundant.

### Post-restore verification

1. Write `dev/validation/anchor_435_checksums_post_restore_20260813.json`
2. Diff against pre-restore manifest; report changed paths
3. Confirm BO CVn LC unchanged (sha256 match pre-restore)
4. Confirm `--fast` still PASS (manifest-db-parity uses snapshot, not LC outputs)

---

## Risk statement

Low risk to anchor gate: `--fast` uses snapshot draft_435, not live photometry outputs.
Restore reverts display/validation artifacts to July anchor state while preserving v3
manifest and Jul-17 HRD cache.

**Approve this plan to execute B.4.**
