CURSOR RESULT - 2026-07-16 - ARCHIVE BACKUP + CLEANUP + PUSH

What I did
Verified offline zips of draft_435 snapshot + live draft, deleted drafts 428-434, narrowed
empty_comp_drop allowlist to draft_435, moved F-428/F-431 evidence out of tmp/, committed +
pushed.

## B1 - Zip verification

### Snapshot -> `C:\ASTRO\backups\draft_000435_snapshot_skysurface_20260716.zip`
```
zip_size_bytes: 4818567858 (~4.82 GB)
zip_sha256: a35d22354666e359ce1bdd9a6eb207d5d768466a67fcdb77c22425eabb3f84a0
src_file_count / zip_file_count: 1932 / 1932
src_uncompressed / zip_uncompressed: 6396938155 / 6396938155
pipeline_meta git_hash: 10d610c0e79ddbd67f91b6c01b1073ca2d3099dd
pass: true
```

### Live -> `C:\ASTRO\backups\draft_000435_anchor_live_20260716.zip`
```
zip_size_bytes: 4818455802 (~4.82 GB)
zip_sha256: a4bb42d255e542b4a516197d5efe1a6304602b331680ac554caf41a244070faf
src_file_count / zip_file_count: 1932 / 1932
src_uncompressed / zip_uncompressed: 6396938155 / 6396938155
pipeline_meta git_hash: 10d610c0e79ddbd67f91b6c01b1073ca2d3099dd
pass: true
```

### VL-ANCHOR-424 historical (untouched)
`C:\ASTRO\backups\vyvar_anchor_424_sigma_floor_20260713_core-bf3743a1.zip` exists
(257667235 bytes).

## B2 - Deletion report

| Item | Action | Size |
|------|--------|------|
| draft_000428 ... draft_000434 | **deleted** | ~48.4 GB combined |
| draft_000435 | **kept** | - |
| draft_000435_snapshot_skysurface_20260716 | **kept** | - |
| tmp/.../pass1_photometry_backup | **deleted** | 0.309 GB |
| tmp/anchor_pair_run.log, anchor_pair_430_431* | **deleted** | ~0.001 GB |
| tmp/f428_*, f431_lost_transform*, headless_forensics | **moved** -> `validation/f428_arc_evidence/` | - |

**Freed total:** ~**48.71 GB**. Remaining drafts: only 435 + snapshot.

## B3 - Allowlist scope

**Was blanket** (`EXPECTED_EXCEPT_FIX_COUNTERS` global). **Narrowed** to
`EXPECTED_EXCEPT_FIX_COUNTERS_BY_DRAFT[435] = {phase2a_empty_comp_drop: 1}`. Other drafts /
other counts still FAIL. Test: `test_except_fix_allowlist_is_draft_scoped`.

## B4 / B5 - Git + STATE

Pushed to `origin/main`. HEAD `b962859`.

```
b962859 chore(archive): backup draft_435 anchor + purge drafts 428-434
0015d29 docs(journal): record LABBE-DET and Anchor #3 PASS
95f262e fix(qa): allowlist draft_435 empty_comp_drop in --full counters
ded815b chore(anchor): cut draft_435 sky-surface anchor and re-enable --full
10d610c fix(labbe-det): canonicalize ensemble SEM join and Labbe RNG purity
```

`--fast` OVERALL PASS (889 passed, 19 skipped) before push. Allowlist narrowing is in
`b962859` (same commit as archive chore; draft-scoped `EXPECTED_EXCEPT_FIX_COUNTERS_BY_DRAFT`).

STATE data items: zip paths + SHA256, freed size, darks ~2026-07-21 on top. JOURNAL closes
F-428->anchor arc.