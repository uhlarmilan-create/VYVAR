# DELETE-OK list for Milan (ANCHOR-516-04)
# Cursor does not delete. Nothing was submitted; 515-era BO CVn files
# were never uploaded. Part B MAG 134/134 identity vs 515 is on record.

Date: 2026-08-18
New freeze (KEEP):
- Archive/Drafts/draft_000516
- Archive/Drafts/draft_000516_snapshot_cleanrebuild_20260818
- Archive/Drafts/draft_000516_p1mini

DELETE-OK (manual, after Milan confirms):
- Archive/Drafts/draft_000435
- Archive/Drafts/draft_000435_p1mini
- Archive/Drafts/draft_000435_snapshot_skysurface_20260716
- Archive/Drafts/draft_000436
- Archive/Drafts/draft_000437
- Archive/Drafts/draft_000509
- Archive/Drafts/draft_000513
- Archive/Drafts/draft_000514
- Archive/Drafts/draft_000515

Notes:
- Historical one-off tools under dev/tools/ and dev/scripts/ still name
  435 snapshot as a measurement target (wide_err_*, closure_step*,
  audit_stage3_*). Those are not live --fast/--full gates. They will
  break after 435 delete; they are not recut.
- Live pytest that touches retired drafts skip-if-missing (draft 514
  in test_comp_qa_pool_guard.py / test_forced_phot_and_weights.py).
  test_s2d photon-transfer fire-proof now uses draft 516.
- Offline zip C:\ASTRO\backups\draft_000435_snapshot_skysurface_20260716.zip
  is outside this list (Milan backup).
