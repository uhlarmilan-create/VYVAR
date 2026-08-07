CURSOR RESULT - 2026-08-07 (DAO-CLOSE)

What I did
Closed the DAO detection workstream: registry companion fix, VYVAR_PARAMS regen,
sigma_g unmeasurable-fraction + ZP RMS census reporting, confusion-blend measurement,
consolidated docs, P1 A/B at same HEAD, four commits, push.

## Acceptance checklist

| # | criterion | result |
|---|-----------|--------|
| 1 | Two P1 core SHAs identical | WITH `aa72e97979a74d5b8297c6bc3624bee668d8bd5f28624de0a708149e286c2636` n=325; WITHOUT (5306c6c, DAO-CLOSE reverted) same SHA n=325 |
| 2 | No group-b arcsec/fwhm_ratio companions | `[]` |
| 3 | test_generated_params_md_is_fresh | PASS |
| 4 | test_ascii_policy | PASS (new files ASCII) |
| 5 | Full suite except 3 named P1 ledger tests | 1259 passed; 3 failed (`test_headless_chain_sha`, `test_p1_snapshot_sha_matches_registered`, `test_p1_census_fingerprint_in_meta`) with VYVAR_INVARIANTS_P1=1; no other failures |
| 6 | artifact_negative draft_501 | 142 (test_draft501_artifact_negative_count PASS) |
| 7 | grep INV- invariants_runtime unchanged | 30; no WIRED_INV_IDS change |
| 8 | docs/VYVAR_DAO_DETECTION.md exists | yes; file:line citations resolve to src_py paths above |
| 9 | unmeasurable fraction per draft | 501: 0.388; 435: 0.842; 500: 0.959 (sigma_g>1.0) |
| 10 | A-6 + DAO threshold closed in roadmap | yes; reopen = two-rig sweep at matched calibration |

## Confusion-blend measurement

Tool: `dev/tools/dao_close_confusion_blend.py`
Results: `dev/results/dao_close_confusion_blend.json`

| draft | n unmatched_in_range | verdict |
|-------|---------------------:|---------|
| 501 | 26 | inconclusive (sample too small) |
| 435 | 81 | refuted_or_inconclusive |
| 500 | 455 | refuted_or_inconclusive |

Test rows show **lower** median local Gaia counts at 1-2 x FWHM than brightness-matched
controls (0 vs 1-2), and summed-neighbour implied G is ~20-24 mag fainter than detection
implied G -- opposite the blend-of-faint-stars prediction.

**Closure verdict:** wide-rig `unmatched_in_range` population is **undecidable at detection
stage** on current evidence. Future evidence (not queued): deeper catalogue, finer plate scale,
per-frame persistence with drift baseline.

## ZP fit RMS (verified)

| draft | RMS (mag) |
|-------|----------:|
| 501 | 0.431 |
| 435 | 0.837 |
| 500 | 0.946 |

## What remains unknown (acceptable to close)

Whether individual wide-rig DAO_ONLY detections are astrophysical, instrumental, or
deblended structure cannot be decided without data this pipeline does not have at detection
stage. VYVAR reports class, implied G, sigma_g, and unmeasurable fraction; consumption remains
gated by snr50_ok and photometry filters. Residual risk is LC quality and census visibility,
not silent catalogue corruption.

## Errors (if any)

None blocking.

## Files changed

- dev/tools/classify_params_scope.py, dev/validation/params_registry.json, docs/VYVAR_PARAMS.md
- src_py/dao_reconcile.py, src_py/photometry_report.py, dev/tests/test_dao_reconcile.py
- dev/tools/dao_close_confusion_blend.py, dev/results/dao_close_confusion_blend.json
- docs/VYVAR_DAO_DETECTION.md, docs/VYVAR_ROADMAP.md, docs/VYVAR_LIMITATIONS.md, docs/VYVAR_DECISIONS.md
- dev/results/CURSOR_RESULT_dao_close.md
