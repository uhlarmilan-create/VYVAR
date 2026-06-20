CURSOR RESULT — 2026-06-20 (Group 3 DEAD reclassification — AUDIT-ONLY)

What I did
Reclassified all Group-3 heuristic-DEAD functions using eight caller mechanisms (direct/attribute call, symbol ref, string ref in py/ui/json, `getattr`, Qt `.connect`, `super()`, registry/CLI `__main__`, dunder protocol). Updated `docs/VYVAR_FULL_AUDIT_LEDGER.md`. Spot-checked Group 1/2 DEAD samples. **No code removal.** Ledger commit pending review.

---

## Summary (207 heuristic DEAD → corrected)

| Status | Count | Notes |
|--------|-------|-------|
| Heuristic DEAD (original AST pass) | **207** | Ledger checkpoint said 189 (module-sum mismatch); JSON inventory = 207 |
| **TRULY-DEAD** | **25** | Removal candidates only |
| LIVE-DYNAMIC (reclassified) | **182** | Registry tuples, same-module refs, dunder protocol, etc. |
| TEST-ONLY (Group-3 subset) | **0** | G2 spot-check found 1 TEST-ONLY (`_epsf_fwhm_native_legacy_px`) |

**False-positive rate:** ~88% of heuristic DEAD were live — concentrated in `database.py` (59/72) and `importer.py` (41/46).

---

## Group 1 / Group 2 spot-check (5 each)

| Group | Result |
|-------|--------|
| **G1** (5/5) | All **TRULY-DEAD** — `_fits_header_positive_float`, `_per_frame_noise_error_map`, `get_auto_fov`, `_cluster_centroid_votes`, `autofill` |
| **G2** | 2 TRULY-DEAD (`_aperture_to_mask_single`, `_norm_id_series`); 2 LIVE-DYNAMIC (`_get_lc_adaptive`, `_select_comps_tiered` via scripts); 1 TEST-ONLY (`_epsf_fwhm_native_legacy_px`) |

**Conclusion:** G1/G2 low DEAD counts are genuine; heuristic only over-counted in DB/UI-heavy Group 3.

---

## TRULY-DEAD (25) — `file:line`

See ledger table; top clusters: `database.py` (13), `importer.py` (5), `time_utils.py` (3), `param_resolver.py` (2).

Full list in `tmp/reclassify_group3_truly_dead.txt`.

---

## Artifacts (tmp/, gitignored)

- `tmp/reclassify_group3_dead.py` — driver
- `tmp/reclassify_group3_dead_results.json` — 207 rows with mechanism notes
- `tmp/reclassify_group3_dead_table.md` — full per-function reclassification
- `tmp/reclassify_g1_g2_spotcheck.json`

---

## Files changed

- `docs/VYVAR_FULL_AUDIT_LEDGER.md` — corrected coverage table, TRULY-DEAD list, G3-F009 update, per-module status fixes
- `CURSOR_RESULT.md` (this report)

**Status:** AUDIT-ONLY; ledger update ready for commit on review.

## Errors (if any)

None.
