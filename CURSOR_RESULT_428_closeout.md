CURSOR RESULT — 2026-07-15 14:05 UTC+2 (F-428 closeout)

What I did
Reworked A3 diagnostic (v2), verified comp-selection isolation (T2), committed F-428 batch locally.
T4 revalidation deferred to Milan UI re-run on clean tree.

---

## T1 — A3 diagnostic v2

**Script:** `scripts/diag_428_unmatched_sep.py`  
**Output:** `tmp/f428_unmatched_sep_v2.txt`

### Root-cause of v1 inconsistencies

| Symptom | Root cause | v2 fix |
|---------|------------|--------|
| p50 ~81″ looked “impossible” | v1 used 0.05° SQL box ORDER BY without cos(dec); brute-force v2 **confirms** p50=81.053″ is real for unmatched DET_* (not a query bug) | Field-bounded Gaia load (13427 rows) + vectorized cos(dec) NN |
| Self-match 26″ with same catalog_id | v1 reported **nearest sky DAO** not **same-cid ms row** | Self-match joins vt.cid → ms.cid; violations are vt VSX coords vs ms WCS coords (5–60″ systematic) |
| 163 vs 69 count | v1 used wrong active_targets path + undefined population | Excluded = vt catalog_id NOT IN active_targets (dedup) = **69**; census excluded sum = **67** (±2 ID normalization) |
| mag=nan | v1 read empty `mag` column only | Fallback `mag` → `phot_g_mean_mag` → `vsx_mag_max`; 65/69 used `vsx_mag_max` |

### v2 header echoes
- Gaia DB: `C:\ASTRO\python\VYVAR\GAIA_DR3\vyvar_gaia_dr3.db`, table `gaia_dr3`, 211712600 total rows
- Field bbox: RA [205.632, 213.320] DEC [39.150, 43.088], **13427** rows queried
- Coords: `ra_deg`/`dec_deg` only (not x/y)

### Unmatched DET_* nearest-Gaia (2724 rows)
| Stat | Value |
|------|-------|
| p50 | **81.053″** |
| p90 | 132.006″ |
| within 1× FWHM (6.234″) | 0/2724 |
| within 2× FWHM | 5/2724 |

Matched masterstars vs Gaia DB by source_id: p50 **2.05″** (sanity: astrometry OK for matched rows).

### DIAG SELF-CHECK: **FAIL** (expected — valid decision input)
1. median 81.053″ ≥ 15″ gate — unmatched DET detections are ~80″ from nearest catalog source on median (why they remain unmatched).
2. 179/249 vt↔ms self-match violations > 2″ — VSX export coords vs platesolved masterstar coords diverge (not a join bug).

**Radius decision:** OPEN (Milan). v2 numbers replace v1.

### Excluded population
- `variable_targets`: 244 unique catalog_id
- `active_targets`: 180 unique catalog_id
- **Excluded set (dedup): 69**
- Phase-0 census: out_of_frame=47, no_dao=17, no_catalog_id=1, saturated=2 → sum **67**
- Heuristic class breakdown: out_of_frame=48, no_dao_gaia_match=20, no_catalog_id=1

---

## T2 — Comp-selection isolation

**Evidence:** `comp_selection_per_target.py:378,523,542` filter `ms["vsx_known_variable"]` from **masterstars** DataFrame. Line **432–433** additionally excludes `variable_target_catalog_ids` (all 244 `variable_targets.csv` Gaia IDs). **Does not read `variability_candidates.csv`.**

| Metric | draft_428 |
|--------|-----------|
| masterstars `vsx_known_variable=True` | **46** |
| 8 known VSX candidates: ms flag | **All 7 located in vt → False** |
| 8 known VSX: in `variable_target_catalog_ids` | **7/7 True** (CSS uses cid `1498486880958321024` in active_targets) |

**Escalation (partial):** Masterstar-level `vsx_known_variable` is **False** for BO CVn, SS CVn, FZ CVn, FY CVn, FU CVn, RX CVn, NSVS 5096293 — a separate stamping gap from F-428-VSXFLAG (variability export path). **Comp pools on draft_428 were still protected** by the `variable_target_catalog_ids` hard filter (line 432). Science impact on **variability field calibration** (which masks `vsx_known_variable` from detector output) is the F-428-VSXFLAG scope; comp-pool contamination from the candidates CSV bug is **not evidenced**.

**T4 may proceed** (Milan re-run); flag stamping should be verified post-fix on clean tree.

---

## T3 — Commit

**Hash:** `adbeca31de33a44ded83f1cd58cd0354ebc3a991` (`adbeca3`)  
**Status:** committed locally; **push pending Milan OK** (per task).

---

## T4 — Revalidation (PENDING Milan)

Blocked on Milan end-to-end UI re-run on committed tree. Checklist for post-run Cursor validation documented in task; includes:
- `pipeline_meta.json`: `git_dirty=false`, T3 hash, `entry_point=run_phase2a`
- LC/proc byte-identity vs pre-fix draft_428
- 8 VSX absent from candidates; AC columns; excluded_targets.csv; infolog UTC; PDF overflow 0
- A-durable MP live test during alignment

---

## T5 — pytest

**859 passed**, 16 skipped (unchanged from fix batch).

## Files changed (closeout delta)
- `scripts/diag_428_unmatched_sep.py` (v2 rework)
- `tmp/f428_unmatched_sep_v2.txt` (generated)
- `CURSOR_RESULT_428_closeout.md`
- `docs/VYVAR_STATE.md`, `docs/VYVAR_ROADMAP.md`, `CHANGELOG.md`
