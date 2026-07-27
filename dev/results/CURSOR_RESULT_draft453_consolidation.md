# CURSOR RESULT - DRAFT-453 CONSOLIDATION (2026-07-27)

READ-ONLY timing + equivalence analysis for BO CVn drafts 451/452/453.
Raw data committed under `dev/results/context/session_20260727/` (**4.78 MB**).

---

## 0. Premise check (rule 0.1)

**What is compared with what:** Three full-path runs of the **same BO CVn raw night** on the **same code
line** (`13fc305`), differing by **entry point** (UI vs headless) and, for 451 only, **sky-surface
state** (pre-`ff08002` vs post-fix). **452 vs 453** isolates entry point at matched sky-surface /
masterstar census (2951 rows, 201 actives). **451 vs 453** isolates sky-surface effect at the same
UI entry point but confounded by masterstar inflation on 451 (6698 vs 2951).

**Comparable targets:** Wall-time targets must come from the **same measurement source** per draft.
The task table's "> 2 h" for 453 is **not comparable** to completed-run folder timestamps or the
452 night-run `SUCCESS` line (measured after completion). Infolog-derived phase spans are **tail-only**
(8000-line ring buffer) and omit calibration/preprocess on UI runs  -  not comparable to 452's full
`[NightRun] [OK]` profile without stating that gap.

---

## 1. Comparison table  -  confirmed / corrected

| Draft | Entry | Sky-surface | Masterstars | Actives | Wall time (measured) |
|-------|-------|-------------|------------:|--------:|---------------------:|
| 451 | UI | **no** | 6698 | 242 | **83.4 min** (folder ctime first?last artifact) |
| 452 | headless | yes | 2951 | 201 | **118.8 min** (`SUCCESS in 7129.1s`, `tmp/c4_night_run_452.log`) |
| 453 | UI | yes | 2951 | 201 | **76.0 min** (folder ctime first?last artifact) |

**Corrections to the task table:**

- **452 wall time:** **118.8 min**, not previously reported in the remediation doc.
- **453 wall time:** Completed run **~76 min**, not > 2 h. Milan's "> 2 h still running" note was
  either mid-run or included non-pipeline wall clock; the finished draft timestamps do not support
  > 2 h pipeline time.
- **451 ~88 min:** Measured **83.4 min**  -  close; same order of magnitude.
- Counts (6698 / 2951 / 201 / 242): **confirmed** from CSV row counts on disk.

**Design note (451 vs 453 confound):** 451 did **more** photometry work (6698 masterstars, 242
actives) yet finished **faster** than 453 (83 vs 76 min is close; both much faster than headless 452).
Any "sky-surface made UI slower" claim is **not supported** by total wall clock; the confound dominates.

---

## 2. Timing profile (primary deliverable)

### 2.1 Sources

| Draft | Timing source | Caveat |
|-------|---------------|--------|
| 452 | Full `tmp/c4_night_run_452.log` `[NightRun] [OK]` markers | Complete pipeline |
| 451, 453 | `infolog_*.txt` tail + folder ctime | **First ~8000 lines lost** (calibration, preprocess, INV-PREP-01); infolog starts mid?MASTERSTAR build |

Artifacts: `phase_timing.csv`, `phase_markers_{451,452,453}.txt`, `config_snapshot_{451,452,453}.json`.

### 2.2 Draft 452  -  full headless phases (seconds)

| Phase | duration_s | n_items |
|-------|-----------:|--------:|
| calibration | 149.2 | 150 frames |
| preprocess | **2743.1** | 150 frames |
| platesolve_align_masterstar | 859.0 | 139 frames |
| photometry_NoFilter_60_2 | **2549.0** | 201 targets |
| (other steps) | < 10 |  -  |
| **wall_clock_total** | **7129.1** |  -  |

**Per-frame preprocess (452):** 2743.1 / 150 = **18.3 s/frame** (includes in-place sky-surface + QC).

### 2.3 Draft 453  -  infolog tail blocks (seconds, incomplete)

| Phase block | duration_s | n_items |
|-------------|-----------:|--------:|
| masterstar_build_tail (visible only) | 6 | 2951 |
| alignment | 857 | 139 |
| phase0_phase1_window | 2177 | 201 |
| phase2a | 805 | 201 |
| reporting_export | 1040 | 201 |
| wall_clock_folder_ctime | 4562 |  -  |

Preprocess/calibration: **not in retained infolog** (not measurable from this artifact).

### 2.4 Draft 451  -  infolog tail blocks (seconds, incomplete)

| Phase block | duration_s | n_items |
|-------------|-----------:|--------:|
| masterstar_build (visible) | 108 | 6698 |
| phase0 (visible) | 2894 |  -  |
| phase1 | 1793 |  -  |
| phase2a | 1026 | 242 |
| reporting_export | 549 | 242 |
| wall_clock_folder_ctime | 5002 |  -  |

### 2.5 Ranked phase deltas (where measurable)

**452 vs 453 (headless slower by 42.8 min wall clock):**

| Rank | Phase | ? (452 ? 453) | Evidence |
|------|-------|--------------:|----------|
| 1 | preprocess | **+2743 s (452 only)** | 452 log; 453 not in infolog |
| 2 | photometry | **+2549 s vs +805 s tail** | 452 log vs 453 infolog phase2a+reporting (~1845 s visible on 453) |
| 3 | platesolve+MS | **+859 s vs ~6 s tail** | 452 log; 453 infolog shows MS already built when log starts |

Headless **preprocess (45.7 min)** alone exceeds the entire UI 453 wall clock (76 min), so UI likely
**reused** calibrated/processed state or skipped re-import steps that headless 452 executed from Raw.

**451 vs 453 (451 slower by 7.4 min wall clock despite more targets):**

| Rank | Phase | ? (451 ? 453) | Evidence |
|------|-------|--------------:|----------|
| 1 | phase2a | **+221 s** | 1026 s vs 805 s (242 vs 201 actives) |
| 2 | phase0/1 window | **+717 s** | 2894 s vs 2177 s (inflated masterstar era vs restored) |
| 3 | alignment | **+287 s** | 1144 s vs 857 s |

**Which phase accounts for 452 slowness:** **`preprocess` (2743 s)** and **`photometry` (2549 s)** on
452; together **~88 min** of the **119 min** total. UI path total wall **76 min** implies major
upstream work was not repeated or not logged.

---

## 3. Named suspects

| Suspect | Verdict | Seconds / notes |
|---------|---------|-----------------|
| **Sky-surface in-place preprocess (`ff08002`)** | **Confirmed expensive on 452** | 2743 s total, 18.3 s/frame. **Not measurable on 453** (preprocess evicted from infolog). Cannot attribute 452?453 gap to sky-surface alone without 453 preprocess timing. |
| **`qc_preprocess_workers=8`** | **Config confirmed identical** on 451/452/453 (`config_snapshot_*.json`). Both paths log `up to 8 process worker(s)` (452 log line 16:03:20; 453 infolog 19:06:45). **Cannot prove** equal utilization without per-frame worker audit. |  -  |
| **`predicted_dilution_factor`** | **Not measurable** | Zero log hits in 452 log or 453 infolog. |
| **`INV-PREP-01` guard** | **Not observable on 453** | 0 infolog hits; preprocess phase not retained. **Cannot confirm** one-frame-per-obs_group from timestamps. |
| **`INV-MS-01` guard** | **Confirmed on 453** | `INV-MS-01 MASTERSTAR purity guard: dao_only_fraction=0.037` at 19:05:40. |
| **VSX?Gaia mixture / Fleming fit** | **Excluded (negligible)** | Single `[DAO] Gaia->DAO reconcile ... fit=fleming1995_erf` line; long GAIA SQL spam is query progress logging during later phases, not a standalone fit loop. |

---

## 4. Equivalence check (452 vs 453)  -  STOP CONDITION

Script: `dev/scripts/draft_ui_equivalence_check.py draft_000452 draft_000453`
Full output: `dev/results/context/session_20260727/equivalence_452_vs_453.txt`
Semantic analysis: `equivalence_semantic.json`

### 4.1 Byte identity (strict criterion)

| Artifact | Count 452/453 | Byte-identical |
|----------|---------------|----------------|
| `lightcurve_*.csv` | 198 / 198 | **NO (0/198)** |
| `comp_quality_*.json` | 0 / 0 | n/a |
| `comparison_stars_per_target.csv` | 1 / 1 | **YES** |

**Stop condition triggered** under strict byte-identity rule.

### 4.2 Divergence decomposition (not averaged away)

1. **Schema:** 452 LC files include **`delta_mag_sysrem`** column (all NaN); 453 omits it. Headless
   452 ran with SysRem export stub; UI 453 did not write the column.
2. **BJD/HJD:** 70/198 files differ at **max |?| = 1.40?10??**  -  float representation only. Cause:
   `pipeline_meta.json` **`observer_location` lat/lon differs** between runs (452:
   50.073658/14.41854; 453: 50.112166/14.698255) despite same site name "Jirny".
3. **Shared photometry columns:** All non-time shared columns match within **1?10??** except path
   metadata in `photometry_summary.csv` (`lc_csv` / `lc_png` draft paths  -  201 rows).

**Catalog / target layer:** `active_targets.csv`, `variable_targets.csv`, `masterstars_full_match.csv`,
`comparison_stars_per_target.csv`  -  **byte-identical** between 452 and 453.

**Conclusion:** Not byte-identical; **scientific divergence is not established**  -  differences are
empty SysRem column, path strings, and sub-nanosecond JD formatting from site-coordinate metadata drift.
**Reference cut must not proceed** until byte identity or an explicit equivalence policy is decided.

### 4.3 Draft 453 infolog extracts (live proof lines)

```
VSX-GAIA XM: n_vsx=873 n_gaia=15085 ... masterstars=205/208 outcome=ok gaia_db_max_g=17.5
FAZA 0 funnel: vsx_bbox=875 -> in_frame=797 -> gaia_id_assigned=651 -> dao_detected=201 -> active=201 | excluded: ... not_target_eligible=596 out_of_frame=78 ...
INV-MS-01 MASTERSTAR purity guard: dao_only_fraction=0.037
[EXO TARGET] funnel: hosts_in_field=82 masterstars_in_frame=2842 promoted=3 sep_max=3 arcsec
```

**INV-PREP-01:** **absent** from saved infolog (preprocess evicted by 8000-line buffer)  -  **not** proof
the guard did not run; **is** proof the UI infolog is an incomplete operator record for early phases.

### 4.4 Acceptance table  -  draft 453 vs 452

| Quantity | 452 | 453 |
|----------|----:|----:|
| pass-1 DAO (453 meta) |  -  | 2552 |
| masterstars rows | 2951 | 2951 |
| DAO_ONLY fraction | 3.69% | 3.69% |
| bg_std | 83.82 ADU | 83.82 ADU |
| sigma_pp (unmasked MAD) | 46.90 ADU | 46.90 ADU |
| DAO threshold | 176.03 ADU | 176.03 ADU |
| active targets | 201 | 201 |
| exo promotions | 3 | 3 |

---

## 5. Data committed

Path: `dev/results/context/session_20260727/` (**4.78 MB**, under 5 MB cap)

| File | Purpose |
|------|---------|
| `draft_452_{active,variable,photometry_summary,comparison_stars,masterstars,pipeline_meta}.*` | 452 science snapshot |
| `draft_453_*` (same six) | 453 science snapshot |
| `phase_timing.csv` | Per-phase durations |
| `phase_markers_{451,452,453}.txt` | Raw log lines used for derivation |
| `equivalence_452_vs_453.txt` | Full equivalence script output |
| `equivalence_semantic.json` | Shared-column diff summary |
| `config_snapshot_{451,452,453}.json` | Workers / sky order / site coords |
| `summary.json` | Machine-readable run summary |

### Retention rule (proposed)

Keep `dev/results/context/session_YYYYMMDD/` for **30 days** or **until the next reference cut**,
whichever comes first; store only CSV/JSON/text (no FITS); cap **5 MB per session**; omit
`lightcurve_*.csv` blobs if over cap (SHA manifest instead).

---

## 6. Stop conditions  -  status

| Condition | Status |
|-----------|--------|
| Comparison table wrong in design-changing way | **Partial:** 453 wall time corrected (76 min, not > 2 h). Design stands. |
| Science files not byte-identical | **TRIGGERED**  -  see ?4. Timing conclusions for 452 vs 453 upstream preprocess are **blocked** until preprocess timing exists for 453. |
| Guard lines absent from 453 infolog | **Partial:** `INV-MS-01` **present**; `INV-PREP-01` **absent** (infolog buffer limit, not proven guard skip). |

---

## 7. Files changed

| File | Action |
|------|--------|
| `dev/results/CURSOR_RESULT_draft453_consolidation.md` | added |
| `dev/results/context/session_20260727/*` | added (22 files, 4.78 MB) |

No source code changes.
