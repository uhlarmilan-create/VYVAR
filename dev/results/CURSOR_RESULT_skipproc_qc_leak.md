CURSOR RESULT - 2026-07-22 (analysis only)

What I did

Read-only investigation of draft_000448 (MN Boo, NoFilter_60_2, 150 lights,
`skip_processed_directory=true`): FITS header census, qc_metrics.csv, OBS_FILES DB
query, infolog grep, and code-path trace for preprocess / alignment / phase2a /
report FWHM sources. No source, test, doc, or draft file modifications.

---

## Symptom recap

- `calibrated/lights/qc_metrics.csv` (written 2026-07-21 20:12) has **130 rows**, all
  `status=ok`, `src==dst` (in-place paths under `NoFilter_60_2/`).
- **20 frames absent** from the CSV (not listed as rejected): 16, 28, 40, 52, 60,
  78, 80, 86, 102, 107, 108, 112, 130, 131, 132, 133, 137, 147, 149, 150.
- `detrended_aligned/lights/NoFilter_60_2/` contains **150 light FITS + MASTERSTAR.fits**
  (151 files total) — all 20 bad frames present downstream.
- Report “FITS Quality Assessment” plots **150 frames**; red FWHM outliers match the
  20 missing-from-CSV frames; dashed limit ? **5.05 px**; plot FWHM scale ? 4.9–5.7 px.
- CSV `fwhm_px` scale ? **3.2–3.8 px** (frame 001 = **3.239**) — different estimator
  than the report plot.

*(Note: task text cited `calibrated/lights/NoFilter_60_2/qc_metrics.csv`; on disk the
CSV is at `calibrated/lights/qc_metrics.csv`, one level above the obs-group folder.)*

---

## E1 — VY_QC header census

**Scope:** all 150 FITS in
`Archive/Drafts/draft_000448/calibrated/lights/NoFilter_60_2/`

| Category | Count |
|----------|------:|
| `VY_QC=ok` (+ `VY_FWHM`, `VYVARPR=True`) | **130** |
| `VY_QC=rejected_*` | **0** |
| **Missing** `VY_QC` / `VY_FWHM` / `VYVARPR` | **20** |

The 20 missing-header frames are exactly the 20 absent from qc_metrics.csv:

| Frame | VY_QC | VY_FWHM | VYVARPR |
|-------|-------|---------|---------|
| 016 | MISSING | — | — |
| 028 | MISSING | — | — |
| 040 | MISSING | — | — |
| 052 | MISSING | — | — |
| 060 | MISSING | — | — |
| 078 | MISSING | — | — |
| 080 | MISSING | — | — |
| 086 | MISSING | — | — |
| 102 | MISSING | — | — |
| 107 | MISSING | — | — |
| 108 | MISSING | — | — |
| 112 | MISSING | — | — |
| 130 | MISSING | — | — |
| 131 | MISSING | — | — |
| 132 | MISSING | — | — |
| 133 | MISSING | — | — |
| 137 | MISSING | — | — |
| 147 | MISSING | — | — |
| 149 | MISSING | — | — |
| 150 | MISSING | — | — |

---

## E2 — Infolog trace

**File:** `Archive/Drafts/draft_000448/infolog_20260721_213235.txt` (8005 lines,
saved 21:32 UTC; preprocess QC completed ~20:12 per `qc_metrics.csv` mtime).

### Requested log lines

| Pattern | Occurrences | Notes |
|---------|------------:|-------|
| `skip_processed_directory=True - running QC in-place on draft lights` | **0** | Preprocess ran before infolog capture window |
| `QC in-place: N ok, M rejected, K errors` | **0** | Same |
| `skip_processed: using X/Y lights FITS (VY_QC=ok)` | **0** | Same |
| `qc_metrics.csv written to ...` | **0** | Same |
| `[RUN VYVAR] Auto FWHM limit=...` | **0** | Analyze/preprocess not in saved infolog |
| `DEBUG: Preprocess DB filter selected ...` | **0** | Same |

**Infolog gap:** saved infolog begins with platesolve/catalog work (~18:14 UTC) and
first RUN VYVAR line is already alignment step 2/303. Preprocess + Analyze logs
(from ~18:08–20:12) were not retained in the ring buffer when infolog was flushed at
21:32.

### Indirect infolog evidence (alignment + phase2a)

```
18:16:58  [RUN VYVAR] [2/303] detrended_aligned/lights: pripravujem zarovnanie (150 snimok z NoFilter_60_2/...)...
```

- **150** per-frame alignment log lines (`zarovnanie MN_Boo_Light_*.fits`) ? **150/150**
  lights aligned, consistent with a **non-filtering** VY_QC gate (not 130/130).

```
19:11:29  [RUN VYVAR] Faza 2A: ProcFrameStore 150 CSV - vypocet FWHM / apertur...
19:24:23  [RUN VYVAR] Faza 2A hotovo: 154 kriviek z 150 snimok -> photometry
```

Phase 2A consumed **150** proc CSVs (all aligned frames including the 20 outliers).

### Reconstructed preprocess values (from artifacts + DB, not from infolog)

| Expected log | Reconstructed value |
|--------------|---------------------|
| Auto FWHM limit | **5.045 px** (`median 4.97 + 1.5×?_MAD 0.0496`; k=1.5 from `config.json`) |
| DB filter selected | **130 rows** at FWHM ? 5.045 |
| QC in-place summary | **130 ok, 0 rejected, 0 errors** (130-row ok-only CSV) |
| skip_processed alignment filter | **150/150** (inferred from alignment count) |

---

## E3 — CSV writer identification

**Writer:** `_qc_enrich_calibrated_in_place` (`src_py/pipeline.py` ~16594–16727),
invoked from `preprocess_calibrated_to_processed` when `skip_processed_directory=True`
(~16780–16796).

Evidence:

1. CSV path `calibrated/lights/qc_metrics.csv` matches `_qc_csv = calibrated_root / "qc_metrics.csv"` with `calibrated_root = resolve_draft_lights_root()` ? `calibrated/lights`.
2. All rows `src==dst` (in-place signature of `_qc_enrich_calibrated_in_place`).
3. **130 ok-only rows** imply `only_paths` was a **130-frame subset** — the in-place
   writer would include rejected rows if those frames were visited.
4. **Alternate writers ruled out:**
   - `app.py` ~767–769 and `night_run.py` ~293–295: guarded by
     `not skip_processed_directory` ? do **not** write in skip mode.
   - No second `qc_metrics.csv` on disk; single mtime 20:12.

**130-frame subset producer:** `calibrated_paths_for_draft_apply_filters`
(`pipeline.py` ~2291–2403), called from `_vyvar_execute_preprocess_pending`
(`app.py` ~709–748) with:

- `quality_filter_draft_id=448`
- `fwhm_max_px = pending["fwhm_limit_px"]` (Auto FWHM from Analyze)

DB query (replayed on `vyvar.sqlite3`, draft 448):

```
auto_limit = 5.045 px  (k=1.5, n_kept=130, n_cut=20)
DB filter pass count = 130
qc_metrics.csv row count = 130
symmetric diff = 0
```

**FWHM source for selection:** `OBS_FILES.FWHM` populated during Analyze RAM QC by
`_quality_inspection_dao_metrics_array` (`run_draft_ram_calibration_qc_to_obs_files`,
~2168–2190). Range **4.865–5.723 px**, median **4.970 px**.

The 20 excluded frames all have OBS `FWHM > 5.045` (e.g. frame 016 = 5.37, 131 = 5.72).

---

## E4 — Alignment entry point

**UI RUN VYVAR path** (`app.py` ~532–543):

1. `_vyvar_execute_preprocess_pending` ? `preprocess_calibrated_to_processed(..., only_paths=p1)`
2. `_vyvar_execute_platesolve_pending` ? `astrometry_align_and_build_masterstar(...)`
   (`app.py` ~891)

**Alignment does pass through the VY_QC filter** at `pipeline.py` ~14304–14310:

```python
if _skip_processed_directory(_cfg_align_root):
    files_all = [fp for fp in files_all if _get_vy_qc_status(fp) == "ok"]
```

**Not bypassed.** Input root with skip mode:
`_archive_preprocess_lights_root` ? `calibrated/lights/NoFilter_60_2` (~1151–1167).

**Why 150 still align:** `_get_vy_qc_status` (~1118–1124) is **fail-open**:

```python
return str(hdul[0].header.get("VY_QC", "ok")).strip().lower()  # missing ? "ok"
```

20 frames without `VY_QC` pass the filter; 130 stamped `VY_QC=ok` also pass ? **150/150**.

---

## E5 — Downstream gates

**Phase 2A** (`photometry_core.py` ~8077–8123): frame list from `proc_*.csv` /
`ProcFrameStore` under `platesolve/`. Optional gates exist but are **off by default**:

- `frame_quality_gate_enabled` ? transparency/PSF-collapse gate
- `frame_align_residual_gate_enabled` ? alignment-residual gate

No `VY_QC` or `qc_metrics.csv` exclusion after alignment. Phase 2A trusts platesolve
output; infolog confirms **150 CSV frames** processed.

**Photometry report QA** reads OBS_FILES `FWHM_PX` for plots (`photometry_report.py`
~3432–3462) — informational only, does not gate science.

**Leak propagation:** confirmed — bad frames reach `detrended_aligned`, proc CSVs,
and phase 2A light curves.

---

## E6 — FWHM estimator split

| Role | Estimator | Code | Typical scale (draft_448) |
|------|-----------|------|---------------------------|
| **(a) qc_metrics.csv / VY_FWHM header** | `_qc_fwhm_elongation` on **full calibrated frame** (photutils segmentation, semimajor/minor ?) | `pipeline.py` ~15901+ | **3.21–3.77 px**, median **3.48**; frame 001 = **3.239** |
| **(b) Report plot + DB FWHM + 130/150 selection** | `_quality_inspection_dao_metrics_array` — DAOStarFinder on **center crop**, moment FWHM | `pipeline.py` ~1378+; stored via `run_draft_ram_calibration_qc_to_obs_files` | **4.865–5.723 px**, median **4.970**; plot limit **5.045** |

**Which produced 130/150 selection:** **(b)** OBS_FILES DAO FWHM via
`calibrated_paths_for_draft_apply_filters` with Auto FWHM limit.

**Limit applied:**

| Key | Value (this run) |
|-----|------------------|
| `auto_fwhm_enabled` | true (default) |
| `auto_fwhm_k_factor` | **1.5** (`config.json`) |
| Computed `auto_limit` | **5.045 px** |
| `qc_fwhm_limit` | 8.0 (in-place reject threshold; **never triggered** — max CSV FWHM 3.77 ? 8.0 and ? 5.045) |

**Critical nuance:** even if in-place QC had visited all 150 frames, **zero** would
have been rejected by `fwhm_reject_limit=5.045` because `_qc_fwhm_elongation` reports
~3.2–3.8 px on this data. Subset selection happened **only** at the DB pre-filter;
in-place QC’s job was header stamping, and 20 frames were never stamped.

Report limit line (~5.05 px) comes from `_qa_fwhm_limit_px` ? `compute_auto_fwhm_limit`
on OBS `FWHM_PX` (`photometry_report.py` ~3210–3252), same k=1.5 logic as Analyze.

No run-local `config_snapshot` artifact found on draft tree; limit reconstructed from
repo `config.json` + OBS_FILES replay (matches reported ~5.05 px).

---

## Root-cause verdict

**CONFIRMED** (with one refinement).

The working hypothesis is correct: skip-mode in-place QC ran on a **130-frame
`only_paths` subset** pre-filtered by OBS DAO FWHM (Auto limit **5.045 px**), leaving
20 frames with **no `VY_QC` header**. The alignment-stage filter at ~14304 is present
but **fail-open** on missing headers (`_get_vy_qc_status` ? `"ok"`), so all **150**
frames entered alignment and photometry.

**Refinement:** this is not merely “subset QC + fail-open” — the dual FWHM estimators
mean in-place QC **could not** have rejected the DAO-flagged outliers even on a full
150-frame visit (segmentation FWHM ~3.5 px vs DAO ~5.0 px and limit 5.045). The
effective gate was the DB pre-filter at preprocess time; header stamping failed to
record exclusion for the 20 skipped frames, and alignment had no fail-closed fallback.

**Ruled out:**

- **E4 bypass:** UI uses `astrometry_align_and_build_masterstar` with the ~14304 block.
- **E5 second CSV writer:** only `_qc_enrich_calibrated_in_place` writes in skip mode;
  app/night_run writers are gated off.

---

## Candidate fix directions (list only — not implemented)

1. **Fail-closed `_get_vy_qc_status` in skip mode** — missing / stale `VY_QC` ?
   `"rejected_missing_qc"` or `"unknown"`, not `"ok"`.
2. **Full-set in-place QC visitation** — ignore `only_paths` for header stamping when
   `skip_processed_directory=True`; apply DB FWHM exclusion separately and stamp
   `VY_QC=rejected_fwhm` on excluded frames.
3. **Explicit reject stamping for DB-filter skips** — when building `only_paths`,
   write `VY_QC=rejected_fwhm` (+ reason header) onto excluded calibrated FITS.
4. **Alignment-side join** — filter against OBS_FILES / qc_metrics / Auto FWHM limit
   in addition to (or instead of) header-only check.
5. **FWHM estimator unification** — use one estimator for DB filter, in-place QC,
   report plot, and reject limits (or store both with explicit roles in headers/CSV).
6. **Infolog durability** — persist preprocess QC summary to draft artifact
   (`pipeline_meta.json`) so skip-mode audits do not depend on ring-buffer timing.

---

## Blast-radius notes

| Consumer | Uses VY_QC / qc_metrics? | Impact if unfixed |
|----------|--------------------------|-------------------|
| `astrometry_align_and_build_masterstar` (~14304) | **VY_QC header** (fail-open) | Bad frames align |
| `resolve_masterstar_input_root` / skip routing | calibrated lights root | Reads same tree |
| Phase 2A | proc CSV enumeration only | All aligned frames photometered |
| `find_qc_metrics_csv` | CSV for meta / INV-FLAT-01 | Incomplete CSV understates rejects |
| Report QA plots | OBS FWHM (not VY_QC) | Shows outliers but does not gate |
| PSF runner | `VY_FWHM`, `VY_QCRMS` headers | May use wrong/stale QC on leaked frames |

**Anchor impact (draft_435):** Anchor #3 / `--full` gate runs **draft_435** through
the legacy **`skip_processed_directory=false`** processed-directory flow (see
`docs/VYVAR_JOURNAL.md` skip_processed arc vs anchor runbook). Current repo
`config.json` has `skip_processed_directory: true`, but anchor protocol exercises the
processed/ path; this specific leak pattern requires **skip mode + DB subset QC +
fail-open headers** — not exercised by the draft_435 anchor unless skip mode is
explicitly enabled for anchor reruns.

---

## Errors

None (analysis completed).

## Files changed

None (read-only task).
