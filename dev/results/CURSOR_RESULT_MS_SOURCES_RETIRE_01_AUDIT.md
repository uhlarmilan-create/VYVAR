CURSOR RESULT - 2026-08-21 (MS-SOURCES-RETIRE-01 Phase 1 audit)

What I did
Read-only consumer/producer audit of `MASTER_SOURCES` across `src_py/`, column mapping vs
`masterstars_full_match.csv` (draft 517 fixture), `csv_only` path assessment, harness
dependency check, and release migration note. Measured `csv_only` ePSF prepare on live corrupt
DB (draft 517). No code or DB writes.

Baseline: local tip `6fec103` (EPSF-DB-01 Gaia amendment); task cites `f54d481` / pushed
`69f4f5e` - same EPSF-DB diagnosis series.

## Output / findings

### 1. Consumer / producer table

| # | File : lines | Role | Operation | Draft scope | Columns / fields used |
|---|--------------|------|-----------|-------------|------------------------|
| 1 | `psf_photometry.py` : 980?998, 1028?1037 | **Reader** (default ePSF) | `SELECT` safe-comp rows | `DRAFT_ID = draft_id` | `SOURCE_ID_GAIA`, `X_MASTER`, `Y_MASTER`, `IS_SAFE_COMP`, `EXCLUSION_REASON` |
| 2 | `psf_photometry.py` : 946?978 | **Reader** (grid ePSF only) | CSV path when `csv_only=True` | same draft CSV dir | `catalog_id`, `x`, `y` + CSV quality masks (no DB) |
| 3 | `psf_photometry.py` : 1066?1073 | **Indirect** | Targeting via `_epsf_allowed_catalog_ids` | platesolve dir | comparison_stars / active_targets CSVs (not DB); **skipped when `csv_only=True`** |
| 4 | `psf_photometry.py` : 634?683 | **Indirect** | `_epsf_augment_candidates_from_detected_pool` | aligned proc CSVs | Uses `db` only for gain/RN via `resolve_gain` ? **not** MASTER_SOURCES; **skipped when `csv_only=True`** |
| 5 | `database.py` : 1348?1416 | **Schema** | `_ensure_master_sources_table` CREATE/ALTER/INDEX | global | Full schema + migrations |
| 6 | `database.py` : 1418?1481 | **Writer** | `replace_master_sources_for_draft` DELETE+INSERT | `DRAFT_ID` | All insert columns (see ?2) |
| 7 | `database.py` : 1483?1497 | **Reader** | `fetch_master_sources_for_draft` SELECT * | `DRAFT_ID` | All columns |
| 8 | `database.py` : 1499?1522 | **Writer** | `update_master_source_safety` UPDATE | row `ID` | `IS_SAFE_COMP`, `EXCLUSION_REASON`, `SAFE_OVERRIDE` ? **defined but unwired** (zero callers in `src_py`) |
| 9 | `pipeline.py` : 14288 | **Writer** | MAKE MASTERSTAR ? `replace_master_sources_for_draft` | `draft_id` | In-memory `rows_ms` dict (see ?2) |
| 10 | `pipeline.py` : 14334?14348 | **Writer** | UPDATE out-of-common-field exclusions | `DRAFT_ID` | `IS_SAFE_COMP`, `EXCLUSION_REASON`, `X_MASTER`, `Y_MASTER` |
| 11 | `pipeline.py` : 14372?14440 | **Reader+Writer** | Stress-test RMS + unstable + VSX flag | `DRAFT_ID` | `STRESS_RMS`, `IS_SAFE_COMP`, `EXCLUSION_REASON`, `PHOT_CATEGORY`, `SOURCE_ID_GAIA`, `RA`, `DE` |
| 12 | `pipeline.py` : 6610?6628 | **Reader** | `write_photometry_plan_files` DB merge | `draft_id` | `SOURCE_ID_GAIA`, `LIKELY_NONLINEAR`, `ON_BAD_COLUMN` ? merged into comparison-star selection input |
| 13 | `ui_components.py` : 172?214 | **Reader** | Photometric Grid QA heatmap | `draft_id` | `G_MAG`, `BP_RP`, `STRESS_RMS`, `IS_SAFE_COMP`, `PHOT_CATEGORY`, `LIKELY_NONLINEAR`, `ON_BAD_COLUMN` |

**Grep coverage:** all `MASTER_SOURCES` / `fetch_master_sources` / `replace_master_sources`
hits in `src_py/` are listed above. Tests: `dev/tests/test_query_local_gaia_g3_f002.py:104`
comment only (no runtime read). No hits in `photometry_core.py`, `draft_provenance.py`, or
`session_baseline_check.py`.

**Production ePSF entry points** (`build_epsf_model`, default `csv_only=False`):
`app.py:1789`, `psf_runner.py:658`, `pipeline.py:15777`.

**Grid ePSF** already uses `csv_only=True` (`psf_photometry.py:1489` via `build_epsf_grid_model`).

---

### 2. Column mapping: MASTER_SOURCES vs `masterstars_full_match.csv`

**MASTER_SOURCES schema** (`database.py:1351?1374` + migrations `:1381?1404`):

`ID`, `DRAFT_ID`, `SOURCE_ID_GAIA`, `X_MASTER`, `Y_MASTER`, `RA`, `DE`, `G_MAG`, `BP_RP`,
`G_FLUX_ERROR_REL`, `NON_SINGLE_STAR`, `PHOT_VARIABLE_FLAG`, `FILTER_NAME`, `PHOT_CATEGORY`,
`RECOMMENDED_APERTURE`, `IS_VAR`, `IS_SATURATED`, `IS_SAFE_COMP`, `EXCLUSION_REASON`,
`STRESS_RMS`, `SAFE_OVERRIDE`, `LIKELY_NONLINEAR`, `ON_BAD_COLUMN`, `CREATED_AT`

**Draft 517 CSV columns** (45 cols, measured): includes `catalog_id`, `x`, `y`, `ra_deg`,
`dec_deg`, `mag`, `catalog_mag`, `phot_g_mean_mag`, `bp_rp`, `likely_saturated`, `is_saturated`,
`is_noisy`, `is_usable`, `photometry_ok`, `vsx_known_variable`, `gaia_dr3_variable_catalog`,
`sigma_g_row`, `zone`, `source_state`, `edge_safe_10px`, ? ? **does not include**
`likely_nonlinear`, `on_bad_column`, `is_safe_comp`, `phot_category`, `stress_rms`,
`exclusion_reason`, or `catalog_known_variable` (last is derived at runtime in ePSF code).

| MASTER_SOURCES column | CSV column / source | Gap? | Derivable from draft files? |
|-----------------------|---------------------|------|----------------------------|
| `SOURCE_ID_GAIA` | `catalog_id` | No | ? |
| `X_MASTER`, `Y_MASTER` | `x`, `y` | No | ? |
| `RA`, `DE` | `ra_deg`, `dec_deg` | No | ? |
| `G_MAG` | `phot_g_mean_mag`, `catalog_mag`, `mag`, `implied_g_mag` | No | ? |
| `BP_RP` | `bp_rp` | No | ? |
| `G_FLUX_ERROR_REL` | `sigma_g_row` (approx.; same Gaia query window) | Partial name | Gaia re-query or MAKE MASTERSTAR row |
| `NON_SINGLE_STAR` | ? | **Gap** | Gaia column at cross-match time (`pipeline.py` rows_ms) |
| `PHOT_VARIABLE_FLAG` | `gaia_dr3_variable_catalog` (bool, not flag text) | Partial | Gaia at cross-match |
| `FILTER_NAME` | ? | **Gap** | `draft_manifest.json` / setup name (`NoFilter_60_2`) |
| `PHOT_CATEGORY` | ? | **Gap** | Computed at MAKE MASTERSTAR (`filt_mag_X_col_Y`) |
| `RECOMMENDED_APERTURE` | ? | **Gap** | Computed at MAKE MASTERSTAR (`recommended_aperture_by_color`) |
| `IS_VAR` | `vsx_known_variable` \| `gaia_dr3_variable_catalog` | Partial | CSV bools |
| `IS_SATURATED` | `is_saturated`, `likely_saturated` | Partial | CSV |
| `IS_SAFE_COMP` | ? | **Gap** | Computed at MAKE MASTERSTAR (+ stress/bbox/VSX updates) |
| `EXCLUSION_REASON` | ? | **Gap** | Same pipeline block |
| `STRESS_RMS` | ? | **Gap** | `stress_test_relative_rms_from_sidecars` at MAKE MASTERSTAR |
| `SAFE_OVERRIDE` | ? | **Gap** | UI/manual; default 0 |
| `LIKELY_NONLINEAR` | ? | **Gap** | Detected during MAKE MASTERSTAR Gaia loop |
| `ON_BAD_COLUMN` | ? | **Gap** | Detected during MAKE MASTERSTAR |
| `ID`, `DRAFT_ID`, `CREATED_AT` | ? | DB metadata | Not needed for file-based science |

**Draft 517 spot-check:** `comparison_stars.csv` (47 cols) also lacks `likely_nonlinear`,
`on_bad_column` ? consistent with EPSF-DB-01 corruption blocking the DB merge in
`write_photometry_plan_files` (EXC-0347 silent pass).

**CSV proxies not used by ePSF today:** `zone` (linear/noise/saturated), `source_state`,
`edge_safe_10px` ? available but not wired to `IS_SAFE_COMP`.

---

### 3. Sidecar verdict

**For ePSF input (primary retire goal): NO JSON sidecar needed.**

Required ePSF fields are present in `masterstars_full_match.csv`:
`catalog_id`, `x`, `y`, `likely_saturated`, `photometry_ok`, plus optional
`is_saturated` / `is_noisy` / `is_usable`. Runtime derives `catalog_known_variable` from
VSX/Gaia bool columns (`psf_photometry.py:854?866`).

Measured on draft 517 (corrupt DB, read-only smoke):
- `csv_only=True` ? **2249** candidates after isolation (proceeds past old failure point)
- `csv_only=False` ? **`DatabaseError: database disk image is malformed`** on MASTER_SOURCES query

**For Photometric Grid QA UI:** sidecar **or** CSV column extension **needed if UI is kept**
? heatmap reads `IS_SAFE_COMP`, `PHOT_CATEGORY`, `STRESS_RMS` from DB only today.

**For comparison-star enrichment (`likely_nonlinear` / `on_bad_column`):** not DB-only ?
values are computed in MAKE MASTERSTAR before DB insert (`pipeline.py:14233?14282`) but
**not written back to CSV today**. Phase 2 should persist them to CSV (or a draft JSON export)
when retiring the writer, not rely on a runtime sidecar.

**Summary:** CSV is complete for ePSF with `csv_only` semantics; no JSON sidecar for ePSF.
Optional draft file export recommended for UI QA and photometry-plan merge (Phase 2 design).

---

### 4. `csv_only` assessment

| Aspect | `csv_only=False` (production default) | `csv_only=True` (grid ePSF today) |
|--------|--------------------------------------|-----------------------------------|
| Star positions | `MASTER_SOURCES` safe-comp (`IS_SAFE_COMP=1`, empty `EXCLUSION_REASON`) ? CSV quality ids | CSV quality rows with finite `x`,`y` |
| Join key | `SOURCE_ID_GAIA` ? `catalog_id` | `catalog_id` only |
| Targeting | `_epsf_allowed_catalog_ids` narrows to masterstar+comp worklist when enough stars | **Disabled** (`use_targeting and not csv_only`) |
| Broad-pool augment | Runs if safe-comp join `< min_stars` | **Skipped** |
| DB dependency | **Hard** on MASTER_SOURCES SELECT | **None** for star selection (db still used for FWHM manifest + gain/RN in augment path) |
| Fail-loud | DB malformed ? crash | Missing `x,y` ? `ValueError`; missing quality cols ? `ValueError` |

**Draft 517 counts:** CSV quality mask ? **2268** rows with finite `x,y`; after full
`_epsf_prepare_stars` isolation ? **2249** (`min_stars=30` satisfied).

**What is missing to ?make csv_only the only path? for global ePSF:**

1. **`build_epsf_model` must pass `csv_only=True`** (or remove DB branch) ? one-line call change.
2. **Behavior change vs legacy DB path:** global ePSF would use full CSV-quality frame coverage
   instead of DB safe-comp subset + targeting + broad-pool augment. Grid build already accepts
   this trade-off (comment at `:947?949`: DB join under-covers edge stars with non-Gaia ids).
3. **Optional parity:** if architect wants DB-equivalent safe-comp filtering without DB, Phase 2
   must either (a) add `is_safe_comp` / `exclusion_reason` columns to CSV at MAKE MASTERSTAR, or
   (b) document intentional widening to CSV quality gates only.
4. **`db` parameter** remains needed for FWHM fallback (`get_epsf_fwhm_from_context`) and
   equipment gain/RN ? not for MASTER_SOURCES.

**Verdict:** A production-grade CSV path **already exists** (`csv_only=True`); the fix is
mostly wiring + retiring DB read/write, not inventing new selection logic.

---

### 5. `--full` / `--fast` dependency check

| Harness | Opens DB? | Queries MASTER_SOURCES? | Migration impact |
|---------|-----------|-------------------------|------------------|
| `--fast` manifest-db-parity (`session_baseline_check.py:295?307`) | Yes (`VyvarDatabase`) | **No** | `_ensure_master_sources_table` runs on open (CREATE/ALTER/INDEX); **SELECT not used** ? explains EPSF-DB-01 `--fast` PASS with corrupt btree |
| `--full` anchor photometry (`:625?645`) | Yes | **No** | `run_full_photometry_pipeline` (`photometry_core.py`) has **zero** MASTER_SOURCES references |
| `--full` inputs | ? | ? | Uses frozen `masterstars_full_match.csv` + `variable_targets.csv` from anchor snapshot |

**Anchor recut expectation (`9902d918` byte-identity):** Phase 1 confirms anchor path **never
reads MASTER_SOURCES**. Recut should hold **if** Phase 2 does not alter photometry_core or anchor
inputs. **Must still confirm by recut after Phase 2** ? do not assume.

**Phase 4 P1 gap confirmed:** `--fast` does not run `PRAGMA quick_check`; corrupt MASTER_SOURCES
was invisible until ePSF RUN.

---

### 6. Release impact (VYVAR-release)

- Release repo ships the same `src_py/` tree (Milan copy per `VYVAR_RELEASE_RUNBOOK.md`).
- Existing user DBs may have populated or **corrupt** `MASTER_SOURCES` (today's incident).
- Phase 2 C3 requirement stands: stop creating the table; `DROP TABLE IF EXISTS MASTER_SOURCES`
  (+ indexes) on migration with **try/except** ? if DROP fails on corrupt btree, log and defer
  to Phase 3 file swap (`_rebuild_table_safely` pattern at `database.py:1245?1264`).
- App open must **not crash** when table is absent or corrupt-un-droppable.
- Phase 3 one-time scripted swap copies readable tables per EPSF-DB-01:
  `EQUIPMENTS`, `TELESCOPE`, `LOCATION`, `CALIBRATION_LIBRARY`, `FITS_HEADER_CACHE`,
  `OBS_QC_PROCESSING_RUN`, `OBS_QC_PROCESSING_FILE`, `FIELD_REGISTRY`, `COMP_STAR_LIBRARY`.
  Skip `LOCATION_OLD` (legacy mirror; EPSF-DB-01 lists 6 rows; no active write path ? confirm
  dead before Phase 3 copy list finalization).
- Users lose DB-stored MASTER_SOURCES per draft; per-draft truth moves to draft directory files
  (already the case for science outputs). No user action except optional DB hygiene on upgrade.

---

### 7. Phase 2?4 preview (blocked until architect review)

| Phase | Key work |
|-------|----------|
| **2 C1** | `_epsf_prepare_stars` / `build_epsf_model`: file-only star selection; fail-loud on missing CSV/columns |
| **2 C2** | Stop `replace_master_sources_for_draft`; migrate `write_photometry_plan_files` + Grid QA off DB |
| **2 C3** | Remove `_ensure_master_sources_table`; safe DROP migration |
| **3** | Fresh DB copy script; rename corrupt file to `vyvar.sqlite3.corrupt-20260821` |
| **4 P1** | `PRAGMA quick_check` in `--fast` (~1.2 s measured EPSF-DB-01) |
| **4 P2?P3** | JOURNAL/DECISIONS/ROADMAP; close EPSF-DB item |

**Operator constraint unchanged:** no DB-writing jobs until Phase 3 completes.

---

## Errors

None during audit. Live measurement: `csv_only=False` on draft 517 reproduces EPSF-DB-01
`DatabaseError` (expected).

## Files changed

- `dev/results/CURSOR_RESULT_MS_SOURCES_RETIRE_01_AUDIT.md` (this file)

**STOP ? Phase 1 complete. Awaiting architect review before Phase 2.**
