# CURSOR RESULT - POST-451 REMEDIATION (2026-07-27)

Three-part remediation from draft 451 analysis. Separate commits per part; `--fast` green after
each code commit; anchor `--full` re-verified after funnel expectation update.

---

## What I did

### Part A - Exoplanet promotion (`926a94c`)

**When:** Earliest symptom draft_450 (2026-07-24, no exo columns). Promotion code since Jun 2026;
path bug latent until full-path runs from non-repo CWD.

**Why silent:** `exoplanet_local_db_path` resolved with `Path.resolve()` against CWD, not
`data_root` (`config.py` / `pipeline.py:5043`). Missing DB -> empty promotion frame.
`_merge_vsx_exoplanet_variable_targets` (`pipeline.py:5229`) dropped exo columns when
`exo.empty` (merge defect compounded empty promotion).

**Columns vanish because:** both no rows AND merge dropped schema (latter fixed always-add columns).

**DB:** `exoplanets/vyvar_exoplanet_local.db`, table `exoplanet_data`, ~14185 rows; reachable.

**Fix:** resolve path in `config.__post_init__`, `require_exoplanet_local_db_path` fail-loud,
funnel log `[EXO TARGET]`, preserve exo columns on empty frame.

**Tests:** `dev/tests/test_exoplanet_promotion_restore.py` (7 passed with local match tests).

---

### Part B - Observability + skip_reason (`63b902d`)

**B.1:** `VSX-GAIA XM:` and `FAZA 0 funnel:` routed through `log_event`
(`vsx_gaia_crossmatch.py:689-703`, `photometry_core.py:12930+`).

**B.2:** `_propagate_phase2a_skip_reason_to_active` writes `ac_skip_reason` -> `skip_reason` on
active export when `n_frames=0`. NaN empty skip_reason no longer blocks write.

**Empty skip_reason counts (n_frames=0, empty skip_reason):**

| Draft | Count | Notes |
|-------|------:|-------|
| 435 snapshot | 1 | R CVn only (before fix, internal `no_comps`) |
| 450 | 1 | R CVn only |
| 451 | 1 | R CVn only |

**B.3 diagnostic (border / missing anchor targets):**

| Target | x, y (451 VT) | edge_margin=50 | 451 exclusion | Border-only? |
|--------|---------------|----------------|---------------|--------------|
| Gaia 1497436125799224960 | 1457.9, 573.0 | in frame | `not_target_eligible` | **No** |
| Gaia 1485913828055470592 | 35.5, 1224.6 | **out** (x<50) | `out_of_frame` | **Yes** |
| Gaia 1497121459315202560 | 1330.1, 1022.3 | in frame | **present in active** | N/A (not missing) |
| HAT-188-0000323 | 1674.4, 1053.6 | in frame | `not_target_eligible` | **No** |

**Reading:** Only Gaia2 is among `out_of_frame=78` with correct fixed-margin exclusion after
`cb78b25` BORDER defer. Gaia2 was **active on 435** at the same coords (annulus-aware safe bbox
included it). Gaia1/HAT losses are **not** border defects. **No stop** - heterogeneous causes,
not silent incorrect edge loss.

Anchor funnel expectation updated (`5078669`): `skip_reason_histogram` now
`{"": 162, "no_comps": 1, "zone_flag": 2}`.

---

### Part C - Sky-surface + guards

**C.1 (`ff08002`):** `_qc_enrich_calibrated_in_place` applies sky subtract when
`preprocess_sky_surface_order > 0` without `VY_CHANNEL`; mosaic Bayer skipped; stamps
`VY_SKYSF`, `VYSKYORD`, `VYSKYP2P`. Mono regression test in `test_osc1_extraction.py`.

**C.2:** Not implemented - pending C.4 measurement. Note: anchor `bg_std` 83.8 vs `sigma_pp`
46.1 => nominal 2.1 sigma threshold operates at ~3.8x pixel noise on anchor (documented, not
fixed here).

**C.3 (`1191579`):** `INV-PREP-01` (large_small_ratio WARN >10x) on one QC frame per obs_group;
`INV-MS-01` (DAO_ONLY WARN >0.10, FAIL >0.25) on masterstar CSV write. Registered in
`docs/VYVAR_INVARIANTS.md`; unit tests in `test_invariants_p2.py`.

**C.4 acceptance:** **NOT RUN** - no calibrated FITS under `Archive/Drafts/draft_000451` or
450 on this machine (CSV/infolog only). Preprocess byte-compare vs `8815c45` and full BO CVn
raw-to-photometry re-run blocked on Milan raw frame availability.

| Quantity | draft 451 | required | anchor | post-fix |
|----------|----------:|---------:|-------:|---------:|
| pass-1 DAO | 8926 | ~2550 | 2552 | pending |
| masterstars rows | 6698 | ~2950 | 2951 | pending |
| DAO_ONLY fraction | 40.4% | <10% | 3.7% | pending |
| bg_std | 62.2 | ~84 | 83.8 | pending |
| DAO threshold | 130.7 | ~175 | 175.4 | pending |
| active targets | 242 | ~160-175 | 165 | pending |
| Group A RMS ratio | 1.007 | ~1.0 | 1.0 | pending |

**C.5 (`5078669`):** `VYVAR_DECISIONS.md` (SKY-SURFACE-RESTORE), `VYVAR_INVARIANTS.md`
(guards), `VYVAR_STATE.md`, `VYVAR_ROADMAP.md` (F-431 T1 fixed). `flow_doc_facts.py` unchanged
(default `preprocess_sky_surface_order=2` already correct). FLOW PDF not regenerated this session.

---

## Gates

| Gate | Result |
|------|--------|
| `--fast` after A/B/C | PASS (1186 passed, 26 skipped) |
| `ruff` | clean on touched files |
| anchor `--full` after B (first run) | FAIL `full-phase0-funnel` (expected: R CVn `no_comps` now exported) |
| anchor `--full` after `5078669` | **PASS** (core `1c48d9fc...` n=325, extended `744bce94...` n=487, plan-regen 875, active 165) |

Anchor SHA checks unchanged on first run: core `1c48d9fc...` n=325, extended `744bce94...`
n=487, plan-regen 875 rows, active 165.

---

## Commits

| Commit | Part |
|--------|------|
| `926a94c` | A exoplanet |
| `63b902d` | B observability + skip_reason |
| `ff08002` | C.1 sky-surface |
| `1191579` | C.3 guards |
| `5078669` | C.5 docs + anchor funnel expectation |

---

## Errors

None blocking code delivery. C.4 blocked on missing raw FITS on disk.

---

## Files changed (committed)

- `src_py/config.py`, `database.py`, `pipeline.py` (A + C)
- `src_py/vsx_gaia_crossmatch.py`, `photometry_core.py` (B)
- `src_py/invariants_runtime.py` (C.3)
- `dev/tests/test_exoplanet_promotion_restore.py`, `test_post451_part_b.py`,
  `test_invariants_p2.py`, `test_osc1_extraction.py`
- `docs/VYVAR_*`, `dev/scripts/session_baseline_check.py`

---

## PRE-PUSH CLOSEOUT (2026-07-27)

### 0. Correction: draft 451 data was present

Listed `C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000451\` before any blocked declaration.
Contents: `Raw/` (150 FITS), `calibrated/lights/NoFilter_60_2/` (150 FITS),
`detrended_aligned/` (140 FITS), `platesolve/`, `infolog_20260727_104233.txt`,
`draft_manifest.json`. The earlier "no calibrated FITS" report was wrong.

---

### 1. C.4 acceptance run (draft_452)

**1.1 Preprocess byte-compare (PASS).** Worktree at `8815c45`; frames 001/050/100 from
`draft_000451/calibrated/` reprocessed with C.1 active. Mean absolute difference **0.0000 ADU**
(all three); `VYSKYORD=2` on post-fix outputs.

**1.2 Full raw-to-photometry re-run (PASS).**
`python src_py/simulate_night_run.py --source Archive/Drafts/draft_000451/Raw --eq 1 --tel 1`
-> **draft_000452** (451 untouched). Log: `tmp/c4_night_run_452.log` / terminal `398612.txt`.

Preprocessing re-ran from `calibrated/` (Step 11 QC in-place, 150 frames). Confirmed stamps on
452 frames vs absent on 451:

| Location | draft 451 | draft 452 |
|----------|-----------|-----------|
| `calibrated/lights/.../BO_CVn_Light_001.fits` | VYSKYORD=None | **VYSKYORD=2, VY_SKYSF=True** |
| `detrended_aligned/BO_CVn_Light_001.fits` | VYSKYORD=None | **VYSKYORD=2, VY_SKYSF=True** |

MASTERSTAR build logged from cold start (not resumed):

```
[DAO pass 1] 2552 detections, 1332 Gaia unmatched
[DAO pass 2] 1225 additional detections from 1332 targeted positions
Per-frame catalog: masterstars_full_match.csv, 2951 rows
```

**1.3 Acceptance table**

| Quantity | draft 451 (broken) | required | anchor | draft 452 (post-fix) |
|----------|-------------------:|---------:|-------:|---------------------:|
| pass-1 DAO | 8926 | ~2550 | 2552 | **2552** |
| masterstars rows | 6698 | ~2950 | 2951 | **2951** |
| DAO_ONLY fraction | 40.4% | <10%, ~4% | 3.7% | **3.69%** (109/2951) |
| bg_std | 62.2 ADU | ~84 ADU | 83.8 | **83.82 ADU** |
| DAO threshold | 130.7 ADU | ~175 ADU | 175.4 | **176.03 ADU** |
| active targets | 242 | **201** (live VT) | 165 (frozen VT) | **201** |
| Group A median RMS ratio vs anchor | 1.007 | 0.95-1.05 | 1.0 | **1.000** |
| sigma_pp | 46.07 ADU | ~46 | 46.13 | **46.90 ADU** |

**Reading on active=201:** The old **~160-175** band was wrong (anchor 165 comes from a frozen VT
with `vsx_variable_targets_mag_limit=14.5`, 245 rows; live VT has 873 rows - see
**PHASE0-ACTIVE-COUNT-VT-CONTEXT**). Correct decomposition: **201** = Group A **163** (243 VT rows
<= 14.5 mag) + Group B **38** (630 VT rows > 14.5 mag). Group B actives carry
`gaia_match_source` **masterstars** (37) or **masterstars_exo** (1) - not `gaia_dr3_direct`
(Phase 0 rejects those as `not_target_eligible`). Group A photometry matches anchor (ratio 1.000).

**Infolog lines (night-run stdout; Part B.1 proof):**

```
VSX-GAIA XM: n_vsx=873 n_gaia=15085 ... masterstars=205/208 outcome=ok gaia_db_max_g=17.5
FAZA 0 funnel: vsx_bbox=875 -> in_frame=797 -> gaia_id_assigned=651 -> dao_detected=201 -> active=201 | excluded: ... not_target_eligible=596 out_of_frame=78 ...
```

**Exoplanet (Part A proof on 452 VT):** 3 hosts promoted with schema preserved:
TOI-3919, TIC 23815847, Gaia DR3 1500746446072739200 (TIC 165412561). Draft 451 VT had **no**
exo columns.

**INV-PREP-01 / INV-MS-01 (pre-push fix):** On draft_452 the guards likely executed (outcomes
match) but `log_event`-only wiring left no lines in headless stdout or on-disk infolog - same
defect class as Part B. **Fix:** dual `LOGGER.info` + `log_event` on both guards; headless
`run_night_pipeline` saves infolog on success. Tests:
`test_inv_ms01_milestone_reaches_headless_logger`,
`test_inv_prep01_milestone_reaches_headless_logger`. UI run (draft 453+) is the direct proof.

**Group A / Group B (vs anchor 165 actives)**

| | draft 451 | draft 452 |
|---|----------:|----------:|
| Group A (shared) | 160 | 163 |
| Group B (451/452-only) | 82 | **38** |
| Group A median lc_rms ratio vs anchor | 1.007 | **1.000** |
| Group B median lc_rms | 0.294 mag | 0.398 mag |
| Group B RED trust | 41 | 20 |

Group B shrank 82 -> 38 as inflated-DAO spurious actives dropped after sky-surface restore.
Remaining 38 are faint VT tail detections (G > 14.5 mag) with legitimate `masterstars` /
`masterstars_exo` identity joins - not shared-anchor targets. Median lc_rms **0.398 mag**, **20**
RED (see **DETECTION-DEPTH-VS-LC-USABILITY**).

**1.4 C.2 decision:** **C.2 NOT needed.** `bg_std` returned to **83.82 ADU** (matches draft_435
exactly). **`sigma_pp` 46.90 vs anchor 46.13:** `MASTERSTAR.fits` pixel arrays are **byte-identical**
between draft_452 and draft_435 (`max_abs_diff=0.0`). The gap is a **measurement inconsistency**:
unmasked MAD on the full frame yields **~46.90 ADU** on both drafts; star-masked MAD (40 px margin
from masterstars CSV) yields **~45.03 ADU** on both. Anchor reference 46.13 used a masked method
with slightly different margin/implementation - not different image content. DAO counts still match
exactly because detection uses catalog x/y/peak, not the sigma estimator mask.

---

### 2. Part A class fix - path resolution audit

**Trigger:** Latent `Path.resolve()` against CWD, not `data_root`. First production symptom
draft_450 (2026-07-24). **Activation arc:** `c99fcec` (2026-07-24) separated `data_root` from
install root while night-run CWD stayed under `Archive/Drafts/`; `3c31bfa` (2026-07-23)
materialized relative catalog paths in `config.json` but did not fix resolution semantics. Not a
single-line break - CWD != data_root exposed the class.

**Uniform rule (`f873085`):** `resolve_config_path(raw, data_root)` for all config paths;
catalog DB opens via `_resolve_catalog_db_path` + `require_*_db_path` fail-loud helpers.
Guard: `dev/tests/test_catalog_db_path_resolution.py`.

| Key | Resolution today | data_root-relative | Missing/unreadable |
|-----|------------------|--------------------|--------------------|
| `gaia_db_path` | `resolve_config_path` in `__post_init__`; `_resolve_catalog_db_path` at query | Yes | **GaiaCatalogError** (require) / FileNotFoundError (query) |
| `vsx_local_db_path` | same | Yes | **VSXCatalogError** (require); query returns `[]` only after path valid |
| `exoplanet_local_db_path` | same | Yes | **ExoplanetCatalogError** (require); query returns `[]` only after path valid |
| `archive_root` | `_path_or_default` -> `resolve_config_path` | Yes | fails at use site if path wrong |
| `calibration_library_root` | same | Yes | fails at use site if path wrong |
| `database_path` | same | Yes | sqlite connect error (vyvar app DB, not catalog) |
| `blind_index_*_path` | `resolve_config_path` after legacy migration | Yes | blind solve fails when index missing |
| `project_root` | install tree | N/A | N/A |

Recorded in `docs/VYVAR_DECISIONS.md` (**CONFIG-PATH-DATA-ROOT**).

---

### 3. Part B.3 loose ends

**3.1 `not_target_eligible` confirmed (VT `gaia_match_source` mismatch):**

```
draft_435 variable_targets.csv:1497436125799224960 ... gaia_match_source=masterstars
draft_451 variable_targets.csv:1497436125799224960 ... gaia_match_source=gaia_dr3_direct -> excluded not_target_eligible

draft_435 variable_targets.csv:HAT-188-0000323 ... gaia_match_source=masterstars
draft_451 variable_targets.csv:HAT-188-0000323 ... gaia_match_source=gaia_dr3_direct -> excluded not_target_eligible
```

Phase 0 requires `gaia_match_source in ("masterstars",)` for target eligibility.

**3.2 Border margin ROADMAP item opened:** **PHASE0-BORDER-MARGIN-GEOMETRY** (MED) in
`docs/VYVAR_ROADMAP.md`. Derive margin from annulus outer radius / FWHM, not fixed 50 px.
Not implemented here. `out_of_frame=78` on 451/452 is consistent with ~12% frame strip.

---

### 4. Remaining docs

| Item | Status |
|------|--------|
| `VYVAR_FLOW_CZ.pdf` | **Regenerated** (`f873085`) |
| `VYVAR_STATE.md` NOT-guaranteed removed | **Done** - C.4 table passed |
| `VYVAR_DECISIONS.md` path rule | **Done** (CONFIG-PATH-DATA-ROOT) |

---

### 5. Gates (closeout)

| Gate | Result |
|------|--------|
| `--fast` | **PASS** (1189 passed, 26 skipped) after `f873085` |
| `ruff` | clean on path-resolution files |
| anchor `--full` | **PASS** on `f873085` (core `1c48d9fc...` n=325, extended `744bce94...` n=487, plan-regen 875, active 165) |

---

### Commits for Milan authorization (local, not pushed)

| Commit | Description |
|--------|-------------|
| `926a94c` | Part A exoplanet promotion + fail-loud |
| `63b902d` | Part B observability + skip_reason |
| `ff08002` | C.1 sky-surface restore |
| `1191579` | C.3 INV-PREP-01 / INV-MS-01 guards |
| `5078669` | C.5 docs + anchor funnel expectation |
| `f873085` | Uniform data_root path resolution + catalog tests + DECISIONS/ROADMAP/PDF |
| `df42d46` | C.4 closeout result + STATE NOT-guaranteed removal |

**Push:** Milan authorized 2026-07-27; see **## PUSH** below.

---

### Stop conditions

None triggered. Preprocess byte-compare 0 ADU; 452 detrended frames regenerated with sky stamps;
Group A RMS ratio 1.000; bg_std restored; anchor `--full` PASS.

---

## PUSH (2026-07-27)

Milan authorized push conditional on items 1-3 below. All satisfied; `--fast` green; anchor `--full`
PASS; `ruff` clean.

### Item 1 - Guard observability (blocking)

**(b) Confirmed:** guards were not wired to headless stdout on draft_452 (`log_event` only).
**Fix:** `LOGGER.info` + `log_event` on `INV-MS-01` and `INV-PREP-01`; `ensure_infolog_logging()`
at headless start; `save_infolog_to_disk()` on success. Unit tests assert both fire on headless
path. Draft 452 has no retroactive infolog file; UI draft 453+ infolog is the live proof.

### Item 2 - sigma_pp 46.90 vs 46.13 (blocking)

`MASTERSTAR.fits` arrays **identical** (452 vs 435 anchor snapshot). **Measurement inconsistency:**
full-frame unmasked MAD ~46.90 ADU; star-masked MAD ~45.03 ADU (40 px margin). Anchor 46.13 from
masked evaluation with different margin. No fix required.

### Item 3 - DECISIONS entries

| ID | Summary |
|----|---------|
| **PHASE0-ACTIVE-COUNT-VT-CONTEXT** | ~160-175 band wrong; 201 = 163 + 38; Group B = masterstars/masterstars_exo |
| **DETECTION-DEPTH-VS-LC-USABILITY** | Detection depth != LC usability; trust RED flags handle Group B |
| **GUARD-HEADLESS-OBSERVABILITY** | Dual-path logging fix for INV guards |

### Gates (push)

| Gate | Result |
|------|--------|
| `pytest -q` | **PASS** (1191 passed, 26 skipped) |
| `--fast` | **PASS** |
| `ruff` | clean |
| anchor `--full` | **PASS** (core `1c48d9fc...` n=325, extended `744bce94...` n=487, plan-regen 875, active 165) |

### Commits pushed

| Commit | Description |
|--------|-------------|
| `926a94c` | Part A exoplanet promotion + fail-loud |
| `63b902d` | Part B observability + skip_reason |
| `ff08002` | C.1 sky-surface restore |
| `1191579` | C.3 INV-PREP-01 / INV-MS-01 guards |
| `5078669` | C.5 docs + anchor funnel expectation |
| `f873085` | Uniform data_root path resolution + catalog tests + DECISIONS/ROADMAP/PDF |
| `df42d46` | C.4 closeout result + STATE update |
| `1b58fe3` | Guard headless observability + infolog save + tests |
| `5192213` | DECISIONS (3 entries) + UI equivalence script + PUSH result |

**Pushed range:** `535d863..5192213` (9 commits).

### Item 5 - UI equivalence prep (ready, not run)

When Milan's UI draft exists (453+):

```bash
python dev/scripts/draft_ui_equivalence_check.py draft_000452 draft_000453
```

Compares science file SHAs (`lightcurve_*.csv`, `comp_quality_*.json`,
`comparison_stars_per_target.csv`), acceptance table, Phase 0 funnel, `VSX-GAIA XM:` line, both
guard outputs from **UI infolog**, exoplanet promotion count (expect 3). Any divergence is a
finding - do not cut reference or touch ledger until both runs agree.
