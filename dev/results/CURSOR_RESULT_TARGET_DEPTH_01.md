CURSOR RESULT - 2026-08-14 TARGET-DEPTH-01

Register ID: TARGET-DEPTH-01
Closes: SNR-DEPTH-01
Status: implemented locally; commit below; push awaits Milan authorization.

## Verdict

Item A: QC snapshot write failed because `OBS_QC_PROCESSING_FILE.RUN_ID` still
FK-referenced `OBS_QC_PROCESSING_RUN_OLD` after a half-finished OBS_DRAFT FK strip.
Heal on open rebuilds FILE against live `OBS_QC_PROCESSING_RUN`; demo write succeeded.

Item B: target admission depth is derived per draft from COMP-POOL machinery.
Forced-photometry `detect_frac` is ceiling-degenerate after SNR-GATE-01, so the
operative path is NP half-SNR at the last fully-complete bin (T-R0). Targets
fainter than the limit stay in `active_targets.csv` with
`skip_photometry=True`, `skip_reason=below_target_depth` (flag, do not omit).

Draft 513 is a trial run under X-R3 (no stamped SHA; status stuck at INGESTED).

---

## Item A -- QC snapshot write

### Cause

`OBS_QC_PROCESSING_FILE.RUN_ID` foreign key still pointed at
`OBS_QC_PROCESSING_RUN_OLD` while the live parent table was already named
`OBS_QC_PROCESSING_RUN`. Inserts into FILE then raised:

`no such table: main.OBS_QC_PROCESSING_RUN_OLD`

Created by a half-finished OBS_DRAFT FK strip / `_rebuild_table_safely` path that
could leave FILE pointing at `RUN_OLD` after `RUN_OLD` was dropped.

### Fix

`src_py/database.py`: detect FILE FK parent; if not `OBS_QC_PROCESSING_RUN`,
rebuild FILE (and RUN if needed) through the existing safe rebuild. Opening
`VyvarDatabase` heals.

### Demonstration

Harness `tmp/target_depth_01_run.py` on live `vyvar.sqlite3`:

- after open: FILE FK parent = `OBS_QC_PROCESSING_RUN`
- `record_qc_processing_apply(513, "target-depth-01-qc-demo", overwrite=True)`
  -> `run_id=392`, `n_files_rows=150`, ok

Tests: `test_qc_file_fk_to_run_old_heals_on_open`,
`test_qc_run_obs_draft_fk_migration_uses_safe_rebuild`.

### Other retired names / incomplete QC

- Still present: `LOCATION_OLD` (ACTIVE-column migration leftover). Not on the QC
  write path. Deferred one-line: heal `LOCATION_OLD` orphans on open.
- Draft statuses at investigation: 512/513/510 = `INGESTED`, 435 = `PROCESSED`.
  Draft 513 status remained `INGESTED` because the QC write failed -- recorded
  status does not describe the photometry that was produced.

---

## Item B -- target depth

### 2.1 Current target-admission path (end to end)

| Step | Criterion | Parameter / value | Where |
|------|-----------|-------------------|-------|
| VSX / exo plan | In-cone VSX (+ exo merge) written to `variable_targets.csv` | planner cone / VSX export | `pipeline` plan |
| Gaia identity | VSX auto: `gaia_match_source=masterstars` and non-empty `catalog_id` | -- | `select_active_targets` |
| Manual / exo | Non-empty `catalog_id` (not VSX-gated) | -- | same |
| MASTERSTAR match | Identity join on Gaia source_id | -- | same |
| Chip interior | x,y inside frame with margin | `phase01_chip_interior_margin_px=50` (default) | same / Phase 0+1 |
| Bright sat heuristic | Exclude if `snr50_ok=False` and mag < 8 | hard-coded mag 8 | same |
| Zone mask | `skip_photometry` if zone in saturated / catalog_only / neznama_zona | `dao_detection_n_equiv=3.78`, sat fraction 0.85 | zone from `_annotate_masterstars_flux_zones` |
| VSX type scope | Mask if type in `vsx_out_of_scope_types` | config list (often empty) | same |
| **Depth (new)** | Mask if mag > draft-derived `target_depth_g` | derived; no config mag/SNR | `derive_target_depth_limit` + `select_active_targets` |

`zone_flag` / `zone` (T-R1): **not photometric linearity**. Computed as
`peak_dao / bg_sigma` versus `dao_detection_n_equiv` (DAO detection significance),
plus a saturation ceiling. `linear` = above the DAO N-sigma cut; `noise` = below
it; `saturated` = peak above sat limit. `noise` does **not** set
`skip_photometry` (only saturated / catalog_only / neznama_zona / vsx scope /
below_target_depth). Bound on draft 512 MASTERSTAR (~360 linear / ~344 noise of
~704) is that significance cut, not a mag floor.

Why G~15-16 can be `linear`: after SNR-GATE-01 those stars are on MASTERSTAR; if
stack peak significance exceeds `dao_detection_n_equiv` they are labelled
`linear` even when single-frame photometry is unusable.

Draft 513 products (existing, pre-mask application counts):

- 218 active_targets; Gaia G ~5.94-15.25; zone 173 linear / 45 noise; 60 lightcurves
- By G bin (active_targets): see `TARGET_DEPTH_01_results.json`

### 2.2 Derived limit

Machinery: COMP-POOL `summarize_stars` (`detect_frac`, `scatter_mad`) and the NP
curve. No config magnitude or SNR constant.

**Finding:** after SNR-GATE-01, forced-photometry `detect_frac` stays near 1.0
through G15 and does **not** reproduce SNR-GATE F2 DAO-repeatability
(G14 frac_median 0.955, G15 0.507). Using forced-photometry detect_frac alone
with half-completeness yields depth ~15.5 and masks nothing on draft 513.

**Rule applied:**

1. Bright sample: mag_g <= 13, finite detect_frac.
2. If p16(detect_frac) is informative (below the bright median): walk 0.5-mag
   bins; last bin with n>=8 and median(detect_frac) >= p16.
3. If ceiling-degenerate (all bright detect_frac ~1) **and** `scatter_mad`
   present (all four drafts here): **NP half-SNR (T-R0)**
   - last 0.5-mag bin with n>=8 and median(detect_frac) >= 1.0 is the forced-
     photometry completeness locus; snr_ref = 1.0857 / scatter_median
   - snr_thr = 0.5 * snr_ref
   - last bin with n>=8 and snr_median >= snr_thr; that upper edge is
     `target_depth_g`
4. If degenerate and no scatter: thr = 0.5 * bright_median detect_frac
   (half-completeness; T-R0 named).

**Same as comp faint limit?** No. Comp Stage-2 faint limit is photon = sys
(~G11 on this rig). A programme target may be fainter than any usable comp; it
must still be measurable. Target depth is that measurability cut.

**Reject vs flag:** **flag** (`skip_photometry` + recorded reason), keep the row.
Basis: LSST forced photometry keeps a row for every reference source and
exposes measurement flags (e.g. `base_PsfFlux_flag`) rather than omitting
undetected forced sources
(https://pipelines.lsst.io/v/v27_0_0/getting-started/multiband-analysis.html;
DP1 forced-photometry tutorial). IRAF `apphot` likewise measures and flags.
Detection-driven catalogues omit rows; VYVAR is a variable-star tool that
already does forced photometry on known VSX positions, so the flag model fits.

### Derived values

| Draft | mode | target_depth_g | n_targets mag>limit | existing LCs mag>limit |
|------:|------|---------------:|--------------------:|-----------------------:|
| 512 | np_half_snr | 13.0 | 2 | 1 |
| 513 | np_half_snr | 15.0 | 15 | 4 |
| 510 | np_half_snr | 13.0 | 2 | 1 |
| 435 | np_half_snr | 15.5 | 0 | 0 |

512/510 depth 13.0 is the shallow pre-SNR-GATE MASTERSTAR edge (few stars past
G13), not a tuned cut.

### T-R3 -- existing LCs that would be masked (not softened)

Draft 513 (4):

- TSVSC1 TN-N130302101-35-67-2 (G=15.101, noise)
- ZTF J134644.81+405436.2 (G=15.092, noise)
- ZTF J141147.12+404808.7 (G=15.162, noise)
- LINEAR 10032668 (G=15.146, noise)

Draft 512/510 (1): CSS_J134925.3+393524 (G=14.195, noise)

### Named non-derived (T-R0)

- `bright_mag_hi=13.0` (bright plateau reference; same family as COMP-POOL / F2)
- `bin_width=0.5`, `min_bin_n=8` (COMP-POOL NP convention)
- `complete_detect_frac=1.0` (strict forced-photometry completeness locus)
- `snr_thr = 0.5 * snr_ref` when detect_frac is ceiling-degenerate
- half-completeness `0.5 * bright_median` when degenerate and scatter absent

### Application

`run_phase0_and_phase1` derives the limit from `per_frame_csv_dir`, passes
`target_depth_g` into `select_active_targets`, writes `target_depth.json`.

COMP-POOL Stage 2 unchanged (C2-R2). SNR-GATE detection depth unchanged.

---

## Draft 513 provenance

Trial under X-R3: no committed-tree SHA in draft 513 products; manifest status
`INGESTED` (QC write failed); nine local commits were pending push when it was
run. Not a reference draft.

---

## Section 6 -- nothing may break

Impact inventory: see `TARGET_DEPTH_01_results.json` `section6.impact_inventory`.
Intended changers: QC FILE FK heal; `below_target_depth` mask. Untouched:
dao_flux, apertures, exported errs, COMP-POOL Stage 2, SNR-GATE detection depth.

| Check | Measured |
|-------|----------|
| Aperture radii vs archived SNR table | n=23; max \|delta\| = 0.10 px (pre-existing builder/product near-tie from WIDE-ERR-LOC-01; not introduced here); archive sha16 `55cd7bf4b2f51d49`; noise-floor helper still legacy |
| dao_flux six BO stars on `proc_BO_CVn_Light_068.csv` | max rel diff = 0.0 (n=6) |
| Exported error bars | re-read identical; cols err / err_photon / err_sem_rel / err_scint_rel; n=134 |
| BO CVn 512 | present; mag 9.72; zone linear; trust GREEN; check_scatter 0.009300; n_clean 5; would not mask |
| BO CVn 513 | present; mag 9.72; zone linear; trust RED; check_scatter 0.011147; n_clean 4; would not mask |
| Iron-gate + kwarg fixtures | returncode 0; still fire |
| `--fast` | see commit footer |

Nothing moved that the inventory did not predict.

---

## Pre-registered rules

| Rule | Fired? |
|------|--------|
| T-R0 | Yes -- half-SNR (and half-completeness fallback) named; not presented as fully derived |
| T-R1 | Yes -- `zone_flag` measures DAO peak significance, not photometric linearity |
| T-R2 | Yes -- no threshold tuned to a count or a star |
| T-R3 | Yes -- 4 LCs on 513 and 1 on 512/510 reported; criterion not softened |

---

## Deferred (one line each)

- Heal `LOCATION_OLD` orphans on DB open (not QC path).
- Forced-photometry `detect_frac` != SNR-GATE F2 per-frame DAO detection fraction; NP half-SNR is the operative target-depth path when detect_frac is ceiling-degenerate.

---

## Register diff

- **SNR-DEPTH-01**: CLOSED (superseded/closed by TARGET-DEPTH-01)
- **TARGET-DEPTH-01**: FIXED (Item A + Item B)

Machine-readable: `dev/results/TARGET_DEPTH_01_results.json`

## Files changed

- `src_py/database.py` -- QC FILE FK heal
- `src_py/comp_pool_noise.py` -- `derive_target_depth_limit`
- `src_py/photometry_core.py` -- wire depth into Phase 0+1; `target_depth.json`
- `dev/tests/test_target_depth_01.py`
- `dev/tests/test_database_sqlite_threading.py`
- `dev/results/TARGET_DEPTH_01_results.json`
- `dev/results/CURSOR_RESULT_TARGET_DEPTH_01.md`
- `docs/VYVAR_AUDIT_2026_REGISTER.md`
