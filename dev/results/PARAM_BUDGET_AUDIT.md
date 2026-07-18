# PARAM-BUDGET-AUDIT - do we really need 304 parameters?

Date: 2026-07-18
Baseline: origin/main `191be0e`
Scope: ANALYSIS ONLY. Zero code/config/registry changes. The only commit is this
deliverable pair (`PARAM_BUDGET_AUDIT.md` + `param_budget_audit.csv`). All claims
cited as `path:line`; honest UNKNOWN where code does not answer. ASCII-only.

Milan's question, restated: do we truly need this many parameters? Do some repeat
each other? Can some be merged into one? This is the evidence pass that feeds the
deletion-oriented wave B - it proposes; Milan disposes.

Companion machine-readable file: `dev/results/param_budget_audit.csv` (304 rows;
Step 1 evidence columns + Step 3 disposition columns).

Inputs reused (not redone): `dev/results/PARAM_SOURCE_AUDIT.md` +
`param_source_audit.csv` (source/ownership axis, 12 DB duplicates, vestigial
SETTINGS table), `dev/validation/params_registry.json` (the 304-key set),
`docs/VYVAR_CONFIG_GUIDE_EN.md`.

---

## Bottom line first

The registry carries **304 keys**. The audit finds that a conservative,
low-argument reduction takes the live surface to **257** keys, and Milan's own
appetite on two larger levers (an 80-key "expert never-touched" hardcode pool and
the observatory-facts move to the DB) can take it below **180** without removing a
single genuinely-tunable science knob.

Nothing here is deleted blind: every DELETE cites the shadowing reader or the "0
readers" fact, and every key that Milan has ever moved in config.json history or
that a test pins is flagged HIGH and left for his individual call.

| Disposition | Count | Meaning |
|-------------|-------|---------|
| KEEP | 200 | Effective, tunable, earns its place (120 core + 80 hardcode candidates - see below) |
| KEEP-GATED | 54 | Feature intentionally OFF awaiting validation; keep with its gate |
| HARDCODE | 20 | Never moved + expert + no plausible per-user tuning -> module constant |
| MERGE | 14 -> 3 | Collapses into 3 structured keys (removes 11 keys) |
| DELETE-DB-DUP | 9 | DB/FITS is authoritative; config copy is a dead fallback |
| DELETE-DEAD | 7 | No effective reader (cited) |

Projected end-state (conservative, HIGH-risk items still pending Milan):
`304 - 7 (dead) - 9 (db-dup) - 20 (hardcode) - 11 (merge collapse) = 257`.

Of the 200 KEEP, **80 are "hardcode candidates"**: expert-tier, never moved in
six weeks of config history, ungated, and read in production - kept as KEEP only
because whether they are worth a knob is a tuning-appetite judgment that is
Milan's, not the code's. If he accepts that pool, the surface drops to ~177.

---

## Method (how each Step-1 column was produced)

- `n_readers`: count of real read sites across `src_py/*.py` **excluding
  `config.py`** (the load/clamp/to_dict plumbing), counting attribute reads
  (`.key`) and quoted uses (getattr / job-dict / `st.get`). Tooling:
  `tmp/_param_budget_build.py` (gitignored scratch). 0 real readers = strong DEAD
  signal; each zero was then traced by hand (`tmp/_param_trace.py`) to separate
  "config-internal only", "dev-script only", and "truly unreferenced".
- `effective?`: does changing the value change behavior on the PRODUCTION path
  today? `no` where the value is overwritten/shadowed before use (cite the
  shadowing site); `gated` where it is only read behind a permanently-OFF gate;
  `yes` otherwise.
- `in_config_json`: present in Milan's current `config.json`.
- `ever_changed`: did the key EVER hold a non-default value across the **15
  commits** that touched `config.json` (full per-version JSON diff, oldest to
  working tree). Only **13 keys** have ever moved.
- `clamp_note`: does an `AppConfig.__post_init__` `min()/max()/clip` narrow the
  documented range. 102 keys are clamped.
- `gated_feature`: key belongs to a feature intentionally OFF; the controlling
  gate key is named.
- `test_touched`: referenced by any file under `dev/tests/` (83 keys).
- `family`: grouping label for Step 2.

Caveat: `n_readers` is a lower-bound signal - a key read only via dynamic
`getattr(cfg, name)` in a loop would under-count. Every 0-reader key in this
report was hand-verified, so no DELETE rests on the raw counter alone.

---

## STEP 1 - evidence matrix (all 304 keys)

Full matrix is `dev/results/param_budget_audit.csv`. Aggregate signals:

- 304 registry keys; **292** present in `config.json`; **12** registry keys are
  code-default-only (not in config.json - see 4.3 metadata contradictions).
- **13** keys with a real reader count of 0 (hand-verified below).
- **13** keys ever changed in config.json history:
  `apply_color_term, blind_index_fine_path, blind_index_wide_path,
  comp_trust_min_comps, gaia_db_path, masterdark_validity_days,
  masterflat_validity_days, observer_alt_m, observer_lat, observer_location_id,
  observer_location_name, observer_lon, temporal_binning_enabled`.
  Every one of these is protected: none is proposed for DELETE/HARDCODE except the
  observer_* block, which is HIGH-flagged for Milan.
- **83** keys are test_touched; **54** keys belong to gated-OFF features.

### 1.1 The 13 zero-reader keys, hand-traced

| key | real readers | verdict |
|-----|--------------|---------|
| `aperture_fwhm_factor_medium` | 0 (config.py + registry gen only) | DEAD - SNR sizing reads only `_small`/`_large` (`pipeline.py:187-188`) |
| `masterstar_log_astroalign` | 0 | DEAD - logging toggle never wired (`config.py:1694` load + to_dict only) |
| `phase01_comparison_proximity_tiebreak` | 0 | DEAD - experimental tie-break never wired into comp selection |
| `phase01_comparison_rms_bin_mag` | 0 | DEAD - experimental rms-bin knob never consumed |
| `calibration_master_ccd_temp_tolerance_c` | 0 prod (config clamp + one test) | DEAD in prod - calibration match never reads it; `test_calibration_library_match.py:328` only asserts the default |
| `export_arcsec_per_px` | 0 in `src_py/` (dev export scripts only) | REDUNDANT - derivable from WCS/optics; duplicates plate scale |
| `k2_ceiling` | 0 | GATED (k2 fit v2 plausibility bound) |
| `k2_fit_enabled` | 0 | GATED (k2 fit v2) |
| `k2_fit_consistency_sigma` | 0 | GATED (k2 fit v2) |
| `k2_fit_lit_factor` | 0 | GATED (k2 fit v2) |
| `k2_fit_min_detectability` | 0 | GATED (k2 fit v2) |
| `psf_spatial_grid` | 0 in `src_py/` (smoke harness only) | GATED (psf spatial) + wiring note (see 4.3) |
| `psf_spatial_min_stars_per_cell` | 0 in `src_py/` (smoke harness only) | GATED (psf spatial) + wiring note |

Bonus finding outside the 304: `sips_dao_fwhm` appears in code/job-dicts with a
default but has **0 readers and is not a registered key** (only
`sips_dao_threshold_sigma` is registered). It is a dead code-level constant, not a
config parameter - noted, not counted.

---

## STEP 2 - duplication and merge analysis

### 2.1 config <-> config: families that could collapse

**MERGE (recommended) - parallel scalar sets that are really one table:**

1. `comp_tier{1..4}_bprp_limit` + `comp_tier{1..4}_weight` -> **one 4-row
   `comp_tiers` table** (columns bprp_limit, weight). They are always read
   together as an indexed set:
   `comp_selection_per_target.py:258-260` and `:1962-1964` and `:2268-2271`,
   `photometry_core.py:8597-8600` and `:12578-12580`, `export_reports.py:1068-1070`,
   `check_star_kmag.py:436-439`. 8 keys -> 1 structured key. Effective; ~6 readers
   each, so MED risk (touches several readers, anchor-gate the refactor).
2. `phase01_tier{1..4}_mag` -> **one 4-row `phase01_tiers` table**. Cleanest
   case: read in exactly ONE place (`photometry_core.py:14918-14921`) and not even
   present in config.json. 4 keys -> 1. LOW risk.
3. `aperture_fwhm_factor_{small,large}` -> **one 2-entry `aperture_snr_sizing`
   map** (`pipeline.py:187-188`). `aperture_fwhm_factor_medium` is DEAD (0
   readers) and dropped, not merged. The base `aperture_fwhm_factor` (20 readers)
   is a DIFFERENT knob (the primary aperture) and stays KEEP. 2 keys -> 1. LOW.

**KEEP-FAMILY (do NOT merge) - same word, different physical quantity/stage:**

4. DAO detection sigmas are **stage-specific and legitimately independent**:
   - `qc_dao_detection_sigma` - QC-preprocess detection (`pipeline.py:14450`)
   - `alignment_detection_sigma` - frame-alignment source detection (`pipeline.py:14164`)
   - `masterstar_dao_threshold_sigma` - MASTERSTAR catalog build (11 readers)
   - `sips_dao_threshold_sigma` - SIPS-mode detection, UI-overridable per run
     (`app.py:530,829,834,843`; 19 readers)
   - `masterstar_dao_pass2_sigma` - MASTERSTAR pass-2 recovery (`pipeline.py:7411,8143`)
   Each drives a different detection pass with a different SNR/false-alarm budget.
   Merging them would couple unrelated stages. KEEP all (the base value happens to
   be similar today, but that is a coincidence, not a shared quantity).
5. Saturation fractions apply to **different quantities** and stay independent:
   - `saturate_limit_fraction` - MASTERSTAR saturation ceiling (`pipeline.py:12189`)
   - `cog_sat_frac` - curve-of-growth saturation cut (gated COG; `pipeline.py:200`)
   - `cal_diag_sat_warn_frac` - calibration-diagnostic WARN threshold (`cal_diag.py:102`)
   Different consumers, different meaning of "the fraction". KEEP-FAMILY.

**Near-duplicates / aliases:**
- `export_arcsec_per_px` vs the WCS-resolved plate scale: the config key is a
  fallback/label only, 0 `src_py/` readers -> DELETE (2.2 / Step 3).
- `blind_index_path` (legacy) vs `blind_index_fine_path` / `blind_index_wide_path`:
  `blind_index_path` is not in config.json (registry-only legacy alias); the live
  code uses fine/wide. Registry cleanup, not a config deletion (4.3).

### 2.2 config <-> DB: the 12 known duplicates (restated) + column scan

From `PARAM_SOURCE_AUDIT.md` 2.2, twelve config keys duplicate a DB/FITS store:

| config key | DB/FITS authority | winner citation | disposition |
|------------|-------------------|-----------------|-------------|
| `gain` | FITS EGAIN/GAIN; DB EQUIPMENTS.GAIN_ADU | `param_resolver.py:375-451` | DELETE-DB-DUP (HIGH) |
| `read_noise` | DB EQUIPMENTS.READNOISE_E; FITS | `param_resolver.py:490-514` | DELETE-DB-DUP (HIGH) |
| `plate_scale_arcsec_per_px` | WCS/CD; DB optics | `photometry_core.py:10060-10169` | DELETE-DB-DUP (HIGH) |
| `phase01_plate_scale_arcsec_per_px` | WCS/CD | `photometry_core.py:10063` | DELETE-DB-DUP (MED) |
| `export_arcsec_per_px` | WCS/optics | derivable; 0 src readers | DELETE-DB-DUP (LOW) |
| `observer_lat` | LOCATION.LATITUDE (per-draft) | `param_resolver.py:639-647` | DELETE-DB-DUP (HIGH) |
| `observer_lon` | LOCATION.LONGITUDE | `param_resolver.py:639-647` | DELETE-DB-DUP (HIGH) |
| `observer_alt_m` | LOCATION.ALTITUDE | `param_resolver.py:639-647` | DELETE-DB-DUP (HIGH) |
| `observer_location_name` | LOCATION.PLACENAME | `config.py:1231-1244` | DELETE-DB-DUP (HIGH) |
| `observer_location_id` | LOCATION fk selector | `config.py:1231-1244` | KEEP (the pointer) |
| `masterdark_validity_days` | config authoritative; SETTINGS vestigial | `config.py:724-725` | KEEP (delete DB SETTINGS side) |
| `masterflat_validity_days` | config authoritative; SETTINGS vestigial | `config.py:724-725` | KEEP (delete DB SETTINGS side) |

Two nuances vs the raw "12 duplicates" list:
- `observer_location_id` is the one observer key to KEEP - it is the FK that
  hydrates lat/lon/alt/name; the other four are the redundant copies.
- `masterdark/masterflat_validity_days` are the opposite polarity: **config is
  authoritative**, the DB SETTINGS copy is the dead one. Keep the config key;
  delete the SETTINGS table row/reader/writer on the DB side.

Additional column scan (EQUIPMENTS/TELESCOPE/LOCATION/SCANNING): no NEW config
duplicate beyond the twelve. `EQUIPMENTS.SATURATE_ADU` is an absolute ADU ceiling
and does not duplicate any config key (`saturate_limit_fraction` is a fraction of
that ceiling, a different quantity). `EQUIPMENTS.FOCAL`/`TELESCOPE.FOCAL` and
`PIXELSIZE` have no config-key twin (plate scale is derived, not stored). `SCANNING`
columns (EXPTIME/FILTERS/BINNING/...) are an import FK snapshot with zero runtime
photometry readers - see 2.4.

### 2.3 config <-> hardcoded constants (module-level parameters)

Live module constants that act as parameters next to registered keys of the same
family (from source-audit 1.5), and whether config should absorb them or they
should absorb config:

- `DENSITY_OVERRIDES` (`config.py:2445`) and
  `CROWDING_LOOSEN/TIGHTEN_OVERRIDES` (`config.py:2472,2478`) - delta tables that
  post-adjust comp/annulus/crowding config values per field class. These are
  effectively a second, hidden tier of the same knobs. Recommend they stay code
  constants (expert-owned deltas) but be **documented in the guide** as the reason
  a run can differ from the config.json value - they are not user parameters.
- `GAIN_SETTING_INDEX_MAP` (`param_resolver.py:85`) - QHY294 gain-index -> e-/ADU.
  Detector fact; belongs with the DB EQUIPMENTS move, not config.
- `K2` literature tables (`k2_extinction.py:36-99`) - correctly hardcoded
  (literature); config only overrides via `k2_defaults_bprp` (KEEP).
- `dao_reconcile` match/blend constants (`dao_reconcile.py:23-30`),
  `crowding_index` GAIN/RN fallbacks 3.17/7.6 (`crowding_index.py:47-48`),
  `_PSF_QUALITY_THRESH` (`psf_photometry.py:2480`), sparse/trust bands
  (`sparse_trust_core.py:18-23`, `trust_flag_core.py:39-42`) - diagnostic
  internals; keep as constants. No config key should absorb them.

Net: the hardcoded constants are correctly on the "not a user knob" side; the
action item is documentation (explain DENSITY/CROWDING overrides in the guide), not
migration.

### 2.4 DB side: zero-reader keys/columns

- `SETTINGS` table (`database.py:2665-2668`): **vestigial**. Seeded with
  `masterdark_validity_days`/`masterflat_validity_days` on every open; `get_setting_int`
  / `set_setting` have zero production callers (only a direct test read). Wave B
  should drop the table + accessors; config keeps the two validity-days keys.
- `SCANNING` columns and `resolve_binning`/`resolve_exptime`
  (`param_resolver.py:624-633`): defined but **zero callers** - FITS headers
  dominate binning/exptime at runtime. DB-side dead surface (report only; DB
  schema change is out of scope for the config wave).

---

## STEP 3 - disposition proposal

One disposition per key is in `param_budget_audit.csv` (columns `disposition`,
`merge_group`, `risk`, `hardcode_candidate`, `justification`). Summary:

### 3.1 DELETE-DEAD (7) - no effective reader, cited

`aperture_fwhm_factor_medium` (LOW), `masterstar_log_astroalign` (LOW),
`phase01_comparison_proximity_tiebreak` (LOW), `phase01_comparison_rms_bin_mag`
(LOW), `calibration_master_ccd_temp_tolerance_c` (MED - one test asserts default),
`frame_width_px` (MED - NAXIS-shadowed, `photometry_core.py:11681-11692`),
`frame_height_px` (MED - NAXIS-shadowed).

### 3.2 DELETE-DB-DUP (9) - DB/FITS authoritative

`gain`, `read_noise`, `plate_scale_arcsec_per_px` (HIGH),
`phase01_plate_scale_arcsec_per_px` (MED), `export_arcsec_per_px` (LOW),
`observer_lat`, `observer_lon`, `observer_alt_m`, `observer_location_name` (HIGH).
This is the Option-B "thin config.json, DB owns facts" move from the source audit;
it is a science-path fallback removal and MUST be anchor-gated.

### 3.3 HARDCODE (20) - internal mechanics, never moved, no user tuning

Blind-solver DBSCAN internals (`blind_cluster_coherence_cap`, `_eps_deg`,
`_min_samples`, `_min_votes`, `_vote_span`, `blind_scale_tol_frac`,
`blind_prefilter_min`), MASTERSTAR/plate-solve solver internals
(`masterstar_odds_k`, `_odds_match_floor`, `_odds_min_quadrants`,
`_false_alarm_p_max`, `_sip_force_rms_guard_ratio`,
`_platesolve_prewrite_rms_max_px`, `_platesolve_prewrite_relaxed_rms_max_px`,
`_platesolve_nn_refine_max_rms_px`, `_solver_use_draft_median_if_hint_sep_deg`,
`_optimizer_mirror_extra_log`, `platesolve_anisotropy_threshold`), and two fit/QC
internals (`sky_adu_fallback`, `moffat_chi2_limit`). All expert-tier, single
reader, never changed in history -> become module constants with a comment. LOW
risk each (none is test_touched or ever_changed).

### 3.4 MERGE (14 keys -> 3 structured keys)

`comp_tiers` (8 -> 1, MED), `phase01_tiers` (4 -> 1, LOW),
`aperture_snr_sizing` (2 -> 1, LOW). See 2.1.

### 3.5 KEEP-GATED (54) - features OFF awaiting validation

Grouped by gate: PSF core `psf_photometry_enabled` (14), PSF neighbor-sub
`psf_neighbor_sub_enabled` (10), COG `cog_aperture_correction_enabled` (8), crowding
classifier `crowding_classifier_enabled` (4), k2 fit v2 `k2_fit_enabled` (5 incl.
`k2_ceiling`), savgol/democratic detrend `savgol_detrend_enabled` (4), frame-quality
gate (3), frame-align residual gate (2), temporal binning (2), sysrem (2). These are
NOT dead - each is a validated-or-pending feature held behind its gate. Keep with
the gate.

### 3.6 KEEP (200)

120 are the genuinely-tunable core (photometry apertures, comp selection,
variability thresholds, trust bands, QC limits, HRD, blind verify, paths). 80 are
**hardcode candidates**: expert-tier + never moved + ungated + has a reader. They
are kept KEEP only because "is this worth a knob?" is Milan's tuning-appetite call,
not something the code can answer. Distribution of the 80: detection 35,
comp_selection 11, reports 8, photometry 6, calibration 6, qc 5, trust 4, paths 3,
observer 1, system 1. All are listed with `hardcode_candidate=yes` in the CSV.

---

## STEP 4 - summary for Milan

### 4.1 Counts and projected end-state

| Disposition | Keys | Removes from live surface |
|-------------|------|---------------------------|
| KEEP | 200 | 0 |
| KEEP-GATED | 54 | 0 |
| HARDCODE | 20 | 20 |
| MERGE (14 -> 3) | 14 | 11 |
| DELETE-DB-DUP | 9 | 9 |
| DELETE-DEAD | 7 | 7 |

- Conservative end-state (HIGH-risk items still pending your call): **257 keys**.
- If you also accept the 80-key hardcode-candidate pool: **~177 keys**.
- The three levers, ranked by payoff/safety:
  1. HARDCODE the 20 solver internals - pure safety, -20, do first.
  2. MERGE the 3 tier tables - -11, LOW/MED, improves clarity most.
  3. DELETE-DEAD the 7 - -7, mostly LOW.
  The DB-dup move (-9) and the 80-key hardcode pool are the big levers but need
  your judgment (science-path change / tuning appetite).

### 4.2 Top merge groups (before -> after)

| group | before | after | risk |
|-------|--------|-------|------|
| `comp_tiers` (bprp_limit + weight, 4 rows) | 8 | 1 | MED |
| `phase01_tiers` (mag, 4 rows) | 4 | 1 | LOW |
| `aperture_snr_sizing` (small+large; drop dead medium) | 2 (+1 dead) | 1 | LOW |

(Only three genuine collapse groups exist; the other same-word families - DAO
sigmas, saturation fractions - are legitimately independent and must NOT be merged,
see 2.1.)

### 4.3 Where the audit contradicts the registry/guide (metadata to fix)

1. **12 registry keys are not in config.json** (code-default-only): `blind_index_path`
   (legacy alias, superseded by fine/wide), `masterstar_dao_pass2_sigma`,
   `masterstar_use_best_frame_fwhm`, `phase01_comparison_fov_fraction`,
   `phase01_tier1_mag..4_mag`, `plate_solve_fov_deg` (computed), `project_root`
   (internal), `qc_preprocess_workers` (env/host), `saturate_limit_fraction`.
   Registry lists them as parameters but they never appear in the config file the
   guide documents - either add them to config.json or mark them internal/computed
   in the registry.
2. **`blind_index_path`** is a registry-only legacy alias; live code uses
   `blind_index_fine_path`/`blind_index_wide_path`. Remove the alias from the
   registry.
3. **PSF spatial keys wiring gap**: `psf_spatial_grid` and
   `psf_spatial_min_stars_per_cell` have 0 `src_py/` readers even though
   `psf_spatial_enabled` is read (`psf_photometry.py:1384`); today only the smoke
   harness passes them. If PSF-spatial is ever ungated, confirm the grid/min-stars
   values are actually threaded into `psf_photometry.py`, or they will be silently
   ineffective. Registry should note "gated + wiring pending".
4. **`phase01_comparison_proximity_tiebreak` / `_rms_bin_mag`**: STATE/registry
   present these as experimental toggles, but they have 0 readers - they are dead,
   not merely OFF. Fix the metadata or wire them.
5. **`masterdark/masterflat_validity_days`**: guide/source model imply the DB
   SETTINGS table stores validity days; in reality config.json is authoritative and
   SETTINGS is vestigial. Document config as the owner.

### 4.4 HIGH-risk proposals needing Milan's individual call (7)

All are DELETE-DB-DUP - the config copy is a dead fallback, but removing it is a
science-path change and each is either ever_changed or test_touched:

| key | why HIGH | recommended handling |
|-----|----------|----------------------|
| `gain` | 47 fallback readers + test_touched; header/DB authoritative | anchor-gate; move to DB-owned |
| `read_noise` | test_touched; DB/FITS authoritative | anchor-gate; move to DB-owned |
| `plate_scale_arcsec_per_px` | test_touched; WCS/DB authoritative | anchor-gate; keep as labelled fallback or drop |
| `observer_lat` | ever_changed + test; science uses draft LOCATION | move to DB; keep `observer_location_id` selector |
| `observer_lon` | ever_changed + test | same |
| `observer_alt_m` | ever_changed + test | same |
| `observer_location_name` | ever_changed; mirror of PLACENAME | same |

`phase01_plate_scale_arcsec_per_px` (MED) and `export_arcsec_per_px` (LOW) are the
lower-risk members of the same DB-dup family.

---

## Appendix - artifacts and reproducibility

- `dev/results/param_budget_audit.csv` - 304 rows: `key, n_readers, effective,
  in_config_json, ever_changed, clamp_note, gated_feature, test_touched, family,
  tier, phase, disposition, merge_group, risk, hardcode_candidate, justification`.
- Evidence built by gitignored scratch scripts under `tmp/`
  (`_param_budget_build.py`, `_param_trace.py`, `_param_families.py`,
  `_param_disposition.py`); no repo code, config, or registry was modified.
- Read-site counting excludes `config.py` plumbing and is a lower bound; every
  0-reader key was hand-verified before any DELETE was proposed.
- `ever_changed` computed from the 15 commits that touched `config.json`
  (per-version JSON value diff, oldest to working tree).
