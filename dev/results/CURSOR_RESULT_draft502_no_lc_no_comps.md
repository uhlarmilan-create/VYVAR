CURSOR RESULT - 2026-08-07 (draft502_no_lc_no_comps)

What I did
READ-ONLY diagnosis of draft_000502 (TOI-1131.01.b field, Newton 300/1200 + C3-26000,
V_60_2, pre_calibrated): zero light curves and empty comparison pool. Verified Section 1
fork, recomputed zone arithmetic from draft_502 masterstars CSV, traced zone -> comps -> LC
chain with file:line citations and on-disk values. No pipeline runs, no code or config changes.

## Output / findings

### 1. Section 1 fork: zone column PRESENT (continue to Section 2)

**Branch:** Column `zone` is present in
`Archive/Drafts/draft_000502/platesolve/V_60_2/masterstars_full_match.csv`.

- Infolog searched: `Archive/Drafts/draft_000502/infolog_20260807_102207.txt`
- Grep for `MASTERSTAR source_type annotate failed`: **no matches**
- CSV written successfully (1668 rows); this is **not** a repeat of draft_501's missing-zone /
  annotate-exception path (INV-MS-01 write-block). Task A (`554f7a8`) did its job: the artefact
  exists and carries zone columns.

**Zone distribution on disk (n=1668):**

| zone      | count |
|-----------|------:|
| noisy3    |  1638 |
| noisy2    |    13 |
| saturated |    10 |
| linear    |     4 |
| noisy1    |     3 |

---

### 2. Draft_502 zone-classifier arithmetic (verified against disk)

Quantities from `masterstars_full_match.csv` (single scalar columns, row-invariant):

| Quantity | Value (ADU unless noted) |
|----------|--------------------------|
| `noise_floor_adu` (`nf_val`) | 33493.93 |
| `saturate_limit_adu_85pct` (`sat_lim`) | 55704.75 |
| Matched-row aperture `flux` median (972 Gaia-matched) | 286.53 |
| `flux_med` = median(`flux_s` where finite and `flux_s < nf_val`) | 117.03 |
| `noisy2_thresh` = `flux_med + 2/3 * (nf_val - flux_med)` | 22368.30 |
| `noisy3_thresh` = `flux_med + 1/3 * (nf_val - flux_med)` | 11242.67 |
| Rows with `flux < nf_val` | 99.2% (1655/1668) |
| DAO_ONLY rows with negative `flux` | 20.4% (142/696) |
| Matched rows failing `snr50_ok` | 60.7% (590/972) |

**Sky context (pre_calibrated pedestal):**

- Sample light frame median ADU: 33487.27 (`non_calibrated/lights/V_60_2/..._23-05-09_V.fits`)
- `pipeline_meta.json` `dynamic_params.sky_adu_per_px`: 33482.23
- `pipeline_meta.json` `sky_surface_p2p_median_adu`: 10.94 (residual flatness, not pedestal)

**Recomputation:** Re-implementing `_annotate_masterstars_flux_zones` logic
(`src_py/pipeline.py:6267-6314`) on draft_502 CSV reproduces zone counts **exactly**
(recomputed == on-disk for all 1668 rows). `is_usable = (zone == "linear") & flux.notna()`
-> **4 True, 1664 False**.

**Four `linear` / `is_usable` stars** (all bright, flux well above `nf_val`):

| catalog_id | mag (G) | flux (ADU) | vsx_known_variable |
|------------|--------:|-----------:|:-------------------:|
| 1625566442031336064 | 10.73 | 42424 | False |
| 1625573829375431296 | 11.00 | 36857 | True (CzeV4348, also active target) |
| 1625469611992999936 | 10.76 | 34290 | False |
| 1625561872186122496 | 11.07 | 33739 | False |

**Section 2 hypothesis confirmed:** Noisy sub-class thresholds compare background-subtracted
aperture sum (`flux_s`) against per-pixel `noise_floor_adu` (`nf_val`). On this pre-calibrated
field (~33.5k ADU pedestal), essentially every star has `flux_s << nf_val`, so nearly all rows
fall below `noisy3_thresh` -> `zone=noisy3` -> `is_usable=False`. Saturation correctly uses
`peak_max_adu` (`pipeline.py:6260-6270); noisy branches do not.

---

### 3. Chain from zone to both symptoms

#### 3.1 zone -> zone_flag in active_targets

- Mapping: `_active_target_zone_flag` (`src_py/photometry_core.py:12700-12714`) passes through
  `linear|noisy1|noisy2|noisy3|saturated` unchanged.
- `select_active_targets` skip gate (`photometry_core.py:13242-13243`):
  `skip_photometry = zone_flag in ("saturated", "catalog_only", "neznama_zona")`.
  **noisy3 is NOT skipped here** -- noisy targets proceed to Phase 1/2A.

**active_targets.csv (n=22):**

| zone_flag | count |
|-----------|------:|
| noisy3    | 18 |
| saturated | 2 |
| linear    | 1 |
| noisy2    | 1 |

| skip_photometry | count |
|-----------------|------:|
| True            | 12 |
| False           | 10 |

| skip_reason | count |
|-------------|------:|
| vsx_type_out_of_scope | 10 |
| no_comps | 10 |
| zone_flag | 2 |

Infolog line 2500: `masked: zone_flag=2 vsx_type_out_of_scope=10` (only 2 saturated targets
masked at Phase 0; 18 noisy3 targets were **not** masked).

#### 3.2 zone -> is_usable -> comparison-star pool

- `is_usable` set at `pipeline.py:6313-6314`:
  `is_usable = zone.eq("linear") & flux_s.notna()` -> **4 global candidates**.
- Comparison selection initial mask (`comp_selection_per_target.py:374-381`):

```python
cand_mask = (
    ms["_dist_deg"].le(max_dist_deg)
    & is_usable
    & ~is_saturated
    & ~is_noisy
    & ~vsx_known_variable
    & ~likely_saturated
)
```

DET-only fallback (`comp_selection_per_target.py:514-523`) adds stars with `snr50_ok` but
still requires `~is_saturated`, `~likely_saturated`, `~vsx_known_variable`; it does **not**
bypass `is_usable` / `~is_noisy`. With 1638 noisy3 + 13 noisy2 stars, the Gaia-matched pool
is effectively the 4 linear stars (minus self-exclusion and vetoes).

**snr50_ok secondary effect:** 590/972 matched rows fail `snr50_ok` (60.7%). This would
further shrink any DET fallback path, but the primary bottleneck is already `is_usable` (4 stars
vs 1668 total) before snr50 is consulted for Gaia-matched comps.

#### 3.3 comparison_stars_per_target.csv: written, empty

- Path: `Archive/Drafts/draft_000502/platesolve/V_60_2/photometry/comparison_stars_per_target.csv`
- **Exists**, 285 bytes, **header only, 0 data rows**.
- Written by `comp_df.to_csv(...)` (`photometry_core.py:15523-15536`) even when empty.
- PDF line `Comparison pool: - (no comparison_stars_per_target.csv)` comes from
  `photometry_report.py:1242-1243` when `comp_df.empty` -- wording says "no ...csv" but the
  file is present with zero rows.

Infolog line 2554: `Faza 0+1 hotovo: 22 cielov, 0 parov porovnavaciek`.
All 12 Phase-1 targets logged `Deltamag uvolneny 2.00 -> 3.00 (pole ma malo kandidatov)`.
`pipeline_meta.except_fix_summary.phase2a_empty_comp_drop = 10`.

**Why even CzeV4348 (linear, G~11) got 0 comps:** After global `is_usable` collapse, only
3 non-self usable comps remain at G~10.7-11.1 (distances 179-1346 arcsec, mag/color filters
pass). For typical targets at G~13-17, |delta mag| to these comps is 3-6 mag, exceeding even
the relaxed 3.0 mag ceiling (`phase01_comparison_max_mag_diff_absolute`). CzeV4348 should have
3 viable mag-matched comps; Phase 1 still assigned 0 pairs -- consistent with downstream
proc-frame RMS / min-frames gating (`max_comp_rms=0.1`, `n_comp_min=2`) eliminating the
survivors, but the **dominant** failure mode for the field is the empty global pool from
`is_usable`, not a CzeV4348-specific edge case.

#### 3.4 Empty comps -> zero light curves

- Phase 2A empty-comp handler: `_phase2a_skip_empty_comps_target`
  (`photometry_core.py:1493-1518`) records `ac_skip_reason=no_comps`.
- Infolog line 2584: `[AC] run summary: applied=0 skipped=22 ({'no_comps': 10, 'unknown': 12})`.
- `photometry_summary.csv`: all 22 targets `n_frames=0`; 10 rows `ac_skip_reason=no_comps`,
  12 rows blank `ac_skip_reason` (10 `vsx_type_out_of_scope` + 2 `zone_flag` saturated skips).
- `photometry/lightcurves/`: **0** LC CSV/PNG data files; 22 `trust_*.csv` sidecars only.
- Infolog line 2587: `Faza 2A hotovo: 0 kriviek z 70 snimok`.

---

### 4. Additional checks

| Check | Result |
|-------|--------|
| `provenance.git_hash` | `0ee01b377b8a55da8f0c7ac889a6f16551fb9b9d` (HEAD at diagnosis) |
| Contains Task A `554f7a8`? | **Yes** (`git merge-base --is-ancestor 554f7a8 0ee01b3` exit 0) |
| `calibration_mode` | `pre_calibrated` (`draft_manifest.json`, `pipeline_meta.json`) |
| `per_frame_saturation_enabled` | **False** (`pipeline_meta.provenance.config_snapshot`, default per `config.py:977-979`) |
| `noise_floor_adu` | 33493.93 |
| `saturate_limit_adu_85pct` | 55704.75 |
| Sky median (pedestal) | ~33487 ADU (sample light frame) |
| DAO census (`dao_only_class`, n=696) | ambiguous_depth=539, artifact_negative=142, indeterminate=15, unmatched_in_range=0 |

---

### 5. Comparison-star gating condition (exact)

Primary candidate mask (`comp_selection_per_target.py:374-381`):

- `_dist_deg <= max_dist_deg` (config: 1.5 deg)
- `is_usable == True` (requires `zone == "linear"` from `pipeline.py:6314`)
- `~is_saturated`, `~is_noisy`, `~vsx_known_variable`, `~likely_saturated`

Additional filters applied downstream in the same module:

- Exclude variable-target catalog IDs (`comp_selection_per_target.py:433-434`)
- `|delta BP-RP| <= comp_max_delta_bprp` (0.99) when color known (`443-450`)
- Gaia nss/extobj Filter A (`469+`)
- `_dist_deg >= min_dist_arcsec/3600` (60 arcsec minimum separation, `417-418`, `550-551`)
- Adaptive `|delta mag|` filter (`565-633`); relaxes to 3.0 mag absolute ceiling when sparse
- DET fallback: `catalog_id.startswith("DET")` + `snr50_ok` + saturation vetoes (`514-523`)
- Proc-frame RMS gate: `comp_rms <= max_comp_rms` (0.1), authoritative, no relax above gate
  (`1637-1676`)
- Final selection requires `len(selected) >= n_comp_min` (config: 2)

**snr50_ok is required only for DET stars**, not for Gaia-matched comps passing `is_usable`.

---

### 6. Root-cause statement

Draft_502's zero light curves and empty comparison pool share a single upstream cause: the
masterstar flux-zone classifier in `_annotate_masterstars_flux_zones` (`pipeline.py:6272-6299`)
compares background-subtracted **aperture sums** against per-pixel **`noise_floor_adu`**, a
category error already recognized and fixed for saturation (which uses `peak_max_adu`). On this
pre-calibrated Newton V field with ~33.5k ADU sky pedestal, 99.2% of stars have aperture flux
far below `nf_val`, so 98.2% are classified `noisy3`, only 4 are `linear`/`is_usable`, and
comparison selection (`is_usable` gate at `comp_selection_per_target.py:376`) cannot build a
pool. Phase 1 writes an empty `comparison_stars_per_target.csv`; Phase 2A skips all photometry
eligible targets with `no_comps` (`photometry_core.py:1493-1518`). This is **not** draft_501's
annotate/write failure (zone column present, post-Task-A code ran); it is the **same defect
class as INV-MS-01** -- an absolute threshold calibrated implicitly for one flux scale (works
by numerical coincidence on VYVAR-calibrated draft_435 with sky ~2k ADU) but destroys another
(pre-calibrated high pedestal). A fix changes science outputs and will need a local P1 A/B
byte-identity gate before golden-ledger update.

---

## Errors (if any)

None (read-only diagnosis).

## Files changed

None (read-only task).
