CURSOR RESULT — 2026-06-19 (EXO-AS-TARGET — Gaia ID audit + saturation report)

What I did
Read-only audit of TOI-1131 Gaia `catalog_id` float corruption narrative, traced float-touch sites, re-verified exclusion/dedup on the **true** id, hardened exo promotion id resolution (`masterstar_row_gaia_key`), reverted validation-script masking. Saturation determination on draft 422 `V_60_2` target star (not whole-frame max).

## Issue 1 — Gaia catalog_id (BLOCKER audit)

### 1. True Gaia source_id (string-safe)

| Source | `source_id` | sep from TOI host |
|--------|-------------|-------------------|
| `GAIA_DR3/vyvar_gaia_dr3.db` nearest match | **1625373404725030528** | 0.0065″ |
| `masterstars_full_match.csv` (`dtype={'catalog_id':str}`) | **1625373404725030528** | — |
| `exoplanets/vyvar_exoplanet_local.db` host position | RA 248.433545°, Dec 61.718334° | — |

**True 19-digit Gaia DR3 id: `1625373404725030528`**

### 2. `...0400` vs `...0528`

| Value | In Gaia DB? | Role |
|-------|-------------|------|
| `1625373404725030528` | **Yes** (RA/Dec match TOI host) | **Correct id** |
| `1625373404725030400` | **No** (lookup returns NULL) | **Float64 corruption** of `...0528` |

Demonstration: `int(np.float64(1625373404725030528))` → `1625373404725030400`. Same via `normalize_gaia_source_id(1.6253734047250304e+18)`.

The STEP 1 assumption that `...0400` was canonical was **wrong**; the pipeline/masterstars CSV already carry the **correct** string `...0528` when read with `dtype=str`.

### 3. Float-touch sites (where `...0400` would appear)

| Site | Risk |
|------|------|
| `pd.read_csv` without `dtype={'catalog_id':str}` on masterstars | infers float64 → scientific string → normalizes to `...0400` |
| `normalize_gaia_source_id(float)` / `normalize_gaia_source_id('1.625e+18')` | returns `...0400` |
| `photometry_core.select_active_targets` `_gaia_id_str()` fallback `int(float(s))` | corrupts if hit |
| VSX path in `write_photometry_plan_files` | uses `_norm_gaia_id` on Gaia DB rows **as strings** (safe) |
| Exo promotion (before fix) | `normalize_gaia_source_id(row.get('catalog_id'))` only — unsafe if row is float-typed |

**Current draft 422 `masterstars_full_match.csv` on disk:** raw CSV quotes `"1625373404725030528"` (correct). Default pandas read → `np.float64(1.6253734047250304e+18)` (display only; normalize → `...0400`).

### 4. Re-verification on TRUE id `...0528`

| Check | Result |
|-------|--------|
| In `comparison_stars.csv` pool | **No** |
| Used as comp (`comparison_stars_per_target`) | **No** (as comp); has Phase-1 comp **rows as target** (expected) |
| Wrong id `...0400` in comp pools | **No** |
| Dedup VSX∩exo same true id | Unit test: one row with both labels when ids match `...0528` |
| Dedup VSX `...0400` + exo `...0528` | **Two rows** (no false merge) — test added |

### 5. Fix at cause (promotion path)

`pipeline._build_exoplanet_promotion_rows_from_masterstars`:
- Apply `catalog_id_series_for_masterstars_export()` on masterstars input (mirror plan write).
- Resolve promoted id via `masterstar_row_gaia_key(row)` (mirror `select_active_targets` / VSX discipline).

Validation script reverted: matches by **exact** `TRUE_GAIA_CID = 1625373404725030528`; no `exo_host_obj_id` pass workaround; no “saturated = LC pass” masking.

## Issue 2 — TOI-1131 saturated → no LC (OBSERVING finding)

**Target star** on `V_60_2` (78 proc frames, catalog_id `...0528`):

| Metric | Value |
|--------|-------|
| `peak_max_adu` (target) | min 28 984, max 58 589, median ~44 627 ADU |
| `saturate_limit_adu` (proc) | 65 535 |
| `saturate_limit_adu_85pct` (masterstars) | 55 704.75 (equipment × 0.85) |
| MASTERSTAR reference `peak_max_adu` | **58 555** > 55 704.75 → `zone=saturated` |
| `zone` on all 78 frames | `saturated` |
| Phase 2A LC | **None** (`skip_photometry=True`) |

**Conclusion:** TOI-1131 is **genuinely saturated** on the 60 s V frames (target peak exceeds 85% equipment ceiling). Skip is **correct pipeline behavior**, not a bug to suppress. Milan needs **shorter exposures** (or a fainter filter setup) for a usable transit LC on this host.

## Validation (production path)

Script: `tmp/validate_exo_as_target_422.py` — `write_photometry_plan_files` + `run_full_photometry_pipeline`.

Acceptance: true `catalog_id` in `variable_targets` + `active_targets`, no float-corrupted id in outputs, exclusion on true id, 8-VSX do-no-harm, active count 9. LC absence reported separately (Issue 2).

## Files changed

- `pipeline.py` — exo promotion id resolution
- `tmp/validate_exo_as_target_422.py` — true-id validation (masking reverted)
- `tests/test_exoplanet_variable_targets_merge.py` — true id + anti-false-dedup test
- `comp_qa_core.py` — locus excludes skip targets without LC (prior do-no-harm)

**No commit / no push** (awaiting sign-off).
