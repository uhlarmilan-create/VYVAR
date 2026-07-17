# VYVAR -- Runbook: Chi_and_H baseline run (re-establish byte-identity anchor)

**Date:** 2026-06-10 -- **Author:** Claude (read-only runbook). Executed by Milan / Cursor on the
Windows tree (data is local; Claude cannot run it). ASCII-only.

## Context

- **Why:** all drafts were deleted, including the byte-identity reference (`draft_000366` /
  `770966c3`). This run re-cuts a fresh anchor.
- **Data:** `C:\ASTRO\python\VYVAR\Archive\Chi_and_H`, **already calibrated** FITS — **must be
  retained** (only non-regenerable input; catalog is zaloha on disk). Filter-wheel
  setups **B / V / R / L** (`B_20_2`, `V_20_2`, `R_20_2`, `L_20_2`). **V** = visual/green
  (`G/` folder); **L** = clear/no-filter (broadband). Rig: Newton 300/1200 + C3-26000, **binning
  2x2 -> ~1.3 arcsec/px**, Dablice.
- **Field:** h & chi Persei (Double Cluster), center forced to **(RA 35.15, Dec 57.13)** because
  the FITS carry no RA/DEC.
- **Anchor type:** this is a **Newton / bin2 (~1.3 arcsec/px)** anchor (reproduces the old
  `draft_000380` class), not the wide-rig `draft_000366` class. Fine for regressing the #3
  short-baseline change (rig-independent). A separate wide-rig anchor can be cut later if wanted.

## Canonical entry point

`python scripts/chiandh_night_run_bvr.py`

What it does (read before running): forces the field center; sets `skip_processed_directory=True`
and `psf_photometry_enabled=False`; runs `VYVAR_CT_PROTOTYPE=1`; registers equipment (Phase A);
runs night pipeline (calibrated import -> platesolve -> MASTERSTAR -> Phase 0/1/2A -> comp_qa ->
trust -> reports); CT target presel hook. Params baked in: `plate_fov_deg=1.25`,
`dao_fwhm_px=3.5`, `dao_threshold_sigma=3.5`, `catalog_match_max_sep_arcsec=3.0`,
`max_catalog_rows=20000`, detect 200-4000 stars. It **restores config on exit**.

## Prerequisites (check before the run)

1. **Config (zaloha only):** `config.json` must use **`GAIA_DR3/zaloha/`** for
   `gaia_db_path`, `blind_index_fine_path`, and `blind_index_wide_path` (G<=16). **Never** point
   at the in-progress full-sky `GAIA_DR3/vyvar_gaia_dr3.db`. No TAP, no field DB, no
   `chiandh_build_field_db.py`. Record `git rev-parse HEAD` in the result JSON.
2. **Pre-calibrated import is healthy:** confirm the canonical proc-CSV glob fix is in the tree --
   `load_proc_pivot` uses `list_proc_csvs` / `PROC_CSV_GLOB="proc_*.csv"`;
   `tests/test_proc_csv_glob.py` passes. This is the fix for `comp_qa_core` silently zeroing
   `n_clean` on pre-calibrated imports (exactly this kind of data).
3. **Code state:** pin with `git rev-parse HEAD` (recorded in result JSON).
4. **Anchor model:** SHA fingerprint + this recipe; ephemeral `draft_*` may be deleted.
   Re-verify by regenerating and comparing SHA to `3f7c9e7a...` / `d5b72d08...`
   (`tests/photometry_sha.py`). Historical pre-drift: `203254fd...` / `95a5515a...`.
   Regression vs historical cut: `compare_photometry_science_meaningful` (PROCESS).
   **RETIRED:** `f4bcc0ee` (truncated draft_385), `d246a5be` (TAP draft_382).

## Run

```
python scripts/chiandh_night_run_bvr.py
```

Result JSON: `tmp/chiandh_bvr_night_run_result.json` (draft_id, draft_dir, timings, masterstar
stats, calibrated/platesolve setups, CT prototype path).

## Post-run verification checklist (gates a valid anchor)

- [ ] `night_run_success == true`; `draft_id` / `draft_dir` recorded.
- [ ] **`photometry_completeness`**: every setup `ok: true` (>=90% summary/active); no truncation.
- [ ] **Plate scale ~= 1.3 arcsec/px** (NOT ~0.65) in MASTERSTAR WCS / `pipeline_meta` -- confirms
      bin2 was handled. If it shows ~0.65 or 9.77, stop and check optics/binning registration.
- [ ] Per-filter setups present for **B / V / R / L** under `platesolve/` (MASTERSTAR.fits each)
      and `calibrated/lights/`.
- [ ] **`n_clean` populated** (non-zero) in the comp_qa sidecars / `photometry_summary.csv` --
      the pre-calibrated n_clean regression (prereq 3) must not recur.
- [ ] Trust distribution recorded: at `comp_trust_min_comps=5` expect **1382 YELLOW / 106 RED**
      (1488 summary rows on draft_387 class). Pre-floor-5 counts (1400/88) are **not** the
      current baseline.
- [ ] **0 PDF overflow** in the SUMMARY MEASURE REPORT (R1 guarantee).
- [ ] CT prototype emitted (`ct_prototype.csv`) -- CT path exercised for B/V/R/L.
- [ ] **`git rev-parse HEAD`** recorded in STATE/JOURNAL (commit hash is not auto-captured).
- [ ] **Compute byte-identity SHA-256** (`compute_photometry_sha`): core + full vs recorded
      `3f7c9e7a...` / `d5b72d08...` — **draft-independent** gate. Record SHAs, zaloha recipe,
      config snapshot, and ephemeral `draft_id` in `VYVAR_STATE.md` / `VYVAR_JOURNAL.md`.
      When comparing to the historical `203254fd...` cut, also run
      `compare_photometry_science_meaningful` (benign-drift acceptance).

SHA helpers: `tests/photometry_sha.py` (`compute_photometry_sha`).

## Notes / provenance to capture

- Anchor is **catalog-dependent**: built against **zaloha** (`GAIA_DR3/zaloha/vyvar_gaia_dr3.db`,
  G<=16). It will legitimately shift when the catalog is rebuilt (e.g. full-sky cut, Gaia DR4).
  Record provenance so a future SHA mismatch is explainable, not alarming.
- Trust distribution at G<=16 differs from the retired TAP field-DB cut (more RED); the anchor
  is numeric SHA and trust-independent.
- PSF stays OFF (enforced by the script). bin2 (~1.3 arcsec/px) is the aperture-workhorse regime;
  it does NOT unblock the PSF/NEIGHBOR-SUB enablement gate (which needs bin1 ~0.65 arcsec/px).
- Post-#3 acceptance: re-run recipe, compare SHA to recorded values (not to a prior draft tree).
  Truncated subsets exercise `short_baseline` (see `VYVAR_LC_QUALITY_SHORT_BASELINE_SPEC.md`).
