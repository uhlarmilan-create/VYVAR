# VYVAR -- Development State

Last updated: **2026-06-11** (trust/anchor/reliability session wrap; zaloha anchor confirmed;
completeness gate; trust-correctness cluster; BLE001 guard).

This is the **entry point**: a snapshot of what is true *now* + an index. It deliberately holds
no history and no open-task detail -- those live in the linked files.

| File | Holds |
|------|-------|
| `docs/VYVAR_ROADMAP.md` | Open work (the only place to look for "what's next"). |
| `docs/VYVAR_DECISIONS.md` | Durable design decisions + *why* they hold. |
| `docs/VYVAR_JOURNAL.md` | Chronological session log (history, append-only). |
| `docs/VYVAR_PROCESS.md` | How we work: Definition of Done, byte-identity discipline, config<->UI parity, tests. |
| `docs/VYVAR_PARAMS.md` | Config-key <-> default <-> clamp <-> UI-location registry. |
| `docs/VYVAR_VALIDATION.md` | Inject-and-recover synthetic validation harness (matrix, FAIL policy). |
| `docs/VYVAR_PIPELINE_CZ.md` | Czech pipeline manual for the paper (ASCII, rev. 2026-06-09). |
| `docs/VYVAR_GAIA_DR3_AUDIT.md` | Gaia DR3 ingest audit (build schema, match, ref mag; 2026-06-10). |
| `docs/VYVAR_LC_QUALITY_SHORT_BASELINE_SPEC.md` | `short_baseline` LC-quality spec #3 (rev b, ready; 2026-06-10). |
| `docs/VYVAR_RUNBOOK.md` | Chi_and_H zaloha-only night-run procedure (alias → baseline runbook). |
| `docs/VYVAR_CHIANDH_BASELINE_RUNBOOK.md` | Chi_and_H baseline re-cut procedure (byte-identity anchor; 2026-06-11). |
| `docs/VYVAR_TRUST_CHECKSTAR_HARDENING_SPEC.md` | Trust Findings A/B + CS-1 hardening (2026-06-11). |
| `docs/VYVAR_CHECKSTAR_SELECTION_SPEC.md` | Check-star selection CS-2..4 (2026-06-11). |
| `docs/VYVAR_COMP_FLOOR_POLICY_SPEC.md` | Comp trust floor policy; Option B adopted. |
| `docs/VYVAR_MATH_PHYS_AUDIT.md` | Math/physics audit (first pass; citation scoping landed). |
| `CITATIONS.bib` | Single source of truth for all algorithm/software citations. |

---

## Mission

A high-automation differential-photometry pipeline that lets amateur astronomers contribute
science with confidence: **trust in the numbers** (comp-stability QA via comp_qa + per-target
trust gate) and **guardrails for non-experts**. Independent extraction cross-validation lives
in the offline `xval_run.py` harness (not in-pipeline). **Aperture photometry remains the
validated workhorse** on the wide rig; **PSF is validated publication-grade at fine scale on
synthetic truth** (draft-367-like, mismatch ~0) but **gated OFF** in production LC until a
real Newton / dense-field draft passes the characterization gate.

## Pipeline (current)

```
Raw -> Calibrate -> QC (in-place) -> Align -> MASTERSTAR + Gaia DR3 catalog
   -> Phase 0+1: comp selection per target (colour tier + stability gate, distance gate)
   -> Phase 2A: differential photometry (Broeg 1/sigma^2 ensemble; SNR-optimal per-star aperture)
   -> comp_qa (Sokolovsky LOO QA, read-only)
   -> trust gate (GREEN/YELLOW/RED per target; comp-health + check-star + lc_quality)
   -> reports/exports (PDF SUMMARY MEASURE REPORT, AAVSO, VarAstro)
```

Plate scale is **WCS-derived** (~9.77 arcsec/px on the wide rig). Differential weighting is **Broeg
(2005) 1/sigma^2**, order-independent. Comp selection ranks colour tier first, stability second
(both gated), proximity as a distance gate only. (See DECISIONS for the rationale.)

## Production defaults (feature flags)

| Area | Flag | Default |
|------|------|---------|
| Comp QA | `comp_qa_enabled` | **ON** |
| Trust gate | `trust_flag_enabled` | **ON** (inform-only) |
| Proximity tie-break | `phase01_comparison_proximity_tiebreak` | OFF (reverted) |
| PSF (all) | `psf_photometry_enabled`, `psf_adaptive_enabled`, `psf_grouper_enabled`, `psf_spatial_enabled` | OFF |
| NEIGHBOR-SUB | `psf_neighbor_sub_enabled` | OFF |
| COG aperture corr. | `cog_aperture_correction_enabled` | OFF |
| Crowding classifier | `crowding_classifier_enabled` | OFF (wide rig) |
| Detrend | `sysrem_enabled`, `savgol_detrend_enabled` | OFF |
| Skip processed/ | `skip_processed_directory` | OFF |

PSF flags stay **OFF** on the wide rig (correct). The PSF path is now **validated-but-gated**:
enable only on characterized fine-scale data after the Brno / Newton characterization gate
(see DECISIONS + ROADMAP).

Comp bounds (user-configurable): Phase-1 selection `phase01_comparison_n_comp_min/max` = **3 / 8**
(unchanged). Trust-only floor `comp_trust_min_comps` = **5** (`strong = min+2` → **7** at
defaults); `check_star_min_epochs` = **5**; CS-2 artefact floor `check_select_rms_floor` =
**1e-4**; CS-4 uses `aperture_correction_max_contamination` = **0.15** when
`contamination_idx` is present. `max_comp_rms` = 0.1; colour cut <= 0.79.

## Rigs (known sets)

| ID | Telescope | Camera | Scale | Site |
|----|-----------|--------|-------|------|
| 1 | Carl-Zeiss 200 mm | QHY294MM | ~9.77 arcsec/px (wide) | Jirny |
| 2 | Newton 300/1200 | C3-26000 | ~0.65 arcsec/px (fine) | Dablice |
| 3 | Noctutec 206/560 f/2.72 | C3-26000 | TBD | TBD |

Per-set config architecture is still pending (ROADMAP: TODO-MULTISET).

## Status snapshot

### Gaia DR3 catalog integration

PM (`pmra`/`pmdec`) and `ruwe` are **NOT** in the DR3 catalog; **deferred to the DR4 build**
(~Dec 2026). Platesolver PM propagation is present but a no-op against DR3. Fine-scale dense
fields carry the GAIA-1 mis-association caveat until DR4 (DECISIONS).

### Reference draft and byte-identity

The anchor is a **SHA fingerprint + regeneration recipe**, not a dependency on any draft tree
surviving. Byte-identity is re-verified by regenerating from the recipe below and comparing
computed SHA to the recorded values (`tests/photometry_sha.py`).

- **Core SHA** `203254fd75ea5874f5986eac3f478260c2e7e5a9c2636bfecf2b31244cfb09ba` (2806 files:
  `lightcurve_*.csv` + `comp_quality_*.json` + `comparison_stars_per_target.csv`).
- **Full SHA** `95a5515a6c15a473b6fcd29d3afe0c3b78d88a2da434f8a1c03f28dbe2783c24` (4285 files:
  core + `comp_qa_*.json`). SHA set excludes `lc_quality_flag` / trust.
- Cut from ephemeral `draft_000386` (2026-06-11); **confirmed** `draft_000387` (byte-identical).
  **RETIRED:** `f4bcc0ee` / `bd0b1792` (draft_385 truncated photometry — false success);
  `d246a5be` / `30a2f461` (draft_382 TAP field DB G<=19.5).

**Regeneration recipe** (`docs/VYVAR_CHIANDH_BASELINE_RUNBOOK.md`) — **no TAP / no field DB:**

1. **Source data (must retain):** `Archive/Chi_and_H` — pre-calibrated FITS (only non-regenerable input).
2. **Catalog + blind index:** `GAIA_DR3/zaloha/vyvar_gaia_dr3.db` (G<=16), zaloha blind PKLs
   (`gaia_triangles_fine.pkl`, `gaia_triangles_wide.pkl`). **Do not read** in-progress
   `GAIA_DR3/vyvar_gaia_dr3.db`.
3. **Run:** `python scripts/chiandh_night_run_bvr.py` (#3 code; Newton bin2 ~1.30"/px).
4. **Verify:** `compute_photometry_sha(draft_root)` core + full vs recorded SHAs.

**Setups (filter-wheel labels):** `B_20_2`, `V_20_2`, `R_20_2`, `L_20_2` — **B/V/R/L** are wheel
positions. **V** = visual/green (`G/` folder); **L** = clear/broadband (`L_20_2` in anchor).

**Provenance at anchor cut (2026-06-11, zaloha):**

| Item | Value |
|------|-------|
| `git_commit` | `39690b7fa398dc0445f8956d318296aeb0cfe725` |
| Catalog | `GAIA_DR3/zaloha/vyvar_gaia_dr3.db` (G<=16) + zaloha blind PKLs |
| Rig | Newton 300/1200 + C3-26000, bin2 ~1.30"/px |
| Ephemeral draft | `draft_000386` (~1401 LCs; deletable) |
| Completeness gate | `night_run.audit_photometry_completeness` — >=90% summary/active per setup |

Scoped trust at cut (`comp_trust_min_comps=5`, floor-5 baseline): **1382 YELLOW / 106 RED**
(1488 summary rows; re-trust on draft_387). Pre-floor-5 counts were 1400/88 — superseded.
Anchor photometry SHA is numeric and trust-independent.
- **Last science-validated wide draft:** `draft_000365` (V842 Her, 127 frames, 143 targets).
- **CT science locked:** h & chi Per `draft_000380` (Johnson-Cousins B/V/Rc); details in JOURNAL.

### PSF photometry (gated OFF in production LC)

Infrastructure complete. Validated **publication-grade on synthetic fine-scale** truth
(draft-367-like, ePSF-vs-star mismatch ~0):

| pillar | result (V3d harness, mag 12-17) |
|--------|----------------------------------|
| ACCURACY | mid-mag bias **<~2%** via brightness-independent **sky-only fit weights** (`psf_weight_mode=sky_only`; Astier 2013 / Lacroix 2025) |
| PRECISION | PSF scatter wins from ~mag 13 |
| UNCERTAINTY | P3 ~1 via **sandwich** variance (`psf_err_mode=sandwich_skyonly`) |
| ePSF FWHM QC | robust azimuthal-profile estimator (EPSF-1); warning band [0.80, 1.25] (diagnostic only) |

Sky estimate: aperture-consistent **annulus** / **residual_annulus** (`psf_sky_method` column).
Real-field enablement **blocked on a Newton / dense-field draft** (incoming Brno data will
unblock after the characterization gate).

### NEIGHBOR-SUB

**VALIDATED_FINE_SCALE_IDLE** -- works at fine scale (A9 HV ~83%, FAIL-SILENT 0), fail-safe
guards + full provenance, gated OFF; no current real use case (draft 367 sparse crowding).
Coarse / under-sampled fields fall back to SAFE_LOW_YIELD (correct REFUSE, not silent deblend).

### Fail-safety #4 (2026-06-08)

- MASTERSTAR WCS persist: **fail-closed** (draft solve fails; Phase 2A blocked for that draft).
- Edge-ok check: **fail-open + loud flag** (`edge_filter_failed` on `variability_candidates.csv`).
- Dead UI modules removed (`ui_photometry_results`, `ui_suspected_lightcurves`).

### Citations (PSF arc)

Astier et al. 2013, Lacroix et al. 2025, Guy et al. 2010, Stetson 1987, Mighell 1999 wired in
`CITATIONS.bib` where the methods run.

### Cross-validation, trust, tests, reporting

- **Cross-validation:** CLOSED for the aperture path (offline `xval_run.py`: sep reproduces
  VYVAR to 0.2 %/frame); in-pipeline `sep_xval` retired 2026-06-03; PSF cross-val deferred.
- **Trust distribution (draft_000365 baseline):** GREEN 69 / YELLOW 59 / RED 15.
- **Tests:** **259 passed / 14 skipped** (last full `tests/` run; incl. BLE001 regression guard).
- **Lint:** `ruff check . --select BLE001,E722` clean (`pyproject.toml` + pre-commit + pytest).
- **Reporting:** R1 overflow guarantee holds (0 violations); R3 (aperture-vs-PSF overlay) pending.

---

## Top of mind (parked — see ROADMAP)

- **Reserved check-star** (hold-one-out by design) — **moves photometry anchor**; defer until
  explicit re-cut discipline.
- **AAVSO-standard output #4** — G→B/V/Rc (Broeg band/colour point).
- **TODO-MULTISET** — per-rig config (wide vs fine).
- **TODO-GS8** — Phase-3 global ZP / multi-night matching.
- **DR4 build** (~Dec 2026) — J2017.5 epoch hook at `vyvar_platesolver.py:63`.
- **PSF / NEIGHBOR-SUB** — needs bin1 ~0.65"/px real data (Brno gate).
- **`build_gaia_catalog.py` adaptive-split** — apply at next full-sky build (DEFERRED this commit).
