# VYVAR -- Development State

Last updated: **2026-06-09**.

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

Comp bounds (user-configurable): `phase01_comparison_n_comp_min/max` = **3 / 8** (the trust gate
and comp_qa derive their thresholds from these). `max_comp_rms` = 0.1; colour cut <= 0.79.

## Rigs (known sets)

| ID | Telescope | Camera | Scale | Site |
|----|-----------|--------|-------|------|
| 1 | Carl-Zeiss 200 mm | QHY294MM | ~9.77 arcsec/px (wide) | Jirny |
| 2 | Newton 300/1200 | C3-26000 | ~0.65 arcsec/px (fine) | Dablice |
| 3 | Noctutec 206/560 f/2.72 | C3-26000 | TBD | TBD |

Per-set config architecture is still pending (ROADMAP: TODO-MULTISET).

## Status snapshot

### Reference draft and byte-identity

- **Byte-identity reference:** `draft_000366` -- numeric photometry SHA **`770966c3...`**
  (283 LC + comp_quality + comparison_stars files). Holds across the PSF accuracy arc,
  EPSF-1 QC fix, and #4 fail-safety work (production PSF gated OFF; error paths only).
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
- **Tests:** **224 passed / 6 skipped** (last full `tests/` run).
- **Reporting:** R1 overflow guarantee holds (0 violations); R3 (aperture-vs-PSF overlay) pending.
