CURSOR RESULT - 2026-09-08 EPSF-XVAL-A2-PREP-01

Date: 2026-09-08. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: 3c1306c.
Class: dev-only kit + read-only reference extraction.
No production code change. Live 516/517 read-only.
Milan runs the kit on Linux; A2-COMPARE is afterwards.

## Architect error 17 (carry-in, on the record)

EPSF-XVAL-A1 left fitrad/psfrad unspecified. The harness tied the
fit radius to the disputed ePSF meta FWHM (3.30 px) instead of the
frame FWHM authority (~5.19 px), on the quantity EPSF-SHAPE-01
flags as suspect. Cite `dev/xval_pythonphot/run_xval_a1.py:528-530`
(`fitrad = float(psf_meta.get("fwhm_px") or 3.3)`). The
FWHM-authority principle existed and was not applied to the
harness. Same class as errors 9-16.

A2 wires the lesson: `SEEING_FWHM` is
`qc_metrics.fwhm_px * WCS plate scale` per frame. Not the ePSF
meta 3.30.

## Refute (governing artifacts, before the kit was written)

1. Snapshot layout is not `SNAPSHOT_DIR/Light_*.fits`.
   `_copy_frozen_anchor_inputs` (`dev/scripts/session_baseline_check.py:716-742`)
   and the C-EXPORT-GAP sandbox put aligned lights at
   `detrended_aligned/lights/NoFilter_60_2/` and QC at
   `calibrated/lights/qc_metrics.csv`. Filenames are
   `BO_CVn_Light_NNN.fits` (134 science + MASTERSTAR). The kit
   searches `*Light_*.fits` under that tree and excludes
   MASTERSTAR. A `Light_*.fits` prefix glob would have found
   nothing.

2. There is no CD matrix. WCS is PC + CDELT=1
   (`PC1_1 ~ -0.002715` deg). Header `SCALE=9.55169` is not the
   WCS scale. Plate scale is computed as in
   `src_py/unit_resolver.py:34-58` (`proj_plane_pixel_scales`).
   Light_001: 9.774 arcsec/px source=`wcs_pc`.
   Architect synthetic used 9.77 arcsec/px; matches WCS, not SCALE.

3. Header `GAIN=0.0` is present and invalid. Using it would
   silently diverge from A1 `g_pt=0.637067`
   (`lightcurve_1498613634033133184_psf.csv` gain_authority line).
   Kit treats GAIN missing or <=0 as fallback 0.637067 and logs
   the source. SATURATE/MAXLIN absent; SATUR_LEVEL=60000 as
   specified.

4. Ensemble members have no dedicated PSF LC (same as A1). Frozen
   comparison uses live proc `psf_flux` for those four IDs.
   Check `1497613731286514432` likewise has proc flux only.

Did not raise. Kit and freeze proceeded with the corrections above.

## What I did

Kit `dev/xval_psfex/` (self-contained; no `src_py` import on the
Linux path):

- `pass1.param` / `pass2.param` -- container-tested keyword lists,
  verbatim.
- `run_all.sh SNAPSHOT_DIR OUT_DIR` -- binary detect
  `source-extractor|sextractor|sex` and `psfex`; fail loud with
  the apt line. Per-frame meta via `frame_meta.py`. Pass1
  FITS_LDAC + listed flags. PSFEx `PSFVAR_DEGREES 2`, then deg 3
  into `OUT_DIR/deg3/`. Pass2 ASCII_HEAD + `-PSF_NAME`.
  `OUT_DIR/run_log.txt`. No tar.
- `RUNBOOK.md` -- exactly five Milan steps.
- `targets.csv` -- 65 IDs (60 PSF LC + 4 ensemble + check),
  ra/dec from live `masterstars_full_match.csv`, provenance
  sha256 prefix `bfa24039`.
- `frame_meta.py` / `extract_vyvar_reference.py` -- helpers
  (Linux run uses only frame_meta).

Frozen VYVAR side (read-only live 516) at
`dev/results/context/session_20260907_epsfxval_a2/vyvar_reference/`:

- `proc_psf_flux.csv` -- 7795 rows from 134 live proc files
- `qc_metrics.csv` -- snapshot copy (the file Linux will read)
- `lightcurves/lightcurve_<target>_psf.csv` -- sidecar has pinned
  ensemble ids / `g_pt=0.637067`
- `ensemble_sidecar.json` -- same four IDs and weights as A1/A1B
- `PROVENANCE.md` -- live G4 hashes

Archive was not written.

Smoke (Windows, no SExtractor): Light_001 ->
SEEING_FWHM=51.847 arcsec (5.30456 px * 9.774 "/px),
gain=0.637067 source=`g_pt_fallback`, satur=60000.

## Pre-registered A2 readings (fixed now; applied at A2-COMPARE)

On M2 (target AND check, same ensemble memberships/weights as
A1/A1B, median offset removed):

- R-A2-1: both <= 3.0 mmag -> AIJ-class external agreement of the
  ePSF chain (the validation EPSF-XVAL-01 exists for).
- R-A2-2: 3-10 mmag -> partial agreement; per-epoch contributors
  tabled; Milan decides next step.
- R-A2-3: > 10 mmag -> the disagreement is not explained by the
  A1 harness weaknesses (spatial PSF + native positions are in);
  root-cause moves to the VYVAR side with EPSF-SHAPE-01 as the
  first suspect.

Diagnostic expectation (no bound): the A2 per-star median offset
spread should collapse versus A1B's -0.38..-0.92 mag if the
single-PSF model was the sheet's cause.

STOP after A2-COMPARE in every case; closure is Milan's.

## Gates

G4 live 516 after the freeze (read-only):

| product | sha256 prefix | verdict |
|---------|---------------|---------|
| masterstars_full_match.csv | bfa24039778f437b... | PASS |
| MASTERSTAR.fits | 13e77cf8a1dcb4e7... | PASS |
| masterstar_epsf.fits | 172f95403beae36d... | PASS |

`--fast --clean` OVERALL PASS (1624 passed, 34 skipped; clean-tree
PASS). No `--full`. No `--full-epsf`. `src_py` does not import
`dev.xval_psfex`.


## Errors (if any)

None that stopped the kit. Header GAIN=0.0 / missing CD / Light_*
glob were caught in the refute and coded around.

## Files changed

dev/xval_psfex/
dev/results/CURSOR_RESULT_EPSF_XVAL_A2_PREP.md
dev/results/context/session_20260907_epsfxval_a2/vyvar_reference/

STOP: kit is ready for Milan's Linux run. A2-COMPARE and
EPSF-XVAL-01 closure remain his.
