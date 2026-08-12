CURSOR RESULT - 2026-08-12 PSF smear / lacosmic root cause

## Premise
Compared draft_435 (frozen, pre-Batch-E CR) vs draft_505/506 (fresh) on identical
BO CVn raw frames. Task hypothesized RAM-handoff alignment resampling; evidence
shows damage is upstream in QC in-place L.A.Cosmic.

## Frame 012 before/after numbers
Comp star native ~(571,101); master-grid ~(570,99).

| Stage | draft_435 | draft_506 | Notes |
|-------|-----------|-----------|-------|
| Raw peak | 17328 | 17328 | identical SHA / sharp |
| Calibrated peak | 17246 | 3751 | 506 VY_COSM=True |
| Aligned @grid | 15699 | 2933 | astroalign both |
| FWHM_mom aligned | 5.15 | ~11.8 | 506 broadened |
| Aperture sum (r~5 native cut) | ~1.05e5 | ~4.4e4 | flux redistributed |

Repro on 435-cal f012:
- objlim=5.0 -> peak 3727, n_cr=4945 (matches 506 damage)
- objlim=8.0 -> peak 17270, n_cr=1556 (core preserved)

All 27 bad frames: objlim=8 peak ratio med/min = 1.00 / 1.00.

## Bad frame indices (27)
12 22 23 27 30 36 37 38 41 43 45 50 57 73 89 90
106 107 108 109 110 111 113 114 132 133 134

## Root cause (exact code)
`src_py/pipeline.py`:
- `_qc_enrich_one_frame` calls `_remove_cosmics_lacosmic` after sky-surface
- `_remove_cosmics_lacosmic` -> `astroscrappy.detect_cosmics` with
  former default `lacosmic_objlim=5.0`
- draft_435 era: `VY_COSM=False` (CR not applied)
- draft_505/506: Batch-E CR default ON; star cores false-positive as cosmics

NOT: astroalign kernel, double resample, RAM-handoff flush, WCS reproject.
Aligned products simply inherit already-eaten calibrated pixels.

## Fix landed
1. `lacosmic_objlim` default 5.0 -> 8.0 (`config.py`, PARAMS.md)
2. QC workers receive enable/sigclip/objlim from parent AppConfig (was always
   fresh `AppConfig()` ignoring overrides)
3. Header GAIN/RDNOISE <=0 treated as missing (`GAIN=0.0` on these frames)
4. Tests: `dev/tests/test_lacosmic_star_core.py` (5 passed)
5. `--fast` OVERALL PASS (1296 passed / 27 skipped)

## Close gate status
| Gate | Status |
|------|--------|
| Cause found + code location | PASS |
| Frame-012 numbers + fix repro | PASS |
| Unit tests + --fast | PASS |
| Fresh draft_506 full LC / TRUST | PENDING (needs Milan re-run; on-disk calibrated already damaged) |
| draft_435 no regression | PENDING re-run (code default change only affects CR path) |

## Data
`dev/results/context/session_20260812_psf_smear_lacosmic/`
- frame_peak_census.csv
- bad_frames.txt
- frame012_psf.txt

## Decision note for Milan
If any field still shows CR-eaten cores at objlim=8, options are raise further,
disable `enable_lacosmic`, or inject real equipment gain/RN into lacosmic (headers
currently carry GAIN=0.0). Do not QC-reject the 27 frames -- raw is good.
