CURSOR RESULT - 2026-09-16 EPSF-AC-02

Date: 2026-09-16. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: 36e5934 (LEDGER-EPSF-DOD-05 tip).
Class: MEASUREMENT, zero new photometry. No production change.
Live 516/517 and Archive read-only. a2/ never staged.

## Criteria quoted from D-EPSF-XVAL-DOD-05 at HEAD

Source: `docs/VYVAR_DECISIONS.md` at `36e5934`.

- 2a LINEARITY (flux scale vs brightness): on the VAL-03 isolated
  set, Theil-Sen slope of d = m_psf_inst - m_L_inst vs (G - 10),
  fitted simultaneously with colour: |b| <= 5.0 mmag/mag. Bootstrap
  std reported; PASS requires |b| <= 5.0 and |b| - 2*std <= 5.0 is
  NOT required (n is small; report both).

- 2b STABILITY of per-star offsets (what differential photometry
  relies on): split-half test - d_s derived on odd epochs, applied
  to even epochs (and vice versa); the robust scatter across stars
  of the residual (d_s,even - d_s,odd) <= 10 mmag. If 2b holds, the
  per-star scale offsets are constants absorbed by ensemble
  normalization and cannot enter a differential LC.

## Inputs confirmed

VAL-03 session only: `large_aperture_lc.csv` (r_L = 4.0 x FWHM),
isolated pass_selection n=18, frozen proc `psf_flux` in that LC,
PSFEx deg2 XPSF/YPSF (a2_compare match_rows_deg2 where present;
else local a2 pass2 NN on masterstars x+1,y+1), masterstars G /
BP-RP via VAL-03 `isolated_candidates.csv`.

Epoch set: n=134. Ordering source: `sorted(unique stem)` from
VAL-03 `large_aperture_lc.csv` (BO_CVn_Light_001 .. _148).

catalog_ids (n=18):
1499906247391001088, 1498735778606786816, 1497528072458898432,
1500727513856914944, 1497674651102612992, 1498062332030906880,
1497837207025312768, 1500460813567859456, 1497617407778562304,
1499200223486564608, 1497758935541325824, 1497145751650265600,
1496994156484645632, 1500486102335278592, 1496315070616056064,
1504489595970703872, 1497953377300128768, 1498677611865494016.

## Part A - criterion 2a (population: VAL-03 isolated n=18)

Simultaneous alternating Theil-Sen (4 rounds) + 2000-sample bootstrap:

| quantity | value |
|---|---:|
| a (mmag) | -127.31 |
| b (mmag/mag) | +2.678 |
| b_boot_std | 21.547 |
| c (mmag/mag BP-RP) | +137.501 |
| c_boot_std | 48.292 |
| robust scatter of resid (1.4826*MAD, mmag) | 40.360 |
| \|b\| | 2.678 |
| \|b\| - 2*boot_std (reported, not required) | -40.416 |

G-only fit (VAL-03 continuity): b = -4.934 mmag/mag, robust
scatter 58.581 mmag (bit-match to VAL-03 summary).

## Part B - criterion 2b (population: VAL-03 isolated n=18)

| split | robust scatter r (mmag) | noise expectation (mmag) | ratio | verdict |
|---|---:|---:|---:|---|
| odd/even by epoch_idx | 5.594 | 4.830 | 1.16 | PASS |
| first vs second half chronological | 6.604 | 4.830 | 1.37 | PASS |

Noise expectation = sqrt(2) * median_s( MAD-scatter(d(s,e)) / sqrt(n/2) ).
Both splits <= 10 mmag. Per-star table: `splithalf_per_star.csv`.

## Part C - RECORD (population: VAL-03 isolated n=18)

| quantity | value |
|---|---:|
| robust scatter d raw (mmag) | 59.4 |
| after colour (mmag) | 45.4 |
| after colour + phase pred (mmag) | 37.4 |
| c (mmag/mag) | +137.5 +/- 48.3 |
| phase Spearman rho (d_after_colour vs pred) | 0.129 |
| p | 0.61 |
| Theil-Sen slope d vs pred | 0.302 |

CORE-03 T1 noise-off surface is NOT confirmed as the dominant
per-star mechanism on this set (rho < 0.5). Remaining ~37 mmag
after colour+phase is the bright-end / residual common-scale floor.

## Readings fired

- R-AC1 PASS: |b|=2.678 <= 5.0.
- R-AC2 PASS: both splits <= 10 mmag.
- R-AC3 RECORD: decomposition as above; phase rho=0.129 (no
  CORE-03 confirmation note).
- R-AC4: criterion 2 of DOD-05 met on 516; closure waits only on
  criterion 3 at the 520 re-cut.

No fix in this task. Sequencing is Milan's.

## G4 (read-only after run)

csv bfa24039 PASS / fits 13e77cf8 PASS / epsf 172f9540 PASS.

## Files

- `dev/xval_psfex/epsf_ac_02.py`
- `dev/results/context/session_20260916_epsf_ac_02/`
  linearity_fit.csv, splithalf_per_star.csv, splithalf_summary.csv,
  decomposition.csv, summary.json
- this file

## Gates

`--fast --clean` OVERALL PASS on `697501f` (1629 passed, 34
skipped; clean-tree PASS). `a2/` never staged.

## STOP

Measurement only. No production change.
