CURSOR RESULT - 2026-09-14 EPSF-XVAL-A2-COMPARE-01

Date: 2026-09-14. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: 69a901d.
Class: MEASUREMENT + dev-only script. No production code change.
Live 516/517 and Archive read-only. EPSF-XVAL-01 closure is Milan's.

## Premise (Rule 0.1)

Compared: VYVAR ePSF instrumental / differential photometry
(frozen `vyvar_reference/proc_psf_flux.csv`, 134 frames, 7795
rows) versus PSFEx pass2 `MAG_PSF` / `FLUX_PSF` from the Linux
A2 kit. A1 compared the same VYVAR side to PythonPhot pkfit
(reference-limited: 34.34 / 31.79 mmag). The two legs share
the target, check, pinned ensemble, and residual construction
(`d` after per-star median, RMS over epochs). They are not
the same external code path.

Expected on-disk layout was
`session_20260907_epsfxval_a2/a2/{run_log.txt,work/,deg3/}`.
Actual Linux `OUT_DIR` is `a2/out/` (`run_log.txt`, `work/`,
`deg3/work/`; last line `done n_frames=134 n_fail=0`; 134
stems each with `*_pass2.cat`, `*.psf`, `meta.json`). Extra:
`a2/draft_000516_snapshot_era04_20260826/` snapshot copy.
Contents match the expected structure one directory down.
Path bound in `dev/xval_psfex/a2_compare.py` as `A2_OUT`.
`a2/` is gitignored (Milan 2026-09-14 local-only). Did not
raise; measurement used the bound actual path.

`targets.csv` n=65, columns catalog_id,ra,dec,role; keyed on
catalog_id; sha prefix `bfa24039`. Frozen sidecar
`lightcurves/lightcurve_1498613634033133184_psf.csv` present.
WCS from snapshot aligned lights
`Archive/Drafts/draft_000516_snapshot_era04_20260826/detrended_aligned/lights/NoFilter_60_2/`
(PC + CDELT; SCALE never used). Pass2 columns match
`pass2.param`. Ensemble for BO CVn is pinned
(`pinned_ensembles.csv` + sidecar `ensemble_source=pinned`):
`1497771992240531712, 1499200223486564608,
1497974027502858240, 1497368849430107904`.

## Matching

5-frame NN probe (stems 001/037/076/109/148; 325 attempts;
213 true-match distances): median 0.455 px, p95 0.793 px,
ambiguity rate 0. Adopted r_match=2.0 px. No silent drops:
shared pass2 row keeps the nearest and flags; two pass2
inside r flags `two_pass2_in_r`.

Census population: 65 `targets.csv` stars x 134 frames.
Deg2 totals: n_matched=5907, n_unmatched=2803, n_flagged=143,
n_nonfinite=500, n_ambiguous=4. Target, check, and all four
ensemble members: 134/134 matched, 0 unmatched, 0 nonfinite
(target FLAGS>0 on 1 epoch, kept).

## M1 (diagnostic; no bound)

Population: all 65 `targets.csv` stars. `d_si = m_psf_VYVAR
- m_PSFEx` with VYVAR `m = -2.5 log10(psf_flux)` from
`proc_psf_flux.csv` (same as `run_xval_a1.py` `_flux_to_inst_mag`)
and PSFEx `MAG_PSF` as-is. Per-star median removed; RMS over
epochs with both finite. Sorted by `phot_g_mean_mag` from live
`masterstars_full_match.csv`. This is the full matchable set,
not only the 60 PSF-LC stars A1 tabulated.

CSV: `a2_compare/m1_per_star.csv` and `m1_per_star_deg3.csv`.

| catalog_id | G | role | n_ok | n_nan | RMS mmag | median d mag |
|---|---:|---|---:|---:|---:|---:|
| 1496795041799526400 | 7.121 | psf_lc | 134 | 0 | 64.55 | -0.803 |
| 1500549977088828160 | 7.745 | psf_lc | 134 | 0 | 49.68 | -0.366 |
| 1497613731286514432 | 8.450 | check | 134 | 0 | 24.98 | -0.317 |
| 1498278351706325248 | 8.605 | psf_lc | 134 | 0 | 19.75 | -0.193 |
| 1497343732462852864 | 9.184 | psf_lc | 134 | 0 | 14.89 | -0.230 |
| 1499200223486564608 | 9.679 | ensemble | 134 | 0 | 21.19 | -0.233 |
| 1498613634033133184 | 9.720 | target | 134 | 0 | 17.78 | -0.248 |
| 1497771992240531712 | 9.752 | ensemble | 134 | 0 | 17.96 | -0.245 |
| 1497683722074089728 | 10.110 | psf_lc | 134 | 0 | 27.07 | -0.297 |
| 1497425371201155072 | 10.202 | psf_lc | 134 | 0 | 20.27 | -0.244 |
| 1496998382733052928 | 10.529 | psf_lc | 134 | 0 | 28.99 | -0.262 |
| 1497418258735289472 | 11.018 | psf_lc | 134 | 0 | 21.74 | -0.207 |
| 1496733984545821696 | 11.142 | psf_lc | 0 | 134 |  |  |
| 1497974027502858240 | 11.172 | ensemble | 134 | 0 | 23.37 | -0.213 |
| 1497245497969274240 | 11.192 | psf_lc | 134 | 0 | 35.85 | -0.256 |
| 1498795809366255488 | 11.308 | psf_lc | 133 | 1 | 23.03 | -0.234 |
| 1497368849430107904 | 11.523 | ensemble | 134 | 0 | 24.34 | -0.236 |
| 1497236186481686016 | 11.881 | psf_lc | 134 | 0 | 29.53 | -0.235 |
| 1498699086702005376 | 12.107 | psf_lc | 134 | 0 | 28.80 | -0.235 |
| 1498298211635183744 | 12.116 | psf_lc | 134 | 0 | 33.65 | -0.251 |
| 1498425548825498112 | 12.537 | psf_lc | 134 | 0 | 45.20 | -0.269 |
| 1498783199341798016 | 12.559 | psf_lc | 134 | 0 | 40.01 | -0.245 |
| 1500424804562041984 | 12.565 | psf_lc | 134 | 0 | 36.85 | -0.222 |
| 1497603835681942400 | 12.700 | psf_lc | 134 | 0 | 55.80 | -0.287 |
| 1497132660589966976 | 12.701 | psf_lc | 134 | 0 | 48.49 | -1.428 |
| 1498486880958321024 | 12.748 | psf_lc | 133 | 1 | 41.40 | -0.212 |
| 1498804639818507904 | 12.900 | psf_lc | 133 | 1 | 39.61 | -0.204 |
| 1498000793739050368 | 12.998 | psf_lc | 116 | 18 | 147.04 | -0.348 |
| 1499084499887740160 | 13.004 | psf_lc | 132 | 2 | 74.23 | -0.570 |
| 1497561779362267392 | 13.020 | psf_lc | 116 | 18 | 67.24 | -0.246 |
| 1496278752372040832 | 13.155 | psf_lc | 132 | 2 | 38.60 | -0.249 |
| 1498027456896444928 | 13.175 | psf_lc | 99 | 35 | 60.13 | -0.293 |
| 1497350638770267520 | 13.294 | psf_lc | 11 | 123 | 1594.32 | -1.816 |
| 1496293286541396480 | 13.415 | psf_lc | 114 | 20 | 59.33 | -0.234 |
| 1498058827337611392 | 13.455 | psf_lc | 109 | 25 | 103.69 | -0.194 |
| 1497639123133258752 | 13.534 | psf_lc | 109 | 25 | 138.55 | -0.236 |
| 1497227287309482624 | 13.561 | psf_lc | 54 | 80 | 60.00 | -0.193 |
| 1485560025830226432 | 13.607 | psf_lc | 88 | 46 | 76.96 | -0.228 |
| 1498617482323461376 | 13.630 | psf_lc | 55 | 79 | 145.79 | -0.173 |
| 1500693841313325696 | 13.690 | psf_lc | 76 | 58 | 77.85 | -0.220 |
| 1500461157165243648 | 13.735 | psf_lc | 25 | 109 | 255.49 | -4.643 |
| 1498842882207281152 | 13.814 | psf_lc | 68 | 66 | 124.21 | -1.679 |
| 1499021174889970816 | 13.819 | psf_lc | 98 | 36 | 127.59 | -0.162 |
| 1499006984318088320 | 13.924 | psf_lc | 75 | 59 | 78.42 | -0.178 |
| 1496037650087948160 | 13.980 | psf_lc | 59 | 75 | 145.69 | -0.197 |
| 1500327978819506944 | 13.981 | psf_lc | 118 | 16 | 80.99 | -0.195 |
| 1497491273179203456 | 13.988 | psf_lc | 48 | 86 | 178.43 | -0.522 |
| 1499210016011946496 | 14.078 | psf_lc | 38 | 96 | 124.24 | -0.214 |
| 1497169940906156032 | 14.195 | psf_lc | 55 | 79 | 107.73 | -1.535 |
| 1497871669842349184 | 14.322 | psf_lc | 25 | 109 | 1791.37 | -0.176 |
| 1485987254816323328 | 14.420 | psf_lc | 5 | 129 | 144.14 | -0.277 |
| 1502012464992313088 | 14.564 | psf_lc | 5 | 129 | 58.47 | -0.118 |
| 1497154753901690624 | 14.580 | psf_lc | 0 | 134 |  |  |

Remaining 12 `targets.csv` rows have NaN G and n_ok of 0-13
(edge / unmatched). Full file has 65 rows.

BO CVn M1 RMS = 17.78 mmag (deg2). A1 M1 on the same star was
49.51 mmag vs PythonPhot.

## M2 (product; the decisive number)

Population: frames where target + all 4 pinned ensemble members
have finite flux on BOTH sides. Combination: AIJ tot_C_cnts,
`delta = -2.5 log10(F_t / sum F_c)`, weights do not enter
(`run_xval_a1.py:449-478`; `photometry_lightcurve.py:677-679`
and `:765-774`). VYVAR fluxes from `proc_psf_flux.csv`. PSFEx
from matched `FLUX_PSF`.

n_finite = 134 / 134. Frames lost to matching: 0 (both sides).

Sidecar cross-check (rebuilt VYVAR target diff vs frozen
`psf_delta_mag`): RMS raw 0.00030 mmag, after-median 0.00029
mmag, n=134. Reference extraction premise holds.

| star | deg2 RMS mmag | deg3 RMS mmag | deg3-deg2 | n_finite |
|---|---:|---:|---:|---:|
| target 1498613634033133184 | 10.48 | 10.60 | +0.13 | 134 |
| check 1497613731286514432 | 21.41 | 21.09 | -0.32 | 134 |

Deg2 is primary. Deg3 is sensitivity only.

Target top-8 |resid| epochs deg2 (mmag): Light_148 (+28.90),
037 (-23.75), 076 (+23.11), 033 (+22.78), 005 (+22.25),
032 (+22.20), 053 (-21.82), 034 (+21.12).

Check top-8 |resid| deg2: Light_014 (+54.80), 089 (+48.96),
013 (+46.71), 032 (+45.30), 053 (+42.81), 076 (+42.26),
035 (+41.82), 041 (-41.35).

## Reading (binding; deg2 primary)

R-A2-3: > 10.0 mmag -> VYVAR-side root cause; first suspect EPSF-SHAPE-01.

Scale (not a reading): aperture-vs-AIJ 1.9503 mmag; A1
(reference-limited) 34.34 / 31.79 mmag; G3 internal bound
~12.5 mmag.

## Architect error 20

Every number above names its population (65-star matchable
set vs 134-frame identical-ensemble set vs 5-frame NN probe).
Do not quote M1 17.78 mmag as the product metric; the
decisive number is M2.

## G4 / gates

Live 516 unchanged: csv `bfa24039` / fits `13e77cf8` /
epsf `172f9540` (all PASS). `--fast --clean` OVERALL PASS
(1629 passed, 34 skipped; clean-tree PASS).
`git check-ignore` covers `a2/`; that tree was never staged.

## Files

- `dev/xval_psfex/a2_compare.py` (zero `src_py` imports of it)
- `.gitignore` entry for `.../a2/`
- `dev/results/context/session_20260907_epsfxval_a2/a2_compare/`
  m1, m2 epochs, match census, probe, frames-lost, summary.json
- this file

## STOP

No reland, no production change, no EPSF-XVAL-01 closure.
That decision is Milan's, on these A2-COMPARE numbers.
