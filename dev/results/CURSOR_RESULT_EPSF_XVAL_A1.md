CURSOR RESULT - 2026-09-07 EPSF-XVAL-A1-PYTHONPHOT-01

Date: 2026-09-07. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: fbb5d58.
Class: MEASUREMENT + dev-only harness. No production code change.
Live 516/517 read-only. EPSF-XVAL-01 does not close here.

## Refute check

1. Literature: PythonPhot `pkfit` is the IDL AstroLib / DAOPHOT 1985
   January 25 port (Stetson 1987 lineage): weighted least-squares fit
   of `scale * (pixel-integrated Gaussian + residual LUT)`. Cited
   `dev/xval_pythonphot/vendor/pkfit.py:5-13` and `:100-103`. This is
   NOT an Anderson & King ePSF. Did not raise; measurement proceeded.
2. Faithful py3: vendored code needed NumPy 2 / astropy / Py2-int-div
   shims (PROVENANCE.md). None change the DAO math. `if fnoise:` left
   alone; harness omits `noiseim`.
3. Snapshot `draft_000516_snapshot_era04_20260826` has aligned lights
   + proc x,y and **no** `masterstar_epsf.fits`. G-EPSF products
   (model SHA `172f9540`, PSF LCs, meta n_stars_used=67, created
   2026-08-22T19:11:31Z) were read from live 516. No sandbox ePSF
   rebuild (~4 h not taken).
4. Live `build_epsf_science_set` is no longer the 2026-08-22 pool
   (`n_after_science_scope=3` today vs meta 68). Meta lists no IDs.
   Pool taken from the retained G-EPSF census
   `dev/results/context/session_20260822_epsf_valid_02_r1r4/r3_build_stars_516.csv`
   (67 rows; same n / timestamp as live meta).
5. `VyvarDatabase()` writes tables; not instantiated. Gain from
   `psf_internal_lc._load_gain_authority` (live
   `gain_photon_transfer.json`, same `g_pt` stamp as the PSF LC
   header). RN from EQUIPMENTS via sqlite `mode=ro` +
   `param_resolver.resolve_read_noise` (cite
   `psf_photometry.py:770-775`).

## What I did

Vendored the minimal PythonPhot set under
`dev/xval_pythonphot/vendor/` (upstream SHA
`7992eb6ba98ee946a4b8d27eb55ce0028f76374e`) with LICENSE.rst and
PROVENANCE.md. Smoke test recovers a daoerf Gaussian vs aper flux
to <0.1%. `src_py` has zero imports of `xval_pythonphot`.

Harness `dev/xval_pythonphot/run_xval_a1.py`: sandbox copy of the
era04 snapshot via `_copy_frozen_anchor_inputs`
(`session_baseline_check.py:716`); per-frame `getpsf` +
`pkfit_norecenter` at snapshot proc x,y; M1/M2; 5-frame recenter
sensitivity. Live Archive never written.

## Output / findings

**Route:** sandbox snapshot aligned lights (134 FITS; Light_002 etc.
already absent from the freeze). PSF LCs + ePSF meta/model from live
516. No ePSF rebuild.

**Gain / RN:** gain=0.637067 e-/ADU source=`g_pt`; RN=15.2 e-
source=`db` (EQUIPMENTS.READNOISE_E=7.6 * XBINNING=2). fitrad=3.3014
px (meta FWHM); psfrad=8.0 px.

**getpsf:** all 134 frames produced a PSF (n_psf_used 66 or 67). No
frame-level getpsf drop.

### M1 (implementation; no hard bound)

`d_si = m_psf_VYVAR - m_PP` on instrumental mag; per-star median
removed; RMS over epochs. Full PSF LC star set (n=60), sorted by G.
CSV: `dev/results/context/session_20260907_epsfxval/m1_per_star.csv`.

| catalog_id | G | n_ok | n_fail | RMS mmag | median d mag |
|---|---:|---:|---:|---:|---:|
| 1496795041799526400 | 7.121 | 134 | 0 | 86.33 | -0.9033 |
| 1500549977088828160 | 7.745 | 134 | 0 | 32.16 | -0.4919 |
| 1498278351706325248 | 8.605 | 134 | 0 | 50.25 | -0.6417 |
| 1497343732462852864 | 9.184 | 134 | 0 | 31.59 | -0.5542 |
| 1498613634033133184 | 9.720 | 134 | 0 | 49.51 | -0.6502 |
| 1497683722074089728 | 10.110 | 134 | 0 | 56.05 | -0.6085 |
| 1497425371201155072 | 10.202 | 134 | 0 | 37.86 | -0.5959 |
| 1496998382733052928 | 10.529 | 134 | 0 | 43.59 | -0.5683 |
| 1497418258735289472 | 11.018 | 134 | 0 | 42.47 | -0.5521 |
| 1496733984545821696 | 11.142 | 0 | 134 |  |  |
| 1497245497969274240 | 11.192 | 134 | 0 | 61.35 | -0.6017 |
| 1498795809366255488 | 11.308 | 134 | 0 | 51.25 | -0.6677 |
| 1497236186481686016 | 11.881 | 134 | 0 | 82.92 | -0.5888 |
| 1498699086702005376 | 12.107 | 134 | 0 | 87.01 | -0.6867 |
| 1498298211635183744 | 12.116 | 134 | 0 | 65.15 | -0.5415 |
| 1498425548825498112 | 12.537 | 134 | 0 | 78.71 | -0.7015 |
| 1498783199341798016 | 12.559 | 134 | 0 | 67.73 | -0.6827 |
| 1500424804562041984 | 12.565 | 134 | 0 | 49.75 | -0.5964 |
| 1497603835681942400 | 12.700 | 134 | 0 | 94.78 | -0.6326 |
| 1497132660589966976 | 12.701 | 134 | 0 | 57.13 | -0.6403 |
| 1498486880958321024 | 12.748 | 134 | 0 | 83.61 | -0.6459 |
| 1498804639818507904 | 12.900 | 134 | 0 | 66.53 | -0.5569 |
| 1498000793739050368 | 12.998 | 133 | 1 | 118.87 | -0.6129 |
| 1499084499887740160 | 13.004 | 134 | 0 | 65.08 | -0.6140 |
| 1497561779362267392 | 13.020 | 134 | 0 | 66.51 | -0.5223 |
| 1496278752372040832 | 13.155 | 134 | 0 | 65.88 | -0.6203 |
| 1498027456896444928 | 13.175 | 134 | 0 | 73.14 | -0.5476 |
| 1497350638770267520 | 13.294 | 125 | 9 | 1317.84 | -1.5120 |
| 1496293286541396480 | 13.415 | 134 | 0 | 91.63 | -0.5467 |
| 1498058827337611392 | 13.455 | 134 | 0 | 120.18 | -0.5287 |
| 1497639123133258752 | 13.534 | 134 | 0 | 104.06 | -0.5848 |
| 1497227287309482624 | 13.561 | 133 | 1 | 85.83 | -0.5491 |
| 1485560025830226432 | 13.607 | 121 | 13 | 118.00 | -0.5634 |
| 1498617482323461376 | 13.630 | 133 | 1 | 93.32 | -0.5291 |
| 1500693841313325696 | 13.690 | 134 | 0 | 87.26 | -0.5938 |
| 1500461157165243648 | 13.735 | 134 | 0 | 42.42 | -0.5496 |
| 1499021174889970816 | 13.819 | 134 | 0 | 94.98 | -0.6124 |
| 1497683996951418880 | 13.891 | 48 | 86 | 894.28 | -0.4559 |
| 1499006984318088320 | 13.924 | 134 | 0 | 120.46 | -0.6474 |
| 1496037650087948160 | 13.980 | 134 | 0 | 132.75 | -0.5912 |
| 1500327978819506944 | 13.981 | 134 | 0 | 73.68 | -0.5825 |
| 1497491273179203456 | 13.988 | 134 | 0 | 120.94 | -0.5290 |
| 1499210016011946496 | 14.078 | 126 | 8 | 119.70 | -0.5373 |
| 1497169940906156032 | 14.195 | 70 | 64 | 61.42 | -0.5439 |
| 1498321301379345408 | 14.274 | 124 | 10 | 391.28 | -0.6279 |
| 1499209638054824320 | 14.309 | 3 | 131 | 170.65 | -0.7778 |
| 1497871669842349184 | 14.322 | 132 | 2 | 1246.54 | -0.5232 |
| 1485987254816323328 | 14.420 | 125 | 9 | 156.94 | -0.5297 |
| 1498752516095473664 | 14.424 | 11 | 123 | 87.67 | -0.6768 |
| 1502012464992313088 | 14.564 | 121 | 13 | 134.56 | -0.5654 |
| 1497154753901690624 | 14.580 | 51 | 83 | 7573.55 | -0.9077 |
| 1497284015237511808 | 14.609 | 39 | 95 | 190.70 | -0.6640 |
| 1499081819828174080 | 14.719 | 53 | 81 | 1737.36 | -0.6155 |
| 1485534187306501376 | 14.816 | 64 | 70 | 1931.22 | -0.6219 |
| 1497096960821764224 | 14.909 | 28 | 106 | 89.61 | -0.6253 |
| 1485987151737107200 | 14.966 | 62 | 72 | 986.30 | -0.6511 |
| 1500418894687086208 | 14.975 | 41 | 93 | 1772.83 | -0.5641 |
| 1498842882207281152 |  | 0 | 134 |  |  |
| 1499842372636900992 |  | 0 | 134 |  |  |
| 1500410236033012352 |  | 0 | 134 |  |  |

BO CVn (target) M1 RMS = 49.51 mmag. Clean-star M1 floor is ~32 mmag
(G~7.7-9.2). Faint-end RMS is a photon-noise / pkfit-fail floor (no
M1 bound). Four LC IDs have n_ok=0 (absent from snapshot proc, or
aper_nan on every frame). Median d ~ -0.55 to -0.65 mag is a
constant scale offset between model families; M1/M2 remove it.

### M2 (product; AIJ-class readings)

Ensemble from target PSF LC sidecar:
`1497771992240531712,1499200223486564608,1497974027502858240,1497368849430107904`
source=`pinned`. Check `1497613731286514432` has no dedicated PSF LC;
VYVAR check fluxes from live proc `psf_flux` (read-only); same
ensemble as the target sidecar.

| star | RMS(diff) mmag | n_finite | reading |
|------|---------------:|---------:|---------|
| target 1498613634033133184 | 34.34 | 134 | R-X3 |
| check 1497613731286514432 | 31.79 | 134 | R-X3 |

R-X3: RMS(diff) > 10.0 mmag -> STOP, root-cause before any validation claim.

R-X3: RMS(diff) > 10.0 mmag -> STOP, root-cause before any validation claim.

Precedent (aperture vs AIJ) = 1.9503 mmag. These RMS values are
above that scale and above the 10 mmag STOP line.

Target top |resid| epochs (after median offset): Light_132 (94.9
mmag), 102 (-93.0), 138 (77.8), 041 (75.6), 114 (72.9), 023 (72.7),
069 (-71.9), 140 (70.1).

Check top |resid| epochs: Light_086 (76.3 mmag), 041 (69.6), 100
(-68.5), 096 (62.4), 001 (60.1), 021 (-54.7), 107 (54.2), 110 (54.1).

### Position sensitivity (5 frames; diagnostic, no bound)

Frames: Light_001, 037, 076, 109, 148. n_pairs=272.
median |dx|=0.317 px, median |dy|=0.343 px.
median flux delta (recenter - norecenter)=160.3 ADU;
median dmag=-51.2 mmag.

### Failures (never silent)

n=1559. By stage: position `missing_in_proc`=1483 (11 catalog IDs
absent from snapshot proc on 134 frames); pkfit `aper_nan`=76.
Zero getpsf frame failures. CSV:
`dev/results/context/session_20260907_epsfxval/failures.csv`.

### Runtime

175.4 s wall (~1.2 s/frame).

## Gates

G4 live 516 after the harness (read-only):

| product | sha256 prefix | verdict |
|---------|---------------|---------|
| masterstars_full_match.csv | bfa24039778f437b... | PASS |
| MASTERSTAR.fits | 13e77cf8a1dcb4e7... | PASS |
| masterstar_epsf.fits | 172f95403beae36d... | PASS |

Expected prefixes csv `bfa24039` / fits `13e77cf8` / epsf `172f9540`.
Live draft 516 was not written.

`--fast --clean` OVERALL PASS (1626 passed, 32 skipped; clean-tree
PASS). No `--full`. No `--full-epsf`. `src_py` does not import
`dev.xval_pythonphot`.

## Errors (if any)

Vendor needed the PROVENANCE py3 shims before the smoke test would
import or index pixels. Measurement itself completed (exit 0).
LinAlgWarnings from `pkfit_norecenter` inverse on ill-conditioned
stars; those fits are in the failure / high-RMS tail, not dropped.

## Files changed

dev/xval_pythonphot/ (harness + vendor + PROVENANCE + LICENSE)
dev/tests/test_xval_pythonphot_vendor.py
dev/tests/conftest.py (vendor collect ignore)
dev/results/CURSOR_RESULT_EPSF_XVAL_A1.md
dev/results/context/session_20260907_epsfxval/

STOP: leg A2 SExtractor+PSFEx on Linux follows; EPSF-XVAL-01 closure is Milan's
