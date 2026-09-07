CURSOR RESULT - 2026-09-07 EPSF-XVAL-A1B-ROOTCAUSE-01

Date: 2026-09-07. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: b2a3ce6.
Class: MEASUREMENT ONLY. No production code change. Harness edits
in `dev/xval_pythonphot/` only. Live 516/517 read-only.
EPSF-XVAL-01 does not close here.

## Architect error 16 (carry-in, on the record)

A1 designed `pkfit_norecenter` at snapshot proc x,y as the primary
mode without first reading what those x,y contain (projection vs
per-frame refit; index convention), and against native
DAOPHOT-lineage practice (Stetson's pkfit fits position; ALLSTAR
always does). The A1 recenter sensitivity (median |dx|=0.32 px,
|dy|=0.34 px, median dmag=-51 mmag) exposed it. Same class as
errors 9-15: design claim without reading the governing artifact.

## Step 1 -- governing code (BEFORE any rerun number)

Read and cited first. Rerun data below is evidence, not the
authority for what the positions are.

(a) What writes the snapshot proc x,y consumed by the A1 harness.

`detect_stars_match_master_reference` (`src_py/pipeline_catalog.py:2448`)
runs DAO FIND (`x_centroid` / `y_centroid` at `:2692-2694`). On
aligned frames (`VY_ALGN`; flag test `:2748`, helper `:150`) it then
calls `_lock_matched_centroids_to_master_grid` (`:2181-2247`, call
site `:2879-2888`). That snap-to-MASTERSTAR then replaces the
coordinate with the integer brightest pixel:

`xo[i] = float(x_lo + int(px))` at `:2244-2245`.

Docstring `:2197`: "sub-pixel centre not required".

Empirical (sandbox snapshot `proc_BO_CVn_Light_00*.csv`, target
`1498613634033133184`): x,y are integers that jitter by 1 px
across frames -- (958, 822), (958, 823), (958, 823). Not a stable
subpixel photocenter. Not a WCS-projected catalog position either:
the lock starts from the master grid, then jumps to the discrete
peak.

(b) What positions the VYVAR PSF path itself fits at.

`psf_photometry.psf_photometry_stars`: `psf_fix_position_enabled`
is not in `config.py`; getattr default is False (`:2868`).
Photutils refits `x_0`, `y_0`. Flux is at the fitted centroid.
Written proc columns `x`, `y` stay the init (`:3230-3231`);
fitted positions go to `x_fit`, `y_fit` (`:3245-3246`).
VYVAR PSF recenters. A1's primary `pkfit_norecenter` at integer
peak pixels does not.

(c) Vendor pkfit / daoerf pixel-center vs numpy.

`dev/xval_pythonphot/vendor/getpsf.py:56-57`: IDL first pixel is
(0, 0). `daoerf.py:58-59` integrates `[x-0.5, x+0.5]`, so integer
N is that pixel's centre. Same 0-based pixel-center convention as
numpy / photutils. A 0-vs-1 or axis-swap mismatch is not predicted
by the governing code; a ~0.3 px residual from integer-peak vs
photocenter (H-B) is.

## What I did

Harness `dev/xval_pythonphot/run_xval_a1b.py` (reuses A1 helpers;
optional `psf_model=` skip-getpsf on the H-A grid). Sandbox reuse
of the era04 copy at `tmp/session_20260907_epsfxval/sandbox/`
(`--skip-copy`). Same stars, same PSF pool (G-EPSF census, n=67),
same gain/RN, same ensemble as A1. Live Archive never written.

Gain=0.637067 e-/ADU source=`g_pt`; RN=15.2 e- source=`db`.
fitrad=3.3014 px; psfrad=8.0 px. Runtime 196.1 s.

## Output / findings

### H-A convention grid (5 frames x 27 variants)

Frames: Light_001, 037, 076, 109, 148. 272 pairs/variant.
`pkfit_norecenter` at x,y+offset; recenter run at that start.
Shift = |r_fit - start|. |dmag| vs A1 baseline = norecenter at
the offset vs norecenter at the original proc x,y.

Unfiltered medians made `swap_xy` look like a zero-shift
"winner". That is a fail artifact: 165/272 `fail_rec`, and the
failed fits leave `x_fit==start` so shift=0, while
median |dmag|=4.27 mag. Table below is **successful recents
only** (`fail_rec<0.5`). Full CSV:
`dev/results/context/session_20260907_epsfxval_a1b/ha_summary.csv`.

| variant | n_ok | n_fail | med shift px | med |dx| | med |dy| | med |dmag| |
|---|---:|---:|---:|---:|---:|---:|
| dx+0.0_dy+0.5 | 269 | 3 | 0.485 | 0.319 | 0.276 | 0.049 |
| dx+0.0_dy+0.0 | 269 | 3 | 0.555 | 0.319 | 0.355 | 0.000 |
| dx+0.5_dy+0.5 | 269 | 3 | 0.649 | 0.497 | 0.274 | 0.092 |
| dx-0.5_dy+0.5 | 269 | 3 | 0.651 | 0.518 | 0.277 | 0.080 |
| dx-0.5_dy+0.0 | 269 | 3 | 0.733 | 0.522 | 0.355 | 0.090 |
| dx+0.5_dy+0.0 | 269 | 3 | 0.758 | 0.498 | 0.355 | 0.069 |
| dx+0.0_dy+1.0 | 269 | 3 | 0.798 | 0.319 | 0.669 | 0.113 |
| dx-0.5_dy+1.0 | 269 | 3 | 0.933 | 0.516 | 0.669 | 0.141 |
| dx+0.5_dy+1.0 | 269 | 3 | 0.937 | 0.498 | 0.668 | 0.181 |
| dx+0.0_dy-0.5 | 269 | 3 | 0.953 | 0.319 | 0.834 | 0.123 |
| dx+1.0_dy+0.5 | 269 | 3 | 1.069 | 0.987 | 0.275 | 0.233 |
| dx+0.5_dy-0.5 | 269 | 3 | 1.072 | 0.502 | 0.831 | 0.173 |
| dx+1.0_dy+0.0 | 269 | 3 | 1.085 | 0.987 | 0.355 | 0.215 |
| dx-1.0_dy+0.5 | 269 | 3 | 1.094 | 1.019 | 0.276 | 0.267 |
| dx-0.5_dy-0.5 | 269 | 3 | 1.096 | 0.522 | 0.834 | 0.205 |
| dx-1.0_dy+0.0 | 269 | 3 | 1.137 | 1.019 | 0.351 | 0.309 |
| dx+1.0_dy+1.0 | 269 | 3 | 1.280 | 0.987 | 0.668 | 0.387 |
| dx-1.0_dy+1.0 | 269 | 3 | 1.282 | 1.019 | 0.678 | 0.360 |
| dx-1.0_dy-0.5 | 269 | 3 | 1.380 | 1.019 | 0.832 | 0.466 |
| dx+1.0_dy-0.5 | 269 | 3 | 1.382 | 0.987 | 0.832 | 0.376 |
| dx+0.0_dy-1.0 | 270 | 2 | 1.404 | 0.321 | 1.335 | 0.402 |
| dx-0.5_dy-1.0 | 270 | 2 | 1.514 | 0.523 | 1.333 | 0.511 |
| dx+0.5_dy-1.0 | 270 | 2 | 1.514 | 0.506 | 1.335 | 0.425 |
| swap_xy | 107 | 165 | 1.746 | 0.931 | 1.163 | 4.267 |
| dx-1.0_dy-1.0 | 271 | 1 | 1.751 | 1.013 | 1.331 | 0.723 |
| dx+1.0_dy-1.0 | 270 | 2 | 1.758 | 0.993 | 1.335 | 0.637 |
| swap_xy_p0.5 | 106 | 166 | 1.786 | 1.064 | 0.826 | 4.021 |

No variant zeroes the recenter shift. Best non-swap
(`dy=+0.5`) only trims 0.07 px vs identity; `+/-1.0` adds ~1 px
as a 1-based guess would; axis-swap fails and (on survivors)
shifts 1.75 px at 4 mag. **H-A is refuted.** Residual ~0.3-0.55 px
is H-B (integer peak vs photocenter), not an index/axis convention
error.

### Full native recenter (134 frames) -- M1 / M2

Same ensemble as A1 (pinned):
`1497771992240531712,1499200223486564608,1497974027502858240,1497368849430107904`.
Same memberships and weights. M2 = RMS after median of
`delta_VYVAR - delta_PP` (local flux-sum, not
`photometry_lightcurve`).

M1 (instrumental `m_VYVAR - m_PP`; per-star median removed).
CSV: `dev/results/context/session_20260907_epsfxval_a1b/m1_recenter.csv`.

| catalog_id | G | n_ok | RMS mmag A1 | RMS mmag A1B | median d A1B |
|---|---:|---:|---:|---:|---:|
| 1500549977088828160 | 7.745 | 134 | 32.16 | 45.34 | -0.487 |
| 1498278351706325248 | 8.605 | 134 | 50.25 | 78.99 | -0.704 |
| 1497343732462852864 | 9.184 | 134 | 31.59 | 39.79 | -0.559 |
| 1498613634033133184 | 9.720 | 134 | 49.51 | 56.29 | -0.646 |

Clean-star M1 floor is not improved by native recenter (32 -> 40-45
mmag on the G 7.7-9.2 pair that set the A1 floor). Per-star median
d remains -0.38 .. -0.92 mag (not a single scale constant).

M2 (product; pre-registered readings):

| star | A1 norecenter mmag | A1B recenter mmag | n_finite |
|------|-------------------:|------------------:|---------:|
| target 1498613634033133184 | 34.34 | 30.84 | 134 |
| check 1497613731286514432 | 31.79 | 38.01 | 134 |

Target improved 3.5 mmag; check worsened 6.2 mmag. Both remain
above 10 mmag.

### H-B -- median fit minus proc x,y (per frame)

134 frames. Median of per-frame medians:

- signed dx = -0.054 px (range -0.37 .. +0.32)
- signed dy = +0.332 px (range -0.61 .. +0.50)
- |dx| = 0.305 px, |dy| = 0.347 px, shift = 0.533 px

Matches the A1 5-frame sensitivity (|dx|=0.317, |dy|=0.343).
Quantifies H-B: proc x,y sit on the integer peak; pkfit's
converged photocenter is systematically ~1/3 px away, with a
+y bias. CSV:
`dev/results/context/session_20260907_epsfxval_a1b/hb_per_frame.csv`.

### H-C -- per-star median offset vs field position

n=56 stars with a finite median d; n=61 with a finite shift.
Map (csv + png):
`dev/results/context/session_20260907_epsfxval_a1b/hc_per_star.csv`
`dev/results/context/session_20260907_epsfxval_a1b/hc_residual_map.png`

| pair | n | Pearson r (p) | Spearman r (p) |
|------|--:|---------------|----------------|
| r_field vs median_d_mag | 56 | -0.189 (0.162) | -0.080 (0.558) |
| r_field vs median_shift | 61 | -0.281 (0.028) | -0.227 (0.079) |
| x vs median_d_mag | 56 | -0.009 (0.948) | +0.067 (0.622) |
| y vs median_d_mag | 56 | -0.069 (0.614) | -0.016 (0.905) |

Median d vs field position is not significant. Shift vs radius
is a weak Pearson signal only. The 2D map is a mixed -0.4..-0.9
mag sheet, not a radial tree-ring. H-C (single getpsf vs gridded
ePSF as a *spatial* mag-residual pattern) is not demonstrated in
this median-d map. The scale offset itself remains, so a
spatially varying PSF can still be the *model-family* cause
without imprinting a clean radius correlation on the median.

### H-D -- per-epoch M2 residual vs frame FWHM

qc_metrics `fwhm_px` (DAO-scale; matched on `Light_NNN`).
n=134/134 finite. FWHM span is tiny: 5.138 .. 5.305 px
(median 5.192, std 0.029).

| star | n | Pearson r (p) | Spearman r (p) |
|------|--:|---------------|----------------|
| target | 134 | -0.050 (0.562) | -0.029 (0.735) |
| check | 134 | -0.444 (7.7e-8) | -0.461 (2.2e-8) |

Target residuals do not track seeing. Check does, but over a
0.17 px FWHM window that cannot budget 30+ mmag of M2.
H-D (getpsf LUT noise correlated with seeing) is not the
target's M2 cause. CSV:
`dev/results/context/session_20260907_epsfxval_a1b/hd_epochs.csv`.

### Failures (never silent)

n=1559. By stage: position `missing_in_proc`=1483 (IDs absent
from snapshot proc); pkfit `aper_nan`=76. Zero getpsf frame
failures. Same accounting as A1. CSV:
`dev/results/context/session_20260907_epsfxval_a1b/failures.csv`.

### Applied reading (verbatim)

R-A1B-3: recentered M2 > 10 mmag -> position was not the (main)
cause; H-C/H-D tables become the primary evidence; no validation
claim; A2 leg becomes the decisive external reference.

Applied on worst(target, check) = 38.01 mmag. Both sides
individually also sit in R-A1B-3.

## Gates

G4 live 516 after the harness (read-only):

| product | sha256 prefix | verdict |
|---------|---------------|---------|
| masterstars_full_match.csv | bfa24039778f437b... | PASS |
| MASTERSTAR.fits | 13e77cf8a1dcb4e7... | PASS |
| masterstar_epsf.fits | 172f95403beae36d... | PASS |

Expected prefixes csv `bfa24039` / fits `13e77cf8` / epsf `172f9540`.
G4 PASS. Live draft 516 was not written.

`--fast --clean` OVERALL PASS (1626 passed, 32 skipped; clean-tree
PASS). No `--full`. No `--full-epsf`. `src_py` does not import
`dev.xval_pythonphot`.

## Errors (if any)

H-A first-pass summary ranked `swap_xy` by unfiltered median
shift (fail-stays-put). Table and `ha_best` recomputed on
`fail_rec<0.5` before this report. LinAlgWarnings from vendor
`pkfit` inverse on ill-conditioned stars; those fits sit in the
failure / high-RMS tail, not dropped.

The H-C png was written from `hc_per_star.csv` after the
measurement loop (same scatter the harness intended); correlations
in `summary.json` are from the loop itself.

## Files changed

dev/xval_pythonphot/run_xval_a1b.py
dev/xval_pythonphot/run_xval_a1.py (optional `psf_model=` reuse)
dev/results/CURSOR_RESULT_EPSF_XVAL_A1B.md
dev/results/context/session_20260907_epsfxval_a1b/

STOP: readings fix wording only. STOP for Milan in every case.
EPSF-XVAL-01 closure remains his, pending leg A2
(SExtractor+PSFEx on Linux).
