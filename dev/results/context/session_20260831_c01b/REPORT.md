# CURSOR RESULT - CONSOLIDATE-01B single geometry, single edge rule, SNR-table removal

Date: 2026-08-31. Branch: consolidate-01. English. ASCII.
Base: origin/consolidate-01 @ 22391ca. Live draft 516 was not written.
Push: origin/consolidate-01 only (after G2 / G-EPSF).

## What I did

One geometry resolver, one on-chip predicate, named sky estimators, and
deletion of the unused SNR-optimal aperture table plus scatter/CoG gate
modules. A2 MEASURE FIRST forbade swapping bbox FWHM from MASTERSTAR
VY_FWHM to qc_metrics (all 134 ok frames differ at FITS-card rounding).

## A2 MEASURE FIRST -- STOP (do not swap)

Script: `a2_measure.py`. Table: `a2_r_out_table.csv`. Summary: `a2_summary.json`.

Config: f=1.35, inner=2.7, outer=5.2.

| Quantity | Value |
| --- | --- |
| n qc rows / n status=ok / n measured | 150 / 134 / 134 |
| n last-ulp identical (qc vs header VY_FWHM) | 0 / 134 |
| max abs delta FWHM (px) | 2.511e-4 (Light_028) |
| max abs delta r_out (px) | 2.511e-4 |
| example Light_001 | qc 5.30455572 vs header 5.3046 |
| MASTERSTAR VY_FWHM | 5.19465 |
| night-median qc (ok) | 5.191733 |
| bbox r_out resolver(MS header) | 27.01218 |
| bbox r_out resolver(night med) | 26.99701 |
| MS vs night-med r_out equal | false |

Cause: FITS card rounding (4-5 decimals), not a science FWHM change.
Photometry already prefers qc via `resolve_frame_fwhm_px`. Bbox FWHM
authority stays MASTERSTAR header VY_FWHM. FWHM-AUTH-01 closure for
aperture geometry is STOPPED pending a ledger decision.

## A1 converted call sites (d320697 inventory -> tip)

| Site (01A frozen line) | Tip | Disposition |
| --- | --- | --- |
| pipeline bbox ~6725 | pipeline.py:6595 | `resolve_aperture_geometry`; MASTERSTAR VY_FWHM; magic 3.5 and 10.5 deleted; missing FWHM/factors raise |
| forced seed masterstar_gaia_accounting.py:478 | :488 | named `SEED_ANNULUS_INNER_PAD_PX=4` / `SEED_ANNULUS_OUTER_PAD_PX=8`; not production resolver (byte-identity of era04 seeds; comment states why) |
| psf `_aperture_annulus_radii_px` | psf_photometry.py:2047 | delegates to resolver via AppConfig factors |
| psf `_psf_annulus_radii_px` | psf_photometry.py:2073 | same |
| xval_run.py:181 | xval_run.py:145 | resolver; sky via `sky_median_mask` |

Resolver: `aperture_policy.resolve_aperture_geometry` at aperture_policy.py:75.
Invalid FWHM or annulus factors raise; never invent.

## A3 sky statistics

New `src_py/sky_estimation.py`:

| Variant | Callers |
| --- | --- |
| `sky_median_mask` | `_sky_pp_from_annulus_image` (photometry_core.py:12383); PSF `_annulus_median_per_px` (values=, min_pix=8); xval harness |
| `sky_exact_mean` | `_estimate_annulus_sky_pp` (photometry_core.py:2021) |
| `sky_clipped_mean_med_std` | forced seed (masterstar_gaia_accounting.py:499) |

Pass-2 hypot annulus 8-12 px is not CircularAnnulus; left alone.
G2 byte-identity is the numeric proof.

## B1 edge predicate (D-1)

`aperture_policy.star_fits_on_chip` / `stars_fit_on_chip` (aperture_policy.py:114 / :137).
4-tuple naxis = precomputed safe bbox already shrunk by r_out; dummy
geometry (0,0,0). 2-tuple naxis = chip NAXIS inset by geometry r_out.

| Consumer | Tip |
| --- | --- |
| Comp-pool candidates | pipeline.py `select_comparison_stars_spatial_grid` :6128 |
| Active targets | photometry_core.py `select_active_targets` :13601 |
| Global comp pool | photometry_core.py `build_global_comp_pool` :14528 |

When `safe_bbox` is None, the existing 50 px PHASE0 fallback is kept
(not merged into EDGE).

## B2 ledger + ROADMAP

EDGE-ANNULUS-01 -> CLOSED-DECIDED in `docs/VYVAR_DECISIONS.md`:
"Edge stars are not used (Milan 2026-08-31). Aperture and annulus must
lie fully on-chip; no partial-annulus mode. Consequence accepted: FR CVn
and the 4 EDGE ids stay outside the 516 product."

ROADMAP NEXT SESSION step 1 closed with that wording.

PHASE0-BORDER-MARGIN-GEOMETRY: NOT merged. Phase 0 uses
`phase01_chip_interior_margin_px` = 50 px. EDGE r_out is ~27 px at
5.2 x FWHM. Different margin. ROADMAP row updated with that note.

## C1 reader grep (D-2)

Grep `aperture_snr_table` (filename and json key) in src_py + dev/tests +
src_py/ui_*.py:

| Location | Role |
| --- | --- |
| src_py/ui_*.py | no matches |
| src_py production | writer/load/precompute deleted this task |
| APERTURE-01 skip at photometry_core enhance_catalog | branch deleted; scalar APERTURE-01 only |
| GS11 | map-only; SNR table fallback removed |
| src_py/validate_lc_crossval.py:81 | comment `FWHM_PX = 2.3976  # from aperture_snr_table.json` -- not a file reader |
| docs/ | historical mentions only (DECISIONS/STATE/JOURNAL/audit) |
| dev/tests | all SNR-table tests deleted |

No UI page or computation reads the table. Delete proceeded.

## C2 deletions

Deleted modules: `src_py/aperture_scatter_select.py`, `src_py/snr_cog_gates.py`.
Deleted tests: test_impl_01_snr_cog, test_impl_02_snr_cog_gates,
test_impl_03_scatter_aperture, test_snr_table_dao_fwhm_authority,
test_snr_table_rn_header. Kept test_t2_aperture_snr.py and
test_snr_gate_01_sky_mad.py.

Config keys removed (registry + config.json + AppConfig, KNOWN_REMOVED_KEYS):
`aperture_selection_criterion`, `aperture_scatter_r_min_px`,
`aperture_scatter_r_max_px`, `aperture_scatter_r_step_px`.
config_runtime 283 -> 279. Kept `aperture_snr_sizing` and
`snr_cog_isolation_fwhm`. Regenerated `docs/VYVAR_PARAMS.md`.

ePSF import-graph touch (same G-EPSF budget): deleted
`_epsf_allowed_catalog_ids`, `_epsf_positions_from_csvs`,
`_psf_fit_region_mask` from psf_photometry.py; deleted psf_runner
`step_1`..`step_5` dead lines. Radii helpers now import aperture_policy.

## C3 product accounting

`ext_aperture` = `photometry_sha_files(..., include_comp_qa=True)` minus
PSF LCs. Patterns: lightcurve_*.csv, comp_quality_*.json,
comparison_stars_per_target.csv, comp_qa_*.json.
`aperture_snr_table.json` is outside the glob.

Expected: era04_aperture core d55fcc9d n=53 byte-identical; ext
cc8b532e n=157 unchanged (table never in the glob). n=157 -> 156
does not apply. G2 confirmed: core d55fcc9d8ad9b552 n=53, ext
cc8b532ee668b9b3 n=157. Deletion was clean.

## Gates

| Gate | Result |
| --- | --- |
| G1 before | PASS 1628 passed / 32 skipped; clean-tree; HEAD 22391ca (`g1_before.txt`) |
| G1 after (dirty tree, --fast) | PASS 1603 passed / 32 skipped (`g1_after.txt`). Count drop = deleted SNR/scatter tests. `--clean` after commit (worktree is HEAD). |
| G2 --full ePSF OFF | PASS (`g2_full.txt`). core era04_aperture d55fcc9d8ad9b552 n=53 byte-identical; ext cc8b532ee668b9b3 n=157 unchanged (table outside glob). full-science-compare n_lc=53 failures=0. Pipeline 1312 s. |
| G-EPSF --full-epsf | PASS (`g_epsf.txt`). epsf01 c743b8ba89f4ac54 n=53. G3 residual BO dem=12.505 / FW dem=4.629, n_full=134 both. ePSF stage 14802 s, 53 PSF LCs. Aperture hashes unchanged. |
| G4 live 516 | PASS after G2 and after G-EPSF: csv bfa24039778f437b / fits 13e77cf8a1dcb4e7 / epsf 172f95403beae36d. |

## STOPs

1. A2 FWHM swap: 134/134 frames differ. Bbox stays MASTERSTAR VY_FWHM.
   FWHM-AUTH-01 for aperture geometry not closed by this task.
2. D-2 was conditional: C1 found no computational/UI reader; delete ran.
3. Spec vs code: forced seed +4/+8 is not the production resolver
   (named pads, documented). PHASE0 50 px is not EDGE r_out (not merged).

## Files changed (pre-commit)

src_py: aperture_policy.py, sky_estimation.py (new), pipeline.py,
photometry_core.py, psf_photometry.py, psf_runner.py, xval_run.py,
masterstar_gaia_accounting.py, config.py; deleted
aperture_scatter_select.py, snr_cog_gates.py.

config.json, dev/validation/params_registry.json, docs/VYVAR_PARAMS.md,
docs/VYVAR_DECISIONS.md, docs/VYVAR_ROADMAP.md.

dev/tests: policy/bbox/GS11/err-bkg/osc2/photometry_core/params dashboard
edits; five SNR/scatter test files deleted.

Session: this directory (measure table, G1 logs, this REPORT).
