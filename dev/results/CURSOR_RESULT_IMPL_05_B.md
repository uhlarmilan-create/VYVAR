# CURSOR RESULT - IMPL-05 Item B (per-magnitude scatter aperture)

Date: 2026-08-16
Baseline: 3927afd / stamp 0000dd8 (Item D)
Tip: (pending commit)
Push: NO

## What I did

Corrected Item B (confirmed ADDENDUM 2): one re-photometry pass with persistence,
magnitude-stratified eval set (3-FWHM isolated), per-bin IMPL-04 decision rule,
per-mag `aperture_scatter_table.json`, INV-APERTURE-* gates, remasure of draft
514 proc CSVs. Registered EMPTY-DAO-01 OPEN (B3).

## B1 - Persistence

**Format: parquet** (`aperture_flux_ladder.parquet` beside
`aperture_scatter_table.json` on the draft; copy under `dev/results/`).

Why parquet not JSON: 75 stars x 22 radii x 134 frames = 221100 long-form rows
(~2.0 MB compressed). Full per-frame fluxes are required to re-decide; JSON
would be a multi-MB text dump.

Columns: `catalog_id`, `G`, `r_px`, `frame_index`, `flux`, `n_frames`.

## B2 - Stratification

Comp mag bins from post-COMP-ASSIGN-02 `comparison_stars_per_target.csv`
(unique comps). Eval stars: `snr_cog_isolation_fwhm=3` x FWHM (~15.6 px).
Target ~10 per bin; shortfall only `[7,8)` (5 isolated field stars available).

## B3 - EMPTY-DAO-01

Registered OPEN in `docs/VYVAR_DECISIONS.md` and
`docs/VYVAR_AUDIT_2026_REGISTER.md`. No investigation.

## Per-mag decision (IMPL-04 rule)

| bin | n_eval | chosen r [px] | branch |
|---|---:|---:|---|
| 7-8 | 5 | 6.5 | flat_upper_edge |
| 8-9 | 10 | 5.0 | flat_upper_edge |
| 9-10 | 10 | 5.5 | flat_upper_edge |
| 10-11 | 10 | 5.5 | flat_upper_edge |
| 11-12 | 10 | 5.0 | flat_upper_edge |
| 12-13 | 10 | 3.0 | flat_upper_edge |
| 13-14 | 10 | 2.0 | genuine_minimum |
| 14-15 | 10 | 3.5 -> clipped 2.0 | flat then monotone |

Filled 0.5-mag table is non-increasing (span 4.5 px). Physics expectation
(bright large / faint ~3-5): agrees at G8/9.7/10.8/11.5 check points.

## INV-APERTURE-*

`impl02_gates.ok=True`. INV-APERTURE-MAG-MONOTONE pass (non-trivial, not flat).
INV-APERTURE-BOUND pass. CoG FLATNESS informational only (prior EE curve).

## Remasure + before/after LOO scatter (mmag)

Proc CSVs remasured at per-mag radii (backup under
`_backup_proc_csv_before_impl05_per_mag`). LOO robust scatter on selected comps
(`IMPL_05_B_comp_scatter_before_after.json`):

| G bin | before r=9.5 | after per-mag |
|---|---:|---:|
| 9-10 | 10.2 | 8.3 |
| 10-11 | 17.8 | **11.9** |
| 11-12 | 34.4 | 22.3 |
| 12-13 | 70.5 | 40.3 |
| 13-14 | 83.8 | 36.2 |

G~10.8 class moves toward the ~14 mmag physics floor (11.9). Faint 14-15 LOO
worsened in this thin sample (blend/variable contamination risk; Item C
single-source addresses candidacy).

## Artifacts

- `dev/tools/impl_05_per_mag_aperture_scan.py`
- `dev/results/IMPL_05_scatter_scan.json`
- `dev/results/IMPL_05_aperture_scatter_table.json`
- `dev/results/IMPL_05_aperture_flux_ladder.parquet`
- `dev/results/IMPL_05_B_comp_scatter_before_after.json`
- draft (gitignored Archive): `aperture_scatter_table.json`,
  `aperture_snr_table.json`, `aperture_flux_ladder.parquet`

## Files changed (this commit)

- `docs/VYVAR_DECISIONS.md` (EMPTY-DAO-01 OPEN)
- `docs/VYVAR_AUDIT_2026_REGISTER.md`
- `dev/tools/impl_05_per_mag_aperture_scan.py`
- `dev/results/IMPL_05_*` + this result
