CURSOR RESULT - 2026-08-22 (EPSF-VALID-02 ACCEPTANCE RERUN)

What I did
Re-exported per-frame PSF columns on draft 516 using frozen `masterstar_epsf.fits` (1475-star
production model) and F1-F3 code. Pre-export proc CSVs backed up for aperture guard.

## Output / findings

### Expectation A -- full coverage
| Metric | Before (P1 on-disk) | After rerun |
|--------|---------------------|-------------|
| CSS_J140918.7+423422 (`1498486880958321024`) psf_ok frames | 35 / 134 | **135 / 135** |
| Collapse at Light_039 | yes (0 ok from frame 35+) | **no** |

Artifact: `dev/results/context/session_20260822_epsf_valid_02_accept/css_target_coverage.csv`

### Expectation B -- UI (code inspection; not Streamlit screenshot)
- Science-set filter: 333 stars via `build_epsf_science_set`
- ProgressColumn: `format="%.1f%%"`
- Build-meta expander: present in `ui_epsf_dashboard.py`

### Expectation C -- INV-PSF-FRAME-01
| Metric | Value |
|--------|-------|
| frames_total | 135 |
| frames_with_zero_ok | **0** |
| fraction | 0.0 |
| policy | **ok** (silent below 20% threshold) |

Job summary: `Archive/Drafts/draft_000516/platesolve/NoFilter_60_2/epsf_photometry_job_summary.json`
Sandbox copy: `dev/results/context/session_20260822_epsf_valid_02_accept/epsf_job_summary.json`

Per-frame n_fit ~230 (science set) vs ~2400 pre-F3; n_ok ~64-76 per frame.

### Aperture guard
`dao_flux` / `mag` / `mag_inst` / `flux` column hashes: **0 mismatches** across 134 proc files
(`aperture_hash_mismatches=0`). PSF columns are the only delta.

Backup: `dev/results/context/session_20260822_epsf_valid_02_accept/proc_backup_pre_accept/`

## Docs impact

None.

## Gate status

Acceptance rerun PASS. Post-series `--full` recut pending (not run in this file).

## Errors

None. Rerun runtime ~84 min (135 frames, sequential export).

## Files changed

- Sandbox: `dev/sandbox/epsf_valid_02_accept_rerun.py`
- Artifacts: `dev/results/context/session_20260822_epsf_valid_02_accept/*`
- On-disk proc CSVs under draft 516 (PSF columns refreshed; aperture byte-identical)

**STOP** -- architect review before F4 production model swap (F4 build code already committed separately).
