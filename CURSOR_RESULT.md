CURSOR RESULT — 2026-06-15 (draft_407 V0612 close — ship + real product)

What I did
Committed Workstreams A + B (simple differential + reporting-column fix). Ran production
`run_full_photometry_pipeline` on draft_407 `g_60_4`, regenerated PDF + LC export, measured vs AIJ.

## Output / findings

### Closure metrics (V0612 Cam, 1111749368289526912, 24 normal frames)

| column | pre-eclipse RMS | shape corr vs AIJ | DoD target |
|--------|-----------------|-------------------|------------|
| `delta_mag` | **0.0118** | **0.985** | ~0.0113 / ~0.949 |
| `mag_calib` | **0.0112** | **0.978** | ~0.0109 / ~0.958 |
| `mag_calib_ct` | **0.0112** | **0.978** | CT off (same as mag_calib) |

Outlier flags: **24/24 normal** (mask-first guard; no ingress `outlier_lo`).

### Deliverables (real VYVAR product)

| artifact | path |
|----------|------|
| PDF LC (`mag_calib_ct`) | `Archive/Drafts/draft_000407/platesolve/g_60_4/VYVAR_report_g_60_4_20260615.pdf` |
| LC CSV (full) | `Archive/Drafts/draft_000407/platesolve/g_60_4/photometry/lightcurves/lightcurve_1111749368289526912.csv` |
| LC export subset | `tmp/phase13/V0612_Cam_draft407_g_lc.csv` |
| Close harness JSON | `tmp/phase13/draft407_close.json` |

Phase 1: 7 comps (C3 absent via tier-ladder + `comp_select_rms_floor`); bin OFF; color term off.

**draft_407 g CLOSED** — clean eclipse on real output, RMS matches validated DoD-A/B.

## Errors (if any)
PDF HRD section warning only (`tuple index out of range`); LC + main report OK.

## Files changed
Committed: config.py, config.json, ui_settings.py, comp_selection_per_target.py,
photometry_core.py, method_lc_output.py, param_resolver.py, photometry_report.py,
CITATIONS.bib, citations.py, docs (SIMPLE_DIFFERENTIAL_SPEC, DECISIONS, STATE, ROADMAP, …),
CURSOR_RESULT.md.
Sandbox: `tmp/phase13/draft407_close.py` (not committed).

Not in commit (parked / separate): `docs/VYVAR_SIGMA_BUDGET_SPEC.md`, Brno solver delta
(`vyvar_platesolver.py`, `pipeline.py`, `app.py`, `ui_dao_stars.py`).
