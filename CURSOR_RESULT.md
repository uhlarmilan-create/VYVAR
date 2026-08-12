CURSOR RESULT - 2026-08-12

What I did
Repo-wide zero-clipping: removed remaining science-path sigma-clip / kappa-sigma /
outlier-rejection and dead clip/CR config+UI. Cosmic cleaning already removed at
0ab686f. Added plain_mean_med_std; master dark/flat stay plain mean/median.

## Output / findings
- params_registry: 272 total; config_runtime 244
- Full site inventory: dev/results/CURSOR_RESULT_remove_all_clip_cr.md
- Fresh BO CVn re-run still required (505/506 calibrated already CR-damaged)
- comp_rms is now unclipped MAD (numbers will change)

## Errors (if any)
None blocking after test updates. Borderline hard gates listed for Milan in the
detail result file (slope filter, max_comp_rms, p2p 0.10 ceiling, align-residual gate).

## Files changed
See dev/results/CURSOR_RESULT_remove_all_clip_cr.md
