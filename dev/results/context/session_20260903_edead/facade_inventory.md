# Post-E-DEAD facade inventory (feeds E-final)

Measured at product SHA `8006c88` after C6.

## pipeline.py (1089 lines)

Physical defs remaining (AST top-level):

| line | kind | name | note |
| --- | --- | --- | --- |
| 209 | FunctionDef | `_analyze_calibrated_qc_one` | STAY; AstroPipeline-adjacent |
| 240 | FunctionDef | `analyze_calibrated_qc` | STAY; REAL caller AstroPipeline |
| 302 | ClassDef | `AstroPipeline` | C-C; out of E-DEAD scope |

Plus constants still physical here: `SAT_LIMIT_*`, `_MASTERSTAR_*`,
`_EXO_HOST_ANNOTATION_COLUMNS`, `_PLATESOLVE_ANISOTROPY_THRESHOLD`, ...
Everything else is a facade re-export (calibrate, catalog, giants,
ui_helpers, gate_helpers, epsf_hooks).

E-final questions: glue dismantling, facade permanence (Milan),
twin-dismantle of SAT_LIMIT to a leaf constants module, test retargets.
The two QC helpers can move with AstroPipeline or stay beside it.

## photometry_core.py (1273 lines)

Physical defs remaining:

| line | kind | name | note |
| --- | --- | --- | --- |
| 628 | FunctionDef | `compute_auto_fwhm_limit` | not in the 27 |
| 721 | FunctionDef | `run_full_photometry_pipeline` | C-D production entry |
| 1262 | FunctionDef | `select_active_targets` | E3 wrap/follow |

`from photometry import *` still star-imports the facade.

E-DEAD names now live in:

- photometry_shared: cog, batch, bbox pair
- photometry_gate_helpers: mask, canon, mad, clamp, labbe trio
- photometry_exports: `_get_lc_star_method`

Header `from photometry_core import` lines in shared/gate_helpers/exports
are remaining E4 glue. `_clamp_err_empty_apertures_min` is also an
E4-style inject onto photometry_shared (None + bind after gate_helpers
load), same pattern as `_clamp_err_empty_apertures_n`.
