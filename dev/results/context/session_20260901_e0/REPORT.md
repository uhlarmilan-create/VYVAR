# CURSOR RESULT - CONSOLIDATE-01E0 split map (measure, move nothing)

Date: 2026-09-01. Branch: consolidate-01. English. ASCII.
Base: origin/consolidate-01 @ 5b1068d (Milan has not pushed main; origin/main
is still 7c086e8). Not a SEL-GHOST-MERGED tip. Implementer: Cursor.
Architect: Claude. No src_py edits. No G2 (nothing on the photometry graph
changed). No G-EPSF.

Premise: compared pipeline.py and photometry_core.py as they sit at 5b1068d
(20124 lines / 256 functions + 2 classes; 17525 lines / 219 functions + 4
classes) against the night-run stage list in the task. The two files have no
named section structure. The map is the cut; byte-identity polices later moves.

## What I did

Built `dev/tools/e0_split_map.py` (AST, deterministic LPA, seed order fixed).
Wrote symbol_map.json, clusters.json, module_map_proposal.md, risk_register.json,
modules_raw.json. Product code untouched.

## M1 Symbol table

`dev/results/context/session_20260901_e0/symbol_map.json`

| file | lines | top-level functions | classes |
| --- | --- | --- | --- |
| pipeline.py | 20124 | 256 | 2 (SkySurfaceOrderConflictError, AstroPipeline) |
| photometry_core.py | 17525 | 219 | 4 |

No name collisions across the two files. `photometry.py` remains a star-import
facade of photometry_core.

Each symbol: name, line span, size, same-file callees, cross-file callees,
same-file callers, external callers (src_py + tests, including import-from,
attr call, getattr, and `pipeline.foo` patch strings), stages reachable,
primary_stage.

Primary stage is same-file BFS from night-run / importer seeds, not a cross-file
paint (that collapsed phase2a into phase0+1 on the first pass). AstroPipeline
methods matching calibrat* / preprocess* seed their callees; the class itself
is assigned calibration.

Stage line totals (sum of def spans; module-level glue between defs is not
in the sums):

| primary_stage | n | lines |
| --- | --- | --- |
| import | 20 | 661 |
| calibration | 77 | 4160 |
| astrometry-MASTERSTAR | 142 | 13744 |
| phase0+1 comp selection | 30 | 4095 |
| photometry-shared | 32 | 1936 |
| phase2a photometry | 111 | 9130 |
| ePSF hooks | 4 | 248 |
| exports-reports | 5 | 94 |
| UI-only | 7 | 240 |
| gate-only | 26 | 943 |
| unreachable | 27 | 902 |

## M2 Clustering

`clusters.json`. Label-propagation, nodes sorted, neighbor-label weight then
lex tie-break, 50 iterations max. Then merge any cluster with cross-edges >
internal-edges into the neighbor with most connecting edges.

- 50 communities after merge.
- Weak clusters remaining: none.
- 142 undirected cross-cluster edges. Top 20 are all weight 1 (unique seams,
  each a future import line). See clusters.json top20_cross_cluster_edges.
- Largest LPA community: 83 defs / 8213 lines around
  `_astrometry_align_impl_body` + `generate_masterstar_and_catalog` (internal
  121, cross 61). Graph-wise it is a module; line-cap forbids shipping it as
  one file. Stage packing peels the giant defs (below).

LPA is diagnostic. Module cuts follow night-run stages.

## M3 Proposed module map

`module_map_proposal.md` (table + per-module def lists + import seams).

Constraints applied:

- Stage boundaries win. pipeline.py vs photometry_core.py never share a
  proposed file.
- Facades stay and re-export. Facade removal is E-final.
- phase0+1 and phase2a are not merged.

### phase2a vs comp-selection (do not auto-merge)

| bucket | n | lines |
| --- | --- | --- |
| phase0+1 | 30 | 4095 |
| phase2a | 111 | 9130 |
| photometry-shared | 32 | 1936 |

Directed call weights:

- comp -> phase2a: 0
- phase2a -> comp: 0
- comp -> shared: 11 / shared -> comp: 3
- phase2a -> shared: 27 / shared -> phase2a: 4

They do not call each other. They share a helper layer. Architect decides
whether any shared name is actually stage-owned; the map keeps them in
photometry_shared.py.

### Line cap

Cap ~4000. Packing slack 4200 so a coherent 4095-line stage is not split for
95 lines. Defs >= 800 lines are peeled to one-def modules (cannot go under
4000 without splitting the function body).

Only `pipeline_calibrate.py` is over cap as a multi-def module: 76 defs /
4121 lines (YES). Least-bad keep. Optional: leave `AstroPipeline` (507 lines)
in the pipeline.py facade to bring the rest under 4000.

Giant one-def modules (function-body split is a later E-task if required):

| module | lines |
| --- | --- |
| pipeline_astrometry__generate_masterstar_and_catalog.py | 2540 |
| photometry_phase2a__phase2a_process_one_target.py | 1611 |
| pipeline_astrometry__detect_stars_and_match_catalog.py | 1372 |
| pipeline_astrometry__export_per_frame_catalogs.py | 1101 |
| photometry_comp__run_phase0_and_phase1.py | 1078 |
| pipeline_astrometry__astrometry_align_impl_body.py | 1049 |
| photometry_phase2a__phase2a_prepare_shared_state.py | 940 |

Remaining astrometry leftovers pack to pipeline_astrometry.py (65 / 3995) and
_2 (73 / 3687). Remaining phase2a leftovers pack to photometry_phase2a.py
(63 / 3995) and _2 (46 / 2584).

### Suggested E1..En order (architect writes the tasks)

Small and isolated first. MP spawn and giant defs last.

1. pipeline_import.py
2. pipeline_epsf_hooks.py / photometry_epsf_hooks.py (G-EPSF if ePSF graph touched)
3. photometry_exports.py, photometry_calibrate.py
4. photometry_shared.py
5. photometry_comp.py then photometry_comp__run_phase0_and_phase1.py
6. photometry_phase2a.py / _2 then the two giant phase2a defs
7. pipeline_calibrate.py (spawn workers; see M4)
8. pipeline_astrometry.py / _2 then the four giant astrometry defs
9. UI / gate / dead last; confirm dead before any delete (this task deletes nothing)

Each E-task: pure moves + import fixes, facades re-export, byte-identity after
each. G-EPSF when the ePSF graph is touched.

## M4 Risk register

`risk_register.json`

- __all__: 34 names on photometry_core (none on pipeline.py). photometry.py
  star-import re-exports them. Every moved __all__ name must stay on the
  photometry_core facade until E-final.
- string / getattr / patch-string: 13 names (tests patch `pipeline.foo` /
  `photometry_core.foo`). E-tasks must keep the dotted path working via the
  facade.
- private test imports: 86 `_`-prefixed names imported from dev/tests
  (e.g. `_vyvar_df_to_csv`, `_photometric_error`, `_calibrate_one_light_disk`).
- MP spawn workers (must remain `pipeline.<name>` for ProcessPoolExecutor
  spawn): `_init_calibrate_batch_worker`, `_calibrate_batch_process_one`,
  `_init_export_per_frame_worker`, `_export_per_frame_ram_worker_task`,
  `_calibrate_one_light_disk`.
- pickle: pipeline.py pickles FITS headers for RAM workers, not function
  objects. The spawn workers above are the real pickle-by-name hazard.
- No cross-file name collisions.

## Gates

| gate | status | detail |
| --- | --- | --- |
| G1 --fast --clean | PASS at 5b1068d | 1612 passed, 32 skipped. clean-tree PASS. db-quick-check WARN waived. Log: g1.txt |
| G2 | SKIP | git diff src_py empty |
| G-EPSF | SKIP | ePSF graph not touched |

## Files changed

- dev/tools/e0_split_map.py
- dev/results/context/session_20260901_e0/* (REPORT, json, module_map_proposal, g1)

## Errors

None that blocked the map. First mapper pass painted phase2a as phase0+1
because run_full_photometry_pipeline BFS-propagated both stages; fixed by
same-file independent reachability.
