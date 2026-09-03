## D-FACADE-PERMANENT-01 (Milan 2026-09-03)

`pipeline.py` and `photometry_core.py` remain PERMANENT thin re-export
facades. `dev/` scripts keep importing through them. Facade removal
is off the table.

## D-CONSTANTS-LEAF-01 (Milan 2026-09-03)

ALL pipeline-physical constants live in leaf module
`pipeline_constants.py` (imports nothing from VYVAR). The SAT_LIMIT
numeric twin in `pipeline_catalog.py` is dismantled.

## D-RUNFULL-HOME-01 (Milan 2026-09-03)

`run_full_photometry_pipeline` stays physically in
`photometry_core.py` (production entry; C-D made permanent).
`analyze_calibrated_qc` + `_analyze_calibrated_qc_one` stay
physically in `pipeline.py` beside `AstroPipeline` (C-C).
