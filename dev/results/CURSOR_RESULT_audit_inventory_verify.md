# Inventory verification sample

OK present: 18
ABSENT confirmed: 2
WRONG/MISSING: 11

MISSING | 1.1 | Scan Source | importer.py
MISSING | 1.1 | Auto-detect optics | optics_autodetect.py
OK | 1.1 | Create Archive | src_py\app.py
OK | 1.1 | RUN VYVAR | src_py\night_run.py
OK | 1.1 | pre-cal import | src_py\app.py
OK | 1.1 | Simulate night run | src_py\night_run.py
MISSING | 1.1 | Draft manifest | draft_provenance.py
OK | 1.1 | Observation finalization | src_py\ui_finalization.py
MISSING | 1.2 | Master Dark | ui_calibration_library.py
MISSING | 1.2 | CAL-DIAG | cal_diag.py
MISSING | 1.2 | CAL stage | cal_stage.py
MISSING | 1.2 | OSC extract | osc_extract.py
OK | 1.2 | BPM dark MAD | src_py\config.py
OK | 1.3 | Quick calibrate | src_py\app.py
OK | 1.3 | RAM QC | src_py\app.py
OK | 1.3 | Preprocess QC enrich | src_py\app.py
OK | 1.3 | SAT-DIAG | src_py\pipeline.py
ABSENT-OK | 1.3 | Cosmic-ray rejection | removed 0ab686f
ABSENT-OK | 1.3 | CR L.A.Cosmic | removed 0ab686f
OK | 1.4 | Plate solve | src_py\app.py
OK | 1.4 | Blind solve | src_py\config.py
MISSING | 1.5 | VSX Gaia crossmatch | vsx_gaia_crossmatch.py
MISSING | 1.5 | TESS | tess_verify.py
OK | 1.6 | Phase 2A | src_py\photometry_core.py
OK | 1.6 | SysRem | src_py\photometry_core.py
MISSING | 1.6 | ePSF | psf_photometry.py
OK | 1.7 | BJD batch | src_py\photometry_core.py
OK | 1.8 | AAVSO export | src_py\export_reports.py
MISSING | 1.8 | Trust flag | trust_flag_core.py
OK | 1.9 | xval harness | src_py\trust_flag_core.py
OK | 1.10 | OSC export gate | src_py\export_reports.py