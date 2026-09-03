E-FINAL progress. Branch consolidate-01.

Phase A DONE:
- 8136282 pipeline_constants leaf + facade re-export; deleted _PIXEL_MATCH_DEBUG_LOGGED
- f2faeb1 pipeline_catalog SAT_LIMIT from leaf; retired test_sat_limit_twin_guard
- ac4c9d2 masterstar_build constants
- d130f4d pipeline_calibrate constants
- 83e4f60 pipeline_astrometry constants
- d8d1ccc photometry_lightcurve constants

Phase B landed:
- fa1ccac + 20f36d7 _fit_subtract follow; E5 test updated
- 3db4e01 _plate_solve_input_bundle follow
- 9b0a414 extract_fits follow into gate_helpers (exc0389)
- 483780a _resolve_git_provenance lambda -> plain re-export
- 2787569 photometry_comp enrich injects
- 449efb0 LAST_EXCLUDED_TARGETS home photometry_comp; PEP 562
- 33328e8 _fill_masterstars follow -> pipeline_astrometry
- a8494e9 _astrometry_align_impl_body -> astrometry_align (call-time; cycle)
- 29cdc40 extract_fits_metadata remaining follow -> fits_meta

G1 C15 --fast FAIL was dirty-tree contamination (exc0312 mid-edit);
--clean PASS on a8494e9. C16 G1 in flight on 29cdc40.

Remaining: catalog_cone, pix2world, AppConfig, calibrate_mp_spawn,
read_flux inject, placeholders, stale call-time, Phase C, decisions/ROADMAP,
G2/G4, report, push.
