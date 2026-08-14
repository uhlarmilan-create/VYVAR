# Register diff for authorization -- PP-KWARG-01 (2026-08-14)

Base: `4a3e855`. Proposed append to `docs/VYVAR_AUDIT_2026_REGISTER.md`.

| Register ID | wave | stage | class | severity | summary | disposition | status |
|-------------|------|-------|-------|----------|---------|-------------|--------|
| **PP-KWARG-01** | 8 | preprocess | P | HIGH | `_pp_kw` splat passed `use_gpu_if_available` to `qc_enrich_calibrated_lights_in_place` (not in signature). Draft 511 failed at MAKE MASTERSTAR. Fixed by removing dead kwarg; static kwarg-compat gate added. | **FIXED** | CLOSED |

## Related

- Not caused by uncommitted CLOSE-IRON-GATES work.
- `preprocess_calibrated_to_processed` deprecated alias retained; production UI/night_run now call `qc_enrich_calibrated_lights_in_place` directly.
